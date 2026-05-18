"""Phase 1 qualitative + quantitative results viz.

Run AFTER training completes. Produces:
  - results/phase1_ddp/figures/rollout_curve.png   (L2 + cosine vs horizon k)
  - results/phase1_ddp/figures/loss_curve.png      (training loss vs step)
  - results/phase1_ddp/figures/clip_<name>.gif      (real video, 4 clips)
  - results/phase1_ddp/figures/clip_<name>_quality.png  (per-clip cosine curve)
  - results/phase1_ddp/PHASE1_RESULT_SUMMARY.md     (writeup)
"""
from __future__ import annotations
import json, sys, random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, "/root/autodl-tmp/world_model")
from src.phase1.dataset import discover_shards, _load_tokens
from src.phase1.heads import PredictiveHead

REPO = Path("/root/autodl-tmp/world_model")
RESULTS_ROOT = Path("/root/autodl-tmp/results/phase1_ddp/vggt_noact")
OUT = RESULTS_ROOT / "figures"
OUT.mkdir(parents=True, exist_ok=True)

device = "cuda:0"
cfg = yaml.safe_load(open(REPO / "configs/phase1/paper_scale.yaml"))
ctx_len = cfg["head"]["context_len"]
horizons = cfg["eval"]["horizons"]  # [1,2,4,8,16,32]

# 1) Find best/last checkpoint
ckpt_dir = RESULTS_ROOT
best = ckpt_dir / "best.pt"
last_ckpts = sorted(ckpt_dir.glob("ckpt_step_*.pt"))
ckpt_path = best if best.exists() else (last_ckpts[-1] if last_ckpts else None)
assert ckpt_path is not None, f"no checkpoint in {ckpt_dir}"
print(f"loading {ckpt_path}")
state = torch.load(ckpt_path, map_location=device, weights_only=False)

# 2) Build head, load weights
head_cfg = cfg["head"]
# PredictiveHead doesn't take token_pool; that's a dataset-level pooling knob.
head_kwargs = {k: v for k, v in head_cfg.items() if k != "token_pool"}
head = PredictiveHead(
    token_dim=cfg["cache"]["token_dim"],
    action_dim=cfg["dataset"]["action_dim"],
    **head_kwargs,
).to(device).eval()
# unwrap DDP module. prefix if needed
sd = state.get("model", state.get("model_state_dict", state))
sd = {k.replace("module.", ""): v for k, v in sd.items()}
head.load_state_dict(sd, strict=False)
print(f"loaded head, params={sum(p.numel() for p in head.parameters())/1e6:.1f} M")

# 3) Parse loss curve from training log (no metrics.jsonl exists)
import re
log_path = Path("/root/autodl-tmp/logs/train_phase1_ddp.log")
if log_path.exists():
    rows = []
    pat = re.compile(r"ep (\d+) step (\d+)\s+loss ([\d.eE+-]+)\s+lr ([\d.eE+-]+)")
    for line in log_path.read_text().splitlines():
        m = pat.search(line)
        if m:
            rows.append((int(m.group(2)), float(m.group(3)), float(m.group(4))))
    if rows:
        steps = [r[0] for r in rows]
        losses = [r[1] for r in rows]
        lrs = [r[2] for r in rows]
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        ax[0].plot(steps, losses, lw=0.6, alpha=0.5, color="C0")
        if len(losses) > 20:
            win = max(20, len(losses)//30)
            sm = np.convolve(losses, np.ones(win)/win, mode="valid")
            ax[0].plot(steps[win-1:], sm, lw=1.8, color="C1", label=f"smoothed (win={win})")
            ax[0].legend()
        ax[0].set_xlabel("step"); ax[0].set_ylabel("loss"); ax[0].set_yscale("log")
        ax[0].set_title(f"Phase 1 training loss (5×Blackwell DDP, vggt_noact)\nfinal: {losses[-1]:.4e} @ step {steps[-1]}")
        ax[0].grid(True, alpha=0.3)
        ax[1].plot(steps, lrs, lw=1.0, color="C2")
        ax[1].set_xlabel("step"); ax[1].set_ylabel("learning rate"); ax[1].set_yscale("log")
        ax[1].set_title("LR schedule (cosine with warmup)")
        ax[1].grid(True, alpha=0.3)
        fig.tight_layout(); fig.savefig(OUT / "loss_curve.png", dpi=120); plt.close()
        print(f"saved {OUT}/loss_curve.png with {len(rows)} log points")
    else:
        print(f"no parseable rows in {log_path}")
else:
    print(f"no log at {log_path}, skipping loss curve")

# 4) Pick 4 sample clips for qualitative — need manifest for jpg paths
manifest = json.load(open(REPO / "data/manifests/paper_scale.json"))
clip_to_frames = {c["clip_id"]: c["frames"] for c in manifest}
random.seed(0)
shards = discover_shards("/root/autodl-tmp/cache/paper_scale")
# Filter for clips long enough for max horizon rollout
min_len = ctx_len + max(horizons) + 1
candidates = [s for s in shards if s.n_frames >= min_len]
print(f"{len(candidates)} clips long enough (>= {min_len} frames)")
sample = random.sample(candidates, k=min(6, len(candidates)))

# 5) For each sample clip: predict + visualize
@torch.no_grad()
def rollout_clip(shard, head):
    """Autoregressive rollout from frames 0..ctx_len-1. Returns (real_tokens [Kmax,D], pred_tokens [Kmax,D], cos[Kmax], l2[Kmax])."""
    data = np.load(shard.path)
    tokens = _load_tokens(data["tokens"])  # [N, P, D]
    if head_cfg.get("token_pool", "mean") == "mean":
        tokens_state = tokens.mean(dim=1)  # [N, D]
    else:
        tokens_state = tokens
    Kmax = max(horizons)
    N = tokens_state.shape[0]
    if N < ctx_len + Kmax + 1:
        return None
    # context: frames 0..ctx_len-1
    ctx = tokens_state[:ctx_len].to(device).unsqueeze(0)  # [1, C, D]
    real = tokens_state[ctx_len : ctx_len + Kmax].to(device)  # [Kmax, D]
    acts = torch.zeros(1, ctx_len, cfg["dataset"]["action_dim"], device=device)
    tgt_a = torch.zeros(1, cfg["dataset"]["action_dim"], device=device)

    preds = []
    cur_ctx = ctx.clone()
    cur_acts = acts.clone()
    for _ in range(Kmax):
        try:
            pred = head(cur_ctx, cur_acts, tgt_a)  # [1, D]
        except TypeError:
            pred = head(cur_ctx, cur_acts)
        preds.append(pred.squeeze(0))
        # roll context: drop first, append pred
        cur_ctx = torch.cat([cur_ctx[:, 1:], pred.unsqueeze(1)], dim=1)
    preds = torch.stack(preds, dim=0).float()  # [Kmax, D]
    real = real.float()
    cos = F.cosine_similarity(preds, real, dim=-1).cpu().numpy()
    l2 = (preds - real).norm(dim=-1).cpu().numpy() / real.norm(dim=-1).clamp(min=1e-5).cpu().numpy()
    return real.cpu(), preds.cpu(), cos, l2

cos_curves, l2_curves, clip_names = [], [], []
for sh in sample:
    name = sh.meta["clip_id"]
    out = rollout_clip(sh, head)
    if out is None:
        continue
    _, _, cos, l2 = out
    cos_curves.append(cos)
    l2_curves.append(l2)
    clip_names.append(name)

    # per-clip quality plot
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    K = len(cos)
    ax[0].plot(range(1, K+1), cos, "o-", color="C0")
    ax[0].set_xlabel("horizon k (frames ahead)"); ax[0].set_ylabel("cosine sim (pred vs real)")
    ax[0].set_title(f"{name}: cosine"); ax[0].grid(True, alpha=0.3); ax[0].set_ylim(0, 1.02)
    ax[1].plot(range(1, K+1), l2, "o-", color="C3")
    ax[1].set_xlabel("horizon k"); ax[1].set_ylabel("L2 / |target|")
    ax[1].set_title(f"{name}: relative L2 error"); ax[1].grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / f"clip_{name}_quality.png", dpi=120); plt.close()

    # Animated gif of the actual clip
    frames_list = clip_to_frames.get(name, [])
    frame_paths = [Path(p) for p in frames_list][:ctx_len + max(horizons) + 1]
    imgs = []
    for fp in frame_paths:
        if fp.exists():
            img = Image.open(fp).convert("RGB").resize((256, 256))
            imgs.append(img)
    if imgs:
        gif_path = OUT / f"clip_{name}.gif"
        imgs[0].save(gif_path, save_all=True, append_images=imgs[1:], duration=100, loop=0)
        print(f"saved {gif_path} ({len(imgs)} frames)")

# 6) Aggregate rollout curve across the sample clips
if cos_curves:
    cos_mean = np.stack(cos_curves).mean(axis=0)
    cos_std = np.stack(cos_curves).std(axis=0)
    l2_mean = np.stack(l2_curves).mean(axis=0)
    l2_std = np.stack(l2_curves).std(axis=0)
    K = len(cos_mean)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(range(1, K+1), cos_mean, "o-", color="C0", label="mean")
    ax[0].fill_between(range(1, K+1), cos_mean - cos_std, cos_mean + cos_std, alpha=0.2)
    ax[0].set_xlabel("horizon k"); ax[0].set_ylabel("cosine sim")
    ax[0].set_title(f"Rollout cosine vs k (n={len(cos_curves)} clips)"); ax[0].grid(True, alpha=0.3); ax[0].set_ylim(0, 1.02)
    ax[1].plot(range(1, K+1), l2_mean, "o-", color="C3", label="mean")
    ax[1].fill_between(range(1, K+1), l2_mean - l2_std, l2_mean + l2_std, alpha=0.2)
    ax[1].set_xlabel("horizon k"); ax[1].set_ylabel("relative L2")
    ax[1].set_title(f"Rollout L2 vs k"); ax[1].grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "rollout_curve.png", dpi=120); plt.close()
    print(f"saved {OUT}/rollout_curve.png")

    # Print summary
    print("\n=== Summary ===")
    for k_idx, k in enumerate([1, 2, 4, 8, 16, 32]):
        if k_idx < K:
            i = k - 1
            print(f"  k={k:2d}  cos={cos_mean[i]:.3f}±{cos_std[i]:.3f}  L2/|y|={l2_mean[i]:.3f}")
    # Save summary json
    json.dump({
        "n_clips_evaluated": len(cos_curves),
        "horizons": [1, 2, 4, 8, 16, 32],
        "cos_mean": cos_mean.tolist(),
        "cos_std": cos_std.tolist(),
        "l2_mean": l2_mean.tolist(),
        "l2_std": l2_std.tolist(),
        "ckpt": str(ckpt_path),
    }, open(OUT / "rollout_summary.json", "w"), indent=2)

print("\nDone. Files in", OUT)
