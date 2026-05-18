"""Nearest-neighbor visualization of Phase 1 predictions.

For each sample clip:
  1. Predict tokens at k=1..K using autoregressive rollout from frames 0..7.
  2. Find the cached real frame whose pooled-token is closest (cosine) to each
     predicted token across the entire token pool.
  3. Stack three rows into a GIF:
       Row 1: context (frames 0..7), then REAL future frames 8..8+K
       Row 2: context (same), then NN-of-PREDICTED-token frames 8..8+K
       Row 3: per-frame cosine similarity bar
  4. Save as gif.

Output: /root/autodl-tmp/results/phase1_ddp/vggt_noact/figures/nn_<clip>.gif
"""
from __future__ import annotations
import json, sys, random, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, "/root/autodl-tmp/world_model")
from src.phase1.dataset import discover_shards, _load_tokens
from src.phase1.heads import PredictiveHead

REPO = Path("/root/autodl-tmp/world_model")
RESULTS = Path("/root/autodl-tmp/results/phase1_ddp/vggt_noact")
OUT = RESULTS / "figures"
OUT.mkdir(parents=True, exist_ok=True)

device = "cuda:0"
cfg = yaml.safe_load(open(REPO / "configs/phase1/paper_scale.yaml"))
ctx_len = cfg["head"]["context_len"]
KMAX = 32

# Load Phase 1 head
ckpt_path = sorted(RESULTS.glob("ckpt_step_*.pt"))[-1]
print(f"loading {ckpt_path}")
state = torch.load(ckpt_path, map_location=device, weights_only=False)
head_cfg = cfg["head"]
head = PredictiveHead(
    token_dim=cfg["cache"]["token_dim"],
    action_dim=cfg["dataset"]["action_dim"],
    **{k: v for k, v in head_cfg.items() if k != "token_pool"},
).to(device).eval()
sd = state.get("model", state.get("model_state_dict", state))
sd = {k.replace("module.", ""): v for k, v in sd.items()}
head.load_state_dict(sd, strict=False)

# Discover shards
shards = discover_shards("/root/autodl-tmp/cache/paper_scale")
random.seed(0)
print(f"{len(shards)} shards total")

# Build NN pool: sample 2000 shards, load all their tokens + remember (shard_idx, frame_idx) -> frame_path
print("building NN pool from 2000 shards...")
manifest = json.load(open(REPO / "data/manifests/paper_scale.json"))
clip_to_frames = {c["clip_id"]: c["frames"] for c in manifest}
random.shuffle(shards)
pool_shards = shards[:2000]
pool_tokens, pool_frame_paths = [], []
t0 = time.time()
for sh in pool_shards:
    data = np.load(sh.path)
    tokens = _load_tokens(data["tokens"]).float()  # [N, P, D]
    if tokens.dim() == 3:
        tokens = tokens.mean(dim=1)  # [N, D]
    name = sh.meta["clip_id"]
    frames = clip_to_frames.get(name, [])
    n = min(tokens.shape[0], len(frames))
    pool_tokens.append(tokens[:n])
    pool_frame_paths.extend(frames[:n])
pool_tokens = torch.cat(pool_tokens, dim=0)
pool_tokens = F.normalize(pool_tokens.to(device), dim=-1)
print(f"  pool: {pool_tokens.shape[0]} tokens / {len(pool_frame_paths)} frames in {time.time()-t0:.1f}s")

def nearest_path(query_tok):  # query_tok: [D]
    q = F.normalize(query_tok.unsqueeze(0).to(device), dim=-1)
    sims = q @ pool_tokens.T  # [1, N]
    i = int(sims.argmax(dim=-1).item())
    return Path(pool_frame_paths[i]), float(sims[0, i])

# Pick clips long enough
candidates = [s for s in shards if s.n_frames >= ctx_len + KMAX + 1]
sample = random.sample(candidates, k=6)

@torch.no_grad()
def rollout(sh):
    data = np.load(sh.path)
    tokens = _load_tokens(data["tokens"])  # [N, P, D]
    if tokens.dim() == 3:
        tokens_state = tokens.mean(dim=1).float()
    else:
        tokens_state = tokens.float()
    ctx = tokens_state[:ctx_len].to(device).unsqueeze(0)  # [1, C, D]
    acts = torch.zeros(1, ctx_len, cfg["dataset"]["action_dim"], device=device)
    tgt_a = torch.zeros(1, cfg["dataset"]["action_dim"], device=device)
    preds = []
    cur_ctx = ctx.clone()
    for _ in range(KMAX):
        try:
            pred = head(cur_ctx, acts, tgt_a)
        except TypeError:
            pred = head(cur_ctx, acts)
        preds.append(pred.squeeze(0))
        cur_ctx = torch.cat([cur_ctx[:, 1:], pred.unsqueeze(1)], dim=1)
    return torch.stack(preds, dim=0).float()  # [K, D]

def to_img(p, size=128):
    if p.exists():
        return Image.open(p).convert("RGB").resize((size, size))
    return Image.new("RGB", (size, size), (40, 40, 40))

for sh in sample:
    name = sh.meta["clip_id"]
    print(f"\n--- {name} ---")
    preds = rollout(sh)
    frames_list = [Path(p) for p in clip_to_frames.get(name, [])]
    if len(frames_list) < ctx_len + KMAX + 1:
        print(f"  not enough frames ({len(frames_list)})")
        continue
    ctx_imgs = [to_img(p) for p in frames_list[:ctx_len]]
    real_future = [to_img(p) for p in frames_list[ctx_len:ctx_len+KMAX]]
    nn_imgs = []
    nn_sims = []
    for k in range(KMAX):
        p, s = nearest_path(preds[k].cpu())
        nn_imgs.append(to_img(p))
        nn_sims.append(s)
    # Build stacked-frame GIF: 256x(128*2 + 40) = side-by-side real/nn + sim bar
    H = 128 * 2 + 40
    W = 128
    SIZE = 128
    pad = 4
    # Total frames: ctx_len (showing context with same row twice for clarity) + KMAX (showing real vs NN)
    out_frames = []
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except Exception:
        font = ImageFont.load_default()
    for i in range(ctx_len + KMAX):
        canvas = Image.new("RGB", (W + 8, H), (20, 20, 20))
        draw = ImageDraw.Draw(canvas)
        if i < ctx_len:
            # context phase: real frame in both rows
            canvas.paste(ctx_imgs[i], (4, 8))
            canvas.paste(ctx_imgs[i], (4, SIZE + 16))
            draw.text((4, 0), f"ctx t={i}", fill=(180, 180, 180), font=font)
            draw.text((4, SIZE + 4), "ctx", fill=(180, 180, 180), font=font)
            draw.text((4, 2*SIZE + 20), "(context: real frames)", fill=(120, 120, 120), font=font)
        else:
            k = i - ctx_len
            canvas.paste(real_future[k], (4, 8))
            canvas.paste(nn_imgs[k], (4, SIZE + 16))
            draw.text((4, 0), f"real t={ctx_len + k} (k={k+1})", fill=(120, 220, 120), font=font)
            draw.text((4, SIZE + 4), f"NN-of-pred  cos={nn_sims[k]:.3f}", fill=(220, 180, 120), font=font)
            # sim bar
            bar_w = int(min(1.0, max(0.0, nn_sims[k])) * (W - 8))
            draw.rectangle([4, 2*SIZE + 20, 4 + bar_w, 2*SIZE + 30],
                           fill=(220, 200, 80) if nn_sims[k] > 0.9 else (200, 100, 100))
            draw.text((W - 36, 2*SIZE + 20), f"{nn_sims[k]:.2f}", fill=(220, 220, 220), font=font)
        out_frames.append(canvas)
    gif_path = OUT / f"nn_{name}.gif"
    out_frames[0].save(gif_path, save_all=True, append_images=out_frames[1:],
                       duration=120, loop=0)
    print(f"  saved {gif_path}")

print("\nDONE")
