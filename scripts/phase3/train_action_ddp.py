"""Phase 3 BC + optional world-coupling trainer for the action policy pi_phi.

Companion to scripts/phase1/train_ddp.py and scripts/phase2/train_generative_ddp.py.
Uses the same multi-node DDP recipe + streaming dataset on the H100 cluster.

Stage 1 (default): w_world=0, pure behavior cloning on extracted OXE actions.
Stage 2: set train.w_world > 0 and provide predictor + generator ckpts. The
self-supervised coupling loss is

    L_world = || G(s, l, pi(s,l)) - D_psi(s, pi(s,l)) ||^2

i.e. "the generator's outcome of the proposed action should match the predictor's
forecast under the same action."

Launch:
    bash scripts/runs/run_multinode.sh phase3 <node_rank>
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.phase2.dataset import collate
from src.phase2.dataset_streaming import build_streaming_datasets
from src.phase2.text_encoder import FrozenCLIPText
from src.phase3.action_head import ActionPolicy, ActionPolicyConfig, count_params

log = logging.getLogger("phase3.train_ddp")


def _ddp_setup() -> tuple[int, int, int]:
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl")
        dist.barrier()
    else:
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
    return rank, local_rank, world


def _is_main(rank: int) -> bool:
    return rank == 0


def _save_ckpt(path: Path, **state) -> None:
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def _latest_ckpt(run_dir: Path):
    cks = sorted(run_dir.glob("ckpt_step_*.pt"))
    return cks[-1] if cks else None


def _lr_at(step: int, total: int, warmup: int) -> float:
    if step < warmup:
        return step / max(1, warmup)
    import math
    p = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1 + math.cos(math.pi * min(1.0, p)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="configs/phase3/h100_paper_scale.yaml")
    ap.add_argument("--out", default="/shared/results/phase3_h100")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.cfg).read_text())
    rank, local_rank, world = _ddp_setup()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    if _is_main(rank):
        log.info("Phase 3 BC trainer. world=%d device=%s", world, device)
        log.info("cfg=%s", json.dumps(cfg, indent=2)[:600])

    out_dir = Path(args.out) / "ckpts"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- model
    pcfg = ActionPolicyConfig(**cfg["policy"])
    model = ActionPolicy(pcfg).to(device)
    if world > 1:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=cfg["ddp"].get("find_unused_parameters", False))
    n_params = count_params(model.module if isinstance(model, DDP) else model)
    if _is_main(rank):
        log.info("policy params: %.1f M", n_params / 1e6)

    # --- data
    train_ds, val_ds = build_streaming_datasets(cfg, rank=rank, world_size=world)
    bs = int(cfg["train"]["batch_size"])
    nw = int(cfg["train"].get("num_workers", 4))
    train_dl = DataLoader(train_ds, batch_size=bs, num_workers=nw, collate_fn=collate)
    val_dl = DataLoader(val_ds, batch_size=bs, num_workers=0, collate_fn=collate)

    # --- text encoder (frozen)
    text_enc = FrozenCLIPText(cfg["text_encoder"]["hf_id"], cfg["text_encoder"]["max_length"]).to(device)
    text_enc.eval()
    for p in text_enc.parameters():
        p.requires_grad = False

    # --- optimizer
    params = [p for p in (model.module if isinstance(model, DDP) else model).parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])

    grad_accum = int(cfg["train"].get("grad_accum", 1))
    epochs = int(cfg["train"]["epochs"])
    warmup = int(cfg["train"]["warmup_steps"])
    log_every = int(cfg["train"].get("log_every_steps", 50))
    ckpt_every = int(cfg["train"].get("ckpt_every_steps", 1000))
    amp_dtype = torch.bfloat16 if cfg["train"].get("amp_dtype", "bfloat16") == "bfloat16" else torch.float16

    # Bound per-epoch iterations to avoid streaming-dataset hangs in DDP.
    try:
        steps_per_epoch = max(1, len(train_ds) // (bs * max(1, world)))
    except Exception:
        steps_per_epoch = 1000  # fallback
    total_steps = steps_per_epoch * epochs
    max_batches = steps_per_epoch * grad_accum

    w_bc = float(cfg["train"].get("w_bc", 1.0))
    # w_world > 0 enables Stage 2 coupling. For Stage 1 BC only, leave 0.
    w_world = float(cfg["train"].get("w_world", 0.0))

    global_step = 0
    start_epoch = 0
    if args.resume:
        ck = _latest_ckpt(out_dir)
        if ck is not None:
            sd = torch.load(ck, map_location="cpu", weights_only=False)
            (model.module if isinstance(model, DDP) else model).load_state_dict(sd["model"])
            opt.load_state_dict(sd["opt"])
            global_step = int(sd["step"])
            start_epoch = int(sd["epoch"])
            if _is_main(rank):
                log.info("resumed from %s @ step %d", ck.name, global_step)

    t0 = time.time()
    for ep in range(start_epoch, epochs):
        if hasattr(train_ds, "set_epoch"):
            train_ds.set_epoch(ep)
        model.train()
        opt.zero_grad(set_to_none=True)
        agg = {"bc": 0.0, "world": 0.0, "tot": 0.0, "n": 0}

        for bi, batch in enumerate(train_dl):
            if bi >= max_batches:
                break
            z = batch["z"].to(device, non_blocking=True)
            cond = text_enc(batch["text"], device)
            actions = batch.get("actions")
            if actions is None:
                # Without actions there is no BC target -- skip
                continue
            actions = actions.to(device, non_blocking=True)

            lr = cfg["train"]["lr"] * _lr_at(global_step, total_steps, warmup)
            for g in opt.param_groups:
                g["lr"] = lr

            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=device.type == "cuda"):
                pred_actions = model(z, cond)            # [B, T, A]
                bc = (pred_actions - actions).pow(2).mean()
                total = w_bc * bc
                world_loss = torch.zeros((), device=device)
                # Stage 2: w_world>0 requires predictor + generator ckpts wired in.
                # Currently a placeholder; implement once Phase 1+2 are trained.
                if w_world > 0:
                    log.warning("w_world > 0 set but Stage 2 coupling not yet wired. Skipping.")
                total_scaled = total / grad_accum

            total_scaled.backward()
            agg["bc"] += float(bc.detach())
            agg["world"] += float(world_loss.detach())
            agg["tot"] += float(total.detach())
            agg["n"] += 1

            if (bi + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(params, cfg["train"]["grad_clip"])
                opt.step()
                opt.zero_grad(set_to_none=True)
                global_step += 1

                if _is_main(rank) and global_step % log_every == 0:
                    log.info("ep %d step %d  bc %.4e  world %.4e  tot %.4e  lr %.2e  %.0fs",
                             ep, global_step,
                             agg["bc"]/agg["n"], agg["world"]/agg["n"],
                             agg["tot"]/agg["n"], lr, time.time() - t0)

                if _is_main(rank) and ckpt_every > 0 and global_step % ckpt_every == 0:
                    _save_ckpt(out_dir / f"ckpt_step_{global_step:08d}.pt",
                               model=(model.module if isinstance(model, DDP) else model).state_dict(),
                               opt=opt.state_dict(),
                               step=global_step, epoch=ep, cfg=cfg)

        if world > 1:
            dist.barrier()

    if _is_main(rank):
        _save_ckpt(out_dir / f"ckpt_step_{global_step:08d}.pt",
                   model=(model.module if isinstance(model, DDP) else model).state_dict(),
                   opt=opt.state_dict(),
                   step=global_step, epoch=epochs, cfg=cfg)
        log.info("done. elapsed=%.0fs", time.time() - t0)


if __name__ == "__main__":
    main()
