"""Download Open X-Embodiment sub-datasets from the jxu124 HF mirror.

Per `docs/DATA_ACQUISITION_SPEC.md`, we want:
    bridge / fractal20220817_data / kuka / taco_play / jaco_play

Downloads tar shards to `--out-dir`. Resumable (idempotent — skips shards already
on disk at the right size). Sequential by default; pass `--workers N` to fetch
N shards in parallel (capped at 4 to keep the proxy happy).

Usage:
    python scripts/download_oxe.py \
        --datasets bridge fractal20220817_data kuka taco_play jaco_play \
        --out-dir /root/autodl-tmp/oxe \
        --workers 2 \
        --max-shards-per-dataset 0   # 0 = all

A shard cap is useful for the partial download strategy described in the spec:
we may not need ALL of KUKA (448 shards) if we only want 4K episodes.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import logging
import os
import sys
import time
from pathlib import Path

# Per HF script: jxu124/OpenX-Embodiment.py — shard counts per dataset.
OXE_SHARD_COUNTS = {
    "fractal20220817_data": 78,
    "kuka": 448,
    "bridge": 49,
    "taco_play": 11,
    "jaco_play": 2,
}
REPO_ID = "jxu124/OpenX-Embodiment"

log = logging.getLogger("download_oxe")


def shard_paths(dataset: str, n: int | None) -> list[str]:
    total = OXE_SHARD_COUNTS[dataset]
    take = total if (n is None or n <= 0) else min(n, total)
    return [f"{dataset}/{dataset}_{i:05d}.tar" for i in range(take)]


def download_one(rel_path: str, out_dir: Path) -> tuple[str, int, float, str]:
    """Download one shard. Returns (path, bytes, seconds, status).

    Status: 'downloaded' if newly fetched, 'cached' if already present.
    """
    from huggingface_hub import hf_hub_download

    out_path = out_dir / rel_path
    if out_path.exists() and out_path.stat().st_size > 1024:
        return (rel_path, out_path.stat().st_size, 0.0, "cached")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    local = hf_hub_download(
        repo_id=REPO_ID,
        filename=rel_path,
        repo_type="dataset",
        local_dir=str(out_dir),
    )
    sz = os.path.getsize(local)
    return (rel_path, sz, time.time() - t0, "downloaded")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--datasets",
        nargs="+",
        default=list(OXE_SHARD_COUNTS.keys()),
        choices=list(OXE_SHARD_COUNTS.keys()),
    )
    ap.add_argument("--out-dir", type=Path, default=Path("/root/autodl-tmp/oxe"))
    ap.add_argument("--workers", type=int, default=2, help="Parallel shard downloads (cap 4).")
    ap.add_argument(
        "--max-shards-per-dataset",
        type=int,
        default=0,
        help="0 means all shards. Otherwise take the first N shards of each dataset.",
    )
    ap.add_argument("--log", type=Path, default=None)
    args = ap.parse_args()

    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if args.log:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(args.log))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
    )

    workers = max(1, min(4, args.workers))
    cap = args.max_shards_per_dataset if args.max_shards_per_dataset > 0 else None
    plan: list[str] = []
    for d in args.datasets:
        plan.extend(shard_paths(d, cap))

    log.info("Plan: %d shards across %d datasets (workers=%d)", len(plan), len(args.datasets), workers)
    log.info("out_dir=%s", args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    total_bytes = 0
    t_start = time.time()
    done = 0
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(download_one, p, args.out_dir): p for p in plan}
        for fut in cf.as_completed(futures):
            rel_path, sz, dt, status = fut.result()
            done += 1
            total_bytes += sz
            mb = sz / 1e6
            mbps = (mb / dt) if dt > 0 else 0
            log.info(
                "[%d/%d] %s %s  %.1f MB  %.0fs  %.1f MB/s  total=%.1f GB",
                done, len(plan), status, rel_path, mb, dt, mbps, total_bytes / 1e9,
            )

    dt_total = time.time() - t_start
    log.info("DONE in %.0fs (%.1f min)  total=%.1f GB  avg=%.1f MB/s",
             dt_total, dt_total / 60, total_bytes / 1e9, (total_bytes / 1e6 / dt_total) if dt_total > 0 else 0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
