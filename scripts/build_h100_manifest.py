"""Build H100-scale manifest from extracted OXE episodes.

Like scripts/build_paper_scale_manifest.py but with bigger per-dataset targets:
    bridge                12K  -> 40K
    fractal20220817_data   8K  -> 30K
    kuka                   4K  -> 20K
    taco_play              3K  -> 3.5K  (it's small)
    jaco_play              1K  -> 1K   (cap)

Total ≈ 90-100K train + 9-10K val clips.

Same schema as paper_scale.json; just bigger numbers. Outputs:
    data/manifests/h100_scale.json
    data/manifests/h100_val_ids.json
    data/manifests/h100_scale_summary.json
"""
from __future__ import annotations
import argparse
import hashlib
import json
import logging
import random
import sys
from pathlib import Path

log = logging.getLogger("build_h100_manifest")

TARGET_TRAIN = {
    "bridge": 40_000,
    "fractal20220817_data": 30_000,
    "kuka": 20_000,
    "taco_play": 3_500,
    "jaco_play": 1_000,
}
VAL_FRACTION = 0.1


def stable_hash01(s: str) -> float:
    h = hashlib.sha256(s.encode()).digest()
    return int.from_bytes(h[:8], "big") / (1 << 64)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--extracted-dir", type=Path, default=Path("/data/oxe_extracted"))
    ap.add_argument("--repo-root", type=Path, default=Path("/data/world_model_workspace/world_model"))
    ap.add_argument("--out-manifest", type=Path, default=None)
    ap.add_argument("--out-val-ids", type=Path, default=None)
    ap.add_argument("--out-summary", type=Path, default=None)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    out_manifest = args.out_manifest or args.repo_root / "data/manifests/h100_scale.json"
    out_val = args.out_val_ids or args.repo_root / "data/manifests/h100_val_ids.json"
    out_summary = args.out_summary or args.repo_root / "data/manifests/h100_scale_summary.json"
    out_manifest.parent.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    entries: list[dict] = []
    summary: dict = {"datasets": {}}
    val_ids: list[str] = []

    for dataset, target in TARGET_TRAIN.items():
        ds_dir = args.extracted_dir / dataset
        if not ds_dir.exists():
            log.warning("dataset %s missing (no dir %s)", dataset, ds_dir)
            summary["datasets"][dataset] = {"n_found": 0, "n_train": 0, "n_val": 0}
            continue
        ep_dirs = sorted(ds_dir.glob("episode_*"))
        n_found = len(ep_dirs)
        log.info("%s: %d extracted episodes (target %d train + %d val)",
                 dataset, n_found, target, int(target * VAL_FRACTION))
        random.shuffle(ep_dirs)
        per_dataset_count = 0
        per_dataset_val = 0
        target_val = int(target * VAL_FRACTION)
        for ep_dir in ep_dirs:
            if per_dataset_count >= target + target_val:
                break
            meta_p = ep_dir / "meta.json"
            if not meta_p.exists():
                continue
            try:
                meta = json.loads(meta_p.read_text())
            except Exception:
                continue
            clip_id = meta.get("clip_id") or f"{dataset}_{int(ep_dir.name.split('_')[-1]):06d}"
            n_frames = int(meta.get("n_frames", 0))
            if n_frames < 8:
                continue
            entry = {
                "clip_id": clip_id,
                "set": "h100_scale",
                "frames": meta.get("frames", []),
                "fps": float(meta.get("fps", args.fps)),
                "meta": {
                    "dataset": dataset,
                    "episode_index": int(ep_dir.name.split('_')[-1]),
                    "task": meta.get("task", ""),
                },
                "gt_depth": None,
            }
            entries.append(entry)
            if stable_hash01(clip_id) < VAL_FRACTION and per_dataset_val < target_val:
                val_ids.append(clip_id)
                per_dataset_val += 1
            per_dataset_count += 1
        summary["datasets"][dataset] = {
            "n_found": n_found,
            "n_train": per_dataset_count - per_dataset_val,
            "n_val": per_dataset_val,
        }

    summary["n_total"] = len(entries)
    summary["n_val"] = len(val_ids)
    log.info("Wrote %d entries (%d val) -> %s", len(entries), len(val_ids), out_manifest)
    out_manifest.write_text(json.dumps(entries))
    out_val.write_text(json.dumps(val_ids))
    out_summary.write_text(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
