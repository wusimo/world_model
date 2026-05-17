"""Build paper_scale.json from extracted OXE episodes.

Reads `meta.json` from every `episode_*/` directory under each dataset, applies
per-dataset target counts from `docs/DATA_ACQUISITION_SPEC.md`, splits into
train/val via stable hash on clip_id, and writes:

    data/manifests/paper_scale.json     # 30K train + 3K val entries
    data/manifests/val_ids.json         # list of val clip_ids
    data/manifests/paper_scale_summary.json   # per-dataset counts/sizes

The manifest entry schema matches `data/manifests/set_a.json`:

    {
      "clip_id": "<dataset>_NNNNNN",
      "set": "paper_scale",
      "frames": [<abs paths to jpg frames>],
      "fps": 15,
      "meta": {"dataset": "<dataset>", "episode_index": NNNNNN, "task": "..."},
      "gt_depth": null
    }
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import sys
from pathlib import Path

log = logging.getLogger("build_manifest")

# Per docs/DATA_ACQUISITION_SPEC.md §2. Totals must sum to ~30K train + ~3K val.
TARGET_TRAIN = {
    "bridge": 12_000,
    "fractal20220817_data": 8_000,
    "kuka": 4_000,
    "taco_play": 3_000,
    "jaco_play": 1_000,
}
VAL_FRACTION = 0.1  # ≈ 3 K val out of 30 K train (per spec)


def stable_hash01(s: str) -> float:
    h = hashlib.sha256(s.encode()).digest()
    return int.from_bytes(h[:8], "big") / (1 << 64)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--extracted-dir", type=Path, default=Path("/root/autodl-tmp/oxe_extracted"))
    ap.add_argument("--repo-root", type=Path, default=Path("/root/autodl-tmp/world_model"))
    ap.add_argument(
        "--out-manifest",
        type=Path,
        default=None,
        help="Defaults to <repo_root>/data/manifests/paper_scale.json",
    )
    ap.add_argument(
        "--out-val-ids",
        type=Path,
        default=None,
        help="Defaults to <repo_root>/data/manifests/val_ids.json",
    )
    ap.add_argument(
        "--out-summary",
        type=Path,
        default=None,
        help="Defaults to <repo_root>/data/manifests/paper_scale_summary.json",
    )
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    out_manifest = args.out_manifest or (args.repo_root / "data/manifests/paper_scale.json")
    out_val_ids = args.out_val_ids or (args.repo_root / "data/manifests/val_ids.json")
    out_summary = args.out_summary or (args.repo_root / "data/manifests/paper_scale_summary.json")
    out_manifest.parent.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)

    all_entries: list[dict] = []
    val_ids: list[str] = []
    per_dataset_stats: dict[str, dict] = {}

    for dataset, target_train in TARGET_TRAIN.items():
        ds_dir = args.extracted_dir / dataset
        if not ds_dir.exists():
            log.warning("dataset dir missing: %s — skipping", ds_dir)
            per_dataset_stats[dataset] = {"available": 0, "train": 0, "val": 0}
            continue

        ep_dirs = sorted(ds_dir.glob("episode_*"))
        log.info("dataset=%s available_episodes=%d target_train=%d",
                 dataset, len(ep_dirs), target_train)

        # Read all meta.json (cheap, ~1KB each).
        metas: list[dict] = []
        for ep_dir in ep_dirs:
            mp = ep_dir / "meta.json"
            if not mp.exists():
                continue
            try:
                m = json.loads(mp.read_text())
            except Exception as e:
                log.warning("bad meta %s: %s", mp, e)
                continue
            metas.append(m)

        if len(metas) == 0:
            log.warning("no episodes for %s", dataset)
            per_dataset_stats[dataset] = {"available": 0, "train": 0, "val": 0}
            continue

        # Stable hash determines train/val. Cap train at target_train.
        train_quota = min(target_train, int(len(metas) * (1 - VAL_FRACTION)))
        val_quota = int(train_quota * VAL_FRACTION / (1 - VAL_FRACTION))

        # Sort by stable hash so the "first N by hash" pick is reproducible.
        metas_sorted = sorted(metas, key=lambda m: stable_hash01(m["clip_id"]))

        train_metas = metas_sorted[:train_quota]
        val_metas = metas_sorted[train_quota:train_quota + val_quota]

        for m, is_val in [(x, False) for x in train_metas] + [(x, True) for x in val_metas]:
            entry = {
                "clip_id": m["clip_id"],
                "set": "paper_scale",
                "frames": m["frames"],
                "fps": args.fps,
                "meta": {
                    "dataset": m["dataset"],
                    "episode_index": m["episode_index"],
                    "task": m.get("task", ""),
                },
                "gt_depth": None,
            }
            all_entries.append(entry)
            if is_val:
                val_ids.append(m["clip_id"])

        per_dataset_stats[dataset] = {
            "available": len(metas),
            "train": train_quota,
            "val": val_quota,
        }

    # Shuffle so dataset chunks don't sit next to each other in the manifest.
    random.shuffle(all_entries)

    out_manifest.write_text(json.dumps(all_entries))
    out_val_ids.write_text(json.dumps(sorted(val_ids)))
    out_summary.write_text(json.dumps({
        "total_entries": len(all_entries),
        "n_val": len(val_ids),
        "per_dataset": per_dataset_stats,
    }, indent=2))

    log.info("Wrote %d entries (%d val) -> %s",
             len(all_entries), len(val_ids), out_manifest)
    log.info("Summary: %s", per_dataset_stats)
    return 0


if __name__ == "__main__":
    sys.exit(main())
