"""Extract per-episode JPG frames from Open X-Embodiment webdataset tar shards.

Per `docs/DATA_ACQUISITION_SPEC.md`:
- Each tar is a webdataset (.tar) with one record per *episode*.
- We decode each episode, write frames to
    /root/autodl-tmp/oxe_extracted/<dataset>/episode_<i>/frame_<j>.jpg
- We also emit per-episode JSON sidecars so the manifest builder can stitch
  them together later without re-reading tars.

Episode-record schema varies per OXE sub-dataset (`OpenX-Embodiment.py` resolves
it lazily from one example). To stay robust, this script walks the record
recursively looking for image arrays / PIL images and the task string.

Output layout per episode:
    <out_dir>/<dataset>/episode_NNNNNN/frame_000000.jpg
    <out_dir>/<dataset>/episode_NNNNNN/meta.json
        {
          "clip_id": "<dataset>_NNNNNN",
          "dataset": "<dataset>",
          "episode_index": NNNNNN,
          "task": "...",
          "n_frames": K,
          "frames": ["<rel_frame_path>", ...]
        }
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import os
import sys
import time
from pathlib import Path

log = logging.getLogger("extract_oxe")

# Frame budget per spec.
MAX_FRAMES_PER_CLIP = 128
MIN_FRAMES_PER_CLIP = 8


# ----------------------------- helpers for unknown-schema episode records -----
def _is_pil_image(x) -> bool:
    try:
        from PIL.Image import Image
        return isinstance(x, Image)
    except Exception:
        return False


def _find_image_seq(node):
    """Walk a nested dict/list looking for a sequence of images. Returns list or None.

    OXE records typically nest like:
        steps -> list of step dicts
        each step has 'observation' -> {'image': PILImage, 'image_2': PILImage, ...}
    or sometimes:
        observations -> {'image': [PILImage, PILImage, ...]}
    """
    # Direct: a list of PIL images
    if isinstance(node, list):
        if node and _is_pil_image(node[0]):
            return node
        # list of dicts: try to extract one image per element
        if node and isinstance(node[0], dict):
            for key in ("image", "rgb_static", "rgb", "front_camera_rgb",
                        "observation.images.exterior_image_1_left", "image_0"):
                if key in node[0] and _is_pil_image(node[0][key]):
                    return [el[key] for el in node if isinstance(el, dict) and key in el]
            # Nested observation
            for key in ("observation", "obs"):
                if key in node[0] and isinstance(node[0][key], dict):
                    inner = [el[key] for el in node if isinstance(el, dict) and key in el]
                    seq = _find_image_seq(inner)
                    if seq:
                        return seq
        # generic recurse
        for el in node:
            seq = _find_image_seq(el)
            if seq:
                return seq
    elif isinstance(node, dict):
        for key in ("image", "rgb_static", "rgb", "front_camera_rgb", "image_0"):
            v = node.get(key)
            if isinstance(v, list) and v and _is_pil_image(v[0]):
                return v
        for v in node.values():
            seq = _find_image_seq(v)
            if seq:
                return seq
    return None


def _find_task_string(node) -> str:
    """Find a natural-language task string. Many OXE datasets include one."""
    if isinstance(node, str):
        # avoid returning tiny/empty/keys
        s = node.strip()
        if 4 <= len(s) <= 400 and " " in s:
            return s
        return ""
    if isinstance(node, bytes):
        try:
            return _find_task_string(node.decode("utf-8", errors="ignore"))
        except Exception:
            return ""
    if isinstance(node, list):
        for el in node:
            s = _find_task_string(el)
            if s:
                return s
    if isinstance(node, dict):
        # prefer canonical keys first
        for key in ("natural_language_instruction", "instruction", "task", "language_instruction"):
            if key in node:
                s = _find_task_string(node[key])
                if s:
                    return s
        for v in node.values():
            s = _find_task_string(v)
            if s:
                return s
    return ""


def extract_tar(
    tar_path: Path,
    dataset: str,
    out_dir: Path,
    episode_offset: int,
    jpg_quality: int = 90,
) -> tuple[int, int]:
    """Extract all episodes from one tar. Returns (n_episodes_written, n_frames_written)."""
    import webdataset as wds
    from PIL import Image as PILImage

    n_eps = 0
    n_frames_total = 0
    ds = wds.WebDataset(str(tar_path)).decode()

    for rec_idx, rec in enumerate(ds):
        # rec keys: __key__, __url__, ... and dataset-specific top-level keys.
        ep_idx = episode_offset + rec_idx
        ep_dir = out_dir / dataset / f"episode_{ep_idx:06d}"
        meta_path = ep_dir / "meta.json"

        # Resume: skip episodes whose meta.json is already present.
        if meta_path.exists():
            try:
                m = json.loads(meta_path.read_text())
                n_eps += 1
                n_frames_total += m.get("n_frames", 0)
                continue
            except Exception:
                pass

        images = _find_image_seq(rec)
        if not images:
            log.debug("skip %s.%d: no image sequence found", tar_path.name, rec_idx)
            continue

        n = len(images)
        if n < MIN_FRAMES_PER_CLIP:
            log.debug("skip %s.%d: only %d frames (< %d)",
                      tar_path.name, rec_idx, n, MIN_FRAMES_PER_CLIP)
            continue

        # Cap to MAX_FRAMES_PER_CLIP — truncate at start per spec.
        if n > MAX_FRAMES_PER_CLIP:
            images = images[-MAX_FRAMES_PER_CLIP:]
            n = MAX_FRAMES_PER_CLIP

        ep_dir.mkdir(parents=True, exist_ok=True)
        frame_paths: list[str] = []
        for i, img in enumerate(images):
            if not _is_pil_image(img):
                # Some datasets may have raw bytes; decode them.
                if isinstance(img, (bytes, bytearray)):
                    img = PILImage.open(io.BytesIO(img))
                else:
                    raise RuntimeError(f"unexpected image type: {type(img)}")
            if img.mode != "RGB":
                img = img.convert("RGB")
            fp = ep_dir / f"frame_{i:06d}.jpg"
            img.save(fp, format="JPEG", quality=jpg_quality)
            frame_paths.append(str(fp.resolve()))

        task = _find_task_string(rec)

        meta = {
            "clip_id": f"{dataset}_{ep_idx:06d}",
            "dataset": dataset,
            "episode_index": ep_idx,
            "task": task,
            "n_frames": n,
            "frames": frame_paths,
            "shard": str(tar_path.name),
        }
        meta_path.write_text(json.dumps(meta))
        n_eps += 1
        n_frames_total += n

    return n_eps, n_frames_total


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", type=Path, default=Path("/root/autodl-tmp/oxe"))
    ap.add_argument("--out-dir", type=Path, default=Path("/root/autodl-tmp/oxe_extracted"))
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--jpg-quality", type=int, default=90)
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

    summary: dict[str, dict] = {}
    for dataset in args.datasets:
        ds_in_dir = args.in_dir / dataset
        if not ds_in_dir.exists():
            log.warning("no tar dir for %s at %s — skipping", dataset, ds_in_dir)
            continue
        tars = sorted(ds_in_dir.glob(f"{dataset}_*.tar"))
        log.info("dataset=%s shards=%d", dataset, len(tars))
        n_eps = 0
        n_frames = 0
        for tar_idx, tar in enumerate(tars):
            t0 = time.time()
            # Episode offset: each shard contributes some episodes; use a wide
            # spacing so shard boundaries are visible in the indices.
            offset = tar_idx * 10_000 + n_eps  # safe upper bound — adjusted below
            # Actually, simpler: keep running count
            offset = n_eps
            eps_in_shard, frames_in_shard = extract_tar(
                tar, dataset, args.out_dir, episode_offset=offset, jpg_quality=args.jpg_quality
            )
            n_eps += eps_in_shard
            n_frames += frames_in_shard
            dt = time.time() - t0
            log.info("  %s: +%d eps (+%d frames) in %.0fs (total %d eps)",
                     tar.name, eps_in_shard, frames_in_shard, dt, n_eps)
        summary[dataset] = {"n_episodes": n_eps, "n_frames": n_frames}

    log.info("SUMMARY: %s", summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
