# Phase 0 — Current status (end of Day 1)

## Done

- Repo scaffolding matches §2 of the brief.
- `src/vggt_wrapper.py` — `encode` / `decode_geometry` / `full_inference`, bf16 autocast.
- `src/data_loader.py` — manifest-driven, uses `vggt.utils.load_fn.load_and_preprocess_images`.
- `src/metrics.py` — depth rel-err with median-ratio alignment, SO(3) geodesic,
  Chamfer, AbsRel, δ<1.25, photometric reprojection L1, token L2/cosine, Pearson.
- `src/viz.py` — depth-pair, heatmap-pair, histogram, scatter.
- `experiments/exp1..4.py` — full runners per §5 of the brief.
- `scripts/smoke_test.py` — downloads VGGT-1B, runs 8-frame forward pass, exercises
  the exact metric path Exp 3 uses.

## Verified by smoke test (GPU 2, bf16, H100 80GB)

| Output | Shape | Notes |
|---|---|---|
| aggregated_tokens (last layer) | `[1, 8, 1374, 2048]` | matches `per_frame_token_delta` expectation |
| depth | `[1, 8, 518, 518, 1]` | squeeze(-1) in experiments |
| point_map | `[1, 8, 518, 518, 3]` | |
| camera_extrinsic | `[1, 8, 3, 4]` | |
| camera_intrinsic | `[1, 8, 3, 3]` | |
| forward-pass time (8 frames) | ~2 s | model load: ~7 min first time (weights download); ~5 s cached |
| resident memory | well under 80 GB | |

Smoke-test token L2 deltas across 8 cartoon/no-overlap frames: `[50.8, 43.6,
54.8, 55.4, 38.0, 37.1, 74.4]` — clearly discriminates scene transitions, so the
Exp 3 signal path is live.

## Not done — blocked on decisions I should not make alone

These gate the actual diagnostic runs, not the code:

1. **Dataset licenses.** DROID/Open-X, ScanNet, nuScenes all require a human to
   accept TOS (and in ScanNet's case, email a signed form). I stopped here
   rather than agreeing on your behalf.
2. **DROID vs Open-X subset choice.** `lerobot/droid_100` is the smallest
   prepackaged option but only has 100 trajectories — fine for 30–50 clips, but
   if you want task diversity beyond DROID you should pick the Open-X sub-datasets.
3. **Whether to run exp3 on a placeholder dataset first.** We could populate Set
   A with the 25 `vggt/examples/kitchen/` frames as a fake "manipulation" clip
   just to burn the end-to-end pipeline once. That's a 5-minute sanity lap, not
   a diagnostic.
4. **GPU budget.** All four experiments on the recommended scale fit in the
   brief's 40–80 H100-hour budget; I have 6× idle H100s available. Happy to fan
   out per-clip across GPUs once data is in place.

## Approximate wall-clock remaining (with data in place)

| Step | Estimate (6× H100) |
|---|---|
| Exp 3 on Set A (30 clips × ~100 frames) | ~15 min |
| Exp 1 on Set A | ~30 min |
| Exp 2 (Set A + Set B with GT) | ~30 min |
| Exp 4 (Set C) | ~15 min |
| Plots + qualitative inspection | ~1 h |
| REPORT.md | ~1 h |

Total ≈ 3–4 wall-clock hours of compute + write-up once datasets land.

## Suggested next step

Confirm dataset approach, then I can either (a) proceed with `lerobot/droid_100`
+ ScanNet (both HF-hosted, no email gate), or (b) wait on Open-X / nuScenes
license acceptance before starting.
