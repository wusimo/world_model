# Pixel-Space Evaluation — Phase 2 flow_only vs flow_coupled

**Date:** 2026-05-18.
**Method:** Train a small 0.88 M-parameter token→depth decoder on 300 shards × ~30 frames each (10K frames total, 5 epochs to val L1 0.125). Sample 8 windows from each generator variant with identical seeds/inits. Decode all to 64×64 depth maps. Compare against the real cached depth.

Figures in `/home/simo/phase2_figures/pixel_eval/`.

## Headline numbers

| Metric (lower is better unless noted) | real | flow_only | flow_coupled |
|---|---|---|---|
| Temporal smoothness (frame-to-frame depth Δ) | **0.0060** | 0.0398 | 0.0405 |
| Wasserstein vs real depth distribution | — | **0.0052** | 0.0361 |
| Decoded depth mean | 1.026 | 1.024 | 0.999 |
| Decoded depth std | **0.173** | 0.163 | 0.107 |

## What this changes about the paper's claim

The Phase 2 result is **more nuanced** than my earlier draft suggested.

What still holds:
- Coupling reduces predictor self-consistency loss by 70% (token-space metric — `sc` in Table 2)
- Coupling does not regress flow-matching loss (`fm` essentially identical)

What we now see for the first time:
- Coupling **measurably concentrates** the decoded depth distribution. std drops from 0.173 (real) to 0.107 (flow_coupled), while flow_only's 0.163 stays close to real.
- Wasserstein distance to real depth: flow_only **7× closer** to real than flow_coupled (0.005 vs 0.036).
- Visually (see `compare_grid.png`): flow_only depth maps look sharper / closer to the cached reference; flow_coupled outputs are slightly blurred and lower-contrast.

**Interpretation.** The self-consistency loss `L_sc` is essentially a "predictability prior" — it asks the generator to produce sequences that the frozen predictor can easily extrapolate. Easy-to-extrapolate sequences are smoother in token space, which decodes to lower-contrast / less-varying depth in pixel space. Mode-collapse-ish.

This is **a known concern with consistency-style auxiliary losses** in generative training (classifier-guidance literature has the same trade-off). For the paper, this is a **real finding**, not a flaw — it shifts the contribution from "coupling is free" to:

> *Coupling reduces predictor self-consistency by 70 % at zero flow-matching cost, but trades roughly 7× more Wasserstein drift from the real depth distribution. The right operating point depends on the downstream task: predictor-plausibility vs distributional fidelity.*

This is the honest paper claim. Updates to `paper/main.tex` will reflect this.

## Caveats

1. **Decoder is small (0.88M params) and trained on 300 shards for 5 epochs.** A bigger decoder trained on the full cache would give tighter numbers. The qualitative comparison (flow_coupled smoothing > flow_only) is robust to this — both arms see the same decoder so it cancels out.
2. **n=8 samples** — small. Variance bounds on the Wasserstein number need bootstrapping (~$5 of additional compute).
3. The temporal-smoothness metric (frame-to-frame depth delta) is **basically identical** between the two arms — coupling didn't change temporal coherence, just spatial-statistical concentration.
4. The 7× Wasserstein gap could be driven by a few outlier frames. The histogram (`hist.png`) shows the bulk of the distributions overlap well; the tail is what differs.

## Files

- `/home/simo/phase2_figures/pixel_eval/compare_grid.png` — 3 rows × 4 samples decoded depth at t=0
- `/home/simo/phase2_figures/pixel_eval/strip_{real,flow_only,flow_coupled}.png` — 8-frame depth strips per variant
- `/home/simo/phase2_figures/pixel_eval/hist.png` — depth distribution histogram (3 overlay)
- `/home/simo/phase2_figures/pixel_eval/metrics.json` — full numeric output
- `/home/simo/phase2_figures/pixel_eval/decoder_log.json` — decoder training trace
