# Phase 2 Feasibility Prototype Report

**Date:** 2026-04-20
**Scope:** Coupling-mechanism feasibility prototype, not paper-scale Phase 2
**Compute:** ~22 min wall-clock across 2 GPUs (GPU 0: flow_only 6.2 min, GPU 1: flow_coupled 16 min)
**Config:** `configs/phase2/default.yaml`, 40 epochs, batch 8, 717 train / 186 val windows across 24/6 DROID clips

## TL;DR

Coupling the generative head to the frozen Phase-1 predictor via self-consistency loss **reduces held-out predictor self-consistency loss on generated rollouts by 41.5%** (0.0392 → 0.0229), while costing 3.6% on pure flow-matching val loss (0.952 → 0.986). The prototype success criterion is clearly met. The coupling mechanism works as designed on this data.

## Architecture

- **Frozen:** VGGT tokens (Phase 1 cache), CLIP ViT-B/32 text encoder, Phase 1 predictor `D_psi` (`vggt_noact/best.pt`).
- **Trainable:** `FlowMatchingGenerator` G_θ (22.6 M params, 6 transformer layers, hidden 512). In coupled variant only, `PhysicsInference` g_φ (6.2 M params).
- **Conditioning:** CLIP-pooled task embedding + init frame tokens.
- **Loss (coupled):** flow matching + `w_sc × L_sc` + `w_ph × L_ph`, with `w_sc=1.0`, `w_ph=0.1`. `z_gen` is produced inline during training by 8-step Euler integration of the flow ODE (gradients flow through the sampler).

## Post-hoc comparison (same sampler, same seed, 24-step Euler)

| Variant       | val_fm ± sem    | val_sc ± sem         | Δval_sc vs baseline |
|---------------|-----------------|----------------------|---------------------|
| `flow_only`   | 0.9518 ± 0.0169 | 0.03921 ± 0.00583    | —                   |
| `flow_coupled`| 0.9863 ± 0.0212 | **0.02293 ± 0.00024**| **−41.5%**          |

**Variance collapse.** `val_sc` standard error drops ~25× under coupling (0.00583 → 0.00024). The coupled generator does not just score better on average — it produces much more consistently predictor-plausible rollouts across held-out windows.

## Training dynamics

- `flow_only` best val_fm = 0.938 at epoch 25; converges in ~20 epochs, plateaus thereafter.
- `flow_coupled` best val_fm = 0.967 at epoch 37; val_fm tracks slightly above flow_only throughout, consistent with the coupling constraint pulling G_θ away from the pure flow-matching optimum. Self-consistency loss (training-time) drops from 0.022 at epoch 0 to 0.016 at epoch 39, with a brief spike to 0.019 around epoch 37 (not critical).
- Physics moment-match loss drops from ~2.8 × 10⁻⁶ to ~1.3 × 10⁻⁶. Absolute magnitude is tiny, suggesting g_φ quickly saturates — see "Limitations" below.

## Interpretation vs. the three sub-claims (Section 2 of the plan)

- **H1 (representation adequacy):** Already validated by Phase 0 + Phase 1. Phase 2 did not stress it further — the generator operates in the same cached VGGT token space and trains stably.
- **H2 (predictive viability):** Phase 1 established this.
- **H3 (physical coupling value):** The prototype is *partial* evidence for H3. We show that the predictor-side of the coupling (`L_sc`) does reshape the generator: coupled rollouts are substantially more plausible under a frozen predictor. This is the mechanism H3 depends on. We do *not* yet show the downstream payoff (OOD generalization, long-horizon stability, policy-training improvement) — those require paper-scale Phase 2 data.

## Prototype success criterion

- ✅ `flow_coupled` val_sc < `flow_only` val_sc on held-out windows → **YES** (−41.5%, clearly outside sem overlap).
- ✅ val_fm comparable (coupling did not tank flow matching) → yes, 3.6% cost.

Conclusion: **green-light scaling up to real Phase 2**. The coupling design is not broken.

## Limitations

1. **Tiny dataset.** 30 DROID clips × 128 frames is 3–4 orders of magnitude smaller than what Phase 2 per the roadmap requires (10–50K text, scene pairs). Effects at scale may differ. In particular, `val_fm` is still > 0.9 — this generator is far from a good sampler; both variants hallucinate rather than reconstruct. The comparison is meaningful only *relative to a shared baseline*.
2. **Physics loss magnitude is tiny** (~10⁻⁶). g_φ on 30 clips likely collapses the moment-match space to near-constant. A meaningful physics-coupling test needs more data diversity; on this prototype, essentially all lift comes from `L_sc`.
3. **Predictor is the `vggt_noact` baseline** (by design — generated rollouts lack actions), and Phase 1 showed `vggt_noact` was already slightly stronger than the action-conditioned predictor. This means the self-consistency loss is evaluated against our *best* Phase-1 predictor, which is the right call but means we haven't tested robustness to predictor choice.
4. **No pixel decode.** Evaluation is entirely in token space (val_fm, val_sc, g_φ moment match). We do not know whether the coupled samples would look better to a human than the uncoupled ones — we only know they are easier for the predictor to extrapolate.

## Plots

- `results/phase2/plot_compare.png` — bar chart of val_fm and val_sc with ±sem.

## Artifacts

- Per-variant ckpts: `results/phase2/runs/{flow_only,flow_coupled}/best.pt`
- Per-variant training logs: `results/phase2/runs/<v>/train_log.json`, `summary.json`
- Stdout: `results/phase2/train_{flow_only,flow_coupled}.log`
- Master runner log: `results/phase2/run_phase2.log`
- Comparison JSON: `results/phase2/comparison.json`

## Next steps for real Phase 2

1. Scale text-scene pair supply. DROID task strings are short and repetitive; consider supplementing with ScanNet/ARKitScenes captions or text-prompted bootstrap data (Section 8 of the proposal).
2. Re-test with `g_φ` output dim and hidden size that are appropriate for the scaled data — current 64-dim phi was overkill here and likely saturated.
3. Ablate coupling weights (`w_sc`, `w_ph`) to find the Pareto frontier of `val_fm` vs `val_sc`.
4. Add pixel-space eval via VGGT's decoder so we can check whether predictor-plausible samples are also perceptually plausible.
5. Schedule-based coupling (turn on `L_sc` only after G_θ has converged in pure flow matching) may avoid the 3.6% val_fm cost.
