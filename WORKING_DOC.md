# GeoPhys-WM Working Doc — What's Been Done, Architecture, Data

**Scope.** High-level summary of work in `Simo/geophys-feasibility/`, the working
repo for the GeoPhys-WM plan (unified generative + predictive world model on a
frozen VGGT geometric latent, with a physics-coupling module).

**Status snapshot (2026-04-20).** Phase 0 feasibility → cautious proceed.
Phase 1 predictive head → trained, VGGT beats DINOv2 by ~7–8×. Phase 2 coupling
prototype → −41.5% predictor self-consistency loss under coupling. Paper-scale
Phase 2/3 not yet run.

---

## 1. What's been done

### Phase 0 — Feasibility diagnostic (frozen VGGT-1B only)
Goal: verify H1, that frozen VGGT tokens are a viable state space for dynamic
manipulation scenes. Four experiments on 1–2 H100, ~40 GPU-hours.

| Exp | Question | Result | Rating |
|---|---|---|---|
| 1 | Temporal coherence across sliding windows (depth / pose / Chamfer) | Depth rel-err 5.5%, rotation 0.08° | Depth **Yellow**, pose **Green** |
| 2 | Static vs dynamic reconstruction quality | Smooth motion/quality profile; VGGT is affine-invariant so absolute AbsRel on ScanNet is not comparable | **Green** (no cliff) |
| 3 ★ | Do token deltas correlate with optical flow? (critical gate) | Median Pearson r = 0.47 (manipulation), 0.52 (driving); 90% p < 0.05 | **Yellow** (above Red 0.2, below Green 0.5) |
| 4 | Cross-domain probe (manipulation vs KITTI) | Driving shows higher variance, bimodal distribution; signal is domain-general | Mixed |

**Verdict:** Cautious proceed. Signal is present and significant; a world model
on these tokens will likely need either (a) light VGGT fine-tuning, or (b) a
dynamics head on top of the frozen backbone. Full write-up in `REPORT.md`.

### Phase 1 — Predictive head on frozen tokens (~7.5 min, 3 GPUs)
Train a small transformer predictor `D_ψ` on pre-cached tokens; compare VGGT vs
DINOv2 and ablate action conditioning.

| Run | Val loss | L2 k=1 | L2 k=8 | cos k=8 |
|---|---|---|---|---|
| `vggt` (action-cond) | 0.0189 | 1.73 | 3.83 | 0.9934 |
| `vggt_noact` (zero-action) | **0.0171** | 1.54 | 3.34 | 0.9939 |
| `dinov2` | 0.1228 | 13.87 | 24.11 | 0.8160 |

**Findings.** VGGT tokens dominate DINOv2 by ~7–8× on L2 at every horizon, and
generalize cleanly (no DINOv2-style overfitting). Action conditioning *regressed*
by ~9% — counterfactual action swaps shift predictions (cf_delta ≈ 0.04) but the
signal hurts more than it helps at this scale. Full write-up in `PHASE1_REPORT.md`.

### Phase 2 — Coupling feasibility prototype (~22 min, 2 GPUs)
Train a generative head `G_θ` (flow matching on VGGT tokens) with two variants,
to test whether the coupling mechanism (predictor self-consistency + physics
moment match) works as designed. **Not paper-scale Phase 2** — only 30 DROID clips.

| Variant | val_fm (flow-matching) | val_sc (predictor self-consistency) |
|---|---|---|
| `flow_only` | 0.9518 ± 0.0169 | 0.03921 ± 0.00583 |
| `flow_coupled` | 0.9863 ± 0.0212 | **0.02293 ± 0.00024** (−41.5%) |

Coupling cost 3.6% on flow matching, gained 41.5% on predictor self-consistency
on held-out windows, with variance collapsing ~25×. Success criterion met →
green-light scaling up. Physics moment-match loss ended near 10⁻⁶ (likely
saturated on 30 clips — needs real Phase 2 data to test meaningfully). Full
write-up in `PHASE2_REPORT.md`.

### Not yet done
- Paper-scale Phase 1: 50–200K manipulation episodes, full baselines, NeurIPS'26 numbers.
- Paper-scale Phase 2: 10–50K (text, scene) pairs with physics coupling at realistic scale.
- Phase 3: closed-loop self-improvement.
- Pixel-space evaluation via VGGT decoder (all current eval is in token space).
- Fine-tuning experiments (frozen VGGT throughout so far).

---

## 2. Architecture

### Backbone (frozen everywhere)
**VGGT-1B** (`facebook/VGGT-1B`), wrapped in `src/vggt_wrapper.py`:
- `encode(images)` → last-layer aggregated tokens, bf16 autocast.
- `decode_geometry(...)` → depth, point maps, camera extrinsic/intrinsic (used
  only for Phase 0 diagnostics; Phase 1/2 operate in token space).
- Tokens are pooled from VGGT's 1374-token output per frame (4 specials + 37×37
  patch grid) down to an **8×8 grid (64 patch tokens, D=2048)** via
  adaptive-avg-pool, cached per clip as bf16 (bitcast to int16 for numpy).

State representation used downstream:
- **Mean-pooled per frame**: `z_t ∈ ℝ^2048` for Phase 1 predictor input.
- **Full 8×8 grid**: `[T, 64, 2048]` for Phase 2 generator / physics module.

### Phase 1 — Predictive head `D_ψ` (~20 M params)
`src/phase1/heads.py`. Vanilla transformer encoder:
- Input: 4 past per-frame tokens + 4 past action embeddings + a learnable query
  slot conditioned on target action.
- 4 layers, hidden 512, 8 heads, dropout 0.1, norm-first, GELU.
- Output: predicted next-frame token (D=2048 for VGGT, 768 for DINOv2 baseline).
- Loss: MSE on tokens. Trained 30 epochs, batch 16, lr 3e-4, AdamW, cosine warmup.
- `vggt_noact` ablation zeroes the action path; same head elsewhere.

### Phase 2 — Generator `G_θ` (~22.6 M params)
`src/phase2/generative.py`. Rectified flow matching in VGGT token space:
- Predicts velocity `v_θ(z_t, t, cond_text, init_frame)` at time `t ∈ [0,1]`.
- Conditioning: frozen CLIP ViT-B/32 pooled text embedding (task string) +
  linear-projected init-frame tokens (broadcast over the T=8 frames as a static
  reference the generator has to deform).
- 6 transformer encoder layers, hidden 512, 8 heads, sinusoidal time embedding.
- Loss: `‖v_θ(z_t, t, c) − (z_1 − z_0)‖²` where `z_t = (1−t)z_0 + t z_1`, `z_0 ∼ 𝒩(0,I)`.
- Sampler: Euler (24 steps at eval, 8 steps during training for coupling — gradients
  flow through the sampler via `sample_with_grad`).

### Phase 2 — Physics inference `g_φ` (~6.2 M params)
`src/phase2/physics.py`. Treats physics as a **latent variable**, not a
supervised target:
- Transformer over the first 4 frames (4 × 64 tokens + CLS), hidden 384, 3 layers.
- Outputs `φ ∈ ℝ^64` — a distributional summary, not parameter-by-parameter.
- Consistency loss: first+second moment match between `φ(z_real)` and `φ(z_gen)`
  on a batch.

### Phase 2 — Coupling loss
`src/phase2/coupling.py`. Two legs:
1. **Predictor self-consistency** `L_sc`: sample future tokens from `G_θ`, feed
   through the *frozen* Phase-1 `vggt_noact` predictor one step at a time,
   require low step-wise MSE. Intuition: plausible rollouts should be easy to
   extrapolate. Zero actions are used (`vggt_noact` predictor) since generated
   trajectories have no real actions.
2. **Physics distribution match** `L_ph`: `g_φ(z_gen)` should look like
   `g_φ(z_real)` in moments.
3. Total: `L_fm + 1.0·L_sc + 0.1·L_ph` in the `flow_coupled` variant.

### What's frozen vs trained

| Module | Phase 0 | Phase 1 | Phase 2 |
|---|---|---|---|
| VGGT-1B backbone | frozen | frozen (cached tokens) | frozen (cached tokens) |
| CLIP text encoder | — | — | frozen |
| Predictor `D_ψ` | — | **trained** | frozen (`vggt_noact/best.pt`) |
| Generator `G_θ` | — | — | **trained** (both variants) |
| Physics `g_φ` | — | — | **trained** (coupled variant only) |

---

## 3. Data

Frozen against manifests under `data/manifests/`; raw frames live outside the
repo. Total <50 GB.

### Sets used

| Set | Role | Source | Size |
|---|---|---|---|
| **A** — manipulation | Phase 0 Exp 1/3; Phase 1 & 2 training | `lerobot/droid_100` (HF), DROID end-effector deltas as actions (A=7: xyz+rxyz+gripper) | 30 clips × ≤128 frames |
| **B** — static | Phase 0 Exp 2 (static baseline, GT depth) | ScanNet subset | 20 scenes × ~20 frames |
| **C** — driving | Phase 0 Exp 4 (cross-domain probe) | KITTI subset | 11 drives × up to 30 frames |

### Phase 1/2 token cache
- One forward pass of VGGT-1B per clip → `.npz` shards in
  `results/phase1/cache_tokens/` containing pooled tokens (8×8×2048 bf16),
  downsampled depth (64×64 fp16), extrinsic (3×4), intrinsic (3×3), actions
  (A=7), states (S=7), frame IDs.
- Window config: W=8 frames, stride 4 (50% overlap), up to 128 frames/clip →
  ~16 windows/clip.
- Train/val split: `val_episode_ids = [3, 9, 14, 19, 24, 29]` (6/30 held out);
  rest train. Same split shared across Phase 1 and Phase 2 so the frozen Phase 1
  predictor is not evaluated on its own training data in Phase 2.
- DINOv2 baseline: parallel cache using `vit_base_patch14_dinov2.lvd142m` at
  image size 518 (to match VGGT's FOV), same 8×8 pool, D=768.

### Text conditioning (Phase 2)
DROID task strings (free-text), fed through frozen CLIP ViT-B/32 text encoder
(max length 32). Empty tasks fallback: `"robot manipulation task"`. This is
deliberately shallow — real Phase 2 needs 10–50K richer (text, scene) pairs per
the roadmap.

### Physics supervision
None. `φ` is a learned 64-dim latent, supervised only through the moment-match
consistency loss and downstream predictor self-consistency. This keeps the
project feasible — no dataset would provide `friction=0.3`-style labels.

---

## 4. Repo map

```
configs/{phase1,phase2}/default.yaml   # one-place config per phase
data/manifests/set_{a,b,c}.json        # frozen clip lists
src/vggt_wrapper.py                    # VGGT encode/decode
src/data_loader.py, metrics.py, viz.py # Phase 0 primitives
experiments/exp1..4.py                 # Phase 0 diagnostics
src/phase1/{cache,dataset,heads,train,eval}.py
src/phase2/{generative,physics,coupling,dataset,text_encoder}.py
scripts/phase2/{train_generative,compare_eval}.py
run_phase1.sh, run_phase2.sh           # multi-GPU orchestration
REPORT.md                              # Phase 0 synthesis
PHASE1_{REPORT,STATUS}.md              # Phase 1 results + restart guide
PHASE2_{REPORT,STATUS}.md              # Phase 2 results + restart guide
PAPER.md                               # paper draft
results/                               # all metrics, plots, ckpts (gitignored)
```

## 5. Headline numbers to keep in mind

- Frozen VGGT token-flow Pearson r: **0.47** manipulation / **0.52** driving.
- Phase 1 VGGT vs DINOv2 next-token L2 at k=8: **3.83 vs 24.11** (~6.3×).
- Phase 1 action conditioning: slight regression (flag for Phase 2 triage).
- Phase 2 coupling gain on held-out predictor self-consistency: **−41.5%** at
  3.6% flow-matching cost; val_sc variance −25×.

## 6. Open questions carried forward

1. Why did action conditioning hurt in Phase 1? (embedding regularization /
   dataset action variance / FiLM vs concat injection)
2. How does the coupling behave with more data (the current physics loss
   saturated at 10⁻⁶)?
3. Should coupling be scheduled (turn on `L_sc` after `G_θ` converges)?
4. Fine-tune VGGT aggregator layers, or keep fully frozen?
5. Pixel-space eval via VGGT decoder — do predictor-plausible samples look
   perceptually plausible?
