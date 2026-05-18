# How the pipeline works — concise explainers

Two reader-friendly explainers for the two things you asked about: what action re-extraction does, and what the Phase 1 VGGT next-token predictor is actually predicting.

---

## 1. Action re-extraction — what + how

### What it does

For every OXE episode we already extracted, write a per-episode `actions.npy` (and `states.npy`) NumPy array next to the frame JPGs. After this pass, each `/root/autodl-tmp/oxe_extracted/<dataset>/episode_NNNNNN/` directory has:

```
frame_000000.jpg  …  frame_000NNN.jpg
meta.json
actions.npy    # (T, 14) float32
states.npy     # (T, 14) float32
```

That `actions.npy` is what `cache.py` will later read (via the `OXEActions` class I described in the handoff doc) so the cached tokens can include real action signals instead of the `NullActions` zeros we trained Phase 1 on.

### Why it's non-trivial

The 5 OXE sub-datasets each define actions differently:

| Dataset | Action keys | Total dims |
|---|---|---|
| `jaco_play` | `world_vector(3)` + `gripper_closedness_action(1)` + `terminate_episode(3)` | 7 |
| `bridge` | `world_vector(3)` + `rotation_delta(3)` + `open_gripper(1)` + `terminate(3)` | 10 |
| `kuka` | `world_vector(3)` + `rotation_delta(3)` + `gripper_closedness_action(1)` + `terminate(3)` | 10 |
| `fractal20220817_data` | `world_vector(3)` + `rotation_delta(3)` + `gripper(1)` + `base_displacement_vector(2)` + `base_displacement_vertical_rotation(1)` + `terminate(3)` | 13 |
| `taco_play` | `actions(7)` (already a flat 7-d vector) | 7 |

There's no canonical unified action space across OXE — this is a known pain point of the dataset. You can't naively concatenate or compare these without a schema.

### The algorithm

A simple **fixed-slot zero-pad** to 14 dimensions. Each slot has a *semantic* meaning that every dataset either fills or leaves zero:

```
[ world_vec_x, world_vec_y, world_vec_z,          # slots 0-2: end-effector translation
  rot_delta_x, rot_delta_y, rot_delta_z,          # slots 3-5: end-effector rotation delta
  gripper,                                        # slot  6:   gripper close/open
  base_disp_x, base_disp_y,                       # slots 7-8: mobile-base XY (Fractal only)
  base_rot_z,                                     # slot  9:   mobile-base yaw
  terminate_0, terminate_1, terminate_2,          # slots 10-12: episode-end one-hot
  spare                                           # slot 13:   reserved
]
```

The extractor reads each step's `action` dict, looks up which keys map to which slots for that dataset, and fills the rest with zeros. For datasets with no mobile base (everything except Fractal), slots 7-9 are always 0 — that's a semantic "this robot has no mobile base," not a learning signal that needs balancing.

For taco_play we splat the 7-d action into the first 7 slots assuming it maps to [world_vec, rot_delta, gripper] convention — this is approximate; the dataset paper would specify exactly.

**Normalization is NOT done here.** It happens at training time via `compute_action_stats(shards)`, which computes per-dim mean/std across each dataset's actions and normalizes. That way the model sees zero-mean / unit-std actions regardless of source dataset.

### Why this is "good enough" for the next experiment

The `vggt_noact` arm of Phase 1 already validates the predictor in token space without actions. The `vggt` arm (action-conditioned) tests whether adding actions *helps* — and prototype results showed it slightly hurts. We want to redo that test at paper scale because action-conditioning being net-negative is the open question. The 14-d slot schema gives the predictor a consistent input across datasets so it can learn per-slot conditioning. If it still doesn't help, that's a publishable negative result.

---

## 2. What is the Phase 1 predictor actually predicting?

### The short version

Given **8 consecutive frames** of a robot manipulation video, the predictor outputs a 2048-dimensional vector that *should be the frozen VGGT-1B encoder's pooled output for frame 9 (the next frame)*. Loss is MSE between the prediction and the cached real VGGT-pooled token for frame 9.

It's not predicting pixels. It's predicting an abstract geometric representation of the next frame.

### Step-by-step

1. **VGGT-1B (frozen).** A pre-trained geometric foundation model from Meta. Input: a window of RGB frames. Output: a 24-layer stack of "aggregator tokens" of shape `[1, T, 1374, 2048]` per layer. Each frame's 1374 tokens factor into 4 special tokens (camera + register) + 37×37 = 1369 patch tokens. The 2048-d embedding *learned by VGGT* encodes the per-patch 3D geometry that VGGT's depth + camera heads decode from. Think: "VGGT's internal language for describing the scene in 3D."

2. **Spatial pooling (cache time).** We adaptive-average-pool the 37×37 patch grid down to 8×8 (so 64 patches per frame), then mean across the 64 patches → **1 vector of dimension 2048 per frame**. This loses the spatial grid but keeps a per-frame "global geometry summary." Final cache shape: `[N_frames, 2048]` int16 (bf16 view) per clip.

3. **Predictor architecture (trainable, 100M params).** A 12-layer Transformer encoder with hidden 1024 and 16 heads. Given context tokens shaped `[batch, 8, 2048]` (8 consecutive frames), it produces 1 target token of shape `[batch, 2048]` — the model's guess for frame 9's pooled VGGT token. Architecture detail: a learnable query embedding attends to the 8 context tokens (positional encoded), goes through 12 layers, outputs through a linear head back to 2048-d.

4. **Loss.** MSE between predicted and cached real frame-9 VGGT token: `||pred − target||²` averaged over the batch.

5. **Autoregressive rollout (at evaluation).** To measure long-horizon quality:
   - Predict frame 9 from frames 1-8 → call this `p_9`.
   - Now predict frame 10 from `[frames 2-8, p_9]` (the model's own previous prediction is fed back as if it were real).
   - Continue for k=1, 2, 4, 8, 16, 32 steps ahead.
   - At each k, compare cosine and L2 of the model's prediction to the cached real VGGT token of that future frame.

### Why this is a meaningful task

There are two strong claims hiding in this setup:

**(a) The VGGT token is a compact geometric summary of a frame.** Tokens encode depth, camera, and surface info that VGGT can decode back to pixel-level 3D structure. If you can predict the next-frame's VGGT token, you implicitly predict the next-frame's geometry.

**(b) Token-space prediction can be much easier than pixel-space prediction.** Pixel-level video prediction has to model lighting, texture, micro-details. Token-level prediction only needs to model the gross geometric evolution — much closer to "rigid-body dynamics" than to "novel view synthesis." If the token space is rich enough, a small predictor on top of frozen VGGT should outperform a much bigger pixel-level predictor.

### What the Phase 1 result says

In our paper-scale Phase 1: cosine sim 0.995 at k=32 (4 sec / ~32 frames ahead) and relative L2 ~0.10. That means **after rolling out 32 token-frames purely from the model's own predictions, the trajectory in VGGT-token space still lies ≤10% from the real trajectory**. The model can keep its predicted "geometry summary" within 10% of reality for 4 seconds of robot motion. That's strong evidence the token space is predictable — and the headline claim of the project.

### What this is NOT (yet)

- **Not pixel video.** Phase 1 alone can't decode predicted tokens back to images because we pool away the patch grid. Decoding requires the un-pooled 1374-token grid, which Phase 2's generator produces.
- **Not action-conditioned (yet).** Current run is `vggt_noact` — actions are zeros. Phase 2 doesn't need actions (text + init-frame conditioning) so this is a separate workstream.
- **Not a closed-loop policy.** The predictor doesn't choose actions. It only forecasts geometry given the prefix.

### Connection to Phase 2

Phase 2's generator `G_θ` produces full token sequences (not just next-frame) conditioned on text. The novel idea: instead of training `G_θ` to only minimize flow-matching loss against real tokens, also minimize the *frozen Phase 1 predictor's* self-consistency loss on `G_θ`'s rollouts. That is, *the generator is asked to generate sequences that the predictor thinks are physically plausible.* This couples the generative and predictive sides of the world model — and the prototype showed it reduces predictor self-consistency error by 41.5%.
