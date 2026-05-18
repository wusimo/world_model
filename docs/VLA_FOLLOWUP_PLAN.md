# VLA Follow-Up Plan — Turning the World Model Into an Agent

**Question.** We have a 540 M-param world model on frozen VGGT tokens (predictor $D_\psi$ + generator $G_\theta$ + physics $g_\phi$). To make it generate *actions* and follow *language instructions* (i.e., act as a true VLA, comparable to OpenVLA / $\pi_0$), what's the concrete path?

**Answer.** Add two heads on top of the same frozen-VGGT cache: an action policy $\pi_\phi$ and a stronger text encoder. Train in three stages over ~3–4 weeks of 64×H100 time. Output: a $\sim$2.5 B-param **geometry-grounded VLA** with a defensible niche against bigger competitors.

This doc is a follow-up to `docs/SCALE_64xH100_PLAN.md` and complements its Phase B. Companion to `docs/PHASE{1,2}_RESULT_2026-05-18.md`.

---

## 1. The target architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│  Frozen VGGT-1B  +  Frozen text encoder (CLIP-L → small LM later)    │
└──────────────────────────────────────────────────────────────────────┘
                                  │
        ┌──────────────┬──────────┴──────────┬──────────────┐
        ▼              ▼                     ▼              ▼
   D_ψ (100 M)    G_θ (400 M)           π_φ (200 M)    g_φ (44 M)
   predictor      generator           ACTION POLICY    physics
   (today)        (today)               (new)         (today)
        │              │                     │              │
        ▼              ▼                     ▼              ▼
    next-token    token sequence       action vector    invariant
    forecast      from text+action     a_t..a_{t+T}     statistics

                            All four heads share the
                            same VGGT cache and (optionally)
                            the same coupling-loss machinery.
```

Total trainable: **~750 M** today + **200 M action head** + **300 M LM upgrade** = **$\sim$1.25 B**. Backbone (VGGT + text) is frozen, $\sim$1.3 B of free pretrained parameters.

---

## 2. The 14-d action vector — already extracted

We already extracted unified actions during the 2026-05-18 run (`scripts/extract_oxe_actions.py`). Slot schema:

```
[ world_vec_x, world_vec_y, world_vec_z,           # slots 0-2   end-effector translation
  rot_delta_x, rot_delta_y, rot_delta_z,           # slots 3-5   end-effector rotation delta
  gripper,                                         # slot  6     close / open
  base_disp_x, base_disp_y,                        # slots 7-8   mobile-base XY (Fractal)
  base_rot_z,                                      # slot  9     mobile-base yaw (Fractal)
  terminate_0, terminate_1, terminate_2,           # slots 10-12 episode-end one-hot
  spare ]                                          # slot 13     reserved
```

Per-episode `actions.npy` is on disk at `/root/autodl-tmp/oxe_extracted/<dataset>/episode_<NNNNNN>/actions.npy`. **No additional data work needed for the BC pretraining stage.**

---

## 3. Three-stage training plan

### Stage 1 — Behavior cloning (1 week, ~2 K H100-h)

Train $\pi_\phi$ from scratch as a supervised behavior-cloning model:

| Knob | Value |
|---|---|
| Input | $T=8$ frames of VGGT tokens $[B, 8, 1374, 2048]$ + CLIP-L/14 task embedding |
| Output | $a_1, \ldots, a_8 \in \mathbb{R}^{14}$ (one action per context frame) |
| Loss | MSE per slot (or cross-entropy on 256 bins per slot, OpenVLA-style) |
| Architecture | 12-layer Transformer × hidden 1024, 200 M params total |
| Schedule | 12 epochs, peak LR $3 \times 10^{-4}$, cosine decay, warmup 2 K |
| Data | 30 K – 300 K OXE clips × $\sim$30 frames each (depending on Phase A scale) |
| Compute | $\sim$2 K H100-h, 1.5 days on 64×H100 |

**Concrete deliverable.** A new `src/phase3/action_head.py` and `scripts/phase3/train_action_ddp.py`. Add to `configs/phase3/action_paper_scale.yaml`. Mirror the Phase 1/2 trainer skeleton — DDP, AMP bf16, streaming dataset, ckpt every 1 K steps.

**Acceptance gate M-V1.** Action MSE on held-out val $\leq$ 0.05 per slot averaged across slots, *or* trajectory replay matching ($k=8$ replayed rollout L2 < 0.2 on test demos).

### Stage 2 — Self-supervised coupling fine-tuning (1 week, ~3 K H100-h)

Add a coupling loss that doesn't require ground-truth actions:

$$\mathcal{L}_\pi = \mathcal{L}_{\text{BC}}(\pi_\phi(s, l), a_{\text{demo}}) + w_{\text{world}} \cdot \mathcal{L}_{\text{world}}(\pi_\phi, G_\theta, D_\psi)$$

where the world-consistency loss is:

1. Sample $a \sim \pi_\phi(s, l)$
2. Generate $s' = G_\theta(s, l, a)$ via the generator
3. Predict $\hat{s}' = D_\psi(s, a)$ via the dynamics predictor
4. $\mathcal{L}_{\text{world}} = \| s' - \hat{s}' \|^2$ — "the generator's outcome of the proposed action should match the predictor's forecast of that action"

**Why this matters.** It gives the policy a *self-supervised signal* without reward labels: "good actions produce predictable outcomes." This is the natural extension of the Phase 2 coupling story to the action head — and is the kind of methodological claim that makes a CoRL paper.

Same compute as Stage 1, ~1.5 days on 64×H100.

**Acceptance gate M-V2.** Coupled $\pi_\phi$ matches BC-only on val MSE AND improves on LIBERO success rate by $\geq$ 5 % over BC-only at the same training compute.

### Stage 3 — Language upgrade (2 weeks, ~10 K H100-h)

CLIP-L/14 produces a single 768-d sentence embedding. For *complex* instructions ("pick up the red cup that's left of the blue plate, only if the lid is open") this is too coarse. Upgrade options:

| Option | Backbone | Params | Compute to align | Pros / cons |
|---|---|---|---|---|
| A | Phi-2 (text-only) | 2.7 B | $\sim$5 K H100-h | small, strong reasoner; needs visual grounding adapter |
| B | Paligemma-2 (3B VLM) | 3 B | $\sim$10 K H100-h | matches $\pi_0$; trained-in visual grounding; bigger |
| C | Qwen2-VL-2B | 2 B | $\sim$8 K H100-h | great visual grounding; weaker action context |
| D | Stay with CLIP-L | 0.4 B | 0 | cheap; insufficient for hard instructions |

**Recommend B (Paligemma-2).** It's the same choice $\pi_0$ made; battle-tested for VLA tasks; pretrained vision–language alignment removes a training hurdle.

**Architecture choice.** Replace the CLIP-L embedding pipe with Paligemma-2's text encoder output. Add a *cross-attention* path from $\pi_\phi$'s Transformer layers into Paligemma's sequence-level text tokens (not just a pooled vector) so the policy can attend to *parts* of the instruction.

**Acceptance gate M-V3.** On LIBERO-Goal (130 tasks) and especially LIBERO-Long (long-horizon), the language-upgraded VLA beats the CLIP-L baseline by $\geq$ 10 percentage points success rate, AND matches OpenVLA-7B on the geometry-heavy subset (precise placement, stacking, fitting).

---

## 4. Data plan beyond what's on disk

| Bucket | Source | Status | Phase that needs it |
|---|---|---|---|
| 30 K OXE clips | already extracted | ✅ | Stage 1, Stage 2 |
| **Full 500 K OXE clips** | currently downloading (Tier 1) | ⏳ ~3 days | Stage 1 + Stage 3 scaling |
| **Per-clip language captions** for the ~30 % of OXE clips with empty task strings | Claude API VLM captions on 4 sample frames | not started | Stage 3 |
| LIBERO benchmark | public, 50 GB | not downloaded | Stage 2, Stage 3 evaluation |
| RoboCasa or FurnitureBench (geometric-heavy benchmarks) | public | not downloaded | Stage 3 evaluation |
| **(Optional)** Real-robot rollouts for online RL | hardware-dependent | n/a | stretch |

**Captioning cost.** At Anthropic API rates, captioning 30 K clips at $\sim$\$0.003 each = $\sim$\$100. One-shot, no infra needed.

**LIBERO download.** $\sim$50 GB, 1 hour. Standard benchmark — same harness as OpenVLA used.

---

## 5. Concrete code deliverables

```
src/phase3/                                       NEW
  action_head.py                                  ActionPolicy module (Transformer over tokens + text)
  inverse_dynamics.py                             IDM head (state, next-state → action) for self-supervised pretraining
  text_encoder_lm.py                              Wrapper around Paligemma-2 / Phi-2 / chosen LM
configs/phase3/
  action_bc.yaml                                  Stage 1 config
  action_coupled.yaml                             Stage 2 config (adds w_world > 0)
  action_lm.yaml                                  Stage 3 config (swaps CLIP-L → Paligemma)
scripts/phase3/
  train_action_ddp.py                             DDP trainer mirroring scripts/phase1/train_ddp.py
  caption_oxe.py                                  Claude API captioner for empty task strings
  eval_libero.py                                  LIBERO evaluation harness
  eval_geometric.py                               geometry-heavy task suite (custom: precise place, stack, fit)
docs/
  PHASE3_VLA_RESULT_<date>.md                     paper-style write-up after each stage
```

Estimated engineering effort: $\sim$3 weeks of focused work for one person, *separate from* the training compute.

---

## 6. Compute and timeline summary

| Stage | What | Wall-clock on 64×H100 | Compute (H100-h) | $ at \$3/h |
|---|---|---|---|---|
| Stage 1 | BC pretraining | $\sim$1.5 days | 2 K | \$6 K |
| Stage 2 | Coupling fine-tune | $\sim$1.5 days | 3 K | \$9 K |
| Stage 3 | LM upgrade + retrain | $\sim$6 days | 10 K | \$30 K |
| Eval | LIBERO + geometric | $\sim$1 day | 500 | \$1.5 K |
| **Total** | full VLA path | **$\sim$10 days** | **$\sim$15 K** | **\$45 K** |

Add Phase A from the scale-up plan (1B predictor + 1.5B generator) and you're at:

- Phase A + VLA = ~25 days on 64×H100, ~30 K H100-h, ~$90 K rental, ~$45 K with reserved discounts.

**One CoRL/ICLR paper falls out at Stage 2 end** ("Geometry-grounded VLAs with predictor coupling"). **A second paper at Stage 3 end** ("LM-conditioned action policies on frozen 3D foundations"). Both are fundable on academic / startup-tier compute.

---

## 7. How this positions against existing VLAs

| Model | Params | Backbone | Action head | Trained on | Geometric grounding |
|---|---|---|---|---|---|
| OpenVLA | 7 B | Prismatic VLM (Llama-2 + DINO/SigLIP) | discretized AR | OXE | implicit |
| $\pi_0$ | 3 B | Paligemma | flow-matching | OXE + proprietary | implicit |
| RT-2 | 55 B | PaLI-X / PaLM-E | discretized AR | OXE + Internet web | implicit |
| **Geom-VLA (this proposal, Stage 3)** | **$\sim$2.5 B** | **VGGT-1B + Paligemma** | **discrete or flow-matching** | **OXE + (optional) caption-augmented** | **explicit** (depth, camera, point map decodable from the same backbone) |

**Niche claim:** explicit 3D structure in the visual backbone via VGGT. Tests this niche on geometry-heavy tasks:
- Precision grasping: gripper position within 1 cm of target
- Stacking: place block on top of another with <2° tilt
- Insertion: peg-in-hole, fit-into-slot
- Pour: orient end-effector for liquid transfer
- Open-loop placement: position object at named 3D location

Existing VLAs win on language-complex tasks; we'd target geometry-precise tasks as the differentiator.

---

## 8. Open research questions (stretch territory)

1. **Hierarchical action heads.** Current $\pi_\phi$ predicts one action per frame. Real tasks need *plans* (open drawer → reach → grasp → close). Implement a high-level head that outputs sub-goals + a low-level head that converts sub-goals to actions.

2. **Reward modeling without demos.** Can we train a reward model from clip-level success labels alone (instead of dense per-frame rewards)? Standard inverse-RL territory; tractable.

3. **VLA + simulator closed loop.** Use our $G_\theta + D_\psi$ as the simulator from the post-training plan, run $\pi_\phi$ in it for online improvement. The full closed loop. Highest-risk, highest-payoff.

4. **Action-conditioned scene generation.** The "Marble-junior" extension *plus* action conditioning: given a 3D scene latent + a language goal + a starting position, generate the action sequence that achieves the goal. Connects all four heads (predictor, generator, scene, action) in one inference. Months of work.

---

## 9. Recommended ordering

**If the goal is "fastest publishable VLA result":**

1. Stage 1 (BC on 30 K OXE clips, CLIP-L language) → submit a baseline paper.
2. Stage 2 (coupling fine-tune) → strengthens the paper with the methodological-novelty story.
3. Skip Stage 3 for v1 of the paper; mark it as future work.

**Total time-to-paper:** ~3 weeks engineering + ~3 days training + ~1 week writing. ~$15 K compute.

**If the goal is "compete with OpenVLA across the board":**

1. Do Phase A (scale up world model first).
2. Do Stage 1 + 2 + 3 + LIBERO/geometric eval.
3. Real-robot demo for a follow-on.

**Total:** 2–3 months wall-clock, $50–100 K compute.

---

## 10. Bottom line

The frozen-VGGT-with-shared-substrate architecture extends naturally to a true VLA. Adding an action policy is **one more head on the same cache**, not a separate project. Adding a real LM backbone is the bigger lift (~10 K H100-h, $30 K compute) but standard engineering. The result is a $\sim$2.5 B-param geometry-grounded VLA that occupies a defensible niche against larger competitors: **smaller, cheaper, and explicit-3D-grounded** versus OpenVLA's bigger language and $\pi_0$'s bigger backbone.

The path is concrete, achievable on 64×H100, and produces 1–2 publishable papers along the way.
