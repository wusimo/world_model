# Scale-Up Plan — 64×H100 / 50TB Storage Tier

**Question:** From what we have today (5×Blackwell, 67 GB OXE subset, 250 GB pooled VGGT cache, 100 M predictor + 400 M generator), what does a 64×H100 + 50 TB tier unlock? Specifically: a 1B–10B VGGT-based world model with generation + prediction + action-conditioning + serving as VLA post-training environment, aspiring toward Marble/point-cloud-class output.

Honest assessment first, then concrete plan.

---

## 1. Reality check — is 64×H100 enough?

**Yes for the core architectural ambition. No for full foundation-scale.**

| Comparable | Total compute | Equivalent on 64×H100 |
|---|---|---|
| Current paper (5×Blackwell, 16 GPU-h)  | $40 of compute | trivial |
| GR-2 (Bytedance) | ~50 K H100-h | ~30 days |
| OpenVLA | ~25 K H100-h | ~16 days |
| V-JEPA-2 | ~100 K H100-h | ~65 days |
| Cosmos (NVIDIA) | ~500 K – 1 M H100-h | impossible (~1 year) |
| Marble (Fei-Fei Li's startup, estimated) | ~200 K – 500 K H100-h, plus proprietary 3D data | impossible / unclear |
| Sora / Veo | 1 M – 10 M+ H100-h | impossible |

**Translation:** 64×H100 ≈ 1500 H100-h/day. You sit comfortably in the *robotics-foundation* tier (GR-2, OpenVLA, V-JEPA-2) but not in the *world-foundation* tier (Sora, Veo, Cosmos, Marble). The architectural claim — *adapter-on-frozen-VGGT* world model — is well-matched to your tier. **Foundation-tier output** (Marble's persistent 3D, Cosmos's photoreal video) would require fundamentally different research at $5–50M of compute, not just bigger versions of what you have.

Where the 50 TB really helps:

| Cache config | Per-frame size | Frames at 50 TB | Clips at 60 fr/clip avg |
|---|---|---|---|
| **Current pooled (1 vec/frame)** | 4 KB | 12.5 B | 200 M |
| **Pooled 8×8 grid (64 vec/frame)** | 256 KB | 200 M | 3.3 M |
| **Full 37×37 patch grid** | 5.6 MB | 9 M | 150 K |
| **Full grid + depth + camera + actions** | ~10 MB | 5 M | 80 K |

So with 50 TB you can either (a) cache a 200 M-frame dataset at pooled-grid resolution, or (b) cache 100 K clips at full geometric resolution. Both unlock different capabilities.

---

## 2. What scales cleanly vs what needs new research

| Capability | Approach | Scale-clean? | Effort |
|---|---|---|---|
| Predictor scaling 100M → 1B | bigger Transformer, same recipe | ✅ pure scaling | 3 days |
| Predictor scaling 1B → 10B | tensor-parallel + FSDP, distributed token cache | ✅ engineering, not research | 1 week |
| Generator scaling 400M → 1–3B | DiT-style flow matching, well-trodden | ✅ pure scaling | 4 days |
| Action conditioning (CFG) | already in flight tonight | ✅ small extension | done after tonight |
| Coupling at scale | mostly engineering | ✅ + risk of mode collapse needs sweep | 1 week |
| **VLA post-training env** | new code: rollout loop + reward + action sampler | ⚠️ engineering + integration | 3–4 weeks |
| Token → pixel RGB decoder | learn a decoder from cache (no architecture invention) | ✅ scaling | 1 week |
| Point cloud output | new decoder head, ground truth pre-cached from VGGT point-map | ⚠️ supervision quality matters | 2 weeks |
| Persistent 3D / gaussian splats (Marble-class) | **fundamentally different output modality**; needs novel decoder + per-scene optimization loop | ❌ research project | 3–6 months |
| Long-horizon (>1 minute) generation | architectural — needs efficient long-context handling | ⚠️ research | 1–2 months |
| Closed-loop VLA training with our model as env | new research: differentiable RL with token-space world model | ❌ research project | 3 months |

**Bottom line:** ~70% of the path is engineering you can buy with compute. ~30% requires new ideas (especially Marble-class persistent 3D and closed-loop VLA training).

---

## 3. The phased plan

### Phase A — Scale the current architecture to 1B (weeks 1–2)

**Goal:** Same architecture as today, 5–10× more params and 10–30× more data. Tests whether the frozen-VGGT story holds at scale.

| Component | Today | Phase A target |
|---|---|---|
| Predictor `D_ψ` | 100 M, 12 layers × 1024 | **1 B**, 24 layers × 2048 |
| Generator `G_θ` | 400 M, 20 layers × 1280 | **1.5 B**, 32 layers × 2048 |
| Physics `g_φ` | 44 M | 100 M |
| Total trainable | 540 M | **2.6 B** |
| Episodes | 30 K | **300 K** (full 5-dataset OXE + Something-Something v2) |
| Frames | 2.4 M | **30 M** |
| Cache size | 250 GB pooled | **4 TB** pooled, or **20 TB** unpooled patch grid |
| Compute | 16 GPU-h | **5–10 K H100-h** (3–7 days on 64 H100) |

**Data plan:** Full OXE 5-dataset (~575 K episodes, ~36 GB tars) + add `bridge_v2` (already in scope), `roboturk`, `robonet`, `nyu_franka_play`, `viola`, `austin_*`, `iamlab_*`. That gets to ~1 M robot episodes. Add Something-Something v2 (220K human-action clips) for action diversity. Cache at **full 37×37 patch grid** so we can decode pixels.

**Acceptance gates:**
- M-A1 (week 1): Cache 100 K clips at full grid. Storage: ~6 TB.
- M-A2 (week 1): 1B predictor reaches k=32 rollout cos ≥ 0.997 on held-out val. (Current at 100M is 0.995.)
- M-A3 (week 2): 1.5B generator with action-conditioned CFG dropout, sc reduction ≥ 50 % at zero fm cost, AND pixel Wasserstein ≤ 0.02 (vs current 0.036). Tests whether the predictability-fidelity trade-off softens with capacity.
- M-A4 (week 2): pixel-space RGB decoder trained from full-grid cache, FID ≤ 80 on held-out clips.

**Risk:** the frozen-VGGT bottleneck. VGGT-1B's representational ceiling may turn out to limit a 10B downstream. If A3/A4 plateau, the right move is Phase B with a different backbone OR a fine-tuneable adapter on VGGT.

### Phase B — Add VLA post-training environment (weeks 3–6)

**Goal:** Make the trained world model usable as a "fast simulator" that VLAs can roll out in to collect off-policy training data.

**What this requires:**

1. **Fast inference path.** Currently sampling a single 8-frame window takes 24 Euler steps × forward pass ≈ ~500 ms. For a VLA gradient step at ~20 Hz, this needs to drop to ~50 ms / window. Options:
   - **Consistency distillation** (1-step or 2-step sampler distilled from the 24-step teacher). Standard recipe, ~1 week of engineering.
   - **TensorRT inference** for the generator. 2–3× speedup, plug-and-play.
   - Together: target ~20 ms/window sampling.

2. **Action interface.** VLA emits actions → world model takes actions → outputs next observation tokens. Key design choice:
   - **A1.** World model outputs VGGT tokens; VLA gets them through a wrapper that decodes to whatever the VLA expects (RGB, depth, etc.). Slower but flexible.
   - **A2.** Modify VLA to consume VGGT tokens directly. Faster but couples us to specific VLAs.
   - Recommend **A1** for generality; **A2** as an optimization later.

3. **Reward signal.** For RL-style VLA training, need a reward. Options:
   - Reproduce target frames (imitation): MSE between rolled-out tokens and demonstration tokens.
   - Learned reward model from preference data (e.g., RLHF-style): too expensive for now.
   - Held-out demonstration likelihood: tractable, well-defined.
   - Recommend: imitation MSE for v1, learned reward model in v2.

4. **Rollout / replay infrastructure.** Standard RL stack. ~2 weeks of engineering, similar to dm_control or Isaac Gym integration.

5. **Closed-loop training experiment.** Take a VLA (start with OpenVLA), train it from real demos, then **continue training in the world model as a simulator**. Show downstream performance on LIBERO improves vs real-only training. This is the *headline claim* of Phase B.

**Compute estimate:** ~10 K H100-h for the full Phase B (mostly VLA fine-tuning, not world-model training). 1 week on 64 H100.

**Acceptance gate:** M-B1: OpenVLA + sim-rollout post-training improves LIBERO success rate by ≥ 5 % over real-only baseline at the same data budget. This is the kind of result that gets a CoRL or NeurIPS paper.

### Phase C — Persistent 3D / Marble-junior (months 2–6)

**Goal:** Move from token sequences to explicit 3D structure. Less ambitious than full Marble (no aim to be deployed for end users), more ambitious than current.

This is **research-territory**. Key bets:

1. **Output modality.** Choose from:
   - **3D gaussian splats** (Marble's bet): per-scene optimization on top of generated tokens.
   - **Point clouds** (your "PointNet" mention): decode tokens to 3D point maps via VGGT's point-map head, then aggregate across time.
   - **NeRF-style fields**: 3D-implicit, requires per-scene training. Slow.
   - Recommend: **point clouds** as the v1 (closest to VGGT's native output), gaussian splats as v2.

2. **Temporal persistence.** Marble's strength is that the scene doesn't change identity across viewpoints/time. Token sequences don't have this by construction. Need a *scene-level latent* that token sequences condition on, learned during training to be invariant across time. ~1 month of architecture exploration.

3. **Training data.** Currently 100% short-clip robot data. Persistent 3D needs longer-horizon scenes with viewpoint diversity. Need to add:
   - ScanNet++ / ARKitScenes (3D scene reconstruction data, ~5 TB)
   - HM3D / Replica (Habitat synthetic scenes)
   - Ego4D long takes (~7 TB usable subset)

4. **Compute.** ~50 K H100-h for the full Phase C training arc. 5 weeks on 64 H100. Realistic but the bottleneck is research velocity, not compute.

**Acceptance gate:** M-C1: given a 5-second clip from a never-seen scene, generate a persistent 3D representation (point cloud or splats) that supports novel-view rendering with PSNR ≥ 22 dB at 5° off-axis. This is below Marble's claimed quality but is a *publishable* "approaching foundation-tier" result.

### Phase D — Closed-loop research (months 4–9)

Stretch territory. Two open research questions:

1. **Can a token-space world model be a useful RL environment?** Currently world models are mostly used for offline data augmentation, not online RL. The latency, divergence-from-real, and reward-hacking concerns are unsolved.

2. **Does scale alone get us to Marble-quality?** Or do we fundamentally need a different output modality?

Both are 3–6 month research questions. Worth pursuing if A/B/C succeed.

---

## 4. Concrete recipes

### Storage allocation (50 TB)

```
/data/oxe_full/                    ~410 GB     raw OXE 5-dataset, all 588 shards
/data/oxe_extended/                ~5 TB       additional OXE sub-datasets + Something-Something
/data/cache_pooled/                ~4 TB       8×8 pooled token cache for 300K clips (Phase A)
/data/cache_fullgrid/              ~20 TB      37×37 patch grid cache for 100K clips (Phase A2)
/data/depth_camera/                ~3 TB       cached VGGT decode outputs (depth + camera per frame)
/data/3d_scenes/                   ~10 TB      ScanNet++, ARKitScenes, HM3D (Phase C)
/data/ego4d_subset/                ~7 TB       curated 5-min snippets from Ego4D
/data/checkpoints/                 ~500 GB     every saved checkpoint across Phases A/B/C
/data/results/                     ~200 GB     figures, eval outputs, paper artifacts
TOTAL                              ~50 TB
```

### Training recipes

**Phase A predictor (1 B params, 64×H100):**
- FSDP + activation checkpointing
- 12 epochs, cosine LR, peak 2e-4, warmup 5K steps
- Batch size 64 per rank × 64 ranks = 4096 effective
- Mixed bfloat16
- Expected wall-clock: **2.5 days** on 64 H100

**Phase A generator (1.5 B params, 64×H100):**
- Same FSDP setup
- 12 epochs, peak LR 1e-4
- Coupling on after epoch 4 (earlier than 6 epochs because we have 10× more steps)
- Sweep `w_sc ∈ {0.5, 1.0, 2.0}` and `w_ph ∈ {0, 0.1}` — 6 runs
- Expected wall-clock per run: **3 days** on 64 H100, **18 days total** for the sweep

### Software dependencies

Net new on top of what we have:

- `torch.distributed.fsdp` for FSDP (already in our PyTorch 2.7)
- `xformers` or `flash-attention-3` for efficient attention at 2K+ context
- `tensorrt` / `torch.compile` for inference optimization
- `gsplat` / `nerfstudio` if pursuing 3D output (Phase C)
- A reward / replay buffer library (we'd build this; ~1 week)

---

## 5. Where you'd sit at the end of the plan

After **Phase A** (~2 weeks): a 2.6 B-param frozen-VGGT world model with action conditioning, sc -50 %+, decoded pixel FID ≤ 80, runnable on 4 H100s at inference. **Direct CoRL/ICLR paper.**

After **Phase B** (~6 weeks): same model + an inference path fast enough to be used as a VLA simulator. OpenVLA + simulator post-training shown to improve LIBERO. **A separate, stronger paper.**

After **Phase C** (~6 months): persistent 3D point cloud output, novel-view PSNR ≥ 22 dB. Approaches but does not match Marble or Cosmos. **A third paper, scoped honestly.**

Total compute: roughly **100–200 K H100-h** across all three phases, or **2–4 months wall-clock on 64×H100 at high utilization**.

**Cost order of magnitude (rental):** 100–200 K H100-h × $3/h = **$300 K – $600 K**. With committed/reserved discounts: **$150 K – $300 K**. Academic-cluster access: ~$0.

---

## 6. Honest gaps from a "Marble-class" or "Cosmos-class" target

Even after Phase C, the project will be *one tier below* Marble / Cosmos. Specifically:

- **No Internet-scale pretraining.** Marble likely pretrains on hundreds of millions of Internet video frames; we'd cap around 30–50 M. The diversity / aesthetic gap is real and not closeable on 64 H100.
- **VGGT-1B is the perception ceiling.** Photorealistic output and complex scene composition require a perception backbone trained on Internet-scale RGB data — Marble's backbone is bigger and broader than VGGT. To match would require retraining the encoder, which is its own foundation-tier project.
- **No human-in-the-loop / RLHF.** Aesthetic quality at the level of consumer-facing products requires preference data. We don't have it and can't easily get it.
- **No deployed-product engineering.** Marble has streaming, mobile rendering, latency targets, content moderation. None of that is research.

**What we CAN credibly claim** with the proposed plan:
- A medium-tier *open* world model with a unified predictive + generative architecture
- Demonstrated downstream-policy improvement (Phase B)
- Approachable 3D output for short clips (Phase C)
- Total cost roughly 1/100th of Marble/Cosmos training

That's a strong research story. It's not a Marble competitor.

---

## 7. What I'd do if I were you

If your goal is **publishable research at the strongest possible tier**: Phase A (2 weeks) → Phase B (4 weeks). Two papers, both fundable. ~$100 K compute.

If your goal is **commercialization** (Marble-class product): the 64×H100 isn't enough. You'd need to rent or partner-into 500–1000 H100s, plus a content/data partnership for Internet-scale video, plus a 5–10 person team.

If you want to **probe the foundation-scale question**: do Phase A + a focused arm of Phase C (point cloud only, no gaussian splats). 3 months on 64 H100. Honest "scaling-laws of frozen-backbone world models" paper.

The first path is the easiest to execute and the easiest to talk to advisors / industry about. I'd recommend it unless you have a specific reason to chase Phase C.
