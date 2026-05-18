# world_model — Session Handoff (2026-05-18)

Local handoff doc. Captures (1) where the current paper-scale run stands, (2) how to check on it from a new session, (3) the Tier 1 / Tier 2 plan for scaling beyond this paper, (4) known issues + workarounds.

Pairs with `world_model_autodl_status.md` (more detailed run state) and `SNAPSHOT.md` (live, auto-updated every 60 s on the remote, synced manually).

---

## 1. Where the run is RIGHT NOW (2026-05-18 13:36 local)

| Stage | Status | Where |
|---|---|---|
| Env (torch 2.7+cu128 + deps + vggt clone) | ✅ done | `/root/autodl-tmp/world_model/.venv` |
| OXE download (88 shards / 67 GB) | ✅ done | `/root/autodl-tmp/oxe/` |
| Frame extraction (66,944 ep / 2.4 M frames) | ✅ done | `/root/autodl-tmp/oxe_extracted/` |
| Manifest (30,881 train + 3,086 val) | ✅ done | `/root/autodl-tmp/world_model/data/manifests/paper_scale.json` |
| VGGT token cache (30,795 / 30,881 npz) | ✅ done (99.7 %, 86 missing — acceptable) | `/root/autodl-tmp/cache/paper_scale/` (~250 GB) |
| **Phase 1 training (single-GPU on cuda:0)** | 🔄 running, ep 1, step 32K, loss 0.0116 | `/root/autodl-tmp/logs/train_single.log` |
| Phase 2 training | pending | — |

**Phase 1 training:** running as `python -m src.phase1.train_ddp --cfg configs/phase1/paper_scale.yaml --name vggt_noact ...` via `nohup` on GPU 0. About 31K steps per epoch, 12 epochs, ~25 steps/s → ETA ~4 hours from launch → **finishes ~17:14 local on 2026-05-18**.

**Key files on the autodl box:**
- Logs: `/root/autodl-tmp/logs/{train_single.log,cache_gpu*.log,oxe_download.log,extract.log,snapshot.log}`
- Snapshot (auto-updated every 60s): `/root/autodl-tmp/SNAPSHOT.md`
- Code: `/root/autodl-tmp/world_model` on branch `autodl-run` (3 commits ahead of `main`, all local — not pushed)
- Results dir: `/root/autodl-tmp/results/phase1_single/vggt_noact/` (checkpoints + curves)

**Background processes (all `nohup`'d, survive disconnect):**
- `snapshot.sh` — updates SNAPSHOT.md every 60s
- `train_single.sh` (pid ~164499) — Phase 1 training, single GPU
- Monitor task `bge3pkfjp` in this local Claude session — dies when you exit; not needed

---

## 2. How to resume next session

Easiest: start a fresh Claude session and paste this in:

> "Check the world_model run on autodl. The run is documented at /home/simo/world_model_handoff.md and /home/simo/world_model_autodl_status.md. Pull the live snapshot from autodl and tell me what's happened since I left."

The new Claude can then:

```
scp autodl:/root/autodl-tmp/SNAPSHOT.md /home/simo/SNAPSHOT.md
ssh autodl 'tail -30 /root/autodl-tmp/logs/train_single.log'
ssh autodl 'ls /root/autodl-tmp/results/phase1_single/vggt_noact/'
```

That gives the full picture. The autodl env (SSH alias, paths, no-sudo, China-proxy) is captured in the auto-memory under `reference_autodl_machine.md` so a fresh Claude can navigate it.

**Common follow-up actions next session:**
- "Show me the loss curve" → `scp` the metrics .json from `results/phase1_single/`
- "Phase 1 done — launch Phase 2" → there's a `scripts/phase2/train_generative_ddp.py`; needs a fresh launcher
- "Fix the val split bug" → see §4 below
- "Try the NCCL fix" → see §4 below

---

## 3. Tier 1 / Tier 2 plan to scale beyond this paper

Context: this run validates the "frozen VGGT + coupling" claim at paper scale (420 M params, 30K clips, ~4000 H100-h compute). To get to FAST-WAM-class capability, the field has shown the right move is **scale the predictor/generator, not the encoder**. Two tiers, in order:

### Tier 1 — Scale the predictor on top of frozen VGGT (1–2 months)

**Goal:** Test whether the frozen-VGGT story scales. Same architecture, 10× more capacity + data.

| Knob | Current | Tier 1 target | Why |
|---|---|---|---|
| Predictor params | 100 M (12L × 1024) | **1 B** (32L × 2048) | Test if next-token prediction quality scales with capacity over frozen tokens |
| Generator params | 300 M | 1 B | Same; generator should track predictor scale |
| Episodes | 30K (subset) | **100K–200K** | Pull the FULL OXE subset (588 shards / ~410 GB) instead of just 88 / 67 GB |
| Data sources | 5 OXE sub-datasets | + DROID-100 full, + Something-Something v2 | More embodiment diversity |
| Frame budget | 128 / clip | 256 / clip | Allows longer-horizon (k=64) evaluation |
| Cache size | ~250 GB | **~2 TB** | Same int16 format, 10× more clips |
| Actions | NullActions (zeros) | **Real OXE actions** extracted per-step | Closes the action-conditioning question Phase 1 left open |
| Compute | 5× Blackwell × 4 h | 5× Blackwell × 1–2 weeks, OR rent an 8× H100 node for a week | |

**Concrete next actions for Tier 1** (in order):

1. **Fix the actions story.** Patch `extract_oxe_frames.py` to also dump a per-episode `actions.npy` and `states.npy` (extracted from `rec["data.pickle"]["steps"][i]["action"]`). Then write an `OXEActions` class in `cache.py` (parallel to `DroidActions`) that reads the .npy. Different datasets have different action dims — zero-pad to a fixed 14-d action vector (covers all 5 OXE sub-datasets).
2. **Re-pull OXE at full size.** Edit `scripts/download_oxe.py` shard caps → set to `0` (= all). Network is ~13 MB/s aggregate observed → 410 GB takes ~9 hours.
3. **Re-extract + re-cache.** Same scripts, just over the bigger raw set. Expect ~2 days for full extraction + cache (caching is CPU-bound at ~0.85 clips/s aggregate on 5 GPUs; 200K clips × 1.2 s = ~67 hours).
4. **Scale the predictor in `configs/phase1/paper_scale.yaml`:**
   ```yaml
   head:
     hidden_dim: 2048   # was 1024
     n_layers: 32       # was 12
     n_heads: 32        # was 16
   train:
     batch_windows: 16  # halve to fit larger model in VRAM
     grad_accum: 2      # effective batch 32×5=160 still
   ```
5. **Need real DDP working** (NCCL fix — see §4). Single-GPU would take a month at Tier 1 scale.

**Acceptance gate:** Phase 1 L2 loss at k=8 plateaus at ≤ 0.8 (current run hits ~3.34 per prototype). If yes → write the "frozen tokens scale" paper.

---

### Tier 2 — Discrete-token autoregressive on top of VGGT (3–6 months)

**Goal:** Switch from continuous-token MSE prediction to discrete-token autoregressive — same recipe as Cosmos / Veo / GR-2. This is where the architecture starts to look like a "real" world model.

| Knob | Tier 1 | Tier 2 target | Why |
|---|---|---|---|
| Token representation | continuous, pooled 8×8 | **discrete codes** (16K–64K vocab, FSQ on pooled VGGT) | Enables AR + variable-length generation; required for SOTA video gen |
| Predictor | continuous MSE | **autoregressive transformer** (next-token logits over codebook) | The standard recipe for billion-frame regimes |
| Predictor params | 1 B | **3–7 B** | At this scale you get emergent multi-task behavior |
| Generator | flow-matching on continuous tokens | **same AR transformer, conditioned on text** (or diffusion-on-codes) | Unifies prediction + generation |
| Data | Tier 1 + Ego4D | + web video (HD-VG, WebVid, panda-70M) | Foundation-scale corpus |
| Total frames | ~10 M | **100 M – 1 B** | Where scaling laws start to bite |
| Compute | 5× Blackwell × 1–2 weeks | **50K–200K H100-h** | Real cluster needed; AutoDL won't cut it |

**Concrete next actions for Tier 2:**

1. **Token codebook.** Train a small FSQ or RVQ on top of pooled VGGT tokens. Codebook size 16K. ~10M frames for training, runs in 1 GPU-week.
2. **Rewrite predictor** as decoder-only transformer over discrete codes. Use Llama-style architecture; ~7 B params. Sequence length: 8 frames × 64 patches × 1 code/patch = 512 tokens per context. Output: 64 next-frame patch codes.
3. **Action tokenization.** Bucket each action dim into 256 bins → 14 action tokens per step. Interleave: `[a1, a2, ..., a14, f1, f2, ..., f64, a1, a2, ...]`.
4. **Text conditioning** via cross-attention to CLIP-L/14 embeddings of task strings.
5. **Training infra.** Need actual multi-node training. AutoDL maxes out at one node; need to rent a cluster (Lambda Labs, RunPod, or academic access).

**Acceptance gate:** Pixel-FID on held-out OXE clips ≤ 60 (current SOTA Cosmos hits ~45). Even hitting 80 would be a publishable result for the architecture's compute budget.

**Honest assessment:** Tier 2 is a genuine research project, 3–6 months of focused work, $50K–200K compute. Worth it if the goal is a foundation world model. Skip it if the goal is just to publish the paper this current run validates.

---

## 4. Known issues + carry-over fixes

### A. ~~NCCL crashes~~ FIXED 2026-05-18 14:15

**Symptom:** DDP training crashed ~1 sec after `head params: 155.63 M` with `CUDA error: an illegal memory access` in NCCL watchdog. Single-GPU ran fine.

**Root cause:** NCCL 2.26.2 (shipped with torch 2.7+cu128) has Blackwell sm_120 bugs.

**Fix applied:** `pip install --upgrade nvidia-nccl-cu12` → 2.30.4. NCCL allreduce + broadcast smoke-tested on 5 Blackwell GPUs ✓. Full training resumed from scratch under DDP at 14.6 steps/rank/s = 2336 samples/s aggregate (vs 800 samples/s single-GPU).

**Also fixed:** `paper_scale.yaml` flipped `ddp.find_unused_parameters: false → true` since `vggt_noact` mode leaves the action embedding params with no gradient → DDP complains otherwise.

**Note:** Pip warned `torch 2.7.0+cu128 requires nvidia-nccl-cu12==2.26.2`. That pin is overly strict — 2.30.4 works fine. If you ever do `pip install torch ...` again, you'll need to re-upgrade NCCL.

### B. val_ids.json schema mismatch — manifest writes clip_ids, splitter expects episode_indexes

**Symptom:** Training logs `train shards: 30795  val shards: 0`. No validation loss ever computed.

**Root cause:** `build_paper_scale_manifest.py` writes `val_ids.json` as a list of clip_id strings (`"bridge_000003"`). `dataset.py:split_shards` checks `sh.episode_index (int) in val_set (strings)` — never matches.

**Fix (15 lines):** Patch `src/phase1/dataset.py:split_shards` to match by `sh.meta.get("clip_id")` instead of `sh.episode_index`. Then re-run training.

This isn't blocking the current run — model trains fine without periodic val. But you can't pick a "best" checkpoint by val loss; only by step count. Worth fixing before Phase 2.

### C. Actions are all zeros (NullActions / `use_actions: false`)

**Why:** OXE doesn't have the DROID LeRobot parquet that `cache.py` was originally written for. SESSION_RECAP_2026-04-29 flagged this as an acceptable workstream split.

**Cost:** We can't test the action-conditioning claim from PHASE1_REPORT.md (Phase 1 §2 hypothesis). The strongest prototype variant was `vggt_noact` anyway — so the current run does validate the headline claim.

**Fix:** See Tier 1 action #1 above (extract real OXE actions during frame extraction).

### D. Disk usage

Current: ~325 GB total on `/root/autodl-tmp` (4.5 TB available).
- OXE raw: 67 GB
- Extracted frames: ~30 GB
- VGGT cache: ~250 GB
- Logs + meta: <1 GB

Tier 1 will need ~2 TB. Still fits.

### E. Not pushed to GitHub yet

The `autodl-run` branch has 3 commits ahead of `main` (all the script + fix work). The remote box has no GH credentials. Next session, either:
- Run `gh auth login` on the remote via `! ssh autodl ...`
- Or pull the branch down via `git fetch autodl autodl-run` after adding an ssh remote
- Or just copy patches via scp and push from local

---

## 5. Auto-memory references

Stored under `~/.claude/projects/-home-simo/memory/`:
- `user_autodl_remote.md` — user works from local box over SSH; don't install Claude remotely
- `reference_autodl_machine.md` — SSH alias `autodl`, paths, proxy
- `feedback_china_proxy.md` — `/etc/network_turbo` proxy for github/HF
- `project_world_model_oxe.md` — paper-scale OXE training pipeline driver

These are loaded automatically into a new Claude session's context — they'll know where things are.

---

**TL;DR — the one thing to do when you come back:**

```
cat /home/simo/SNAPSHOT.md     # see where things are
ssh autodl 'tail -30 /root/autodl-tmp/logs/train_single.log'
```

If `train_single.log` shows step ≥ 350K, training is done. Otherwise it's still running. Either way, the autodl box keeps working until you tell it to stop.
