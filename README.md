# GeoPhys-WM

Unified generative + predictive world model on a frozen VGGT geometric latent, with physics-inference coupling. This repo hosts the feasibility study (Phase 0), the predictive head (Phase 1), the generative head + coupling prototype (Phase 2), and the **paper-scale results on 30K Open X-Embodiment clips** trained 2026-05-18.

**Current state (2026-05-18).** Phase 0, 1, 2-prototype, and **paper-scale Phase 1 + Phase 2 (`flow_coupled`)** all done. `flow_only` baseline currently running. A draft paper (LaTeX) is committed under `paper/`.

| Headline number | Value | Reference |
|---|---|---|
| Phase 1 rollout cosine @ k=32 | **0.995** | `docs/PHASE1_RESULT_2026-05-18.md` |
| Phase 2 self-consistency reduction under coupling | **−70 %** (0.026 → 0.008) | `docs/PHASE2_RESULT_2026-05-18.md` |
| Cost on flow-matching loss from coupling | **zero** | same |
| Total compute, both phases | ~16 GPU-h on 5×Blackwell, ~\$40 | `paper/main.tex` §4 |
| Reproduction time, single 5×Blackwell node | ~4 hours | one `bash` command, see below |

A short strategic summary in [`docs/ONE_PAGER.md`](docs/ONE_PAGER.md); the paper-scale write-ups and the draft paper are the most up-to-date authoritative sources.

## Where to start reading

| I want to… | Read this |
|---|---|
| **The paper draft (LaTeX, compilable)** | [`paper/main.tex`](paper/main.tex) — compile with `tectonic main.tex` or `pdflatex main` |
| Paper-scale Phase 1 result (cosine 0.995 @ k=32) | [`docs/PHASE1_RESULT_2026-05-18.md`](docs/PHASE1_RESULT_2026-05-18.md) |
| Paper-scale Phase 2 result (−70 % sc under coupling) | [`docs/PHASE2_RESULT_2026-05-18.md`](docs/PHASE2_RESULT_2026-05-18.md) |
| Concise project pitch (1 page) | [`docs/ONE_PAGER.md`](docs/ONE_PAGER.md) |
| What the predictor predicts / what action re-extraction does | [`docs/EXPLAINERS_2026-05-18.md`](docs/EXPLAINERS_2026-05-18.md) |
| Hand-off (current run state, Tier-1/2 scale-up plans, known issues) | [`docs/SESSION_RECAP_2026-05-18.md`](docs/SESSION_RECAP_2026-05-18.md) |
| Plain-language explainer (no ML background) | [`docs/EXPERIMENTS_EXPLAINED.md`](docs/EXPERIMENTS_EXPLAINED.md) |
| Detailed long-form technical reference | [`docs/WORKING_DOC.md`](docs/WORKING_DOC.md) |
| Paper-scale milestone plan (Phase 2) | [`PHASE2_PAPER_PLAN.md`](PHASE2_PAPER_PLAN.md) |
| 50-H100 stretch tier | [`docs/STRETCH_PLAN.md`](docs/STRETCH_PLAN.md) |
| Phase 0 feasibility study | [`PAPER.md`](PAPER.md), [`REPORT.md`](REPORT.md) |
| 30-clip prototype Phase 1 results | [`PHASE1_REPORT.md`](PHASE1_REPORT.md) |
| 30-clip prototype Phase 2 results | [`PHASE2_REPORT.md`](PHASE2_REPORT.md) |
| Data acquisition spec (OXE) | [`docs/DATA_ACQUISITION_SPEC.md`](docs/DATA_ACQUISITION_SPEC.md) |
| Archived earlier handoffs / status docs | [`docs/archive/`](docs/archive/) |

## Repo layout

```
configs/
  phase1/{default,paper_scale}.yaml      predictor configs
  phase2/{default,paper_scale}.yaml      generator + coupling configs
src/
  vggt_wrapper.py                        VGGT encode / decode_geometry
  data_loader.py, metrics.py, viz.py     shared utilities
  phase1/                                cache, dataset, heads, train_ddp, eval
  phase2/                                generator, physics, coupling, text_encoder
experiments/exp1..4.py                   Phase 0 four diagnostics
scripts/
  download_oxe.py                        Open X-Embodiment HF mirror puller (subset / full)
  extract_oxe_frames.py                  webdataset tar → JPGs + meta.json
  extract_oxe_actions.py                 unified 14-d action slot extractor (5 OXE subsets)
  build_paper_scale_manifest.py          30K-train + 3K-val manifest builder
  phase2/{train_generative_ddp,compare_eval,pixel_eval}.py
  runs/                                  ops scripts used to drive the paper-scale run
    install_torch.sh                     torch 2.7+cu128 (Blackwell sm_120)
    snapshot.sh                          periodic SNAPSHOT.md writer
    extract_orchestrator.sh              extract each dataset as it lands
    cache_run.sh                         shard manifest across N GPUs, run cache.py per shard
    train_phase1_launcher.sh             torchrun wrapper for Phase 1 DDP
    train_phase2_coupled.sh, train_phase2_flow_only.sh
    download_full_oxe.sh                 full Tier-1 OXE pull (~410 GB)
  eval/                                  result-viz scripts
    phase1_viz.py                        rollout curves + per-clip GIFs
    phase1_nn_demo.py                    nearest-neighbor-of-predicted-token visualization
    phase2_viz.py                        fm/sc/ph loss panels + coupling-zoom
tests/                                   smoke tests for Phase 1/2
data/manifests/{set_*,paper_scale}.json  clip lists
docs/
  ONE_PAGER.md                           executive summary + arch diagrams
  PHASE{1,2}_RESULT_2026-05-18.md        paper-scale results
  EXPLAINERS_2026-05-18.md               cross-discipline explainers (added 2026-05-18)
  SESSION_RECAP_2026-05-18.md            handoff / current run state
  EXPERIMENTS_EXPLAINED.md, WORKING_DOC.md, STRETCH_PLAN.md, DATA_ACQUISITION_SPEC.md
  archive/                               older session recaps + completed handoffs
paper/
  main.tex                               CoRL-targeted paper draft
  figures/*.png                          generated figures used by the draft
```

## Install + run

```bash
bash setup.sh                            # Phase 0 setup (uv venv, vggt clone)
# For Blackwell sm_120 (RTX PRO 6000): use the alternative installer
bash scripts/runs/install_torch.sh       # torch 2.7+cu128 + NCCL 2.30+

source .venv/bin/activate
python scripts/smoke_test.py             # Phase 0 sanity (downloads ~5 GB VGGT weights)
python tests/test_phase1_smoke.py
python tests/test_phase2_smoke.py
```

Requires: CUDA GPU (Ampere+ for bf16; sm_120 Blackwell needs the alt installer), Python 3.12, ~10 GB weights + ~250 GB cache for paper-scale.

## Reproducing the paper-scale results

```bash
# 1. Acquire OXE subset (~67 GB, ~85 min on a 13 MB/s line)
bash scripts/runs/download_full_oxe.sh   # or use --max-shards-per-dataset N for subset

# 2. Extract per-episode JPGs + actions
python scripts/extract_oxe_frames.py --in-dir /path/to/oxe --out-dir /path/to/extracted \
    --datasets jaco_play taco_play kuka fractal20220817_data bridge
python scripts/extract_oxe_actions.py    # adds actions.npy / states.npy per episode

# 3. Build the paper-scale manifest (30K train + 3K val)
python scripts/build_paper_scale_manifest.py

# 4. Cache VGGT tokens, sharded across N GPUs (~5 hours on 5×Blackwell)
bash scripts/runs/cache_run.sh

# 5. Phase 1: train the predictor (~25 min on 5×Blackwell DDP)
bash scripts/runs/train_phase1_launcher.sh

# 6. Phase 2: train the generator with coupling (~3 hours)
bash scripts/runs/train_phase2_coupled.sh
# (Optional) flow-matching-only baseline for comparison
bash scripts/runs/train_phase2_flow_only.sh

# 7. Produce figures and result summaries
python scripts/eval/phase1_viz.py
python scripts/eval/phase1_nn_demo.py
python scripts/eval/phase2_viz.py
```

## Phase 0 / 1 / 2 prototype workflows (smaller-scale)

```bash
# Phase 0 — four feasibility diagnostics on cached DROID tokens
python experiments/exp1_temporal_coherence.py
python experiments/exp2_static_vs_dynamic.py
python experiments/exp3_token_dynamics.py
python experiments/exp4_cross_domain.py

# Phase 1 prototype (30-clip DROID, multiple backbone arms)
bash run_phase1.sh

# Phase 2 prototype (flow_only vs flow_coupled on 30 clips)
bash run_phase2.sh
python -m scripts.phase2.compare_eval
python -m scripts.phase2.pixel_eval
```

## What's still missing for the paper

The two **headline empirical claims** of `docs/ONE_PAGER.md` are done. The following items remain (see `docs/PHASE2_RESULT_2026-05-18.md` for full discussion):

- [ ] `flow_only` baseline arm (running 2026-05-18 22:10, ~1 h)
- [ ] Action-conditioned Phase 1 retrain — action extraction done; cache + train pending (~6 h)
- [ ] **CFG action-dropout on the Phase 2 generator** — closes the ONE_PAGER's "one model, two modes" claim (~3 h)
- [ ] Pixel-space VGGT-decoded eval of Phase 2 outputs (~1 h after `flow_only`)
- [ ] Coupling-weight sweep, 9 runs (`w_sc × w_ph`) (~27 GPU-h)
- [ ] LIBERO downstream policy evaluation (original Phase 2 plan headline figure; not started)
- [ ] Held-out validation metrics (val_ids.json schema fix is a 15-line change)

## Citing / sharing

If you're discussing the project externally, please cite the specific phase report that contains the result you're using. The top-level `REPORT.md` and `PAPER.md` are **Phase 0 only**; current paper-scale numbers live in `paper/main.tex` and `docs/PHASE{1,2}_RESULT_2026-05-18.md`.

## Contact

See the git log for commit authors. Branch `autodl-run` carries the 2026-05-18 paper-scale work.
