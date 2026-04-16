# GeoPhys-WM Phase 0 Feasibility

Diagnostic pipeline evaluating whether frozen VGGT-1B tokens are a viable state
representation for a world model on dynamic robotic manipulation video.

**This is a diagnostic only.** No world-model code lives here. No VGGT fine-tuning.
Base VGGT-1B only (no VGGT-World / VGGT-DP / VGGDrive variants).

## Layout

```
configs/default.yaml          one-place config
src/vggt_wrapper.py           encode / decode_geometry / full_inference
src/data_loader.py            manifest-driven clip loader
src/metrics.py                pure metric functions
src/viz.py                    plotting helpers
experiments/exp1..4.py        the four diagnostics
scripts/smoke_test.py         end-to-end sanity check
data/manifests/*.json         clip lists (checked in; raw frames are not)
results/                      generated outputs (gitignored)
REPORT.md                     synthesis + go/no-go (written last)
```

## Install

```bash
bash setup.sh
source .venv/bin/activate
python scripts/smoke_test.py        # downloads ~5GB weights on first run
```

Requires: CUDA GPU (Ampere+ for bf16), Python 3.11, ~10GB for weights & deps,
plus up to ~50GB once datasets are populated.

## Running the experiments

All four experiments read from `configs/default.yaml` and write into `results/`.

```bash
python experiments/exp1_temporal_coherence.py --limit 2     # quick sanity
python experiments/exp1_temporal_coherence.py               # full Set A
python experiments/exp2_static_vs_dynamic.py
python experiments/exp3_token_dynamics.py
python experiments/exp4_cross_domain.py
```

Execution order follows §11 of the brief: **Exp 3 is the most important** — if
median Pearson r < 0.2 on Set A, stop and report pivot.

## Data

Populate `data/manifests/set_{a,b,c}.json` before running. See `data/README.md`
for sources, licenses, and selection criteria. The pipeline is frozen against
manifests — nothing is fetched at run time.

## Interpreting results

| Experiment | Green | Yellow | Red |
|---|---|---|---|
| Exp 1 depth rel-err | <5% | 5–15% | >20% |
| Exp 1 rot Δ | <2° | 2–5° | >5° |
| Exp 3 median r | >0.5 | 0.2–0.5 | <0.2 |
| Exp 2 quality-vs-motion | smooth | linear decline | cliff drop |

Synthesis + final go/no-go in `REPORT.md` (written after all four runs complete).
