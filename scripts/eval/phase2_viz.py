"""Phase 2 (flow_coupled) qualitative + quantitative results viz.

Produces:
  - figures/loss_curve.png       (fm + sc + ph + total vs step, log y, coupling boundary marked)
  - figures/coupling_zoom.png    (zoomed sc and ph after coupling activation)
  - figures/loss_summary.json    (the headline numbers)
"""
from __future__ import annotations
import json, re
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

LOG = Path("/root/autodl-tmp/logs/train_phase2.log")
OUT = Path("/root/autodl-tmp/results/phase2_ddp/figures")
OUT.mkdir(parents=True, exist_ok=True)

pat = re.compile(r"ep (\d+) step (\d+)\s+fm ([\d.eE+-]+)\s+sc ([\d.eE+-]+)\s+ph ([\d.eE+-]+)\s+tot ([\d.eE+-]+)")
rows = []
for line in LOG.read_text().splitlines():
    m = pat.search(line)
    if m:
        rows.append({
            "ep": int(m.group(1)), "step": int(m.group(2)),
            "fm": float(m.group(3)), "sc": float(m.group(4)),
            "ph": float(m.group(5)), "tot": float(m.group(6)),
        })
print(f"parsed {len(rows)} rows from {LOG}")

steps = np.array([r["step"] for r in rows])
fm = np.array([r["fm"] for r in rows])
sc = np.array([r["sc"] for r in rows])
ph = np.array([r["ph"] for r in rows])
tot = np.array([r["tot"] for r in rows])

# Find first step with sc>0 (coupling activation)
coupling_idx = np.where(sc > 1e-6)[0]
coupling_step = int(steps[coupling_idx[0]]) if len(coupling_idx) > 0 else None
print(f"coupling activated at step {coupling_step}")

# Main loss curve — 4 panels (fm, sc, ph, total)
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for ax, vals, name, color in zip(axes.ravel(),
                                  [fm, sc, ph, tot],
                                  ["fm (flow matching)", "sc (predictor self-consistency)",
                                   "ph (physics consistency)", "tot (weighted sum)"],
                                  ["C0", "C3", "C2", "C4"]):
    ax.plot(steps, vals, lw=0.8, alpha=0.6, color=color)
    if len(vals) > 20:
        win = max(20, len(vals)//30)
        sm = np.convolve(vals, np.ones(win)/win, mode="valid")
        ax.plot(steps[win-1:], sm, lw=1.6, color=color, label=f"smoothed")
    if coupling_step is not None and "fm" not in name:
        ax.axvline(coupling_step, color="k", ls="--", alpha=0.5, label=f"coupling on @ step {coupling_step}")
    ax.set_xlabel("step"); ax.set_ylabel(name); ax.set_yscale("log" if name != "tot" else "linear")
    ax.set_title(name); ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
fig.suptitle("Phase 2 (flow_coupled) training losses — 5×Blackwell DDP, 30K OXE clips")
fig.tight_layout()
fig.savefig(OUT / "loss_curve.png", dpi=120)
plt.close()
print(f"saved {OUT}/loss_curve.png")

# Zoom on coupling effect
if coupling_step is not None:
    coup_steps = steps[coupling_idx]
    coup_sc = sc[coupling_idx]
    coup_ph = ph[coupling_idx]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(coup_steps, coup_sc, lw=0.8, alpha=0.5, color="C3")
    if len(coup_sc) > 10:
        win = max(10, len(coup_sc)//20)
        sm = np.convolve(coup_sc, np.ones(win)/win, mode="valid")
        ax[0].plot(coup_steps[win-1:], sm, lw=2.0, color="C3", label="smoothed")
        ax[0].axhline(coup_sc[0], color="gray", ls=":", label=f"initial: {coup_sc[0]:.4f}")
        final_sc = sm[-1]
        ax[0].axhline(final_sc, color="C1", ls=":", label=f"final: {final_sc:.4f} (Δ {(final_sc/coup_sc[0]-1)*100:.0f}%)")
        ax[0].legend(); ax[0].grid(True, alpha=0.3)
    ax[0].set_xlabel("step (post-coupling)"); ax[0].set_ylabel("sc loss")
    ax[0].set_title("Predictor self-consistency reduction under coupling")

    ax[1].plot(coup_steps, coup_ph, lw=0.8, alpha=0.5, color="C2")
    if len(coup_ph) > 10:
        win = max(10, len(coup_ph)//20)
        sm = np.convolve(coup_ph, np.ones(win)/win, mode="valid")
        ax[1].plot(coup_steps[win-1:], sm, lw=2.0, color="C2", label="smoothed")
    ax[1].set_xlabel("step"); ax[1].set_ylabel("ph loss"); ax[1].set_yscale("log")
    ax[1].set_title("Physics-consistency reduction"); ax[1].grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "coupling_zoom.png", dpi=120); plt.close()
    print(f"saved {OUT}/coupling_zoom.png")

# Summary numbers
summary = {
    "n_steps": int(len(rows)),
    "final_step": int(steps[-1]),
    "final_fm": float(fm[-1]),
    "final_sc": float(sc[-1]),
    "final_ph": float(ph[-1]),
    "final_tot": float(tot[-1]),
    "fm_start": float(fm[0]),
    "coupling_activated_at_step": coupling_step,
}
if coupling_step is not None:
    sc_at_coupling_start = float(sc[coupling_idx[0]])
    sc_final = float(sc[coupling_idx[-1]])
    summary.update({
        "sc_at_coupling_start": sc_at_coupling_start,
        "sc_final": sc_final,
        "sc_relative_reduction_pct": float((sc_final / sc_at_coupling_start - 1) * 100),
    })
json.dump(summary, open(OUT / "loss_summary.json", "w"), indent=2)
print(f"\nSummary:\n{json.dumps(summary, indent=2)}")
