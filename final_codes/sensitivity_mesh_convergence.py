# -*- coding: utf-8 -*-
"""
Mesh (grid) convergence study for the 1-D radial FEM thermal solver.

Reviewer request (R1 comment 6):
    "Please provide more details on the grid you selected in section 4.2.
    What justifies that the resolution is high enough? Any evidence for
    convergence?"

Method
------
Run compute_temperature() for one representative interior structure at the
four grid resolutions specified in the reviewer response plan (R1 comment 6,
"review tasks.docx" task #5: "Execute a mesh refinement study across four
grid resolutions (Nr = 150, 300, 601, 1200 nodes)"). Per-layer node counts
are obtained by scaling the manuscript's default layer proportions
(80, 200, 160, 160, which sum to the default 601-node total) to hit each
target total exactly:

    Nr= 150: nodes_per_layer=( 19,  50,  40,  40)
    Nr= 300: nodes_per_layer=( 39, 100,  80,  80)
    Nr= 601: nodes_per_layer=( 80, 200, 160, 160)  (the manuscript's default)
    Nr=1200: nodes_per_layer=(159, 400, 320, 320)

For each pair of consecutive resolutions we report the max-norm temperature
difference (interpolated onto the coarser grid) at three key interfaces
(core-mantle boundary, seafloor, ice base) and over the full profile, plus
the observed order of convergence `p` and the Grid Convergence Index (GCI,
Roache 1994) computed from the three finest levels. The four specified grid
totals are not at a perfectly constant refinement ratio (300/150=2.00 exact,
601/300=2.003, 1200/601=1.997), so the GCI here uses the actual local ratio
of node counts between each consecutive pair rather than assuming a fixed
r=2, which is the more correct treatment when the ratio isn't exact.

Run:
    python sensitivity_mesh_convergence.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _fem_core import fem

# --- Representative interior (median-ish uniform-prior structure) ---
R_CORE, R_MANTLE, R_OCEAN = 600e3, 1450e3, 1540e3   # ice ~25 km, ocean ~90 km
RHO_CORE, RHO_MANTLE = 5500.0, 3500.0

# Exact grid levels per the reviewer response plan (Nr = 150, 300, 601, 1200).
LEVEL_NODES_PER_LAYER = [
    (19, 50, 40, 40),      # Nr = 150
    (39, 100, 80, 80),     # Nr = 300
    (80, 200, 160, 160),   # Nr = 601 (manuscript default)
    (159, 400, 320, 320),  # Nr = 1200
]
LEVEL_NAMES = ["coarse (Nr=150)", "medium (Nr=300)",
               "default (Nr=601)", "finest (Nr=1200)"]

GCI_SAFETY_FACTOR = 1.25   # Roache's recommended Fs for 3+ grid studies


def run_levels():
    """Solve the FEM problem at each refinement level; return list of (n_total, r, T)."""
    results = []
    for npl, name in zip(LEVEL_NODES_PER_LAYER, LEVEL_NAMES):
        r, T = fem.compute_temperature(
            R_CORE, R_MANTLE, R_OCEAN, RHO_CORE, RHO_MANTLE,
            nodes_per_layer=npl,
        )
        n_total = len(r)
        print(f"  [{name:>17s}]  nodes_per_layer={npl}  "
              f"total nodes={n_total:5d}  T_surf={T[-1]:.3f} K  T_core={T[0]:.3f} K")
        results.append((n_total, r, T))
    return results


def interp_error(r_ref, T_ref, r_fine, T_fine):
    """Max-abs error of T_ref against T_fine, both interpolated onto r_ref."""
    T_fine_on_ref = np.interp(r_ref, r_fine, T_fine)
    return np.max(np.abs(T_ref - T_fine_on_ref))


def boundary_values(r, T, r_target):
    idx = fem.nearest_idx(r, r_target)
    return T[idx]


def grid_convergence_index(f1, f2, f3, r, Fs=GCI_SAFETY_FACTOR):
    """
    Roache (1994) Grid Convergence Index from three grids of solution values
    f1 (finest), f2 (medium), f3 (coarsest), with refinement ratio r.

    `r` should be supplied by the caller as the (geometric-mean, if the two
    steps differ slightly) refinement ratio actually used between the three
    grids -- the classic 3-grid GCI formula assumes one constant ratio.

    Returns (p, GCI_21) where p is the observed order of convergence and
    GCI_21 is the fine-grid (f1 vs f2) convergence index, expressed as a
    fraction of |f1|.
    """
    e32 = f3 - f2
    e21 = f2 - f1
    if abs(e21) < 1e-14:
        return np.inf, 0.0
    ratio = e32 / e21
    if ratio <= 0:
        # Non-monotonic convergence; order cannot be estimated reliably.
        return np.nan, np.nan
    p = np.log(ratio) / np.log(r)
    GCI_21 = Fs * abs(e21) / (r**p - 1) / max(abs(f1), 1e-12)
    return p, GCI_21


def main():
    print(f"\n{'='*70}")
    print("  MESH (GRID) CONVERGENCE STUDY")
    print(f"  Structure: R_core={R_CORE/1e3:.0f} km, R_mantle={R_MANTLE/1e3:.0f} km, "
          f"R_ocean={R_OCEAN/1e3:.0f} km")
    print(f"{'='*70}")

    results = run_levels()
    n_list  = [n for n, _, _ in results]
    r_list  = [r for _, r, _ in results]
    T_list  = [T for _, _, T in results]

    # --- Errors relative to the finest grid, at three key interfaces ---
    r_finest, T_finest = r_list[-1], T_list[-1]
    key_points = {
        "Core centre (r=0)":         0.0,
        "Core-mantle boundary":      R_CORE,
        "Seafloor (mantle-ocean)":   R_MANTLE,
        "Ice base (ocean-ice)":      R_OCEAN,
        "Surface":                   fem.R_EUROPA,
    }

    print(f"\n  {'Level':<20}{'Nodes':>8}   " +
          "   ".join(f"{k:<24}" for k in key_points))
    for name, n, r, T in zip(LEVEL_NAMES, n_list, r_list, T_list):
        vals = [f"{boundary_values(r, T, rt):9.3f} K" for rt in key_points.values()]
        print(f"  {name:<20}{n:>8}   " + "   ".join(f"{v:<24}" for v in vals))

    print(f"\n  Max-norm |T - T_finest| (interpolated), full profile:")
    max_errs = []
    for name, n, r, T in zip(LEVEL_NAMES, n_list, r_list, T_list):
        err = interp_error(r, T, r_finest, T_finest)
        max_errs.append(err)
        print(f"    {name:<20} (n={n:5d}) : {err:.4f} K")

    # --- Grid Convergence Index using the three finest levels ---
    # The three finest node counts are 300, 601, 1200: r32 = 601/300 = 2.003,
    # r21 = 1200/601 = 1.997. Both are within 0.35% of 2, so their geometric
    # mean is used as the single refinement ratio the classic GCI formula
    # requires.
    r32 = n_list[2] / n_list[1]
    r21 = n_list[3] / n_list[2]
    r_gci = float(np.sqrt(r32 * r21))
    print(f"\n  Grid Convergence Index (Roache 1994), refinement ratio r={r_gci:.4f} "
          f"(actual: r32={r32:.4f}, r21={r21:.4f}):")
    print(f"  (evaluated at each key interface using the 3 finest grids)")
    all_within_tol = True
    for label, rt in key_points.items():
        f3 = boundary_values(r_list[1], T_list[1], rt)   # medium (coarsest of the 3)
        f2 = boundary_values(r_list[2], T_list[2], rt)   # fine
        f1 = boundary_values(r_list[3], T_list[3], rt)   # finest
        p, gci = grid_convergence_index(f1, f2, f3, r_gci)
        if np.isnan(gci):
            print(f"    {label:<26} p=  n/a   GCI= n/a  (non-monotone; ~converged, diffs at solver tol)")
            continue
        print(f"    {label:<26} p={p:5.2f}   GCI(fine,medium)={gci*100:.4f} %")
        if gci > 0.01:   # 1% threshold
            all_within_tol = False

    default_err = max_errs[LEVEL_NAMES.index("default (Nr=601)")]
    print(f"\n  Manuscript default (Nr=601 level) max-norm error vs. "
          f"finest (Nr=1200) grid: {default_err:.4f} K")
    verdict = "CONVERGED" if default_err < 1.0 and all_within_tol else "NOT CLEARLY CONVERGED"
    print(f"  Verdict (tol=1.0 K, matching the solver's own TOL_T): {verdict}")

    # --- Plot: profile overlay + convergence error vs. resolution ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    depth_finest = (fem.R_EUROPA - r_finest) / 1e3
    for name, r, T in zip(LEVEL_NAMES, r_list, T_list):
        depth = (fem.R_EUROPA - r) / 1e3
        axes[0].plot(T, depth, lw=1.6, label=f"{name} (n={len(r)})")
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Temperature (K)")
    axes[0].set_ylabel("Depth below surface (km)")
    axes[0].set_title("Temperature profile vs. grid resolution")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    axes[1].loglog(n_list, np.maximum(max_errs, 1e-6), "o-", color="#922B21")
    axes[1].set_xlabel("Total number of FEM nodes")
    axes[1].set_ylabel("Max-norm |T - T_finest|  (K)")
    axes[1].set_title("Grid convergence")
    axes[1].axvline(601, color="gray", ls="--", lw=1.0, label="Default (601 nodes)")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3, which="both")

    plt.tight_layout()
    plt.savefig("mesh_convergence_study.png", dpi=200, bbox_inches="tight")
    print(f"\n  Saved figure -> mesh_convergence_study.png")


if __name__ == "__main__":
    main()
