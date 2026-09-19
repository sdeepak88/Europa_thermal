# -*- coding: utf-8 -*-
"""
Constant vs. temperature-dependent ice thermal conductivity comparison.

Reviewer request (R2 major 8):
    "In the thermal model (Section 4), the thermal conductivity of ice is
    assumed constant, however experiments have shown that ice Ih thermal
    conductivity is proportional to 1/T (Hobbs 1974; Andersson and Inaba
    2005). This is likely important for Europa where ice shell temperatures
    vary from 110 K at the surface to 270 K at the base. Please address the
    effects of a temperature-dependent thermal conductivity on the
    resulting temperature profiles and consider implementing it in the
    thermal model."

Implementation
--------------
compute_temperature() in FEM+GravityInversion.py now accepts
`ice_k_mode="constant"` (the original K_ICE=2.5 W/m/K baseline) or
`ice_k_mode="temperature_dependent"`, which evaluates
k_ice_andersson_inaba(T) = 567/T W/m/K at each node, each fixed-point
iteration, using the current temperature estimate (see FEM+GravityInversion.py
section 6). This script runs both modes for a representative structure and
for the ensemble-median structures of both prior variants (if cached
solutions are available), and reports the resulting change in the ice
geotherm, conductive-lid thickness, and surface heat flux.

Run:
    python sensitivity_ice_conductivity.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _fem_core import fem

R_CORE, R_MANTLE, R_OCEAN = 600e3, 1450e3, 1540e3
RHO_CORE, RHO_MANTLE = 5500.0, 3500.0


def run_pair(R_core, R_mantle, R_ocean, rho_core, rho_mantle, label):
    r_c, T_c, diag_c = fem.compute_temperature(
        R_core, R_mantle, R_ocean, rho_core, rho_mantle,
        ice_k_mode="constant", return_diagnostics=True)
    r_t, T_t, diag_t = fem.compute_temperature(
        R_core, R_mantle, R_ocean, rho_core, rho_mantle,
        ice_k_mode="temperature_dependent", return_diagnostics=True)

    ice_idx_c = np.where(r_c >= R_ocean)[0]
    ice_idx_t = np.where(r_t >= R_ocean)[0]
    k_ai_at_surface = fem.k_ice_andersson_inaba(fem.T_SURFACE)
    k_ai_at_base     = fem.k_ice_andersson_inaba(T_c[ice_idx_c[0]])

    print(f"\n  --- {label} ---")
    print(f"    k_ice (Andersson & Inaba) at T_surface={fem.T_SURFACE:.1f} K: "
          f"{k_ai_at_surface:.3f} W/m/K   (constant baseline K_ICE={fem.K_ICE:.2f})")
    print(f"    k_ice (Andersson & Inaba) at ice-base T~{T_c[ice_idx_c[0]]:.1f} K: "
          f"{k_ai_at_base:.3f} W/m/K")
    print(f"    Nu_i           : constant-k={diag_c['Nu_i']:.3f}   "
          f"temp-dep-k={diag_t['Nu_i']:.3f}   "
          f"(identical by construction -- the stagnant-lid Ra/Nu scaling still uses "
          f"the constant-K_ICE diffusivity; see FEM+GravityInversion.py note)")
    print(f"    q_surface      : constant-k={diag_c['q_surface']*1e3:.2f} mW/m^2   "
          f"temp-dep-k={diag_t['q_surface']*1e3:.2f} mW/m^2   "
          f"(delta {1e3*(diag_t['q_surface']-diag_c['q_surface']):+.2f} mW/m^2)")
    print(f"    T_seafloor     : constant-k={diag_c['T_seafloor']:.3f} K   "
          f"temp-dep-k={diag_t['T_seafloor']:.3f} K")

    # Depth at which the ice reaches within 1 K of the seafloor temperature.
    def conductive_lid_depth(r, T, i_idx):
        depth = (fem.R_EUROPA - r[i_idx]) / 1e3
        T_ice = T[i_idx]
        near_base = np.where(np.abs(T_ice - T_ice[0]) < 1.0)[0]
        return depth[near_base[-1]] if len(near_base) else depth[0]

    lid_c = conductive_lid_depth(r_c, T_c, ice_idx_c)
    lid_t = conductive_lid_depth(r_t, T_t, ice_idx_t)
    print(f"    Isothermal-base extent (within 1K of T_seafloor): "
          f"constant-k={lid_c:.2f} km   temp-dep-k={lid_t:.2f} km")

    return (r_c, T_c, ice_idx_c), (r_t, T_t, ice_idx_t), label


def main():
    print(f"\n{'='*70}")
    print("  ICE THERMAL CONDUCTIVITY: CONSTANT vs. TEMPERATURE-DEPENDENT")
    print(f"  k_ice_andersson_inaba(T) = 567/T  W/m/K  (R2 major 8)")
    print(f"{'='*70}")

    runs = []
    runs.append(run_pair(R_CORE, R_MANTLE, R_OCEAN, RHO_CORE, RHO_MANTLE,
                          "Representative structure (25 km ice, 90 km ocean)"))

    # Also compare for the ensemble-median structure of any cached MC runs,
    # if present (so this reflects the manuscript's actual accepted models,
    # not just one illustrative case).
    for tag in ["uniform", "geochemical", "smoketest", "smoketest2"]:
        sol_file = f"solutions_{tag}.npy"
        if os.path.exists(sol_file):
            sol = np.load(sol_file)
            R_core_med, R_mantle_med, R_ocean_med = np.median(sol[:, 0]), np.median(sol[:, 1]), np.median(sol[:, 2])
            rho_core_med, rho_mantle_med = np.median(sol[:, 3]), np.median(sol[:, 4])
            runs.append(run_pair(
                R_core_med, R_mantle_med, R_ocean_med, rho_core_med, rho_mantle_med,
                f"Median structure, tag={tag!r} (N={len(sol)})"))

    # --- Plot the representative-structure comparison ---
    (r_c, T_c, ice_idx_c), (r_t, T_t, ice_idx_t), label = runs[0]
    depth_c = (fem.R_EUROPA - r_c[ice_idx_c]) / 1e3
    depth_t = (fem.R_EUROPA - r_t[ice_idx_t]) / 1e3

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].plot(T_c[ice_idx_c], depth_c, color="#7D6608", lw=2.0, label="Constant K_ICE")
    axes[0].plot(T_t[ice_idx_t], depth_t, color="#1A5276", lw=2.0, ls="--",
                 label="T-dependent (Andersson & Inaba 2005)")
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Temperature (K)")
    axes[0].set_ylabel("Depth below surface (km)")
    axes[0].set_title(f"Ice-shell geotherm\n{label}")
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3)

    T_grid = np.linspace(110, 270, 100)   # Andersson & Inaba (2005) validity range
    axes[1].plot(T_grid, np.full_like(T_grid, fem.K_ICE), color="#7D6608", lw=2.0,
                 label=f"Constant K_ICE = {fem.K_ICE} W/m/K")
    axes[1].plot(T_grid, fem.k_ice_andersson_inaba(T_grid), color="#1A5276", lw=2.0, ls="--",
                 label="Andersson & Inaba (2005): 567/T")
    axes[1].set_xlabel("Temperature (K)")
    axes[1].set_ylabel("Ice thermal conductivity (W/m/K)")
    axes[1].set_title("Conductivity parametrisation")
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("ice_conductivity_comparison.png", dpi=200, bbox_inches="tight")
    print(f"\n  Saved figure -> ice_conductivity_comparison.png")


if __name__ == "__main__":
    main()
