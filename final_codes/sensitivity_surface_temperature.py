# -*- coding: utf-8 -*-
"""
Surface boundary-condition sensitivity sweep.

Reviewer request (R1 comment 7):
    "The top boundary condition uses a fixed surface temperature. However,
    the surface temperature of Europa can vary by about 50 degC (e.g. Travis
    et al 2012). Please discuss the sensitivity of the model to the surface
    temperature."

Method
------
Sweep the fixed surface Dirichlet boundary condition T_surface across the
equatorial-to-polar range noted by the reviewer (the baseline 113.15 K
+/- 25 K, i.e. a 50 K / 50 degC full range spanning ~88-138 K, bracketing
Europa's sub-solar-equator to polar-night surface temperatures) for the
median structure of a representative accepted ensemble. Report the resulting
perturbation to the conductive geotherm, the ice-shell thermal gradient, and
the mantle/ice Nusselt numbers and heat fluxes.

Run:
    python sensitivity_surface_temperature.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _fem_core import fem

R_CORE, R_MANTLE, R_OCEAN = 600e3, 1450e3, 1540e3
RHO_CORE, RHO_MANTLE = 5500.0, 3500.0

T_SURFACE_BASELINE = fem.T_SURFACE          # 113.15 K
HALF_RANGE = 25.0                           # K; 50 K/50 degC full range (Travis et al. 2012)
T_SURFACE_VALUES = np.linspace(T_SURFACE_BASELINE - HALF_RANGE,
                                T_SURFACE_BASELINE + HALF_RANGE, 9)


def main():
    print(f"\n{'='*70}")
    print("  SURFACE TEMPERATURE BOUNDARY-CONDITION SENSITIVITY")
    print(f"  Baseline T_surface = {T_SURFACE_BASELINE:.2f} K "
          f"({T_SURFACE_BASELINE-273.15:.2f} degC)")
    print(f"  Sweep range: {T_SURFACE_VALUES[0]:.2f} - {T_SURFACE_VALUES[-1]:.2f} K "
          f"(+/-{HALF_RANGE:.0f} K, Travis et al. 2012)")
    print(f"{'='*70}\n")

    rows = []
    profiles = []
    for Ts in T_SURFACE_VALUES:
        r, T, diag = fem.compute_temperature(
            R_CORE, R_MANTLE, R_OCEAN, RHO_CORE, RHO_MANTLE,
            t_surface=Ts, return_diagnostics=True,
        )
        # Near-surface thermal gradient (last 5 km of ice shell).
        depth = (fem.R_EUROPA - r) / 1e3
        near_surf = depth <= 5.0
        grad = np.polyfit(depth[near_surf], T[near_surf], 1)[0]   # K/km (negative: warms with depth)

        rows.append({
            "T_surface": Ts,
            "T_seafloor": diag["T_seafloor"],
            "Nu_i": diag["Nu_i"],
            "Nu_m": diag["Nu_m"],
            "q_surface": diag["q_surface"],
            "near_surface_gradient_K_per_km": grad,
        })
        profiles.append((r, T))
        print(f"  T_surface={Ts:7.2f} K ({Ts-273.15:6.2f} degC)  ->  "
              f"T_seafloor={diag['T_seafloor']:7.3f} K   Nu_i={diag['Nu_i']:5.2f}   "
              f"q_surface={diag['q_surface']*1e3:6.2f} mW/m^2   "
              f"near-surface grad={grad:7.2f} K/km")

    T_seafloor_arr = np.array([row["T_seafloor"] for row in rows])
    print(f"\n  Seafloor temperature range across the surface-BC sweep: "
          f"{T_seafloor_arr.max()-T_seafloor_arr.min():.4f} K "
          f"(min {T_seafloor_arr.min():.3f}, max {T_seafloor_arr.max():.3f})")
    print("  -> the ocean-ice boundary is pinned at the pressure-freezing point "
          "and the ocean is convectively isothermal, so the deep interior is "
          "essentially insulated from the surface-BC choice; the sensitivity "
          "is confined to the conductive ice-shell lid.")

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    cmap = plt.get_cmap("coolwarm")
    for i, (Ts, (r, T)) in enumerate(zip(T_SURFACE_VALUES, profiles)):
        depth = (fem.R_EUROPA - r) / 1e3
        ice_mask = depth <= 30
        col = cmap(i / (len(T_SURFACE_VALUES) - 1))
        axes[0].plot(T[ice_mask], depth[ice_mask], color=col, lw=1.6,
                     label=f"{Ts:.1f} K")
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Temperature (K)")
    axes[0].set_ylabel("Depth below surface (km)")
    axes[0].set_title("Ice-shell geotherm vs. surface boundary condition")
    axes[0].legend(fontsize=7, title="T_surface", ncol=2)
    axes[0].grid(alpha=0.3)

    axes[1].plot(T_SURFACE_VALUES - 273.15,
                 [row["near_surface_gradient_K_per_km"] for row in rows],
                 "o-", color="#922B21")
    axes[1].set_xlabel("T_surface (degC)")
    axes[1].set_ylabel("Near-surface thermal gradient (K/km, top 5 km)")
    axes[1].set_title("Near-surface gradient sensitivity")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("surface_temperature_sensitivity.png", dpi=200, bbox_inches="tight")
    print(f"\n  Saved figure -> surface_temperature_sensitivity.png")


if __name__ == "__main__":
    main()
