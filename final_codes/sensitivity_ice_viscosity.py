# -*- coding: utf-8 -*-
"""
Ice reference viscosity sensitivity sweep.

Reviewer request (R2 minor 6, re: Eq. 14 / Table 8):
    "The reference viscosity of ice near the melting point is dependent on
    the unknown grain size, and is typically explored as a range from 1e13
    to 1e15 Pa s because this has implications for convection and tidal
    heating." / "How does ice shell Nu change with different reference
    viscosity?"

Method
------
Sweep ETA_ICE_REF (viscosity of ice Ih at the melting point, used as the
Arrhenius reference in the stagnant-lid Nusselt-number scaling) across the
grain-size-driven uncertainty range 1e13 - 1e15 Pa s, holding geometry and
tidal partitioning fixed at a representative structure. Report the
resulting ice-shell Nusselt number Nu_i, the basal Rayleigh number Ra_b,
and the convective regime (conducting sub-critical lid vs. convecting).

Run:
    python sensitivity_ice_viscosity.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _fem_core import fem

R_CORE, R_MANTLE, R_OCEAN = 600e3, 1450e3, 1540e3
RHO_CORE, RHO_MANTLE = 5500.0, 3500.0

ETA_ICE_REF_BASELINE = fem.ETA_ICE_REF   # 1e14 Pa s
ETA_ICE_REF_VALUES = np.geomspace(1e13, 1e15, 9)


def main():
    print(f"\n{'='*70}")
    print("  ICE REFERENCE VISCOSITY SENSITIVITY SWEEP  (Table 8)")
    print(f"  Baseline ETA_ICE_REF = {ETA_ICE_REF_BASELINE:.2e} Pa s")
    print(f"  Sweep range (grain-size uncertainty): "
          f"{ETA_ICE_REF_VALUES[0]:.2e} - {ETA_ICE_REF_VALUES[-1]:.2e} Pa s")
    print(f"{'='*70}\n")

    kappa_ice = fem.K_ICE / (fem.RHO_ICE * fem.CP_ICE)
    ice_thk = fem.R_EUROPA - R_OCEAN

    rows = []
    for eta in ETA_ICE_REF_VALUES:
        r, T, diag = fem.compute_temperature(
            R_CORE, R_MANTLE, R_OCEAN, RHO_CORE, RHO_MANTLE,
            eta_ice_ref=eta, return_diagnostics=True,
        )
        i_idx = np.where(r >= R_OCEAN)[0]
        Nu_i, Ra_b = fem.stagnant_lid_nusselt(
            T[i_idx[0]], T[i_idx[-1]], ice_thk,
            fem.RHO_ICE, fem.G_SURF, fem.ALPHA_ICE, kappa_ice,
            eta, fem.T_ICE_REF, fem.E_ICE, Nu_cap=50.0, return_ra=True,
        )
        regime = "convecting" if Nu_i > 1.001 else "conducting (sub-critical)"
        rows.append(dict(eta=eta, Nu_i=Nu_i, Ra_b=Ra_b, regime=regime,
                          T_seafloor=diag["T_seafloor"], q_surface=diag["q_surface"]))
        print(f"  eta_ice_ref={eta:9.2e} Pa s  ->  Nu_i={Nu_i:6.2f}   "
              f"Ra_b={Ra_b:10.3e}   [{regime}]   "
              f"q_surface={diag['q_surface']*1e3:6.2f} mW/m^2")

    print(f"\n  Table 8 (regenerated): ice-shell Nu vs. reference viscosity")
    print(f"  {'eta_ice_ref (Pa s)':<22}{'Ra_b':<14}{'Nu_i':<8}{'Regime'}")
    for row in rows:
        print(f"  {row['eta']:<22.2e}{row['Ra_b']:<14.3e}{row['Nu_i']:<8.2f}{row['regime']}")

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    Nu_arr = np.array([row["Nu_i"] for row in rows])
    Ra_arr = np.array([row["Ra_b"] for row in rows])

    axes[0].loglog(ETA_ICE_REF_VALUES, Nu_arr, "o-", color="#1A5276")
    axes[0].axvline(ETA_ICE_REF_BASELINE, color="gray", ls="--", lw=1.0, label="Baseline (1e14 Pa s)")
    axes[0].set_xlabel("Ice reference viscosity (Pa s)")
    axes[0].set_ylabel("Ice-shell Nusselt number Nu_i")
    axes[0].set_title("Nu_i vs. ice reference viscosity")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3, which="both")

    axes[1].loglog(Ra_arr, Nu_arr, "o-", color="#7D6608")
    axes[1].set_xlabel("Basal Rayleigh number Ra_b")
    axes[1].set_ylabel("Nu_i")
    axes[1].set_title("Nu_i vs. Ra_b (stagnant-lid scaling)")
    axes[1].grid(alpha=0.3, which="both")

    plt.tight_layout()
    plt.savefig("ice_viscosity_sensitivity.png", dpi=200, bbox_inches="tight")
    print(f"\n  Saved figure -> ice_viscosity_sensitivity.png")


if __name__ == "__main__":
    main()
