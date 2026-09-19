# -*- coding: utf-8 -*-
"""
Global tidal power sensitivity sweep.

Reviewer request (R2 major 3):
    "This study makes a big assumption of the heat budget of Europa,
    assigning a ... total tidal heating of 1 TW. ... our knowledge of the
    tidal heating of Europa is far less constrained without a measured
    complex tidal Love number. Please address the uncertainty in the
    P_tidal = 1 TW value and its effect on the results."

Method
------
Sweep the total global tidal power P_tidal (module default: 1e12 W) across
the range of published Europa tidal-heating estimates, log-spaced from
0.1 TW to 5 TW, holding the interior structure fixed at a representative
(median-ish) accepted structure. For each value, solve the FEM thermal
problem and report the resulting mantle/ice Nusselt numbers, core-mantle
and seafloor temperatures, and boundary heat fluxes.

Note: the current model takes ice/ocean/mantle/core layer *thicknesses* as
Monte-Carlo inputs (constrained by gravity+MOI), not as an output of the
thermal solve -- there is no free "equilibrium ice shell thickness" internal
to compute_temperature for fixed geometry. This sweep therefore reports the
sensitivity of the solved *temperature field and convective vigor* to
P_tidal at fixed geometry, which is what the FEM solver actually predicts;
translating that into an equilibrium-thickness sensitivity would require an
outer shell-thickness iteration, which is out of scope for this solver.

Run:
    python sensitivity_tidal_power.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _fem_core import fem

R_CORE, R_MANTLE, R_OCEAN = 600e3, 1450e3, 1540e3
RHO_CORE, RHO_MANTLE = 5500.0, 3500.0

P_TIDAL_BASELINE = fem.P_TIDAL   # 1e12 W
P_TIDAL_VALUES = np.geomspace(0.1e12, 5e12, 12)


def main():
    print(f"\n{'='*70}")
    print("  GLOBAL TIDAL POWER SENSITIVITY SWEEP")
    print(f"  Baseline P_tidal = {P_TIDAL_BASELINE:.3e} W")
    print(f"  Sweep range: {P_TIDAL_VALUES[0]:.3e} - {P_TIDAL_VALUES[-1]:.3e} W")
    print(f"{'='*70}\n")

    rows = []
    profiles = []
    for P in P_TIDAL_VALUES:
        r, T, diag = fem.compute_temperature(
            R_CORE, R_MANTLE, R_OCEAN, RHO_CORE, RHO_MANTLE,
            p_tidal=P, return_diagnostics=True,
        )
        rows.append(dict(P_tidal=P, **diag))
        profiles.append((r, T))
        print(f"  P_tidal={P:9.3e} W  ->  Nu_m={diag['Nu_m']:7.2f}  Nu_i={diag['Nu_i']:5.2f}  "
              f"T_cmb={diag['T_cmb']:8.2f} K  T_seafloor={diag['T_seafloor']:7.3f} K  "
              f"q_surf={diag['q_surface']*1e3:6.2f} mW/m^2  q_seafloor={diag['q_seafloor']*1e3:6.2f} mW/m^2  "
              f"conduction_branch={diag['conduction_branch']}")

    Nu_m_arr    = np.array([row["Nu_m"] for row in rows])
    Tcmb_arr    = np.array([row["T_cmb"] for row in rows])
    q_seaf_arr  = np.array([row["q_seafloor"] for row in rows])

    baseline_idx = int(np.argmin(np.abs(P_TIDAL_VALUES - P_TIDAL_BASELINE)))
    print(f"\n  Relative to baseline (P_tidal={P_TIDAL_BASELINE:.2e} W, "
          f"Nu_m={Nu_m_arr[baseline_idx]:.2f}, T_cmb={Tcmb_arr[baseline_idx]:.1f} K):")
    print(f"    At P_tidal=0.1 TW : Nu_m={Nu_m_arr[0]:.2f} "
          f"({100*(Nu_m_arr[0]/Nu_m_arr[baseline_idx]-1):+.1f} %),  "
          f"T_cmb={Tcmb_arr[0]:.1f} K ({Tcmb_arr[0]-Tcmb_arr[baseline_idx]:+.1f} K)")
    print(f"    At P_tidal=5.0 TW : Nu_m={Nu_m_arr[-1]:.2f} "
          f"({100*(Nu_m_arr[-1]/Nu_m_arr[baseline_idx]-1):+.1f} %),  "
          f"T_cmb={Tcmb_arr[-1]:.1f} K ({Tcmb_arr[-1]-Tcmb_arr[baseline_idx]:+.1f} K)")
    print(f"    Seafloor heat flux spans {q_seaf_arr.min()*1e3:.1f} - "
          f"{q_seaf_arr.max()*1e3:.1f} mW/m^2 over the swept range "
          f"(vs. {q_seaf_arr[baseline_idx]*1e3:.1f} mW/m^2 at baseline).")

    # --- Plot ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    axes[0].semilogx(P_TIDAL_VALUES, Nu_m_arr, "o-", color="#7D6608", label="Nu_m (mantle)")
    axes[0].semilogx(P_TIDAL_VALUES, [row["Nu_i"] for row in rows], "s-",
                      color="#1A5276", label="Nu_i (ice)")
    axes[0].axvline(P_TIDAL_BASELINE, color="gray", ls="--", lw=1.0, label="Baseline (1 TW)")
    axes[0].set_xlabel("Global tidal power P_tidal (W)")
    axes[0].set_ylabel("Nusselt number")
    axes[0].set_title("Convective vigor vs. tidal power")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3, which="both")

    axes[1].semilogx(P_TIDAL_VALUES, Tcmb_arr - 273.15, "o-", color="#922B21")
    axes[1].axvline(P_TIDAL_BASELINE, color="gray", ls="--", lw=1.0)
    axes[1].set_xlabel("Global tidal power P_tidal (W)")
    axes[1].set_ylabel("Core-mantle boundary T (degC)")
    axes[1].set_title("Core temperature vs. tidal power")
    axes[1].grid(alpha=0.3, which="both")

    axes[2].semilogx(P_TIDAL_VALUES, np.array([row["q_seafloor"] for row in rows]) * 1e3,
                      "o-", color="#2980B9", label="Seafloor")
    axes[2].semilogx(P_TIDAL_VALUES, np.array([row["q_surface"] for row in rows]) * 1e3,
                      "s-", color="#9B59B6", label="Surface")
    axes[2].axvline(P_TIDAL_BASELINE, color="gray", ls="--", lw=1.0)
    axes[2].set_xlabel("Global tidal power P_tidal (W)")
    axes[2].set_ylabel("Conductive heat flux (mW/m^2)")
    axes[2].set_title("Boundary heat flux vs. tidal power")
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.3, which="both")

    plt.tight_layout()
    plt.savefig("tidal_power_sensitivity.png", dpi=200, bbox_inches="tight")
    print(f"\n  Saved figure -> tidal_power_sensitivity.png")


if __name__ == "__main__":
    main()
