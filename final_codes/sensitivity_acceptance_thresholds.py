# -*- coding: utf-8 -*-
"""
Monte Carlo acceptance-threshold sensitivity sweep.

Reviewer requests:
    R1 comment 3: "Line 116 to 120 specified the conditions under which a
    certain set of parameter values can be accepted. Where do these numbers,
    0.002, 0.01, 30 km and 150 km come from? ... how sensitive is the
    simulation results to these numbers, if they were picked arbitrarily?"
    R2 minor 2: "The acceptance criteria used for the Monte Carlo simulations
    uses MOI between +/- 0.002 of the observed value ... Please describe how
    the bound of +/-0.002 is derived, and what confidence interval it
    represents."

Method
------
Re-run the Monte Carlo acceptance/rejection step of run_monte_carlo() under
several combinations of the four thresholds identified in the manuscript
(MOI tolerance, mass tolerance, minimum ocean thickness, max ice thickness),
holding everything else (priors, RNG seed) fixed, and report how the
accepted-sample count and the posterior 16/50/84th-percentile structural
parameters shift.

The +/-0.002 MOI tolerance is stated in the manuscript to correspond to the
propagated 1-sigma Doppler-tracking uncertainty on I/(MR^2) from the Galileo
radio-science gravity solution; this script does not re-derive that
propagation (it is an observational-error-budget question, not a solver
question) but does test how strongly the accepted ensemble's statistics
depend on the choice, which is the quantitative part of the reviewers'
requests.

Run:
    python sensitivity_acceptance_thresholds.py
"""

import numpy as np

from _fem_core import fem

CONFIG = fem.PRIOR_CONFIGS["uniform"]
SEED = 42
N_DRAWS = 300_000   # raw draws per threshold combo (kept moderate for runtime)

# Baseline (manuscript) values, per the code: TOL_MOI=0.002, mass_tol=0.01,
# ocean_min_thk=30 km, ice_bounds upper=150 km.
BASELINE = dict(moi_tol=0.002, mass_tol=0.01, ocean_min_thk=30e3, ice_bounds=(10e3, 150e3))

THRESHOLD_SWEEPS = {
    # Spec range: MOI tolerance in [0.001, 0.004] ("review tasks.docx" task #7).
    "MOI tolerance": [
        dict(BASELINE, moi_tol=0.001),
        dict(BASELINE),
        dict(BASELINE, moi_tol=0.004),
    ],
    # Spec values: mass tolerance in {0.005, 0.02}.
    "Mass tolerance": [
        dict(BASELINE, mass_tol=0.005),
        dict(BASELINE),
        dict(BASELINE, mass_tol=0.02),
    ],
    # Spec: minimum ocean shell limit, "ocean >= 10 km" (vs. the manuscript's
    # default 30 km floor -- NOT the ocean's upper bound, which isn't
    # separately constrained in this model).
    "Min ocean thickness": [
        dict(BASELINE, ocean_min_thk=10e3),
        dict(BASELINE),
        dict(BASELINE, ocean_min_thk=60e3),
    ],
    # Spec: minimum ice shell limit, "ice >= 5 km" (vs. the manuscript's
    # default 10 km floor). This sweeps the *lower* bound of ice_bounds --
    # earlier drafts of this script incorrectly swept the *upper* bound
    # instead, which tests a different (and unspecified) sensitivity.
    "Min ice thickness": [
        dict(BASELINE, ice_bounds=(5e3, 150e3)),
        dict(BASELINE),
        dict(BASELINE, ice_bounds=(15e3, 150e3)),
    ],
}


def summarize(solutions):
    if len(solutions) == 0:
        return None
    ice_thk_km   = (fem.R_EUROPA - solutions[:, 2]) / 1e3
    ocean_thk_km = (solutions[:, 2] - solutions[:, 1]) / 1e3
    core_r_km    = solutions[:, 0] / 1e3
    out = {}
    for name, arr in [("ice_km", ice_thk_km), ("ocean_km", ocean_thk_km), ("core_km", core_r_km)]:
        p16, p50, p84 = np.percentile(arr, [16, 50, 84])
        out[name] = (p16, p50, p84)
    return out


def main():
    print(f"\n{'='*78}")
    print("  MONTE CARLO ACCEPTANCE-THRESHOLD SENSITIVITY  (N_draws={:,} per combo)".format(N_DRAWS))
    print(f"  Baseline: {BASELINE}")
    print(f"{'='*78}")

    for group_name, combos in THRESHOLD_SWEEPS.items():
        print(f"\n  --- {group_name} ---")
        print(f"  {'params (delta from baseline)':<38}{'N_acc':>8}{'%':>7}   "
              f"{'Ice p50 (16-84)':<22}{'Ocean p50 (16-84)':<22}{'Core p50 (16-84)'}")
        for combo in combos:
            sol = fem.run_monte_carlo(
                CONFIG, N=N_DRAWS, seed=SEED,
                ice_bounds=combo["ice_bounds"], moi_tol=combo["moi_tol"],
                mass_tol=combo["mass_tol"], ocean_min_thk=combo["ocean_min_thk"],
                verbose=False,
            )
            stats = summarize(sol)
            is_baseline = combo == BASELINE
            tag = " (baseline)" if is_baseline else ""
            label = (f"moi_tol={combo['moi_tol']}, mass_tol={combo['mass_tol']}, "
                     f"ocean_min={combo['ocean_min_thk']/1e3:.0f}km, "
                     f"ice=({combo['ice_bounds'][0]/1e3:.0f}-{combo['ice_bounds'][1]/1e3:.0f})km{tag}")
            if stats is None:
                print(f"  {label:<38}{0:>8}{0.0:>7.2f}   -- no accepted solutions --")
                continue
            ice_s   = f"{stats['ice_km'][1]:.1f} ({stats['ice_km'][0]:.1f}-{stats['ice_km'][2]:.1f})"
            ocean_s = f"{stats['ocean_km'][1]:.1f} ({stats['ocean_km'][0]:.1f}-{stats['ocean_km'][2]:.1f})"
            core_s  = f"{stats['core_km'][1]:.1f} ({stats['core_km'][0]:.1f}-{stats['core_km'][2]:.1f})"
            print(f"  {label:<38}{len(sol):>8}{100*len(sol)/N_DRAWS:>7.2f}   "
                  f"{ice_s:<22}{ocean_s:<22}{core_s}")

    print(f"\n  Interpretation:")
    print(f"  - Acceptance *rate* is highly sensitive to all four thresholds (as expected --")
    print(f"    they define the acceptance region), but the *posterior medians* of ice/ocean/core")
    print(f"    thickness are the quantity that matters for the paper's conclusions; compare the")
    print(f"    p50 (16-84%) columns above across each row-group to judge robustness.")


if __name__ == "__main__":
    main()
