# -*- coding: utf-8 -*-
"""
Europa Interior Modeling — Multi-Gravity-Solution Comparison
=============================================================
Runs the gravity-constrained Monte Carlo + FEM thermal pipeline for each
published Europa J2/C20 gravity solution and plots all median temperature
profiles on a single comparison figure.



References
----------
Anderson et al. 1997  Science 276, 1236–1239
Anderson et al. 1998  Science 281, 2019–2022
Jacobson et al. 1999  AJ 119, 453–460   (reported in Gomez Casajus 2021 Table 1)
Gomez Casajus et al. 2021  Icarus 358, 114187
"""

import warnings
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

# ─────────────────────────────────────────────────────────────────────────────
# 1.  PHYSICAL CONSTANTS  (SI)
# ─────────────────────────────────────────────────────────────────────────────

G          = 6.67430e-11
R_EUROPA   = 1_565e3
M_EUROPA   = 4.7998e22
OMEGA      = 2.047e-5
M_JUP      = 1.898e27
E_ORB      = 0.009
Q_TIDAL    = 100
A_EUROPA   = 6.709e8
G_SURF     = 1.308          # m s⁻²  (matched to the canonical model)

RHO_ICE    = 920.0
RHO_OCEAN  = 1_030.0

K_ICE      = 2.5
K_OCEAN    = 0.49
K_MANTLE   = 3.5
K_CORE     = 25.0

CP_OCEAN   = 4_000
CP_MANTLE  = 1_000
CP_ICE     = 2_000          # J kg⁻¹ K⁻¹

ALPHA_OCEAN   = 3e-4
ALPHA_MANTLE  = 3e-5
ALPHA_ICE     = 1.6e-4      # K⁻¹

ETA_OCEAN  = 1e-3
ETA_MANTLE = 3e19

H_MANTLE_RADIOGENIC = 5e-8

T_SURFACE  = 113.15

# ── Rheology: Arrhenius temperature-dependent viscosity ──────────────────────
R_GAS = 8.314               # J mol⁻¹ K⁻¹

# Silicate mantle (olivine, diffusion creep)
ETA_MANTLE_REF = ETA_MANTLE # Pa s  reference viscosity at T_MANTLE_REF
T_MANTLE_REF   = 1600.0     # K
E_MANTLE       = 300e3      # J mol⁻¹  activation energy (dry olivine)

# Ice Ih (melting-point reference)
ETA_ICE_REF    = 1e14       # Pa s  viscosity at the melting point
T_ICE_REF      = 273.0      # K
E_ICE          = 60e3       # J mol⁻¹  activation energy for ice Ih

# Stagnant-lid convection scaling (Solomatov 1995; Reese et al. 1999)
A_NU           = 0.53       # prefactor in  Nu = A_NU · Ra_b^(1/3) · θ^(-4/3)

# Thermal convection iteration (stabilised fixed-point solver)
MAX_THERMAL_ITER = 300
RELAX_LOGK       = 0.10     # under-relaxation of log(k_eff)
NU_SEED          = 15.0     # initial mantle Nu (convecting seed)
NU_MANTLE_MAX    = 2_000.0  # hard cap on mantle Nu
TOL_T            = 0.5      # convergence tolerance on max |ΔT| between iters [K]

# ── Ocean convection (effective-conductivity treatment, UNCAPPED Nu_o) ───────
# The ocean is solved as a convecting layer whose conductivity is enhanced by
# Nu_o = NU_OCEAN_PREF · Ra_o^(1/3), with NO upper cap. For water this lands at
# Nu_o ~ 10⁶–10⁷, so the ocean becomes essentially isothermal. A convecting seed
# avoids a huge first-iteration transient.
NU_OCEAN_PREF    = 0.10    # prefactor in  Nu_o = NU_OCEAN_PREF · Ra_o^(1/3)
NU_OCEAN_SEED    = 1.0e4   # initial ocean Nu (convecting seed; uncapped solver)

# MOI acceptance tolerance
TOL_MOI  = 0.002

# ── Global tidal power  (canonical form: (21/2)·k2/Q, k2 = 0.23) ─────────────
n = 2.05e-5
P_TIDAL = 1e12

# Partition of the total tidal power across the dissipating layers
TIDAL_FRACTION = {
    "mantle": 0.20,   # silicate mantle (solid, viscoelastic)
    "ocean":  0.00,   # liquid water — negligible viscous tidal heating
    "ice":    0.80,   # ice shell (solid, viscoelastic)
}

# ─────────────────────────────────────────────────────────────────────────────
# 2.  PUBLISHED EUROPA J2 / C20 GRAVITY SOLUTIONS
# ─────────────────────────────────────────────────────────────────────────────
# Each entry carries:
#   j2       : J2 value (dimensionless, ×10⁻⁶ expressed as float)
#   j2_sigma : 1-σ uncertainty (same units)
#   note     : short description
#   color    : line color for comparison plot
#   ls       : linestyle
#
# All solutions apply the hydrostatic equilibrium constraint J2/C22 = 10/3,
# so C20 = -J2 / sqrt(5).  The Radau–Darwin relation then gives I_bar from J2.

GRAVITY_SOLUTIONS = {
    "Anderson et. al.,1997\n": {
        "j2"      : 432.0e-6,
        "j2_sigma": 24.0e-6,          # Gomez Casajus 2021 Table 1
        "note"    : "Anderson et al. 1997, Science 276, 1236 (2 flyby, E4+E6)",
        "color"   : "#E74C3C",        # red
        "ls"      : "--",
    },
    "Anderson et. al.,1998\n": {
        "j2"      : 435.5e-6,
        "j2_sigma": 8.2e-6,           # original code value
        "note"    : "Anderson et al. 1998, Science 281, 2019 (4 flybys, canonical)",
        "color"   : "#2980B9",        # blue
        "ls"      : "-",
    },
    "Jacobson et. al.,2000\n": {
        "j2"      : 417.0e-6,
        "j2_sigma": 6.0e-6,           # Gomez Casajus 2021 Table 1
        "note"    : "Jacobson et al. 1999",
        "color"   : "#27AE60",        # green
        "ls"      : "-.",
    },
    "GomezCasajus et. al.,2021\n": {
        "j2"      : 461.39e-6,
        "j2_sigma": 7.84e-6,          # large because hydrostatic not assumed in
                                      # the orbit fit; SOL-B then applies it post-hoc
        "note"    : "Gomez Casajus et al. 2021, Icarus 358, 114187 SOL-B (Juno-corrected, hydrostatic)",
        "color"   : "#8E44AD",        # purple
        "ls"      : ":",
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# 3.  RADAU–DARWIN  (one function, parameterised on J2)
# ─────────────────────────────────────────────────────────────────────────────

def rotational_parameter() -> float:
    return OMEGA**2 * R_EUROPA**3 / (G * M_EUROPA)

Q_ROT = rotational_parameter()


def ibar_from_j2(j2: float) -> float | None:
    """
    Derive normalised MOI  I/(MR²)  from J2 via the Radau–Darwin relation
    assuming hydrostatic equilibrium.

    kf = 6*J2 / (5 * q_rot)  — fluid Love number proxy
    I_bar = (2/3) * [1 - (2/5) * sqrt((4-kf)/(1+kf))]
    """
    kf = (6.0 * j2) / (5.0 * Q_ROT)
    if not (0 < kf < 4):
        return None
    return (2.0 / 3.0) * (1.0 - (2.0 / 5.0) * np.sqrt((4.0 - kf) / (1.0 + kf)))


# ─────────────────────────────────────────────────────────────────────────────
# 4.  INTERIOR STRUCTURE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def shell_mass(r1, r2, rho): return (4/3) * np.pi * rho * (r2**3 - r1**3)
def shell_moi (r1, r2, rho): return (8/15) * np.pi * rho * (r2**5 - r1**5)


def build_layers(R_core, R_mantle, R_ocean, rho_core, rho_mantle):
    return [
        (0.0,      R_core,   rho_core),
        (R_core,   R_mantle, rho_mantle),
        (R_mantle, R_ocean,  RHO_OCEAN),
        (R_ocean,  R_EUROPA, RHO_ICE),
    ]


def mass_and_ibar(layers):
    M = sum(shell_mass(r1, r2, rho) for r1, r2, rho in layers)
    I = sum(shell_moi (r1, r2, rho) for r1, r2, rho in layers)
    return M, I / (M * R_EUROPA**2)


def misfit_for_target(R_core, R_mantle, R_ocean, rho_core, rho_mantle,
                       ibar_target, tol=TOL_MOI):
    if not (0 < R_core < R_mantle < R_ocean < R_EUROPA):
        return np.inf
    if R_ocean - R_mantle < 30e3:
        return np.inf
    layers = build_layers(R_core, R_mantle, R_ocean, rho_core, rho_mantle)
    M_model, Ibar_model = mass_and_ibar(layers)
    if abs(M_model - M_EUROPA) / M_EUROPA > 0.01:
        return np.inf
    return abs(Ibar_model - ibar_target) / tol


# ─────────────────────────────────────────────────────────────────────────────
# 5.  MONTE CARLO  (uniform priors, parameterised on ibar_target)
# ─────────────────────────────────────────────────────────────────────────────

PRIOR_UNIFORM = {
    "R_core"    : (400e3, 800e3),
    "mantle_thk": (750e3, 1_000e3),
    "ocean_thk" : (60e3,  150e3),
    "rho_core"  : (4_000, 6_500),
    "rho_mantle": (3_000, 4_000),
}


def run_monte_carlo(ibar_target: float, N: int = 100_000,
                    seed: int = 42) -> np.ndarray:
    """
    Monte Carlo inversion targeting ibar_target.

    Returns accepted solutions as (n_accepted, 5) array:
    [R_core, R_mantle, R_ocean, rho_core, rho_mantle] all in SI.
    """
    rng = np.random.default_rng(seed)
    p   = PRIOR_UNIFORM
    accepted = []

    for _ in range(N):
        R_core     = rng.uniform(*p["R_core"])
        mantle_thk = rng.uniform(*p["mantle_thk"])
        ocean_thk  = rng.uniform(*p["ocean_thk"])
        rho_core   = rng.uniform(*p["rho_core"])
        rho_mantle = rng.uniform(*p["rho_mantle"])

        R_mantle = R_core + mantle_thk
        R_ocean  = R_mantle + ocean_thk
        ice_thk  = R_EUROPA - R_ocean

        if not (10e3 <= ice_thk <= 150e3):
            continue

        if misfit_for_target(R_core, R_mantle, R_ocean,
                              rho_core, rho_mantle, ibar_target) < 1.0:
            accepted.append([R_core, R_mantle, R_ocean, rho_core, rho_mantle])

    solutions = np.array(accepted) if accepted else np.empty((0, 5))
    return solutions


# ─────────────────────────────────────────────────────────────────────────────
# 6.  FEM THERMAL SOLVER  (canonical physics)
# ─────────────────────────────────────────────────────────────────────────────

def freezing_point(ice_thickness_m: float) -> float:
    P_oib = RHO_ICE * G_SURF * ice_thickness_m
    return 273.15 - 0.0075 * (P_oib / 1e5)


def build_grid(R_core, R_mantle, R_ocean,
               nodes_per_layer=(80, 200, 160, 160)):
    n0, n1, n2, n3 = nodes_per_layer
    r = np.concatenate([
        np.linspace(0,        R_core,   n0 + 1, endpoint=True),
        np.linspace(R_core,   R_mantle, n1 + 1, endpoint=True)[1:],
        np.linspace(R_mantle, R_ocean,  n2 + 1, endpoint=True)[1:],
        np.linspace(R_ocean,  R_EUROPA, n3 + 1, endpoint=True)[1:],
    ])
    return r


def nearest_idx(r, r_target):
    return int(np.argmin(np.abs(r - r_target)))


def element_stiffness(r1, r2, k):
    L    = r2 - r1
    geom = (r1**2 + r1 * r2 + r2**2) / 3.0
    return (k / L) * geom * np.array([[ 1., -1.], [-1.,  1.]])


def element_load(r1, r2, H):
    L = r2 - r1
    return (H * L / 12.0) * np.array([
        3*r1**2 + 2*r1*r2 + r2**2,
        r1**2   + 2*r1*r2 + 3*r2**2,
    ])


# ── Rheology: Arrhenius viscosity + stagnant-lid convection ──────────────────

def arrhenius_viscosity(T, eta_ref, T_ref, E_act):
    """Creep viscosity  η(T) = η_ref · exp[(E/R)(1/T − 1/T_ref)]   [Pa s]."""
    T = np.maximum(np.asarray(T, float), 1.0)
    return eta_ref * np.exp((E_act / R_GAS) * (1.0 / T - 1.0 / T_ref))


def stagnant_lid_nusselt(T_base, T_top, d, rho, g, alpha, kappa,
                         eta_ref, T_ref, E_act, Nu_cap=None):
    """
    Nusselt number for a layer with strongly temperature-dependent (Arrhenius)
    viscosity, in the stagnant-lid regime.
    """
    dT    = max(T_base - T_top, 1.0)
    eta_b = float(arrhenius_viscosity(T_base, eta_ref, T_ref, E_act))
    Ra_b  = rho * g * alpha * dT * d**3 / (kappa * eta_b)
    theta = max((E_act / R_GAS) * dT / T_base**2, 1e-6)

    if Ra_b < 20.9 * theta**4:
        return 1.0                                          # sub-critical → conduction

    Nu = max(A_NU * Ra_b ** (1.0 / 3.0) * theta ** (-4.0 / 3.0), 1.0)
    return Nu if Nu_cap is None else min(Nu, Nu_cap)


def compute_temperature(R_core, R_mantle, R_ocean, rho_core, rho_mantle,
                        H_mantle_val=H_MANTLE_RADIOGENIC,
                        tidal_fraction=None, return_diagnostics=False):
    """
    Steady-state 1-D spherical FEM heat-conduction solve with convective
    effective-conductivity iterations.

    Ocean = convecting layer with an UNCAPPED effective Nusselt number Nu_o
    (replaces the earlier prescribed-adiabat + basal-TBL treatment); ocean nodes
    are solved, only the ocean-ice boundary is pinned at the freezing point.
    Mantle & ice = Arrhenius stagnant-lid convection.
    Tidal power partitioned per TIDAL_FRACTION.
    """
    r  = build_grid(R_core, R_mantle, R_ocean)
    Nr = len(r)

    # -- Distribute the total tidal power across the dissipating layers -------
    frac = TIDAL_FRACTION if tidal_fraction is None else tidal_fraction
    f_sum = frac.get("mantle", 0.0) + frac.get("ocean", 0.0) + frac.get("ice", 0.0)
    if f_sum <= 0:
        raise ValueError("tidal_fraction values must sum to a positive number.")
    f_mantle = frac.get("mantle", 0.0) / f_sum
    f_ocean  = frac.get("ocean",  0.0) / f_sum
    f_ice    = frac.get("ice",    0.0) / f_sum

    V_mantle = (4 / 3) * np.pi * (R_mantle**3 - R_core**3)
    V_ocean  = (4 / 3) * np.pi * (R_ocean**3  - R_mantle**3)
    V_ice    = (4 / 3) * np.pi * (R_EUROPA**3 - R_ocean**3)

    H_tidal_mantle = f_mantle * P_TIDAL / V_mantle if V_mantle > 0 else 0.0
    H_tidal_ocean  = f_ocean  * P_TIDAL / V_ocean  if V_ocean  > 0 else 0.0
    H_tidal_ice    = f_ice    * P_TIDAL / V_ice    if V_ice    > 0 else 0.0

    # Node groups for each layer (fixed across iterations)
    m_idx = np.where((r >= R_core)   & (r < R_mantle))[0]
    o_idx = np.where((r >= R_mantle) & (r < R_ocean ))[0]
    i_idx = np.where( r >= R_ocean)[0]

    k_profile = np.where(r < R_core,   K_CORE,
                np.where(r < R_mantle, NU_SEED * K_MANTLE,
                np.where(r < R_ocean,  NU_OCEAN_SEED * K_OCEAN,
                                       K_ICE))).astype(float)

    H_profile = np.where(r < R_core,   0.0,
                np.where(r < R_mantle, H_mantle_val + H_tidal_mantle,
                np.where(r < R_ocean,  H_tidal_ocean,
                                       H_tidal_ice)))

    # Thermal diffusivities / kinematic viscosity
    kappa_mantle = K_MANTLE / (rho_mantle * CP_MANTLE)
    kappa_ice    = K_ICE    / (RHO_ICE    * CP_ICE)
    kappa_ocean  = K_OCEAN  / (RHO_OCEAN  * CP_OCEAN)
    nu_ocean     = ETA_OCEAN / RHO_OCEAN

    ice_thk   = R_EUROPA - R_ocean
    T_freeze  = freezing_point(ice_thk)

    idx_oib   = nearest_idx(r, R_ocean)
    idx_surf  = Nr - 1

    Nu_m = Nu_i = 1.0
    Nu_o = NU_OCEAN_SEED
    T_old = None
    converged = False
    iters_used = MAX_THERMAL_ITER

    for iteration in range(MAX_THERMAL_ITER):
        K_sp = lil_matrix((Nr, Nr))
        F    = np.zeros(Nr)

        for j in range(Nr - 1):
            k_j, k_j1 = k_profile[j], k_profile[j + 1]
            k_e = 2 * k_j * k_j1 / (k_j + k_j1) if (k_j + k_j1) > 0 else 0.0

            H_e = 0.5 * (H_profile[j] + H_profile[j + 1])

            Ke = element_stiffness(r[j], r[j + 1], k_e)
            Fe = element_load(r[j], r[j + 1], H_e)

            K_sp[j,     j]     += Ke[0, 0]
            K_sp[j,     j + 1] += Ke[0, 1]
            K_sp[j + 1, j]     += Ke[1, 0]
            K_sp[j + 1, j + 1] += Ke[1, 1]
            F[j]     += Fe[0]
            F[j + 1] += Fe[1]

        # Centre: symmetry (zero flux)
        K_sp[0, :] = 0
        K_sp[0, 0] = 1
        K_sp[0, 1] = -1
        F[0]       = 0

        # OIB phase-change boundary (top of ocean / base of ice) -- the single
        # pinned anchor for the ocean. Ocean interior + seafloor nodes are now
        # solved (no adiabat pinning); their temperatures emerge from the FEM
        # with the Nu_o-enhanced ocean conductivity.
        K_sp[idx_oib, :]       = 0
        K_sp[idx_oib, idx_oib] = 1
        F[idx_oib]             = T_freeze

        # Surface boundary
        K_sp[idx_surf, :]        = 0
        K_sp[idx_surf, idx_surf] = 1
        F[idx_surf]              = T_SURFACE

        T = spsolve(K_sp.tocsr(), F)

        # -- Mantle: Arrhenius stagnant-lid convection ----------------------
        Nu_m = stagnant_lid_nusselt(
            T[m_idx[0]], T[m_idx[-1]], R_mantle - R_core,
            rho_mantle, G_SURF, ALPHA_MANTLE, kappa_mantle,
            ETA_MANTLE_REF, T_MANTLE_REF, E_MANTLE, Nu_cap=NU_MANTLE_MAX,
        )

        # -- Ice shell: Arrhenius stagnant-lid convection -------------------
        if len(i_idx) > 5:
            Nu_i = stagnant_lid_nusselt(
                T[i_idx[0]], T[i_idx[-1]], R_EUROPA - R_ocean,
                RHO_ICE, G_SURF, ALPHA_ICE, kappa_ice,
                ETA_ICE_REF, T_ICE_REF, E_ICE, Nu_cap=50.0,
            )
        else:
            Nu_i = 1.0

        # -- Ocean: convective effective conductivity, UNCAPPED Nu_o --------
        # Isoviscous Rayleigh number across the ocean; Nu_o = pref.Ra_o^(1/3)
        # with no upper limit. The (Nu up => dT down => Ra down) feedback is
        # self-limiting, and the log-space relaxation below keeps it stable.
        if len(o_idx) > 5:
            dT_o = max(T[o_idx[0]] - T[o_idx[-1]], 1e-6)
            d_o  = R_ocean - R_mantle
            Ra_o = (G_SURF * ALPHA_OCEAN * dT_o * d_o**3) / (kappa_ocean * nu_ocean)
            Nu_o = max(1.0, NU_OCEAN_PREF * Ra_o ** (1.0 / 3.0))   # NO cap
        else:
            Nu_o = 1.0

        # -- Convergence on the temperature field itself --------------------
        if T_old is not None and np.max(np.abs(T - T_old)) < TOL_T:
            converged = True
            iters_used = iteration + 1
            break
        T_old = T.copy()

        # -- Under-relax effective conductivity in LOG space (mantle+ice+ocean) --
        k_profile[m_idx] = np.exp(
            (1 - RELAX_LOGK) * np.log(k_profile[m_idx])
            + RELAX_LOGK * np.log(Nu_m * K_MANTLE))
        k_profile[i_idx] = np.exp(
            (1 - RELAX_LOGK) * np.log(k_profile[i_idx])
            + RELAX_LOGK * np.log(Nu_i * K_ICE))
        k_profile[o_idx] = np.exp(
            (1 - RELAX_LOGK) * np.log(k_profile[o_idx])
            + RELAX_LOGK * np.log(Nu_o * K_OCEAN))

    if not converged:
        warnings.warn(
            f"FEM convection loop did not converge after {MAX_THERMAL_ITER} "
            f"iterations (Nu_m={Nu_m:.3f}, Nu_i={Nu_i:.3f}).",
            RuntimeWarning, stacklevel=2
        )

    if not return_diagnostics:
        return r, T

    diag = {
        "converged":         converged,
        "iterations":        iters_used,
        "Nu_m":              float(Nu_m),
        "Nu_i":              float(Nu_i),
        "Nu_o":              float(Nu_o),           # uncapped ocean Nusselt number
        "T_core_centre":     float(T[0]),
        "T_cmb":             float(T[m_idx[0]]),
        "T_seafloor":        float(T[o_idx[0]]),     # mantle-ocean boundary node (solved)
        "conduction_branch": bool(Nu_m < 2.0),
    }
    return r, T, diag


# ─────────────────────────────────────────────────────────────────────────────
# 7.  PER-SOLUTION PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_one_solution(label: str, grav: dict,
                     N_mc: int = 100_000, seed: int = 42) -> dict | None:
    """
    Full pipeline for one published J2 solution.

    Returns a result dict or None if no solutions accepted / Radau invalid.
    """
    print(f"\n{'='*65}")
    print(f"  {label.replace(chr(10), ' ')}")
    print(f"  {grav['note']}")
    print(f"  J2 = {grav['j2']*1e6:.1f} ×10⁻⁶  (±{grav['j2_sigma']*1e6:.1f})")
    print(f"{'='*65}")

    # Radau–Darwin target
    ibar = ibar_from_j2(grav["j2"])
    if ibar is None:
        print(f"  [SKIP] Radau–Darwin returned None for J2={grav['j2']*1e6:.1f}e-6")
        return None
    print(f"  Target I/(MR²) = {ibar:.5f}")

    # Monte Carlo
    solutions = run_monte_carlo(ibar, N=N_mc, seed=seed)
    if len(solutions) == 0:
        print(f"  [SKIP] No accepted solutions.")
        return None
    n_acc = len(solutions)
    print(f"  Accepted: {n_acc:,} / {N_mc:,}  ({100*n_acc/N_mc:.2f} %)")

    # Median structure
    R_core_med    = np.median(solutions[:, 0])
    R_mantle_med  = np.median(solutions[:, 1])
    R_ocean_med   = np.median(solutions[:, 2])
    rho_core_med  = np.median(solutions[:, 3])
    rho_mantle_med= np.median(solutions[:, 4])

    # Summary statistics
    ice_thk_km    = (R_EUROPA - solutions[:, 2]) / 1e3
    ocean_thk_km  = (solutions[:, 2] - solutions[:, 1]) / 1e3
    core_r_km     = solutions[:, 0] / 1e3
    p16i, p50i, p84i = np.percentile(ice_thk_km,   [16, 50, 84])
    p16o, p50o, p84o = np.percentile(ocean_thk_km,  [16, 50, 84])
    p16c, p50c, p84c = np.percentile(core_r_km,     [16, 50, 84])
    print(f"  Ice shell  : {p50i:.0f} (+{p84i-p50i:.0f} / -{p50i-p16i:.0f}) km")
    print(f"  Ocean      : {p50o:.0f} (+{p84o-p50o:.0f} / -{p50o-p16o:.0f}) km")
    print(f"  Core radius: {p50c:.0f} (+{p84c-p50c:.0f} / -{p50c-p16c:.0f}) km")

    # FEM for median structure (with diagnostics from the canonical engine)
    print(f"  Computing FEM thermal profile for median structure ...")
    r_med, T_med, diag = compute_temperature(
        R_core_med, R_mantle_med, R_ocean_med,
        rho_core_med, rho_mantle_med, return_diagnostics=True)

    # Key temperatures
    T_core_C   = T_med[nearest_idx(r_med, 0)]            - 273.15
    T_mantle_C = T_med[nearest_idx(r_med, R_mantle_med)] - 273.15
    T_ocean_C  = T_med[nearest_idx(r_med, R_ocean_med)]  - 273.15
    print(f"  T(core centre)     = {T_core_C:.1f} °C")
    print(f"  T(mantle–ocean)    = {T_mantle_C:.4f} °C   (seafloor, solved)")
    print(f"  T(ocean–ice base)  = {T_ocean_C:.1f} °C")
    print(f"  Nu_m = {diag['Nu_m']:.1f}   Nu_i = {diag['Nu_i']:.2f}   "
          f"Nu_o = {diag['Nu_o']:.3e}   converged = {diag['converged']}")

    return {
        "label"         : label,
        "grav"          : grav,
        "ibar"          : ibar,
        "solutions"     : solutions,
        "R_core_med"    : R_core_med,
        "R_mantle_med"  : R_mantle_med,
        "R_ocean_med"   : R_ocean_med,
        "rho_core_med"  : rho_core_med,
        "rho_mantle_med": rho_mantle_med,
        "r_med"         : r_med,
        "T_med"         : T_med,
        "diag"          : diag,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 8.  COMPARISON PLOT  (median T-profiles for all solutions)
# ─────────────────────────────────────────────────────────────────────────────

def plot_comparison(results: list[dict], save_path: str | None = None) -> None:
    """
    Plot all median temperature profiles on a single panel.

    Layer labels are placed in data coordinates at the midpoint of each
    layer (using the Anderson+1998 median structure as reference), so they
    always sit inside the correct layer regardless of axis scaling.
    """

    fig, ax = plt.subplots(figsize=(10, 12))

    # ── Draw one line per accepted solution ─────────────────────────────────
    for res in results:
        lbl   = res["label"].replace("\n", " ")
        col   = res["grav"]["color"]
        ls    = res["grav"]["ls"]
        r_med = res["r_med"]
        T_med = res["T_med"]
        T_C   = T_med - 273.15
        depth = (R_EUROPA - r_med) / 1e3      # km, surface = 0

        ax.plot(T_C, depth, color=col, ls=ls, lw=2.4,
                label=f"{lbl}  (I̅={res['ibar']:.4f})")

        # Per-solution boundary lines (thin dotted, matching color)
        for r_b in [res["R_ocean_med"], res["R_mantle_med"], res["R_core_med"]]:
            d_b = (R_EUROPA - r_b) / 1e3
            ax.axhline(d_b, color=col, lw=0.7, alpha=0.45, ls=":")

    # ── Layer labels in data coordinates ────────────────────────────────────
    # Use Anderson+1998 median boundaries as the reference midpoints.
    ref = next((r for r in results
                if "Anderson+1998" in r["label"].replace("\n", " ")), results[0])
    d_ocean  = (R_EUROPA - ref["R_ocean_med"])  / 1e3
    d_mantle = (R_EUROPA - ref["R_mantle_med"]) / 1e3
    d_core   = (R_EUROPA - ref["R_core_med"])   / 1e3
    d_max    = R_EUROPA / 1e3

    # x position: 88% of the way across the current x-range (set after plotting)
    ax.invert_yaxis()   # surface at top — must call before get_ylim
    x_lo, x_hi = ax.get_xlim()
    x_label = x_lo + 0.88 * (x_hi - x_lo)

    layer_specs = [
        (d_ocean  / 2,              "Ice Shell", "#1A5276"),   # mid of 0→d_ocean
        ((d_ocean + d_mantle) / 2,  "Ocean",     "#2471A3"),
        ((d_mantle + d_core)  / 2,  "Mantle",    "#7D6608"),
        ((d_core  + d_max)    / 2,  "Core",      "#922B21"),
    ]
    for y_mid, name, col in layer_specs:
        ax.text(x_label, y_mid, name,
                ha="center", va="center", fontsize=10,
                color=col, fontweight="bold",
                bbox=dict(fc="white", ec=col, alpha=0.80,
                          boxstyle="round,pad=0.3"))

    # ── Axes formatting ───────────────────────────────────────────────────────
    ax.set_xlabel("Temperature (°C)", fontsize=12)
    ax.set_ylabel("Depth below surface (km)", fontsize=12)
    ax.set_title("Europa Median Interior Temperature — by Gravity Solution",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="lower left", fontsize=9, framealpha=0.88)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\n  Saved comparison plot → {save_path}")
        plt.close(fig)
    else:
        plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# 9.  SUMMARY TABLE
# ─────────────────────────────────────────────────────────────────────────────

def print_summary_table(results: list[dict]) -> None:
    """Print a compact summary table of all solutions side by side."""
    hdr = f"{'Solution':<35} {'J2×10⁻⁶':>9} {'I̅':>7} {'Ice(km)':>8} " \
          f"{'Ocean(km)':>10} {'CoreR(km)':>10} {'T_core(°C)':>11} {'Nu_m':>7}"
    print(f"\n{'─'*len(hdr)}")
    print("  COMPARISON TABLE — Median Parameters by J2 Solution")
    print(f"{'─'*len(hdr)}")
    print(f"  {hdr}")
    print(f"  {'─'*(len(hdr)-2)}")
    for res in results:
        lbl  = res["label"].replace("\n", " ")[:33]
        j2v  = res["grav"]["j2"] * 1e6
        ibar = res["ibar"]
        ice  = (R_EUROPA - res["R_ocean_med"]) / 1e3
        oce  = (res["R_ocean_med"] - res["R_mantle_med"]) / 1e3
        cor  = res["R_core_med"] / 1e3
        tc   = res["T_med"][nearest_idx(res["r_med"], 0)] - 273.15
        num  = res["diag"]["Nu_m"]
        print(f"  {lbl:<35} {j2v:>9.1f} {ibar:>7.4f} {ice:>8.1f} "
              f"{oce:>10.1f} {cor:>10.1f} {tc:>11.1f} {num:>7.1f}")
    print(f"{'─'*len(hdr)}\n")


# ─────────────────────────────────────────────────────────────────────────────
# 10.  MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    os.makedirs("europa_gravity_comparison", exist_ok=True)

    print(f"\n{'='*65}")
    print("  Europa Multi-Gravity-Solution Interior Comparison")
    print(f"  Global tidal power: {P_TIDAL:.3e} W")
    print(f"  Tidal partition   : mantle {TIDAL_FRACTION['mantle']:.0%} / "
          f"ice {TIDAL_FRACTION['ice']:.0%}")
    print(f"  Rotational parameter q_rot: {Q_ROT:.6f}")
    print(f"{'='*65}")

    # ── Run pipeline for each published J2 solution ──────────────────────────
    all_results = []
    for label, grav in GRAVITY_SOLUTIONS.items():
        res = run_one_solution(label, grav, N_mc=100_000, seed=42)
        if res is not None:
            all_results.append(res)

    if not all_results:
        print("\n[ERROR] No valid solutions for any dataset. Check prior ranges.")
        raise SystemExit(1)

    # ── Print summary table ───────────────────────────────────────────────────
    print_summary_table(all_results)

    # ── Save individual median profiles to CSV ────────────────────────────────
    for res in all_results:
        tag  = res["label"].replace("\n", "_").replace(" ", "_").replace("+", "")
        tag  = "".join(c for c in tag if c.isalnum() or c in "_-")
        depth_km = (R_EUROPA - res["r_med"]) / 1e3
        T_C      = res["T_med"] - 273.15
        fname = f"europa_gravity_comparison/profile_{tag}.csv"
        np.savetxt(fname,
                   np.column_stack([res["r_med"], depth_km, T_C]),
                   delimiter=",",
                   header="r_m,depth_km,T_degC",
                   comments="")
        print(f"  Profile saved → {fname}")

    # ── Comparison plot ───────────────────────────────────────────────────────
    plot_comparison(
        all_results,
        save_path="europa_gravity_comparison/median_T_profiles_all_solutions.png",
    )

    print("\n[DONE] All results in europa_gravity_comparison/")
