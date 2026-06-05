# -*- coding: utf-8 -*-
"""
Europa interior structure & thermal model.

Pipeline:
  1. Invert the observed gravity (J2 / C20) for a normalised MOI target.
  2. Monte-Carlo sample four-layer interiors consistent with mass + MOI.
  3. Solve a 1-D radial FEM heat-conduction problem with convection handled
     via effective Nusselt-number enhancement of conductivity.
  4. Summarise and plot the accepted structures and temperature profiles.
"""

import warnings
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve


# =============================================================================
# 1.  PHYSICAL CONSTANTS  (SI units)
# =============================================================================

# --- Bulk / orbital properties of Europa ---
G        = 6.67430e-11   # gravitational constant [N m^2 kg^-2]
R_EUROPA = 1_565e3       # mean radius [m]
M_EUROPA = 4.7998e22     # total mass [kg]
OMEGA    = 2.047e-5      # synchronous rotation rate [rad s^-1]
G_SURF   = 1.308         # surface gravity [m s^-2]

# --- Observed gravity field (Anderson et al.) ---
J2_OBS  = 435.5e-6
C20_OBS = -J2_OBS / np.sqrt(5)

# --- Fixed layer densities [kg m^-3] ---
RHO_ICE   = 920.0
RHO_OCEAN = 1_030.0      # ~50 PSU sea water

# --- Thermal conductivities [W m^-1 K^-1] ---
K_ICE    = 2.5
K_OCEAN  = 0.49
K_MANTLE = 3.5
K_CORE   = 25.0

# --- Specific heat capacities [J kg^-1 K^-1] ---
CP_OCEAN  = 4_000
CP_MANTLE = 1_000
CP_ICE    = 2_000

# --- Thermal expansion coefficients [K^-1] ---
ALPHA_OCEAN  = 3e-4
ALPHA_MANTLE = 3e-5
ALPHA_ICE    = 1.6e-4

# --- Dynamic viscosities [Pa s] ---
ETA_OCEAN  = 1e-3        # liquid water
ETA_MANTLE = 3e19        # reference mantle viscosity

# --- Internal heating & boundary condition ---
H_MANTLE_RADIOGENIC = 5e-8     # radiogenic heating [W m^-3] (Schubert et al. 2004)
T_SURFACE           = 113.15   # fixed surface temperature [K] (-160 degC)

# --- Rheology: Arrhenius temperature-dependent viscosity ---
R_GAS = 8.314            # universal gas constant [J mol^-1 K^-1]

# Silicate mantle (dry olivine, diffusion creep)
ETA_MANTLE_REF = ETA_MANTLE   # reference viscosity at T_MANTLE_REF [Pa s]
T_MANTLE_REF   = 1600.0       # reference temperature [K]
E_MANTLE       = 300e3        # activation energy [J mol^-1]

# Ice Ih (melting-point reference)
ETA_ICE_REF = 1e14            # viscosity at the melting point [Pa s]
T_ICE_REF   = 273.0           # reference temperature [K]
E_ICE       = 60e3            # activation energy [J mol^-1]

# Stagnant-lid convection scaling (Solomatov 1995; Reese et al. 1999):
#   Nu = A_NU * Ra_b^(1/3) * theta^(-4/3)
A_NU = 0.53

# --- Thermal convection solver controls ---
MAX_THERMAL_ITER = 300        # iteration cap for the fixed-point conductivity loop
RELAX_LOGK       = 0.10       # log-space under-relaxation factor for k_eff
NU_SEED          = 15.0       # initial mantle Nusselt seed (convecting)
NU_MANTLE_MAX    = 2_000.0    # hard cap on mantle Nusselt number
TOL_T            = 0.5        # convergence tolerance on max|dT| between iters [K]

# Ocean convection (uncapped effective-conductivity treatment):
#   Nu_o = NU_OCEAN_PREF * Ra_o^(1/3)   (no upper cap -> near-isothermal ocean)
NU_OCEAN_PREF = 0.10
NU_OCEAN_SEED = 1.0e4         # initial ocean Nusselt seed (convecting)

# --- MOI acceptance tolerance ---
TOL_MOI = 0.002


# =============================================================================
# 2.  GRAVITY INVERSION HELPERS
# =============================================================================

def rotational_parameter() -> float:
    """Dimensionless ratio of centrifugal to gravitational acceleration."""
    return OMEGA**2 * R_EUROPA**3 / (G * M_EUROPA)


Q_ROT = rotational_parameter()


def shell_mass(r1: float, r2: float, rho: float) -> float:
    """Mass of a uniform-density spherical shell [kg]."""
    return (4 / 3) * np.pi * rho * (r2**3 - r1**3)


def shell_moi(r1: float, r2: float, rho: float) -> float:
    """Polar moment of inertia of a uniform-density spherical shell [kg m^2]."""
    return (8 / 15) * np.pi * rho * (r2**5 - r1**5)


def radau_darwin_ibar(C20: float) -> float | None:
    """Infer the normalised MOI I/(MR^2) from C20 via Radau-Darwin theory."""
    J2 = -C20 * np.sqrt(5)
    kf = (6 * J2) / (5 * Q_ROT)
    if not (0 < kf < 4):
        return None
    return (2 / 3) * (1 - (2 / 5) * np.sqrt((4 - kf) / (1 + kf)))


IBAR_TARGET = radau_darwin_ibar(C20_OBS)
print(f"Target I/(MR^2) from Radau-Darwin: {IBAR_TARGET:.5f}")


def build_layers(
    R_core: float, R_mantle: float, R_ocean: float,
    rho_core: float, rho_mantle: float
) -> list[tuple[float, float, float]]:
    """Return the four-layer (core, mantle, ocean, ice) interior description."""
    return [
        (0.0,      R_core,   rho_core),
        (R_core,   R_mantle, rho_mantle),
        (R_mantle, R_ocean,  RHO_OCEAN),
        (R_ocean,  R_EUROPA, RHO_ICE),
    ]


def mass_and_ibar(layers: list) -> tuple[float, float]:
    """Total mass [kg] and normalised MOI for the given layer stack."""
    M = sum(shell_mass(r1, r2, rho) for r1, r2, rho in layers)
    I = sum(shell_moi(r1, r2, rho) for r1, r2, rho in layers)
    return M, I / (M * R_EUROPA**2)


def misfit(
    R_core: float, R_mantle: float, R_ocean: float,
    rho_core: float, rho_mantle: float,
    tol: float = TOL_MOI
) -> float:
    """Dimensionless misfit; returns inf when priors/constraints are violated."""
    # Enforce geometric ordering of the interfaces.
    if not (0 < R_core < R_mantle < R_ocean < R_EUROPA):
        return np.inf
    # Minimum 30 km ocean thickness (physical prior).
    if R_ocean - R_mantle < 30e3:
        return np.inf
    layers = build_layers(R_core, R_mantle, R_ocean, rho_core, rho_mantle)
    M_model, Ibar_model = mass_and_ibar(layers)
    # Mass constraint: within +/-1 %.
    if abs(M_model - M_EUROPA) / M_EUROPA > 0.01:
        return np.inf
    return abs(Ibar_model - IBAR_TARGET) / tol


# =============================================================================
# 3.  MONTE CARLO SAMPLER
# =============================================================================

# Prior specifications for the two run variants.
PRIOR_CONFIGS = {
    "uniform": {
        "label":       "All-Uniform Priors",
        "R_core":      ("uniform", 400e3, 800e3),
        "mantle_thk":  ("uniform", 750e3, 1_000e3),
        "ocean_thk":   ("uniform", 60e3,  150e3),
        "rho_core":    ("uniform", 4_000, 6_500),
        "rho_mantle":  ("uniform", 3_000, 4_000),
    },
    "gaussian": {
        "label":       "Gaussian Core/Mantle + Uniform Ocean/Ice",
        "R_core":      ("normal",  600e3, 100e3),
        "mantle_thk":  ("normal",  900e3, 150e3),
        "ocean_thk":   ("uniform", 60e3,  150e3),
        "rho_core":    ("normal",  5_500, 500),
        "rho_mantle":  ("normal",  3_500, 500),
    },
}


def _sample(rng: np.random.Generator, spec: tuple) -> float:
    """Draw a single sample from a ('uniform'|'normal', a, b) prior spec."""
    kind = spec[0]
    if kind == "uniform":
        return rng.uniform(spec[1], spec[2])
    elif kind == "normal":
        return rng.normal(spec[1], spec[2])
    raise ValueError(f"Unknown prior kind: {kind!r}")


def run_monte_carlo(
    config: dict,
    N: int = 100_000,
    seed: int = 42,
    ice_bounds: tuple = (10e3, 150e3),
) -> np.ndarray:
    """Sample N interiors and return those passing the mass/MOI/ice filters."""
    rng = np.random.default_rng(seed)
    accepted = []
    for _ in range(N):
        R_core     = _sample(rng, config["R_core"])
        mantle_thk = _sample(rng, config["mantle_thk"])
        ocean_thk  = _sample(rng, config["ocean_thk"])
        rho_core   = _sample(rng, config["rho_core"])
        rho_mantle = _sample(rng, config["rho_mantle"])

        R_mantle = R_core + mantle_thk
        R_ocean  = R_mantle + ocean_thk
        ice_thk  = R_EUROPA - R_ocean

        if not (ice_bounds[0] <= ice_thk <= ice_bounds[1]):
            continue

        if misfit(R_core, R_mantle, R_ocean, rho_core, rho_mantle) < 1.0:
            accepted.append([R_core, R_mantle, R_ocean, rho_core, rho_mantle])

    solutions = np.array(accepted)
    print(f"  Accepted: {len(solutions):,} / {N:,}  ({100*len(solutions)/N:.2f} %)")
    return solutions


# =============================================================================
# 4.  STATISTICAL SUMMARY & H_TIDAL REPORT
# =============================================================================

def percentile_summary(solutions: np.ndarray, config: dict) -> dict:
    """Print and return 16/50/84-percentile ranges for derived parameters and H_tidal."""
    R_core_arr     = solutions[:, 0]
    R_mantle_arr   = solutions[:, 1]
    R_ocean_arr    = solutions[:, 2]
    
    R_core_km      = R_core_arr / 1e3
    mantle_thk_km  = (R_mantle_arr - R_core_arr) / 1e3
    ocean_thk_km   = (R_ocean_arr - R_mantle_arr) / 1e3
    ice_thk_km     = (R_EUROPA - R_ocean_arr) / 1e3
    rho_core_arr   = solutions[:, 3]
    rho_mantle_arr = solutions[:, 4]

    # --- Compute Volumetric Tidal Heating (H_tidal) ---
    f_sum    = sum(TIDAL_FRACTION.values())
    f_mantle = TIDAL_FRACTION.get("mantle", 0.0) / f_sum
    f_ice    = TIDAL_FRACTION.get("ice", 0.0) / f_sum
    
    V_mantle_arr = (4 / 3) * np.pi * (R_mantle_arr**3 - R_core_arr**3)
    V_ice_arr    = (4 / 3) * np.pi * (R_EUROPA**3 - R_ocean_arr**3)
    
    H_tidal_mantle_arr = f_mantle * P_TIDAL / V_mantle_arr
    H_tidal_ice_arr    = f_ice * P_TIDAL / V_ice_arr

    params = {
        "Core radius (km)":       R_core_km,
        "Mantle thickness (km)":  mantle_thk_km,
        "Ocean thickness (km)":   ocean_thk_km,
        "Ice thickness (km)":     ice_thk_km,
        "Core density (kg/m^3)":  rho_core_arr,
        "Mantle density (kg/m^3)": rho_mantle_arr,
    }

    print(f"\n{'-'*60}")
    print(f"  GRAVITY-CONSTRAINED RANGES  |  {config['label']}")
    print(f"  Uncertainty: 16-84 percentile  (~1 sigma equivalent)")
    print(f"{'-'*60}")

    summary = {}
    for name, vals in params.items():
        p16, p50, p84 = np.percentile(vals, [16, 50, 84])
        summary[name] = (p16, p50, p84)
        print(f"  {name}")
        print(f"    Median : {p50:.1f}  (+{p84-p50:.1f} / -{p50-p16:.1f})")
        print(f"    16-84% : {p16:.1f} - {p84:.1f}")

    # --- Print H_tidal Median and Range ---
    print(f"\n  VOLUMETRIC TIDAL HEATING (H_tidal) [W/m^3]")
    print(f"  Mantle:")
    print(f"    Median : {np.median(H_tidal_mantle_arr):.3e}")
    print(f"    Range  : {np.min(H_tidal_mantle_arr):.3e} to {np.max(H_tidal_mantle_arr):.3e}")
    print(f"  Ice Shell:")
    print(f"    Median : {np.median(H_tidal_ice_arr):.3e}")
    print(f"    Range  : {np.min(H_tidal_ice_arr):.3e} to {np.max(H_tidal_ice_arr):.3e}")

    return summary


# =============================================================================
# 5.  TIDAL POWER & PHASE-CHANGE BOUNDARY
# =============================================================================

# Total global tidal dissipation [W] and its distribution across layers.
P_TIDAL = 1e12
print(f"\nGlobal tidal power: {P_TIDAL:.3e} W")

TIDAL_FRACTION = {
    "mantle": 0.20,   # solid silicate mantle (viscoelastic)
    "ocean":  0.00,   # liquid water (negligible viscous heating)
    "ice":    0.80,   # solid ice shell (viscoelastic)
}


def freezing_point(ice_thickness_m: float) -> float:
    """Pressure-depressed freezing temperature at the ocean-ice boundary [K]."""
    P_oib = RHO_ICE * G_SURF * ice_thickness_m            # overburden pressure [Pa]
    return 273.15 - 0.0075 * (P_oib / 1e5)


# =============================================================================
# 6.  RHEOLOGY — ARRHENIUS VISCOSITY + STAGNANT-LID CONVECTION
# =============================================================================

def arrhenius_viscosity(T, eta_ref: float, T_ref: float, E_act: float):
    """Creep viscosity eta(T) = eta_ref * exp[(E/R)(1/T - 1/T_ref)]  [Pa s]."""
    T = np.maximum(np.asarray(T, float), 1.0)
    return eta_ref * np.exp((E_act / R_GAS) * (1.0 / T - 1.0 / T_ref))


def stagnant_lid_nusselt(
    T_base: float, T_top: float, d: float,
    rho: float, g: float, alpha: float, kappa: float,
    eta_ref: float, T_ref: float, E_act: float,
    Nu_cap: float | None = None,
) -> float:
    """Nusselt number for a strongly T-dependent (Arrhenius) stagnant-lid layer."""
    dT    = max(T_base - T_top, 1.0)
    eta_b = float(arrhenius_viscosity(T_base, eta_ref, T_ref, E_act))
    Ra_b  = rho * g * alpha * dT * d**3 / (kappa * eta_b)
    theta = max((E_act / R_GAS) * dT / T_base**2, 1e-6)

    # Sub-critical Rayleigh number -> pure conduction.
    if Ra_b < 20.9 * theta**4:
        return 1.0

    Nu = max(A_NU * Ra_b ** (1.0 / 3.0) * theta ** (-4.0 / 3.0), 1.0)
    return Nu if Nu_cap is None else min(Nu, Nu_cap)


# =============================================================================
# 7.  FEM RADIAL GRID
# =============================================================================

def build_grid(
    R_core: float, R_mantle: float, R_ocean: float,
    nodes_per_layer: tuple = (80, 200, 160, 160)
) -> np.ndarray:
    """Build a non-uniform radial node distribution, one block per layer."""
    n0, n1, n2, n3 = nodes_per_layer
    r = np.concatenate([
        np.linspace(0,        R_core,   n0 + 1, endpoint=True),
        np.linspace(R_core,   R_mantle, n1 + 1, endpoint=True)[1:],
        np.linspace(R_mantle, R_ocean,  n2 + 1, endpoint=True)[1:],
        np.linspace(R_ocean,  R_EUROPA, n3 + 1, endpoint=True)[1:],
    ])
    return r


def nearest_idx(r: np.ndarray, r_target: float) -> int:
    """Index of the grid node closest to r_target."""
    return int(np.argmin(np.abs(r - r_target)))


# =============================================================================
# 8.  FEM ELEMENT ROUTINES  (spherical 1-D conduction)
# =============================================================================

def element_stiffness(r1: float, r2: float, k: float) -> np.ndarray:
    """2-node element conductance matrix for spherical-shell conduction."""
    L    = r2 - r1
    geom = (r1**2 + r1 * r2 + r2**2) / 3.0
    return (k / L) * geom * np.array([[ 1.0, -1.0],
                                       [-1.0,  1.0]])


def element_load(r1: float, r2: float, H: float) -> np.ndarray:
    """2-node element load vector for a uniform volumetric source H."""
    L = r2 - r1
    return (H * L / 12.0) * np.array([
        3 * r1**2 + 2 * r1 * r2 + r2**2,
        r1**2     + 2 * r1 * r2 + 3 * r2**2,
    ])


# =============================================================================
# 9.  TEMPERATURE SOLVER
# =============================================================================
# Convection is represented by enhancing each layer's conductivity with an
# effective Nusselt number, iterated to convergence in log-space:
#   - Mantle & ice : Arrhenius stagnant-lid Nu (capped).
#   - Ocean        : Nu_o = NU_OCEAN_PREF * Ra_o^(1/3), uncapped, driving the
#                    ocean to near-isothermal at the freezing point.
# Only the ocean-ice boundary is pinned (phase-change anchor); the seafloor
# temperature emerges from the solve. Interfaces use harmonic conductivity
# averaging.

def compute_temperature(
    R_core: float, R_mantle: float, R_ocean: float,
    rho_core: float, rho_mantle: float,
    H_mantle_val: float = H_MANTLE_RADIOGENIC,
    tidal_fraction: dict | None = None,
    return_diagnostics: bool = False,
):
    r = build_grid(R_core, R_mantle, R_ocean)
    Nr = len(r)

    # --- Distribute total tidal power across the dissipating layers ---
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

    # --- Per-layer node groups (fixed across iterations) ---
    m_idx = np.where((r >= R_core)   & (r < R_mantle))[0]
    o_idx = np.where((r >= R_mantle) & (r < R_ocean ))[0]
    i_idx = np.where( r >= R_ocean)[0]

    # --- Initial conductivity and heating profiles (convecting seeds) ---
    k_profile = np.where(r < R_core,   K_CORE,
                np.where(r < R_mantle, NU_SEED * K_MANTLE,
                np.where(r < R_ocean,  NU_OCEAN_SEED * K_OCEAN,
                                       K_ICE))).astype(float)

    H_profile = np.where(r < R_core,   0.0,
                np.where(r < R_mantle, H_mantle_val + H_tidal_mantle,
                np.where(r < R_ocean,  H_tidal_ocean,
                                       H_tidal_ice)))

    # --- Thermal diffusivities and kinematic viscosity ---
    kappa_mantle = K_MANTLE / (rho_mantle * CP_MANTLE)
    kappa_ice    = K_ICE    / (RHO_ICE    * CP_ICE)
    kappa_ocean  = K_OCEAN  / (RHO_OCEAN  * CP_OCEAN)
    nu_ocean     = ETA_OCEAN / RHO_OCEAN

    ice_thk  = R_EUROPA - R_ocean
    T_freeze = freezing_point(ice_thk)

    idx_oib  = nearest_idx(r, R_ocean)
    idx_surf = Nr - 1

    Nu_m = Nu_i = 1.0
    Nu_o = NU_OCEAN_SEED
    T_old = None
    converged = False
    iters_used = MAX_THERMAL_ITER

    for iteration in range(MAX_THERMAL_ITER):
        K_sp = lil_matrix((Nr, Nr))
        F    = np.zeros(Nr)

        # Assemble global stiffness matrix and load vector.
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

        # Centre: symmetry / zero-flux condition.
        K_sp[0, :] = 0
        K_sp[0, 0] = 1
        K_sp[0, 1] = -1
        F[0]       = 0

        # Ocean-ice boundary: pinned at the freezing point (phase-change anchor).
        K_sp[idx_oib, :]       = 0
        K_sp[idx_oib, idx_oib] = 1
        F[idx_oib]             = T_freeze

        # Surface: fixed temperature.
        K_sp[idx_surf, :]        = 0
        K_sp[idx_surf, idx_surf] = 1
        F[idx_surf]              = T_SURFACE

        T = spsolve(K_sp.tocsr(), F)

        # --- Mantle: Arrhenius stagnant-lid convection ---
        Nu_m = stagnant_lid_nusselt(
            T[m_idx[0]], T[m_idx[-1]], R_mantle - R_core,
            rho_mantle, G_SURF, ALPHA_MANTLE, kappa_mantle,
            ETA_MANTLE_REF, T_MANTLE_REF, E_MANTLE, Nu_cap=NU_MANTLE_MAX,
        )

        # --- Ice shell: Arrhenius stagnant-lid convection ---
        if len(i_idx) > 5:
            Nu_i = stagnant_lid_nusselt(
                T[i_idx[0]], T[i_idx[-1]], R_EUROPA - R_ocean,
                RHO_ICE, G_SURF, ALPHA_ICE, kappa_ice,
                ETA_ICE_REF, T_ICE_REF, E_ICE, Nu_cap=50.0,
            )
        else:
            Nu_i = 1.0

        # --- Ocean: uncapped convective effective conductivity ---
        if len(o_idx) > 5:
            dT_o = max(T[o_idx[0]] - T[o_idx[-1]], 1e-6)
            d_o  = R_ocean - R_mantle
            Ra_o = (G_SURF * ALPHA_OCEAN * dT_o * d_o**3) / (kappa_ocean * nu_ocean)
            Nu_o = max(1.0, NU_OCEAN_PREF * Ra_o ** (1.0 / 3.0))
        else:
            Nu_o = 1.0

        # --- Convergence test on the temperature field ---
        if T_old is not None and np.max(np.abs(T - T_old)) < TOL_T:
            converged = True
            iters_used = iteration + 1
            break
        T_old = T.copy()

        # --- Log-space under-relaxation of effective conductivity ---
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
        "Nu_o":              float(Nu_o),
        "T_core_centre":     float(T[0]),
        "T_cmb":             float(T[m_idx[0]]),
        "T_seafloor":        float(T[o_idx[0]]),
        "conduction_branch": bool(Nu_m < 2.0),
    }
    return r, T, diag


# =============================================================================
# 10.  THERMAL CONVERGENCE DIAGNOSTICS
# =============================================================================

def diagnostics_summary(diags: list[dict], tag: str) -> None:
    """Print convergence statistics and flag suspect (cold/non-converged) runs."""
    n = len(diags)
    if n == 0:
        print("  No diagnostics to report.")
        return

    n_failed     = sum(1 for d in diags if not d["converged"])
    n_conduction = sum(1 for d in diags if d["conduction_branch"])
    Nu_m  = np.array([d["Nu_m"]  for d in diags])
    iters = np.array([d["iterations"] for d in diags])
    T_cmb = np.array([d["T_cmb"] for d in diags]) - 273.15   # degC

    print(f"\n{'-'*60}")
    print(f"  THERMAL CONVERGENCE DIAGNOSTICS  |  tag={tag!r}  (N={n})")
    print(f"{'-'*60}")
    print(f"  Converged          : {n - n_failed:,} / {n:,}  "
          f"({100*(n-n_failed)/n:.1f} %)")
    print(f"  Conduction branch  : {n_conduction:,}  (Nu_m < 2 - suspect/cold)")
    print(f"  Iterations         : min {iters.min()}, "
          f"median {int(np.median(iters))}, max {iters.max()} "
          f"(cap {MAX_THERMAL_ITER})")
    print(f"  Nu_m  16/50/84%    : "
          f"{np.percentile(Nu_m,16):.1f} / {np.percentile(Nu_m,50):.1f} / "
          f"{np.percentile(Nu_m,84):.1f}   "
          f"[min {Nu_m.min():.2f}, max {Nu_m.max():.1f}]")
    print(f"  T_cmb 16/50/84%    : "
          f"{np.percentile(T_cmb,16):.0f} / {np.percentile(T_cmb,50):.0f} / "
          f"{np.percentile(T_cmb,84):.0f} degC   "
          f"[min {T_cmb.min():.0f}, max {T_cmb.max():.0f}]")

    # Explicitly list the worst offenders for inspection / culling.
    bad = [i for i, d in enumerate(diags)
           if (not d["converged"]) or d["conduction_branch"]]
    if bad:
        print(f"\n  ! {len(bad)} suspect solution(s) (index -> Nu_m, T_cmb, conv):")
        for i in bad[:15]:
            d = diags[i]
            print(f"      [{i:4d}]  Nu_m={d['Nu_m']:7.2f}  "
                  f"T_cmb={d['T_cmb']-273.15:6.0f}degC  "
                  f"conv={d['converged']}")
        if len(bad) > 15:
            print(f"      ... and {len(bad)-15} more")
    else:
        print(f"\n  All solutions converged on the convecting branch.")


# =============================================================================
# 11.  PLOTTING — PARAMETER HISTOGRAMS
# =============================================================================

def plot_histograms(solutions: np.ndarray, config: dict, tag: str) -> None:
    """Histogram each derived structural parameter with 16/50/84% markers."""
    R_core_km     = solutions[:, 0] / 1e3
    mantle_thk_km = (solutions[:, 1] - solutions[:, 0]) / 1e3
    ocean_thk_km  = (solutions[:, 2] - solutions[:, 1]) / 1e3
    ice_thk_km    = (R_EUROPA - solutions[:, 2]) / 1e3
    rho_core_arr  = solutions[:, 3]
    rho_mantle_arr= solutions[:, 4]

    data   = [R_core_km, mantle_thk_km, ocean_thk_km,
              ice_thk_km, rho_core_arr, rho_mantle_arr]
    labels = ["Core radius (km)", "Mantle thickness (km)",
              "Ocean thickness (km)", "Ice thickness (km)",
              "Core density (kg m^-3)", "Mantle density (kg m^-3)"]
    colors = ["#922B21", "#7D6608", "#1A5276", "#AED6F1", "#6C3483", "#1E6B3C"]

    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    fig.suptitle(f"Gravity-Constrained Parameter Distributions\n{config['label']}",
                 fontsize=13, fontweight="bold")
    axes = axes.flatten()

    for ax, d, lab, col in zip(axes, data, labels, colors):
        p16, p50, p84 = np.percentile(d, [16, 50, 84])
        ax.hist(d, bins=40, color=col, alpha=0.80, edgecolor="white", linewidth=0.4)
        ax.axvline(p50, color="black", lw=1.8, label=f"Median {p50:.1f}")
        ax.axvline(p16, color="black", lw=1.0, ls="--", alpha=0.7)
        ax.axvline(p84, color="black", lw=1.0, ls="--", alpha=0.7)
        ax.set_xlabel(lab, fontsize=10)
        ax.set_ylabel("Count", fontsize=9)
        ax.legend(fontsize=8, framealpha=0.6)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"histograms_{tag}.png", dpi=300, bbox_inches="tight")
    plt.show()


# =============================================================================
# 12.  PLOTTING — TEMPERATURE PROFILES
# =============================================================================

def plot_temperature_profiles(
    T_profiles_list: list[tuple[np.ndarray, np.ndarray]],
    R_core_med: float, R_mantle_med: float, R_ocean_med: float,
    T_median_structure: tuple[np.ndarray, np.ndarray],
    config: dict,
    tag: str,
) -> None:
    """Plot the ensemble of temperature profiles with layer boundaries."""
    r_ref, T_ref = T_median_structure
    depth_ref = (R_EUROPA - r_ref) / 1e3                  # km, surface = 0

    # Interpolate every profile onto the reference grid.
    T_matrix = []
    for r_i, T_i in T_profiles_list:
        T_matrix.append(np.interp(r_ref, r_i, T_i))
    T_matrix = np.array(T_matrix)

    T_low  = np.percentile(T_matrix, 16, axis=0) - 273.15
    T_high = np.percentile(T_matrix, 84, axis=0) - 273.15
    T_med  = np.median(T_matrix, axis=0) - 273.15
    T_ref_C = T_ref - 273.15

    depth_core    = (R_EUROPA - R_core_med)   / 1e3
    depth_mantle  = (R_EUROPA - R_mantle_med) / 1e3
    depth_ocean   = (R_EUROPA - R_ocean_med)  / 1e3
    depth_surface = 0.0

    fig = plt.figure(figsize=(10, 13))
    gs  = gridspec.GridSpec(1, 2, width_ratios=[3, 1], wspace=0.05)
    ax  = fig.add_subplot(gs[0])
    ax_ice = fig.add_subplot(gs[1])

    # Main panel: spread band, sampled profiles, medians.
    ax.fill_betweenx(depth_ref, T_low, T_high,
                     color="steelblue", alpha=0.25,
                     label="16-84% Spread")

    inside = [i for i, T_i in enumerate(T_matrix)
              if np.all((T_i - 273.15 >= T_low) & (T_i - 273.15 <= T_high))]
    if inside:
        rng_plot = np.random.default_rng(0)
        chosen = rng_plot.choice(inside, min(20, len(inside)), replace=False)
        for idx in chosen:
            ax.plot(T_matrix[idx] - 273.15, depth_ref,
                    color="steelblue", alpha=0.18, lw=0.8)

    ax.plot(T_med, depth_ref, "k-", lw=2.0, label="Median of All Profiles")
    ax.plot(T_ref_C, depth_ref, color="#C0392B", lw=2.5,
            label="Median Gravity Structure")

    # Layer-boundary lines.
    boundary_style = dict(lw=1.2, alpha=0.85)
    ax.axhline(depth_surface, color="#9B59B6", ls="--", label="Surface", **boundary_style)
    ax.axhline(depth_ocean,   color="#2980B9", ls="--", label=f"Ocean-Ice  ({depth_ocean:.0f} km)", **boundary_style)
    ax.axhline(depth_mantle,  color="#7D6608", ls="--", label=f"Mantle-Ocean  ({depth_mantle:.0f} km)", **boundary_style)
    ax.axhline(depth_core,    color="#922B21", ls="--", label=f"Core-Mantle  ({depth_core:.0f} km)", **boundary_style)

    label_x = 0.97

    def mid_depth(d1, d2):
        return (d1 + d2) / 2

    layer_labels = [
        (mid_depth(depth_surface, depth_ocean),  "Ice Shell", "#2C3E50"),
        (mid_depth(depth_ocean,   depth_mantle), "Ocean",     "#1A5276"),
        (mid_depth(depth_mantle,  depth_core),   "Mantle",    "#7D6608"),
        (mid_depth(depth_core,    max(depth_ref)),"Core",     "#922B21"),
    ]
    for yd, lbl, col in layer_labels:
        ax.text(label_x, yd, lbl,
                transform=ax.get_yaxis_transform(),
                ha="right", va="center", fontsize=9,
                color=col, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7, ec=col))

    ax.set_xlabel("Temperature (degC)", fontsize=12)
    ax.set_ylabel("Depth below surface (km)", fontsize=12)
    ax.invert_yaxis()
    ax.set_title(
        f"Interior Temperature Profiles of Europa\n{config['label']}",
        fontsize=12, fontweight="bold"
    )
    ax.legend(loc="lower left", fontsize=8, framealpha=0.85)
    ax.grid(alpha=0.3)

    # Inset panel: ice-shell zoom.
    ice_mask = depth_ref <= depth_ocean + 5
    ax_ice.fill_betweenx(depth_ref[ice_mask],
                         T_low[ice_mask], T_high[ice_mask],
                         color="steelblue", alpha=0.3)
    ax_ice.plot(T_med[ice_mask], depth_ref[ice_mask], "k-", lw=1.8)
    ax_ice.plot(T_ref_C[ice_mask], depth_ref[ice_mask], color="#C0392B", lw=2.0)
    ax_ice.axhline(depth_ocean, color="#2980B9", ls="--", lw=1.0)
    ax_ice.invert_yaxis()
    ax_ice.set_xlabel("T (degC)", fontsize=9)
    ax_ice.set_title("Ice Shell\nZoom", fontsize=9)
    ax_ice.grid(alpha=0.3)
    ax_ice.tick_params(labelsize=8)
    ax_ice.yaxis.set_label_position("right")
    ax_ice.yaxis.tick_right()

    plt.savefig(f"temperature_profiles_{tag}.png", dpi=300, bbox_inches="tight")
    plt.show()

    # Report key boundary temperatures for the median structure.
    idx_core_med   = nearest_idx(r_ref, 0)
    idx_mantle_med = nearest_idx(r_ref, R_mantle_med)
    idx_ocean_med  = nearest_idx(r_ref, R_ocean_med)

    print(f"\n  Key temperatures - median gravity structure:")
    print(f"    Core centre          : {T_ref[idx_core_med]   - 273.15:.1f} degC")
    print(f"    Mantle-Ocean boundary: {T_ref[idx_mantle_med] - 273.15:.1f} degC")
    print(f"    Ocean-Ice boundary   : {T_ref[idx_ocean_med]  - 273.15:.1f} degC")
    print(f"    Surface              : {T_SURFACE - 273.15:.1f} degC")


# =============================================================================
# 13.  PERSISTENCE
# =============================================================================

def save_run(tag: str, solutions: np.ndarray,
             T_profiles_list: list, r_list: list) -> None:
    """Persist accepted solutions and their temperature/grid arrays to disk."""
    np.save(f"solutions_{tag}.npy", solutions)
    np.save(f"T_profiles_{tag}.npy", np.array([T for _, T in T_profiles_list]))
    np.save(f"r_grids_{tag}.npy",    np.array([r for r, _ in T_profiles_list]))
    print(f"  Saved solutions and T-profiles to disk (tag={tag!r})")


def load_run(tag: str):
    """Reload a previously saved run (solutions + paired (r, T) profiles)."""
    sols = np.load(f"solutions_{tag}.npy")
    Ts   = np.load(f"T_profiles_{tag}.npy")
    rs   = np.load(f"r_grids_{tag}.npy")
    return sols, list(zip(rs, Ts))


# =============================================================================
# 14.  MAIN DRIVER
# =============================================================================

def run_variant(tag: str, config: dict, N: int = 100_000, seed: int = 42,
                force_recompute: bool = False) -> None:
    """Run one prior variant end-to-end: sample, summarise, solve, and plot."""
    print(f"\n{'='*65}")
    print(f"  VARIANT: {config['label']}")
    print(f"{'='*65}")

    # --- Stage 1: Monte-Carlo sampling (load cached if available) ---
    sol_file = f"solutions_{tag}.npy"
    if not force_recompute and os.path.exists(sol_file):
        solutions = np.load(sol_file)
        print(f"  Loaded {len(solutions):,} solutions from {sol_file}")
    else:
        solutions = run_monte_carlo(config, N=N, seed=seed)
        np.save(sol_file, solutions)

    if len(solutions) == 0:
        print("  No accepted solutions - check prior ranges or constraints.")
        return

    # --- Stage 2: statistics and histograms ---
    _ = percentile_summary(solutions, config)

    R_core_med    = np.median(solutions[:, 0])
    R_mantle_med  = np.median(solutions[:, 1])
    R_ocean_med   = np.median(solutions[:, 2])
    rho_core_med  = np.median(solutions[:, 3])
    rho_mantle_med= np.median(solutions[:, 4])

    plot_histograms(solutions, config, tag)

    # --- Stage 3: FEM thermal profiles (load cached if available) ---
    tp_file = f"T_profiles_{tag}.npy"
    rg_file = f"r_grids_{tag}.npy"
    if not force_recompute and os.path.exists(tp_file):
        Ts = np.load(tp_file)
        rs = np.load(rg_file)
        T_profiles_list = list(zip(rs, Ts))
        print(f"  Loaded {len(T_profiles_list):,} T-profiles from disk")
    else:
        print(f"\n  Computing FEM thermal profiles for {len(solutions):,} solutions...")
        T_profiles_list = []
        diags = []
        for i, sol in enumerate(solutions):
            r_i, T_i, diag_i = compute_temperature(*sol, return_diagnostics=True)
            T_profiles_list.append((r_i, T_i))
            diags.append(diag_i)
            if (i + 1) % max(1, len(solutions) // 10) == 0:
                print(f"    {i+1}/{len(solutions)}", end="\r", flush=True)
        print()
        diagnostics_summary(diags, tag)
        save_run(tag, solutions, T_profiles_list,
                 [r for r, _ in T_profiles_list])

    # --- Stage 4: temperature profile plot for the median structure ---
    r_med, T_med_struct = compute_temperature(
        R_core_med, R_mantle_med, R_ocean_med,
        rho_core_med, rho_mantle_med
    )

    plot_temperature_profiles(
        T_profiles_list,
        R_core_med, R_mantle_med, R_ocean_med,
        (r_med, T_med_struct),
        config, tag
    )


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    # Run both prior variants; set force_recompute=True to regenerate from scratch.
    run_variant("uniform",  PRIOR_CONFIGS["uniform"],  N=100_000, seed=42,  force_recompute=True)
    run_variant("gaussian", PRIOR_CONFIGS["gaussian"], N=100_000, seed=123, force_recompute=True)
