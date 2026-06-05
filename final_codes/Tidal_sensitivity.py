# -*- coding: utf-8 -*-
"""
Europa interior model + tidal-heating partition sensitivity (single file).
Self-contained: no external project imports. Just needs numpy/scipy/matplotlib.
"""
# -*- coding: utf-8 -*-
import warnings
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

# ─────────────────────────────────────────────────────────────────────────────
# 1.  PHYSICAL CONSTANTS  (all SI)
# ─────────────────────────────────────────────────────────────────────────────

G          = 6.67430e-11   # N m² kg⁻²
R_EUROPA   = 1_565e3       # m
M_EUROPA   = 4.7998e22     # kg
OMEGA      = 2.047e-5      # rad s⁻¹  (synchronous rotation)
M_JUP      = 1.898e27      # kg
E_ORB      = 0.009         # orbital eccentricity
Q_TIDAL    = 100           # tidal quality factor
A_EUROPA   = 6.709e8       # m  (semi-major axis)
G_SURF     = 1.308         # m s⁻²  (surface gravity)

# Observed gravity — Anderson et al.
J2_OBS  = 435.5e-6
C20_OBS = -J2_OBS / np.sqrt(5)

# Fixed layer densities
RHO_ICE   = 920.0          # kg m⁻³
RHO_OCEAN = 1_030.0        # kg m⁻³  (~50 PSU)

# Thermal properties
K_ICE    = 2.5             # W m⁻¹ K⁻¹
K_OCEAN  = 0.49
K_MANTLE = 3.5
K_CORE   = 25.0

CP_OCEAN  = 4_000          # J kg⁻¹ K⁻¹
CP_MANTLE = 1_000
CP_ICE    = 2_000          # J kg⁻¹ K⁻¹

ALPHA_OCEAN  = 3e-4        # K⁻¹  thermal expansion
ALPHA_MANTLE = 3e-5
ALPHA_ICE    = 1.6e-4      # K⁻¹

ETA_OCEAN  = 1e-3          # Pa s  dynamic viscosity (water)
ETA_MANTLE = 3e19          # Pa s  Reference dynamic viscosity for mantle

H_MANTLE_RADIOGENIC = 5e-8 # W m⁻³  (Schubert et al. 2004)

# Boundary temperatures
T_SURFACE = 113.15         # K  (-160 °C)

# ── Rheology: Arrhenius temperature-dependent viscosity ──────────────────────
R_GAS = 8.314              # J mol⁻¹ K⁻¹  universal gas constant

# Silicate mantle (olivine, diffusion creep)
ETA_MANTLE_REF = ETA_MANTLE   # Pa s  reference viscosity at T_MANTLE_REF (= 3e19)
T_MANTLE_REF   = 1600.0       # K     reference temperature
E_MANTLE       = 300e3        # J mol⁻¹  activation energy (dry olivine, ~240–540e3)

# Ice Ih (melting-point reference)
ETA_ICE_REF    = 1e14         # Pa s  viscosity at the melting point
T_ICE_REF      = 273.0        # K
E_ICE          = 60e3         # J mol⁻¹  activation energy for ice Ih (~50–60e3)

# Stagnant-lid convection scaling (Solomatov 1995; Reese et al. 1999)
A_NU           = 0.53         # prefactor in  Nu = A_NU · Ra_b^(1/3) · θ^(-4/3)

# FEM grid
NR       = 1200
MAX_ITER = 30
RELAX    = 0.3
TOL_CONV = 1e-3

# Thermal convection iteration (stabilised fixed-point solver)
# HYPERPARAMETERS ADJUSTED FOR OCEAN DYNAMICS
MAX_THERMAL_ITER = 300     # Increased to allow the extreme ocean Nu to settle
RELAX_LOGK       = 0.10    # Decreased to prevent wild oscillations in ocean conductivity
NU_SEED          = 15.0    # initial mantle Nu (convecting seed; avoids conduction runaway)
NU_MANTLE_MAX    = 2_000.0 # hard cap on mantle Nu (kills transient spikes)
TOL_T            = 0.5     # convergence tolerance on max |ΔT| between iterations [K]

# ── Ocean convection (effective-conductivity treatment, UNCAPPED Nu_o) ───────

NU_OCEAN_PREF    = 0.10    # prefactor in  Nu_o = NU_OCEAN_PREF · Ra_o^(1/3)
NU_OCEAN_SEED    = 1.0e4   # initial ocean Nu (convecting seed; uncapped solver)

# MOI acceptance tolerance
TOL_MOI  = 0.002

# ─────────────────────────────────────────────────────────────────────────────
# 2.  GRAVITY INVERSION — helper functions
# ─────────────────────────────────────────────────────────────────────────────

def rotational_parameter() -> float:
    """Dimensionless ratio of centrifugal to gravitational acceleration."""
    return OMEGA**2 * R_EUROPA**3 / (G * M_EUROPA)


Q_ROT = rotational_parameter()


def shell_mass(r1: float, r2: float, rho: float) -> float:
    """Mass of a uniform-density spherical shell [kg]."""
    return (4 / 3) * np.pi * rho * (r2**3 - r1**3)


def shell_moi(r1: float, r2: float, rho: float) -> float:
    """Polar moment of inertia of a uniform-density spherical shell [kg m²]."""
    return (8 / 15) * np.pi * rho * (r2**5 - r1**5)


def radau_darwin_ibar(C20: float) -> float | None:
    """
    Infer the normalised MOI  I/(MR²)  from C20 via Radau–Darwin theory.
    """
    J2 = -C20 * np.sqrt(5)
    kf = (6 * J2) / (5 * Q_ROT)
    if not (0 < kf < 4):
        return None
    return (2 / 3) * (1 - (2 / 5) * np.sqrt((4 - kf) / (1 + kf)))


IBAR_TARGET = radau_darwin_ibar(C20_OBS)
print(f"Target I/(MR²) from Radau–Darwin: {IBAR_TARGET:.5f}")


def build_layers(
    R_core: float, R_mantle: float, R_ocean: float,
    rho_core: float, rho_mantle: float
) -> list[tuple[float, float, float]]:
    """Return the four-layer description of Europa's interior."""
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
    """Scalar misfit measure (dimensionless)."""
    # Geometric ordering
    if not (0 < R_core < R_mantle < R_ocean < R_EUROPA):
        return np.inf
    # Minimum ocean thickness (30 km physical prior)
    if R_ocean - R_mantle < 30e3:
        return np.inf
    layers = build_layers(R_core, R_mantle, R_ocean, rho_core, rho_mantle)
    M_model, Ibar_model = mass_and_ibar(layers)
    # Mass constraint ±1 %
    if abs(M_model - M_EUROPA) / M_EUROPA > 0.01:
        return np.inf
    return abs(Ibar_model - IBAR_TARGET) / tol


# ─────────────────────────────────────────────────────────────────────────────
# 3.  MONTE CARLO SAMPLER
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# 5.  GLOBAL TIDAL POWER
# ─────────────────────────────────────────────────────────────────────────────
n = 2.05e-5
# P_TIDAL = (21 / 2) * 0.23 *G *n* (M_JUP)**2 * R_EUROPA**5 * E_ORB**2 / (Q_TIDAL * A_EUROPA**6)
P_TIDAL = 1e12
print(f"\nGlobal tidal power: {P_TIDAL:.3e} W")

TIDAL_FRACTION = {
    "mantle": 0.20,   # silicate mantle (solid, viscoelastic)
    "ocean":  0.00,   # liquid water — negligible viscous tidal heating
    "ice":    0.80,   # ice shell (solid, viscoelastic)
}

def freezing_point(ice_thickness_m: float) -> float:
    P_oib = RHO_ICE * G_SURF * ice_thickness_m     # Pa
    return 273.15 - 0.0075 * (P_oib / 1e5)         # K


# ─────────────────────────────────────────────────────────────────────────────
# 5b.  RHEOLOGY — Arrhenius viscosity + stagnant-lid convection
# ─────────────────────────────────────────────────────────────────────────────

def arrhenius_viscosity(T, eta_ref: float, T_ref: float, E_act: float):
    """
    Creep viscosity  η(T) = η_ref · exp[(E/R)(1/T − 1/T_ref)]   [Pa s].
    """
    T = np.maximum(np.asarray(T, float), 1.0)
    return eta_ref * np.exp((E_act / R_GAS) * (1.0 / T - 1.0 / T_ref))


def stagnant_lid_nusselt(
    T_base: float, T_top: float, d: float,
    rho: float, g: float, alpha: float, kappa: float,
    eta_ref: float, T_ref: float, E_act: float,
    Nu_cap: float | None = None,
) -> float:
    """
    Nusselt number for a layer with strongly temperature-dependent
    (Arrhenius) viscosity, in the stagnant-lid regime.
    """
    dT    = max(T_base - T_top, 1.0)
    eta_b = float(arrhenius_viscosity(T_base, eta_ref, T_ref, E_act))
    Ra_b  = rho * g * alpha * dT * d**3 / (kappa * eta_b)
    theta = max((E_act / R_GAS) * dT / T_base**2, 1e-6)

    if Ra_b < 20.9 * theta**4:
        return 1.0                                          # sub-critical → conduction

    Nu = max(A_NU * Ra_b ** (1.0 / 3.0) * theta ** (-4.0 / 3.0), 1.0)
    return Nu if Nu_cap is None else min(Nu, Nu_cap)


# ─────────────────────────────────────────────────────────────────────────────
# 6.  FEM RADIAL GRID
# ─────────────────────────────────────────────────────────────────────────────

def build_grid(
    R_core: float, R_mantle: float, R_ocean: float,
    nodes_per_layer: tuple = (80, 200, 160, 160)
) -> np.ndarray:
    n0, n1, n2, n3 = nodes_per_layer
    r = np.concatenate([
        np.linspace(0,        R_core,   n0 + 1, endpoint=True),
        np.linspace(R_core,   R_mantle, n1 + 1, endpoint=True)[1:],
        np.linspace(R_mantle, R_ocean,  n2 + 1, endpoint=True)[1:],
        np.linspace(R_ocean,  R_EUROPA, n3 + 1, endpoint=True)[1:],
    ])
    return r

def nearest_idx(r: np.ndarray, r_target: float) -> int:
    return int(np.argmin(np.abs(r - r_target)))


# ─────────────────────────────────────────────────────────────────────────────
# 7.  FEM ELEMENT ROUTINES
# ─────────────────────────────────────────────────────────────────────────────

def element_stiffness(r1: float, r2: float, k: float) -> np.ndarray:
    L    = r2 - r1
    geom = (r1**2 + r1 * r2 + r2**2) / 3.0
    return (k / L) * geom * np.array([[ 1.0, -1.0],
                                       [-1.0,  1.0]])

def element_load(r1: float, r2: float, H: float) -> np.ndarray:
    L = r2 - r1
    return (H * L / 12.0) * np.array([
        3 * r1**2 + 2 * r1 * r2 + r2**2,
        r1**2     + 2 * r1 * r2 + 3 * r2**2,
    ])


# ─────────────────────────────────────────────────────────────────────────────
# 8.  TEMPERATURE SOLVER  —  ocean = convecting layer with UNCAPPED Nu_o
# ─────────────────────────────────────────────────────────────────────────────


def compute_temperature(
    R_core: float, R_mantle: float, R_ocean: float,
    rho_core: float, rho_mantle: float,
    H_mantle_val: float = H_MANTLE_RADIOGENIC,
    tidal_fraction: dict | None = None,
    return_diagnostics: bool = False,
):

    r = build_grid(R_core, R_mantle, R_ocean)
    Nr = len(r)

    # ── Distribute the total tidal power across the dissipating layers ───────
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

        # OIB phase-change boundary (top of ocean / base of ice) — the single
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

        # ── Mantle: Arrhenius stagnant-lid convection ──────────────────────
        Nu_m = stagnant_lid_nusselt(
            T[m_idx[0]], T[m_idx[-1]], R_mantle - R_core,
            rho_mantle, G_SURF, ALPHA_MANTLE, kappa_mantle,
            ETA_MANTLE_REF, T_MANTLE_REF, E_MANTLE, Nu_cap=NU_MANTLE_MAX,
        )

        # ── Ice shell: Arrhenius stagnant-lid convection ───────────────────
        if len(i_idx) > 5:
            Nu_i = stagnant_lid_nusselt(
                T[i_idx[0]], T[i_idx[-1]], R_EUROPA - R_ocean,
                RHO_ICE, G_SURF, ALPHA_ICE, kappa_ice,
                ETA_ICE_REF, T_ICE_REF, E_ICE, Nu_cap=50.0,
            )
        else:
            Nu_i = 1.0

        # ── Ocean: convective effective conductivity, UNCAPPED Nu_o ────────
        # Isoviscous Rayleigh number across the ocean; Nu_o = pref·Ra_o^(1/3)
        # with no upper limit. The (Nu↑ ⇒ ΔT↓ ⇒ Ra↓) feedback is self-limiting,
        # and the log-space relaxation below keeps the climb stable.
        # ── Ice shell: Arrhenius stagnant-lid convection ───────────────────
        if len(i_idx) > 5:
            # Use the max internal temperature, which responds to H_tidal_ice
            T_ice_max = np.max(T[i_idx]) 
            
            Nu_i = stagnant_lid_nusselt(
                T_ice_max, T[i_idx[-1]], R_EUROPA - R_ocean,
                RHO_ICE, G_SURF, ALPHA_ICE, kappa_ice,
                ETA_ICE_REF, T_ICE_REF, E_ICE, Nu_cap=50.0,
            )
        else:
            Nu_i = 1.0

        # ── Convergence on the temperature field itself ────────────────────
        if T_old is not None and np.max(np.abs(T - T_old)) < TOL_T:
            converged = True
            iters_used = iteration + 1
            break
        T_old = T.copy()

        # ── Under-relax effective conductivity in LOG space (mantle+ice+ocean) ──
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
        "T_seafloor":        float(T[o_idx[0]]),     # mantle–ocean boundary node (solved)
        "conduction_branch": bool(Nu_m < 2.0),
    }
    return r, T, diag


# ─────────────────────────────────────────────────────────────────────────────
#  TIDAL-HEATING PARTITION SENSITIVITY  (ice ↔ mantle split)
# ─────────────────────────────────────────────────────────────────────────────
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import csv

# ── 1. Representative interior: median of the uniform-prior gravity ensemble ──
print("\nDeriving median gravity-constrained structure (uniform priors)…")
sols = run_monte_carlo(PRIOR_CONFIGS["uniform"], N=60_000, seed=42)
R_core   = float(np.median(sols[:, 0]))
R_mantle = float(np.median(sols[:, 1]))
R_ocean  = float(np.median(sols[:, 2]))
rho_core = float(np.median(sols[:, 3]))
rho_man  = float(np.median(sols[:, 4]))

print(f"  R_core   = {R_core/1e3:7.1f} km")
print(f"  R_mantle = {R_mantle/1e3:7.1f} km   (mantle thk {(R_mantle-R_core)/1e3:.1f} km)")
print(f"  R_ocean  = {R_ocean/1e3:7.1f} km   (ocean thk  {(R_ocean-R_mantle)/1e3:.1f} km)")
print(f"  ice thk  = {(R_EUROPA-R_ocean)/1e3:7.1f} km")
print(f"  rho_core = {rho_core:7.0f} kg/m3 | rho_mantle = {rho_man:.0f} kg/m3")
print(f"  P_TIDAL  = {P_TIDAL:.3e} W")

# ── 2. Ice : Mantle split grid (ocean = 0) ────────────────────────────────────
# (ice%, mantle%) running from all-ice to all-mantle
splits = [(100, 0), (80, 20), (60, 40), (50, 50),
          (40, 60), (20, 80), (0, 100)]

rows = []
profiles = []   # (ice%, mantle%, r, T)

for ice_pct, man_pct in splits:
    tf = {"mantle": man_pct / 100.0, "ocean": 0.0, "ice": ice_pct / 100.0}
    r, T, d = compute_temperature(
        R_core, R_mantle, R_ocean, rho_core, rho_man,
        tidal_fraction=tf, return_diagnostics=True,
    )
    profiles.append((ice_pct, man_pct, r, T))

    # mid-shell ice temperature (interior responds to ice tidal heating;
    # the hottest point is always the pinned melting-point base, so use mid-depth)
    i_idx = np.where(r >= R_ocean)[0]
    T_ice_mid = float(T[i_idx[len(i_idx) // 2]]) - 273.15

    rows.append({
        "ice_pct":       ice_pct,
        "man_pct":       man_pct,
        "P_ice_TW":      ice_pct / 100.0 * P_TIDAL / 1e12,
        "P_man_TW":      man_pct / 100.0 * P_TIDAL / 1e12,
        "T_core_C":      d["T_core_centre"] - 273.15,
        "T_cmb_C":       d["T_cmb"]         - 273.15,
        "T_seafloor_C":  d["T_seafloor"]    - 273.15,
        "Nu_o":          d["Nu_o"],                   # uncapped ocean Nusselt
        "T_ice_mid_C":   T_ice_mid,
        "Nu_m":          d["Nu_m"],
        "Nu_i":          d["Nu_i"],
        "converged":     d["converged"],
        "regime":        "convecting" if d["Nu_m"] >= 2.0 else "conduction",
    })

# ── 3. Console table ──────────────────────────────────────────────────────────
hdr = (f"{'Ice%':>5}{'Man%':>5}{'P_ice':>8}{'P_man':>8}"
       f"{'T_core':>9}{'T_CMB':>9}{'T_sflr':>10}{'Nu_o':>11}{'T_ice_md':>10}"
       f"{'Nu_m':>9}{'Nu_i':>7}{'regime':>13}")
print("\n" + "═" * len(hdr))
print("  TIDAL-SPLIT SENSITIVITY  (median gravity structure)")
print("  P in TW;  temperatures in °C;  Nu_o = uncapped ocean Nusselt")
print("═" * len(hdr))
print(hdr)
print("─" * len(hdr))
for x in rows:
    print(f"{x['ice_pct']:>5}{x['man_pct']:>5}"
          f"{x['P_ice_TW']:>8.3f}{x['P_man_TW']:>8.3f}"
          f"{x['T_core_C']:>9.1f}{x['T_cmb_C']:>9.1f}{x['T_seafloor_C']:>10.4f}"
          f"{x['Nu_o']:>11.3e}{x['T_ice_mid_C']:>10.1f}"
          f"{x['Nu_m']:>9.1f}{x['Nu_i']:>7.2f}{x['regime']:>13}")
print("═" * len(hdr))

# ── 4. CSV ────────────────────────────────────────────────────────────────────
csv_path = "tidal_split_sensitivity.csv"
with open(csv_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["ice_pct", "mantle_pct", "P_ice_TW", "P_mantle_TW",
                "T_core_centre_C", "T_CMB_C", "T_seafloor_C",
                "Nu_ocean", "T_ice_mid_C",
                "Nu_mantle", "Nu_ice", "regime", "converged"])
    for x in rows:
        w.writerow([x["ice_pct"], x["man_pct"],
                    f"{x['P_ice_TW']:.4f}", f"{x['P_man_TW']:.4f}",
                    f"{x['T_core_C']:.2f}", f"{x['T_cmb_C']:.2f}",
                    f"{x['T_seafloor_C']:.5f}",
                    f"{x['Nu_o']:.4e}",
                    f"{x['T_ice_mid_C']:.2f}",
                    f"{x['Nu_m']:.2f}", f"{x['Nu_i']:.3f}",
                    x["regime"], x["converged"]])
print(f"\nSaved table → {csv_path}")

# ─────────────────────────────────────────────────────────────────────────────
# 5. FIGURES  —  each plot saved as its own PNG
# ─────────────────────────────────────────────────────────────────────────────
cmap = plt.cm.coolwarm
norm = Normalize(vmin=0, vmax=100)     # colour by MANTLE fraction

man_fracs    = np.array([x["man_pct"]      for x in rows])
ice_fracs    = 100 - man_fracs
T_core_arr   = np.array([x["T_core_C"]     for x in rows])
T_cmb_arr    = np.array([x["T_cmb_C"]      for x in rows])
T_sflr_arr   = np.array([x["T_seafloor_C"] for x in rows])
T_icemid_arr = np.array([x["T_ice_mid_C"]  for x in rows])
Nu_m_arr     = np.array([x["Nu_m"]         for x in rows])

saved = []

# (1) Temperature profiles vs depth -------------------------------------------
fig, ax = plt.subplots(figsize=(8, 10))
for ice_pct, man_pct, r, T in profiles:
    depth = (R_EUROPA - r) / 1e3
    ax.plot(T - 273.15, depth, lw=1.8, color=cmap(norm(man_pct)),
            label=f"ice {ice_pct} / man {man_pct}")
depth_core   = (R_EUROPA - R_core)   / 1e3
depth_mantle = (R_EUROPA - R_mantle) / 1e3
depth_ocean  = (R_EUROPA - R_ocean)  / 1e3
for d_, c in [(0.0, "#9B59B6"), (depth_ocean, "#2980B9"),
              (depth_mantle, "#7D6608"), (depth_core, "#922B21")]:
    ax.axhline(d_, color=c, ls="--", lw=1.0, alpha=0.8)
ax.invert_yaxis()
ax.set_xlabel("Temperature (°C)", fontsize=11)
ax.set_ylabel("Depth below surface (km)", fontsize=11)
ax.set_title("Interior temperature vs. ice : mantle tidal split",
             fontsize=12, fontweight="bold")
ax.grid(alpha=0.3)
ax.legend(fontsize=8, loc="upper right", title="split (%)", title_fontsize=9)
sm = ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
# cbar = fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.02)
# cbar.set_label("Mantle tidal fraction (%)", fontsize=9)
fig.savefig("01_temperature_profiles.png", dpi=300, bbox_inches="tight")
plt.close(fig); saved.append("01_temperature_profiles.png")

# (2) Silicate-interior temperatures vs mantle fraction -----------------------
fig, ax = plt.subplots(figsize=(7, 5.5))
ax.plot(man_fracs, T_core_arr, "o-", color="#922B21", label="Core centre")
ax.plot(man_fracs, T_cmb_arr,  "s-", color="#C0392B", label="Core–Mantle boundary")
ax.plot(man_fracs, T_sflr_arr, "^-", color="#7D6608", label="Seafloor (Mantle–Ocean)")
ax.set_xlabel("Mantle tidal fraction (%)", fontsize=11)
ax.set_ylabel("Temperature (°C)", fontsize=11)
ax.set_title("Silicate-interior temperatures", fontsize=12, fontweight="bold")
ax.grid(alpha=0.3); ax.legend(fontsize=9)
fig.savefig("02_silicate_temperatures.png", dpi=200, bbox_inches="tight")
plt.close(fig); saved.append("02_silicate_temperatures.png")

# (3) Ice-shell mid-depth temperature vs ice fraction -------------------------
fig, ax = plt.subplots(figsize=(7, 5.5))
ax.plot(ice_fracs, T_icemid_arr, "o-", color="#1A5276")
ax.set_xlabel("Ice tidal fraction (%)", fontsize=11)
ax.set_ylabel("Mid ice-shell T (°C)", fontsize=11)
ax.set_title("Ice shell mid-depth temperature", fontsize=12, fontweight="bold")
ax.grid(alpha=0.3)
fig.savefig("03_ice_midshell_temperature.png", dpi=200, bbox_inches="tight")
plt.close(fig); saved.append("03_ice_midshell_temperature.png")

# (4) Mantle convective vigour vs mantle fraction -----------------------------
fig, ax = plt.subplots(figsize=(7, 5.5))
ax.plot(man_fracs, Nu_m_arr, "o-", color="#1E6B3C")
ax.set_xlabel("Mantle tidal fraction (%)", fontsize=11)
ax.set_ylabel("Mantle Nusselt  Nu_m", fontsize=11)
ax.set_title("Mantle convective vigour", fontsize=12, fontweight="bold")
ax.grid(alpha=0.3)
fig.savefig("04_mantle_nusselt.png", dpi=200, bbox_inches="tight")
plt.close(fig); saved.append("04_mantle_nusselt.png")

# (5) Summary table (rendered) ------------------------------------------------
fig, ax = plt.subplots(figsize=(8.5, 3.2))
ax.axis("off")
cell_text = [[f"{x['ice_pct']}/{x['man_pct']}",
              f"{x['T_cmb_C']:.0f}",
              f"{x['T_seafloor_C']:.4f}",
              f"{x['Nu_o']:.2e}",
              f"{x['T_ice_mid_C']:.0f}",
              f"{x['Nu_m']:.1f}"] for x in rows]
tbl = ax.table(cellText=cell_text,
               colLabels=["ice/man", "T_CMB (°C)", "T_sflr (°C)",
                          "Nu_o", "T_ice_mid (°C)", "Nu_m"],
               cellLoc="center", loc="center")
tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1.0, 1.4)
for j in range(6):
    tbl[0, j].set_facecolor("#34495E")
    tbl[0, j].set_text_props(color="white", fontweight="bold")
ax.set_title("Tidal-split sensitivity summary", fontsize=12, fontweight="bold", pad=14)
fig.savefig("05_summary_table.png", dpi=200, bbox_inches="tight")
plt.close(fig); saved.append("05_summary_table.png")

# (6) Ocean convective vigour: uncapped Nu_o vs mantle fraction --------------
Nu_o_arr = np.array([x["Nu_o"] for x in rows])
fig, ax = plt.subplots(figsize=(7, 5.5))
ax.plot(man_fracs, Nu_o_arr, "o-", color="#117A65")
ax.set_yscale("log")
ax.set_xlabel("Mantle tidal fraction (%)", fontsize=11)
ax.set_ylabel("Ocean Nusselt  Nu_o  (uncapped)", fontsize=11)
ax.set_title("Ocean convective vigour\n(effective-conductivity, no cap)",
             fontsize=12, fontweight="bold")
ax.grid(alpha=0.3, which="both")
ax.annotate("Nu_o ~ 10⁶–10⁷ ⇒ ocean stays near-isothermal,\n"
            "but Nu_o (and the seafloor T) rise with the\n"
            "mantle heat flux as the split shifts to mantle",
            xy=(0.04, 0.96), xycoords="axes fraction", va="top", ha="left",
            fontsize=8.5, color="#555", style="italic")
fig.savefig("06_ocean_nusselt.png", dpi=200, bbox_inches="tight")
plt.close(fig); saved.append("06_ocean_nusselt.png")

print("\nSaved plots:")
for f in saved:
    print("  •", f)
