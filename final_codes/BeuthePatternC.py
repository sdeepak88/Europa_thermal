import numpy as np
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# Beuthe-based tidal heating function
# ─────────────────────────────────────────────
def H_tidal_beuthe(H_mean, lat_deg, lon_deg, a=1.0, b=0.3, c=0.2):
    theta = np.radians(90 - lat_deg)
    phi   = np.radians(lon_deg)
    H = H_mean * (a + b * np.cos(2 * theta) + c * np.cos(2 * phi))
    return np.maximum(H, 0)          # FIX #4: np.maximum works for scalars AND arrays

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
R_ice_top    = 1560e3
R_core_center = 0
R_core_top   = 500e3
R_mantle_top = 1400e3

k_ice    = 2.5
k_ocean  = 0.49
k_mantle = 3.5
k_core   = 25

g            = 1.315
alpha_ocean  = 3e-4
rho_ocean    = 1040
cp_ocean     = 4000
mu_ocean     = 1.8e-3
kappa_ocean  = k_ocean / (rho_ocean * cp_ocean)
nu_ocean     = mu_ocean / rho_ocean

alpha_mantle = 2e-5
rho_mantle   = 3300
cp_mantle    = 1000
mu_mantle    = 1e20
kappa_mantle = k_mantle / (rho_mantle * cp_mantle)
nu_mantle    = mu_mantle / rho_mantle

rho_ice  = 920
cp_ice   = 2000

rho_core = 7400
cp_core  = 800

include_tidal_heating_in_mantle = False
H_mantle = 3.4e-7 if include_tidal_heating_in_mantle else 5e-8

# Orbital and tidal parameters
G          = 6.67430e-11
M_jup      = 1.898e27
e          = 0.009
Q          = 100
a_europa   = 6.709e8

R_surf         = R_ice_top
R_oce          = R_mantle_top
P_tidal        = 1e12
volume_ice_ref = (4/3) * np.pi * (R_surf**3 - R_oce**3)
H_tidal_ref    = P_tidal / volume_ice_ref

# Grid
Nr = 900
r  = np.linspace(R_core_center, R_ice_top, Nr)
dr = np.abs(r[1] - r[0])

plot_mask = (r >= 1400e3) & (r <= R_ice_top)
r_plot    = r[plot_mask]

# Harmonic mean helper
def harmonic_mean(a, b):
    return 2 * a * b / (a + b + 1e-12)

# ─────────────────────────────────────────────
# Ice thickness now varies with latitude
# ─────────────────────────────────────────────
def get_ice_thickness(lat_deg):
    """
    Linear interpolation: 10 km at the equator, 30 km at the pole.
    Physically motivated by lower tidal heating at the poles.
    """
    lat_abs = abs(lat_deg)
    return (10e3 + 20e3 * (lat_abs / 90))   # 10 km (equator) → 30 km (pole)

# ─────────────────────────────────────────────
# Main simulation
# ─────────────────────────────────────────────
def simulate_for_location(lat_deg, lon_deg):
    T_core        = 1350
    ice_thickness = get_ice_thickness(lat_deg)
    R_ocean_top   = R_ice_top - ice_thickness

    H_tidal   = H_tidal_beuthe(H_tidal_ref, lat_deg, lon_deg)

    volume_ice = (4/3) * np.pi * (R_ice_top**3 - R_ocean_top**3)
    H_scaled   = H_tidal * volume_ice_ref / volume_ice

    idx_mantle_top = np.argmin(np.abs(r - R_mantle_top))
    idx_ocean_top  = np.argmin(np.abs(r - R_ocean_top))

    T = np.linspace(T_core, -160, Nr)

    Nu_mantle_prev, Nu_ocean_prev = 0.0, 0.0

    for outer in range(200):                     # outer loop: converge Nu
        T_mantle_top    = T[idx_mantle_top]
        T_ocean_top_val = T[idx_ocean_top]

        delta_T_mantle = max(T_core - T_mantle_top, 1)
        delta_T_ocean  = max(T_mantle_top - T_ocean_top_val, 1)

        d_mantle = R_mantle_top - R_core_top
        d_ocean  = R_ocean_top  - R_mantle_top

        Ra_mantle = (g * alpha_mantle * delta_T_mantle * d_mantle**3) / (kappa_mantle * nu_mantle)
        Ra_ocean  = (g * alpha_ocean  * delta_T_ocean  * d_ocean**3)  / (kappa_ocean  * nu_ocean)

        Nu_mantle = 0.3  * Ra_mantle**(1/4)
        Nu_mantle = min(Nu_mantle, 100)          # cap: prevents unphysical k_eff in coarse 1-D model
        Nu_ocean  = min(0.1 * Ra_ocean**(1/3), 100)

        # Check outer convergence
        if abs(Nu_mantle - Nu_mantle_prev) < 1e-4 and abs(Nu_ocean - Nu_ocean_prev) < 1e-4:
            break
        Nu_mantle_prev, Nu_ocean_prev = Nu_mantle, Nu_ocean

        # Build property profiles
        k_profile   = np.zeros(Nr)
        H_profile   = np.zeros(Nr)
        rho_profile = np.zeros(Nr)
        cp_profile  = np.zeros(Nr)

        for j in range(Nr):
            if r[j] > R_ocean_top:          # ice shell (including surface point)
                k_profile[j]   = k_ice
                H_profile[j]   = H_scaled
                rho_profile[j] = rho_ice
                cp_profile[j]  = cp_ice
            elif r[j] > R_mantle_top:       # ocean
                k_profile[j]   = Nu_ocean * k_ocean
                H_profile[j]   = 0
                rho_profile[j] = rho_ocean
                cp_profile[j]  = cp_ocean
            elif r[j] > R_core_top:         # silicate mantle
                k_profile[j]   = Nu_mantle * k_mantle
                H_profile[j]   = H_mantle
                rho_profile[j] = rho_mantle
                cp_profile[j]  = cp_mantle
            else:                            # iron core
                k_profile[j]   = k_core
                H_profile[j]   = 0
                rho_profile[j] = rho_core
                cp_profile[j]  = cp_core

        # Inner loop: solve steady-state heat equation with fixed k
        k_left = harmonic_mean(k_profile[:-2], k_profile[1:-1])
        k_right = harmonic_mean(k_profile[1:-1], k_profile[2:])
        k_eff = k_left + k_right
        H_term = (H_profile[1:-1] * dr**2) / k_eff

        for _ in range(5000):
            T_new = T.copy()
            
            # Vectorized calculation (replaces the 'for j' loop)
            T_new[1:-1] = (k_left * T[:-2] + k_right * T[2:]) / k_eff + H_term

            # Boundary conditions
            T_new[0]             = T_core
            T_new[-1]            = -160
            T_new[idx_ocean_top] = -3

            if np.max(np.abs(T_new - T)) < 1e-3:
                break
            T = T_new

    return T, ice_thickness

# ─────────────────────────────────────────────
# Run for equator and pole
# ─────────────────────────────────────────────
lat_equator = 0
lat_pole    = 90
fixed_lon   = 90

T_equator, ice_eq   = simulate_for_location(lat_equator, fixed_lon)
T_pole,    ice_pole = simulate_for_location(lat_pole,    fixed_lon)

ice_eq_km   = ice_eq   / 1e3
ice_pole_km = ice_pole / 1e3

# ─────────────────────────────────────────────
# Figure 1: Temperature profiles
# ─────────────────────────────────────────────
plt.figure(figsize=(8, 6))                                        
plt.plot(r_plot / 1e3, T_equator[plot_mask],
         label=f"Equator ({ice_eq_km:.0f} km ice shell)",
         color="orange", linewidth=2)
plt.plot(r_plot / 1e3, T_pole[plot_mask],
         label=f"North Pole ({ice_pole_km:.0f} km ice shell)",
         color="royalblue", linewidth=2)
plt.xlabel("Radial Distance from Europa's Centre (km)")
plt.ylabel("Temperature (°C)")
plt.title(
    f"Europa 1-D Steady-State Temperature Profile\n"
    f"Ice Shell + Ocean Layer | Longitude {fixed_lon}°E | "
    f"Beuthe Tidal Heating (Pattern C)"
)
plt.grid(True, alpha=0.4)
plt.legend()
plt.tight_layout()
plt.show()

print(f"H_tidal at Equator (lon={fixed_lon}°): {H_tidal_beuthe(H_tidal_ref, 0,  fixed_lon):.4e} W/m³")
print(f"H_tidal at Pole    (lon={fixed_lon}°): {H_tidal_beuthe(H_tidal_ref, 90, fixed_lon):.4e} W/m³")

# ─────────────────────────────────────────────
# Figure 2: Global tidal heating map (0–360°, no normalisation)
# ─────────────────────────────────────────────
lats = np.linspace(-90, 90,  181)
lons = np.linspace(0,  360, 361)
LON, LAT = np.meshgrid(lons, lats)

H_map = H_tidal_beuthe(1, LAT, LON)   # vectorised 

plt.figure(figsize=(10, 5))            
plt.imshow(H_map, extent=[0, 360, -90, 90],
           origin='lower', cmap='RdBu_r', aspect='auto')
plt.colorbar(label='Relative Tidal Heating (normalised to mean = 1)')
plt.xlabel('Longitude (°E)')
plt.ylabel('Latitude (°N)')
plt.title(
    "Europa Global Tidal Heating Pattern (Beuthe Pattern C Approximation)\n"
    "Longitude 0°–360°E | No Standard-Deviation Normalisation"
)
plt.tight_layout()
plt.show()

# ─────────────────────────────────────────────
# Figure 3: Global map with σ-normalisation (0–360°)
# ─────────────────────────────────────────────
H_mean       = np.mean(H_map)
H_std        = np.std(H_map)
H_normalized = (H_map - H_mean) / H_std

plt.figure(figsize=(8, 5))            # FIX #8
plt.imshow(H_normalized, extent=[0, 360, -90, 90],
           origin='lower', cmap='RdBu_r', aspect='auto',
           vmin=-2.5, vmax=2.5)
plt.colorbar(label='Normalised Tidal Heating (σ units)')
plt.xlabel('Longitude (°E)')
plt.ylabel('Latitude (°N)')
plt.title(
    "Europa Global Tidal Heating Pattern (Beuthe Pattern C) — σ-Normalised\n"
    "Longitude 0°–360°E | Colour Scale ±2.5σ"
)
plt.savefig(f"Pattern0-360.png", dpi=300, bbox_inches="tight")
plt.tight_layout()
plt.show()

# ─────────────────────────────────────────────
# Figure 4: Sub-Jovian-centred map (−90° to +90°)
# ─────────────────────────────────────────────
lats_beuthe = np.linspace(-90, 90, 181)
lons_beuthe = np.linspace(-90, 90, 181)
LON_B, LAT_B = np.meshgrid(lons_beuthe, lats_beuthe)

# Wrap negative lons to [0,360), then apply +90° shift to centre on sub-Jovian point
lon_corrected = (np.where(LON_B >= 0, LON_B, LON_B + 360) + 90) % 360

H_beuthe        = H_tidal_beuthe(1, LAT_B, lon_corrected)
H_beuthe_norm   = (H_beuthe - np.mean(H_beuthe)) / np.std(H_beuthe)

plt.figure(figsize=(8, 6))             # FIX #8
plt.imshow(H_beuthe_norm,
           extent=[-90, 90, -90, 90],
           origin='lower', cmap='RdBu_r', aspect='auto',
           vmin=-2.5, vmax=2.5)
plt.colorbar(label='Normalised Tidal Heating (σ units)')
plt.xlabel('Longitude from Sub-Jovian Point (°)')
plt.ylabel('Latitude (°N)')
plt.title(
    "Europa Tidal Heating Pattern Centred on Sub-Jovian Point (Beuthe Pattern C)\n"
    "σ-Normalised | Longitude −90° to +90° | Colour Scale ±2.5σ"
)
plt.savefig(f"Pattern90-90.png", dpi=300, bbox_inches="tight")
plt.tight_layout()
plt.show()
