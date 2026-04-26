"""
Configuration file based on Montesinos et al. (2021) ApJ 910:31
Implementing a Jupiter-mass planet with CPD at 10 AU, 270° (West)

Paper parameters:
- Star: Solar-type, T_eff = 6000 K, M = 1 M_sun
- Disk: 4 to 25 AU
- Planet mass: 1 M_Jupiter
- Planet luminosity: L_p = 10^-3 L_sun
- Radial position: r_p = 10 AU
- Azimuthal position: φ_p = 270° (West)
- CPD temperature: ~1060 K (with feedback)
- Scale height enhancement: H/r from 0.05 to 0.2 (factor ~4 in paper)
- Initial aspect ratio: H/r = 0.05
"""

import numpy as np

### Constants ###
au = 149.6e+11
mas2rad = np.pi / 180 / 3600. / 1000.

### Dust opacities ###
dustkappas = [
    "['Astrosilcarbgra']",
    "['astrosilicateoptool']",
    "['Astrosilcarb001']",
    "['Astrosilcarboptool']",
    "['Draine84-Astrosil-01micron']",
    "['Draine84Astrosil095-carbon005-01micron']"
]

### Stellar Parameters (Solar-type from paper) ###
tstar          = 6000        # Solar-type star (paper page 5)
rstar          = '1.0*rs'    # Solar radius
mstar          = '1.0*ms'    # Solar mass
istar_sphere   = 1
pc             = 140          # Distance: 140 pc (paper page 5)
incl           = 13           # Inclination: 13° (paper page 5)

### Disk Parameters (4-25 AU from paper) ###
mdisk          = 0#"5e-3*ms"   # ~5×10^-3 M_sun (paper page 4)
sig0           = 23.14
rin            = '4.0*au'    # Inner radius: 4 AU (paper page 4)
rdisk          = '25.0*au'   # Outer radius: 25 AU (paper page 4)
hrdisk         = 0.05        # Initial H/r = 0.05 (paper page 4)
plsig1         = 0           # Surface density exponent (Σ ∝ r^0, paper page 4)
plh            = 0.0         # No flaring in paper setup
hrpivot        = '10.0*au'   # Pivot at planet location
sigma_type     = 0

### Puffed up inner rim - DISABLED (not in paper) ###
hpr_prim_rout  = 0.0
prim_rout      = 0.0
srim_rout      = 0.0
srim_plsig     = 0.0

### Dust parameters ###
dustkappa      = dustkappas[1]
gsmax          = 0.1
gsmin          = 0.1
mixabun        = [1.00, 0.00]
bgdens         = 1e-50

### Grid Parameters ###
wbound         = [0.1, 0.9, 2, 7.0, 25., 1e4]
nw             = [30, 30, 20, 30, 30]
xbound         = ['0.02*au', '3.9*au', '4.1*au', '24*au', '26*au']
nx             = [50, 400, 200, 50]
ybound         = ["0.", "pi/3.", "pi/2.", "2.*pi/3.", "pi"]
ny             = [90, 90, 90, 90]
zbound         = ["0.", "pi", "2*pi"]
nz             = [300, 300]

### Computational Parameters ###
nphot          = '3e+8'
nphot_scat     = '3e+8'
nphot_spec     = '1e+5'
threads        = 30
modified_random_walk = 1
scattering_mode_max = 1
mc_scat_maxtauabs = 15

### Image Parameters ###
npix           = 2200
wav            = 1.0         # λ = 1 μm for scattered light (paper page 5)
phi            = 0
sizeau         = 30          # Adjusted for 25 AU disk
nostar         = False
arcsec         = False
log            = True
au             = True
dpi            = 600

### Asymmetric structures - BASELINE (all disabled) ###
h_fourier_aj             = [0.0, 0.0, 0.0, 0.0, 0.0]
h_fourier_bj             = [0.0, 0.0, 0.0, 0.0, 0.0]
h_modulation_strength    = 1.0
h_asymmetry_factor       = 1.0
sig_fourier_aj           = [0.0, 0.0, 0.0, 0.0, 0.0]
sig_fourier_bj           = [0.0, 0.0, 0.0, 0.0, 0.0]
sig_modulation_strength  = 1.0
sig_asymmetry_factor     = 1.0

# Spiral structures - DISABLED
h_spiral_amp             = 0.0
sig_spiral_amp           = 0.0
spiral_pitch             = 1.0
n_arms                   = 2
spiral_width_phi         = 0.5
spiral_sharpness         = 1.0

# Vortex structures - CPD at 10 AU, 270° (West)
# Based on Montesinos et al. 2021: radiative pressure creates optically thick bump
h_vortex_amp             = [5]            # Enhancement factor: H/r: 0.05 → 0.2 (factor ~4-6)
h_vortex_phi0            = ["0"]            # 
h_vortex_r0              = ['10.0*au']      # Planet at 10 AU (paper page 4)
h_vortex_width_phi       = [0.0175]          # ~10° azimuthal width (0.175 rad ≈ 10°, from paper Fig. 7)
h_vortex_width_r         = ['0.3*au']       # Radial width ~Hill radius (paper Fig. 4 shows ~2 AU height, ~0.3 AU radial)
sig_vortex_amp           = [5]            # No density enhancement in baseline
sig_vortex_phi0          = ["0"]
sig_vortex_r0            = ['10.0*au']
sig_vortex_width_phi     = [0.03]
sig_vortex_width_r       = ['0.3*au']
vortex_sharpness         = 2.0              # Sharp transitions (paper Fig. 4 shows steep edges)

# Radial damping - DISABLED
use_radial_damping       = False
azimuthal_r_max          = "0.0*au"
azimuthal_r_width        = "1.0*au"

# Warped disk - DISABLED
enable_warp              = False
warp_amplitude           = 0.0
warp_phase               = 0.0
warp_mode                = 1

# Inner edge shadow - DISABLED
use_inner_edge_shadow    = False
inner_edge_radius        = "4.0*au"
inner_edge_width         = "0.5*au"
inner_edge_height        = 1.0
inner_edge_azimuthal     = False
inner_edge_phi           = 0.0
inner_edge_phi_width     = 0.785

vertical_steepness       = 1.0
