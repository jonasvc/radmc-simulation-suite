"""
Reference Configuration: Two-Armed Spiral
Protoplanetary disk with moderate spiral structure
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

### Stellar Parameters ###
tstar          = 10000
rstar          = '2.0*rs'
mstar          = '2.5*ms'
istar_sphere   = 1
pc             = 144
incl           = 45

### Disk Parameters ###
mdisk          = "0.01*ms"
sig0           = 0
rin            = '0.5*au'
rdisk          = '200.0*au'
hrdisk         = 0.117
plsig1         = -1
plh            = 0.35
hrpivot        = '100.0*au'
sigma_type     = 0

### Puffed up inner rim ###
hpr_prim_rout  = 0.034
prim_rout      = 2.8
srim_rout      = 3
srim_plsig     = -0.5

### Dust parameters ###
dustkappa      = dustkappas[1]
gsmax          = 0.1
gsmin          = 0.1
mixabun        = [1.00, 0.00]

### Grid Parameters ###
wbound         = [0.1, 0.9, 2, 7.0, 25., 1e4]
nw             = [30, 30, 20, 30, 30]
xbound         = ['0.02*au', '0.49*au', '0.51*au','190*au','220*au']
nx             = [100, 300, 200, 100]
ybound         = ["0.", "pi/3.", "pi/2.", "2.*pi/3.", "pi"]
ny             = [10, 90, 90, 10]
zbound         = ["0.", "pi", "2*pi"]
nz             = [180, 180]

### Computational Parameters ###
nphot          = '1e+7'
nphot_scat     = '1e+8'
nphot_spec     = '1e+5'
threads        = 32
modified_random_walk = 1
scattering_mode_max = 0
mc_scat_maxtauabs = 30

### Image Parameters ###
npix           = 2200
wav            = 2.2
phi            = 0
sizeau         = 36
nostar         = False
arcsec         = False
log            = True
au             = True
dpi            = 600

### Asymmetric structures ###
h_fourier_aj             = [0.0, 0.0, 0.0, 0.0, 0.0]
h_fourier_bj             = [0.0, 0.0, 0.0, 0.0, 0.0]
h_modulation_strength    = 1.0
h_asymmetry_factor       = 1.0
sig_fourier_aj           = [0.0, 0.0, 0.0, 0.0, 0.0]
sig_fourier_bj           = [0.0, 0.0, 0.0, 0.0, 0.0]
sig_modulation_strength  = 1
sig_asymmetry_factor     = 1

# Spiral structures - ENABLED
h_spiral_amp             = 0.15      # Scale height spiral amplitude
sig_spiral_amp           = 0.3       # Density spiral amplitude
spiral_pitch             = 1.0       # Pitch angle
n_arms                   = 2         # Two-armed spiral
spiral_width_phi         = 0.5       # Azimuthal width
spiral_sharpness         = 1         # Sharpness

# Vortex structures - DISABLED
h_vortex_amp             = [0.0, 0.0]
h_vortex_phi0            = [0, "pi"]
h_vortex_r0              = ['80.0*au', '80.0*au']
h_vortex_width_phi       = [0.1, 0.1]
h_vortex_width_r         = ['5.0*au','5.0*au']
sig_vortex_amp           = [0.0, 0.0]
sig_vortex_phi0          = [0, "pi"]
sig_vortex_r0            = ['80.0*au', '80.0*au']
sig_vortex_width_phi     = [0.1, 0.1]
sig_vortex_width_r       = ['5.0*au','5.0*au']
vortex_sharpness         = 1.0

# Radial damping - DISABLED
use_radial_damping       = False
azimuthal_r_max          = "0.0 * au"
azimuthal_r_width        = "0.0 * au"

# Warped disk - DISABLED
enable_warp              = False
warp_amplitude           = 0.15
warp_phase               = 0.0
warp_mode                = 1

# Inner edge shadow - DISABLED
use_inner_edge_shadow    = False
inner_edge_radius        = "0.5*au"
inner_edge_width         = "1.0*au"
inner_edge_height        = 3.0
inner_edge_azimuthal     = False
inner_edge_phi           = 0.0
inner_edge_phi_width     = 0.785

vertical_steepness       = 1
