"""
Configuration file for RADMC-3D simulations
Vortex Structure Example
"""

import numpy as np

### Constants ###
au = 149.6e+11  # astronomical unit in cm
mas2rad = np.pi / 180 / 3600. / 1000.  # mas to rad conversion

### List of available dust opacities ###
dustkappas = [
    "['Astrosilcarbgra']",
    "['astrosilicateoptool']",
    "['Astrosilcarb001']",
    "['Astrosilcarboptool']",
    "['Draine84-Astrosil-01micron']",
    "['Draine84Astrosil095-carbon005-01micron']"
]

### Stellar Parameters ###

tstar          = 10000            # Effective temp
rstar          = '2.0*rs'          # stellar radius in To
mstar          = '2.5*ms'          # stellar mass in Mo 
istar_sphere   = 1                 # Star as point source or sphere (1)
pc             = 144               # Distance in parsec 
incl           = 45                # Inclination angle in degrees (for SED & Images)

### Disk Parameters ###

mdisk          = "0.01*ms"         # Total mass in the disk. Dust mass is 1% of this mass
sig0           = 0                 # Use this value at rdisk of mdisk=0
rin            = '0.5*au'          # Disk inner radius in au
rdisk          = '200.0*au'        # Disk outer radius in au 
hrdisk         = 0.117             # Pressure-scale-height over radius ratio at 100.0au
plsig1         = -1                # Power exponent of the surface density law
plh            = 0.35              # Disk flaring index (If you take beta from paper, probably take (beta-1) for this value
hrpivot        = '100.0*au'        # Radius to compute the psh
sigma_type     = 0                 # 1 = exponential tapering of the disk at outer radius

### Puffed up inner rim (set all to 0 to disable) ###

hpr_prim_rout  = 0.034             # Pressure scale height at rin
prim_rout      = 2.8               # Outer boundary of puffed um inner rim in terms of r_in
srim_rout      = 3                 # Outer boundary if inner rims smoothing in terms of r_in
srim_plsig     = -0.5              # power exponent of density reduction in the inner rim

### Dust parameters ###

dustkappa      = dustkappas[1]     # Choose the dustkappa-file
gsmax          = 0.1               # Maximum grain size
gsmin          = 0.1               # Minimum grain size
mixabun        = [1.00, 0.00]       # Mixing abundance for multiple dustkappa files
bgdens         = 1e-50
### Grid Parameters ###

wbound         = [0.1, 0.9, 2, 7.0, 25., 1e4]   
nw             = [30, 30, 20, 30,30]   
xbound         = ['0.02*au', '0.49*au', '0.51*au','190*au','220*au']
nx             = [100,300,200,100]          
ybound         = ["0.", "pi/3.", "pi/2.", "2.*pi/3.", "pi"]
ny             = [10,90,90,10]
zbound         = ["0.", "pi", "2*pi"]
nz             = [180,180]

### Computational Parameters ###

nphot          = '1e+7'            # Number of Photons for the Monte-Carlo run
nphot_scat     = '1e+8'            # Number of Photons for Scattering
nphot_spec     = '1e+5'            # Number of Photons for the SED 
threads        = 32                # Number of Processor Threads to use
modified_random_walk = 1           # Enable/Disable Modified random walk 
scattering_mode_max = 0            # Scattering Mode (0=None, 1=Isotropic, 2=Anisotropic)
mc_scat_maxtauabs = 30             # Maximum number of times a photon gets absorbed before being destroyed 

### Image Parameters (VLTI Standart for AT) ###

npix           = 2200              # Number of pixels on the rectangular images
wav            = 2.2               # Wavelength of the image in micron
incl           = 45                 # Inclination angle of the source
phi            = 0                 # Azimuthal rotation angle of the source in the model space
sizeau         = 36                # Diameter of the image in au
nostar         = False             # If True the calculated images will not contain stellar emission
arcsec         = False             # If True image axis will have the unit arcsec (requires dpc)
log            = True
au             = True
dpi            = 600

### Additional Parameters for Asymmetric structures ###
# Fourier modulation
h_fourier_aj             = [0.0, 0.0, 0.0, 0.0, 0.0]  # Fourier cosine coefficients A_j for scale height (j=0,1,2,3,4)
h_fourier_bj             = [0.0, 0.0, 0.0, 0.0, 0.0]  # Fourier sine coefficients B_j for scale height (j=0,1,2,3,4)
h_modulation_strength    = 1.0                        # Amplification factor for scale height Fourier modulation
h_asymmetry_factor       = 1.0                        # Asymmetry factor for exponential reduction (higher = stronger shadows)
sig_fourier_aj           = [0.0, 0.0, 0.0, 0.0, 0.0]  # Fourier cosine coefficients A_j for density (j=0,1,2,3,4)
sig_fourier_bj           = [0.0, 0.0, 0.0, 0.0, 0.0]  # Fourier sine coefficients B_j for density (j=0,1,2,3,4)
sig_modulation_strength  = 1                          # Amplification factor for surface density Fourier modulation
sig_asymmetry_factor     = 1                          # Asymmetry factor for surface density exponential reduction

#Spiral structures
h_spiral_amp             = 0                          # Amplitude of spiral scale height variation
sig_spiral_amp           = 0                          # Amplitude of spiral density variation
spiral_pitch             = 1                          # Pitch angle of spiral structure (used for both h and sig)
n_arms                   = 2                          # Number of spiral arms
spiral_width_phi         = 0.5                        # Azimuthal width of spiral arms
spiral_sharpness         = 1                          # Sharpness of spiral structures (>1 uses tanh profile)

# Vortex Strucutures
h_vortex_amp             = [0.0, 0.0]                 # Amplitude of vortex scale height enhancement
h_vortex_phi0            = [0, "pi"]                  # Azimuthal location of vortex center for scale height (radians)
h_vortex_r0              = ['80.0*au', '80.0*au']     # Radial location of vortex center for scale height
h_vortex_width_phi       = [0.1, 0.1]                 # Azimuthal width of vortex for scale height (radians)
h_vortex_width_r         = ['5.0*au','5.0*au']        # Radial width of vortex for scale height
sig_vortex_amp           = [2.0, 0.0]                 # Amplitude of vortex density enhancement
sig_vortex_phi0          = [0, "pi"]                  # Azimuthal location of vortex center for density (radians)
sig_vortex_r0            = ['80.0*au', '0']           # Radial location of vortex center for density
sig_vortex_width_phi     = [0.3, 0.1]                 # Azimuthal width of vortex for density (radians)
sig_vortex_width_r       = ['15.0*au','0']            # Radial width of vortex for density
vortex_sharpness         = 1.0                        # Sharpness of vortex structures


# Radial damping of the Azimuthal modulation
use_radial_damping       = False                      #
azimuthal_r_max          = "0.0 * au"                 # Radius where azimuthal variations start to be damped (0 = no damping)
azimuthal_r_width        = "0.0 * au"                 # Width of damping transition region

# Warped disk (Experimental feature, not final)
enable_warp              = False                      # 
warp_amplitude           = 0.15                       # Warp amplitude in units of scale height
warp_phase               = 0.0                        # Phase shift of the warp pattern
warp_mode                = 1                          # Azimuthal mode number of the warp

# Creating azimuthally dependend walls at the inner edge to create shadows
use_inner_edge_shadow    = False                      # Use raised inner edge as shadow source
inner_edge_radius        = "0.5*au"                   # Radius of the inner edge enhancement
inner_edge_width         = "1.0*au"                   # Width of the inner edge enhancement
inner_edge_height        = 3.0                        # Height factor relative to normal scale height
inner_edge_azimuthal     = False                      # Azimuthally localized inner edge
inner_edge_phi           = 0.0                        # Azimuthal position of inner edge (radians)
inner_edge_phi_width     = 0.785                      # Azimuthal width of inner edge (radians)

vertical_steepness       = 1                         # Steepness of vertical density profile (disabled if 0)
