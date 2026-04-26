#!/usr/bin/env python3
"""
Interferometry Analysis Module (using radmc3dPy)
================================================

Direct implementation using radmc3dPy's getVisibility() and getClosurePhase()
Based on your INTERFEROMETRY.ipynb notebook.
"""

import numpy as np
import os
import contextlib

# Try to import radmc3dPy
try:
    from radmc3dPy import image as radmc_image
    RADMC3D_AVAILABLE = True
except ImportError:
    RADMC3D_AVAILABLE = False
    print("Warning: radmc3dPy not available!")


# =============================================================================
# BASELINE DEFINITIONS (from your notebook)
# =============================================================================

def get_vlti_baselines():
    """Get VLTI baseline configurations"""
    
    # Baseline lengths [m]
    baselines = {
        'A0_B5': 25.69,
        'J2_J6': 98.55,
        'A0_J2': 104.77,
        'A0_J6': 158.63,
        'B5_J2': 106.61,
        'B5_J6': 174.44,
        'J2_K0': 32.27,
        'G1_J2': 58.09,
        'A0_G1': 55.55,
        'G1_K0': 84.58,
        'A0_K0': 121.83,
        'A0_D0': 30.47,
        'D0_G1': 36.33,
        'B5_D0': 43.44,
        'B5_G1': 49.85,
        'K0_J3': 30.25,
        'G2_D0': 38.82,
        'K0_G2': 56.09,
        'G2_J3': 63.92,
        'K0_D0': 91.36,
        'D0_J3': 102.69,
        'B2_C1': 10.56,
        'A0_B2': 12.68,
        'A0_C1': 13.88,
        'D0_C1': 21.16,
        'B2_D0': 31.72,
    }
    
    # Position angles [deg]
    position_angles = {
        'A0_B5': -69.38,
        'J2_J6': 126.04,
        'A0_J2': -156.47,
        'A0_J6': 166.19,
        'B5_J2': -170.39,
        'B5_J6': 159.21,
        'J2_K0': 138.21,
        'G1_J2': -178.72,
        'A0_G1': -133.15,
        'G1_K0': 166.18,
        'A0_K0': -170.39,
        'A0_D0': -170.39,
        'D0_G1': -102.64,
        'B5_D0': 154.13,
        'B5_G1': -160.69,
        'K0_J3': 113.03,
        'G2_D0': -9.35,
        'K0_G2': 22.60,
        'G2_J3': 174.36,
        'K0_D0': 9.61,
        'D0_J3': 172.96,
        'B2_C1': 166.17,
        'A0_B2': -86.54,
        'A0_C1': -133.14,
        'D0_C1': -13.80,
        'B2_D0': 166.19,
    }
    
    # Sort by name
    names = sorted(baselines.keys())
    bl = np.array([baselines[n] for n in names])
    pa = np.array([position_angles[n] for n in names])
    
    return {'baselines': bl, 'position_angles': pa, 'names': names}


def get_closure_triangles():
    """Get closure phase triangles (from your notebook)"""
    
    # Define triangles by baseline names
    triangles = [
        ['A0_B5', 'B5_J2', 'A0_J2'],
        ['A0_J2', 'J2_J6', 'A0_J6'],
        ['B5_J2', 'J2_J6', 'B5_J6'],
        ['A0_B5', 'B5_J6', 'A0_J6'],
        ['G1_J2', 'J2_K0', 'G1_K0'],
        ['A0_G1', 'G1_J2', 'A0_J2'],
        ['A0_J2', 'J2_K0', 'A0_K0'],
        ['A0_G1', 'G1_K0', 'A0_K0'],
        ['A0_B5', 'B5_D0', 'A0_D0'],
        ['B5_D0', 'D0_G1', 'B5_G1'],
        ['A0_D0', 'D0_G1', 'A0_G1'],
        ['K0_G2', 'G2_J3', 'K0_J3'],
        ['K0_G2', 'G2_D0', 'K0_D0'],
        ['G2_D0', 'D0_J3', 'G2_J3'],
        ['K0_D0', 'D0_J3', 'K0_J3'],
        ['A0_B2', 'B2_C1', 'A0_C1'],
        ['A0_D0', 'D0_C1', 'A0_C1'],
        ['B2_D0', 'D0_C1', 'B2_C1'],
    ]
    
    vlti_config = get_vlti_baselines()
    bl_dict = dict(zip(vlti_config['names'], vlti_config['baselines']))
    pa_dict = dict(zip(vlti_config['names'], vlti_config['position_angles']))
    
    # Convert to baseline triplets
    bl_triplets = []
    pa_triplets = []
    max_baselines = []
    
    for tri in triangles:
        bl_tri = [bl_dict[name] for name in tri]
        pa_tri = [pa_dict[name] for name in tri]
        
        # Fix PA for third baseline (from notebook: pa-180)
        pa_tri[2] = pa_tri[2] - 180
        
        bl_triplets.append(bl_tri)
        pa_triplets.append(pa_tri)
        max_baselines.append(max(bl_tri))
    
    triangle_names = ['_'.join(tri) for tri in triangles]
    
    return {
        'triangles': bl_triplets,
        'position_angles': pa_triplets,
        'names': triangle_names,
        'max_baselines': np.array(max_baselines)
    }


# =============================================================================
# VISIBILITY CALCULATION (1:1 from notebook)
# =============================================================================

def calculate_visibilities(fits_file, dpc=144, wavelength_idx=None):
    """
    Calculate visibilities using radmc3dPy.getVisibility()
    
    Parameters:
    -----------
    fits_file : str
        Path to FITS file (or image.out)
    dpc : float
        Distance in parsecs (default: 144 for AB Aur)
    wavelength_idx : int, optional
        Which wavelength to use. If None, uses all.
    
    Returns:
    --------
    dict with visibility data
    """
    if not RADMC3D_AVAILABLE:
        raise ImportError("radmc3dPy is required for visibility calculation")
    
    # Get VLTI baselines
    vlti = get_vlti_baselines()
    bl = vlti['baselines']
    pa = vlti['position_angles']
    pa_rad = np.deg2rad(pa)
    
    # Read image
    print(f"Reading {fits_file}...")
    with open(os.devnull, 'w') as devnull:
        with contextlib.redirect_stdout(devnull):
            im = radmc_image.readImage(fits_file)
    
    # Calculate visibilities (1:1 from notebook)
    print("Calculating visibilities...")
    with open(os.devnull, 'w') as devnull:
        with contextlib.redirect_stdout(devnull):
            vis_rad = im.getVisibility(bl=bl, pa=pa_rad, dpc=dpc)
    
    # Mapping wavelengths and compute total flux (1:1 from notebook)
    iwav_indices = [np.argmin(np.abs(im.wav - wav)) for wav in vis_rad["wav"]]
    total_flux_jy = [np.sum(im.imageJyppix[:, :, iwav]) for iwav in iwav_indices]
    total_flux_jy = np.array(total_flux_jy)
    
    print(f"Total flux (Jy): {total_flux_jy/dpc**2}")
    
    # Normalized visibility amplitudes (1:1 from notebook)
    vis_amp = vis_rad["amp"][:,:] * 1e23 / (total_flux_jy/dpc**2)
    
    # Computing squared visibilities (1:1 from notebook)
    vis2 = vis_amp**2
    
    # Select wavelength if specified
    if wavelength_idx is not None:
        u = vis_rad["u"][:, wavelength_idx:wavelength_idx+1]
        v = vis_rad["v"][:, wavelength_idx:wavelength_idx+1]
        wav = np.array([vis_rad["wav"][wavelength_idx]])
        vis2_sel = vis2[:, wavelength_idx:wavelength_idx+1]
        vis_amp_sel = vis_amp[:, wavelength_idx:wavelength_idx+1]
    else:
        u = vis_rad["u"]
        v = vis_rad["v"]
        wav = vis_rad["wav"]
        vis2_sel = vis2
        vis_amp_sel = vis_amp
    
    return {
        'u': u,
        'v': v,
        'vis2': vis2_sel,
        'vis_amp': vis_amp_sel,
        'wav': wav,
        'bl': vis_rad["bl"],
        'pa': pa,
        'bl_names': vlti['names']
    }


def calculate_visibilities(run_dir, dpc=144, wavelength_idx=None):
    """
    Calculate visibilities using radmc3dPy.getVisibility()
    Reads image.out from run directory (1:1 from notebook)
    
    Parameters:
    -----------
    run_dir : str
        Path to run directory containing image.out
    dpc : float
        Distance in parsecs (default: 144 for AB Aur)
    wavelength_idx : int, optional
        Which wavelength to use. If None, uses all.
    
    Returns:
    --------
    dict with visibility data
    """
    if not RADMC3D_AVAILABLE:
        raise ImportError("radmc3dPy is required for visibility calculation")
    
    # Get VLTI baselines
    vlti = get_vlti_baselines()
    bl = vlti['baselines']
    pa = vlti['position_angles']
    pa_rad = np.deg2rad(pa)
    
    # Change to run directory and read image.out (1:1 from notebook)
    old_dir = os.getcwd()
    os.chdir(run_dir)
    
    try:
        # Read the Image (1:1 from notebook)
        datei = "image.out"
        im = radmc_image.readImage(datei)
        
        # Calculate visibilities (1:1 from notebook)
        with open(os.devnull, 'w') as devnull:
            with contextlib.redirect_stdout(devnull):
                vis_rad = im.getVisibility(bl=bl, pa=pa_rad, dpc=dpc)
        
        # Mapping wavelengths and compute total flux (1:1 from notebook)
        iwav_indices = [np.argmin(np.abs(im.wav - wav)) for wav in vis_rad["wav"]]
        total_flux_jy = [np.sum(im.imageJyppix[:, :, iwav]) for iwav in iwav_indices]
        total_flux_jy = np.array(total_flux_jy)
        
        print(f"Total flux (Jy): {total_flux_jy/dpc**2}")
        
        # Normalized visibility amplitudes (1:1 from notebook)
        vis_amp = vis_rad["amp"][:,:] * 1e23 / (total_flux_jy/dpc**2)
        
        # Computing squared visibilities (1:1 from notebook)
        vis2 = vis_amp**2
        
        # Select wavelength if specified
        if wavelength_idx is not None:
            u = vis_rad["u"][:, wavelength_idx:wavelength_idx+1]
            v = vis_rad["v"][:, wavelength_idx:wavelength_idx+1]
            wav = np.array([vis_rad["wav"][wavelength_idx]])
            vis2_sel = vis2[:, wavelength_idx:wavelength_idx+1]
            vis_amp_sel = vis_amp[:, wavelength_idx:wavelength_idx+1]
        else:
            u = vis_rad["u"]
            v = vis_rad["v"]
            wav = vis_rad["wav"]
            vis2_sel = vis2
            vis_amp_sel = vis_amp
        
        result = {
            'u': u,
            'v': v,
            'vis2': vis2_sel,
            'vis_amp': vis_amp_sel,
            'wav': wav,
            'bl': vis_rad["bl"],
            'pa': pa,
            'bl_names': vlti['names']
        }
        
    finally:
        os.chdir(old_dir)
    
    return result


def calculate_visibilities_custom(run_dir, baselines, position_angles, baseline_names=None, 
                                  dpc=144, wavelength_idx=None):
    """
    Calculate visibilities with custom baselines
    
    Parameters:
    -----------
    run_dir : str
        Path to run directory containing image.out
    baselines : array-like
        Custom baseline lengths in meters
    position_angles : array-like
        Custom position angles in degrees
    baseline_names : list, optional
        Names for the baselines
    dpc : float
        Distance in parsecs
    wavelength_idx : int, optional
        Which wavelength to use
    
    Returns:
    --------
    dict with visibility data
    """
    if not RADMC3D_AVAILABLE:
        raise ImportError("radmc3dPy is required for visibility calculation")
    
    pa_rad = np.deg2rad(position_angles)
    
    if baseline_names is None:
        baseline_names = [f'BL{i+1}' for i in range(len(baselines))]
    
    # Change to run directory and read image.out
    old_dir = os.getcwd()
    os.chdir(run_dir)
    
    try:
        # Read the Image
        datei = "image.out"
        im = radmc_image.readImage(datei)
        
        # Calculate visibilities
        with open(os.devnull, 'w') as devnull:
            with contextlib.redirect_stdout(devnull):
                vis_rad = im.getVisibility(bl=baselines, pa=pa_rad, dpc=dpc)
        
        # Mapping wavelengths and compute total flux
        iwav_indices = [np.argmin(np.abs(im.wav - wav)) for wav in vis_rad["wav"]]
        total_flux_jy = [np.sum(im.imageJyppix[:, :, iwav]) for iwav in iwav_indices]
        total_flux_jy = np.array(total_flux_jy)
        
        print(f"Total flux (Jy): {total_flux_jy/dpc**2}")
        
        # Normalized visibility amplitudes
        vis_amp = vis_rad["amp"][:,:] * 1e23 / (total_flux_jy/dpc**2)
        
        # Computing squared visibilities
        vis2 = vis_amp**2
        
        # Select wavelength if specified
        if wavelength_idx is not None:
            u = vis_rad["u"][:, wavelength_idx:wavelength_idx+1]
            v = vis_rad["v"][:, wavelength_idx:wavelength_idx+1]
            wav = np.array([vis_rad["wav"][wavelength_idx]])
            vis2_sel = vis2[:, wavelength_idx:wavelength_idx+1]
            vis_amp_sel = vis_amp[:, wavelength_idx:wavelength_idx+1]
        else:
            u = vis_rad["u"]
            v = vis_rad["v"]
            wav = vis_rad["wav"]
            vis2_sel = vis2
            vis_amp_sel = vis_amp
        
        result = {
            'u': u,
            'v': v,
            'vis2': vis2_sel,
            'vis_amp': vis_amp_sel,
            'wav': wav,
            'bl': baselines,
            'pa': position_angles,
            'bl_names': baseline_names
        }
        
    finally:
        os.chdir(old_dir)
    
    return result


def calculate_closure_phases(run_dir, dpc=144, wavelength_idx=None):
    """
    Calculate visibilities using radmc3dPy.getVisibility()
    Reads image.out from run directory (1:1 from notebook)
    
    Parameters:
    -----------
    run_dir : str
        Path to run directory containing image.out
    dpc : float
        Distance in parsecs (default: 144 for AB Aur)
    wavelength_idx : int, optional
        Which wavelength to use. If None, uses all.
    
    Returns:
    --------
    dict with visibility data
    """
    if not RADMC3D_AVAILABLE:
        raise ImportError("radmc3dPy is required for visibility calculation")
    
    # Get VLTI baselines
    vlti = get_vlti_baselines()
    bl = vlti['baselines']
    pa = vlti['position_angles']
    pa_rad = np.deg2rad(pa)
    
    # Change to run directory and read image.out (1:1 from notebook)
    old_dir = os.getcwd()
    os.chdir(run_dir)
    
    try:
        # Read the Image (1:1 from notebook)
        datei = "image.out"
        im = radmc_image.readImage(datei)
        
        # Calculate visibilities (1:1 from notebook)
        with open(os.devnull, 'w') as devnull:
            with contextlib.redirect_stdout(devnull):
                vis_rad = im.getVisibility(bl=bl, pa=pa_rad, dpc=dpc)
        
        # Mapping wavelengths and compute total flux (1:1 from notebook)
        iwav_indices = [np.argmin(np.abs(im.wav - wav)) for wav in vis_rad["wav"]]
        total_flux_jy = [np.sum(im.imageJyppix[:, :, iwav]) for iwav in iwav_indices]
        total_flux_jy = np.array(total_flux_jy)
        
        print(f"Total flux (Jy): {total_flux_jy/dpc**2}")
        
        # Normalized visibility amplitudes (1:1 from notebook)
        vis_amp = vis_rad["amp"][:,:] * 1e23 / (total_flux_jy/dpc**2)
        
        # Computing squared visibilities (1:1 from notebook)
        vis2 = vis_amp**2
        
        # Select wavelength if specified
        if wavelength_idx is not None:
            u = vis_rad["u"][:, wavelength_idx:wavelength_idx+1]
            v = vis_rad["v"][:, wavelength_idx:wavelength_idx+1]
            wav = np.array([vis_rad["wav"][wavelength_idx]])
            vis2_sel = vis2[:, wavelength_idx:wavelength_idx+1]
            vis_amp_sel = vis_amp[:, wavelength_idx:wavelength_idx+1]
        else:
            u = vis_rad["u"]
            v = vis_rad["v"]
            wav = vis_rad["wav"]
            vis2_sel = vis2
            vis_amp_sel = vis_amp
        
        result = {
            'u': u,
            'v': v,
            'vis2': vis2_sel,
            'vis_amp': vis_amp_sel,
            'wav': wav,
            'bl': vis_rad["bl"],
            'pa': pa,
            'bl_names': vlti['names']
        }
        
    finally:
        os.chdir(old_dir)
    
    return result


def calculate_closure_phases(run_dir, dpc=144, wavelength_idx=None):
    """
    Calculate closure phases using radmc3dPy.getClosurePhase()
    Reads image.out from run directory (1:1 from notebook)
    
    Parameters:
    -----------
    run_dir : str
        Path to run directory containing image.out
    dpc : float
        Distance in parsecs
    wavelength_idx : int, optional
        Which wavelength to use
    
    Returns:
    --------
    dict with closure phase data
    """
    if not RADMC3D_AVAILABLE:
        raise ImportError("radmc3dPy is required for closure phase calculation")
    
    # Get triangles
    tri_config = get_closure_triangles()
    bl_cp = tri_config['triangles']
    pa_cp = tri_config['position_angles']
    
    # Convert PA (from notebook: pa_cp = (90-pa_cp) %360)
    pa_cp = np.array(pa_cp)
    pa_cp = (90 - pa_cp) % 360
    
    # Change to run directory and read image.out (1:1 from notebook)
    old_dir = os.getcwd()
    os.chdir(run_dir)
    
    try:
        # Read the Image (1:1 from notebook)
        datei = "image.out"
        im = radmc_image.readImage(datei)
        
        # Calculate closure phases (1:1 from notebook)
        with open(os.devnull, 'w') as devnull:
            with contextlib.redirect_stdout(devnull):
                cp = im.getClosurePhase(bl=bl_cp, pa=pa_cp, dpc=dpc)
        
        cp_radmc = cp["cp"][:,:]
        
        # Max baseline per triangle
        max_baselines = tri_config['max_baselines']
        
        # Select wavelength if specified
        if wavelength_idx is not None:
            cp_sel = cp_radmc[:, wavelength_idx:wavelength_idx+1]
            wav = np.array([cp["wav"][wavelength_idx]])
        else:
            cp_sel = cp_radmc
            wav = cp["wav"]
        
        result = {
            'cp': cp_sel,
            'triangles': bl_cp,
            'triangle_names': tri_config['names'],
            'max_baselines': max_baselines,
            'wavelengths': wav
        }
        
    finally:
        os.chdir(old_dir)
    
    return result
    """
    Calculate closure phases using radmc3dPy.getClosurePhase()
    
    Parameters:
    -----------
    fits_file : str
        Path to FITS file
    dpc : float
        Distance in parsecs
    wavelength_idx : int, optional
        Which wavelength to use
    
    Returns:
    --------
    dict with closure phase data
    """
    if not RADMC3D_AVAILABLE:
        raise ImportError("radmc3dPy is required for closure phase calculation")
    
    # Get triangles
    tri_config = get_closure_triangles()
    bl_cp = tri_config['triangles']
    pa_cp = tri_config['position_angles']
    
    # Convert PA (from notebook: pa_cp = (90-pa_cp) %360)
    pa_cp = np.array(pa_cp)
    pa_cp = (90 - pa_cp) % 360
    
    # Read image
    print(f"Reading {fits_file}...")
    with open(os.devnull, 'w') as devnull:
        with contextlib.redirect_stdout(devnull):
            im = radmc_image.readImage(fits_file)
    
    # Calculate closure phases (1:1 from notebook)
    print("Calculating closure phases...")
    with open(os.devnull, 'w') as devnull:
        with contextlib.redirect_stdout(devnull):
            cp = im.getClosurePhase(bl=bl_cp, pa=pa_cp, dpc=dpc)
    
    cp_radmc = cp["cp"][:,:]
    
    # Max baseline per triangle
    max_baselines = tri_config['max_baselines']
    
    # Select wavelength if specified
    if wavelength_idx is not None:
        cp_sel = cp_radmc[:, wavelength_idx:wavelength_idx+1]
        wav = np.array([cp["wav"][wavelength_idx]])
    else:
        cp_sel = cp_radmc
        wav = cp["wav"]
    
    return {
        'cp': cp_sel,
        'triangles': bl_cp,
        'triangle_names': tri_config['names'],
        'max_baselines': max_baselines,
        'wavelengths': wav
    }


# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def plot_uv_coverage(vis_data, ax=None, show_negative=True):
    """Plot UV coverage"""
    import matplotlib.pyplot as plt
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    
    u = vis_data['u'] / 1e6  # Convert to Mega-lambda
    v = vis_data['v'] / 1e6
    
    for i in range(u.shape[0]):
        if show_negative:
            u_plot = np.concatenate([u[i, :], -u[i, :]])
            v_plot = np.concatenate([v[i, :], -v[i, :]])
        else:
            u_plot = u[i, :]
            v_plot = v[i, :]
        
        ax.scatter(u_plot, v_plot, s=2, alpha=0.6)
    
    ax.set_xlabel('U [Mλ]', fontsize=12, fontweight='bold')
    ax.set_ylabel('V [Mλ]', fontsize=12, fontweight='bold')
    ax.set_title('UV Coverage', fontsize=14, fontweight='bold')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    return ax


def plot_visibility_amplitude(vis_data, ax=None, wavelength_colored=True):
    """Plot visibility amplitudes vs spatial frequency"""
    import matplotlib.pyplot as plt
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    baselines = vis_data['bl']
    wavelengths = vis_data['wav']
    vis2 = vis_data['vis2']
    
    n_wavelengths = len(wavelengths)
    
    if wavelength_colored and n_wavelengths > 1:
        cmap = plt.cm.inferno
        
        for i_bl in range(len(baselines)):
            for i_wav in range(n_wavelengths):
                spatial_freq = baselines[i_bl] / wavelengths[i_wav]
                color = cmap(i_wav / (n_wavelengths - 1))
                ax.scatter(spatial_freq, vis2[i_bl, i_wav], s=30, color=color, 
                          marker='.', alpha=0.8)
        
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=wavelengths[0], vmax=wavelengths[-1]))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Wavelength [µm]', fontsize=12)
    else:
        for i_bl in range(len(baselines)):
            spatial_freq = baselines[i_bl] / wavelengths[0]
            ax.scatter(spatial_freq, vis2[i_bl, 0], s=30, marker='.', alpha=0.8)
    
    ax.set_xlabel('Spatial Frequency [10⁶ cycles/rad]', fontsize=12, fontweight='bold')
    ax.set_ylabel('Visibility²', fontsize=12, fontweight='bold')
    ax.set_title('Visibility Amplitudes', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    return ax


def plot_closure_phases(cp_data, ax=None, wavelength_colored=True):
    """Plot closure phases vs spatial frequency (like notebook)"""
    import matplotlib.pyplot as plt
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    cp = cp_data['cp']
    max_baselines = cp_data['max_baselines']
    wavelengths = cp_data['wavelengths']
    
    n_wavelengths = len(wavelengths)
    
    # Calculate spatial frequency: baseline / wavelength (from notebook)
    # spatial_freq = baseline_cp[:, np.newaxis] / cp["wav"]
    
    if wavelength_colored and n_wavelengths > 1:
        cmap = plt.cm.inferno
        
        for i_tri in range(len(max_baselines)):
            for i_wav in range(n_wavelengths):
                # Spatial frequency in 10^6 cycles/rad
                spatial_freq = max_baselines[i_tri] / wavelengths[i_wav]
                color = cmap(i_wav / (n_wavelengths - 1))
                ax.scatter(spatial_freq, cp[i_tri, i_wav], s=30, 
                          color=color, marker='.', alpha=0.8)
        
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=wavelengths[0], vmax=wavelengths[-1]))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Wavelength [µm]', fontsize=12)
    else:
        # Single wavelength - plot all triangles
        spatial_freqs = max_baselines / wavelengths[0]
        ax.scatter(spatial_freqs, cp[:, 0], s=30, marker='.', alpha=0.8)
    
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel('Spatial Frequency [10⁶ cycles/rad]', fontsize=12, fontweight='bold')
    ax.set_ylabel('Closure Phase [°]', fontsize=12, fontweight='bold')
    ax.set_title('Closure Phases', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    return ax


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def process_fits_file(run_dir, dpc=144, wavelength_idx=None):
    """Complete processing pipeline - reads image.out from run directory"""
    
    vis_data = calculate_visibilities(run_dir, dpc=dpc, wavelength_idx=wavelength_idx)
    cp_data = calculate_closure_phases(run_dir, dpc=dpc, wavelength_idx=wavelength_idx)
    
    print("✅ Processing complete!")
    print(f"  - {len(vis_data['bl'])} baselines")
    print(f"  - {len(cp_data['triangles'])} closure triangles")
    print(f"  - {len(vis_data['wav'])} wavelength(s)")
    
    return {
        'visibilities': vis_data,
        'closure_phases': cp_data
    }


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        fits_file = sys.argv[1]
        result = process_fits_file(fits_file)
        
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        plot_uv_coverage(result['visibilities'], ax=axes[0])
        plot_visibility_amplitude(result['visibilities'], ax=axes[1])
        plot_closure_phases(result['closure_phases'], ax=axes[2])
        
        plt.tight_layout()
        plt.show()
    else:
        print("Usage: python interferometry_v2.py <fits_file>")