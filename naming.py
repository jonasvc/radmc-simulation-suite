"""
Module for automatic naming and categorization of simulations.

Folder structure:

  Sandbox runs:
    simulations/sandbox/<category>/<YYYY-MM-DD_HHMM_name>/

  IRYSS runs:
    simulations/iryss/<source>/<opacity>_incl<inclination>/<YYYY-MM-DD_HHMM_name>/

  Batch runs follow the same top-level split:
    simulations/sandbox/batch/<YYYY-MM-DD_HHMM_name>/<NNN_suffix>/
    simulations/iryss/<source>/<opacity>_incl<inclination>/batch/<YYYY-MM-DD_HHMM_name>/<NNN_suffix>/
"""

import os


def determine_category(params):
    """
    Determine simulation category based on active features.
    Returns one of: baseline, fourier, spiral, vortex, warp,
    inner_edge, combined_<features>
    """
    active_features = []

    # Spiral
    if params.get('h_spiral_amp', 0) > 0 or params.get('sig_spiral_amp', 0) > 0:
        n_arms = params.get('n_arms', 0)
        active_features.append(f'spiral_{n_arms}arms' if n_arms > 0 else 'spiral')

    # Vortex
    vortex_h   = params.get('h_vortex_amp',  [0.0, 0.0])
    vortex_sig = params.get('sig_vortex_amp', [0.0, 0.0])
    if isinstance(vortex_h, (list, tuple)):
        has_vortex = any(v > 0 for v in vortex_h) or any(v > 0 for v in vortex_sig)
    else:
        has_vortex = vortex_h > 0 or vortex_sig > 0
    if has_vortex:
        active_features.append('vortex')

    # Fourier
    fourier_h_a   = params.get('h_fourier_aj',  [0.0] * 5)
    fourier_h_b   = params.get('h_fourier_bj',  [0.0] * 5)
    fourier_sig_a = params.get('sig_fourier_aj', [0.0] * 5)
    fourier_sig_b = params.get('sig_fourier_bj', [0.0] * 5)
    if (any(v != 0 for v in fourier_h_a) or any(v != 0 for v in fourier_h_b) or
            any(v != 0 for v in fourier_sig_a) or any(v != 0 for v in fourier_sig_b)):
        active_features.append('fourier')

    # Warp
    if params.get('enable_warp', False):
        active_features.append('warp')

    # Inner edge shadow
    if params.get('use_inner_edge_shadow', False):
        active_features.append('inner_edge')

    if not active_features:
        return 'baseline'
    if len(active_features) == 1:
        return active_features[0]
    return 'combined_' + '_'.join(active_features)


def format_timestamp(timestamp):
    """
    Convert full timestamp (20231115_143022) to short form (2023-11-15_1430).
    """
    try:
        date_part, time_part = timestamp.split('_')
        date_fmt = f"{date_part[:4]}-{date_part[4:6]}-{date_part[6:8]}"
        time_fmt = time_part[:4]
        return f"{date_fmt}_{time_fmt}"
    except Exception:
        return timestamp


def generate_run_directory(base_dir, base_name, params, timestamp, iryss_meta=None):
    """
    Generate run directory and name.

    Parameters
    ----------
    base_dir : str
        Root simulations directory (e.g. '/home/main/RADMC/Simulations')
    base_name : str
        User-provided name for this run
    params : dict
        Simulation parameters
    timestamp : str
        Raw timestamp (%Y%m%d_%H%M%S)
    iryss_meta : dict or None
        If provided, must contain keys: 'source', 'opacity', 'inclination'
        Example: {'source': 'AB-Aur', 'opacity': 'astrosil', 'inclination': '45'}

    Returns
    -------
    run_dir : str
        Full path to the run directory
    run_name : str
        Short folder name (without path)
    """
    short_ts = format_timestamp(timestamp)
    run_name = f"{short_ts}_{base_name}"

    if iryss_meta:
        source  = iryss_meta['source']
        opacity = iryss_meta['opacity']
        incl    = iryss_meta['inclination']
        run_dir = os.path.join(
            base_dir, 'iryss', source,
            f"{opacity}_incl{incl}",
            run_name
        )
    else:
        category = determine_category(params)
        run_dir  = os.path.join(base_dir, 'sandbox', category, run_name)

    return run_dir, run_name


def generate_batch_run_directory(base_dir, base_name, name_suffix,
                                  params, batch_idx, base_timestamp,
                                  iryss_meta=None):
    """
    Generate directory for a single run inside a batch.

    Layout (sandbox):
        simulations/sandbox/batch/<YYYY-MM-DD_HHMM_name>/<NNN_suffix>/

    Layout (iryss):
        simulations/iryss/<source>/<opacity>_incl<incl>/batch/<YYYY-MM-DD_HHMM_name>/<NNN_suffix>/

    Returns
    -------
    run_dir : str
    run_name : str
    batch_root : str
    """
    short_ts = format_timestamp(base_timestamp)

    if iryss_meta:
        source  = iryss_meta['source']
        opacity = iryss_meta['opacity']
        incl    = iryss_meta['inclination']
        batch_root = os.path.join(
            base_dir, 'iryss', source,
            f"{opacity}_incl{incl}",
            'batch', f"{short_ts}_{base_name}"
        )
    else:
        batch_root = os.path.join(
            base_dir, 'sandbox', 'batch', f"{short_ts}_{base_name}"
        )

    run_name = f"{batch_idx:03d}_{name_suffix}"
    run_dir  = os.path.join(batch_root, run_name)

    return run_dir, run_name, batch_root
