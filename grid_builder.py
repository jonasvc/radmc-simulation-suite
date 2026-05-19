"""
Smart adaptive grid builder for RADMC-3D disk simulations.

Builds an optimal non-uniform spherical r x theta x phi grid that concentrates
resolution only where active structures require it.  For inactive (zero-amplitude)
structures the corresponding resolution overhead is zero.

Public API
----------
build_smart_grid(ppar, verbose=True) -> dict
    Main entry point.  Returns a dict with keys
    xbound, nx, ybound, ny, zbound, nz  (ready to pass to problemSetupDust).

CLI
---
python grid_builder.py --config config_baseline_asym.py [--dry-run] [--plot]
"""

from __future__ import print_function
import argparse
import importlib.util
import os
import sys

import numpy as np


# ---------------------------------------------------------------------------
# Physical constants (CGS)
# ---------------------------------------------------------------------------
AU  = 1.496e13   # cm
PI  = np.pi
MS  = 1.989e33   # g
RS  = 6.96e10    # cm

_EVAL_NS = {'au': AU, 'AU': AU, 'pi': PI, 'PI': PI,
             'ms': MS, 'MS': MS, 'rs': RS, 'RS': RS, 'np': np}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ev(val):
    """Evaluate a param value that may be a string expression or numeric."""
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        try:
            return float(eval(val.strip(), dict(_EVAL_NS)))
        except Exception:
            return val
    if isinstance(val, (list, tuple)):
        return [_ev(v) for v in val]
    return val


def _ev_list(val, default=None):
    """Evaluate to a flat list of floats."""
    if val is None:
        return [0.0] if default is None else default
    result = _ev(val)
    if isinstance(result, list):
        return [float(x) for x in result]
    try:
        return [float(result)]
    except (TypeError, ValueError):
        return [0.0]


def _nonzero(val):
    """True if any element is non-zero."""
    try:
        arr = np.atleast_1d(np.array(_ev_list(val), dtype=float))
        return bool(np.any(arr != 0.0))
    except Exception:
        return bool(val)


# ---------------------------------------------------------------------------
# Active structure detection
# ---------------------------------------------------------------------------

def _detect_active(ppar):
    """Return dict of which model structures have non-zero amplitudes."""
    return {
        'fourier_h':   _nonzero(ppar.get('h_fourier_aj', [0])) or
                       _nonzero(ppar.get('h_fourier_bj', [0])),
        'fourier_sig': _nonzero(ppar.get('sig_fourier_aj', [0])) or
                       _nonzero(ppar.get('sig_fourier_bj', [0])),
        'spiral_h':    abs(_ev(ppar.get('h_spiral_amp', 0))) > 0,
        'spiral_sig':  abs(_ev(ppar.get('sig_spiral_amp', 0))) > 0,
        'vortex_h':    _nonzero(ppar.get('h_vortex_amp', [0])),
        'vortex_sig':  _nonzero(ppar.get('sig_vortex_amp', [0])),
        'inner_rim':   _ev(ppar.get('prim_rout', 0.0)) >= 1.0,
        'srim':        _ev(ppar.get('srim_rout', 0.0)) > 0.0,
        'gaps':        any(abs(df) > 0 and abs(df) < 1.0
                          for df in _ev_list(ppar.get('gap_drfact', [0.0]))),
        'warp':        bool(_ev(ppar.get('enable_warp', False))),
        'inner_edge':  bool(_ev(ppar.get('use_inner_edge_shadow', False))),
    }


# ---------------------------------------------------------------------------
# r-grid
# ---------------------------------------------------------------------------

def _build_r_grid(ppar, active):
    """
    Build radial segment boundaries and cell counts.

    Returns
    -------
    xbound : list of float  (cm)
    nx     : list of int
    segs   : list of dict   (for summary printing)
    """
    rin   = _ev(ppar['rin'])
    rdisk = _ev(ppar['rdisk'])

    # ---- collect breakpoints -------------------------------------------------
    bps = [0.1 * rin, rin]

    rim_end = rin

    if active['inner_rim']:
        r_prim = _ev(ppar['prim_rout']) * rin
        bps.append(r_prim)
        rim_end = max(rim_end, r_prim)

    if active['srim']:
        r_srim = _ev(ppar['srim_rout']) * rin
        if r_srim > rim_end * 1.005:
            bps.append(r_srim)
            rim_end = max(rim_end, r_srim)

    # Ensure rim_end > rin by at least a small amount so we always have a
    # separate inner segment even when no puffed rim is active.
    if rim_end <= rin * 1.005:
        rim_end = rin * 1.5
        bps.append(rim_end)

    # Gap edges
    if active['gaps']:
        grins  = _ev_list(ppar.get('gap_rin',    [0.0]))
        grouts = _ev_list(ppar.get('gap_rout',   [0.0]))
        gdfs   = _ev_list(ppar.get('gap_drfact', [0.0]))
        for rg_in, rg_out, df in zip(grins, grouts, gdfs):
            if abs(df) > 0 and abs(df) < 1.0 and rg_out > rim_end:
                rg_in  = max(rim_end * 1.01, rg_in)
                rg_out = min(rdisk * 0.995, rg_out)
                if rg_in < rg_out:
                    bps.extend([rg_in, rg_out])

    # Vortex zones
    for amp_k, r0_k, wr_k in [
        ('h_vortex_amp',   'h_vortex_r0',   'h_vortex_width_r'),
        ('sig_vortex_amp', 'sig_vortex_r0', 'sig_vortex_width_r'),
    ]:
        amps = np.atleast_1d(_ev_list(ppar.get(amp_k, [0.0])))
        r0s  = np.atleast_1d(_ev_list(ppar.get(r0_k,  [50*AU])))
        wrs  = np.atleast_1d(_ev_list(ppar.get(wr_k,  [10*AU])))
        for amp, r0, wr in zip(amps, r0s, wrs):
            if amp > 0 and rin < r0 < rdisk:
                lo = max(rim_end * 1.001, r0 - 2.0*wr)
                hi = min(rdisk  * 0.999, r0 + 2.0*wr)
                if lo < hi:
                    bps.extend([lo, hi])

    bps.append(rdisk)
    bps.append(max(rdisk * 1.1, rdisk + 20*AU))

    # ---- deduplicate --------------------------------------------------------
    bps = sorted(set(bps))
    clean = [bps[0]]
    for b in bps[1:]:
        if b > clean[-1] * 1.005:
            clean.append(b)
    bps = clean

    # ---- assign cell counts per segment -------------------------------------
    rim_log   = np.log(rim_end / rin)
    main_log  = np.log(rdisk  / rim_end)

    xbound = []
    nx     = []
    segs   = []

    for i in range(len(bps) - 1):
        r_lo, r_hi = bps[i], bps[i+1]

        if r_hi <= rin:
            # pre-disk buffer
            n     = 20
            label = 'pre-disk buffer'

        elif r_lo < rdisk and r_hi <= rim_end * 1.001:
            # inner rim / srim region  — 50 cells total, log-proportional
            seg_log = np.log(r_hi / max(r_lo, rin))
            frac    = seg_log / rim_log if rim_log > 0 else 1.0
            n       = max(8, round(50 * frac))
            if active['inner_rim'] and active['srim']:
                label = 'inner rim / srim'
            elif active['inner_rim']:
                label = 'inner rim'
            elif active['srim']:
                label = 'srim taper'
            else:
                label = 'inner disk'

        elif r_lo >= rdisk:
            # outer buffer
            n     = 10
            label = 'outer buffer'

        else:
            # main disk
            seg_log = np.log(r_hi / r_lo)
            frac    = seg_log / main_log if main_log > 0 else 1.0

            # Is this a vortex zone?
            in_vortex = False
            for amp_k, r0_k, wr_k in [
                ('h_vortex_amp',   'h_vortex_r0',   'h_vortex_width_r'),
                ('sig_vortex_amp', 'sig_vortex_r0', 'sig_vortex_width_r'),
            ]:
                amps = np.atleast_1d(_ev_list(ppar.get(amp_k, [0.0])))
                r0s  = np.atleast_1d(_ev_list(ppar.get(r0_k,  [50*AU])))
                wrs  = np.atleast_1d(_ev_list(ppar.get(wr_k,  [10*AU])))
                for amp, r0, wr in zip(amps, r0s, wrs):
                    if amp > 0 and r_lo < r0 + 2*wr and r_hi > r0 - 2*wr:
                        in_vortex = True

            if in_vortex:
                n     = max(12, round(100 * frac * 3.0))
                label = 'vortex zone'
            else:
                n     = max(5, round(100 * frac))
                label = 'main disk'

        xbound.append(r_lo)
        nx.append(n)
        segs.append({'r_lo': r_lo/AU, 'r_hi': r_hi/AU, 'n': n, 'label': label})

    xbound.append(bps[-1])
    return xbound, nx, segs


# ---------------------------------------------------------------------------
# theta-grid
# ---------------------------------------------------------------------------

def _build_theta_grid(ppar):
    """
    Build a colatitude grid concentrated toward the midplane (theta=pi/2).

    Uses 6 segments total (3 per hemisphere), with the innermost segment
    determined by the expected disk scale-height extent.

    Returns
    -------
    ybound : list of float (radians)
    ny     : list of int
    info   : dict  (for summary)
    """
    rin      = _ev(ppar['rin'])
    rdisk    = _ev(ppar['rdisk'])
    hrdisk   = _ev(ppar['hrdisk'])
    hrpivot  = _ev(ppar['hrpivot'])
    plh      = _ev(ppar['plh'])

    # Scale height at outer disk
    H_outer  = hrdisk * (rdisk / hrpivot)**plh * rdisk

    # Angular half-extent of the disk (3 scale heights, with 50% margin)
    sin_dt   = min(0.98, 3.0 * H_outer / rdisk)
    delta_th = np.arcsin(sin_dt) * 1.5          # with margin
    delta_th = min(delta_th, PI/4)

    # Puffed inner rim adds extra vertical extent
    prim_rout    = _ev(ppar.get('prim_rout',    0.0))
    hpr_prim     = _ev(ppar.get('hpr_prim_rout', 0.0))
    if prim_rout >= 1.0 and hpr_prim > 0.0:
        delta_th = max(delta_th, 3.0 * hpr_prim * 1.5)

    # Segment breakpoints (upper hemisphere, symmetric lower)
    th_mid        = PI / 2.0
    th_disk_edge  = th_mid - delta_th          # boundary between atm / disk
    th_polar      = 0.30                        # sparse polar cap boundary

    # Guard: disk_edge must sit between polar and midplane
    th_disk_edge  = max(th_polar + 0.05, min(th_disk_edge, th_mid - 0.05))

    # Cell counts
    n_polar = 8    # [0, th_polar]
    n_atm   = 15   # [th_polar, th_disk_edge]
    n_mid   = 40   # [th_disk_edge, pi/2]  — densest

    # Build ybound / ny for full sphere
    ybound = [
        0.0,
        th_polar,
        th_disk_edge,
        th_mid,
        PI - th_disk_edge,
        PI - th_polar,
        PI,
    ]
    ny = [n_polar, n_atm, n_mid, n_mid, n_atm, n_polar]

    # Cell size near midplane (rad/cell)
    cell_th_mid = (th_mid - th_disk_edge) / n_mid

    # Scale height in theta at outer disk
    sh_th = H_outer / rdisk    # rad

    info = {
        'n_total'      : sum(ny),
        'th_disk_edge' : th_disk_edge,
        'cell_th_mid'  : cell_th_mid,
        'sh_th'        : sh_th,
        'cells_per_sh' : sh_th / cell_th_mid if cell_th_mid > 0 else 0,
    }
    return ybound, ny, info


# ---------------------------------------------------------------------------
# phi-grid
# ---------------------------------------------------------------------------

def _compute_n_phi(ppar, active):
    """Return the optimal number of phi cells based on active structures."""
    n_phi = 120   # baseline minimum (enough for smooth images)

    # Fourier modes: need ≥ 6 cells per half-cycle at highest active mode
    if active['fourier_h'] or active['fourier_sig']:
        aj = _ev_list(ppar.get('h_fourier_aj',   [0.0]))
        bj = _ev_list(ppar.get('h_fourier_bj',   [0.0]))
        as_ = _ev_list(ppar.get('sig_fourier_aj', [0.0]))
        bs  = _ev_list(ppar.get('sig_fourier_bj', [0.0]))
        all_c = aj + bj + as_ + bs
        # highest non-zero index
        j_max = 0
        for idx, c in enumerate(all_c):
            if abs(c) > 0:
                j_max = max(j_max, idx % max(len(aj), 1))
        n_phi = max(n_phi, 8 * max(1, j_max) * 2)

    # Spirals: need ~8 cells per spiral arm width over 2*pi
    if active['spiral_h'] or active['spiral_sig']:
        n_arms = max(1, int(_ev(ppar.get('n_arms', 1))))
        width  = _ev(ppar.get('spiral_width_phi', 0.5))
        if width > 0:
            n_phi = max(n_phi, int(np.ceil(2*PI / width * 8)) * n_arms)

    # Vortices: need ~8 cells per (narrowest active) vortex width in phi
    if active['vortex_h'] or active['vortex_sig']:
        min_w = float('inf')
        for amp_k, wp_k in [('h_vortex_amp',   'h_vortex_width_phi'),
                             ('sig_vortex_amp', 'sig_vortex_width_phi')]:
            amps  = _ev_list(ppar.get(amp_k, [0.0]))
            wphis = _ev_list(ppar.get(wp_k,  [0.5]))
            for amp, wp in zip(amps, wphis):
                if amp > 0 and wp > 0:
                    min_w = min(min_w, wp)
        if min_w < float('inf'):
            n_phi = max(n_phi, int(np.ceil(2*PI / min_w * 8)))

    # Round up to next multiple of 12 (divisible by both 3 and 4)
    n_phi = int(np.ceil(n_phi / 12) * 12)

    # Safety cap
    if n_phi > 1200:
        n_phi = 1200

    return n_phi


def _build_phi_grid(n_phi):
    """Uniform phi grid over [0, 2pi]. Returns (zbound, nz)."""
    half = n_phi // 2
    return [0.0, PI, 2.0*PI], [half, n_phi - half]


# ---------------------------------------------------------------------------
# Cell interface arrays (for write_amr_grid and validation)
# ---------------------------------------------------------------------------

def _interfaces(bounds, counts, log_spaced):
    """
    Compute cell wall positions from segment bounds + counts.

    For all segments except the last: includes left boundary, excludes right.
    For the last segment: includes both boundaries.
    Requires counts[-1] >= 2.
    """
    pts = []
    n_segs = len(counts)
    for i, (lo, hi, n) in enumerate(zip(bounds[:-1], bounds[1:], counts)):
        n = max(2, n) if i == n_segs - 1 else max(1, n)
        if i < n_segs - 1:
            frac = np.arange(n, dtype=float) / float(n)
        else:
            frac = np.arange(n, dtype=float) / float(n - 1)
        if log_spaced:
            seg_pts = lo * (hi / lo) ** frac
        else:
            seg_pts = lo + (hi - lo) * frac
        pts.extend(seg_pts.tolist())
    return np.array(pts)


def _nr_from_grid(xbound, nx):
    xi = _interfaces(xbound, nx, log_spaced=True)
    return len(xi) - 1, xi


def _nth_from_grid(ybound, ny):
    yi = _interfaces(ybound, ny, log_spaced=False)
    return len(yi) - 1, yi


def _nph_from_grid(zbound, nz):
    zi = _interfaces(zbound, nz, log_spaced=False)
    return len(zi) - 1, zi


# ---------------------------------------------------------------------------
# RAM estimate
# ---------------------------------------------------------------------------

def _estimate_ram_gib(ncell, n_spec=1):
    """Lower-bound RAM estimate (GiB) matching single_run.estimate_grid_memory."""
    B = 8  # float64 bytes
    python_setup   = 5  * ncell * B
    dust_arrays    = 2  * n_spec * ncell * B
    radmc_core     = (2 * n_spec + 4) * ncell * B
    return (python_setup + dust_arrays + radmc_core) / 1024**3


# ---------------------------------------------------------------------------
# Validation and summary printing
# ---------------------------------------------------------------------------

def _print_summary(xbound, nx, ybound, ny, zbound, nz,
                   r_segs, th_info, active, orig_ppar):
    """Print a formatted grid summary to stdout."""
    nr,  xi = _nr_from_grid(xbound, nx)
    nth, yi = _nth_from_grid(ybound, ny)
    nph, zi = _nph_from_grid(zbound, nz)

    total = nr * nth * nph

    # Original grid size for comparison
    orig_nr  = sum(orig_ppar['nx']) if isinstance(orig_ppar.get('nx'), list) else int(orig_ppar.get('nx', 0))
    orig_nth = sum(orig_ppar['ny']) if isinstance(orig_ppar.get('ny'), list) else int(orig_ppar.get('ny', 0))
    orig_nph = sum(orig_ppar['nz']) if isinstance(orig_ppar.get('nz'), list) else int(orig_ppar.get('nz', 0))
    orig_total = orig_nr * orig_nth * orig_nph

    n_spec = 1

    ram_new  = _estimate_ram_gib(total, n_spec)
    ram_orig = _estimate_ram_gib(orig_total, n_spec) if orig_total else 0.0

    W = 68
    print('\n' + '─' * W)
    print('  Smart Grid Builder — Summary')
    print('─' * W)

    # Active structures
    on  = [k for k, v in active.items() if v]
    off = [k for k, v in active.items() if not v]
    print(f'  Active structures  : {", ".join(on) if on else "none"}')
    print(f'  Inactive (skipped) : {", ".join(off) if off else "none"}')
    print()

    # r-grid
    print(f'  r-grid  {nr} cells ({len(nx)} segments)')
    for s in r_segs:
        lo_str = f'{s["r_lo"]:>8.2f}'
        hi_str = f'{s["r_hi"]:<8.2f}'
        print(f'    [{lo_str}, {hi_str}] au  {s["n"]:>4d} cells  ({s["label"]})')
    # dr/r range in main disk
    if len(xi) > 1:
        dr_r = np.diff(xi) / xi[:-1]
        print(f'    dr/r range: {dr_r.min()*100:.1f}% – {dr_r.max()*100:.1f}%')
    print()

    # theta-grid
    print(f'  θ-grid  {nth} cells ({len(ny)} segments, midplane-concentrated)')
    th_pairs = list(zip(ybound[:-1], ybound[1:], ny))
    for lo, hi, n in th_pairs:
        print(f'    [{lo:5.3f}, {hi:5.3f}] rad  {n:>4d} cells')
    cps = th_info['cells_per_sh']
    print(f'    Cell size near midplane: {th_info["cell_th_mid"]:.4f} rad  '
          f'({cps:.1f} cells / scale-height at outer disk)')
    print()

    # phi-grid
    print(f'  φ-grid  {nph} cells (uniform)')
    reasons = []
    if not any(active[k] for k in ('fourier_h', 'fourier_sig',
                                    'spiral_h', 'spiral_sig',
                                    'vortex_h', 'vortex_sig')):
        reasons.append('no active azimuthal structures → minimum 120')
    if active['spiral_h'] or active['spiral_sig']:
        reasons.append('set by spiral width')
    if active['vortex_h'] or active['vortex_sig']:
        reasons.append('set by vortex width')
    if reasons:
        print(f'    ({"; ".join(reasons)})')
    print()

    # Totals
    print(f'  {"TOTAL CELLS":20s}  {total:>12,d}')
    if orig_total:
        reduction = orig_total / total if total else 0
        print(f'  {"Config grid":20s}  {orig_total:>12,d}  ({orig_nr} × {orig_nth} × {orig_nph})')
        print(f'  {"Reduction":20s}  {reduction:>11.0f}×')
    print()
    print(f'  RAM lower bound (new)    : {ram_new:.2f} GiB')
    if orig_total:
        print(f'  RAM lower bound (config) : {ram_orig:.2f} GiB')

    # Warnings
    warnings = []
    if total > 50_000_000:
        warnings.append(f'Grid exceeds 50 M cells — consider reducing active structure sharpness')
    if active['vortex_h'] or active['vortex_sig']:
        wphis = (_ev_list(orig_ppar.get('h_vortex_width_phi',  [0.5])) +
                 _ev_list(orig_ppar.get('sig_vortex_width_phi', [0.5])))
        if any(wp < 0.05 for wp in wphis):
            warnings.append('vortex_width_phi < 0.05 rad → Nφ very large; '
                            'consider increasing vortex_width_phi')

    if warnings:
        print()
        for w in warnings:
            print(f'  ⚠  {w}')

    print('─' * W + '\n')


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def build_smart_grid(ppar, verbose=True):
    """
    Compute an optimal RADMC-3D spherical grid from a pipeline params dict.

    Parameters
    ----------
    ppar    : dict  — same format as the Pipeline config params dict.
              String values like '0.5*au' are evaluated automatically.
    verbose : bool  — print the summary table (default True).

    Returns
    -------
    dict with keys:
        xbound, nx   — radial segment boundaries (cm) and cell counts
        ybound, ny   — colatitude boundaries (rad) and cell counts
        zbound, nz   — azimuthal boundaries (rad) and cell counts
    """
    active = _detect_active(ppar)

    xbound, nx, r_segs = _build_r_grid(ppar, active)
    ybound, ny, th_info = _build_theta_grid(ppar)

    n_phi          = _compute_n_phi(ppar, active)
    zbound, nz     = _build_phi_grid(n_phi)

    if verbose:
        _print_summary(xbound, nx, ybound, ny, zbound, nz,
                       r_segs, th_info, active, ppar)

    return {
        'xbound': xbound,
        'nx':     nx,
        'ybound': ybound,
        'ny':     ny,
        'zbound': zbound,
        'nz':     nz,
    }


# ---------------------------------------------------------------------------
# Standalone: write amr_grid.inp directly
# ---------------------------------------------------------------------------

def write_amr_grid(xbound, nx, ybound, ny, zbound, nz, fname='amr_grid.inp'):
    """Write a RADMC-3D compatible amr_grid.inp for a regular spherical grid."""
    nr,  xi = _nr_from_grid(xbound, nx)
    nth, yi = _nth_from_grid(ybound, ny)
    nph, zi = _nph_from_grid(zbound, nz)

    print(f'Writing {fname}  ({nr} × {nth} × {nph} = {nr*nth*nph:,} cells)')
    with open(fname, 'w') as f:
        f.write('1\n')      # iformat
        f.write('0\n')      # grid_style (0 = regular)
        f.write('100\n')    # coordsystem (100 = spherical)
        f.write('0\n')      # gridinfo
        f.write('1 1 1\n')  # active dimensions
        f.write(f'{nr} {nth} {nph}\n')
        for v in xi:
            f.write('%.9e\n' % v)
        for v in yi:
            f.write('%.9e\n' % v)
        for v in zi:
            f.write('%.9e\n' % v)


def _make_diagnostic_plot(xbound, nx, ybound, ny, ppar, active, outfile=None):
    """Save (or show) a two-panel figure: dr/r vs r, and dtheta vs theta."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib not available — skipping diagnostic plot')
        return

    nr, xi = _nr_from_grid(xbound, nx)
    nth, yi = _nth_from_grid(ybound, ny)
    rin   = _ev(ppar['rin'])
    rdisk = _ev(ppar['rdisk'])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # --- panel 1: dr/r vs r --------------------------------------------------
    ax = axes[0]
    r_c  = 0.5 * (xi[:-1] + xi[1:])
    dr_r = np.diff(xi) / r_c
    ax.semilogy(r_c / AU, dr_r * 100, color='steelblue', linewidth=1.2)
    ax.axvline(rin   / AU, color='orange', lw=1.0, ls='--', label='rin')
    ax.axvline(rdisk / AU, color='red',    lw=1.0, ls='--', label='rdisk')

    # Structure markers
    if active['inner_rim']:
        ax.axvline(_ev(ppar['prim_rout']) * rin / AU,
                   color='purple', lw=0.8, ls=':', label='prim_rout')
    if active['srim']:
        ax.axvline(_ev(ppar['srim_rout']) * rin / AU,
                   color='teal', lw=0.8, ls=':', label='srim_rout')
    for amp_k, r0_k in [('h_vortex_amp',   'h_vortex_r0'),
                         ('sig_vortex_amp', 'sig_vortex_r0')]:
        amps = _ev_list(ppar.get(amp_k, [0.0]))
        r0s  = _ev_list(ppar.get(r0_k,  [0.0]))
        for amp, r0 in zip(amps, r0s):
            if amp > 0:
                ax.axvline(r0 / AU, color='green', lw=0.8, ls='-.',
                           label='vortex_r0')

    ax.set_xlabel('r (au)')
    ax.set_ylabel('dr/r (%)')
    ax.set_title(f'r-grid  ({nr} cells)')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # --- panel 2: dtheta vs theta --------------------------------------------
    ax = axes[1]
    th_c  = 0.5 * (yi[:-1] + yi[1:])
    dth   = np.diff(yi)
    ax.semilogy(np.degrees(th_c), np.degrees(dth), color='darkorange', linewidth=1.2)

    # Shade disk region
    H_outer = (_ev(ppar['hrdisk']) *
               (_ev(ppar['rdisk']) / _ev(ppar['hrpivot']))**_ev(ppar['plh']) *
               _ev(ppar['rdisk']))
    sin_dt   = min(0.98, 3.0 * H_outer / _ev(ppar['rdisk']))
    delta_th = np.degrees(np.arcsin(sin_dt))
    ax.axvspan(90 - delta_th, 90 + delta_th, alpha=0.15, color='green',
               label='disk ±3H')
    ax.axvline(90, color='gray', lw=0.8, ls='--', label='midplane')

    ax.set_xlabel('θ (degrees)')
    ax.set_ylabel('dθ (degrees)')
    ax.set_title(f'θ-grid  ({nth} cells)')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    fig.suptitle('Smart Grid — cell spacing diagnostic', fontsize=11)
    fig.tight_layout()

    if outfile:
        fig.savefig(outfile, dpi=150, bbox_inches='tight')
        print(f'  Diagnostic plot saved to {outfile}')
    else:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _load_config_as_dict(config_path):
    """Import a pipeline config .py file and return its attributes as a dict."""
    spec   = importlib.util.spec_from_file_location('_grid_cfg', config_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    params = {}
    for key in dir(module):
        if key.startswith('__'):
            continue
        val = getattr(module, key)
        if not str(type(val)).startswith("<class 'module'>"):
            params[key] = val
    return params


def main():
    parser = argparse.ArgumentParser(
        description='Smart adaptive grid builder for RADMC-3D disk simulations.')
    parser.add_argument('--config', default=None,
                        help='Path to a pipeline config .py file '
                             '(default: Pipeline/config.py)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show summary only — do not write amr_grid.inp')
    parser.add_argument('--plot', action='store_true',
                        help='Save a diagnostic plot grid_builder_diag.png')
    parser.add_argument('--output', default='amr_grid.inp',
                        help='Output file name (default: amr_grid.inp)')
    args = parser.parse_args()

    # Locate config file
    if args.config:
        cfg_path = args.config
    else:
        here = os.path.dirname(os.path.abspath(__file__))
        cfg_path = os.path.join(here, 'config.py')

    if not os.path.exists(cfg_path):
        print(f'Error: config file not found: {cfg_path}')
        sys.exit(1)

    print(f'Loading config: {cfg_path}')
    ppar = _load_config_as_dict(cfg_path)

    result = build_smart_grid(ppar, verbose=True)

    if args.plot:
        _make_diagnostic_plot(
            result['xbound'], result['nx'],
            result['ybound'], result['ny'],
            ppar, _detect_active(ppar),
            outfile='grid_builder_diag.png',
        )

    if args.dry_run:
        print('--dry-run: amr_grid.inp NOT written.')
    else:
        write_amr_grid(
            result['xbound'], result['nx'],
            result['ybound'], result['ny'],
            result['zbound'], result['nz'],
            fname=args.output,
        )


if __name__ == '__main__':
    main()
