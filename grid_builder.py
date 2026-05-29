"""
Smart adaptive grid builder for RADMC-3D disk simulations.

Evaluates the dust density model on a coarse probe grid, measures the actual
|delta ln rho| gradient in r, theta, and phi, then marches to place cell
interfaces so that every cell satisfies |delta ln rho| <= threshold.

Public API
----------
build_smart_grid(ppar, threshold=0.1, verbose=True) -> dict
    Returns {xbound, nx, ybound, ny, zbound, nz} ready for problemSetupDust.

CLI
---
python grid_builder.py --config config.py [--threshold 0.05] [--dry-run] [--plot]
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
AU  = 1.496e13
PI  = np.pi
MS  = 1.989e33
RS  = 6.96e10

_EVAL_NS = {'au': AU, 'AU': AU, 'pi': PI, 'PI': PI,
             'ms': MS, 'MS': MS, 'rs': RS, 'RS': RS, 'np': np}

# Max |d ln rho / d ln r| allowed on cells adjacent to empty space, so the hard
# sigma cutoffs at rin/rdisk (and drfact=0 gaps) get a few cells, not hundreds.
_EDGE_GCAP = 20.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ev(val):
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
    try:
        arr = np.atleast_1d(np.array(_ev_list(val), dtype=float))
        return bool(np.any(arr != 0.0))
    except Exception:
        return bool(val)


def _refine_inner_wall(r_if, rin, ppar, rim_cells):
    """
    Force rim_cells across the first pressure scale height above rin.

    The probe grid has at most 1-2 points spanning the density step at rin,
    so the central-difference gradient there is badly underestimated and the
    march produces cells that are too large.

    This function computes the correct cell size analytically: the density
    jumps from 0 to its disk value over a length scale of H_rin (the pressure
    scale height at rin), so that is the scale the cells must resolve — exactly
    what a correctly measured gradient would give. The numerical probe route
    is skipped only because it is unreliable at this boundary, not because
    the gradient physics is ignored.
    """
    if rim_cells <= 0:
        return r_if

    hrdisk  = float(_ev(ppar.get('hrdisk',  0.1)))
    hrpivot = float(_ev(ppar.get('hrpivot', 100.0 * AU)))
    plh     = float(_ev(ppar.get('plh',     0.0)))

    H_rin   = hrdisk * (rin / hrpivot) ** plh * rin   # scale height at inner rim
    r_wall  = rin + H_rin                              # outer edge of the forced zone

    r_dense = np.geomspace(rin, r_wall, rim_cells + 1)

    # Keep everything outside [rin, r_wall], replace interior with dense grid
    r_keep  = r_if[(r_if <= rin) | (r_if >= r_wall)]
    return np.unique(np.concatenate([r_keep, r_dense]))


def _guarantee_rim_cells(r_if, rin, hrdisk=0.1, prim_rout=1.0, n_min=10):
    """Ensure at least n_min cells across the inner rim region regardless of threshold.

    When a puffed inner rim is active, the structure extends to prim_rout*rin, so
    the refinement zone covers [rin, prim_rout*rin].  Without a puffed rim, the
    zone falls back to one pressure scale height [rin, rin*(1+2*hrdisk)].
    """
    if prim_rout > 1.0:
        r_outer = rin * prim_rout
    else:
        r_outer = rin * (1.0 + 2.0 * hrdisk)
    n_existing = int(np.sum((r_if > rin) & (r_if < r_outer)))
    if n_existing >= n_min:
        return r_if
    new_pts = np.geomspace(rin, r_outer, n_min + 2)[1:-1]
    return np.unique(np.concatenate([r_if, new_pts]))


# ---------------------------------------------------------------------------
# Active structure detection
# ---------------------------------------------------------------------------

_PHI_STRUCTURE_KEYS = ('fourier_h', 'fourier_sig', 'spiral_h', 'spiral_sig',
                       'vortex_h', 'vortex_sig', 'inner_edge')


def _detect_active(ppar):
    return {
        'fourier_h':      _nonzero(ppar.get('h_fourier_aj', [0])) or
                          _nonzero(ppar.get('h_fourier_bj', [0])),
        'fourier_sig':    _nonzero(ppar.get('sig_fourier_aj', [0])) or
                          _nonzero(ppar.get('sig_fourier_bj', [0])),
        'spiral_h':       abs(_ev(ppar.get('h_spiral_amp', 0))) > 0,
        'spiral_sig':     abs(_ev(ppar.get('sig_spiral_amp', 0))) > 0,
        'vortex_h':       _nonzero(ppar.get('h_vortex_amp', [0])),
        'vortex_sig':     _nonzero(ppar.get('sig_vortex_amp', [0])),
        'inner_rim':      _ev(ppar.get('prim_rout', 0.0)) >= 1.0,
        'srim':           _ev(ppar.get('srim_rout', 0.0)) > 0.0,
        'gaps':           any(df != 1.0 and go > gi for gi, go, df in
                              zip(_ev_list(ppar.get('gap_rin',    [0.0])),
                                  _ev_list(ppar.get('gap_rout',   [0.0])),
                                  _ev_list(ppar.get('gap_drfact', [1.0])))),
        'warp':           bool(_ev(ppar.get('enable_warp', False))),
        'inner_edge':     bool(_ev(ppar.get('use_inner_edge_shadow', False))) and
                          bool(_ev(ppar.get('inner_edge_azimuthal', False))),
        'inner_edge_any': bool(_ev(ppar.get('use_inner_edge_shadow', False))),
        'radial_damping': bool(_ev(ppar.get('use_radial_damping', False))) and
                          _ev(ppar.get('azimuthal_r_max',   0.0)) > 0 and
                          _ev(ppar.get('azimuthal_r_width', 0.0)) > 0,
    }


def _has_phi_structure(active):
    return any(active[k] for k in _PHI_STRUCTURE_KEYS)


# ---------------------------------------------------------------------------
# Density evaluation -- replicates ppdisk_complete.py core physics
# ---------------------------------------------------------------------------

def _eval_damping(ppar, rcyl, active):
    """Radial damping factor DAMP(r) in [0, 1] applied to azimuthal modulations."""
    if active.get('radial_damping', False):
        r_max = _ev(ppar.get('azimuthal_r_max',   0.0))
        r_w   = _ev(ppar.get('azimuthal_r_width', 0.0))
        if r_max > 0 and r_w > 0:
            return 0.5 * (1.0 - np.tanh((rcyl - r_max) / r_w))
    return np.ones_like(rcyl) if hasattr(rcyl, '__len__') else 1.0


def _eval_fourier_mod(ppar, phi, which):
    """Sum of Fourier terms: sum_j [A_j cos(j phi) + B_j sin(j phi)]."""
    prefix = 'h_fourier_' if which == 'h' else 'sig_fourier_'
    ajs = np.array(_ev_list(ppar.get(prefix + 'aj', [0.0])))
    bjs = np.array(_ev_list(ppar.get(prefix + 'bj', [0.0])))
    result = np.zeros_like(phi)
    for j, (a, b) in enumerate(zip(ajs, bjs), start=1):
        if a != 0 or b != 0:
            result += a * np.cos(j * phi) + b * np.sin(j * phi)
    return result


def _eval_hp(ppar, rcyl, phi, active):
    """
    Evaluate pressure scale height H(r_cyl, phi) including all active modulations.
    Replicates the H computation in ppdisk_complete.getDustDensity.
    """
    hrdisk  = _ev(ppar['hrdisk'])
    hrpivot = _ev(ppar['hrpivot'])
    plh     = _ev(ppar['plh'])

    H = hrdisk * (rcyl / hrpivot) ** plh * rcyl

    # Puffed inner rim (exact formula from ppdisk_complete.py)
    if active.get('inner_rim', False):
        rin      = _ev(ppar['rin'])
        p_rout   = _ev(ppar['prim_rout'])
        hpr_prim = _ev(ppar.get('hpr_prim_rout', 0.0))
        if p_rout >= 1.0 and hpr_prim > 0.0:
            hpr0 = hrdisk * (p_rout * rin / hrpivot) ** plh
            if hpr0 > 1e-30:
                dummy  = np.log10(max(hpr0 / hpr_prim, 1e-30)) / np.log10(max(p_rout, 1.001))
                # Clamp the power base away from 0 (poles, rcyl->0) so a
                # negative dummy cannot raise (rcyl/rin) to a huge value and
                # overflow -- same safe_r guard used elsewhere in this module.
                rcyl_safe = np.maximum(rcyl, rin * 1e-6)
                H_prim = hpr_prim * (rcyl_safe / rin) ** dummy * rcyl
                H      = np.maximum(H, H_prim)

    # Inner edge shadow: raised wall at edge_radius
    if active.get('inner_edge_any', False):
        edge_r = _ev(ppar.get('inner_edge_radius', _ev(ppar['rin'])))
        edge_w = _ev(ppar.get('inner_edge_width',  AU))
        edge_h = _ev(ppar.get('inner_edge_height', 2.0))
        r_prof = 0.5 * (1.0 - np.tanh((rcyl - edge_r) / max(edge_w / 4.0, 1e-30 * AU)))
        if active.get('inner_edge', False):
            phi_c  = _ev(ppar.get('inner_edge_phi',       0.0))
            phi_w  = _ev(ppar.get('inner_edge_phi_width', PI / 4.0))
            p_prof = np.exp(-0.5 * ((phi - phi_c) / max(phi_w, 1e-6)) ** 2)
            H      = H * (1.0 + (edge_h - 1.0) * r_prof * p_prof)
        else:
            H = H * (1.0 + (edge_h - 1.0) * r_prof)

    DAMP = _eval_damping(ppar, rcyl, active)

    # Fourier modulation of H
    if active.get('fourier_h', False):
        fs   = _eval_fourier_mod(ppar, phi, 'h')
        ms   = _ev(ppar.get('h_modulation_strength', 1.0))
        asym = _ev(ppar.get('h_asymmetry_factor',    1.0))
        fs_m = fs * ms
        fmod = np.where(fs_m > 0, 1.0 + asym * fs_m, 1.0 + fs_m)
        H    = H * (1.0 + DAMP * (fmod - 1.0))

    # Spiral modulation of H
    if active.get('spiral_h', False):
        rin    = _ev(ppar['rin'])
        amp    = _ev(ppar.get('h_spiral_amp',    0.0))
        pitch  = _ev(ppar.get('spiral_pitch',    1.0))
        n_arms = max(1, int(_ev(ppar.get('n_arms', 2))))
        width  = _ev(ppar.get('spiral_width_phi', 0.5))
        sharp  = _ev(ppar.get('spiral_sharpness', 1.0))
        angle  = phi - pitch * np.log(np.maximum(rcyl, rin * 1e-6) / rin)
        if n_arms > 1:
            period = 2.0 * PI / n_arms
            angle  = angle % period
            angle  = np.where(angle > period / 2.0, period - angle, angle)
        if sharp > 1:
            pattern = 0.5 * (1.0 - np.tanh((np.abs(angle) - width / 2.0) / max(width / 4.0, 1e-9)))
        else:
            pattern = np.exp(-0.5 * (angle / max(width / 2.0, 1e-9)) ** 2)
        H = H * (1.0 + DAMP * amp * pattern)

    # Vortex modulation of H
    if active.get('vortex_h', False):
        amps  = _ev_list(ppar.get('h_vortex_amp',       [0.0]))
        r0s   = _ev_list(ppar.get('h_vortex_r0',        [50 * AU]))
        phi0s = _ev_list(ppar.get('h_vortex_phi0',      [0.0]))
        wrs   = _ev_list(ppar.get('h_vortex_width_r',   [10 * AU]))
        wphis = _ev_list(ppar.get('h_vortex_width_phi', [0.5]))
        for amp, r0, phi0_v, wr, wphi in zip(amps, r0s, phi0s, wrs, wphis):
            if amp == 0:
                continue
            phi0 = _ev(phi0_v) if isinstance(phi0_v, str) else float(phi0_v)
            dr   = (rcyl - r0) / max(wr,   1e-30)
            dphi = ((phi - phi0 + PI) % (2 * PI)) - PI
            dphi = dphi / max(wphi, 1e-30)
            H    = H * (1.0 + DAMP * amp * np.exp(-0.5 * dr ** 2) * np.exp(-0.5 * dphi ** 2))

    # Warp (experimental)
    if active.get('warp', False):
        warp_amp   = _ev(ppar.get('warp_amplitude', 0.1))
        warp_phase = _ev(ppar.get('warp_phase',     0.0))
        warp_mode  = int(_ev(ppar.get('warp_mode',  1)))
        H = H * (1.0 + warp_amp * np.cos(warp_mode * (phi - warp_phase)))

    return H


def _eval_sigma(ppar, rcyl, phi, active):
    """
    Evaluate surface density Sigma(r_cyl, phi) including all active modulations.
    Returns relative units (normalised at rdisk=1 before mdisk scaling).
    """
    rin    = _ev(ppar['rin'])
    rdisk  = _ev(ppar['rdisk'])
    plsig1 = _ev(ppar.get('plsig1', -1.5))
    stype  = int(_ev(ppar.get('sigma_type', 0)))

    safe_r = np.maximum(rcyl, rin * 1e-6)
    if stype == 1:
        taper = np.exp(-(safe_r / rdisk) ** (2.0 - plsig1))
        sig   = (safe_r / rdisk) ** (-plsig1) * taper
    else:
        sig   = (safe_r / rdisk) ** plsig1

    # srim inner taper
    if active.get('srim', False):
        srim_r     = _ev(ppar['srim_rout']) * rin
        srim_plsig = _ev(ppar.get('srim_plsig', -0.5))
        if stype == 0:
            sig_at_srim = (srim_r / rdisk) ** plsig1
        else:
            sig_at_srim = ((srim_r / rdisk) ** (-plsig1) *
                           np.exp(-(srim_r / rdisk) ** (2.0 - plsig1)))
        sig_rim = sig_at_srim * (safe_r / srim_r) ** srim_plsig
        sig     = np.where(rcyl < srim_r, sig_rim, sig)

    # Mask outside disk
    sig = np.where((rcyl < rin) | (rcyl > rdisk), 0.0, sig)

    DAMP = _eval_damping(ppar, rcyl, active)

    # Fourier modulation of sigma
    if active.get('fourier_sig', False):
        fs   = _eval_fourier_mod(ppar, phi, 'sig')
        ms   = _ev(ppar.get('sig_modulation_strength', 1.0))
        asym = _ev(ppar.get('sig_asymmetry_factor',    1.0))
        fs_m = fs * ms
        fmod = np.where(fs_m > 0, 1.0 + asym * fs_m, 1.0 + fs_m)
        sig  = sig * (1.0 + DAMP * (fmod - 1.0))

    # Spiral modulation of sigma
    if active.get('spiral_sig', False):
        amp    = _ev(ppar.get('sig_spiral_amp',   0.0))
        pitch  = _ev(ppar.get('spiral_pitch',     1.0))
        n_arms = max(1, int(_ev(ppar.get('n_arms', 2))))
        width  = _ev(ppar.get('spiral_width_phi', 0.5))
        sharp  = _ev(ppar.get('spiral_sharpness', 1.0))
        angle  = phi - pitch * np.log(np.maximum(rcyl, rin * 1e-6) / rin)
        if n_arms > 1:
            period = 2.0 * PI / n_arms
            angle  = angle % period
            angle  = np.where(angle > period / 2.0, period - angle, angle)
        if sharp > 1:
            pattern = 0.5 * (1.0 - np.tanh((np.abs(angle) - width / 2.0) / max(width / 4.0, 1e-9)))
        else:
            pattern = np.exp(-0.5 * (angle / max(width / 2.0, 1e-9)) ** 2)
        sig = sig * (1.0 + DAMP * amp * pattern)

    # Vortex modulation of sigma
    if active.get('vortex_sig', False):
        amps  = _ev_list(ppar.get('sig_vortex_amp',       [0.0]))
        r0s   = _ev_list(ppar.get('sig_vortex_r0',        [50 * AU]))
        phi0s = _ev_list(ppar.get('sig_vortex_phi0',      [0.0]))
        wrs   = _ev_list(ppar.get('sig_vortex_width_r',   [10 * AU]))
        wphis = _ev_list(ppar.get('sig_vortex_width_phi', [0.5]))
        for amp, r0, phi0_v, wr, wphi in zip(amps, r0s, phi0s, wrs, wphis):
            if amp == 0:
                continue
            phi0 = _ev(phi0_v) if isinstance(phi0_v, str) else float(phi0_v)
            dr   = (rcyl - r0) / max(wr,   1e-30)
            dphi = ((phi - phi0 + PI) % (2 * PI)) - PI
            dphi = dphi / max(wphi, 1e-30)
            sig  = sig * (1.0 + DAMP * amp * np.exp(-0.5 * dr ** 2) * np.exp(-0.5 * dphi ** 2))

    # Gap depletion / enhancement (model applies rho *= df for any df != 1)
    if active.get('gaps', False):
        grins  = _ev_list(ppar.get('gap_rin',    [0.0]))
        grouts = _ev_list(ppar.get('gap_rout',   [0.0]))
        gdfs   = _ev_list(ppar.get('gap_drfact', [1.0]))
        for g_in, g_out, df in zip(grins, grouts, gdfs):
            if df != 1.0 and g_out > g_in:
                sig = np.where((rcyl > g_in) & (rcyl < g_out), sig * df, sig)

    return sig


def _eval_probe_density(ppar, active, nr=400, nth=160, nph=None):
    """
    Evaluate rho(r, theta, phi) on a probe grid.

    Returns (r_c, th_c, ph_c, rho) where rho has shape (nr, nth, nph).
    rho is proportional to the physical dust density (no absolute normalisation).
    """
    rin   = _ev(ppar['rin'])
    rdisk = _ev(ppar['rdisk'])

    # Probe extends beyond [rin, rdisk] to capture edge transitions cleanly
    r_c  = np.geomspace(0.08 * rin, 1.15 * rdisk, nr)
    th_c = np.linspace(0.01, PI - 0.01, nth)

    if nph is None:
        nph = 60 if _has_phi_structure(active) else 1
    ph_c = (np.linspace(0.0, 2.0 * PI, nph, endpoint=False)
            if nph > 1 else np.array([PI]))

    R    = r_c[:, None, None]
    TH   = th_c[None, :, None]
    PH   = ph_c[None, None, :]

    RCYL = R  * np.sin(TH)
    Z    = R  * np.cos(TH)

    H   = _eval_hp(ppar, RCYL, PH, active)
    SIG = _eval_sigma(ppar, RCYL, PH, active)

    H   = np.maximum(H, 1e-30 * rin)
    SIG = np.maximum(SIG, 0.0)

    z_h = Z / H
    rho = (SIG / H) * np.exp(-0.5 * z_h ** 2)

    return r_c, th_c, ph_c, rho


# ---------------------------------------------------------------------------
# Smoothing and marching algorithms
# ---------------------------------------------------------------------------

def _smooth(arr, n=7):
    """N-point box smoother (scipy if available, fallback to pure numpy)."""
    arr = np.asarray(arr, dtype=float)
    try:
        from scipy.ndimage import uniform_filter1d
        return uniform_filter1d(arr, size=n)
    except ImportError:
        k   = n // 2
        out = arr.copy()
        for i in range(len(arr)):
            lo     = max(0, i - k)
            hi     = min(len(arr), i + k + 1)
            out[i] = arr[lo:hi].mean()
        return out


def _march_tv(x_probe, g, threshold, x_min, x_max, n_max=1500, label=None):
    """
    Place log-spaced interfaces so each cell spans the same *total variation*
    of ln(rho): the integral of g d(ln x) over a cell equals `threshold`.

    A local-step march (x_next = x*(1 + threshold/g(x))) uses only the gradient
    at the current point, so it steps right over a local density *maximum* (a
    vortex or ring peak) where g -> 0 -- leaving one giant cell across the
    feature.  Integrating the gradient instead accumulates both the up- and
    down-slope variation, so a boundary is always placed across a peak.  Empty
    space (g preset to 0 by the caller) contributes no variation and collapses
    to a single coarse cell.

    n_max is a safety backstop: if the segment needs more than n_max cells the
    count is clipped (cells spread evenly in variation, not truncated) and a
    one-line warning is printed so an under-resolved grid is never silent.
    """
    lnx = np.log(x_probe)
    # Cumulative variation:  cv(x) = integral_{x_probe[0]}^{x} g d(ln x)
    cv  = np.concatenate([[0.0], np.cumsum(0.5 * (g[1:] + g[:-1]) * np.diff(lnx))])
    lo, hi       = np.log(float(x_min)), np.log(float(x_max))
    cv_lo, cv_hi = np.interp([lo, hi], lnx, cv)
    total = cv_hi - cv_lo

    n = int(np.ceil(total / threshold)) if total > 0 else 1
    if n > n_max:
        n = n_max
        if label:
            print(f"  WARNING: {label} hit {n_max}-cell cap -- feature "
                  f"under-resolved; raise --threshold or n_max")
    n = max(1, n)

    # Interfaces at equal increments of cumulative variation, inverted back to x
    targets = cv_lo + np.linspace(0.0, total, n + 1)
    lnif    = np.interp(targets, cv, lnx)
    lnif[0], lnif[-1] = lo, hi
    return np.exp(lnif)


def _march_lin(x_probe, g, threshold, x_min, x_max, min_g=None,
               n_max=1000, label=None):
    """
    Build linearly-spaced interfaces by marching from x_min to x_max.
    At each position, step size dx = threshold / g(x).

    min_g floors the gradient (caps the largest cell).  n_max is a safety
    backstop that does NOT bind in normal use: a first pass marches with no
    ceiling, and only if that would place more than n_max cells do we re-march
    with a gradient ceiling g_max = threshold*n_max/span (spreading the budget
    evenly rather than truncating into one giant cell) and warn.
    """
    if min_g is None:
        min_g = threshold / max(float(x_max) - float(x_min), 1e-30) * 3.0
    x_min = float(x_min)
    x_max = float(x_max)
    span  = x_max - x_min

    def run(g_ceiling, stop_at_nmax):
        interfaces = [x_min]
        x = x_min
        while x < x_max * (1.0 - 1e-9):
            gi     = min(max(float(np.interp(x, x_probe, g)), min_g), g_ceiling)
            x_next = x + threshold / gi
            if x_next >= x_max:
                break
            interfaces.append(x_next)
            x = x_next
            if stop_at_nmax and len(interfaces) - 1 >= n_max:
                return None
        interfaces.append(x_max)
        return np.array(interfaces)

    out = run(np.inf, stop_at_nmax=True)
    if out is not None:
        return out
    if label:
        print(f"  WARNING: {label} hit {n_max}-cell cap -- feature "
              f"under-resolved; raise --threshold or n_max")
    g_max = threshold * n_max / span if span > 0 else np.inf
    return run(g_max, stop_at_nmax=False)


# ---------------------------------------------------------------------------
# Interface builders from probe density
# ---------------------------------------------------------------------------

def _build_r_interfaces(r_c, th_c, rho, threshold, rin):
    """
    Build r interfaces from |d ln rho / d ln r| measured at the midplane.
    Using the midplane slice avoids picking up near-zero polar densities that
    would create false huge gradients at rin/rdisk where sigma transitions.

    The march is split at rin so the cavity (empty, one coarse cell) and the
    disk are handled separately.

    Gradient floor handling: only genuinely-empty regions (cavity, vacuum gaps
    with drfact=0, beyond rdisk) are excluded from refinement -- detected as
    density below 1e-30 of the peak, NOT a fixed fraction of the global max.
    An earlier 1e-4-of-peak floor flattened the entire faint outer disk (which
    is far below 1e-4 of the bright inner rim), so outer-disk structure and the
    radial power-law were never resolved.  Cells adjacent to empty space (the
    hard sigma cutoffs at rin/rdisk) have their gradient capped so the
    artificial cliff gets a few cells rather than hundreds.
    """
    i_mid = np.argmin(np.abs(th_c - PI / 2.0))
    dn    = max(1, len(th_c) // 20)
    i_lo  = max(0, i_mid - dn)
    i_hi  = min(len(th_c), i_mid + dn + 1)
    rho_mid = rho[:, i_lo:i_hi, :].max(axis=(1, 2))
    peak    = float(rho_mid.max())
    tiny    = peak * 1e-30 if peak > 0 else 1e-100
    empty   = rho_mid <= tiny

    g_r = np.abs(np.gradient(np.log(np.maximum(rho_mid, tiny)), np.log(r_c)))
    # Cells touching empty space are the artificial rin/rdisk (and drfact=0)
    # cliffs; cap their gradient so they don't consume the whole cell budget.
    edge = np.zeros_like(empty)
    edge[1:]  |= empty[:-1]
    edge[:-1] |= empty[1:]
    edge      &= ~empty
    g_r = np.where(empty, 0.0, g_r)
    g_r = np.where(edge,  np.minimum(g_r, _EDGE_GCAP), g_r)
    g_r = _smooth(g_r, n=3)

    # Cavity: empty -> collapses to one coarse cell.  Disk: resolves the real
    # density variation (power-law, gaps, vortices) to `threshold` per cell.
    r_if_cav  = _march_tv(r_c, g_r, threshold, r_c[0], rin,     n_max=1500,
                          label='r-march (cavity)')
    r_if_disk = _march_tv(r_c, g_r, threshold, rin,    r_c[-1], n_max=1500,
                          label='r-march (disk)')
    return np.concatenate([r_if_cav, r_if_disk[1:]])


def _build_th_interfaces(r_c, th_c, rho, threshold, hrdisk=0.1):
    """
    Build theta interfaces from the maximum |d ln rho / d theta| over all (r, phi).
    The grid is symmetric around pi/2 (exact midplane boundary).
    Uses a per-r cap of 1% of the local midplane density so we only resolve to
    ~3 scale heights at each radius -- this prevents raised-H features (inner
    edge shadow, puffed rim) from extending the resolved theta region to many
    scale heights above the midplane.
    """
    # Cap at 1% of local (r, phi) midplane density so we resolve to ~3 scale
    # heights everywhere.  This prevents raised-H features (shadow, puffed rim)
    # from extending the resolved region deep into the polar atmosphere.
    # Floor: always cap at least to 1e-4 of the global max so the disk-edge
    # theta transition (rcyl crossing rin/rdisk as theta changes) is never
    # resolved with a floor of 1e-100.
    i_mid = np.argmin(np.abs(th_c - PI / 2.0))
    rho_mid_rph = rho[:, i_mid, :]                           # (nr, nph) midplane slice
    rho_cap_rph = np.maximum(rho_mid_rph * 1e-2,
                             float(rho.max()) * 1e-4)        # (nr, nph)
    rho_cap_rph = rho_cap_rph[:, None, :]                    # (nr, 1, nph) broadcast

    log_rho = np.log(np.maximum(rho, rho_cap_rph))
    g_th_3d = np.abs(np.gradient(log_rho, th_c, axis=1))

    # Worst-case requirement over all r and phi
    g_th = g_th_3d.max(axis=(0, 2))
    # Physics-based floor: at least 4 cells per disk scale height near the midplane.
    # hrdisk is H/r at the pivot radius, so arctan(hrdisk) ~ hrdisk is the
    # scale height angle. This constraint is threshold-independent so the disk
    # surface is always resolved regardless of the chosen gradient threshold.
    dth_scale = float(hrdisk) / 4.0
    # Also cap at 0.10 rad (~6 deg) as a hard geometric maximum.
    dth_max   = min(dth_scale, 0.10)
    min_g_th  = threshold / dth_max
    g_th = np.maximum(_smooth(g_th, n=7), min_g_th)

    # Build upper hemisphere only (theta from th_c[0] toward pi/2)
    mask   = th_c <= PI / 2.0 + 1e-6
    th_up  = th_c[mask]
    g_up   = g_th[:len(th_up)]

    th_if_up = _march_lin(th_up, g_up, threshold,
                          x_min=th_c[0], x_max=PI / 2.0,
                          min_g=min_g_th, n_max=800, label='theta-march')
    th_if_up[-1] = PI / 2.0   # exact midplane -- required by radmc3dPy

    # Mirror for lower hemisphere (skip duplicate pi/2)
    th_if_lo = PI - th_if_up[::-1][1:]
    return np.concatenate([th_if_up, th_if_lo])


def _build_ph_interfaces(r_c, th_c, ph_c, rho, threshold, active):
    """
    Build phi interfaces from |d ln rho / d phi| measured at the midplane.
    Using the midplane region avoids large-z disk atmosphere gradients (e.g.
    from the inner edge shadow raising H in the shadow direction, which creates
    a huge phi contrast at several scale heights even though physically
    irrelevant for the grid).  Returns [0, 2pi] for axisymmetric models.
    """
    if not _has_phi_structure(active) or len(ph_c) <= 1:
        return np.array([0.0, 2.0 * PI])

    i_mid = np.argmin(np.abs(th_c - PI / 2.0))
    dn    = max(1, len(th_c) // 10)           # ~10% of probe cells around midplane
    i_lo  = max(0, i_mid - dn)
    i_hi  = min(len(th_c), i_mid + dn + 1)
    rho_mid = rho[:, i_lo:i_hi, :]            # (nr, n_mid, nph)

    rho_cap = max(float(rho_mid.max()) * 1e-2, 1e-100)
    log_rho = np.log(np.maximum(rho_mid, rho_cap))
    g_ph_3d = np.abs(np.gradient(log_rho, ph_c, axis=2))

    g_ph = g_ph_3d.max(axis=(0, 1))
    g_ph = np.maximum(_smooth(g_ph, n=3), threshold / (PI / 6.0))

    ph_if      = _march_lin(ph_c, g_ph, threshold,
                            x_min=0.0, x_max=2.0 * PI,
                            min_g=threshold / (PI / 6.0),
                            n_max=1080, label='phi-march')
    ph_if[-1]  = 2.0 * PI
    return ph_if


# ---------------------------------------------------------------------------
# Convert interface arrays to (bounds, counts) format for radmc3dPy
# ---------------------------------------------------------------------------

def _to_segments(interfaces):
    """
    Convert an interface array [x0, x1, ..., xN] to (bounds, counts) where
    bounds = [x0,...,xN] and counts = [1,...,1, 2] (N-1 elements).

    This produces N-1 cells with boundaries at the provided positions.
    The last segment has count=2 so _interfaces() includes both endpoints.
    """
    N      = len(interfaces)
    bounds = [float(v) for v in interfaces]
    if N == 2:
        return bounds, [1]
    counts = [1] * (N - 2) + [2]
    return bounds, counts


def _to_segments_r(interfaces):
    """Like _to_segments but expresses r-bounds as '<value>*au' strings.

    This makes the returned xbound human-readable in the same format the user
    writes manually in the params file.  _interfaces() evaluates these strings
    via _ev(), so all downstream code is transparent to the change.
    """
    N      = len(interfaces)
    bounds = [f'{float(v) / AU:.6g}*au' for v in interfaces]
    # 6 sig figs is ample for any realistic grid, but guard against two
    # adjacent interfaces ever rounding to the same string (which would make a
    # zero-width cell): if the round-tripped values are not strictly
    # increasing, fall back to full precision.
    vals = [float(_ev(b)) for b in bounds]
    if any(hi <= lo for lo, hi in zip(vals[:-1], vals[1:])):
        bounds = [f'{float(v) / AU:.12g}*au' for v in interfaces]
    if N == 2:
        return bounds, [1]
    counts = [1] * (N - 2) + [2]
    return bounds, counts


def _to_segments_th(th_if):
    """
    Like _to_segments but replaces the float closest to pi/2 with the
    string 'pi/2.' -- required by radmc3dPy's reggrid.py boundary check.
    """
    bounds_raw, counts = _to_segments(th_if)
    bounds = []
    for v in bounds_raw:
        if abs(v - PI / 2.0) < 1e-9:
            bounds.append("pi/2.")
        else:
            bounds.append(float(v))
    return bounds, counts


# ---------------------------------------------------------------------------
# Cell interface arrays (used by write_amr_grid, summary, and diagnostic plot)
# ---------------------------------------------------------------------------

def _interfaces(bounds, counts, log_spaced):
    pts    = []
    n_segs = len(counts)
    bounds = [float(_ev(b)) if isinstance(b, str) else float(b) for b in bounds]
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
    B = 8
    return ((5 + 2 * n_spec + 2 * n_spec + 4) * ncell * B) / 1024 ** 3


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def _print_summary(xbound, nx, ybound, ny, zbound, nz,
                   active, orig_ppar, threshold, rim_cells=0,
                   nphot_scat_rec=None, nphot_scat_cfg=None):
    nr,  xi = _nr_from_grid(xbound, nx)
    nth, yi = _nth_from_grid(ybound, ny)
    nph, zi = _nph_from_grid(zbound, nz)
    total   = nr * nth * nph

    orig_nr  = sum(orig_ppar['nx']) if isinstance(orig_ppar.get('nx'), list) else int(orig_ppar.get('nx', 0))
    orig_nth = sum(orig_ppar['ny']) if isinstance(orig_ppar.get('ny'), list) else int(orig_ppar.get('ny', 0))
    orig_nph = sum(orig_ppar['nz']) if isinstance(orig_ppar.get('nz'), list) else int(orig_ppar.get('nz', 0))
    orig_total = orig_nr * orig_nth * orig_nph

    ram_new  = _estimate_ram_gib(total)
    ram_orig = _estimate_ram_gib(orig_total) if orig_total else 0.0

    W = 72
    print('\n' + '-' * W)
    print('  Smart Grid Builder -- Summary  (density-gradient adaptive)')
    print('-' * W)
    print(f'  Threshold          : {threshold:.3g}  '
          f'(max |delta ln rho| per cell = {threshold*100:.1f}%)')
    if rim_cells > 0:
        hrdisk  = float(_ev(orig_ppar.get('hrdisk',  0.1)))
        hrpivot = float(_ev(orig_ppar.get('hrpivot', 100.0 * AU)))
        plh     = float(_ev(orig_ppar.get('plh',     0.0)))
        rin_v   = _ev(orig_ppar['rin'])
        H_rin   = hrdisk * (rin_v / hrpivot) ** plh * rin_v
        print(f'  Rim refinement     : {rim_cells} cells in '
              f'[rin, rin + H_rin]  (H_rin = {H_rin/AU:.4f} au, '
              f'analytical gradient resolution)')

    on  = [k for k, v in active.items() if v]
    off = [k for k, v in active.items() if not v]
    print(f'  Active structures  : {", ".join(on) if on else "none"}')
    print(f'  Inactive (skipped) : {", ".join(off) if off else "none"}')
    print()

    rin   = _ev(orig_ppar['rin'])
    rdisk = _ev(orig_ppar['rdisk'])
    dr_r  = np.diff(xi) / xi[:-1]
    print(f'  r-grid  {nr} cells')
    print(f'    dr/r range: {dr_r.min()*100:.2f}% - {dr_r.max()*100:.1f}%  '
          f'(rin={rin/AU:.2f} au, rdisk={rdisk/AU:.0f} au)')
    print()

    ybound_f = [float(_ev(b)) if isinstance(b, str) else float(b) for b in ybound]
    dth = np.diff(ybound_f)
    th_mid_cell = dth[len(dth) // 2]
    print(f'  th-grid {nth} cells')
    print(f'    dtheta range: {np.degrees(dth.min()):.3f} - {np.degrees(dth.max()):.2f} deg  '
          f'(near midplane: {np.degrees(th_mid_cell):.3f} deg)')
    print()

    if nph == 1:
        print(f'  phi-grid 1 cell  (axisymmetric -- 2D mode, no spoke artifacts)')
    else:
        dph = np.diff(zi)
        print(f'  phi-grid {nph} cells')
        print(f'    dphi range: {np.degrees(dph.min()):.2f} - {np.degrees(dph.max()):.2f} deg')
    print()

    print(f'  {"TOTAL CELLS":20s}  {total:>12,d}')
    if orig_total:
        reduction = orig_total / total if total else 0
        print(f'  {"Config grid":20s}  {orig_total:>12,d}  ({orig_nr} x {orig_nth} x {orig_nph})')
        print(f'  {"Reduction":20s}  {reduction:>11.1f}x')
    print()
    print(f'  RAM lower bound (new)    : {ram_new:.2f} GiB')
    if orig_total:
        print(f'  RAM lower bound (config) : {ram_orig:.2f} GiB')

    if nphot_scat_rec is not None:
        print()
        cfg_str = f'{int(nphot_scat_cfg):.2e}' if nphot_scat_cfg else '?'
        status  = 'OK' if (nphot_scat_cfg and nphot_scat_cfg >= nphot_scat_rec) else 'TOO LOW'
        print(f'  nphot_scat (config)      : {cfg_str}  [{status}]')
        print(f'  nphot_scat (recommended) : {nphot_scat_rec:.2e}  (10 photons/cell)')
        if nphot_scat_cfg and nphot_scat_cfg < nphot_scat_rec:
            print(f'  -> pipeline will use {nphot_scat_rec:.2e} to avoid MC noise')

    if total > 50_000_000:
        print(f'\n  WARNING: Grid exceeds 50 M cells -- consider raising threshold')

    print('-' * W + '\n')


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def build_smart_grid(ppar, threshold=0.1, rim_cells=30, verbose=True):
    """
    Compute an optimal RADMC-3D spherical grid from a pipeline params dict.

    Evaluates the dust density model on a coarse probe grid, measures the
    actual |delta ln rho| gradient in every direction, and places cell
    interfaces so each cell satisfies |delta ln rho| <= threshold.

    Parameters
    ----------
    ppar      : dict   -- pipeline config params (string values like '0.5*au' are ok).
    threshold : float  -- max |delta ln rho| per cell (default 0.1 = 10%).
    rim_cells : int    -- cells across the first scale height above rin,
                          computed analytically from H_rin because the probe
                          gradient is unreliable at the density step there
                          (default 30). Set to 0 to disable.
    verbose   : bool   -- print the summary table.

    Returns
    -------
    dict with keys xbound, nx, ybound, ny, zbound, nz, nphot_scat_rec.
    """
    active = _detect_active(ppar)

    # Evaluate the density model on a probe grid
    r_c, th_c, ph_c, rho = _eval_probe_density(ppar, active)

    # Build interface arrays from measured density gradients
    rin    = _ev(ppar['rin'])
    hrdisk = float(_ev(ppar.get('hrdisk', 0.1)))
    r_if  = _build_r_interfaces(r_c, th_c, rho, threshold, rin)
    prim_rout = float(_ev(ppar.get('prim_rout', 1.0)))
    r_if  = _guarantee_rim_cells(r_if, rin, hrdisk=hrdisk, prim_rout=prim_rout, n_min=10)
    r_if  = _refine_inner_wall(r_if, rin, ppar, rim_cells)
    th_if = _build_th_interfaces(r_c, th_c, rho, threshold, hrdisk)
    ph_if = _build_ph_interfaces(r_c, th_c, ph_c, rho, threshold, active)

    # Convert to (bounds, counts) format for radmc3dPy / problemSetupDust
    xbound, nx = _to_segments_r(r_if)
    ybound, ny = _to_segments_th(th_if)

    if len(ph_if) == 2:                    # axisymmetric: radmc3dPy needs nz=[0] to
        zbound, nz = [0.0, float(2.0 * PI)], [0]  # write act_dim=0 (phi off) in amr_grid.inp
    else:
        zbound, nz = _to_segments(ph_if)

    # Recommended nphot_scat: ~10 MC photon hits per cell
    nr  = len(r_if) - 1
    nth = len(th_if) - 1
    nph = len(ph_if) - 1
    total_cells    = nr * nth * nph
    exp            = int(np.ceil(np.log10(max(10 * total_cells, 1))))
    nphot_scat_rec = int(10 ** exp)

    if verbose:
        _print_summary(xbound, nx, ybound, ny, zbound, nz,
                       active, ppar, threshold, rim_cells=rim_cells,
                       nphot_scat_rec=nphot_scat_rec,
                       nphot_scat_cfg=_ev(ppar.get('nphot_scat', 0)))

    return {
        'xbound':         xbound,
        'nx':             nx,
        'ybound':         ybound,
        'ny':             ny,
        'zbound':         zbound,
        'nz':             nz,
        'nphot_scat_rec': nphot_scat_rec,
    }


# ---------------------------------------------------------------------------
# Standalone: write amr_grid.inp directly
# ---------------------------------------------------------------------------

def write_amr_grid(xbound, nx, ybound, ny, zbound, nz, fname='amr_grid.inp'):
    nr,  xi = _nr_from_grid(xbound, nx)
    nth, yi = _nth_from_grid(ybound, ny)
    # nz=[0] means axisymmetric (phi inactive) -- act_dim[2]=0, no phi interfaces
    axisym = (isinstance(nz, list) and len(nz) == 1 and nz[0] == 0)
    if axisym:
        nph = 1
        zi  = np.array([0.0, 2.0 * PI])
    else:
        nph, zi = _nph_from_grid(zbound, nz)
    act_phi = 0 if axisym else 1

    print(f'Writing {fname}  ({nr} x {nth} x {nph} = {nr*nth*nph:,} cells, phi_active={act_phi})')
    with open(fname, 'w') as f:
        f.write('1\n')
        f.write('0\n')
        f.write('100\n')
        f.write('0\n')
        f.write(f'1 1 {act_phi}\n')
        f.write(f'{nr} {nth} {nph}\n')
        for v in xi:
            f.write('%.9e\n' % v)
        for v in yi:
            f.write('%.9e\n' % v)
        for v in zi:
            f.write('%.9e\n' % v)


# ---------------------------------------------------------------------------
# Diagnostic plot
# ---------------------------------------------------------------------------

def _make_diagnostic_plot(xbound, nx, ybound, ny, ppar, active, outfile=None):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib not available -- skipping diagnostic plot')
        return

    nr, xi = _nr_from_grid(xbound, nx)
    nth, yi = _nth_from_grid(ybound, ny)
    rin   = _ev(ppar['rin'])
    rdisk = _ev(ppar['rdisk'])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    r_c  = 0.5 * (xi[:-1] + xi[1:])
    dr_r = np.diff(xi) / r_c
    ax.semilogy(r_c / AU, dr_r * 100, color='steelblue', linewidth=1.2)
    ax.axvline(rin   / AU, color='orange', lw=1.0, ls='--', label='rin')
    ax.axvline(rdisk / AU, color='red',    lw=1.0, ls='--', label='rdisk')
    if active.get('inner_rim'):
        ax.axvline(_ev(ppar['prim_rout']) * rin / AU,
                   color='purple', lw=0.8, ls=':', label='prim_rout')
    if active.get('srim'):
        ax.axvline(_ev(ppar['srim_rout']) * rin / AU,
                   color='teal', lw=0.8, ls=':', label='srim_rout')
    ax.set_xlabel('r (au)')
    ax.set_ylabel('dr/r (%)')
    ax.set_title(f'r-grid  ({nr} cells, density-adaptive)')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    th_c = 0.5 * (yi[:-1] + yi[1:])
    dth  = np.diff(yi)
    ax.semilogy(np.degrees(th_c), np.degrees(dth), color='darkorange', linewidth=1.2)
    ax.axvline(90, color='gray', lw=0.8, ls='--', label='midplane')
    hpr = _ev(ppar.get('hpr_prim_rout', 0.0))
    if hpr > 0 and active.get('inner_rim'):
        ax.axvline(90 - np.degrees(hpr), color='purple', lw=0.8, ls=':',
                   label=f'shadow angle ({hpr:.3g} rad)')
    ax.set_xlabel('theta (deg)')
    ax.set_ylabel('dtheta (deg)')
    ax.set_title(f'theta-grid  ({nth} cells, density-adaptive)')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    fig.suptitle('Smart Grid -- density-adaptive cell spacing', fontsize=11)
    fig.tight_layout()
    if outfile:
        fig.savefig(outfile, dpi=150, bbox_inches='tight')
        print(f'  Diagnostic plot saved to {outfile}')
    else:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Spatial grid visualization
# ---------------------------------------------------------------------------

def _make_spatial_plot(xbound, nx, ybound, ny, ppar, active, outfile=None):
    """
    2D spatial grid visualization in (R, z) coordinates.
    Left panel : full disk — density colormap + cell boundaries.
    Right panel : inner rim zoom — shows the analytical rim refinement.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm
    except ImportError:
        print('matplotlib not available -- skipping spatial plot')
        return

    nr, xi = _nr_from_grid(xbound, nx)
    nth, yi = _nth_from_grid(ybound, ny)

    rin       = _ev(ppar['rin'])
    rdisk     = _ev(ppar['rdisk'])
    hrdisk    = float(_ev(ppar.get('hrdisk',    0.1)))
    hrpivot   = float(_ev(ppar.get('hrpivot',   100.0 * AU)))
    plh       = float(_ev(ppar.get('plh',       0.0)))
    prim_rout = float(_ev(ppar.get('prim_rout', 1.0)))
    H_rin     = hrdisk * (rin / hrpivot) ** plh * rin

    # Interface corner coordinates (nr+1, nth+1) in au
    R_if = xi[:, None] * np.sin(yi[None, :]) / AU
    Z_if = xi[:, None] * np.cos(yi[None, :]) / AU

    # Cell-centre density (nr, nth) — phi=0, valid for axisymmetric models
    r_c  = 0.5 * (xi[:-1] + xi[1:])
    th_c = 0.5 * (yi[:-1] + yi[1:])
    R_c  = r_c[:, None] * np.sin(th_c[None, :])
    Z_c  = r_c[:, None] * np.cos(th_c[None, :])
    PH_c = np.zeros_like(R_c)

    H_c   = _eval_hp(ppar, R_c, PH_c, active)
    SIG_c = _eval_sigma(ppar, R_c, PH_c, active)
    H_c   = np.maximum(H_c, 1e-30 * rin)
    rho_c = np.maximum(SIG_c, 0.0) / H_c * np.exp(-0.5 * (Z_c / H_c) ** 2)

    rho_max = float(rho_c.max()) if rho_c.max() > 0 else 1.0
    rho_c   = np.maximum(rho_c, rho_max * 1e-8)
    norm    = LogNorm(vmin=rho_max * 1e-8, vmax=rho_max)

    rin_au   = rin  / AU
    rdisk_au = rdisk / AU
    H_rin_au = H_rin / AU

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), facecolor='#0a0a0a')

    panels = [
        dict(r_hi   = rdisk_au * 1.05,
             z_hi   = rdisk_au * 0.35,
             step_r = max(1, nr // 60),
             step_th= max(1, nth // 40),
             title  = f'Full disk  ({nr} r × {nth} θ = {nr*nth:,} cells)',
             zoom   = False),
        dict(r_hi   = max(prim_rout, 2.5) * rin_au * 1.4,
             z_hi   = max(prim_rout, 2.5) * rin_au * 0.55,
             step_r = 1,                       # draw every r-interface in zoom
             step_th= max(1, nth // 60),
             title  = (f'Inner rim  (rin = {rin_au:.2f} au,  '
                       f'H_rin = {H_rin_au:.4f} au)'),
             zoom   = True),
    ]

    for ax, p in zip(axes, panels):
        ax.set_facecolor('black')

        # Density as pcolormesh on the actual curvilinear (r, theta) → (R, z) grid
        ax.pcolormesh(R_if, Z_if, rho_c,
                      norm=norm, cmap='inferno',
                      shading='flat', rasterized=True)

        # r-interface arcs — one polyline per radial shell
        for i in range(0, nr + 1, p['step_r']):
            if xi[i] / AU > p['r_hi'] * 1.02:
                continue
            ax.plot(R_if[i, :], Z_if[i, :], '-',
                    color='white', lw=0.15, alpha=0.4)

        # theta-interface rays — one polyline per meridional boundary
        r_mask = xi / AU <= p['r_hi'] * 1.02
        for j in range(0, nth + 1, p['step_th']):
            ax.plot(R_if[r_mask, j], Z_if[r_mask, j], '-',
                    color='white', lw=0.15, alpha=0.4)

        # Reference lines
        ax.axhline(0, color='#777', lw=0.5, ls='--', alpha=0.5)
        ax.axvline(rin_au, color='cyan', lw=0.9, ls='--', alpha=0.85,
                   label=f'rin = {rin_au:.2f} au')
        if p['zoom']:
            ax.axvline(prim_rout * rin_au, color='orange', lw=0.9, ls=':',
                       alpha=0.85,
                       label=f'prim_rout×rin = {prim_rout * rin_au:.2f} au')
            ax.axvline(rin_au + H_rin_au, color='lime', lw=0.9, ls=':',
                       alpha=0.85,
                       label=f'rin + H_rin = {rin_au + H_rin_au:.3f} au')

        ax.set_xlim(0, p['r_hi'])
        ax.set_ylim(-p['z_hi'], p['z_hi'])
        ax.set_xlabel('R  (au)', color='white', fontsize=11)
        ax.set_ylabel('z  (au)', color='white', fontsize=11)
        ax.set_title(p['title'], color='white', fontsize=10, pad=8)
        ax.tick_params(colors='white', which='both')
        for spine in ax.spines.values():
            spine.set_edgecolor('#444')
        ax.legend(fontsize=8, loc='upper right',
                  facecolor='#1a1a1a', labelcolor='white', edgecolor='#555')

    fig.suptitle('RADMC-3D Grid — density-adaptive cell spacing',
                 color='white', fontsize=13)
    fig.tight_layout(rect=[0, 0, 0.93, 1])

    cbar_ax = fig.add_axes([0.95, 0.12, 0.015, 0.75])
    sm = plt.cm.ScalarMappable(cmap='inferno', norm=norm)
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('log₁₀ ρ  (rel.)', color='white', fontsize=10)
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')
    cbar.outline.set_edgecolor('#444')

    if outfile:
        fig.savefig(outfile, dpi=150, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        print(f'  Spatial plot saved to {outfile}')
    else:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Grid analysis plot
# ---------------------------------------------------------------------------

def _make_analysis_plot(xbound, nx, ybound, ny, ppar, active,
                        threshold=0.1, outfile=None):
    """
    Third diagnostic figure — two panels in (R, z):
      Left  : max |Δ ln ρ| per cell  — shows where the threshold is met/exceeded.
      Right : cell aspect ratio log₁₀(r·dθ / dr) — shows elongated cells that
              can cause ray-tracing artefacts even when the gradient is fine.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize, TwoSlopeNorm
    except ImportError:
        print('matplotlib not available -- skipping analysis plot')
        return

    nr, xi = _nr_from_grid(xbound, nx)
    nth, yi = _nth_from_grid(ybound, ny)

    rin   = _ev(ppar['rin'])
    rdisk = _ev(ppar['rdisk'])

    # Corner coordinates  (nr+1, nth+1) in au
    R_if = xi[:, None] * np.sin(yi[None, :]) / AU
    Z_if = xi[:, None] * np.cos(yi[None, :]) / AU

    # Cell-centre density  (nr, nth)
    r_c  = 0.5 * (xi[:-1] + xi[1:])
    th_c = 0.5 * (yi[:-1] + yi[1:])
    R_c  = r_c[:, None] * np.sin(th_c[None, :])
    Z_c  = r_c[:, None] * np.cos(th_c[None, :])
    PH_c = np.zeros_like(R_c)

    H_c   = _eval_hp(ppar, R_c, PH_c, active)
    SIG_c = _eval_sigma(ppar, R_c, PH_c, active)
    H_c   = np.maximum(H_c, 1e-30 * rin)
    rho_c = np.maximum(SIG_c, 0.0) / H_c * np.exp(-0.5 * (Z_c / H_c) ** 2)
    rho_max = float(rho_c.max()) if rho_c.max() > 0 else 1.0
    rho_c   = np.maximum(rho_c, rho_max * 1e-8)
    log_rho = np.log(rho_c)

    # ── |Δ ln ρ| per cell (forward difference in each direction) ─────────────
    grad_r  = np.zeros((nr, nth))
    grad_th = np.zeros((nr, nth))
    grad_r[:-1, :]  = np.abs(np.diff(log_rho, axis=0))
    grad_r[-1,  :]  = grad_r[-2, :]
    grad_th[:, :-1] = np.abs(np.diff(log_rho, axis=1))
    grad_th[:, -1]  = grad_th[:, -2]
    delta_max = np.maximum(grad_r, grad_th)

    # ── Cell aspect ratio  log₁₀(r·dθ / dr) ─────────────────────────────────
    dr_cell  = np.diff(xi)    # (nr,)
    dth_cell = np.diff(yi)    # (nth,)
    ds_th    = r_c[:, None] * dth_cell[None, :]   # (nr, nth) arc length in θ
    ds_r     = dr_cell[:, None] * np.ones((1, nth))
    log_asp  = np.log10(np.maximum(ds_th / ds_r, 1e-9))

    rin_au   = rin  / AU
    rdisk_au = rdisk / AU
    r_hi     = rdisk_au * 1.05
    z_hi     = rdisk_au * 0.35

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), facecolor='#0a0a0a')

    # ── Panel 1: gradient satisfaction ───────────────────────────────────────
    ax = axes[0]
    ax.set_facecolor('black')

    norm_g = TwoSlopeNorm(vmin=0, vcenter=threshold, vmax=threshold * 8)
    im1    = ax.pcolormesh(R_if, Z_if, delta_max,
                           norm=norm_g, cmap='RdYlGn_r',
                           shading='flat', rasterized=True)
    # White contour exactly at the threshold
    ax.contour(R_c / AU, Z_c / AU, delta_max,
               levels=[threshold], colors='white',
               linewidths=1.0, linestyles='--', alpha=0.9)

    ax.axhline(0, color='#777', lw=0.5, ls='--', alpha=0.5)
    ax.axvline(rin_au, color='cyan', lw=0.9, ls='--', alpha=0.75,
               label=f'rin = {rin_au:.2f} au')
    ax.set_xlim(0, r_hi)
    ax.set_ylim(-z_hi, z_hi)
    ax.set_xlabel('R  (au)', color='white', fontsize=11)
    ax.set_ylabel('z  (au)', color='white', fontsize=11)
    ax.set_title(f'Max |Δ ln ρ| per cell  —  white dashed = threshold ({threshold})',
                 color='white', fontsize=10, pad=8)
    ax.tick_params(colors='white')
    for sp in ax.spines.values():
        sp.set_edgecolor('#444')
    ax.legend(fontsize=8, facecolor='#1a1a1a', labelcolor='white', edgecolor='#555')

    cb1 = fig.colorbar(im1, ax=ax, fraction=0.046, pad=0.04)
    cb1.set_label('|Δ ln ρ|', color='white', fontsize=10)
    cb1.ax.yaxis.set_tick_params(color='white')
    plt.setp(cb1.ax.yaxis.get_ticklabels(), color='white')
    cb1.outline.set_edgecolor('#444')
    cb1.ax.axhline(y=threshold, color='white', lw=1.2, ls='--')

    # ── Panel 2: cell aspect ratio ────────────────────────────────────────────
    ax = axes[1]
    ax.set_facecolor('black')

    asp_lim = min(max(abs(float(log_asp.max())), abs(float(log_asp.min()))), 3.0)
    im2 = ax.pcolormesh(R_if, Z_if, log_asp,
                        vmin=-asp_lim, vmax=asp_lim, cmap='RdBu',
                        shading='flat', rasterized=True)
    # White contour at aspect ratio = 1 (isotropic)
    ax.contour(R_c / AU, Z_c / AU, log_asp,
               levels=[0.0], colors='white',
               linewidths=1.0, linestyles='-', alpha=0.9)

    ax.axhline(0, color='#777', lw=0.5, ls='--', alpha=0.5)
    ax.axvline(rin_au, color='cyan', lw=0.9, ls='--', alpha=0.75,
               label=f'rin = {rin_au:.2f} au')
    ax.set_xlim(0, r_hi)
    ax.set_ylim(-z_hi, z_hi)
    ax.set_xlabel('R  (au)', color='white', fontsize=11)
    ax.set_ylabel('z  (au)', color='white', fontsize=11)
    ax.set_title('Cell aspect ratio  log₁₀(r·dθ / dr)  —  white = isotropic',
                 color='white', fontsize=10, pad=8)
    ax.tick_params(colors='white')
    for sp in ax.spines.values():
        sp.set_edgecolor('#444')
    ax.legend(fontsize=8, facecolor='#1a1a1a', labelcolor='white', edgecolor='#555')

    cb2 = fig.colorbar(im2, ax=ax, fraction=0.046, pad=0.04)
    cb2.set_label('log₁₀(r·dθ / dr)', color='white', fontsize=10)
    cb2.ax.yaxis.set_tick_params(color='white')
    plt.setp(cb2.ax.yaxis.get_ticklabels(), color='white')
    cb2.outline.set_edgecolor('#444')
    cb2.ax.axhline(y=0.0, color='white', lw=1.2, ls='-')

    fig.suptitle(f'RADMC-3D Grid Analysis  (threshold = {threshold})',
                 color='white', fontsize=13)
    fig.tight_layout()

    if outfile:
        fig.savefig(outfile, dpi=150, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        print(f'  Analysis plot saved to {outfile}')
    else:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _load_config_as_dict(config_path):
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
        description='Density-adaptive grid builder for RADMC-3D disk simulations.')
    parser.add_argument('--config', default=None,
                        help='Path to a pipeline config .py file (default: Pipeline/config.py)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show summary only -- do not write amr_grid.inp')
    parser.add_argument('--plot', action='store_true',
                        help='Save a diagnostic plot grid_builder_diag.png')
    parser.add_argument('--output', default='amr_grid.inp',
                        help='Output file name (default: amr_grid.inp)')
    parser.add_argument('--threshold', type=float, default=0.1,
                        help='Max |delta ln rho| per cell (default: 0.1 = 10%%). '
                             'Smaller -> finer grid. Typical range: 0.03-0.3.')
    parser.add_argument('--rim-cells', type=int, default=30,
                        help='Cells forced across the first scale height above rin, '
                             'bypassing the probe gradient (default: 30). Set 0 to disable.')
    args = parser.parse_args()

    if args.config:
        cfg_path = args.config
    else:
        here     = os.path.dirname(os.path.abspath(__file__))
        cfg_path = os.path.join(here, 'config.py')

    if not os.path.exists(cfg_path):
        print(f'Error: config file not found: {cfg_path}')
        sys.exit(1)

    print(f'Loading config: {cfg_path}')
    ppar   = _load_config_as_dict(cfg_path)
    result = build_smart_grid(ppar, threshold=args.threshold,
                              rim_cells=args.rim_cells, verbose=True)

    if args.plot:
        _make_diagnostic_plot(
            result['xbound'], result['nx'],
            result['ybound'], result['ny'],
            ppar, _detect_active(ppar),
            outfile='grid_builder_diag.png',
        )
        _make_spatial_plot(
            result['xbound'], result['nx'],
            result['ybound'], result['ny'],
            ppar, _detect_active(ppar),
            outfile='grid_builder_spatial.png',
        )
        _make_analysis_plot(
            result['xbound'], result['nx'],
            result['ybound'], result['ny'],
            ppar, _detect_active(ppar),
            threshold=args.threshold,
            outfile='grid_builder_analysis.png',
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
