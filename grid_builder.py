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


def _n_log(r_lo, r_hi, slope, threshold, n_min=5, n_max=500):
    step = threshold / max(abs(slope), 0.05)
    return max(n_min, min(n_max, int(np.ceil(np.log(r_hi / r_lo) / step))))


def _n_lin(lo, hi, step, n_min=3, n_max=300):
    return max(n_min, min(n_max, int(np.ceil((hi - lo) / step))))


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
        'gaps':           any(abs(df) > 0 and abs(df) < 1.0
                              for df in _ev_list(ppar.get('gap_drfact', [0.0]))),
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
                H_prim = hpr_prim * (rcyl / rin) ** dummy * rcyl
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

    # Gap depletion
    if active.get('gaps', False):
        grins  = _ev_list(ppar.get('gap_rin',    [0.0]))
        grouts = _ev_list(ppar.get('gap_rout',   [0.0]))
        gdfs   = _ev_list(ppar.get('gap_drfact', [1.0]))
        for g_in, g_out, df in zip(grins, grouts, gdfs):
            if abs(df) > 0 and abs(df) < 1.0 and g_out > g_in:
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


def _march_log(x_probe, g, threshold, x_min, x_max, min_g=0.02):
    """
    Build log-spaced interfaces by marching from x_min to x_max.
    At each position, step size dr/r = threshold / g(r).
    """
    interfaces = [float(x_min)]
    x = float(x_min)
    x_max = float(x_max)
    while x < x_max * (1.0 - 1e-9):
        gi     = max(float(np.interp(x, x_probe, g)), min_g)
        x_next = x * (1.0 + threshold / gi)
        if x_next >= x_max:
            break
        interfaces.append(x_next)
        x = x_next
    interfaces.append(x_max)
    return np.array(interfaces)


def _march_lin(x_probe, g, threshold, x_min, x_max, min_g=None):
    """
    Build linearly-spaced interfaces by marching from x_min to x_max.
    At each position, step size dx = threshold / g(x).
    """
    if min_g is None:
        min_g = threshold / max(float(x_max) - float(x_min), 1e-30) * 3.0
    interfaces = [float(x_min)]
    x = float(x_min)
    x_max = float(x_max)
    while x < x_max * (1.0 - 1e-9):
        gi     = max(float(np.interp(x, x_probe, g)), min_g)
        x_next = x + threshold / gi
        if x_next >= x_max:
            break
        interfaces.append(x_next)
        x = x_next
    interfaces.append(x_max)
    return np.array(interfaces)


# ---------------------------------------------------------------------------
# Interface builders from probe density
# ---------------------------------------------------------------------------

def _build_r_interfaces(r_c, th_c, rho, threshold):
    """
    Build r interfaces from |d ln rho / d ln r| measured at the midplane.
    Using the midplane slice avoids picking up near-zero polar densities that
    would create false huge gradients at rin/rdisk where sigma transitions.
    """
    i_mid = np.argmin(np.abs(th_c - PI / 2.0))
    dn    = max(1, len(th_c) // 20)           # ~5% of probe cells around midplane
    i_lo  = max(0, i_mid - dn)
    i_hi  = min(len(th_c), i_mid + dn + 1)
    rho_mid   = rho[:, i_lo:i_hi, :].max(axis=(1, 2))   # (nr,)
    rho_floor = max(float(rho_mid.max()) * 1e-4, 1e-100)
    rho_mid   = np.maximum(rho_mid, rho_floor)
    g_r = np.abs(np.gradient(np.log(rho_mid), np.log(r_c)))
    g_r = np.maximum(_smooth(g_r, n=7), 0.02)
    return _march_log(r_c, g_r, threshold, r_c[0], r_c[-1], min_g=0.02)


def _build_th_interfaces(r_c, th_c, rho, threshold):
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
    # Floor: at most 0.25 rad/cell (prevents single-cell steps wider than ~14 deg)
    g_th = np.maximum(_smooth(g_th, n=7), threshold / 0.25)

    # Build upper hemisphere only (theta from th_c[0] toward pi/2)
    mask   = th_c <= PI / 2.0 + 1e-6
    th_up  = th_c[mask]
    g_up   = g_th[:len(th_up)]

    th_if_up = _march_lin(th_up, g_up, threshold,
                          x_min=th_c[0], x_max=PI / 2.0,
                          min_g=threshold / 0.25)
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
                            min_g=threshold / (PI / 6.0))
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
                   active, orig_ppar, threshold,
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

def build_smart_grid(ppar, threshold=0.1, verbose=True):
    """
    Compute an optimal RADMC-3D spherical grid from a pipeline params dict.

    Evaluates the dust density model on a coarse probe grid, measures the
    actual |delta ln rho| gradient in every direction, and places cell
    interfaces so each cell satisfies |delta ln rho| <= threshold.

    Parameters
    ----------
    ppar      : dict   -- pipeline config params (string values like '0.5*au' are ok).
    threshold : float  -- max |delta ln rho| per cell (default 0.1 = 10%).
    verbose   : bool   -- print the summary table.

    Returns
    -------
    dict with keys xbound, nx, ybound, ny, zbound, nz, nphot_scat_rec.
    """
    active = _detect_active(ppar)

    # Evaluate the density model on a probe grid
    r_c, th_c, ph_c, rho = _eval_probe_density(ppar, active)

    # Build interface arrays from measured density gradients
    r_if  = _build_r_interfaces(r_c, th_c, rho, threshold)
    th_if = _build_th_interfaces(r_c, th_c, rho, threshold)
    ph_if = _build_ph_interfaces(r_c, th_c, ph_c, rho, threshold, active)

    # Convert to (bounds, counts) format for radmc3dPy / problemSetupDust
    xbound, nx = _to_segments(r_if)
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
                       active, ppar, threshold,
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
    result = build_smart_grid(ppar, threshold=args.threshold, verbose=True)

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
