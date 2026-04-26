# plots.py
# Creates plots from RADMC-3D simulation results.
# Files are saved with simple readable names, no run name stamping.

import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as plb
from radmc3dPy import analyze, natconst


def mirror_tauy(data):
    # Mirror tauy data across the midplane for symmetric representation.
    print("Mirroring tauy for symmetric representation...")
    theta_midplane_idx = np.argmin(np.abs(data.grid.y - np.pi / 2))
    tauy_symmetric = data.tauy.copy()
    for i in range(theta_midplane_idx):
        mirror_idx = 2 * theta_midplane_idx - i
        if mirror_idx < tauy_symmetric.shape[1]:
            tauy_symmetric[:, mirror_idx, :] = data.tauy[:, i, :]
    print("Mirroring completed")
    return tauy_symmetric


def plot_sed(spec, star, grid, pc, run_dir, reference_file=None):
    plt.figure()
    plb.title(r'SED')
    plt.xscale('log')
    plt.yscale('log')
    if reference_file is not None:
        data = np.loadtxt(reference_file)
        plt.plot(data[:, 0], data[:, 1],
                 label=f'{os.path.basename(reference_file)}')
    analyze.plotSpectrum(a=spec, nufnu=True, micron=True, xlg=True,
                         ylg=True, dpc=float(pc), oplot=True, label=r'Disk')
    flux = star.fnustar / (pc ** 2)
    flux = np.reshape(flux, len(spec))
    plt.plot(grid.wav, grid.freq * flux, label='Stellar contribution')
    plt.xlabel(r'$\lambda$ [$\mu$m]')
    plt.ylabel(r'log $\nu F_{\nu}$ [erg.s$^{-1}$.cm$^{-2}$]')
    plt.xlim(0.1, 3000)
    plt.ylim(1e-15, 10 ** -6)
    plt.legend(loc='lower left')
    plt.savefig(os.path.join(run_dir, "SED.png"), dpi=250, bbox_inches='tight')
    plt.close()


def plot_dust_density(data, run_dir, wav=2.2):
    tauy_symmetric = mirror_tauy(data)
    plb.figure()
    plb.title(r'Dust density contours with $\tau=1$')
    c1 = plb.contourf(data.grid.x / natconst.au,
                      np.pi / 2. - data.grid.y,
                      np.log10(data.rhodust[:, :, 0, 0].T), 30)
    plb.xlabel('r [AU]')
    plb.ylabel(r'$\pi/2-\theta$')
    plb.xscale('log')
    cb = plb.colorbar(c1)
    cb.ax.yaxis.labelpad = 20
    cb.set_label(r'$\log_{10}{\rho}$', rotation=270.)
    c2 = plb.contour(data.grid.x / natconst.au,
                     np.pi / 2. - data.grid.y,
                     tauy_symmetric[:, :, 0].T,
                     [1.0], colors='w', linestyles='solid')
    plb.clabel(c2, inline=1, fontsize=10, fmt='%g')
    plb.savefig(os.path.join(run_dir, "dust_density.png"), dpi=300)
    plb.close()


def plot_dust_temperature(data, run_dir):
    plb.figure()
    plb.title(r'Dust temperature contours')
    c3 = plb.contourf(data.grid.x / natconst.au,
                      np.pi / 2. - data.grid.y,
                      data.dusttemp[:, :, 0, 0].T, 30)
    plb.xlabel('r [AU]')
    plb.ylabel(r'$\pi/2-\theta$')
    plb.xscale('log')
    cb = plb.colorbar(c3)
    cb.set_label('T [K]', rotation=270.)
    c4 = plb.contour(data.grid.x / natconst.au,
                     np.pi / 2. - data.grid.y,
                     data.dusttemp[:, :, 0, 0].T,
                     10, colors='k', linestyles='solid')
    plb.clabel(c4, inline=1, fontsize=10)
    cb.ax.yaxis.labelpad = 20
    plb.savefig(os.path.join(run_dir, "dust_temperature.png"), dpi=300)
    plb.close()


def plot_temp_dens_combined(data, data_dens, run_dir):
    plb.figure()
    plb.title(r'Temperature and density structure')
    plt.xlim(0.5, 200)
    plt.ylim(0, 0.4)
    c5 = plb.contourf(data_dens.grid.x / natconst.au,
                      np.pi / 2 - data_dens.grid.y,
                      np.log10(data_dens.rhodust[:, :, 0, 0].T), 30)
    plb.xlabel('r [AU]')
    plb.ylabel(r'$\pi/2-\theta$')
    plb.xscale('log')
    plb.colorbar(c5, label=r'$\log_{10}(\rho)$')
    c_lines = plb.contour(data.grid.x / natconst.au,
                          np.pi / 2 - data.grid.y,
                          data.dusttemp[:, :, 0, 0].T,
                          30, linestyles='solid')
    plb.clabel(c_lines, inline=1, fontsize=10)
    plb.savefig(os.path.join(run_dir, "temperature_density.png"), dpi=300)
    plb.close()


def plot_density_zoom(data, run_dir):
    tauy_symmetric = mirror_tauy(data)
    plb.figure()
    plb.title(r'Dust density contours with $\tau=1$ (zoom)')
    c7 = plb.contourf(data.grid.x / natconst.au,
                      np.pi / 2. - data.grid.y,
                      np.log10(data.rhodust[:, :, 0, 0].T), 30)
    plb.xlabel('r [AU]')
    plb.ylabel(r'$\pi/2-\theta$')
    plb.xscale('log')
    cb = plb.colorbar(c7)
    cb.ax.yaxis.labelpad = 20
    cb.set_label(r'$\log_{10}{\rho}$', rotation=270.)
    c8 = plb.contour(data.grid.x / natconst.au,
                     np.pi / 2. - data.grid.y,
                     tauy_symmetric[:, :, 0].T,
                     [1.0], colors='w', linestyles='solid')
    plb.clabel(c8, inline=1, fontsize=10, fmt='%g')
    plt.ylim(0, 0.5)
    plt.xlim(0.5,)
    plb.savefig(os.path.join(run_dir, "density_zoom.png"), dpi=300)
    plb.close()


def create_all_plots(run_dir, pc, wav=2.2, reference_file=None,
                     name=None, timestamp=None):
    spec      = analyze.readSpectrum(fname=os.path.join(run_dir, 'spectrum.out'))
    star      = analyze.readStars()
    grid      = analyze.readGrid()
    plot_sed(spec, star, grid, pc, run_dir, reference_file)
    data      = analyze.readData(dtemp=True, ddens=True)
    opac      = analyze.readOpac(ext=['astrosilicateoptool'])
    data.getTau(wav=wav)
    plot_dust_density(data, run_dir, wav)
    plot_dust_temperature(data, run_dir)
    data_dens = analyze.readData(ddens=True)
    plot_temp_dens_combined(data, data_dens, run_dir)
    plot_density_zoom(data, run_dir)
