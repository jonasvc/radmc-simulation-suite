<div align="center">

# RADMC Simulation Suite `v0.1-alpha`

**A config-driven RADMC-3D workflow for high-resolution protoplanetary-disk simulations.**

![status](https://img.shields.io/badge/status-alpha-f0a020)
![RADMC-3D](https://img.shields.io/badge/RADMC--3D-simulation%20suite-3b82f6)
![Python](https://img.shields.io/badge/python-3.x-3776ab)
![focus](https://img.shields.io/badge/focus-protoplanetary%20disks-16a34a)

`radmc-protodisk-suite-alpha`

</div>

---

## Overview

RADMC ProtoDisk Suite is a Python layer around RADMC-3D and `radmc3dPy`. It handles the repetitive work around disk simulations: building input files, launching RADMC-3D, tracking long runs, saving outputs, plotting diagnostics, and keeping runs organized.

The suite is aimed at spherical-grid disk models with large cell counts, high photon numbers, scattered-light images, SEDs, image cubes, and repeated parameter studies.

This is an alpha release. The workflow is usable for research runs, but paths, defaults, and interfaces may still change.

## Workflow

```mermaid
flowchart LR
    C[Config] --> S[Setup]
    S --> R[RADMC inputs]
    R --> M[mctherm]
    M --> E[SED]
    E --> I[Image or cube]
    I --> P[Plots and FITS]
    P --> L[Run folder and logbook]
```

## At A Glance

| Part | What it does |
| --- | --- |
| Config system | Uses Python configs for stellar, disk, dust, grid, image, and Monte-Carlo parameters |
| RADMC runner | Executes setup, thermal Monte-Carlo, SED, image, and cube phases |
| Progress UI | Shows phase progress, photon progress, and wavelength progress |
| Smart grid | Builds non-uniform spherical grids from active disk structures |
| Large-grid checks | Estimates memory use and validates key RADMC input and output files |
| Batch mode | Runs parameter sweeps with separate output folders |
| Plotting | Produces SED, density, temperature, and combined diagnostic plots |
| Logbook | Records parameters, runtime, status, errors, and run locations |
| Interferometry | Computes visibilities, closure phases, and UV coverage from image output |
| Analysis GUI | Opens completed runs and combines plotting with interferometry tools |

## Disk Model Features

The default model setup focuses on protoplanetary disks on spherical grids. Current config controls include:

- Stellar temperature, radius, mass, distance, and inclination
- Disk mass, inner radius, outer radius, scale height, and flaring
- Dust opacity choice and dust mixture settings
- Wavelength grid and camera wavelength setup
- Spherical `r x theta x phi` grid setup
- Thermal, scattering, and SED photon counts
- OpenMP thread and runtime settings
- Single-image and image-cube camera settings
- Puffed-up inner rim and smoothed rim regions
- Fourier modulation of scale height and surface density
- Spiral height and density perturbations
- Vortices and ring-like density structures
- Gap-like density structures
- Radial damping for azimuthal perturbations
- Experimental warped disk terms
- Azimuthally localized inner-edge shadowing
- Vertical density steepness control

Reference configs are kept in `configs/` for baseline, shadow, spiral, vortex, gap, warped, and project-specific runs.

## Smart Grid Builder

The smart grid builder creates a non-uniform spherical grid from the active structure in the config. It concentrates cells near the regions that usually need them: the inner rim, disk midplane, gaps, vortices, spirals, and sharp radial transitions.

It reports:

- Radial, polar, and azimuthal cell counts
- Total cell count
- Lower-bound RAM estimate
- Detected active structures
- Large-grid warnings
- Suggested scattering photon count for image calculations

Standalone check:

```bash
python grid_builder.py --config config.py --dry-run --plot
```

## Running A Simulation

Start the launcher:

```bash
python main.py
```

The launcher asks for:

- Run name
- Sandbox or Project-style layout
- Default, reference, or custom config
- Single run or batch run
- Advanced progress UI or raw RADMC output
- Optional single image
- Optional image cube
- Optional reference SED overlay
- Optional smart grid builder

A single run goes through:

1. Setup
2. Model configuration
3. Thermal Monte-Carlo
4. SED calculation
5. Image or cube generation
6. File saving
7. Diagnostic plotting
8. Logbook update

## Output Layout

The output root is set in `main.py` through `SIMULATIONS_ROOT`.

Current default:

```text
/home/main/RADMC/Simulations
```

Sandbox runs:

```text
Simulations/sandbox/<category>/<YYYY-MM-DD_HHMM_run-name>/
```

Project runs:

```text
Simulations/project/<source>/<opacity>_incl<inclination>/<YYYY-MM-DD_HHMM_run-name>/
```

Each run folder stores the config snapshot, log file, RADMC input files, RADMC output files, spectra, images, plots, and status information.

## Large-Grid Safeguards

The suite checks the generated RADMC files before launching expensive runs and checks important outputs afterward.

Validated files include:

- `amr_grid.inp`
- `wavelength_micron.inp`
- `stars.inp`
- `dustopac.inp`
- `radmc3d.inp`
- `dust_density.binp`
- `dust_temperature.bdat`

Large-run support also includes binary density output, optional chunked density writing, memory estimates, and cleanup of temporary files after successful runs.

## Images, SEDs, And Cubes

The pipeline can compute:

- SEDs over the configured wavelength grid
- Single-wavelength images
- Spectral image cubes using `camera_wavelength_micron.inp`
- FITS files through `radmc3dPy`
- Quick-look PNG files

For thermal Monte-Carlo and image scattering runs, progress is based on photon counts. For SED and wavelength-dependent work, progress is based on wavelength steps.

## Plotting

The plotting module writes standard diagnostics into the run folder:

- SED plots
- SED plots with AB Aur reference data
- Dust density contours
- Dust temperature contours
- Combined temperature and density plots
- Zoomed density plots

## Batch Mode

Batch mode starts from a base config and applies parameter overrides for each run. Each batch member gets its own output folder, merged config snapshot, log file, RADMC files, plots, runtime, and status entry.

## Logbook

The logbook records completed and failed runs so previous simulations remain searchable.

```bash
python view_logbook.py view 20
python view_logbook.py search mdisk="0.01*ms" n_arms=2
python view_logbook.py export simulation_logbook.csv
```

## Interferometry And GUI

The interferometry tools work on RADMC image output and compute synthetic observables for comparison with interferometric data.

They include:

- VLTI-style baseline definitions
- Custom baseline and position-angle support
- Visibility amplitudes
- Closure phases
- UV-coverage plots
- Visibility and closure-phase plotting helpers

The public GUI entry point is:

```bash
python radmc_analysis_gui_with_interfero.py
```

It can inspect completed runs, plot SEDs, show density and temperature structure, inspect FITS images when `astropy` is installed, and run the interferometry analysis tools.

## Patched RADMC-3D Support

The config can point to a specific RADMC executable through `radmc3d_exe`. This can be a standard RADMC-3D build or a modified fork.

When the executable supports them, the pipeline writes the additional scattering controls:

```text
mcscat_phi_coarsen
mc_peeledoff
```

It also exposes OpenMP-related settings:

```text
OMP_DYNAMIC
OMP_PROC_BIND
OMP_PLACES
setthreads
```

## Main Files

| Path | Purpose |
| --- | --- |
| `main.py` | Interactive launcher |
| `single_run.py` | Core RADMC run workflow |
| `batch_run.py` | Parameter sweep runner |
| `config.py` | Default simulation config |
| `configs/` | Reference and example configs |
| `config_loader.py` | Default, reference, and custom config loading |
| `grid_builder.py` | Smart spherical grid builder |
| `plots.py` | SED, density, and temperature plots |
| `export.py` | Simulation logbook writer |
| `view_logbook.py` | Logbook view, search, and export helper |
| `interferometry.py` | Visibility and closure-phase tools |
| `radmc_analysis_gui_with_interfero.py` | Analysis GUI with interferometry tools |

## Requirements

Python packages are listed in `requirements.txt`.

Core packages:

- `numpy`
- `scipy`
- `pandas`
- `openpyxl`
- `matplotlib`
- `rich`
- `psutil`
- `filelock`
- `radmc3dPy`

External requirements:

- RADMC-3D executable
- Working `radmc3dPy` installation
- OpenMP-capable RADMC build for threaded runs

Optional packages:

- `PyQt5` for the GUI
- `astropy` for FITS viewing

## Alpha Status

This is a working research suite, not a polished package release. Before running it on another machine, check:

- `SIMULATIONS_ROOT` in `main.py`
- `radmc3d_exe` in the config
- RADMC-3D and `radmc3dPy` paths
- Available memory for large grids
- Whether the selected RADMC executable supports the patched controls
- Whether the selected reference config matches the science case

