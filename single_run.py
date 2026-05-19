"""
Module for running a single RADMC-3D simulation.
Supports 'Advanced UI' (Visual Progress) and 'Raw' (Standard RADMC Output).
Supports Single Image and Image Cube generation.
"""

import os
import sys
import shutil
import logging
import time
import subprocess
import shlex
import re
import ast
import struct
from contextlib import contextmanager
from radmc3dPy import analyze, setup, image
from terminal_ui import print_success, print_error
from streaming_density import problem_setup_dust_streaming, should_stream_density


# ===========================================================================
# DETAILED LOGGING HELPERS
# ===========================================================================

IGNORE_RADMC_ERROR_LINES_IF_RETURN_OK = True


def _as_count(value):
    """Return the total cell count for a radmc3dPy grid count parameter."""
    if isinstance(value, (list, tuple)):
        return int(sum(value))
    return int(value)


def _list_length_from_config_value(value):
    if isinstance(value, (list, tuple)):
        return len(value)
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return 1
        if isinstance(parsed, (list, tuple)):
            return len(parsed)
    return 1


def estimate_grid_memory(params):
    nx = _as_count(params['nx'])
    ny = _as_count(params['ny'])
    nz = _as_count(params['nz'])
    ncell = nx * ny * nz
    nspec = max(
        1,
        _list_length_from_config_value(params.get('mixabun', [1.0])),
        _list_length_from_config_value(params.get('dustkappa', [1.0])),
    )

    # Lower-bound estimate for the dominant resident arrays used by the Python
    # setup and RADMC-3D thermal MC. Temporary arrays can push this higher.
    bytes_per_double = 8
    python_density_arrays = 5 * ncell * bytes_per_double
    dust_species_arrays = 2 * nspec * ncell * bytes_per_double
    radmc_core_arrays = (2 * nspec + 4) * ncell * bytes_per_double
    lower_bound_gib = (
        python_density_arrays + dust_species_arrays + radmc_core_arrays
    ) / 1024.0**3

    return {
        'nx': nx,
        'ny': ny,
        'nz': nz,
        'ncell': ncell,
        'nspec': nspec,
        'lower_bound_gib': lower_bound_gib,
    }


def log_grid_memory_estimate(params):
    estimate = estimate_grid_memory(params)
    msg = (
        "Grid estimate: "
        f"{estimate['nx']} x {estimate['ny']} x {estimate['nz']} = "
        f"{estimate['ncell']:,} cells, "
        f"~{estimate['lower_bound_gib']:.1f} GiB lower-bound working memory"
    )
    print(msg)
    logging.info(msg)

    if estimate['ncell'] > 100_000_000:
        warning = (
            "WARNING: This grid is very large for a single-node RADMC-3D run. "
            "Expect setup and mctherm memory use to be substantially above the estimate."
        )
        print(warning)
        logging.warning(warning)

    return estimate


def validate_radmc_input_files(params):
    """Check the generated RADMC-3D input files before launching mctherm."""
    required_files = [
        "amr_grid.inp",
        "wavelength_micron.inp",
        "stars.inp",
        "dustopac.inp",
        "radmc3d.inp",
        "dust_density.binp",
    ]
    missing = [fname for fname in required_files if not os.path.exists(fname)]
    if missing:
        raise RuntimeError("Missing RADMC-3D input files: " + ", ".join(missing))

    empty = [fname for fname in required_files if os.path.getsize(fname) == 0]
    if empty:
        raise RuntimeError("Empty RADMC-3D input files: " + ", ".join(empty))

    ncell_expected = (
        _as_count(params['nx']) *
        _as_count(params['ny']) *
        _as_count(params['nz'])
    )
    with open("dust_density.binp", "rb") as rfile:
        header = rfile.read(32)
    if len(header) != 32:
        raise RuntimeError("dust_density.binp is too small to contain a valid binary header")

    iformat, precision, ncell, nspec = struct.unpack("=qqqq", header)
    if iformat != 1:
        raise RuntimeError(f"dust_density.binp has unsupported iformat={iformat}; expected 1")
    if precision not in (4, 8):
        raise RuntimeError(f"dust_density.binp has unsupported precision={precision}; expected 4 or 8")
    if ncell != ncell_expected:
        raise RuntimeError(
            f"dust_density.binp cell count {ncell:,} does not match grid {ncell_expected:,}"
        )
    if nspec < 1:
        raise RuntimeError(f"dust_density.binp has invalid dust species count nspec={nspec}")

    expected_size = 32 + ncell * nspec * precision
    actual_size = os.path.getsize("dust_density.binp")
    if actual_size != expected_size:
        raise RuntimeError(
            "dust_density.binp size mismatch: "
            f"expected {expected_size:,} bytes from header, found {actual_size:,} bytes"
        )

    msg = (
        "RADMC input check: dust_density.binp "
        f"ncell={ncell:,}, nspec={nspec}, precision={precision}, "
        f"size={actual_size / 1024.0**3:.2f} GiB"
    )
    print(msg)
    logging.info(msg)


def validate_mctherm_output():
    """Fail early if RADMC-3D returned without writing a usable temperature file."""
    fname = "dust_temperature.bdat"
    if not os.path.exists(fname):
        raise RuntimeError(
            "RADMC-3D mctherm finished/returned without creating dust_temperature.bdat. "
            "This usually means RADMC-3D stopped before the photon loop, often during "
            "large-grid memory allocation. Check the RADMC lines in log.txt and the "
            "cluster stderr/OOM status."
        )
    size = os.path.getsize(fname)
    if size == 0:
        raise RuntimeError("RADMC-3D created an empty dust_temperature.bdat file")
    logging.info(f"[MCTHERM_OUTPUT] {fname} size={size:,} bytes")


def log_phase_start(phase_name):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    logging.info("=" * 70)
    logging.info(f"[PHASE_START] {phase_name}")
    logging.info(f"[TIMESTAMP] {timestamp}")
    logging.info("=" * 70)
    return time.time()


def log_phase_end(phase_name, start_time):
    end_time = time.time()
    duration = end_time - start_time
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    if duration < 60:
        duration_str = f"{duration:.2f} seconds"
    elif duration < 3600:
        duration_str = f"{duration/60:.2f} minutes ({duration:.1f} seconds)"
    else:
        duration_str = f"{duration/3600:.2f} hours ({duration/60:.1f} minutes)"
    logging.info("-" * 70)
    logging.info(f"[PHASE_END] {phase_name}")
    logging.info(f"[TIMESTAMP] {timestamp}")
    logging.info(f"[DURATION] {duration_str}")
    logging.info("-" * 70)
    logging.info("")


def log_command(command_str, cwd=None):
    logging.info(f"[CMD] {command_str}")
    if cwd:
        logging.info(f"[CWD] {cwd}")
    else:
        logging.info(f"[CWD] {os.getcwd()}")


@contextmanager
def suppress_output():
    """Redirects stdout/stderr to devnull at OS level."""
    with open(os.devnull, "w") as devnull:
        old_stdout = os.dup(1)
        old_stderr = os.dup(2)
        try:
            os.dup2(devnull.fileno(), 1)
            os.dup2(devnull.fileno(), 2)
            yield
        finally:
            os.dup2(old_stdout, 1)
            os.dup2(old_stderr, 2)
            os.close(old_stdout)
            os.close(old_stderr)


#########################################################################
### RAW TRACKER
#########################################################################

class RawTracker:
    def start(self): pass
    def stop(self): pass
    def start_phase(self, name): print(f"\n>>> Starting Phase: {name}")
    def complete_phase(self, name): print(f">>> Completed Phase: {name}\n")
    def log(self, msg): pass
    def set_phase_total(self, n): pass
    def update_progress(self, n): pass
    def print_summary(self): pass


#################################################################
### RADMC-3D COMMAND EXECUTION
#################################################################

def build_radmc_env(params):
    """Build environment overrides for RADMC-3D/OpenMP runs."""
    env = os.environ.copy()
    mapping = {
        'omp_stacksize': 'OMP_STACKSIZE',
        'omp_dynamic': 'OMP_DYNAMIC',
        'omp_proc_bind': 'OMP_PROC_BIND',
        'omp_places': 'OMP_PLACES',
    }
    for param_key, env_key in mapping.items():
        if param_key in params and params[param_key] is not None:
            value = str(params[param_key]).strip()
            if value and value.lower() not in ('none', 'default', 'auto'):
                env[env_key] = value
    return env


def run_radmc_command(command_str, tracker, total_photons=None, env=None):
    """Execute a RADMC-3D command with mode-dependent output handling."""
    log_command(command_str)
    if env is not None:
        for key in ('OMP_STACKSIZE', 'OMP_DYNAMIC', 'OMP_PROC_BIND', 'OMP_PLACES'):
            if key in env:
                logging.info(f"[ENV] {key}={env[key]}")
    cmd_start_time = time.time()
    cmd_args = shlex.split(command_str)

    if not shutil.which(cmd_args[0]):
        print(f"Error: Command '{cmd_args[0]}' not found!")
        logging.error(f"Command not found: {cmd_args[0]}")
        raise FileNotFoundError(f"Command not found: {cmd_args[0]}")

    # RAW MODE
    if isinstance(tracker, RawTracker):
        process = subprocess.Popen(
            cmd_args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, env=env)
        all_output = []
        error_found = False
        error_msg = None
        with process.stdout:
            for line in iter(process.stdout.readline, ''):
                print(line, end='')
                logging.info("[RADMC] " + line.rstrip())
                all_output.append(line)
                if 'ERROR' in line.upper() and not error_found:
                    error_found = True
                    error_msg = line.strip()
        return_code = process.wait()
        logging.info(f"[RETURN_CODE] {return_code}")
        if error_found:
            if IGNORE_RADMC_ERROR_LINES_IF_RETURN_OK and return_code == 0:
                logging.warning(f"RADMC-3D ERROR line ignored (return code 0): {error_msg}")
            else:
                logging.error(f"RADMC-3D ERROR detected: {error_msg}")
                raise RuntimeError(f"RADMC-3D ERROR: {error_msg}")
        elif return_code != 0:
            logging.error(f"Command failed with return code {return_code}")
            raise RuntimeError(f"Command failed with code {return_code}")
        return

    # ADVANCED MODE
    if total_photons:
        try:
            safe_total = int(float(total_photons))
            tracker.set_phase_total(safe_total)
        except ValueError:
            tracker.log(f"[yellow]Warning: Could not parse total_photons '{total_photons}'.[/yellow]")

    photon_pattern = re.compile(r"Photon\s+nr[:.]?\s+(\d+)", re.IGNORECASE)
    error_found = False
    error_msg = None

    process = subprocess.Popen(
        cmd_args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, env=env)

    with process.stdout:
        for line in iter(process.stdout.readline, ''):
            clean_line = line.strip()
            if clean_line:
                logging.info("[RADMC] " + clean_line)
                tracker.log(clean_line)
                if 'ERROR' in clean_line.upper() and not error_found:
                    error_found = True
                    error_msg = clean_line
                if total_photons:
                    match = photon_pattern.search(clean_line)
                    if match:
                        try:
                            current_photon = int(match.group(1))
                            tracker.update_progress(current_photon)
                        except ValueError:
                            pass

    return_code = process.wait()
    logging.info(f"[RETURN_CODE] {return_code}")

    if error_found:
        if IGNORE_RADMC_ERROR_LINES_IF_RETURN_OK and return_code == 0:
            tracker.log("[yellow]RADMC-3D ERROR line ignored (return code 0).[/yellow]")
            logging.warning(f"RADMC-3D ERROR line ignored (return code 0): {error_msg}")
        else:
            tracker.log(f"[red bold]RADMC-3D ERROR detected![/red bold]")
            logging.error(f"RADMC-3D ERROR detected: {error_msg}")
            raise RuntimeError(f"RADMC-3D ERROR: {error_msg}")
    elif return_code != 0:
        tracker.log(f"[red bold]Process failed with code {return_code}[/red bold]")
        logging.error(f"Command failed with return code {return_code}")
        raise RuntimeError(f"Command failed with code {return_code}")


#################################################################
### CLEANUP
#################################################################

# Files generated by RADMC-3D that should be removed from the
# pipeline directory after a successful run.
# These have all been copied to run_dir in Phase 6 beforehand.
RADMC_TEMP_FILES = [
    "dust_temperature.bdat",
    "spectrum.out",
    "image.out",
    "amr_grid.inp",
    "stars.inp",
    "dust_density.binp",
    "dustopac.inp",
    "wavelength_micron.inp",
    "radmc3d.inp",
    "problem_params.inp",
    "ppdisk_complete.inp",   # default parfile written by radmc3dPy
    "ppdisk_complete_amr.inp",
]


def cleanup_pipeline_dir():
    """
    Remove temporary RADMC-3D files from the pipeline (working) directory.
    Called after all files have been safely copied to run_dir.
    Files that don't exist are silently skipped.
    """
    removed = []
    skipped = []
    for fname in RADMC_TEMP_FILES:
        if os.path.exists(fname):
            try:
                os.remove(fname)
                removed.append(fname)
            except Exception as e:
                logging.warning(f"Could not remove {fname}: {e}")
                skipped.append(fname)
    logging.info(f"[CLEANUP] Removed {len(removed)} temp files: {removed}")
    if skipped:
        logging.warning(f"[CLEANUP] Could not remove: {skipped}")
    return removed, skipped


#################################################################
### MAIN SIMULATION FUNCTION
#################################################################

def run_single_simulation(params, run_dir, name, timestamp, make_images=False,
                          make_image_cube=False,
                          wavelength=2.2, threads=32, ui_mode='advanced',
                          smart_grid_params=None):
    """
    Run a single RADMC-3D simulation.
    Supports generation of Single Images and Spectral Image Cubes.
    """

    phases = ["Setup", "Configure Model", "MC Thermal", "SED Calculation"]
    if make_images or make_image_cube:
        phases.append("Generate Image")
    phases.append("Save Files")

    estimates = {
        "MC Thermal":     None,
        "SED Calculation": None,
        "Generate Image":  None,
    }

    if ui_mode == 'advanced':
        from terminal_ui import AdvancedPhaseTracker
        tracker = AdvancedPhaseTracker(phases, estimates)
        use_silencer = True
    else:
        tracker = RawTracker()
        use_silencer = False

    tracker.start()
    start_time = time.time()

    try:
        ######################
        ### Phase 1: Setup ###
        ######################

        tracker.start_phase("Setup")
        phase_start = log_phase_start("Setup")
        model_name = params.get('model_name', 'ppdisk_complete')
        radmc3d_exe = params.get('radmc3d_exe', 'radmc3d')
        radmc_env = build_radmc_env(params)
        if use_silencer:
            with suppress_output():
                analyze.writeDefaultParfile(model_name)
        else:
            analyze.writeDefaultParfile(model_name)
        log_phase_end("Setup", phase_start)
        tracker.complete_phase("Setup")

        ################################
        ### Phase 2: Configure Model ###
        ################################

        tracker.start_phase("Configure Model")
        phase_start = log_phase_start("Configure Model")

        # Apply smart grid overrides if provided
        effective_params = dict(params)
        if smart_grid_params is not None:
            for k in ('xbound', 'nx', 'ybound', 'ny', 'zbound', 'nz'):
                effective_params[k] = smart_grid_params[k]

        grid_estimate = log_grid_memory_estimate(effective_params)

        dust_setup_args = {
            'xbound': effective_params['xbound'], 'nx': effective_params['nx'],
            'ybound': effective_params['ybound'], 'ny': effective_params['ny'],
            'zbound': effective_params['zbound'], 'nz': effective_params['nz'],
            'wbound': params['wbound'], 'nw': params['nw'],
            'rstar': params['rstar'], 'mstar': params['mstar'], 'tstar': params['tstar'],
            'istar_sphere': params['istar_sphere'],
            'mdisk': params['mdisk'], 'sig0': params['sig0'],
            'rin': params['rin'], 'rdisk': params['rdisk'],
            'hrdisk': params['hrdisk'], 'hrpivot': params['hrpivot'],
            'plsig1': params['plsig1'], 'plh': params['plh'],
            'sigma_type': params['sigma_type'],
            'hpr_prim_rout': params['hpr_prim_rout'], 'prim_rout': params['prim_rout'],
            'srim_rout': params['srim_rout'], 'srim_plsig': params['srim_plsig'],
            'dustkappa_ext': params['dustkappa'],
            'gsmax': params['gsmax'], 'gsmin': params['gsmin'],
            'mixabun': params['mixabun'],
            'nphot': params['nphot'], 'nphot_scat': params['nphot_scat'],
            'nphot_spec': params['nphot_spec'],
            'modified_random_walk': params['modified_random_walk'],
            'scattering_mode_max': params['scattering_mode_max'],
            'h_fourier_aj': params['h_fourier_aj'], 'h_fourier_bj': params['h_fourier_bj'],
            'sig_fourier_aj': params['sig_fourier_aj'], 'sig_fourier_bj': params['sig_fourier_bj'],
            'h_modulation_strength': params['h_modulation_strength'],
            'h_asymmetry_factor': params['h_asymmetry_factor'],
            'sig_asymmetry_factor': params['sig_asymmetry_factor'],
            'sig_modulation_strength': params['sig_modulation_strength'],
            'h_spiral_amp': params['h_spiral_amp'], 'sig_spiral_amp': params['sig_spiral_amp'],
            'spiral_pitch': params['spiral_pitch'], 'n_arms': params['n_arms'],
            'spiral_width_phi': params['spiral_width_phi'],
            'spiral_sharpness': params['spiral_sharpness'],
            'h_vortex_amp': params['h_vortex_amp'], 'h_vortex_phi0': params['h_vortex_phi0'],
            'h_vortex_r0': params['h_vortex_r0'], 'h_vortex_width_phi': params['h_vortex_width_phi'],
            'h_vortex_width_r': params['h_vortex_width_r'],
            'sig_vortex_amp': params['sig_vortex_amp'], 'sig_vortex_phi0': params['sig_vortex_phi0'],
            'sig_vortex_r0': params['sig_vortex_r0'], 'sig_vortex_width_phi': params['sig_vortex_width_phi'],
            'sig_vortex_width_r': params['sig_vortex_width_r'],
            'vortex_sharpness': params['vortex_sharpness'],
            'use_radial_damping': params['use_radial_damping'],
            'azimuthal_r_max': params['azimuthal_r_max'],
            'azimuthal_r_width': params['azimuthal_r_width'],
            'enable_warp': params['enable_warp'],
            'warp_amplitude': params['warp_amplitude'],
            'warp_phase': params['warp_phase'],
            'warp_mode': params['warp_mode'],
            'use_inner_edge_shadow': params['use_inner_edge_shadow'],
            'inner_edge_radius': params['inner_edge_radius'],
            'inner_edge_width': params['inner_edge_width'],
            'inner_edge_height': params['inner_edge_height'],
            'inner_edge_azimuthal': params['inner_edge_azimuthal'],
            'inner_edge_phi': params['inner_edge_phi'],
            'inner_edge_phi_width': params['inner_edge_phi_width'],
            'vertical_steepness': params['vertical_steepness'],
            'bgdens': params['bgdens'],
            'binary': True,
        }
        for optional_key in ('grid_style', 'crd_sys', 'levelMaxLimit', 'threshold'):
            if optional_key in params:
                dust_setup_args[optional_key] = params[optional_key]

        use_streaming_density, density_writer_reason = should_stream_density(
            params=effective_params,
            estimate=grid_estimate,
            model_name=model_name,
        )
        density_writer_msg = (
            "Density writer: "
            + ("streaming/chunked" if use_streaming_density else "normal radmc3dPy")
            + f" ({density_writer_reason})"
        )
        tracker.log(density_writer_msg)
        logging.info(density_writer_msg)

        if use_streaming_density:
            streaming_kwargs = dict(dust_setup_args)
            streaming_kwargs['chunk_nphi'] = params.get('density_chunk_nphi', 8)
            streaming_kwargs['density_precision'] = params.get('density_binary_precision', 8)
            if use_silencer:
                with suppress_output():
                    problem_setup_dust_streaming(model_name, **streaming_kwargs)
            else:
                problem_setup_dust_streaming(model_name, **streaming_kwargs)
        else:
            if use_silencer:
                with suppress_output():
                    setup.problemSetupDust(model_name, **dust_setup_args)
            else:
                setup.problemSetupDust(model_name, **dust_setup_args)

        with open("radmc3d.inp", "a") as f:
            f.write(f"mc_scat_maxtauabs = {params['mc_scat_maxtauabs']}\n")

        validate_radmc_input_files(effective_params)

        log_phase_end("Configure Model", phase_start)
        tracker.complete_phase("Configure Model")

        ###########################
        ### Phase 3: MC Thermal ###
        ###########################

        tracker.start_phase("MC Thermal")
        phase_start = log_phase_start("MC Thermal")
        run_radmc_command(
            f'{radmc3d_exe} mctherm setthreads {threads} sloppy',
            tracker, total_photons=params['nphot'], env=radmc_env)
        validate_mctherm_output()
        log_phase_end("MC Thermal", phase_start)
        tracker.complete_phase("MC Thermal")

        ################################
        ### Phase 4: SED Calculation ###
        ################################

        tracker.start_phase("SED Calculation")
        phase_start = log_phase_start("SED Calculation")
        run_radmc_command(
            f'{radmc3d_exe} sed incl {params["incl"]} setthreads {threads} sloppy',
            tracker, total_photons=params['nphot_spec'], env=radmc_env)
        log_phase_end("SED Calculation", phase_start)
        tracker.complete_phase("SED Calculation")

        #######################
        ### Phase 5: Images ###
        #######################

        if make_images or make_image_cube:
            tracker.start_phase("Generate Image")
            phase_start = log_phase_start("Generate Image")

            def compute_and_save(mode_type, lambda_arg, fits_prefix, png_prefix):
                tracker.log(f"Computing {mode_type} ({lambda_arg})...")
                logging.info(f"Computing {mode_type} using {lambda_arg}")
                cmd = (f"{radmc3d_exe} image "
                       f"npix {params['npix']} "
                       f"incl {params['incl']} "
                       f"sizeau {params['sizeau']} "
                       f"{lambda_arg} "
                       f"phi {params['phi']} "
                       f"setthreads {threads}")
                if params['nostar']:
                    cmd += " nostar"
                run_radmc_command(cmd, tracker, env=radmc_env)
                im = image.readImage()
                fits_filename = f'{fits_prefix}_{name}_{timestamp}.fits'
                im.writeFits(fits_filename, dpc=params['pc'])
                shutil.move(fits_filename, os.path.join(run_dir, fits_filename))
                tracker.log(f"Saved {fits_filename}")
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                png_filename = f'{png_prefix}_{name}_{timestamp}.png'
                fig = plt.figure()
                try:
                    image.plotImage(im, au=True, log=True, cmap='gist_heat')
                    fig.tight_layout()
                    fig.savefig(png_filename, dpi=300, bbox_inches='tight')
                    shutil.move(png_filename, os.path.join(run_dir, png_filename))
                except Exception as e:
                    tracker.log(f"[yellow]Warning: Could not plot PNG for {mode_type}: {e}[/yellow]")
                finally:
                    plt.close(fig)

            if make_images:
                compute_and_save(
                    mode_type="Single Image",
                    lambda_arg=f"lambda {wavelength}",
                    fits_prefix="Img",
                    png_prefix="Image")

            if make_image_cube:
                compute_and_save(
                    mode_type="Image Cube",
                    lambda_arg="loadlambda",
                    fits_prefix="ImgCube",
                    png_prefix="ImageCube")

            log_phase_end("Generate Image", phase_start)
            tracker.complete_phase("Generate Image")

        #####################
        ### Phase 6: Save ###
        #####################

        tracker.start_phase("Save Files")
        phase_start = log_phase_start("Save Files")
        logging.info("Reading output files and saving to run directory")

        spec = analyze.readSpectrum(fname='spectrum.out')
        shutil.copy("spectrum.out", os.path.join(run_dir, "spectrum.out"))
        star = analyze.readStars()
        grid = analyze.readGrid()

        files_to_save = [
            ("problem_params.inp",   "problem_params.inp"),
            ("radmc3d.inp",          "radmc3d.inp"),
            ("amr_grid.inp",         "amr_grid.inp"),
            ("stars.inp",            "stars.inp"),
            ("dust_density.binp",    "dust_density.binp"),
            ("image.out",            "image.out"),
            ("dustopac.inp",         "dustopac.inp"),
            ("wavelength_micron.inp","wavelength_micron.inp"),
            ("dust_temperature.bdat","dust_temperature.bdat"),
        ]

        if 'dustkappa' in params:
            dustkappa_file = params['dustkappa']
            dustkappa_filename = os.path.basename(dustkappa_file)
            files_to_save.append((dustkappa_filename, dustkappa_filename))

        for ref_file in ["ABAur_Dominik.txt", "ABAur_Dullemond.txt"]:
            if os.path.exists(ref_file):
                files_to_save.append((ref_file, ref_file))

        saved_count = 0
        for src, dst in files_to_save:
            if os.path.exists(src):
                shutil.copy(src, os.path.join(run_dir, dst))
                saved_count += 1

        log_phase_end("Save Files", phase_start)
        tracker.complete_phase("Save Files")

        tracker.stop()

        try:
            from export import log_simulation
            log_simulation(params, run_dir, name, timestamp,
                           (time.time() - start_time) / 60, "SUCCESS")
        except Exception:
            pass

        tracker.print_summary()
        return spec, star, grid

    except Exception as e:
        tracker.stop()
        runtime_minutes = (time.time() - start_time) / 60
        error_msg = str(e)
        logging.error(f"Simulation failed: {error_msg}")
        print_error(f"Simulation failed: {error_msg}")
        try:
            from export import log_simulation
            log_simulation(params, run_dir, name, timestamp,
                           runtime_minutes, status="FAILED", error_msg=error_msg)
        except Exception:
            pass
        raise e
