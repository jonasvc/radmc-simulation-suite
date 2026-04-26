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
from contextlib import contextmanager
from radmc3dPy import analyze, setup, image
from terminal_ui import print_success, print_error


# ===========================================================================
# DETAILED LOGGING HELPERS
# ===========================================================================

IGNORE_RADMC_ERROR_LINES_IF_RETURN_OK = True


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

def run_radmc_command(command_str, tracker, total_photons=None):
    """Execute a RADMC-3D command with mode-dependent output handling."""
    log_command(command_str)
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
            text=True, bufsize=1)
        all_output = []
        error_found = False
        error_msg = None
        with process.stdout:
            for line in iter(process.stdout.readline, ''):
                print(line, end='')
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
        text=True, bufsize=1)

    with process.stdout:
        for line in iter(process.stdout.readline, ''):
            clean_line = line.strip()
            if clean_line:
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
                          wavelength=2.2, threads=32, ui_mode='advanced'):
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
        if use_silencer:
            with suppress_output():
                analyze.writeDefaultParfile('ppdisk_complete')
        else:
            analyze.writeDefaultParfile('ppdisk_complete')
        log_phase_end("Setup", phase_start)
        tracker.complete_phase("Setup")

        ################################
        ### Phase 2: Configure Model ###
        ################################

        tracker.start_phase("Configure Model")
        phase_start = log_phase_start("Configure Model")

        dust_setup_args = {
            'xbound': params['xbound'], 'nx': params['nx'],
            'ybound': params['ybound'], 'ny': params['ny'],
            'zbound': params['zbound'], 'nz': params['nz'],
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

        if use_silencer:
            with suppress_output():
                setup.problemSetupDust('ppdisk_complete', **dust_setup_args)
        else:
            setup.problemSetupDust('ppdisk_complete', **dust_setup_args)

        with open("radmc3d.inp", "a") as f:
            f.write(f"mc_scat_maxtauabs = {params['mc_scat_maxtauabs']}\n")

        log_phase_end("Configure Model", phase_start)
        tracker.complete_phase("Configure Model")

        ###########################
        ### Phase 3: MC Thermal ###
        ###########################

        tracker.start_phase("MC Thermal")
        phase_start = log_phase_start("MC Thermal")
        run_radmc_command(
            f'radmc3d mctherm setthreads {threads} sloppy',
            tracker, total_photons=params['nphot'])
        log_phase_end("MC Thermal", phase_start)
        tracker.complete_phase("MC Thermal")

        ################################
        ### Phase 4: SED Calculation ###
        ################################

        tracker.start_phase("SED Calculation")
        phase_start = log_phase_start("SED Calculation")
        run_radmc_command(
            f'radmc3d sed incl {params["incl"]} setthreads {threads} sloppy',
            tracker, total_photons=params['nphot_spec'])
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
                cmd = (f"radmc3d image "
                       f"npix {params['npix']} "
                       f"incl {params['incl']} "
                       f"sizeau {params['sizeau']} "
                       f"{lambda_arg} "
                       f"phi {params['phi']} "
                       f"setthreads {threads}")
                if params['nostar']:
                    cmd += " nostar"
                run_radmc_command(cmd, tracker)
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
