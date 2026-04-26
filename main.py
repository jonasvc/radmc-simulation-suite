"""
Main entry point for RADMC-3D simulations.
Handles user input and coordinates single or batch runs.
"""

import os
import sys
import datetime
import logging
import time
import shutil
from single_run import run_single_simulation
from plots import create_all_plots
from terminal_ui import print_banner, print_success, print_error, print_system_info, print_parameter_table

# Root directory for all simulation output
SIMULATIONS_ROOT = "/home/main/RADMC/Simulations"


def get_iryss_meta():
    """
    Prompt the user for IRYSS-specific metadata.
    Returns a dict with keys: source, opacity, inclination.
    """
    print("\n--- IRYSS Run Configuration ---")
    source      = input("  Source object (e.g. AB-Aur, HD-142527): ").strip()
    opacity     = input("  Opacity label (e.g. astrosil, draine84):  ").strip()
    inclination = input("  Inclination in degrees (e.g. 45):          ").strip()

    if not source or not opacity or not inclination:
        print("All IRYSS fields are required. Exiting.")
        sys.exit(1)

    return {
        'source':      source,
        'opacity':     opacity,
        'inclination': inclination,
    }


def get_user_inputs():
    """Collect user inputs for simulation configuration."""

    name = input("Please define a name for this run: ").strip()
    if not name:
        print("Name cannot be empty. Exiting.")
        sys.exit(1)

    # Run type: sandbox or IRYSS
    print("\nRun type:")
    print("1 - Sandbox  (general testing and exploration)")
    print("2 - IRYSS    (project run: source, opacity, inclination)")
    type_choice = input("Please choose 1 or 2: ").strip()

    if type_choice == "1":
        iryss_meta = None
    elif type_choice == "2":
        iryss_meta = get_iryss_meta()
    else:
        print("Invalid choice. Exiting.")
        sys.exit(1)

    # Config selection
    print("\nSelect configuration:")
    print("1 - Default (config.py)")
    print("2 - Reference configs (baseline, spiral, etc.)")
    print("3 - Custom config file")
    config_choice = input("Please choose 1, 2, or 3: ").strip()

    config_name = None
    if config_choice == "2":
        from config_loader import REFERENCE_CONFIGS
        print("\nAvailable reference configurations:")
        for i, key in enumerate(REFERENCE_CONFIGS.keys(), start=1):
            print(f"  {i} - {key}")
        ref_choice = input("\nEnter configuration number: ").strip()
        try:
            ref_idx = int(ref_choice) - 1
            if 0 <= ref_idx < len(REFERENCE_CONFIGS):
                config_name = list(REFERENCE_CONFIGS.keys())[ref_idx]
            else:
                print("Invalid number. Exiting.")
                sys.exit(1)
        except ValueError:
            print("Please enter a valid number. Exiting.")
            sys.exit(1)
    elif config_choice == "3":
        custom_path = input("Enter path to custom config file: ").strip()
        if os.path.exists(custom_path):
            config_name = custom_path
        else:
            print(f"File not found: {custom_path}. Exiting.")
            sys.exit(1)

    # Run mode
    print("\nSelect run mode:")
    print("1 - Single run")
    print("2 - Batch run")
    mode_choice = input("Please choose 1 or 2: ").strip()
    if mode_choice == "1":
        run_mode = "single"
    elif mode_choice == "2":
        run_mode = "batch"
    else:
        print("Invalid choice. Exiting.")
        sys.exit(1)

    # UI mode
    print("\nSelect Output Mode:")
    print("1 - Advanced UI (visual progress bars)")
    print("2 - Raw Output (standard RADMC-3D terminal output)")
    ui_choice = input("Please choose 1 or 2: ").strip()
    ui_mode = "advanced" if ui_choice == "1" else "raw"

    # Image configuration
    print("\n--- Image Configuration ---")
    make_images     = False
    make_image_cube = False
    wavelength      = 2.2

    if input("Compute images/cubes? (y/n): ").strip().lower() == 'y':
        if input("  > Compute SINGLE image at specific wavelength? (y/n): ").strip().lower() == 'y':
            make_images = True
            try:
                wavelength = float(input("    > Wavelength (micron): ").strip() or "2.2")
            except ValueError:
                wavelength = 2.2
        if input("  > Compute spectral IMAGE CUBE? (y/n): ").strip().lower() == 'y':
            make_image_cube = True

    # Reference SED
    print("\nReference SED for AB Aur:")
    print("1 - ABAur_Dominik.txt")
    print("2 - ABAur_Dullemond.txt")
    print("3 - None")
    choice = input("Please choose 1, 2 or 3: ").strip()
    reference_sed = {"1": "ABAur_Dominik.txt", "2": "ABAur_Dullemond.txt"}.get(choice)

    return {
        'name':            name,
        'iryss_meta':      iryss_meta,
        'config_name':     config_name,
        'make_images':     make_images,
        'make_image_cube': make_image_cube,
        'wavelength':      wavelength,
        'reference_sed':   reference_sed,
        'run_mode':        run_mode,
        'ui_mode':         ui_mode,
    }


def setup_logging(run_dir):
    """Set up logging to log.txt inside the run directory."""
    log_file = os.path.join(run_dir, "log.txt")
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s %(levelname)s: %(message)s'
    )


def get_params_dict(config_module):
    """Extract non-dunder, non-module attributes from a config module."""
    params = {}
    for key in dir(config_module):
        if not key.startswith('__'):
            value = getattr(config_module, key)
            if not str(type(value)).startswith("<class 'module'>"):
                params[key] = value
    return params


def resolve_config_path(config_name):
    """Return the filesystem path for the given config name."""
    if config_name is None:
        return "config.py"
    from config_loader import REFERENCE_CONFIGS
    if config_name in REFERENCE_CONFIGS:
        return os.path.join('configs', REFERENCE_CONFIGS[config_name])
    return config_name


def run_single_mode(user_inputs, timestamp):
    """Execute a single simulation run."""

    name            = user_inputs['name']
    iryss_meta      = user_inputs['iryss_meta']
    config_name     = user_inputs['config_name']
    make_images     = user_inputs['make_images']
    make_image_cube = user_inputs['make_image_cube']
    wavelength      = user_inputs['wavelength']
    reference_sed   = user_inputs['reference_sed']
    ui_mode         = user_inputs['ui_mode']

    # Load config
    from config_loader import load_config
    if config_name:
        config = load_config(config_name)
        print(f"\nUsing configuration: {config_name}")
    else:
        import config
        print("\nUsing default configuration: config.py")

    params = get_params_dict(config)

    from naming import generate_run_directory, determine_category
    run_dir, run_name = generate_run_directory(
        SIMULATIONS_ROOT, name, params, timestamp,
        iryss_meta=iryss_meta
    )
    category = determine_category(params) if not iryss_meta else "iryss"

    print_banner("single", name, category, timestamp)
    print_system_info()
    print_parameter_table(params, show_all=False)

    os.makedirs(run_dir, exist_ok=True)
    setup_logging(run_dir)

    logging.info(f"Run name:      {run_name}")
    logging.info(f"Configuration: {config_name or 'config.py'}")
    logging.info(f"UI mode:       {ui_mode}")
    if iryss_meta:
        logging.info(f"IRYSS source:      {iryss_meta['source']}")
        logging.info(f"IRYSS opacity:     {iryss_meta['opacity']}")
        logging.info(f"IRYSS inclination: {iryss_meta['inclination']}")

    start_time = time.time()

    spec, star, grid = run_single_simulation(
        params=params,
        run_dir=run_dir,
        name=run_name,
        timestamp=timestamp,
        make_images=make_images,
        make_image_cube=make_image_cube,
        wavelength=wavelength,
        threads=params['threads'],
        ui_mode=ui_mode,
    )

    create_all_plots(
        run_dir=run_dir,
        pc=params['pc'],
        wav=wavelength,
        reference_file=reference_sed,
    )

    # Cleanup temp files from pipeline dir now that plots are done
    from single_run import cleanup_pipeline_dir
    removed, skipped = cleanup_pipeline_dir()
    print(f"Cleanup: removed {len(removed)} temporary files from pipeline directory.")

    config_path = resolve_config_path(config_name)
    if os.path.exists(config_path):
        shutil.copy(config_path, os.path.join(run_dir, "config.py"))

    runtime = (time.time() - start_time) / 60
    logging.info(f"Runtime: {runtime:.2f} minutes")

    print("\n")
    print_success(f"Completed in {runtime:.1f} minutes!")
    print_success(f"Results: {run_dir}")
    print("\n")


def main():
    print("\n" + "=" * 60)
    print("RADMC-3D Simulation Suite")
    print("=" * 60 + "\n")

    user_inputs = get_user_inputs()
    timestamp   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    if user_inputs['run_mode'] == 'single':
        run_single_mode(user_inputs, timestamp)
    elif user_inputs['run_mode'] == 'batch':
        from batch_run import run_batch_mode
        run_batch_mode(user_inputs, timestamp)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
