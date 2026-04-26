"""
Module for running batch simulations with multiple parameter combinations.

Folder layout (sandbox):
    simulations/sandbox/batch/<YYYY-MM-DD_HHMM_name>/
    +-- batch_summary.txt
    +-- config_base.py
    +-- batch_run.py
    +-- 001_suffix-a/
    +-- 002_suffix-b/

Folder layout (IRYSS):
    simulations/iryss/<source>/<opacity>_incl<incl>/batch/<YYYY-MM-DD_HHMM_name>/
    +-- batch_summary.txt
    +-- config_base.py
    +-- batch_run.py
    +-- 001_suffix-a/
    +-- 002_suffix-b/

Edit param_combinations below to define your parameter grid.
"""

import os
import time
import logging
import shutil
import json
from main import setup_logging, get_params_dict, resolve_config_path, SIMULATIONS_ROOT
from single_run import run_single_simulation
from plots import create_all_plots


###############################################
### DEFINE YOUR PARAMETER COMBINATIONS HERE ###
###############################################

param_combinations = [
    {
        "name_suffix": "scat_2e10",
        "nphot_scat":  "2e+10",
    },
]


########################
### HELPER FUNCTIONS ###
########################

def merge_params(base_params, override_params):
    """Merge base parameters with per-combination overrides."""
    merged = base_params.copy()
    merged.update({k: v for k, v in override_params.items() if k != 'name_suffix'})
    return merged


###########################
### BATCH RUN EXECUTION ###
###########################

def run_batch_mode(user_inputs, base_timestamp):
    """Execute batch runs with multiple parameter combinations."""

    base_name       = user_inputs['name']
    iryss_meta      = user_inputs['iryss_meta']
    config_name     = user_inputs['config_name']
    make_images     = user_inputs['make_images']
    make_image_cube = user_inputs.get('make_image_cube', False)
    wavelength      = user_inputs['wavelength']
    reference_sed   = user_inputs['reference_sed']
    ui_mode         = user_inputs['ui_mode']

    print("\n" + "=" * 60)
    print(f"BATCH MODE: {len(param_combinations)} combinations")
    print(f"Type:        {'IRYSS' if iryss_meta else 'Sandbox'}")
    print(f"Base config: {config_name or 'config.py'}")
    if iryss_meta:
        print(f"Source:      {iryss_meta['source']}")
        print(f"Opacity:     {iryss_meta['opacity']}")
        print(f"Inclination: {iryss_meta['inclination']}°")
    print("=" * 60 + "\n")

    # Load base parameters
    from config_loader import load_config, get_params_dict_from_config
    if config_name:
        config      = load_config(config_name)
        base_params = get_params_dict_from_config(config)
        print(f"Loaded base config: {config_name}\n")
    else:
        import config as cfg
        base_params = get_params_dict(cfg)
        print("Loaded default config.py\n")

    # Build batch root directory
    from naming import generate_batch_run_directory, format_timestamp
    short_ts   = format_timestamp(base_timestamp)

    if iryss_meta:
        source  = iryss_meta['source']
        opacity = iryss_meta['opacity']
        incl    = iryss_meta['inclination']
        batch_root = os.path.join(
            SIMULATIONS_ROOT, 'iryss', source,
            f"{opacity}_incl{incl}",
            'batch', f"{short_ts}_{base_name}"
        )
    else:
        batch_root = os.path.join(
            SIMULATIONS_ROOT, 'sandbox', 'batch', f"{short_ts}_{base_name}"
        )

    os.makedirs(batch_root, exist_ok=True)

    # Copy base config and batch script to batch root
    config_path = resolve_config_path(config_name)
    if os.path.exists(config_path):
        shutil.copy(config_path, os.path.join(batch_root, "config_base.py"))
    if os.path.exists("batch_run.py"):
        shutil.copy("batch_run.py", os.path.join(batch_root, "batch_run.py"))

    batch_start = time.time()
    results     = []

    for idx, param_combo in enumerate(param_combinations, start=1):

        name_suffix = param_combo.get('name_suffix', f'combo_{idx}')
        params      = merge_params(base_params, param_combo)

        run_name = f"{idx:03d}_{name_suffix}"
        run_dir  = os.path.join(batch_root, run_name)
        os.makedirs(run_dir, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"[{idx}/{len(param_combinations)}]  {run_name}")
        print("=" * 60)

        setup_logging(run_dir)

        logging.info(f"Batch combination: {run_name}")
        logging.info(f"Base config: {config_name or 'config.py'}")
        if iryss_meta:
            logging.info(f"IRYSS source:      {iryss_meta['source']}")
            logging.info(f"IRYSS opacity:     {iryss_meta['opacity']}")
            logging.info(f"IRYSS inclination: {iryss_meta['inclination']}")
        logging.info("Modified parameters:")
        for k, v in param_combo.items():
            if k != 'name_suffix':
                logging.info(f"  {k} = {v}")

        run_start = time.time()
        try:
            spec, star, grid = run_single_simulation(
                params=params,
                run_dir=run_dir,
                name=run_name,
                timestamp=base_timestamp,
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

            # Save merged config snapshot
            with open(os.path.join(run_dir, "config.py"), 'w') as f:
                f.write("# Auto-generated merged config for this combination\n")
                f.write(f"# Base: {config_name or 'config.py'}\n")
                f.write(f"# Overrides: {json.dumps({k: str(v) for k, v in param_combo.items() if k != 'name_suffix'})}\n\n")
                for k, v in params.items():
                    f.write(f"{k} = {repr(v)}\n")

            runtime = (time.time() - run_start) / 60
            logging.info(f"Runtime: {runtime:.2f} min")
            results.append({
                'idx': idx, 'name': run_name,
                'runtime': runtime, 'status': 'SUCCESS'
            })
            print(f"Done in {runtime:.2f} min")

        except Exception as e:
            runtime = (time.time() - run_start) / 60
            logging.error(f"FAILED: {e}")
            results.append({
                'idx': idx, 'name': run_name,
                'runtime': runtime, 'status': 'FAILED', 'error': str(e)
            })
            print(f"FAILED: {e}")

    # Summary
    total_runtime = (time.time() - batch_start) / 60
    n_ok   = sum(1 for r in results if r['status'] == 'SUCCESS')
    n_fail = sum(1 for r in results if r['status'] == 'FAILED')

    print("\n" + "=" * 60)
    print("BATCH COMPLETE")
    print(f"  Success: {n_ok}   Failed: {n_fail}")
    print(f"  Total:   {total_runtime:.1f} min ({total_runtime/60:.2f} h)")
    print(f"  Folder:  {batch_root}")
    print("=" * 60 + "\n")

    summary_path = os.path.join(batch_root, "batch_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("BATCH SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Base name:   {base_name}\n")
        f.write(f"Type:        {'IRYSS' if iryss_meta else 'Sandbox'}\n")
        if iryss_meta:
            f.write(f"Source:      {iryss_meta['source']}\n")
            f.write(f"Opacity:     {iryss_meta['opacity']}\n")
            f.write(f"Inclination: {iryss_meta['inclination']}°\n")
        f.write(f"Base config: {config_name or 'config.py'}\n")
        f.write(f"Timestamp:   {base_timestamp}\n")
        f.write(f"Total runs:  {len(param_combinations)}\n")
        f.write(f"Success:     {n_ok}\n")
        f.write(f"Failed:      {n_fail}\n")
        f.write(f"Total time:  {total_runtime:.1f} min\n\n")
        f.write("-" * 60 + "\n")
        for r in results:
            symbol = "OK" if r['status'] == 'SUCCESS' else "FAIL"
            f.write(f"[{symbol}] {r['name']:30s}  {r['runtime']:6.2f} min\n")
            if r['status'] == 'FAILED':
                f.write(f"      Error: {r.get('error', '?')}\n")

    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    print("This module is called from main.py - run: python main.py")
