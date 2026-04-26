"""
Config Loader Utility
Allows loading reference configurations or custom config files
"""

import sys
import os
import importlib.util


# Available reference configurations
REFERENCE_CONFIGS = {
    'baseline': 'config_baseline.py',
    'shadow': 'config_shadow.py',
    'shadow_baseline': 'config_baseline_asym.py',
    'spiral': 'config_spiral.py',
    'vortex': 'config_vortex.py',
    'spiral_phys': 'spiral_structure.py',
    'vortex_phys': 'vortex_structure.py',
    'gap_phys': 'gap_structure.py',
    'warp_phys': 'warped_structure.py',
    'planet': 'config_montesinos_cpd.py',
    'combined': 'config_combined.py',
    'lowgrid': 'config_baseline_asym_grid.py',
    'Matter': 'config_baseline_asym_Matter.py',
}


def list_available_configs():
    """
    List all available reference configurations
    """
    print("\nAvailable reference configurations:")
    print("-" * 40)
    for key, filename in REFERENCE_CONFIGS.items():
        filepath = os.path.join('configs', filename)
        if os.path.exists(filepath):
            print(f"  {key:15s} -> {filename}")
        else:
            print(f"  {key:15s} -> {filename} [NOT FOUND]")
    print("-" * 40)
    print("\nDefault: config.py (current directory)")
    print()


def load_config(config_name=None):
    """
    Load a configuration file
    
    Parameters:
    -----------
    config_name : str or None
        Name of reference config ('baseline', 'spiral', etc.)
        If None, loads default config.py
        
    Returns:
    --------
    config_module : module
        Loaded configuration module
        
    Examples:
    ---------
    # Load default config
    config = load_config()
    
    # Load baseline reference
    config = load_config('baseline')
    
    # Load custom config file
    config = load_config('my_custom_config.py')
    """
    
    # Determine config file path
    if config_name is None:
        # Default: load config.py from current directory
        config_path = 'config.py'
        module_name = 'config'
    elif config_name in REFERENCE_CONFIGS:
        # Load reference config from configs/ directory
        config_path = os.path.join('configs', REFERENCE_CONFIGS[config_name])
        module_name = f'config_{config_name}'
    elif config_name.endswith('.py'):
        # Load custom config file
        config_path = config_name
        module_name = os.path.splitext(os.path.basename(config_name))[0]
    else:
        # Try as reference config first, then as filename
        if os.path.exists(os.path.join('configs', f'{config_name}.py')):
            config_path = os.path.join('configs', f'{config_name}.py')
            module_name = f'config_{config_name}'
        elif os.path.exists(config_name):
            config_path = config_name
            module_name = os.path.splitext(os.path.basename(config_name))[0]
        else:
            raise FileNotFoundError(f"Config file not found: {config_name}")
    
    # Check if file exists
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    # Load the module
    spec = importlib.util.spec_from_file_location(module_name, config_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    
    print(f"Loaded configuration: {config_path}")
    
    return config_module


def get_params_dict_from_config(config_module):
    """
    Extract parameters from a config module
    
    Parameters:
    -----------
    config_module : module
        Configuration module
        
    Returns:
    --------
    params : dict
        Dictionary of parameters
    """
    params = {}
    for key in dir(config_module):
        if not key.startswith('__'):
            value = getattr(config_module, key)
            # Skip module imports
            if not str(type(value)).startswith("<class 'module'>"):
                params[key] = value
    return params


def copy_config_to_run_dir(config_path, run_dir, run_name, timestamp):
    """
    Copy the config file to the run directory
    
    Parameters:
    -----------
    config_path : str
        Path to config file
    run_dir : str
        Run directory
    run_name : str
        Run name
    timestamp : str
        Timestamp
    """
    import shutil
    if os.path.exists(config_path):
        dest = os.path.join(run_dir, f"config_{run_name}_{timestamp}.py")
        shutil.copy(config_path, dest)


# Command line interface
if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == 'list':
            list_available_configs()
        else:
            config_name = sys.argv[1]
            try:
                config = load_config(config_name)
                print(f"\nSuccessfully loaded: {config_name}")
                print("\nKey parameters:")
                print(f"  mdisk: {config.mdisk}")
                print(f"  hrdisk: {config.hrdisk}")
                print(f"  h_spiral_amp: {config.h_spiral_amp}")
                print(f"  sig_spiral_amp: {config.sig_spiral_amp}")
            except Exception as e:
                print(f"Error loading config: {e}")
    else:
        list_available_configs()
