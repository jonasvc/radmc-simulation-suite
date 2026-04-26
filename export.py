"""
Module for logging all simulations to a central Excel/CSV file
Each simulation appends its parameters to the logbook
"""

import pandas as pd
import os
import json
from datetime import datetime
from filelock import FileLock
import warnings

# Suppress openpyxl warnings
warnings.filterwarnings('ignore', category=UserWarning, module='openpyxl')


class SimulationLogbook:
    """
    Central logbook for all simulations
    Appends data after each simulation run
    """
    
    def __init__(self, logbook_path="/home/main/RADMC/Simulations/simulation_logbook.xlsx"):
        """
        Initialize logbook
        
        Parameters:
        -----------
        logbook_path : str
            Path to central logbook file
        """
        self.logbook_path = logbook_path
        self.lock_path = logbook_path + ".lock"
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(logbook_path)), exist_ok=True)
    
    def add_simulation(self, params, run_dir, name, timestamp, runtime_minutes, 
                      status="SUCCESS", error_msg=None):
        """
        Add a simulation to the logbook
        
        Parameters:
        -----------
        params : dict
            All simulation parameters
        run_dir : str
            Directory where results are saved
        name : str
            Simulation name
        timestamp : str
            Timestamp of simulation
        runtime_minutes : float
            Runtime in minutes
        status : str
            SUCCESS or FAILED
        error_msg : str, optional
            Error message if failed
        """
        
        # Prepare row data
        row_data = {
            # Meta information
            'Timestamp': timestamp,
            'Name': name,
            'Status': status,
            'Runtime_min': round(runtime_minutes, 2),
            'Base_Directory': os.path.basename(os.path.dirname(run_dir)),
            'Directory': run_dir,
            'Error': error_msg if error_msg else '',
            
            # Stellar parameters
            'tstar': params.get('tstar', ''),
            'rstar': params.get('rstar', ''),
            'mstar': params.get('mstar', ''),
            'istar_sphere': params.get('istar_sphere', ''),
            'pc': params.get('pc', ''),
            'incl': params.get('incl', ''),
            
            # Disk parameters
            'mdisk': params.get('mdisk', ''),
            'rin': params.get('rin', ''),
            'rdisk': params.get('rdisk', ''),
            'hrdisk': params.get('hrdisk', ''),
            'plsig1': params.get('plsig1', ''),
            'plh': params.get('plh', ''),
            'hrpivot': params.get('hrpivot', ''),
            'sigma_type': params.get('sigma_type', ''),
            'sig0': params.get('sig0', ''),
            
            # Inner rim parameters
            'hpr_prim_rout': params.get('hpr_prim_rout', ''),
            'prim_rout': params.get('prim_rout', ''),
            'srim_rout': params.get('srim_rout', ''),
            'srim_plsig': params.get('srim_plsig', ''),
            
            # Dust parameters
            'dustkappa': str(params.get('dustkappa', '')),
            'gsmax': params.get('gsmax', ''),
            'gsmin': params.get('gsmin', ''),
            'mixabun': str(params.get('mixabun', '')),
            
            # Grid parameters
            'wbound': str(params.get('wbound', '')),
            'nw': str(params.get('nw', '')),
            'xbound': str(params.get('xbound', '')),
            'nx': str(params.get('nx', '')),
            'ybound': str(params.get('ybound', '')),
            'ny': str(params.get('ny', '')),
            'zbound': str(params.get('zbound', '')),
            'nz': str(params.get('nz', '')),
            
            # Computational parameters
            'nphot': params.get('nphot', ''),
            'nphot_scat': params.get('nphot_scat', ''),
            'nphot_spec': params.get('nphot_spec', ''),
            'threads': params.get('threads', ''),
            'modified_random_walk': params.get('modified_random_walk', ''),
            'scattering_mode_max': params.get('scattering_mode_max', ''),
            'mc_scat_maxtauabs': params.get('mc_scat_maxtauabs', ''),
            
            # Image parameters
            'npix': params.get('npix', ''),
            'phi': params.get('phi', ''),
            'sizeau': params.get('sizeau', ''),
            'nostar': params.get('nostar', ''),
            
            # Spiral parameters
            'h_spiral_amp': params.get('h_spiral_amp', ''),
            'sig_spiral_amp': params.get('sig_spiral_amp', ''),
            'spiral_pitch': params.get('spiral_pitch', ''),
            'n_arms': params.get('n_arms', ''),
            'spiral_width_phi': params.get('spiral_width_phi', ''),
            'spiral_sharpness': params.get('spiral_sharpness', ''),
            
            # Vortex parameters
            'h_vortex_amp': str(params.get('h_vortex_amp', '')),
            'h_vortex_phi0': str(params.get('h_vortex_phi0', '')),
            'h_vortex_r0': str(params.get('h_vortex_r0', '')),
            'h_vortex_width_phi': str(params.get('h_vortex_width_phi', '')),
            'h_vortex_width_r': str(params.get('h_vortex_width_r', '')),
            'sig_vortex_amp': str(params.get('sig_vortex_amp', '')),
            'sig_vortex_phi0': str(params.get('sig_vortex_phi0', '')),
            'sig_vortex_r0': str(params.get('sig_vortex_r0', '')),
            'sig_vortex_width_phi': str(params.get('sig_vortex_width_phi', '')),
            'sig_vortex_width_r': str(params.get('sig_vortex_width_r', '')),
            'vortex_sharpness': params.get('vortex_sharpness', ''),
            
            # Fourier parameters
            'h_fourier_aj': str(params.get('h_fourier_aj', '')),
            'h_fourier_bj': str(params.get('h_fourier_bj', '')),
            'sig_fourier_aj': str(params.get('sig_fourier_aj', '')),
            'sig_fourier_bj': str(params.get('sig_fourier_bj', '')),
            'h_modulation_strength': params.get('h_modulation_strength', ''),
            'h_asymmetry_factor': params.get('h_asymmetry_factor', ''),
            'sig_modulation_strength': params.get('sig_modulation_strength', ''),
            'sig_asymmetry_factor': params.get('sig_asymmetry_factor', ''),
            
            # Radial damping
            'use_radial_damping': params.get('use_radial_damping', ''),
            'azimuthal_r_max': params.get('azimuthal_r_max', ''),
            'azimuthal_r_width': params.get('azimuthal_r_width', ''),
            
            # Warp
            'enable_warp': params.get('enable_warp', ''),
            'warp_amplitude': params.get('warp_amplitude', ''),
            'warp_phase': params.get('warp_phase', ''),
            'warp_mode': params.get('warp_mode', ''),
            
            # Inner edge shadow
            'use_inner_edge_shadow': params.get('use_inner_edge_shadow', ''),
            'inner_edge_radius': params.get('inner_edge_radius', ''),
            'inner_edge_width': params.get('inner_edge_width', ''),
            'inner_edge_height': params.get('inner_edge_height', ''),
            'inner_edge_azimuthal': params.get('inner_edge_azimuthal', ''),
            'inner_edge_phi': params.get('inner_edge_phi', ''),
            'inner_edge_phi_width': params.get('inner_edge_phi_width', ''),
            
            # Vertical steepness
            'vertical_steepness': params.get('vertical_steepness', ''),
        }
        
        # Convert to DataFrame row
        new_row = pd.DataFrame([row_data])
        
        # Use file lock to prevent concurrent writes (for batch runs)
        lock = FileLock(self.lock_path, timeout=10)
        
        try:
            with lock:
                # Load existing data or create new
                if os.path.exists(self.logbook_path):
                    try:
                        existing_df = pd.read_excel(self.logbook_path, engine='openpyxl')
                        # Append new row
                        updated_df = pd.concat([existing_df, new_row], ignore_index=True)
                    except Exception as e:
                        print(f"Warning: Could not read existing logbook, creating new one. Error: {e}")
                        updated_df = new_row
                else:
                    updated_df = new_row
                
                # Save to Excel
                updated_df.to_excel(self.logbook_path, index=False, engine='openpyxl')
                
                # Also save as CSV backup
                csv_path = self.logbook_path.replace('.xlsx', '.csv')
                updated_df.to_csv(csv_path, index=False)
                
        except Exception as e:
            print(f"Error writing to logbook: {e}")
            print("Attempting to save to backup location...")
            
            # Backup: save to run directory
            backup_path = os.path.join(run_dir, f"logbook_backup_{timestamp}.xlsx")
            new_row.to_excel(backup_path, index=False, engine='openpyxl')
            print(f"Saved backup to: {backup_path}")
    
    def get_summary(self, last_n=10):
        """
        Get summary of last N simulations
        
        Parameters:
        -----------
        last_n : int
            Number of recent simulations to show
        
        Returns:
        --------
        DataFrame with last N simulations
        """
        if not os.path.exists(self.logbook_path):
            print("No logbook found yet.")
            return None
        
        df = pd.read_excel(self.logbook_path, engine='openpyxl')
        return df.tail(last_n)
    
    def search(self, **criteria):
        """
        Search logbook for specific parameters
        
        Example:
            logbook.search(mdisk="0.01*ms", n_arms=2)
        
        Returns:
        --------
        DataFrame with matching simulations
        """
        if not os.path.exists(self.logbook_path):
            print("No logbook found yet.")
            return None
        
        df = pd.read_excel(self.logbook_path, engine='openpyxl')
        
        # Filter by criteria
        mask = pd.Series([True] * len(df))
        for key, value in criteria.items():
            if key in df.columns:
                mask &= (df[key] == value)
        
        return df[mask]
    
    def export_to_csv(self, output_path=None):
        """
        Export logbook to CSV
        """
        if output_path is None:
            output_path = self.logbook_path.replace('.xlsx', '.csv')
        
        if not os.path.exists(self.logbook_path):
            print("No logbook found yet.")
            return
        
        df = pd.read_excel(self.logbook_path, engine='openpyxl')
        df.to_csv(output_path, index=False)
        print(f"Exported to: {output_path}")
        return output_path


def view_logbook(logbook_path="/home/main/RADMC/Simulations/simulation_logbook.xlsx", last_n=20):
    """
    Quick function to view the logbook
    
    Parameters:
    -----------
    logbook_path : str
        Path to logbook
    last_n : int
        Number of recent simulations to show
    """
    if not os.path.exists(logbook_path):
        print("No logbook found yet. Run some simulations first!")
        return
    
    df = pd.read_excel(logbook_path, engine='openpyxl')
    
    print("\n" + "="*80)
    print(f"SIMULATION LOGBOOK ({len(df)} total simulations)")
    print("="*80)
    
    # Show summary statistics
    print(f"\nStatus Summary:")
    print(df['Status'].value_counts().to_string())
    
    print(f"\nTotal Runtime: {df['Runtime_min'].sum():.1f} minutes ({df['Runtime_min'].sum()/60:.1f} hours)")
    print(f"Average Runtime: {df['Runtime_min'].mean():.1f} minutes")
    
    # Show last N simulations
    print(f"\n{'-'*80}")
    print(f"Last {last_n} simulations:")
    print("-"*80)
    
    recent = df.tail(last_n)
    
    # Show key columns
    display_cols = ['Timestamp', 'Name', 'Status', 'Runtime_min', 'mdisk', 'hrdisk', 
                   'h_spiral_amp', 'sig_spiral_amp', 'n_arms']
    display_cols = [col for col in display_cols if col in recent.columns]
    
    print(recent[display_cols].to_string(index=False))
    print("="*80 + "\n")


# Convenience function for scripts
def log_simulation(params, run_dir, name, timestamp, runtime_minutes, 
                  status="SUCCESS", error_msg=None,
                  logbook_path="/home/main/RADMC/Simulations/simulation_logbook.xlsx"):
    """
    Convenience function to log a simulation
    
    Usage in single_run.py or batch_run.py:
        from export import log_simulation
        log_simulation(params, run_dir, name, timestamp, runtime)
    """
    logbook = SimulationLogbook(logbook_path)
    logbook.add_simulation(params, run_dir, name, timestamp, runtime_minutes, 
                          status, error_msg)
    print(f"✓ Logged to: {logbook_path}")


if __name__ == "__main__":
    # Demo/test
    print("Simulation Logbook Module")
    print("\nTo view logbook, run:")
    print("  from export import view_logbook")
    print("  view_logbook()")