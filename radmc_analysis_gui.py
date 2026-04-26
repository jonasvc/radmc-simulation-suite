#!/usr/bin/env python3
"""
RADMC-3D Analysis GUI
=====================
A flexible PyQt5-based GUI for analyzing RADMC-3D simulation data.

Features:
- SED Plotting
- Temperature & Density Contours
- FITS Image Viewer (requires astropy)
- Parameter Inspection Table (New Tab)

Usage:
------
python radmc_analysis_gui.py
"""

import sys
import os
import glob
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QComboBox, 
                             QFileDialog, QTabWidget, QGroupBox, QDoubleSpinBox,
                             QCheckBox, QTextEdit, QSplitter, QLineEdit, QListWidget,
                             QMessageBox, QSpinBox, QTableWidget, QTableWidgetItem, QHeaderView,
                             QAbstractItemView)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QColor

# --- RADMC3D Import ---
try:
    from radmc3dPy import analyze
    RADMC3D_AVAILABLE = True
except ImportError:
    RADMC3D_AVAILABLE = False
    print("Warning: radmc3dPy not found. RADMC data analysis limited.")

# --- Astropy Import ---
try:
    from astropy.io import fits
    ASTROPY_AVAILABLE = True
except ImportError:
    ASTROPY_AVAILABLE = False
    print("Warning: astropy not found. FITS viewing disabled.")


class MplCanvas(FigureCanvas):
    """Matplotlib canvas for embedding in PyQt5"""
    def __init__(self, parent=None, width=8, height=6, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super(MplCanvas, self).__init__(self.fig)


class RADMCAnalysisGUI(QMainWindow):
    """Main GUI window for RADMC-3D analysis"""
    
    def __init__(self):
        super().__init__()
        self.current_run_dir = None
        self.spectrum = None
        self.grid = None
        self.star = None
        self.init_ui()
        
    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle('RADMC-3D Analysis Tool')
        self.setGeometry(100, 100, 1400, 900)
        
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # Create splitter for resizable panels
        splitter = QSplitter(Qt.Horizontal)
        
        # Left panel: Controls
        left_panel = self.create_control_panel()
        splitter.addWidget(left_panel)
        
        # Right panel: Plots & Data
        right_panel = self.create_plot_panel()
        splitter.addWidget(right_panel)
        
        # Set initial sizes (30% controls, 70% plots)
        splitter.setSizes([400, 1000])
        
        main_layout.addWidget(splitter)
        
        self.statusBar().showMessage('Ready')
        
    def create_control_panel(self):
        """Create the control panel with all settings"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Title
        title = QLabel('RADMC-3D Analysis')
        title.setFont(QFont('Arial', 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # === Run Selection Group ===
        run_group = QGroupBox('Run Selection')
        run_layout = QVBoxLayout()
        
        # Browse button
        browse_btn = QPushButton('Browse for Run Directory')
        browse_btn.clicked.connect(self.browse_run_directory)
        run_layout.addWidget(browse_btn)
        
        # Current run display
        self.current_run_label = QLabel('No run loaded')
        self.current_run_label.setWordWrap(True)
        self.current_run_label.setStyleSheet('padding: 5px; background-color: #f0f0f0;')
        run_layout.addWidget(self.current_run_label)
        
        # Quick path input
        quick_path_layout = QHBoxLayout()
        self.quick_path_input = QLineEdit()
        self.quick_path_input.setPlaceholderText('Or paste path here...')
        quick_load_btn = QPushButton('Load')
        quick_load_btn.clicked.connect(self.load_from_quick_path)
        quick_path_layout.addWidget(self.quick_path_input)
        quick_path_layout.addWidget(quick_load_btn)
        run_layout.addLayout(quick_path_layout)
        
        # Recent runs list
        self.recent_runs_list = QListWidget()
        self.recent_runs_list.setMaximumHeight(100)
        self.recent_runs_list.itemDoubleClicked.connect(self.load_from_recent)
        run_layout.addWidget(QLabel('Recent runs:'))
        run_layout.addWidget(self.recent_runs_list)
        
        run_group.setLayout(run_layout)
        layout.addWidget(run_group)
        
        # === Plot Settings Group (Tab-specific) ===
        plot_group = QGroupBox('Settings')
        plot_layout = QVBoxLayout()
        
        # Create a stacked widget to show different settings per tab
        self.settings_stack = QWidget()
        settings_stack_layout = QVBoxLayout(self.settings_stack)
        settings_stack_layout.setContentsMargins(0, 0, 0, 0)
        
        # --- 1. SED Settings ---
        self.sed_settings = QWidget()
        sed_settings_layout = QVBoxLayout(self.sed_settings)
        
        self.sed_logscale_cb = QCheckBox('Log scale')
        self.sed_logscale_cb.setChecked(True)
        sed_settings_layout.addWidget(self.sed_logscale_cb)
        
        # Wavelength range
        wl_layout = QHBoxLayout()
        wl_layout.addWidget(QLabel('λ range [µm]:'))
        self.wl_min_spin = QDoubleSpinBox()
        self.wl_min_spin.setRange(0.1, 10000)
        self.wl_min_spin.setValue(0.1)
        self.wl_max_spin = QDoubleSpinBox()
        self.wl_max_spin.setRange(0.1, 10000)
        self.wl_max_spin.setValue(1000.0)
        wl_layout.addWidget(self.wl_min_spin)
        wl_layout.addWidget(QLabel('-'))
        wl_layout.addWidget(self.wl_max_spin)
        sed_settings_layout.addLayout(wl_layout)
        
        # Reference SED
        self.show_reference_cb = QCheckBox('Show reference SED')
        self.show_reference_cb.setChecked(False)
        sed_settings_layout.addWidget(self.show_reference_cb)
        
        self.reference_combo = QComboBox()
        self.reference_combo.addItems(['None', 'ABAur_Dominik.txt', 'ABAur_Dullemond.txt'])
        self.reference_combo.setEnabled(False)
        sed_settings_layout.addWidget(self.reference_combo)
        
        self.show_reference_cb.stateChanged.connect(
            lambda state: self.reference_combo.setEnabled(state == 2)
        )
        
        # --- 2. Temperature Settings ---
        self.temp_settings = QWidget()
        temp_settings_layout = QVBoxLayout(self.temp_settings)
        
        temp_settings_layout.addWidget(QLabel('Temperature plot settings:'))
        self.temp_cmap_combo = QComboBox()
        self.temp_cmap_combo.addItems(['hot', 'plasma', 'viridis', 'inferno'])
        temp_cmap_layout = QHBoxLayout()
        temp_cmap_layout.addWidget(QLabel('Colormap:'))
        temp_cmap_layout.addWidget(self.temp_cmap_combo)
        temp_settings_layout.addLayout(temp_cmap_layout)
        
        # --- 3. Density Settings ---
        self.density_settings = QWidget()
        density_settings_layout = QVBoxLayout(self.density_settings)
        
        density_settings_layout.addWidget(QLabel('Density plot settings:'))
        
        # Colormap
        self.dens_cmap_combo = QComboBox()
        self.dens_cmap_combo.addItems(['viridis', 'plasma', 'inferno', 'cividis'])
        dens_cmap_layout = QHBoxLayout()
        dens_cmap_layout.addWidget(QLabel('Colormap:'))
        dens_cmap_layout.addWidget(self.dens_cmap_combo)
        density_settings_layout.addLayout(dens_cmap_layout)
        
        # Tau=1 surface options
        self.show_tau_cb = QCheckBox('Show τ=1 surface')
        self.show_tau_cb.setChecked(True)
        density_settings_layout.addWidget(self.show_tau_cb)
        
        # Tau direction
        tau_dir_layout = QHBoxLayout()
        tau_dir_layout.addWidget(QLabel('τ direction:'))
        self.tau_direction_combo = QComboBox()
        self.tau_direction_combo.addItems(['tauy (vertical)', 'taux (radial)'])
        tau_dir_layout.addWidget(self.tau_direction_combo)
        density_settings_layout.addLayout(tau_dir_layout)
        
        # Tau wavelength
        tau_wav_layout = QHBoxLayout()
        tau_wav_layout.addWidget(QLabel('τ wavelength [µm]:'))
        self.tau_wavelength_spin = QDoubleSpinBox()
        self.tau_wavelength_spin.setRange(0.1, 1000)
        self.tau_wavelength_spin.setValue(2.2)
        self.tau_wavelength_spin.setDecimals(2)
        tau_wav_layout.addWidget(self.tau_wavelength_spin)
        density_settings_layout.addLayout(tau_wav_layout)

        # --- 4. FITS Image Settings ---
        self.image_settings = QWidget()
        image_settings_layout = QVBoxLayout(self.image_settings)
        
        image_settings_layout.addWidget(QLabel('FITS Image settings:'))
        
        # File Selection
        self.fits_file_combo = QComboBox()
        image_settings_layout.addWidget(QLabel('Select FITS file:'))
        image_settings_layout.addWidget(self.fits_file_combo)
        
        # Refresh Button
        refresh_fits_btn = QPushButton('Refresh File List')
        refresh_fits_btn.clicked.connect(self.scan_fits_files)
        image_settings_layout.addWidget(refresh_fits_btn)
        
        # Colormap
        self.img_cmap_combo = QComboBox()
        self.img_cmap_combo.addItems(['inferno', 'magma', 'viridis', 'gray', 'gist_heat'])
        img_cmap_layout = QHBoxLayout()
        img_cmap_layout.addWidget(QLabel('Colormap:'))
        img_cmap_layout.addWidget(self.img_cmap_combo)
        image_settings_layout.addLayout(img_cmap_layout)
        
        # Scaling
        self.img_log_cb = QCheckBox('Log Scale')
        self.img_log_cb.setChecked(True)
        image_settings_layout.addWidget(self.img_log_cb)

        # Slice Selection (for 3D/4D cubes)
        self.slice_spin = QSpinBox() 
        self.slice_spin.setRange(0, 1000)
        self.slice_spin.setValue(0)
        slice_layout = QHBoxLayout()
        slice_layout.addWidget(QLabel('Cube Slice (Index):'))
        slice_layout.addWidget(self.slice_spin)
        image_settings_layout.addLayout(slice_layout)

        # --- 5. Parameters Settings (Empty) ---
        self.param_settings = QWidget()
        self.param_settings.setLayout(QVBoxLayout())
        self.param_settings.layout().addWidget(QLabel("See Table in Right Panel ->"))
        
        # Add all setting widgets to stack
        settings_stack_layout.addWidget(self.sed_settings)
        settings_stack_layout.addWidget(self.temp_settings)
        settings_stack_layout.addWidget(self.density_settings)
        settings_stack_layout.addWidget(self.image_settings)
        settings_stack_layout.addWidget(self.param_settings)
        
        # Initially show SED settings
        self.temp_settings.hide()
        self.density_settings.hide()
        self.image_settings.hide()
        self.param_settings.hide()
        
        plot_layout.addWidget(self.settings_stack)
        plot_group.setLayout(plot_layout)
        layout.addWidget(plot_group)
        
        # === Action Buttons ===
        action_group = QGroupBox('Actions')
        action_layout = QVBoxLayout()
        
        plot_sed_btn = QPushButton('Plot SED')
        plot_sed_btn.clicked.connect(self.plot_sed)
        action_layout.addWidget(plot_sed_btn)
        
        plot_temp_btn = QPushButton('Plot Temperature')
        plot_temp_btn.clicked.connect(self.plot_temperature)
        action_layout.addWidget(plot_temp_btn)
        
        plot_density_btn = QPushButton('Plot Density')
        plot_density_btn.clicked.connect(self.plot_density)
        action_layout.addWidget(plot_density_btn)
        
        plot_image_btn = QPushButton('Plot FITS Image')
        plot_image_btn.clicked.connect(self.plot_fits_image)
        action_layout.addWidget(plot_image_btn)
        
        show_info_btn = QPushButton('Show File Overview')
        show_info_btn.clicked.connect(self.show_run_info)
        action_layout.addWidget(show_info_btn)
        
        action_group.setLayout(action_layout)
        layout.addWidget(action_group)
        
        # === Info Display ===
        info_group = QGroupBox('Quick Log')
        info_layout = QVBoxLayout()
        
        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        self.info_text.setMaximumHeight(150)
        info_layout.addWidget(self.info_text)
        
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)
        
        # Add stretch to push everything to the top
        layout.addStretch()
        
        return panel
    
    def create_plot_panel(self):
        """Create the plotting panel with tabs"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Create tab widget for multiple plots
        self.tab_widget = QTabWidget()
        
        # Connect tab change to update settings
        self.tab_widget.currentChanged.connect(self.on_tab_changed)
        
        # Tab 1: SED Plot
        self.sed_canvas = MplCanvas(self, width=8, height=6)
        self.sed_toolbar = NavigationToolbar(self.sed_canvas, self)
        sed_widget = QWidget()
        sed_layout = QVBoxLayout(sed_widget)
        sed_layout.addWidget(self.sed_toolbar)
        sed_layout.addWidget(self.sed_canvas)
        self.tab_widget.addTab(sed_widget, 'SED')
        
        # Tab 2: Temperature Plot
        self.temp_canvas = MplCanvas(self, width=8, height=6)
        self.temp_toolbar = NavigationToolbar(self.temp_canvas, self)
        temp_widget = QWidget()
        temp_layout = QVBoxLayout(temp_widget)
        temp_layout.addWidget(self.temp_toolbar)
        temp_layout.addWidget(self.temp_canvas)
        self.tab_widget.addTab(temp_widget, 'Temperature')
        
        # Tab 3: Density Plot
        self.density_canvas = MplCanvas(self, width=8, height=6)
        self.density_toolbar = NavigationToolbar(self.density_canvas, self)
        density_widget = QWidget()
        density_layout = QVBoxLayout(density_widget)
        density_layout.addWidget(self.density_toolbar)
        density_layout.addWidget(self.density_canvas)
        self.tab_widget.addTab(density_widget, 'Density')

        # Tab 4: FITS Image Plot
        self.image_canvas = MplCanvas(self, width=8, height=6)
        self.image_toolbar = NavigationToolbar(self.image_canvas, self)
        image_widget = QWidget()
        image_layout = QVBoxLayout(image_widget)
        image_layout.addWidget(self.image_toolbar)
        image_layout.addWidget(self.image_canvas)
        self.tab_widget.addTab(image_widget, 'FITS Image')

        # Tab 5: Parameters Table (NEW)
        self.param_table = QTableWidget()
        self.param_table.setColumnCount(2)
        self.param_table.setHorizontalHeaderLabels(['Parameter', 'Value'])
        header = self.param_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        self.param_table.setEditTriggers(QAbstractItemView.NoEditTriggers) # Read only
        self.tab_widget.addTab(self.param_table, 'Parameters')
        
        layout.addWidget(self.tab_widget)
        
        return panel
    
    def on_tab_changed(self, index):
        """Handle tab change to show appropriate settings"""
        # Hide all settings first
        self.sed_settings.hide()
        self.temp_settings.hide()
        self.density_settings.hide()
        self.image_settings.hide()
        self.param_settings.hide()
        
        # Show settings for current tab
        if index == 0:  # SED
            self.sed_settings.show()
        elif index == 1:  # Temperature
            self.temp_settings.show()
        elif index == 2:  # Density
            self.density_settings.show()
        elif index == 3:  # FITS Image
            self.image_settings.show()
        elif index == 4: # Parameters
            self.param_settings.show()
    
    # === Data Loading Methods ===
    
    def browse_run_directory(self):
        """Open file dialog to select run directory"""
        directory = QFileDialog.getExistingDirectory(
            self, 
            'Select Run Directory',
            os.path.expanduser('~'),
            QFileDialog.ShowDirsOnly
        )
        if directory:
            self.load_run_directory(directory)
    
    def load_from_quick_path(self):
        """Load run from quick path input"""
        path = self.quick_path_input.text().strip()
        if path and os.path.isdir(path):
            self.load_run_directory(path)
        else:
            QMessageBox.warning(self, 'Invalid Path', 'The specified path does not exist.')
    
    def load_from_recent(self, item):
        """Load run from recent runs list"""
        path = item.text()
        if os.path.isdir(path):
            self.load_run_directory(path)
    
    def load_run_directory(self, directory):
        """Load RADMC-3D data from a run directory"""
        self.current_run_dir = directory
        self.current_run_label.setText(f'Current: {directory}')
        self.statusBar().showMessage(f'Loading data from {directory}...')
        
        # Add to recent runs
        if self.recent_runs_list.count() == 0 or self.recent_runs_list.item(0).text() != directory:
            self.recent_runs_list.insertItem(0, directory)
            if self.recent_runs_list.count() > 10:
                self.recent_runs_list.takeItem(10)
        
        try:
            # Change to run directory
            old_dir = os.getcwd()
            os.chdir(directory)
            
            # Load spectrum if available
            if os.path.exists('spectrum.out'):
                if RADMC3D_AVAILABLE:
                    self.spectrum = analyze.readSpectrum(fname='spectrum.out')
                    self.info_text.append('✓ Loaded spectrum.out')
                else:
                    self.info_text.append('✗ Cannot load spectrum (radmc3dPy not available)')
            
            # Load grid if available
            if os.path.exists('amr_grid.inp'):
                if RADMC3D_AVAILABLE:
                    self.grid = analyze.readGrid()
                    self.info_text.append('✓ Loaded grid')
                else:
                    self.info_text.append('✗ Cannot load grid (radmc3dPy not available)')
            
            # Load star if available
            if os.path.exists('stars.inp'):
                if RADMC3D_AVAILABLE:
                    self.star = analyze.readStars()
                    self.info_text.append('✓ Loaded stellar parameters')
                else:
                    self.info_text.append('✗ Cannot load stars (radmc3dPy not available)')
            
            # Scan for FITS files
            self.scan_fits_files()
            
            # Load Parameters to Table (NEW)
            self.load_parameters_to_table()

            os.chdir(old_dir)
            self.statusBar().showMessage(f'Successfully loaded: {directory}')
            
        except Exception as e:
            os.chdir(old_dir)
            self.statusBar().showMessage(f'Error loading data: {str(e)}')
            QMessageBox.critical(self, 'Error', f'Failed to load data:\n{str(e)}')

    def scan_fits_files(self):
        """Scan directory for .fits files"""
        if self.current_run_dir is None:
            return

        self.fits_file_combo.clear()
        try:
            # Find all .fits files
            fits_files = [f for f in os.listdir(self.current_run_dir) if f.endswith('.fits')]
            fits_files.sort()
            
            if fits_files:
                self.fits_file_combo.addItems(fits_files)
                self.info_text.append(f'✓ Found {len(fits_files)} FITS files')
            else:
                self.fits_file_combo.addItem("No .fits files found")
                self.info_text.append('! No .fits files found in directory')
                
        except Exception as e:
            print(f"Error scanning fits: {e}")
            
    def load_parameters_to_table(self):
        """Reads problem_params_*.inp and populates the table tab"""
        self.param_table.setRowCount(0) # Clear table
        
        # Find file
        params_files = glob.glob(os.path.join(self.current_run_dir, 'problem_params_*.inp'))
        # Fallback
        if not params_files and os.path.exists(os.path.join(self.current_run_dir, 'problem_params.inp')):
             params_files = [os.path.join(self.current_run_dir, 'problem_params.inp')]
             
        if not params_files:
            return

        params_filepath = params_files[0]
        
        # Read and parse
        params = []
        try:
            with open(params_filepath, 'r') as f:
                for line in f:
                    if line.strip().startswith('#') or '=' not in line:
                        continue
                    parts = line.split('=', 1)
                    key = parts[0].strip()
                    # Remove inline comments
                    val = parts[1].split('#')[0].strip()
                    params.append((key, val))
        except Exception as e:
            print(f"Error reading params: {e}")
            return
            
        # Sort or define priority
        # Let's put important ones top, others below
        priority_keys = ['mdisk', 'rdisk', 'rin', 'incl', 'dustkappa', 'nphot', 'bgdens']
        sorted_params = []
        
        # Add priority items first
        for p_key in priority_keys:
            for k, v in params:
                if k == p_key:
                    sorted_params.append((k, v))
        
        # Add the rest
        for k, v in params:
            if k not in priority_keys:
                sorted_params.append((k, v))
                
        # Fill Table
        self.param_table.setRowCount(len(sorted_params))
        for row, (key, val) in enumerate(sorted_params):
            self.param_table.setItem(row, 0, QTableWidgetItem(key))
            self.param_table.setItem(row, 1, QTableWidgetItem(val))
            
            # Highlight priority keys
            if key in priority_keys:
                self.param_table.item(row, 0).setBackground(QColor('#e6f3ff'))
                self.param_table.item(row, 1).setBackground(QColor('#e6f3ff'))
    
    # === Plotting Methods ===
    
    def plot_sed(self):
        """Plot the SED using the actual plots.py implementation"""
        if self.spectrum is None:
            QMessageBox.warning(self, 'No Data', 'Please load a run directory first.')
            return
        
        if self.star is None or self.grid is None:
            QMessageBox.warning(self, 'Incomplete Data', 
                              'Need spectrum, star, and grid data for SED plot.')
            return
        
        ax = self.sed_canvas.axes
        ax.clear()
        
        try:
            # Get distance in parsec (you can make this configurable later)
            pc = 140.0  # Default distance for AB Aur
            
            # Apply wavelength range
            wl_min = self.wl_min_spin.value()
            wl_max = self.wl_max_spin.value()
            
            # Plot reference SED if checkbox is checked
            if self.show_reference_cb.isChecked():
                reference_file = self.reference_combo.currentText()
                if reference_file and reference_file != "None":
                    try:
                        ref_data = np.loadtxt(reference_file)
                        ref_wav = ref_data[:, 0]
                        ref_flux = ref_data[:, 1]
                        ax.loglog(ref_wav, ref_flux, 'g--', linewidth=1.5, 
                                 label=os.path.basename(reference_file), alpha=0.7)
                    except Exception as e:
                        print(f"Could not load reference SED: {e}")
            
            # Extract spectrum data
            if isinstance(self.spectrum, np.ndarray):
                if len(self.spectrum.shape) == 2 and self.spectrum.shape[1] == 2:
                    wav = self.spectrum[:, 0]
                    fnu = self.spectrum[:, 1]
                else:
                    wav = self.grid.wav
                    if len(self.spectrum.shape) > 1:
                        fnu = self.spectrum[:, 0]
                    else:
                        fnu = self.spectrum
            elif hasattr(self.spectrum, 'wav') and hasattr(self.spectrum, 'fnu'):
                wav = self.spectrum.wav
                fnu = self.spectrum.fnu
                if len(fnu.shape) > 1:
                    fnu = fnu[:, 0]
            else:
                raise ValueError(f"Unexpected spectrum structure: {type(self.spectrum)}")
            
            # Make sure arrays have same length
            if len(wav) != len(fnu):
                min_len = min(len(wav), len(fnu))
                wav = wav[:min_len]
                fnu = fnu[:min_len]
            
            # Calculate nu*F_nu for the disk
            freq = 2.99792458e14 / wav  # c/lambda in Hz (c in micron/s)
            nufnu_disk = freq * fnu / (pc**2)  # Scale by distance squared
            
            # Plot disk SED
            ax.loglog(wav, nufnu_disk, 'b-', linewidth=2, label='Disk')
            
            # Plot stellar contribution
            try:
                flux_star = self.star.fnustar / (pc**2)
                if len(flux_star.shape) > 1:
                    flux_star = flux_star.flatten()
                flux_star = np.reshape(flux_star, len(self.grid.wav))
                nufnu_star = self.grid.freq * flux_star
                ax.loglog(self.grid.wav, nufnu_star, 
                         'r--', linewidth=2, label='Stellar contribution', alpha=0.7)
            except Exception as e:
                print(f"Could not plot stellar contribution: {e}")
            
            ax.set_xlabel(r'$\lambda$ [$\mu$m]', fontsize=12)
            ax.set_ylabel(r'$\nu F_{\nu}$ [erg s$^{-1}$ cm$^{-2}$]', fontsize=12)
            ax.set_title('Spectral Energy Distribution', fontsize=14, fontweight='bold')
            ax.set_xlim(wl_min, wl_max)
            ax.set_ylim(1e-15, 1e-6)
            ax.legend(loc='lower left')
            ax.grid(True, alpha=0.3, which='both')
            
            self.sed_canvas.draw()
            self.statusBar().showMessage('SED plotted successfully')
            
        except Exception as e:
            QMessageBox.critical(self, 'Error', f'Failed to plot SED:\n{str(e)}')
            import traceback
            traceback.print_exc()
    
    def plot_temperature(self):
        """Plot temperature distribution using plots.py implementation"""
        if self.current_run_dir is None:
            QMessageBox.warning(self, 'No Data', 'Please load a run directory first.')
            return
        
        ax = self.temp_canvas.axes
        ax.clear()
        self.temp_canvas.fig.clear()
        ax = self.temp_canvas.fig.add_subplot(111)
        self.temp_canvas.axes = ax
        
        if not RADMC3D_AVAILABLE:
            ax.text(0.5, 0.5, 'radmc3dPy required for temperature plots',
                   ha='center', va='center', transform=ax.transAxes)
            self.temp_canvas.draw()
            return
        
        try:
            old_dir = os.getcwd()
            os.chdir(self.current_run_dir)
            self.statusBar().showMessage('Loading temperature data...')
            
            data = analyze.readData(dtemp=True, binary=True)
            from radmc3dPy import natconst
            cmap = self.temp_cmap_combo.currentText()
            
            c = ax.contourf(
                data.grid.x / natconst.au,
                np.pi / 2. - data.grid.y,
                data.dusttemp[:, :, 0, 0].T,
                30, cmap=cmap
            )
            
            c_lines = ax.contour(
                data.grid.x / natconst.au,
                np.pi / 2. - data.grid.y,
                data.dusttemp[:, :, 0, 0].T,
                10, colors='k', linestyles='solid', linewidths=0.5
            )
            ax.clabel(c_lines, inline=1, fontsize=8)
            
            ax.set_xlabel('r [AU]', fontsize=12)
            ax.set_ylabel(r'$\pi/2-\theta$', fontsize=12)
            ax.set_xscale('log')
            ax.set_title('Dust Temperature Contours', fontsize=14, fontweight='bold')
            
            cbar = self.temp_canvas.fig.colorbar(c, ax=ax)
            cbar.set_label('T [K]', rotation=270, labelpad=20, fontsize=12)
            
            os.chdir(old_dir)
            self.temp_canvas.draw()
            self.statusBar().showMessage('Temperature plot created')
            
        except Exception as e:
            os.chdir(old_dir)
            QMessageBox.critical(self, 'Error', f'Failed to plot Temp:\n{str(e)}')

    def plot_density(self):
        """Plot density distribution with optional tau=1 surface"""
        if self.current_run_dir is None:
            QMessageBox.warning(self, 'No Data', 'Please load a run directory first.')
            return
        
        ax = self.density_canvas.axes
        ax.clear()
        self.density_canvas.fig.clear()
        ax = self.density_canvas.fig.add_subplot(111)
        self.density_canvas.axes = ax
        
        if not RADMC3D_AVAILABLE:
            ax.text(0.5, 0.5, 'radmc3dPy required for density plots',
                   ha='center', va='center', transform=ax.transAxes)
            self.density_canvas.draw()
            return
        
        try:
            old_dir = os.getcwd()
            os.chdir(self.current_run_dir)
            self.statusBar().showMessage('Loading density data...')
            
            data = analyze.readData(dtemp=True, ddens=True, binary=True)
            show_tau = self.show_tau_cb.isChecked()
            tau_direction = self.tau_direction_combo.currentText()
            use_tauy = 'tauy' in tau_direction.lower() or 'vertical' in tau_direction.lower()
            tau_wavelength = self.tau_wavelength_spin.value()
            
            has_tau = False
            if show_tau:
                try:
                    # Dynamically find the correct dustkappa file extension
                    kappa_files = glob.glob('dustkappa_*.inp')
                    
                    if not kappa_files:
                        print("Warning: No dustkappa_*.inp file found, skipping tau calc.")
                    else:
                        # Extract name
                        kappa_ext = kappa_files[0].replace('dustkappa_', '').replace('.inp', '')
                        opac = analyze.readOpac(ext=[kappa_ext])
                        
                        data.getTau(wav=tau_wavelength)
                        has_tau = True
                        self.statusBar().showMessage(f'Calculating τ at {tau_wavelength} µm...')
                except Exception as e:
                    print(f"Warning: Could not calculate tau: {e}")
                    self.statusBar().showMessage(f'Could not calculate τ: {e}')
            
            from radmc3dPy import natconst
            cmap = self.dens_cmap_combo.currentText()
            
            c = ax.contourf(
                data.grid.x / natconst.au,
                np.pi / 2. - data.grid.y,
                np.log10(data.rhodust[:, :, 0, 0].T),
                30, cmap=cmap
            )
            
            ax.set_xlabel('r [AU]', fontsize=12)
            ax.set_ylabel(r'$\pi/2-\theta$', fontsize=12)
            ax.set_xscale('log')
            ax.set_title('Dust Density Contours', fontsize=14, fontweight='bold')
            
            cbar = self.density_canvas.fig.colorbar(c, ax=ax)
            cbar.set_label(r'$\log_{10}(\rho)$ [g/cm$^3$]', rotation=270, labelpad=20, fontsize=12)
            
            if show_tau and has_tau:
                if use_tauy and hasattr(data, 'tauy'):
                    theta_midplane_idx = np.argmin(np.abs(data.grid.y - np.pi/2))
                    tau_data = data.tauy.copy()
                    for i in range(theta_midplane_idx):
                        mirror_idx = 2 * theta_midplane_idx - i
                        if mirror_idx < tau_data.shape[1]:
                            tau_data[:, mirror_idx, :] = data.tauy[:, i, :]
                    
                    c_tau = ax.contour(
                        data.grid.x / natconst.au,
                        np.pi / 2. - data.grid.y,
                        tau_data[:, :, 0].T,
                        [1.0], colors='white', linestyles='solid', linewidths=2
                    )
                    ax.clabel(c_tau, inline=1, fontsize=10, fmt=r'$\tau_y=1$')
                    
                elif hasattr(data, 'taux'):
                    c_tau = ax.contour(
                        data.grid.x / natconst.au,
                        np.pi / 2. - data.grid.y,
                        data.taux[:, :, 0].T,
                        [1.0], colors='white', linestyles='solid', linewidths=2
                    )
                    ax.clabel(c_tau, inline=1, fontsize=10, fmt=r'$\tau_x=1$')
            
            os.chdir(old_dir)
            self.density_canvas.draw()
            self.statusBar().showMessage('Density plot created')
            
        except Exception as e:
            os.chdir(old_dir)
            QMessageBox.critical(self, 'Error', f'Failed to plot Density:\n{str(e)}')

    def plot_fits_image(self):
        """Plot the selected FITS file"""
        if not ASTROPY_AVAILABLE:
            QMessageBox.warning(self, 'Error', 'Astropy is not installed. Cannot load FITS.')
            return

        if self.current_run_dir is None:
            QMessageBox.warning(self, 'No Data', 'Please load a run directory first.')
            return

        filename = self.fits_file_combo.currentText()
        if not filename or filename == "No .fits files found":
            QMessageBox.warning(self, 'No File', 'Please select a valid FITS file.')
            return

        filepath = os.path.join(self.current_run_dir, filename)
    
        ax = self.image_canvas.axes
        ax.clear()
        self.image_canvas.fig.clear()
        ax = self.image_canvas.fig.add_subplot(111)
        self.image_canvas.axes = ax

        try:
            hdul = fits.open(filepath)
            data = hdul[0].data
            header = hdul[0].header
            hdul.close()

            # Handle Dimensions
            plot_data = data
            slice_idx = self.slice_spin.value()
        
            # --- Calculate wavelength for this slice ---
            wav_micron = None
            if 'CRVAL3' in header and 'CDELT3' in header:
                # CRVAL3 is the reference wavelength, CDELT3 is the increment
                crval3 = header['CRVAL3']
                cdelt3 = header['CDELT3']
                crpix3 = header.get('CRPIX3', 1.0)  # Reference pixel, usually 1
                # Wavelength = CRVAL3 + (slice_idx - CRPIX3 + 1) * CDELT3
                wav_micron = crval3 + (slice_idx - crpix3 + 1) * cdelt3
            elif 'WAVELEN' in header:
                wav_micron = header['WAVELEN']
            elif 'WAVE' in header:
                wav_micron = header['WAVE']
            
            # Peel off extra dimensions
            while len(plot_data.shape) > 2:
                if plot_data.shape[0] > slice_idx:
                    plot_data = plot_data[slice_idx] 
                else:
                    plot_data = plot_data[0]
            
            # Format Title
            if wav_micron is not None:
                try:
                    wav_val = float(wav_micron)
                    title_text = f'Image at {wav_val:.3f} µm (slice {slice_idx})'
                except ValueError:
                    title_text = f'{filename} (slice {slice_idx})'
            else:
                title_text = f'{filename} (slice {slice_idx}, wavelength unknown)'
            
            # Apply Log Scale
            if self.img_log_cb.isChecked():
                plot_data = np.abs(plot_data)
                valid_min = np.min(plot_data[plot_data > 0]) if np.any(plot_data > 0) else 1e-20
                plot_data = np.maximum(plot_data, valid_min)
                plot_data = np.log10(plot_data)
                label_prefix = r'$\log_{10}$ '
            else:
                label_prefix = ''
    
            cmap = self.img_cmap_combo.currentText()
            
            # Plot
            im = ax.imshow(plot_data, origin='lower', cmap=cmap)
            
            # Colorbar
            cbar = self.image_canvas.fig.colorbar(im, ax=ax)
            cbar.set_label(f'{label_prefix}Intensity', rotation=270, labelpad=20)
            
            ax.set_title(title_text, fontsize=12, fontweight='bold')
            ax.set_xlabel('Pixels (X)')
            ax.set_ylabel('Pixels (Y)')
            
            self.image_canvas.draw()
            self.statusBar().showMessage(f'Plotted {filename}')
    
        except Exception as e:
            QMessageBox.critical(self, 'Error', f'Failed to plot FITS:\n{str(e)}')
            import traceback
            traceback.print_exc()

    def show_run_info(self):
        """Display information about the current run"""
        if self.current_run_dir is None:
            QMessageBox.information(self, 'No Run Loaded', 'Please load a run directory first.')
            return
        
        info = f"Run Directory: {self.current_run_dir}\n\n"
        
        files_to_check = [
            'spectrum.out',
            'dust_temperature.bdat',
            'dust_density.binp',
            'amr_grid.inp',
            'radmc3d.inp',
            'stars.inp',
        ]
        
        info += "Available files:\n"
        for fname in files_to_check:
            fpath = os.path.join(self.current_run_dir, fname)
            if os.path.exists(fpath):
                size_mb = os.path.getsize(fpath) / (1024**2)
                info += f"  ✓ {fname} ({size_mb:.2f} MB)\n"
            else:
                info += f"  ✗ {fname}\n"
        
        self.info_text.clear()
        self.info_text.setText(info)


def main():
    """Main entry point"""
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = RADMCAnalysisGUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()