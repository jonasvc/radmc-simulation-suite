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

# --- Interferometry Import ---
try:
    from interferometry import (calculate_visibilities, calculate_visibilities_custom, 
                                   calculate_closure_phases,
                                   plot_uv_coverage, plot_visibility_amplitude,
                                   plot_closure_phases)
    INTERFEROMETRY_AVAILABLE = True
except ImportError:
    INTERFEROMETRY_AVAILABLE = False
    print("Warning: interferometry_v2 module not found. Interferometry tab disabled.")


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
        self.vis_data = None
        self.cp_data = None
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
        
        # === File Info Button ===
        show_info_btn = QPushButton('Show File Overview')
        show_info_btn.clicked.connect(self.show_run_info)
        layout.addWidget(show_info_btn)
        
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
        
        # Plot button
        plot_sed_btn = QPushButton('📊 Plot SED')
        plot_sed_btn.clicked.connect(self.plot_sed)
        sed_settings_layout.addWidget(plot_sed_btn)
        
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
        
        # Plot button
        plot_temp_btn = QPushButton('🌡️ Plot Temperature')
        plot_temp_btn.clicked.connect(self.plot_temperature)
        temp_settings_layout.addWidget(plot_temp_btn)
        
        # --- 3. Density Settings ---
        self.density_settings = QWidget()
        density_settings_layout = QVBoxLayout(self.density_settings)
        
        density_settings_layout.addWidget(QLabel('Density plot settings:'))
        
        # Plot Type Selection
        plot_type_layout = QHBoxLayout()
        plot_type_layout.addWidget(QLabel('Plot type:'))
        self.density_plot_type_combo = QComboBox()
        self.density_plot_type_combo.addItems(['Standard 2D', 'Multi-Azimuth (2×3)', 'Single Azimuth'])
        plot_type_layout.addWidget(self.density_plot_type_combo)
        density_settings_layout.addLayout(plot_type_layout)
        
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
        
        # --- Azimuthal angle for Single Azimuth plot ---
        azimuth_layout = QHBoxLayout()
        azimuth_layout.addWidget(QLabel('Azimuthal angle [rad]:'))
        self.azimuth_angle_spin = QDoubleSpinBox()
        self.azimuth_angle_spin.setRange(0, 2*np.pi)
        self.azimuth_angle_spin.setValue(np.pi/2)
        self.azimuth_angle_spin.setDecimals(3)
        self.azimuth_angle_spin.setSingleStep(0.1)
        azimuth_layout.addWidget(self.azimuth_angle_spin)
        density_settings_layout.addLayout(azimuth_layout)
        
        # --- Plot Limits ---
        density_settings_layout.addWidget(QLabel('Plot limits:'))
        
        # R limits
        r_limits_layout = QHBoxLayout()
        r_limits_layout.addWidget(QLabel('R [AU]:'))
        self.r_min_spin = QDoubleSpinBox()
        self.r_min_spin.setRange(0.01, 1000)
        self.r_min_spin.setValue(0.5)
        self.r_min_spin.setDecimals(2)
        r_limits_layout.addWidget(self.r_min_spin)
        r_limits_layout.addWidget(QLabel('to'))
        self.r_max_spin = QDoubleSpinBox()
        self.r_max_spin.setRange(0.01, 10000)
        self.r_max_spin.setValue(200)
        self.r_max_spin.setDecimals(1)
        r_limits_layout.addWidget(self.r_max_spin)
        density_settings_layout.addLayout(r_limits_layout)
        
        # Theta limits
        theta_limits_layout = QHBoxLayout()
        theta_limits_layout.addWidget(QLabel('π/2-θ:'))
        self.theta_min_spin = QDoubleSpinBox()
        self.theta_min_spin.setRange(-1, 1)
        self.theta_min_spin.setValue(-0.5)
        self.theta_min_spin.setDecimals(2)
        theta_limits_layout.addWidget(self.theta_min_spin)
        theta_limits_layout.addWidget(QLabel('to'))
        self.theta_max_spin = QDoubleSpinBox()
        self.theta_max_spin.setRange(-1, 1)
        self.theta_max_spin.setValue(0.5)
        self.theta_max_spin.setDecimals(2)
        theta_limits_layout.addWidget(self.theta_max_spin)
        density_settings_layout.addLayout(theta_limits_layout)
        
        # Plot button
        plot_density_btn = QPushButton('Plot Density')
        plot_density_btn.clicked.connect(self.plot_density)
        density_settings_layout.addWidget(plot_density_btn)

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
        
        # Radial profile normalization
        self.normalize_radial_cb = QCheckBox('Normalize radial profile')
        self.normalize_radial_cb.setChecked(False)
        image_settings_layout.addWidget(self.normalize_radial_cb)
        
        # Plot buttons
        plot_image_btn = QPushButton('🖼️ Plot FITS Image')
        plot_image_btn.clicked.connect(self.plot_fits_image)
        image_settings_layout.addWidget(plot_image_btn)
        
        plot_radial_btn = QPushButton('📈 Plot Radial Profile')
        plot_radial_btn.clicked.connect(self.plot_radial_profile_simple)
        image_settings_layout.addWidget(plot_radial_btn)

        # --- 5. Interferometry Settings ---
        self.interfero_settings = QWidget()
        interfero_settings_layout = QVBoxLayout(self.interfero_settings)
        
        interfero_settings_layout.addWidget(QLabel('Interferometry settings:'))
        
        # Distance in parsecs with Data button in middle
        dpc_layout = QHBoxLayout()
        dpc_layout.addWidget(QLabel('Distance [pc]:'))
        self.dpc_input = QLineEdit()
        self.dpc_input.setText('144')  # Default for AB Aur
        self.dpc_input.setMaximumWidth(80)
        dpc_layout.addWidget(self.dpc_input)
        
        dpc_layout.addStretch()  # Space before Data button
        
        show_interfero_data_btn = QPushButton('Data')
        show_interfero_data_btn.clicked.connect(self.show_interferometry_data)
        show_interfero_data_btn.setMaximumWidth(60)
        dpc_layout.addWidget(show_interfero_data_btn)
        
        dpc_layout.addStretch()  # Space after Data button
        
        interfero_settings_layout.addLayout(dpc_layout)
        
        # Baseline mode selection
        interfero_settings_layout.addWidget(QLabel('Baseline mode:'))
        self.baseline_mode_combo = QComboBox()
        self.baseline_mode_combo.addItems(['VLTI (all 26)', 'Custom baselines'])
        interfero_settings_layout.addWidget(self.baseline_mode_combo)
        
        # Custom baselines button (only visible when custom mode selected)
        self.custom_bl_btn = QPushButton('Edit Custom Baselines')
        self.custom_bl_btn.clicked.connect(self.edit_custom_baselines)
        self.custom_bl_btn.setVisible(False)
        self.baseline_mode_combo.currentIndexChanged.connect(
            lambda idx: self.custom_bl_btn.setVisible(idx == 1)
        )
        interfero_settings_layout.addWidget(self.custom_bl_btn)
        
        # Store custom baselines
        self.custom_baselines = []
        self.custom_pas = []
        self.custom_bl_names = []
        
        # Plot selection
        interfero_settings_layout.addWidget(QLabel('Select plot:'))
        self.interfero_plot_combo = QComboBox()
        self.interfero_plot_combo.addItems(['All 3 plots', 'UV Coverage only', 
                                            'Visibility² only', 'Closure Phase only'])
        self.interfero_plot_combo.currentIndexChanged.connect(self.replot_interferometry)
        interfero_settings_layout.addWidget(self.interfero_plot_combo)
        
        # Wavelength slice
        interfero_wl_layout = QHBoxLayout()
        interfero_wl_layout.addWidget(QLabel('Wavelength slice:'))
        self.interfero_wl_spin = QSpinBox()
        self.interfero_wl_spin.setRange(0, 100)
        self.interfero_wl_spin.setValue(0)
        interfero_wl_layout.addWidget(self.interfero_wl_spin)
        interfero_settings_layout.addLayout(interfero_wl_layout)
        
        self.use_all_wavelengths_cb = QCheckBox('Use all wavelengths')
        self.use_all_wavelengths_cb.setChecked(False)
        self.use_all_wavelengths_cb.stateChanged.connect(
            lambda state: self.interfero_wl_spin.setEnabled(state == 0)
        )
        interfero_settings_layout.addWidget(self.use_all_wavelengths_cb)
        
        self.wavelength_color_cb = QCheckBox('Color by wavelength')
        self.wavelength_color_cb.setChecked(True)
        self.wavelength_color_cb.stateChanged.connect(self.replot_interferometry)
        interfero_settings_layout.addWidget(self.wavelength_color_cb)
        
        self.show_negative_uv_cb = QCheckBox('Show -u,-v points')
        self.show_negative_uv_cb.setChecked(True)
        self.show_negative_uv_cb.stateChanged.connect(self.replot_interferometry)
        interfero_settings_layout.addWidget(self.show_negative_uv_cb)
        
        # Calculate button
        calc_interfero_btn = QPushButton('📡 Calculate Interferometry')
        calc_interfero_btn.clicked.connect(self.calculate_interferometry)
        interfero_settings_layout.addWidget(calc_interfero_btn)
        
        interfero_settings_layout.addStretch()

        # --- 6. Parameters Settings (Empty) ---
        self.param_settings = QWidget()
        self.param_settings.setLayout(QVBoxLayout())
        self.param_settings.layout().addWidget(QLabel("See Table in Right Panel ->"))
        
        # Add all setting widgets to stack
        settings_stack_layout.addWidget(self.sed_settings)
        settings_stack_layout.addWidget(self.temp_settings)
        settings_stack_layout.addWidget(self.density_settings)
        settings_stack_layout.addWidget(self.image_settings)
        settings_stack_layout.addWidget(self.interfero_settings)
        settings_stack_layout.addWidget(self.param_settings)
        
        # Initially show SED settings
        self.temp_settings.hide()
        self.density_settings.hide()
        self.image_settings.hide()
        self.interfero_settings.hide()
        self.param_settings.hide()
        
        plot_layout.addWidget(self.settings_stack)
        plot_group.setLayout(plot_layout)
        layout.addWidget(plot_group)
        
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

        # Tab 5: Interferometry
        self.interfero_canvas = MplCanvas(self, width=12, height=12)
        self.interfero_toolbar = NavigationToolbar(self.interfero_canvas, self)
        interfero_widget = QWidget()
        interfero_layout = QVBoxLayout(interfero_widget)
        interfero_layout.addWidget(self.interfero_toolbar)
        interfero_layout.addWidget(self.interfero_canvas)
        self.tab_widget.addTab(interfero_widget, 'Interferometry')


        # Tab 6: Parameters Table (NEW)
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
        self.interfero_settings.hide()
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
        elif index == 4:  # Interferometry
            self.interfero_settings.show()
        elif index == 5:  # Parameters
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
        
        if not RADMC3D_AVAILABLE:
            ax = self.density_canvas.axes
            ax.clear()
            ax.text(0.5, 0.5, 'radmc3dPy required for density plots',
                   ha='center', va='center', transform=ax.transAxes)
            self.density_canvas.draw()
            return
        
        plot_type = self.density_plot_type_combo.currentText()
        
        if plot_type == 'Standard 2D':
            self._plot_density_standard()
        elif plot_type == 'Multi-Azimuth (2×3)':
            self._plot_density_multi_azimuth()
        elif plot_type == 'Single Azimuth':
            self._plot_density_single_azimuth()
    
    def _plot_density_standard(self):
        """Standard 2D density plot (original functionality)"""
        ax = self.density_canvas.axes
        ax.clear()
        self.density_canvas.fig.clear()
        ax = self.density_canvas.fig.add_subplot(111)
        self.density_canvas.axes = ax
        
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
                    kappa_files = glob.glob('dustkappa_*.inp')
                    if not kappa_files:
                        print("Warning: No dustkappa_*.inp file found, skipping tau calc.")
                    else:
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
            ax.set_xlim(self.r_min_spin.value(), self.r_max_spin.value())
            ax.set_ylim(self.theta_min_spin.value(), self.theta_max_spin.value())
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
    
    def _plot_density_multi_azimuth(self):
        """Multi-azimuth plot (2x3 grid)"""
        self.density_canvas.fig.clear()
        
        try:
            old_dir = os.getcwd()
            os.chdir(self.current_run_dir)
            self.statusBar().showMessage('Loading density data for multi-azimuth plot...')
            
            data = analyze.readData(dtemp=True, ddens=True, binary=True)
            show_tau = self.show_tau_cb.isChecked()
            tau_direction = self.tau_direction_combo.currentText()
            use_tauy = 'tauy' in tau_direction.lower() or 'vertical' in tau_direction.lower()
            tau_wavelength = self.tau_wavelength_spin.value()
            
            # Calculate tau if needed
            has_tau = False
            if show_tau:
                try:
                    kappa_files = glob.glob('dustkappa_*.inp')
                    if kappa_files:
                        kappa_ext = kappa_files[0].replace('dustkappa_', '').replace('.inp', '')
                        opac = analyze.readOpac(ext=[kappa_ext])
                        data.getTau(wav=tau_wavelength)
                        has_tau = True
                except Exception as e:
                    print(f"Warning: Could not calculate tau: {e}")
            
            from radmc3dPy import natconst
            cmap = self.dens_cmap_combo.currentText()
            
            # Mirror tauy for symmetric representation
            theta_midplane_idx = np.argmin(np.abs(data.grid.y - np.pi/2))
            if has_tau and use_tauy and hasattr(data, 'tauy'):
                tauy_symmetric = data.tauy.copy()
                for i in range(theta_midplane_idx):
                    mirror_idx = 2 * theta_midplane_idx - i
                    if mirror_idx < tauy_symmetric.shape[1]:
                        tauy_symmetric[:, mirror_idx, :] = data.tauy[:, i, :]
            
            # Create 2x3 subplot grid
            axes = self.density_canvas.fig.subplots(2, 3)
            self.density_canvas.fig.subplots_adjust(hspace=0.35, wspace=0.4, left=0.08, right=0.92)
            
            # Select 6 azimuthal angles
            PHI_ANGLE_LIST = np.linspace(data.grid.z.min(), data.grid.z.max(), 6, endpoint=True)
            
            for i, PHI_ANGLE_RAD in enumerate(PHI_ANGLE_LIST):
                phi_index = np.argmin(np.abs(data.grid.z - PHI_ANGLE_RAD))
                row = i // 3
                col = i % 3
                ax = axes[row, col]
                
                print(f"Plotting at phi = {data.grid.z[phi_index]:.2f} rad")
                
                # Dust density contour
                c1 = ax.contourf(data.grid.x/natconst.au, np.pi/2.-data.grid.y,
                                 np.log10(data.rhodust[:,:,phi_index,0].T), 30, cmap=cmap)
                
                ax.set_title(r'$\phi = {:.2f}$ rad'.format(data.grid.z[phi_index]), fontsize=11, pad=8)
                ax.set_xlabel('r [AU]', fontsize=10)
                ax.set_ylabel(r'$\pi/2-\theta$', fontsize=10, labelpad=2)
                ax.set_xscale('log')
                ax.set_xlim(self.r_min_spin.value(), self.r_max_spin.value())
                ax.set_ylim(self.theta_min_spin.value(), self.theta_max_spin.value())
                ax.tick_params(labelsize=8)
                
                # Tau=1 contour if available
                if show_tau and has_tau and use_tauy:
                    c2 = ax.contour(data.grid.x/natconst.au, np.pi/2.-data.grid.y,
                                    tauy_symmetric[:,:,phi_index].T, [1.0],
                                    colors='w', linestyles='solid', linewidths=1.5)
                    ax.clabel(c2, inline=1, fontsize=8, fmt=r'$\tau=1$')
            
            # Add colorbar
            cb = self.density_canvas.fig.colorbar(c1, ax=axes.ravel().tolist(), 
                                                   orientation='vertical', fraction=0.02, pad=0.04)
            cb.set_label(r'$\log_{10}(\rho)$ [g/cm$^3$]', rotation=270, labelpad=20, fontsize=11)
            
            os.chdir(old_dir)
            self.density_canvas.draw()
            self.statusBar().showMessage('Multi-azimuth density plot created')
            
        except Exception as e:
            os.chdir(old_dir)
            QMessageBox.critical(self, 'Error', f'Failed to plot multi-azimuth density:\n{str(e)}')
    
    def _plot_density_single_azimuth(self):
        """Single azimuth plot at specified angle"""
        self.density_canvas.fig.clear()
        ax = self.density_canvas.fig.add_subplot(111)
        self.density_canvas.axes = ax
        
        try:
            old_dir = os.getcwd()
            os.chdir(self.current_run_dir)
            self.statusBar().showMessage('Loading density data for single azimuth plot...')
            
            data = analyze.readData(dtemp=True, ddens=True, binary=True)
            show_tau = self.show_tau_cb.isChecked()
            tau_direction = self.tau_direction_combo.currentText()
            use_tauy = 'tauy' in tau_direction.lower() or 'vertical' in tau_direction.lower()
            tau_wavelength = self.tau_wavelength_spin.value()
            PHI_ANGLE_RAD = self.azimuth_angle_spin.value()
            
            # Calculate tau if needed
            has_tau = False
            if show_tau:
                try:
                    kappa_files = glob.glob('dustkappa_*.inp')
                    if kappa_files:
                        kappa_ext = kappa_files[0].replace('dustkappa_', '').replace('.inp', '')
                        opac = analyze.readOpac(ext=[kappa_ext])
                        data.getTau(wav=tau_wavelength)
                        has_tau = True
                except Exception as e:
                    print(f"Warning: Could not calculate tau: {e}")
            
            from radmc3dPy import natconst
            cmap = self.dens_cmap_combo.currentText()
            
            # Find closest phi index
            phi_index = np.argmin(np.abs(data.grid.z - PHI_ANGLE_RAD))
            print(f"Single plot at phi = {data.grid.z[phi_index]:.2f} rad")
            
            # Mirror tauy for symmetric representation
            theta_midplane_idx = np.argmin(np.abs(data.grid.y - np.pi/2))
            if has_tau and use_tauy and hasattr(data, 'tauy'):
                tauy_symmetric = data.tauy.copy()
                for i in range(theta_midplane_idx):
                    mirror_idx = 2 * theta_midplane_idx - i
                    if mirror_idx < tauy_symmetric.shape[1]:
                        tauy_symmetric[:, mirror_idx, :] = data.tauy[:, i, :]
            
            # Plot density
            c1 = ax.contourf(data.grid.x/natconst.au, np.pi/2.-data.grid.y,
                            np.log10(data.rhodust[:,:,phi_index,0].T), 30, cmap=cmap)
            
            ax.set_title(r'Dust density with $\tau=1$ at $\phi = {:.2f}$ rad'.format(data.grid.z[phi_index]),
                        fontsize=12, fontweight='bold')
            ax.set_xlabel('r [AU]', fontsize=12)
            ax.set_ylabel(r'$\pi/2-\theta$', fontsize=12)
            ax.set_xscale('log')
            ax.set_xlim(self.r_min_spin.value(), self.r_max_spin.value())
            ax.set_ylim(self.theta_min_spin.value(), self.theta_max_spin.value())
            
            cb = self.density_canvas.fig.colorbar(c1, ax=ax)
            cb.set_label(r'$\log_{10}{\rho}$', rotation=270, labelpad=15)
            
            # Tau=1 contour if available
            if show_tau and has_tau and use_tauy:
                c2 = ax.contour(data.grid.x/natconst.au, np.pi/2.-data.grid.y,
                               tauy_symmetric[:,:,phi_index].T, [1.0],
                               colors='w', linestyles='solid', linewidths=2)
                ax.clabel(c2, inline=1, fontsize=10)
            
            os.chdir(old_dir)
            self.density_canvas.draw()
            self.statusBar().showMessage(f'Single azimuth density plot created at φ={PHI_ANGLE_RAD:.2f} rad')
            
        except Exception as e:
            os.chdir(old_dir)
            QMessageBox.critical(self, 'Error', f'Failed to plot single azimuth density:\n{str(e)}')

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

            # --- Try to find wavelength for title ---
            # Try multiple common keywords used in FITS headers
            wav_micron = None
            if 'WAVELEN' in header:
                wav_micron = header['CRVAL3']
            elif 'WAVE' in header:
                wav_micron = header['WAVE']
            elif 'RESTWAV' in header:
                wav_micron = header['RESTWAV']
            
            # Format Title
            if wav_micron is not None:
                try:
                    wav_val = float(wav_micron)
                    title_text = f'Image at {wav_val:.2f} micron'
                except ValueError:
                    title_text = filename
            else:
                title_text = "Image at" + " (Wavelen unknown)"
                print("DEBUG HEADER KEYS:", list(header.keys())) # If it fails, check console

            # Handle Dimensions
            plot_data = data
            slice_idx = self.slice_spin.value()
            
            # Peel off extra dimensions
            while len(plot_data.shape) > 2:
                if plot_data.shape[0] > slice_idx:
                    plot_data = plot_data[slice_idx] 
                else:
                    plot_data = plot_data[0] # Default to 0
            
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

    def plot_radial_profile_simple(self):
        """Plot X and Y axis profiles through center (like the notebook code)"""
        if self.current_run_dir is None:
            QMessageBox.warning(self, 'No Data', 'Please load a run directory first.')
            return
        
        # Get FITS file
        filename = self.fits_file_combo.currentText()
        if not filename or filename == "No .fits files found":
            QMessageBox.warning(self, 'No FITS', 'Please select a valid FITS file.')
            return
        
        filepath = os.path.join(self.current_run_dir, filename)
        
        try:
            with fits.open(filepath) as hdul:
                data = hdul[0].data
                header = hdul[0].header
            
            # Get wavelength slice
            wavelength_idx = self.slice_spin.value()
            
            # Handle cube vs single image (squeeze if needed)
            while data.ndim > 2:
                if wavelength_idx < data.shape[0]:
                    data = data[wavelength_idx]
                else:
                    data = data[0]
            
            # Get pixel scale in mas
            pixel_scale = None
            unit = None
            
            # Check CTYPE for unit
            if 'CTYPE1' in header:
                ctype = header['CTYPE1'].lower()
                if 'arcsec' in ctype:
                    unit = 'arcsec'
                elif 'mas' in ctype:
                    unit = 'mas'
                elif 'deg' in ctype:
                    unit = 'deg'
            
            # Check CUNIT1
            if unit is None and 'CUNIT1' in header:
                cunit = header['CUNIT1'].lower()
                if 'arcsec' in cunit or 'as' in cunit:
                    unit = 'arcsec'
                elif 'mas' in cunit:
                    unit = 'mas'
                elif 'deg' in cunit:
                    unit = 'deg'
            
            # Get pixel scale value
            for key in ['CDELT1', 'CD1_1', 'PXSCALE']:
                if key in header:
                    pixel_scale_raw = abs(header[key])
                    
                    # Convert to mas based on unit
                    if unit == 'arcsec':
                        pixel_scale = pixel_scale_raw * 1000  # arcsec -> mas
                    elif unit == 'mas':
                        pixel_scale = pixel_scale_raw  # already mas
                    elif unit == 'deg':
                        pixel_scale = pixel_scale_raw * 3600 * 1000  # deg -> mas
                    else:
                        # Guess based on magnitude
                        if pixel_scale_raw > 1:
                            pixel_scale = pixel_scale_raw  # assume mas
                        elif pixel_scale_raw > 0.001:
                            pixel_scale = pixel_scale_raw * 1000  # assume arcsec
                        else:
                            pixel_scale = pixel_scale_raw * 3600 * 1000  # assume deg
                    break
            
            if pixel_scale is None:
                pixel_scale = 1.0
            
            # Find center of image
            ny, nx = data.shape
            cx, cy = nx // 2, ny // 2
            
            # Extract X and Y axis profiles through center
            x_profile = data[cy, :]
            y_profile = data[:, cx]
            
            # Normalize if checkbox is checked
            if self.normalize_radial_cb.isChecked():
                # Normalize to peak = 1
                x_max = np.max(x_profile[x_profile > 0]) if np.any(x_profile > 0) else 1.0
                y_max = np.max(y_profile[y_profile > 0]) if np.any(y_profile > 0) else 1.0
                x_profile = x_profile / x_max
                y_profile = y_profile / y_max
                ylabel = 'Normalized Intensity'
                title_suffix = ' (normalized)'
            else:
                ylabel = 'Intensity'
                title_suffix = ''
            
            # Create separation arrays in mas
            x_sep = (np.arange(nx) - cx) * pixel_scale
            y_sep = (np.arange(ny) - cy) * pixel_scale
            
            # Plot on image canvas
            self.image_canvas.fig.clear()
            ax = self.image_canvas.fig.add_subplot(111)
            
            # Plot X and Y profiles
            ax.plot(x_sep, x_profile, 'b-', label='X-axis', linewidth=1.5)
            ax.plot(y_sep, y_profile, 'r-', label='Y-axis', linewidth=1.5)
            
            # Formatting
            ax.set_xlabel('Separation [mas]', fontsize=12)
            ax.set_ylabel(ylabel, fontsize=12)
            ax.set_title(f'Total intensity{title_suffix}', fontsize=13)
            ax.set_yscale('log')
            ax.set_xscale('symlog', linthresh=1)  # Symmetric log scale
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3)
            
            # Set y-limits
            if not self.normalize_radial_cb.isChecked():
                valid_data = data[data > 0]
                if len(valid_data) > 0:
                    ax.set_ylim(valid_data.min() * 0.5, valid_data.max() * 2)
            
            self.image_canvas.fig.tight_layout()
            self.image_canvas.draw()
            
            # Stay on FITS Image tab (already there)
            
            self.statusBar().showMessage('✅ Radial profile plotted (X and Y cuts)')
            
        except Exception as e:
            QMessageBox.critical(self, 'Error', f'Failed to plot radial profile:\n{str(e)}')
            import traceback
            traceback.print_exc()

    def edit_custom_baselines(self):
        """Dialog to edit custom baselines"""
        dialog = QDialog(self)
        dialog.setWindowTitle('Custom Baselines')
        dialog.resize(600, 400)
        
        layout = QVBoxLayout(dialog)
        
        layout.addWidget(QLabel('Enter custom baselines (one per line):'))
        layout.addWidget(QLabel('Format: name,baseline[m],position_angle[deg]'))
        layout.addWidget(QLabel('Example: ALMA_BL1,100,45'))
        
        text_edit = QTextEdit()
        
        # Load existing custom baselines
        if self.custom_baselines:
            lines = []
            for name, bl, pa in zip(self.custom_bl_names, self.custom_baselines, self.custom_pas):
                lines.append(f"{name},{bl},{pa}")
            text_edit.setText('\n'.join(lines))
        else:
            # Default example
            text_edit.setText("ALMA_BL1,100,45\nALMA_BL2,200,90\nNOEMA_BL1,150,30")
        
        layout.addWidget(text_edit)
        
        # Buttons
        btn_layout = QHBoxLayout()
        
        save_btn = QPushButton('Save')
        def save_baselines():
            lines = text_edit.toPlainText().strip().split('\n')
            self.custom_baselines = []
            self.custom_pas = []
            self.custom_bl_names = []
            
            errors = []
            for i, line in enumerate(lines, 1):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                    
                parts = line.split(',')
                if len(parts) != 3:
                    errors.append(f"Line {i}: Expected 3 values (name,baseline,pa)")
                    continue
                
                try:
                    name = parts[0].strip()
                    bl = float(parts[1].strip())
                    pa = float(parts[2].strip())
                    
                    self.custom_bl_names.append(name)
                    self.custom_baselines.append(bl)
                    self.custom_pas.append(pa)
                except ValueError as e:
                    errors.append(f"Line {i}: {str(e)}")
            
            if errors:
                QMessageBox.warning(dialog, 'Parsing Errors', '\n'.join(errors))
            else:
                self.statusBar().showMessage(f'✓ Loaded {len(self.custom_baselines)} custom baselines')
                dialog.accept()
        
        save_btn.clicked.connect(save_baselines)
        btn_layout.addWidget(save_btn)
        
        cancel_btn = QPushButton('Cancel')
        cancel_btn.clicked.connect(dialog.reject)
        btn_layout.addWidget(cancel_btn)
        
        layout.addLayout(btn_layout)
        
        dialog.exec_()
    
    def calculate_interferometry(self):
        """Calculate interferometric observables (computation heavy)"""
        if not INTERFEROMETRY_AVAILABLE:
            QMessageBox.warning(self, 'Error', 
                'Interferometry module not available. Please ensure interferometry_v2.py is in the same directory.')
            return
        
        if self.current_run_dir is None:
            QMessageBox.warning(self, 'No Data', 'Please load a run directory first.')
            return
        
        # Check if image.out exists in run directory
        image_out_path = os.path.join(self.current_run_dir, 'image.out')
        if not os.path.exists(image_out_path):
            QMessageBox.warning(self, 'No image.out', 
                'image.out not found in run directory. Please generate it with RADMC-3D first.')
            return
        
        # Check baseline mode
        baseline_mode = self.baseline_mode_combo.currentText()
        use_custom = baseline_mode == 'Custom baselines'
        
        if use_custom and not self.custom_baselines:
            QMessageBox.warning(self, 'No Custom Baselines', 
                'Please define custom baselines first using "Edit Custom Baselines" button.')
            return
        
        try:
            self.statusBar().showMessage('Calculating visibilities...')
            QApplication.processEvents()
            
            # Get wavelength selection
            if self.use_all_wavelengths_cb.isChecked():
                wavelength_idx = None
            else:
                wavelength_idx = self.interfero_wl_spin.value()
            
            # Get distance from text field and validate
            try:
                dpc = float(self.dpc_input.text())
                if dpc <= 0:
                    raise ValueError("Distance must be positive")
            except ValueError as e:
                QMessageBox.warning(self, 'Invalid Distance', 
                    f'Please enter a valid positive number for distance.\nError: {e}')
                return
            
            # Calculate visibilities with appropriate baselines
            if use_custom:
                # Use custom baselines
                import numpy as np
                self.vis_data = calculate_visibilities_custom(
                    self.current_run_dir, 
                    baselines=np.array(self.custom_baselines),
                    position_angles=np.array(self.custom_pas),
                    baseline_names=self.custom_bl_names,
                    dpc=dpc, 
                    wavelength_idx=wavelength_idx
                )
            else:
                # Use VLTI baselines
                self.vis_data = calculate_visibilities(self.current_run_dir, dpc=dpc, wavelength_idx=wavelength_idx)
            
            self.statusBar().showMessage('Calculating closure phases...')
            QApplication.processEvents()
            
            # Calculate closure phases (only for VLTI mode, skip for custom)
            if not use_custom:
                self.cp_data = calculate_closure_phases(self.current_run_dir, dpc=dpc, wavelength_idx=wavelength_idx)
            else:
                self.cp_data = None  # No predefined triangles for custom baselines
            
            # Update status
            n_baselines = len(self.vis_data['bl'])
            n_wavelengths = len(self.vis_data['wav'])
            
            if self.cp_data:
                n_triangles = len(self.cp_data['triangles'])
                self.statusBar().showMessage(
                    f'✅ Calculated at {dpc} pc: {n_baselines} baselines, '
                    f'{n_triangles} triangles, {n_wavelengths} wavelength(s)'
                )
            else:
                self.statusBar().showMessage(
                    f'✅ Calculated at {dpc} pc: {n_baselines} custom baselines, '
                    f'{n_wavelengths} wavelength(s)'
                )
            
            # Automatically plot after calculation
            self.replot_interferometry()
            
        except Exception as e:
            QMessageBox.critical(self, 'Error', f'Failed to calculate interferometry:\n{str(e)}')
            import traceback
            traceback.print_exc()
            self.statusBar().showMessage('Error calculating interferometry')
    
    def replot_interferometry(self):
        """Replot interferometry with current settings (no recalculation)"""
        # Silently return if no data calculated yet
        if self.vis_data is None:
            return
        
        try:
            self.statusBar().showMessage('Updating plot...')
            QApplication.processEvents()
            
            # Clear figure
            self.interfero_canvas.fig.clear()
            
            # Get plot selection
            plot_selection = self.interfero_plot_combo.currentText()
            
            # Get plot options
            show_neg = self.show_negative_uv_cb.isChecked()
            wl_colored = self.wavelength_color_cb.isChecked()
            
            if plot_selection == 'All 3 plots':
                if self.cp_data is None:
                    QMessageBox.warning(self, 'No Closure Phases',
                        'Closure phases not available for custom baselines. Please select a different plot.')
                    return
                    
                # Create 2x2 grid (will use 3 plots)
                ax1 = self.interfero_canvas.fig.add_subplot(2, 2, 1)
                ax2 = self.interfero_canvas.fig.add_subplot(2, 2, 2)
                ax3 = self.interfero_canvas.fig.add_subplot(2, 2, (3, 4))
                
                plot_uv_coverage(self.vis_data, ax=ax1, show_negative=show_neg)
                plot_visibility_amplitude(self.vis_data, ax=ax2, wavelength_colored=wl_colored)
                plot_closure_phases(self.cp_data, ax=ax3, wavelength_colored=wl_colored)
                
            elif plot_selection == 'UV Coverage only':
                ax = self.interfero_canvas.fig.add_subplot(111)
                plot_uv_coverage(self.vis_data, ax=ax, show_negative=show_neg)
                
            elif plot_selection == 'Visibility² only':
                ax = self.interfero_canvas.fig.add_subplot(111)
                plot_visibility_amplitude(self.vis_data, ax=ax, wavelength_colored=wl_colored)
                
            elif plot_selection == 'Closure Phase only':
                if self.cp_data is None:
                    QMessageBox.warning(self, 'No Closure Phases',
                        'Closure phases not available for custom baselines.')
                    return
                ax = self.interfero_canvas.fig.add_subplot(111)
                plot_closure_phases(self.cp_data, ax=ax, wavelength_colored=wl_colored)
            
            self.interfero_canvas.fig.tight_layout()
            self.interfero_canvas.draw()
            
            # Update status
            n_wavelengths = len(self.vis_data['wav'])
            
            self.statusBar().showMessage(
                f'✅ Plot updated: {plot_selection} ({n_wavelengths} wavelength(s))'
            )
            
        except Exception as e:
            QMessageBox.critical(self, 'Error', f'Failed to plot:\n{str(e)}')
            import traceback
            traceback.print_exc()

    def show_interferometry_data(self):
        """Display interferometry data in tables"""
        if self.vis_data is None or self.cp_data is None:
            QMessageBox.warning(self, 'No Data', 
                'Please calculate interferometry first.')
            return
        
        # Create dialog window
        dialog = QWidget()
        dialog.setWindowTitle('Interferometry Data')
        dialog.resize(900, 700)
        
        layout = QVBoxLayout(dialog)
        
        # Create tab widget for different data views
        tabs = QTabWidget()
        
        # --- Tab 1: Visibilities ---
        vis_widget = QWidget()
        vis_layout = QVBoxLayout(vis_widget)
        
        vis_table = QTableWidget()
        n_baselines = len(self.vis_data['bl'])
        n_wavelengths = len(self.vis_data['wav'])
        
        # Setup table: columns = baseline name, baseline [m], wavelength [um], u [Mlambda], v [Mlambda], Vis²
        vis_table.setRowCount(n_baselines * n_wavelengths)
        vis_table.setColumnCount(6)
        vis_table.setHorizontalHeaderLabels(['Baseline', 'Length [m]', 'λ [µm]', 'u [Mλ]', 'v [Mλ]', 'Vis²'])
        
        row = 0
        for i_bl in range(n_baselines):
            for i_wav in range(n_wavelengths):
                vis_table.setItem(row, 0, QTableWidgetItem(self.vis_data['bl_names'][i_bl]))
                vis_table.setItem(row, 1, QTableWidgetItem(f"{self.vis_data['bl'][i_bl]:.2f}"))
                vis_table.setItem(row, 2, QTableWidgetItem(f"{self.vis_data['wav'][i_wav]:.4f}"))
                vis_table.setItem(row, 3, QTableWidgetItem(f"{self.vis_data['u'][i_bl, i_wav]/1e6:.4f}"))
                vis_table.setItem(row, 4, QTableWidgetItem(f"{self.vis_data['v'][i_bl, i_wav]/1e6:.4f}"))
                vis_table.setItem(row, 5, QTableWidgetItem(f"{self.vis_data['vis2'][i_bl, i_wav]:.6f}"))
                row += 1
        
        vis_table.resizeColumnsToContents()
        vis_layout.addWidget(vis_table)
        
        # Add export button
        export_vis_btn = QPushButton('Export to CSV')
        export_vis_btn.clicked.connect(lambda: self.export_vis_data())
        vis_layout.addWidget(export_vis_btn)
        
        tabs.addTab(vis_widget, 'Visibilities')
        
        # --- Tab 2: Closure Phases ---
        cp_widget = QWidget()
        cp_layout = QVBoxLayout(cp_widget)
        
        cp_table = QTableWidget()
        n_triangles = len(self.cp_data['triangles'])
        
        # Setup table: triangle name, max baseline [m], wavelength [um], spatial freq [Mlambda], CP [deg]
        cp_table.setRowCount(n_triangles * n_wavelengths)
        cp_table.setColumnCount(5)
        cp_table.setHorizontalHeaderLabels(['Triangle', 'Max BL [m]', 'λ [µm]', 'Spatial Freq [Mλ]', 'CP [°]'])
        
        row = 0
        for i_tri in range(n_triangles):
            for i_wav in range(n_wavelengths):
                spatial_freq = self.cp_data['max_baselines'][i_tri] / self.cp_data['wavelengths'][i_wav]
                cp_table.setItem(row, 0, QTableWidgetItem(self.cp_data['triangle_names'][i_tri]))
                cp_table.setItem(row, 1, QTableWidgetItem(f"{self.cp_data['max_baselines'][i_tri]:.2f}"))
                cp_table.setItem(row, 2, QTableWidgetItem(f"{self.cp_data['wavelengths'][i_wav]:.4f}"))
                cp_table.setItem(row, 3, QTableWidgetItem(f"{spatial_freq:.4f}"))
                cp_table.setItem(row, 4, QTableWidgetItem(f"{self.cp_data['cp'][i_tri, i_wav]:.4f}"))
                row += 1
        
        cp_table.resizeColumnsToContents()
        cp_layout.addWidget(cp_table)
        
        # Add export button
        export_cp_btn = QPushButton('Export to CSV')
        export_cp_btn.clicked.connect(lambda: self.export_cp_data())
        cp_layout.addWidget(export_cp_btn)
        
        tabs.addTab(cp_widget, 'Closure Phases')
        
        # --- Tab 3: Summary ---
        summary_widget = QWidget()
        summary_layout = QVBoxLayout(summary_widget)
        
        summary_text = QTextEdit()
        summary_text.setReadOnly(True)
        
        summary = f"""
Interferometry Data Summary
===========================

Baselines: {n_baselines}
Closure Triangles: {n_triangles}
Wavelengths: {n_wavelengths}

Wavelength Range: {self.vis_data['wav'][0]:.4f} - {self.vis_data['wav'][-1]:.4f} µm

Baseline Range: {min(self.vis_data['bl']):.2f} - {max(self.vis_data['bl']):.2f} m

Vis² Range: {self.vis_data['vis2'].min():.6f} - {self.vis_data['vis2'].max():.6f}

Closure Phase Range: {self.cp_data['cp'].min():.4f}° - {self.cp_data['cp'].max():.4f}°

Baselines:
{chr(10).join([f"  {name}: {bl:.2f} m" for name, bl in zip(self.vis_data['bl_names'], self.vis_data['bl'])])}
"""
        summary_text.setText(summary)
        summary_layout.addWidget(summary_text)
        
        tabs.addTab(summary_widget, 'Summary')
        
        layout.addWidget(tabs)
        
        # Close button
        close_btn = QPushButton('Close')
        close_btn.clicked.connect(dialog.close)
        layout.addWidget(close_btn)
        
        dialog.show()
        
        # Keep reference to prevent garbage collection
        self.interfero_data_dialog = dialog
    
    def export_vis_data(self):
        """Export visibility data to CSV"""
        if self.vis_data is None:
            return
        
        filename, _ = QFileDialog.getSaveFileName(self, 'Save Visibility Data', 
                                                   'visibilities.csv', 'CSV Files (*.csv)')
        if filename:
            try:
                import csv
                with open(filename, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['Baseline', 'Length_m', 'Wavelength_um', 'u_Mlambda', 'v_Mlambda', 'Vis2'])
                    
                    n_baselines = len(self.vis_data['bl'])
                    n_wavelengths = len(self.vis_data['wav'])
                    
                    for i_bl in range(n_baselines):
                        for i_wav in range(n_wavelengths):
                            writer.writerow([
                                self.vis_data['bl_names'][i_bl],
                                f"{self.vis_data['bl'][i_bl]:.2f}",
                                f"{self.vis_data['wav'][i_wav]:.6f}",
                                f"{self.vis_data['u'][i_bl, i_wav]/1e6:.6f}",
                                f"{self.vis_data['v'][i_bl, i_wav]/1e6:.6f}",
                                f"{self.vis_data['vis2'][i_bl, i_wav]:.8f}"
                            ])
                
                self.statusBar().showMessage(f'✅ Visibility data exported to {filename}')
            except Exception as e:
                QMessageBox.critical(self, 'Export Error', f'Failed to export:\n{str(e)}')
    
    def export_cp_data(self):
        """Export closure phase data to CSV"""
        if self.cp_data is None:
            return
        
        filename, _ = QFileDialog.getSaveFileName(self, 'Save Closure Phase Data', 
                                                   'closure_phases.csv', 'CSV Files (*.csv)')
        if filename:
            try:
                import csv
                with open(filename, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['Triangle', 'Max_Baseline_m', 'Wavelength_um', 'Spatial_Freq_Mlambda', 'Closure_Phase_deg'])
                    
                    n_triangles = len(self.cp_data['triangles'])
                    n_wavelengths = len(self.cp_data['wavelengths'])
                    
                    for i_tri in range(n_triangles):
                        for i_wav in range(n_wavelengths):
                            spatial_freq = self.cp_data['max_baselines'][i_tri] / self.cp_data['wavelengths'][i_wav]
                            writer.writerow([
                                self.cp_data['triangle_names'][i_tri],
                                f"{self.cp_data['max_baselines'][i_tri]:.2f}",
                                f"{self.cp_data['wavelengths'][i_wav]:.6f}",
                                f"{spatial_freq:.6f}",
                                f"{self.cp_data['cp'][i_tri, i_wav]:.6f}"
                            ])
                
                self.statusBar().showMessage(f'✅ Closure phase data exported to {filename}')
            except Exception as e:
                QMessageBox.critical(self, 'Export Error', f'Failed to export:\n{str(e)}')

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