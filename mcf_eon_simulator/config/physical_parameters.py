#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Physical Layer Parameters for C+L Band MCF System
"""

import numpy as np

# Physical Constants
PLANCK_CONSTANT = 6.626e-34  # J⋅s
SPEED_OF_LIGHT = 3e8         # m/s
REFERENCE_FREQUENCY = 193.4e12  # Hz (1550 nm)

# Fiber Parameters
class FiberParameters:
    def __init__(self):
        # Dispersion parameters at 1550nm reference
        self.beta2 = -21.86e-27  # ps²/m
        self.beta3 = 0.1331e-39  # ps³/m  
        self.beta4 = -2.7e-55    # ps⁴/m
        
        # Nonlinearity parameters
        self.n2 = 2.6e-20        # m²/W
        self.effective_area_um2 = 80  # μm²
        self.effective_area_m2 = 80e-12  # m²
        
        # Loss parameters (frequency dependent)
        self.base_loss_db_km = 0.21
        self.loss_variation_db_km = 0.02
        
        # Channel parameters
        self.symbol_rate_gbaud = 64
        self.symbol_rate_hz = 64e9
        
    def get_attenuation_coefficient(self, frequency_hz):
        """Get frequency-dependent attenuation coefficient"""
        wavelength_nm = SPEED_OF_LIGHT / frequency_hz * 1e9
        
        # Realistic SSMF attenuation
        if wavelength_nm < 1530:  # L-band
            rayleigh_factor = (1550 / wavelength_nm) ** 0.25
            alpha_db_km = self.base_loss_db_km * (0.98 + 0.04 * rayleigh_factor)
        elif wavelength_nm > 1570:  # C-band
            ir_factor = 1 + 0.003 * (wavelength_nm - 1570) / 50
            alpha_db_km = self.base_loss_db_km * ir_factor
        else:  # Transition region
            alpha_db_km = self.base_loss_db_km
            
        # Convert to linear units (1/m)
        alpha_linear_per_m = alpha_db_km * np.log(10) / (10 * 1000)
        return alpha_linear_per_m

# Amplifier Parameters
class AmplifierParameters:
    def __init__(self):
        # Noise figures by band
        self.c_band_nf_db = 4.5  # EDFA C-band
        self.l_band_nf_db = 5.0  # EDFA L-band
        
        # Gain parameters
        self.max_gain_db = 25.0
        self.min_gain_db = 15.0
        self.target_output_power_dbm = 0.0
        
        # Saturation parameters
        self.max_output_power_dbm = 20.0
        self.gain_flatness_tolerance_db = 1.0

# MCF Coupling Parameters
class MCFParameters:
    def __init__(self):
        self.num_cores = 4
        self.core_pitch_um = 43.0
        self.core_pitch_m = 43.0e-6
        
        # Coupling matrix for 4-core square layout
        # Core adjacency: 0-1, 0-2, 1-3, 2-3 (square configuration)
        self.adjacency_matrix = np.array([
            [0, 1, 1, 0],  # Core 0 adjacent to cores 1, 2
            [1, 0, 0, 1],  # Core 1 adjacent to cores 0, 3  
            [1, 0, 0, 1],  # Core 2 adjacent to cores 0, 3
            [0, 1, 1, 0]   # Core 3 adjacent to cores 1, 2
        ])
        
        # Number of adjacent cores per core
        self.adjacent_cores_count = [2, 2, 2, 2]  # Square layout
        
        # Mode coupling parameters
        self.mode_coupling_base = 1e-4
        self.coupling_length_dependence = 1.0
        
    def get_coupling_coefficient(self, frequency_hz):
        """Calculate frequency-dependent coupling coefficient"""
        wavelength_m = SPEED_OF_LIGHT / frequency_hz
        
        # Simplified frequency-dependent coupling
        coupling_coeff = self.mode_coupling_base * (1550e-9 / wavelength_m) ** 0.5
        return coupling_coeff

# Raman Gain Parameters
class RamanParameters:
    def __init__(self):
        # Raman gain peaks for silica fiber
        self.raman_peaks_thz = [13.2, 15.8, 17.6]  # Frequency shifts
        self.raman_amplitudes = [1.0, 0.4, 0.2]    # Relative amplitudes
        self.raman_widths_thz = [2.5, 3.0, 3.5]    # Spectral widths
        self.raman_efficiency = 0.65e-13           # m/W
        
    def get_raman_gain_coefficient(self, pump_freq_hz, signal_freq_hz):
        """Calculate Raman gain coefficient between pump and signal"""
        freq_diff_hz = abs(pump_freq_hz - signal_freq_hz)
        freq_diff_thz = freq_diff_hz / 1e12
        
        gain_total = 0
        for peak_freq, amplitude, width in zip(self.raman_peaks_thz, 
                                             self.raman_amplitudes, 
                                             self.raman_widths_thz):
            # Lorentzian profile
            gain_component = amplitude * (width/2)**2 / ((freq_diff_thz - peak_freq)**2 + (width/2)**2)
            gain_total += gain_component
        
        # Include frequency scaling
        if pump_freq_hz > signal_freq_hz:  # Stokes process
            raman_gain = gain_total * self.raman_efficiency * pump_freq_hz / signal_freq_hz
        else:  # Anti-Stokes process  
            raman_gain = -gain_total * self.raman_efficiency * signal_freq_hz / pump_freq_hz
            
        return raman_gain

# Initialize global parameter objects
FIBER_PARAMS = FiberParameters()
AMPLIFIER_PARAMS = AmplifierParameters()
MCF_PARAMS = MCFParameters()
RAMAN_PARAMS = RamanParameters()