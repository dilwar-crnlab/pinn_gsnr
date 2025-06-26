#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Multi-Core Fiber Link Implementation
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from config.system_config import *
from config.physical_parameters import *

@dataclass
class MCFLinkState:
    """State tracking for MCF link"""
    
    # Spectrum occupancy [core][channel] -> bool
    spectrum_usage: np.ndarray = field(default_factory=lambda: np.zeros((MCF_CORES, TOTAL_CHANNELS), dtype=bool))
    
    # Power levels [core][channel] -> float (Watts)
    power_levels: np.ndarray = field(default_factory=lambda: np.zeros((MCF_CORES, TOTAL_CHANNELS)))
    
    # Active requests per core-channel
    active_requests: Dict[Tuple[int, int], List[int]] = field(default_factory=dict)
    
    # ICXT matrix [core][core] -> float
    icxt_matrix: np.ndarray = field(default_factory=lambda: np.zeros((MCF_CORES, MCF_CORES)))
    
    # Amplifier states per span
    amplifier_gains: Dict[int, Dict[str, float]] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize arrays if needed"""
        if self.spectrum_usage.size == 0:
            self.spectrum_usage = np.zeros((MCF_CORES, TOTAL_CHANNELS), dtype=bool)
        if self.power_levels.size == 0:
            self.power_levels = np.zeros((MCF_CORES, TOTAL_CHANNELS))
        if self.icxt_matrix.size == 0:
            self.icxt_matrix = np.zeros((MCF_CORES, MCF_CORES))

class MCFLink:
    """Multi-Core Fiber Link Class"""
    
    def __init__(self, link_id: int, source: int, destination: int, 
                 distance_km: float, num_spans: int = None):
        """
        Initialize MCF Link
        
        Args:
            link_id: Unique link identifier
            source: Source node ID
            destination: Destination node ID  
            distance_km: Total link distance
            num_spans: Number of spans (calculated if None)
        """
        self.link_id = link_id
        self.source = source
        self.destination = destination
        self.distance_km = distance_km
        
        # Calculate number of spans
        if num_spans is None:
            self.num_spans = max(1, int(np.ceil(distance_km / FIBER_SPAN_LENGTH_KM)))
        else:
            self.num_spans = num_spans
            
        self.span_length_km = distance_km / self.num_spans
        
        # Initialize link state
        self.state = MCFLinkState()
        
        # Physical parameters
        self.fiber_params = FIBER_PARAMS
        self.mcf_params = MCF_PARAMS
        
        # Initialize amplifier states
        for span in range(self.num_spans):
            self.state.amplifier_gains[span] = {
                'C': 20.0,  # Default 20 dB gain
                'L': 20.0
            }
        
        # Channel-to-frequency mapping
        self.setup_frequency_grid()
        
    def setup_frequency_grid(self):
        """Setup frequency grid for C+L bands"""
        # L-band channels (0 to L_BAND_CHANNELS-1)
        l_band_freqs = np.linspace(L_BAND_START_THZ, L_BAND_END_THZ, L_BAND_CHANNELS) * 1e12
        
        # C-band channels (L_BAND_CHANNELS to TOTAL_CHANNELS-1)  
        c_band_freqs = np.linspace(C_BAND_START_THZ, C_BAND_END_THZ, C_BAND_CHANNELS) * 1e12
        
        # Combined frequency grid
        self.frequencies_hz = np.concatenate([l_band_freqs, c_band_freqs])
        self.wavelengths_nm = SPEED_OF_LIGHT / self.frequencies_hz * 1e9
        
        # Band classification
        self.channel_bands = ['L'] * L_BAND_CHANNELS + ['C'] * C_BAND_CHANNELS
        
    def get_available_channels(self, core: int, band: str = None) -> List[int]:
        """
        Get list of available channels for a specific core
        
        Args:
            core: Core index (0-3)
            band: Band filter ('C', 'L', or None for both)
            
        Returns:
            List of available channel indices
        """
        available = []
        
        for ch in range(TOTAL_CHANNELS):
            if not self.state.spectrum_usage[core, ch]:
                if band is None or self.channel_bands[ch] == band:
                    available.append(ch)
                    
        return available
    
    def get_contiguous_channels(self, core: int, num_channels: int, 
                              band: str = None) -> Optional[List[int]]:
        """
        Find contiguous available channels
        
        Args:
            core: Core index
            num_channels: Number of required contiguous channels
            band: Band preference ('C', 'L', or None)
            
        Returns:
            List of contiguous channel indices or None if not available
        """
        available = self.get_available_channels(core, band)
        
        if len(available) < num_channels:
            return None
            
        # Find contiguous sequence
        for i in range(len(available) - num_channels + 1):
            candidate = available[i:i + num_channels]
            
            # Check if channels are contiguous
            if all(candidate[j+1] - candidate[j] == 1 for j in range(len(candidate)-1)):
                return candidate
                
        return None
    
    def allocate_channels(self, core: int, channels: List[int], 
                         request_id: int, power_w: float) -> bool:
        """
        Allocate channels to a request
        
        Args:
            core: Core index
            channels: List of channel indices to allocate
            request_id: Request ID
            power_w: Power level per channel
            
        Returns:
            True if allocation successful
        """
        # Check availability
        for ch in channels:
            if self.state.spectrum_usage[core, ch]:
                return False
                
        # Allocate channels
        for ch in channels:
            self.state.spectrum_usage[core, ch] = True
            self.state.power_levels[core, ch] = power_w
            
            # Track request
            key = (core, ch)
            if key not in self.state.active_requests:
                self.state.active_requests[key] = []
            self.state.active_requests[key].append(request_id)
            
        # Update ICXT matrix
        self.update_icxt_matrix()
        
        return True
    
    def deallocate_channels(self, core: int, channels: List[int], request_id: int):
        """
        Deallocate channels from a request
        
        Args:
            core: Core index
            channels: List of channel indices to deallocate
            request_id: Request ID
        """
        for ch in channels:
            if self.state.spectrum_usage[core, ch]:
                self.state.spectrum_usage[core, ch] = False
                self.state.power_levels[core, ch] = 0.0
                
                # Remove request tracking
                key = (core, ch)
                if key in self.state.active_requests:
                    if request_id in self.state.active_requests[key]:
                        self.state.active_requests[key].remove(request_id)
                    if not self.state.active_requests[key]:
                        del self.state.active_requests[key]
        
        # Update ICXT matrix
        self.update_icxt_matrix()
    
    def update_icxt_matrix(self):
        """Update inter-core crosstalk matrix"""
        self.state.icxt_matrix.fill(0.0)
        
        for core_i in range(MCF_CORES):
            for core_j in range(MCF_CORES):
                if core_i != core_j and MCF_PARAMS.adjacency_matrix[core_i, core_j]:
                    # Calculate ICXT between adjacent cores
                    total_power_j = np.sum(self.state.power_levels[core_j, :])
                    if total_power_j > 0:
                        # Simplified ICXT calculation
                        coupling_coeff = MCF_PARAMS.get_coupling_coefficient(
                            np.mean(self.frequencies_hz))
                        distance_m = self.distance_km * 1000
                        
                        # Power coupling
                        icxt_power = coupling_coeff * total_power_j * distance_m
                        self.state.icxt_matrix[core_i, core_j] = icxt_power
    
    def get_core_utilization(self) -> Dict[int, float]:
        """Get utilization percentage per core"""
        utilization = {}
        for core in range(MCF_CORES):
            used_channels = np.sum(self.state.spectrum_usage[core, :])
            utilization[core] = used_channels / TOTAL_CHANNELS
        return utilization
    
    def get_band_utilization(self) -> Dict[str, float]:
        """Get utilization per band across all cores"""
        band_util = {'C': 0.0, 'L': 0.0}
        
        for band in ['C', 'L']:
            band_channels = [i for i, b in enumerate(self.channel_bands) if b == band]
            total_band_slots = len(band_channels) * MCF_CORES
            used_band_slots = 0
            
            for core in range(MCF_CORES):
                for ch in band_channels:
                    if self.state.spectrum_usage[core, ch]:
                        used_band_slots += 1
                        
            band_util[band] = used_band_slots / total_band_slots if total_band_slots > 0 else 0.0
            
        return band_util
    
    def get_link_statistics(self) -> Dict:
        """Get comprehensive link statistics"""
        return {
            'link_id': self.link_id,
            'source': self.source,
            'destination': self.destination,
            'distance_km': self.distance_km,
            'num_spans': self.num_spans,
            'core_utilization': self.get_core_utilization(),
            'band_utilization': self.get_band_utilization(),
            'total_active_channels': np.sum(self.state.spectrum_usage),
            'average_power_per_core': [np.mean(self.state.power_levels[core, :]) 
                                     for core in range(MCF_CORES)],
            'icxt_levels': self.state.icxt_matrix.tolist()
        }
    
    def check_channel_quality(self, core: int, channel: int, 
                            required_gsnr_db: float) -> bool:
        """
        Check if channel meets QoT requirements
        
        Args:
            core: Core index
            channel: Channel index
            required_gsnr_db: Required GSNR threshold
            
        Returns:
            True if channel meets QoT requirements
        """
        # This would integrate with GSNR calculator
        # For now, simplified check based on ICXT levels
        
        adjacent_icxt = 0.0
        for adj_core in range(MCF_CORES):
            if MCF_PARAMS.adjacency_matrix[core, adj_core]:
                adjacent_icxt += self.state.icxt_matrix[core, adj_core]
        
        # Convert to dB
        icxt_db = 10 * np.log10(adjacent_icxt + 1e-15)
        
        # Simplified QoT check (would be replaced by full GSNR calculation)
        estimated_penalty_db = max(0, icxt_db + 26.82)  # ICXT threshold for PM-64QAM
        
        return estimated_penalty_db < 1.0  # 1 dB penalty threshold
    
    def __str__(self):
        """String representation"""
        return f"MCFLink({self.source}->{self.destination}, {self.distance_km}km, {self.num_spans}spans)"
    
    def __repr__(self):
        return self.__str__()