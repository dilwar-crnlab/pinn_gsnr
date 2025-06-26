#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Request Class for MCF EON Simulator
"""

from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np

@dataclass
class MCFRequest:
    """Request class for multi-core fiber EON"""
    
    # Basic request parameters
    request_id: int
    source: int
    destination: int
    bandwidth_gbps: int
    arrival_time: float
    holding_time: float
    
    # Assignment results
    assigned_path: Optional[List[int]] = None
    assigned_cores: Optional[List[int]] = None
    assigned_channels: Optional[List[List[int]]] = None  # Channels per core
    assigned_bands: Optional[List[str]] = None
    modulation_formats: Optional[List[int]] = None
    actual_bitrates: Optional[List[int]] = None
    
    # QoT metrics
    end_to_end_gsnr_db: Optional[float] = None
    per_channel_gsnr_db: Optional[List[float]] = None
    limiting_impairment: Optional[str] = None  # 'ASE', 'NLI', 'ICXT'
    
    # Resource allocation details
    num_slices: int = 1
    slice_bandwidths: Optional[List[int]] = None
    total_spectrum_slots: int = 0
    spectral_efficiency: float = 0.0
    
    # Status tracking
    is_blocked: bool = False
    blocking_reason: Optional[str] = None  # 'spectrum', 'gsnr', 'path', 'icxt'
    is_active: bool = False
    departure_time: Optional[float] = None
    
    # Performance metrics
    setup_time: float = 0.0
    qot_computation_time: float = 0.0
    allocation_attempts: int = 0
    
    def __post_init__(self):
        """Initialize derived fields"""
        if self.assigned_channels is None:
            self.assigned_channels = [[] for _ in range(4)]  # 4 cores
        if self.assigned_cores is None:
            self.assigned_cores = []
        if self.assigned_bands is None:
            self.assigned_bands = []
        if self.modulation_formats is None:
            self.modulation_formats = []
        if self.actual_bitrates is None:
            self.actual_bitrates = []
        if self.per_channel_gsnr_db is None:
            self.per_channel_gsnr_db = []
        if self.slice_bandwidths is None:
            self.slice_bandwidths = [self.bandwidth_gbps]
    
    def add_slice(self, core: int, channels: List[int], band: str, 
                  modulation_format: int, bitrate: int, gsnr_db: float):
        """Add a slice assignment to the request"""
        if core not in self.assigned_cores:
            self.assigned_cores.append(core)
        
        # Add channels to the specific core
        self.assigned_channels[core].extend(channels)
        self.assigned_bands.append(band)
        self.modulation_formats.append(modulation_format)
        self.actual_bitrates.append(bitrate)
        self.per_channel_gsnr_db.extend([gsnr_db] * len(channels))
        
        # Update spectrum usage
        self.total_spectrum_slots += len(channels)
        
    def calculate_spectral_efficiency(self):
        """Calculate spectral efficiency in bps/Hz"""
        if self.total_spectrum_slots > 0:
            total_bitrate = sum(self.actual_bitrates)
            # Assuming 100 GHz channel spacing
            total_bandwidth_hz = self.total_spectrum_slots * 100e9
            self.spectral_efficiency = (total_bitrate * 1e9) / total_bandwidth_hz
        return self.spectral_efficiency
    
    def get_resource_summary(self):
        """Get summary of allocated resources"""
        return {
            'request_id': self.request_id,
            'bandwidth_requested': self.bandwidth_gbps,
            'bandwidth_allocated': sum(self.actual_bitrates),
            'cores_used': len(self.assigned_cores),
            'channels_used': self.total_spectrum_slots,
            'num_slices': self.num_slices,
            'spectral_efficiency': self.spectral_efficiency,
            'end_to_end_gsnr_db': self.end_to_end_gsnr_db,
            'is_blocked': self.is_blocked,
            'blocking_reason': self.blocking_reason
        }
    
    def is_successfully_allocated(self):
        """Check if request is successfully allocated"""
        return (not self.is_blocked and 
                sum(self.actual_bitrates) >= self.bandwidth_gbps and
                len(self.assigned_cores) > 0)
    
    def get_core_utilization(self):
        """Get utilization per core"""
        core_util = {}
        for i, core in enumerate(self.assigned_cores):
            core_util[core] = {
                'channels': len(self.assigned_channels[core]),
                'bitrate': self.actual_bitrates[i] if i < len(self.actual_bitrates) else 0,
                'band': self.assigned_bands[i] if i < len(self.assigned_bands) else 'Unknown'
            }
        return core_util

@dataclass 
class RequestStatistics:
    """Statistics tracking for requests"""
    
    total_requests: int = 0
    blocked_requests: int = 0
    successful_requests: int = 0
    
    # Blocking reasons
    blocked_spectrum: int = 0
    blocked_gsnr: int = 0  
    blocked_icxt: int = 0
    blocked_path: int = 0
    
    # Per bandwidth class
    requests_per_class: List[int] = field(default_factory=lambda: [0] * 6)
    blocked_per_class: List[int] = field(default_factory=lambda: [0] * 6)
    
    # Per core utilization
    core_utilization: List[int] = field(default_factory=lambda: [0] * 4)
    
    # QoT statistics
    average_gsnr_db: float = 0.0
    min_gsnr_db: float = float('inf')
    max_gsnr_db: float = 0.0
    
    # Spectral efficiency
    total_spectral_efficiency: float = 0.0
    average_spectral_efficiency: float = 0.0
    
    def update_request_stats(self, request: MCFRequest):
        """Update statistics with new request"""
        self.total_requests += 1
        
        if request.is_blocked:
            self.blocked_requests += 1
            # Update blocking reason counters
            if request.blocking_reason == 'spectrum':
                self.blocked_spectrum += 1
            elif request.blocking_reason == 'gsnr':
                self.blocked_gsnr += 1
            elif request.blocking_reason == 'icxt':
                self.blocked_icxt += 1
            elif request.blocking_reason == 'path':
                self.blocked_path += 1
        else:
            self.successful_requests += 1
            
            # Update QoT statistics
            if request.end_to_end_gsnr_db is not None:
                self.average_gsnr_db = ((self.average_gsnr_db * (self.successful_requests - 1) + 
                                       request.end_to_end_gsnr_db) / self.successful_requests)
                self.min_gsnr_db = min(self.min_gsnr_db, request.end_to_end_gsnr_db)
                self.max_gsnr_db = max(self.max_gsnr_db, request.end_to_end_gsnr_db)
            
            # Update core utilization
            for core in request.assigned_cores:
                self.core_utilization[core] += 1
                
            # Update spectral efficiency
            request.calculate_spectral_efficiency()
            self.total_spectral_efficiency += request.spectral_efficiency
            self.average_spectral_efficiency = (self.total_spectral_efficiency / 
                                              self.successful_requests)
    
    def get_blocking_probability(self):
        """Calculate overall blocking probability"""
        if self.total_requests == 0:
            return 0.0
        return self.blocked_requests / self.total_requests
    
    def get_blocking_breakdown(self):
        """Get breakdown of blocking reasons"""
        if self.blocked_requests == 0:
            return {}
        
        return {
            'spectrum': self.blocked_spectrum / self.blocked_requests,
            'gsnr': self.blocked_gsnr / self.blocked_requests,
            'icxt': self.blocked_icxt / self.blocked_requests,
            'path': self.blocked_path / self.blocked_requests
        }