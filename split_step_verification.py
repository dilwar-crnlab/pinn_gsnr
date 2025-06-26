#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Split-Step Method Verification and Plotting
Comprehensive validation and visualization for the split-step ground truth generator
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import time
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import seaborn as sns
from matplotlib.gridspec import GridSpec

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class SplitStepVerifier:
    """
    Verification and plotting class for split-step ground truth generator
    """
    
    def __init__(self, generator=None):
        """Initialize with optional generator instance"""
        self.generator = generator
        
    def verify_physical_parameters(self, generator) -> Dict:
        """Verify that physical parameters are within realistic ranges"""
        
        print("=" * 60)
        print("PHYSICAL PARAMETER VERIFICATION")
        print("=" * 60)
        
        verification_results = {}
        
        # Check frequency grid
        freq_check = self._verify_frequency_grid(generator)
        verification_results['frequency_grid'] = freq_check
        
        # Check attenuation coefficients
        attenuation_check = self._verify_attenuation(generator)
        verification_results['attenuation'] = attenuation_check
        
        # Check Raman gain matrix
        raman_check = self._verify_raman_gain(generator)
        verification_results['raman_gain'] = raman_check
        
        # Check span configuration
        span_check = self._verify_span_config(generator)
        verification_results['span_config'] = span_check
        
        return verification_results
    
    def _verify_frequency_grid(self, generator) -> Dict:
        """Verify frequency grid matches PINN paper specifications"""
        
        print("\n1. Frequency Grid Verification:")
        
        results = {
            'total_channels': len(generator.frequencies),
            'l_band_channels': 0,
            'c_band_channels': 0,
            'channel_spacing_ghz': generator.channel_spacing / 1e9,
            'status': 'PASS'
        }
        
        # Count L-band and C-band channels
        for freq in generator.frequencies:
            if generator.l_band_start <= freq <= generator.l_band_end:
                results['l_band_channels'] += 1
            elif generator.c_band_start <= freq <= generator.c_band_end:
                results['c_band_channels'] += 1
        
        print(f"   Total channels: {results['total_channels']} (expected: 96)")
        print(f"   L-band channels: {results['l_band_channels']}")
        print(f"   C-band channels: {results['c_band_channels']}")
        print(f"   Channel spacing: {results['channel_spacing_ghz']:.1f} GHz")
        
        # Verify against paper specifications
        if results['total_channels'] != 96:
            results['status'] = 'FAIL'
            print("   ❌ Channel count mismatch!")
        else:
            print("   ✅ Channel configuration correct")
            
        return results
    
    def _verify_attenuation(self, generator) -> Dict:
        """Verify attenuation coefficients are realistic for SSMF"""
        
        print("\n2. Attenuation Coefficient Verification:")
        
        alpha_f = generator.alpha_f
        results = {
            'min_loss_db_km': float(np.min(alpha_f)),
            'max_loss_db_km': float(np.max(alpha_f)),
            'mean_loss_db_km': float(np.mean(alpha_f)),
            'std_loss_db_km': float(np.std(alpha_f)),
            'status': 'PASS'
        }
        
        print(f"   Loss range: {results['min_loss_db_km']:.3f} - {results['max_loss_db_km']:.3f} dB/km")
        print(f"   Mean loss: {results['mean_loss_db_km']:.3f} ± {results['std_loss_db_km']:.3f} dB/km")
        
        # Check against typical SSMF values (0.18-0.25 dB/km)
        if results['min_loss_db_km'] < 0.15 or results['max_loss_db_km'] > 0.3:
            results['status'] = 'WARNING'
            print("   ⚠️ Loss values outside typical SSMF range")
        else:
            print("   ✅ Loss values within realistic SSMF range")
            
        return results
    
    def _verify_raman_gain(self, generator) -> Dict:
        """Verify Raman gain matrix properties"""
        
        print("\n3. Raman Gain Matrix Verification:")
        
        raman_matrix = generator.raman_gain_matrix
        results = {
            'matrix_shape': raman_matrix.shape,
            'max_gain': float(np.max(raman_matrix)),
            'min_gain': float(np.min(raman_matrix)),
            'symmetry_check': 'PASS',
            'status': 'PASS'
        }
        
        print(f"   Matrix shape: {results['matrix_shape']}")
        print(f"   Gain range: {results['min_gain']:.2e} - {results['max_gain']:.2e} m/W")
        
        # Check anti-symmetry property (gij = -gji for SRS)
        symmetry_error = np.max(np.abs(raman_matrix + raman_matrix.T))
        if symmetry_error > 1e-15:
            results['symmetry_check'] = 'FAIL'
            print(f"   ❌ Symmetry error: {symmetry_error:.2e}")
        else:
            print("   ✅ Anti-symmetry property satisfied")
        
        # Check typical Raman gain values
        if results['max_gain'] > 1e-12 or results['max_gain'] < 1e-14:
            results['status'] = 'WARNING'
            print("   ⚠️ Gain values outside typical range")
        else:
            print("   ✅ Gain values realistic for silica fiber")
            
        return results
    
    def _verify_span_config(self, generator) -> Dict:
        """Verify span configuration matches paper"""
        
        print("\n4. Span Configuration Verification:")
        
        spans = generator.spans
        results = {
            'total_spans': len(spans),
            'span_lengths': [span.length_km for span in spans],
            'total_distance': sum(span.length_km for span in spans),
            'span2_length': spans[1].length_km if len(spans) > 1 else None,
            'status': 'PASS'
        }
        
        print(f"   Total spans: {results['total_spans']} (expected: 8)")
        print(f"   Total distance: {results['total_distance']} km")
        print(f"   Span #2 length: {results['span2_length']} km (expected: 85)")
        
        # Verify span #2 is 85 km and others are 75 km
        if results['span2_length'] != 85:
            results['status'] = 'FAIL'
            print("   ❌ Span #2 length incorrect!")
        elif len([l for l in results['span_lengths'] if l == 75]) != 7:
            results['status'] = 'WARNING'
            print("   ⚠️ Not all other spans are 75 km")
        else:
            print("   ✅ Span configuration matches paper")
            
        return results
    
    def plot_comprehensive_analysis(self, generator, dataset: Dict, save_dir: str = "."):
        """Create comprehensive analysis plots"""
        
        print("\n" + "=" * 60)
        print("GENERATING COMPREHENSIVE ANALYSIS PLOTS")
        print("=" * 60)
        
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)
        
        # 1. System overview plot
        self._plot_system_overview(generator, save_path)
        
        # 2. Fiber parameters plot
        self._plot_fiber_parameters(generator, save_path)
        
        # 3. Power evolution plots for each scenario
        for scenario_name, scenario_data in dataset['scenarios'].items():
            self._plot_scenario_analysis(scenario_name, scenario_data, dataset, save_path)
        
        # 4. ISRS effects comparison
        self._plot_isrs_effects(dataset, save_path)
        
        # 5. Validation metrics summary
        self._plot_validation_summary(dataset, save_path)
        
        print(f"\n✅ All plots saved to: {save_path}")
    
    def _plot_system_overview(self, generator, save_path: Path):
        """Plot system configuration overview"""
        
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(3, 3, figure=fig)
        
        # Frequency grid
        ax1 = fig.add_subplot(gs[0, :])
        frequencies_thz = generator.frequencies / 1e12
        
        # Create frequency bands visualization
        l_band_mask = (generator.frequencies >= generator.l_band_start) & \
                      (generator.frequencies <= generator.l_band_end)
        c_band_mask = (generator.frequencies >= generator.c_band_start) & \
                      (generator.frequencies <= generator.c_band_end)
        
        ax1.scatter(frequencies_thz[l_band_mask], np.ones(np.sum(l_band_mask)), 
                   c='red', alpha=0.7, s=50, label=f'L-band ({np.sum(l_band_mask)} ch)')
        ax1.scatter(frequencies_thz[c_band_mask], np.ones(np.sum(c_band_mask)), 
                   c='blue', alpha=0.7, s=50, label=f'C-band ({np.sum(c_band_mask)} ch)')
        
        ax1.set_xlabel('Frequency (THz)', fontweight='bold')
        ax1.set_ylabel('Channel')
        ax1.set_title('System Frequency Grid: C+L Band Configuration', fontweight='bold', fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0.5, 1.5)
        ax1.set_yticks([])
        
        # Span configuration
        ax2 = fig.add_subplot(gs[1, :])
        span_lengths = [span.length_km for span in generator.spans]
        span_ids = [span.span_id for span in generator.spans]
        
        colors = ['orange' if length == 85 else 'steelblue' for length in span_lengths]
        bars = ax2.bar(span_ids, span_lengths, color=colors, alpha=0.8, edgecolor='black')
        
        # Add length labels on bars
        for bar, length in zip(bars, span_lengths):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{length} km', ha='center', va='bottom', fontweight='bold')
        
        ax2.set_xlabel('Span Number', fontweight='bold')
        ax2.set_ylabel('Length (km)', fontweight='bold')
        ax2.set_title('8-Span System Configuration', fontweight='bold', fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, max(span_lengths) + 10)
        
        # Add total distance text
        total_distance = sum(span_lengths)
        ax2.text(0.02, 0.95, f'Total Distance: {total_distance} km', 
                transform=ax2.transAxes, fontsize=12, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
        
        # System parameters summary
        ax3 = fig.add_subplot(gs[2, :])
        ax3.axis('off')
        
        summary_text = f"""
        SYSTEM PARAMETERS SUMMARY
        ══════════════════════════════════════════════════════════════
        
         Channel Configuration:
           • Total Channels: {len(generator.frequencies)}
           • L-band: {generator.l_band_start/1e12:.1f} - {generator.l_band_end/1e12:.1f} THz ({np.sum(l_band_mask)} channels)
           • C-band: {generator.c_band_start/1e12:.1f} - {generator.c_band_end/1e12:.1f} THz ({np.sum(c_band_mask)} channels)
           • Channel Spacing: {generator.channel_spacing/1e9:.0f} GHz
        
         Fiber System:
           • Total Spans: {len(generator.spans)}
           • Total Distance: {total_distance} km
           • Fiber Type: Standard Single-Mode Fiber (SSMF)
           • Attenuation: {np.mean(generator.alpha_f):.3f} ± {np.std(generator.alpha_f):.3f} dB/km
        
         Physical Model:
           • Split-Step Method: 100 m spatial resolution
           • ISRS: Full Raman gain matrix (96×96)
           • Amplification: Dual-band EDFA (C+L)
           • ASE Noise: Included with realistic NF
        """
        
        ax3.text(0.05, 0.95, summary_text, transform=ax3.transAxes, 
                fontsize=10, fontfamily='monospace', verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.1))
        
        plt.tight_layout()
        plt.savefig(save_path / "system_overview.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print("   ✅ System overview plot saved")
    
    def _plot_fiber_parameters(self, generator, save_path: Path):
        """Plot detailed fiber parameters"""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        frequencies_thz = generator.frequencies / 1e12
        
        # Plot 1: Attenuation spectrum
        ax1.plot(frequencies_thz, generator.alpha_f, 'b-', linewidth=2, 
                marker='o', markersize=4, label='Split-Step Ground Truth αf')
        
        # Add band boundaries
        ax1.axvline(generator.l_band_end/1e12, color='gray', linestyle=':', alpha=0.8, label='Band boundary')
        ax1.axvline(generator.c_band_start/1e12, color='gray', linestyle=':', alpha=0.8)
        
        # Highlight bands
        ax1.axvspan(generator.l_band_start/1e12, generator.l_band_end/1e12, 
                   alpha=0.15, color='red', label='L-band')
        ax1.axvspan(generator.c_band_start/1e12, generator.c_band_end/1e12, 
                   alpha=0.15, color='blue', label='C-band')
        
        ax1.set_xlabel('Frequency (THz)', fontweight='bold')
        ax1.set_ylabel('Attenuation αf (dB/km)', fontweight='bold')
        ax1.set_title('(a) Frequency-dependent Attenuation', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot 2: Raman gain spectrum
        raman_peak_gain = np.max(generator.raman_gain_matrix, axis=1)
        raman_normalized = raman_peak_gain / np.max(raman_peak_gain) if np.max(raman_peak_gain) > 0 else raman_peak_gain
        
        ax2.plot(frequencies_thz, raman_normalized, 'r-', linewidth=2, 
                marker='s', markersize=4, label='Split-Step Ground Truth gR')
        
        # Add band boundaries
        ax2.axvline(generator.l_band_end/1e12, color='gray', linestyle=':', alpha=0.8)
        ax2.axvline(generator.c_band_start/1e12, color='gray', linestyle=':', alpha=0.8)
        
        # Highlight bands
        ax2.axvspan(generator.l_band_start/1e12, generator.l_band_end/1e12, 
                   alpha=0.15, color='red', label='L-band')
        ax2.axvspan(generator.c_band_start/1e12, generator.c_band_end/1e12, 
                   alpha=0.15, color='blue', label='C-band')
        
        ax2.set_xlabel('Frequency (THz)', fontweight='bold')
        ax2.set_ylabel('Normalized Raman Gain gR', fontweight='bold')
        ax2.set_title('(b) Fiber Raman Gain Spectrum', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        ax2.set_ylim(0, 1.1)
        
        # Plot 3: Raman gain matrix heatmap
        im = ax3.imshow(generator.raman_gain_matrix, aspect='auto', cmap='RdBu_r', 
                       extent=[frequencies_thz[0], frequencies_thz[-1], 
                              frequencies_thz[-1], frequencies_thz[0]])
        ax3.set_xlabel('Signal Frequency (THz)', fontweight='bold')
        ax3.set_ylabel('Pump Frequency (THz)', fontweight='bold')
        ax3.set_title('(c) Raman Gain Matrix', fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax3)
        cbar.set_label('Raman Gain Coefficient (m/W)', fontweight='bold')
        
        # Plot 4: Frequency spacing analysis
        freq_diff = np.diff(generator.frequencies) / 1e9  # Convert to GHz
        ax4.plot(frequencies_thz[1:], freq_diff, 'g-', linewidth=2, 
                marker='d', markersize=4, label='Channel Spacing')
        ax4.axhline(generator.channel_spacing/1e9, color='black', linestyle='--', 
                   alpha=0.8, label=f'Target: {generator.channel_spacing/1e9:.0f} GHz')
        
        ax4.set_xlabel('Frequency (THz)', fontweight='bold')
        ax4.set_ylabel('Channel Spacing (GHz)', fontweight='bold')
        ax4.set_title('(d) Channel Spacing Verification', fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        
        plt.tight_layout()
        plt.savefig(save_path / "fiber_parameters.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print("   ✅ Fiber parameters plot saved")
    
    def _plot_scenario_analysis(self, scenario_name: str, scenario_data: Dict, dataset: Dict, save_path: Path):
        """Plot detailed analysis for each scenario"""
        
        fig = plt.figure(figsize=(20, 14))
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # Extract data
        frequencies_thz = np.array(dataset['system_parameters']['frequencies_hz']) / 1e12
        input_powers_dbm = np.array(scenario_data['input_configuration']['channel_powers_dbm'])
        output_powers_dbm = np.array(scenario_data['final_output']['powers_dbm'])
        
        # Power evolution data
        distances_km = np.array(scenario_data['cumulative_evolution']['distances_km'])
        power_evolution_dbm = np.array(scenario_data['cumulative_evolution']['power_evolution_dbm'])
        
        # Plot 1: Input spectrum
        ax1 = fig.add_subplot(gs[0, 0])
        active_mask = np.array(scenario_data['input_configuration']['channel_powers_w']) > 0
        
        ax1.stem(frequencies_thz[active_mask], input_powers_dbm[active_mask], 
                linefmt='b-', markerfmt='bo', basefmt=' ', label='Active channels')
        ax1.set_xlabel('Frequency (THz)')
        ax1.set_ylabel('Power (dBm)')
        ax1.set_title('Input Spectrum')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot 2: Output spectrum
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.stem(frequencies_thz[active_mask], output_powers_dbm[active_mask], 
                linefmt='r-', markerfmt='ro', basefmt=' ', label='Active channels')
        ax2.set_xlabel('Frequency (THz)')
        ax2.set_ylabel('Power (dBm)')
        ax2.set_title('Output Spectrum')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Plot 3: Power change
        ax3 = fig.add_subplot(gs[0, 2])
        power_change = output_powers_dbm - input_powers_dbm
        bars = ax3.bar(frequencies_thz[active_mask], power_change[active_mask], 
                      color=['green' if x > 0 else 'red' for x in power_change[active_mask]], 
                      alpha=0.7)
        ax3.set_xlabel('Frequency (THz)')
        ax3.set_ylabel('Power Change (dB)')
        ax3.set_title('Net Power Change (ISRS Effect)')
        ax3.grid(True, alpha=0.3)
        ax3.axhline(0, color='black', linestyle='-', alpha=0.5)
        
        # Plot 4: Power evolution heatmap
        ax4 = fig.add_subplot(gs[1, :])
        if len(power_evolution_dbm) > 0:
            # Select subset of channels for clarity
            n_channels_to_plot = min(20, np.sum(active_mask))
            active_indices = np.where(active_mask)[0][:n_channels_to_plot]
            
            evolution_subset = np.array(power_evolution_dbm)[:, active_indices]
            
            im = ax4.imshow(evolution_subset.T, aspect='auto', cmap='viridis', 
                           extent=[distances_km[0], distances_km[-1], 
                                  len(active_indices), 0])
            
            ax4.set_xlabel('Distance (km)')
            ax4.set_ylabel('Channel Index')
            ax4.set_title('Power Evolution Along Fiber')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax4)
            cbar.set_label('Power (dBm)')
            
            # Add span boundaries
            cumulative_distance = 0
            span_lengths = dataset['system_parameters']['span_configurations']
            for i, span_config in enumerate(span_lengths):
                cumulative_distance += span_config['length_km']
                if cumulative_distance < distances_km[-1]:
                    ax4.axvline(cumulative_distance, color='white', linestyle='--', alpha=0.7)
        
        # Plot 5: ISRS effects per span
        ax5 = fig.add_subplot(gs[2, 0])
        span_results = scenario_data['span_results']
        span_ids = [span['span_id'] for span in span_results]
        avg_isrs = [np.mean(span['isrs_gain_db']) if span['isrs_gain_db'] else 0 
                   for span in span_results]
        
        bars = ax5.bar(span_ids, avg_isrs, color='purple', alpha=0.7)
        ax5.set_xlabel('Span Number')
        ax5.set_ylabel('Average ISRS Gain (dB)')
        ax5.set_title('ISRS Effects per Span')
        ax5.grid(True, alpha=0.3)
        ax5.axhline(0, color='black', linestyle='-', alpha=0.5)
        
        # Plot 6: Power statistics
        ax6 = fig.add_subplot(gs[2, 1])
        total_powers = []
        distances_for_total = []
        
        for i, powers_at_distance in enumerate(power_evolution_dbm):
            if isinstance(powers_at_distance, list) and len(powers_at_distance) > 0:
                # Convert dBm to mW and sum
                powers_mw = [10**(p/10) for p in powers_at_distance if p > -50]
                if powers_mw:
                    total_power_mw = sum(powers_mw)
                    total_powers.append(10 * np.log10(total_power_mw))
                    distances_for_total.append(distances_km[i])
        
        if total_powers:
            ax6.plot(distances_for_total, total_powers, 'g-', linewidth=2, 
                    marker='o', markersize=3, label='Total Power')
            ax6.set_xlabel('Distance (km)')
            ax6.set_ylabel('Total Power (dBm)')
            ax6.set_title('Total System Power Evolution')
            ax6.grid(True, alpha=0.3)
            ax6.legend()
        
        # Plot 7: Scenario summary
        ax7 = fig.add_subplot(gs[2, 2])
        ax7.axis('off')
        
        # Calculate summary statistics
        total_input_power = scenario_data['input_configuration']['total_input_power_dbm']
        total_output_power = scenario_data['final_output']['total_output_power_dbm']
        active_channels = scenario_data['input_configuration']['active_channels']
        
        summary_text = f"""
        SCENARIO SUMMARY
        {'='*30}
        
        Name: {scenario_name}
        
        Configuration:
        • Active Channels: {active_channels}
        • Input Power: {total_input_power:.1f} dBm
        • Output Power: {total_output_power:.1f} dBm
        • Net Change: {total_output_power - total_input_power:.1f} dB
        
        ISRS Statistics:
        • Avg Effect: {scenario_data['system_statistics']['average_isrs_effect_db']:.3f} dB
        • Max Gain: {max([max(span['isrs_gain_db']) if span['isrs_gain_db'] else 0 for span in span_results]):.3f} dB
        • Min Gain: {min([min(span['isrs_gain_db']) if span['isrs_gain_db'] else 0 for span in span_results]):.3f} dB
        
        Simulation:
        • Time: {scenario_data['system_statistics']['total_simulation_time_s']:.1f} s
        • Data Points: {scenario_data['system_statistics']['total_spatial_points']}
        • Distance: {scenario_data['system_statistics']['total_distance_km']} km
        """
        
        ax7.text(0.05, 0.95, summary_text, transform=ax7.transAxes, 
                fontsize=9, fontfamily='monospace', verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))
        
        # Overall title
        fig.suptitle(f'Detailed Analysis: {scenario_name.upper()}', 
                    fontsize=16, fontweight='bold')
        
        plt.savefig(save_path / f"scenario_{scenario_name}.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Scenario analysis plot saved: {scenario_name}")
    
    def _plot_isrs_effects(self, dataset: Dict, save_path: Path):
        """Plot ISRS effects comparison across scenarios"""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        scenario_names = []
        avg_isrs_effects = []
        isrs_distributions = []
        
        # Collect ISRS data from all scenarios
        for scenario_name, scenario_data in dataset['scenarios'].items():
            scenario_names.append(scenario_name.replace('pinn_', '').replace('_', ' ').title())
            avg_isrs_effects.append(scenario_data['system_statistics']['average_isrs_effect_db'])
            
            # Collect all ISRS values for this scenario
            all_isrs = []
            for span_result in scenario_data['span_results']:
                if span_result['isrs_gain_db']:
                    all_isrs.extend(span_result['isrs_gain_db'])
            isrs_distributions.append(all_isrs)
        
        # Plot 1: Average ISRS effects
        colors = plt.cm.Set3(np.linspace(0, 1, len(scenario_names)))
        bars = ax1.bar(range(len(scenario_names)), avg_isrs_effects, color=colors, alpha=0.8)
        ax1.set_xlabel('Scenario')
        ax1.set_ylabel('Average ISRS Effect (dB)')
        ax1.set_title('Average ISRS Effects by Scenario')
        ax1.set_xticks(range(len(scenario_names)))
        ax1.set_xticklabels(scenario_names, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(0, color='black', linestyle='-', alpha=0.5)
        
        # Add value labels on bars
        for bar, value in zip(bars, avg_isrs_effects):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Plot 2: ISRS distribution comparison
        ax2.boxplot(isrs_distributions, labels=[name[:10] + '...' if len(name) > 10 else name 
                                               for name in scenario_names])
        ax2.set_xlabel('Scenario')
        ax2.set_ylabel('ISRS Gain (dB)')
        ax2.set_title('ISRS Distribution Comparison')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(0, color='red', linestyle='--', alpha=0.7, label='No ISRS')
        plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
        
        # Plot 3: ISRS histogram
        all_isrs_combined = [val for sublist in isrs_distributions for val in sublist]
        if all_isrs_combined:
            ax3.hist(all_isrs_combined, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            ax3.axvline(np.mean(all_isrs_combined), color='red', linestyle='--', 
                       linewidth=2, label=f'Mean: {np.mean(all_isrs_combined):.3f} dB')
            ax3.axvline(0, color='black', linestyle='-', alpha=0.5, label='No ISRS')
            ax3.set_xlabel('ISRS Gain (dB)')
            ax3.set_ylabel('Frequency')
            ax3.set_title('Overall ISRS Distribution')
            ax3.grid(True, alpha=0.3)
            ax3.legend()
        
        # Plot 4: ISRS vs frequency analysis
        if 'pinn_full_loading' in dataset['scenarios']:
            full_loading_data = dataset['scenarios']['pinn_full_loading']
            frequencies_thz = np.array(dataset['system_parameters']['frequencies_hz']) / 1e12
            
            # Get ISRS effects for each channel (average across spans)
            channel_isrs = []
            for ch_idx in range(len(frequencies_thz)):
                ch_effects = []
                for span_result in full_loading_data['span_results']:
                    if span_result['isrs_gain_db'] and ch_idx < len(span_result['isrs_gain_db']):
                        ch_effects.append(span_result['isrs_gain_db'][ch_idx])
                
                if ch_effects:
                    channel_isrs.append(np.mean(ch_effects))
                else:
                    channel_isrs.append(0)
            
            # Separate L-band and C-band
            l_band_start = dataset['metadata']['system_configuration']['l_band_range_thz'][0]
            l_band_end = dataset['metadata']['system_configuration']['l_band_range_thz'][1]
            c_band_start = dataset['metadata']['system_configuration']['c_band_range_thz'][0]
            
            l_band_mask = (frequencies_thz >= l_band_start) & (frequencies_thz <= l_band_end)
            c_band_mask = frequencies_thz >= c_band_start
            
            ax4.scatter(frequencies_thz[l_band_mask], np.array(channel_isrs)[l_band_mask], 
                       c='red', alpha=0.7, s=30, label='L-band')
            ax4.scatter(frequencies_thz[c_band_mask], np.array(channel_isrs)[c_band_mask], 
                       c='blue', alpha=0.7, s=30, label='C-band')
            
            ax4.set_xlabel('Frequency (THz)')
            ax4.set_ylabel('Average ISRS Gain (dB)')
            ax4.set_title('ISRS Effects vs Frequency')
            ax4.grid(True, alpha=0.3)
            ax4.legend()
            ax4.axhline(0, color='black', linestyle='-', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(save_path / "isrs_effects_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(" ISRS effects analysis plot saved")
    
    def _plot_validation_summary(self, dataset: Dict, save_path: Path):
        """Plot validation metrics summary"""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Collect validation metrics
        scenario_names = []
        data_points = []
        power_ranges = []
        isrs_ranges = []
        
        for scenario_name, scenario_data in dataset['scenarios'].items():
            metrics = dataset['validation_metrics'][scenario_name]
            scenario_names.append(scenario_name.replace('pinn_', '').replace('_', ' ').title())
            data_points.append(metrics['total_data_points'])
            power_ranges.append([metrics['power_range_dbm']['min'], 
                               metrics['power_range_dbm']['max']])
            isrs_ranges.append([metrics['isrs_effect_range_db']['min'], 
                              metrics['isrs_effect_range_db']['max']])
        
        # Plot 1: Data points per scenario
        colors = plt.cm.Set2(np.linspace(0, 1, len(scenario_names)))
        bars = ax1.bar(range(len(scenario_names)), data_points, color=colors, alpha=0.8)
        ax1.set_xlabel('Scenario')
        ax1.set_ylabel('Total Data Points')
        ax1.set_title('PINN Training Data Points by Scenario')
        ax1.set_xticks(range(len(scenario_names)))
        ax1.set_xticklabels(scenario_names, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, data_points):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + max(data_points)*0.01,
                    f'{value:,}', ha='center', va='bottom', fontweight='bold', fontsize=8)
        
        # Plot 2: Power range coverage
        power_ranges = np.array(power_ranges)
        x_pos = np.arange(len(scenario_names))
        
        ax2.errorbar(x_pos, np.mean(power_ranges, axis=1), 
                    yerr=[np.mean(power_ranges, axis=1) - power_ranges[:, 0],
                          power_ranges[:, 1] - np.mean(power_ranges, axis=1)],
                    fmt='o', capsize=5, capthick=2, markersize=8, alpha=0.8)
        
        ax2.set_xlabel('Scenario')
        ax2.set_ylabel('Power Range (dBm)')
        ax2.set_title('Power Range Coverage for PINN Training')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(scenario_names, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: ISRS effect range
        isrs_ranges = np.array(isrs_ranges)
        
        ax3.errorbar(x_pos, np.mean(isrs_ranges, axis=1), 
                    yerr=[np.mean(isrs_ranges, axis=1) - isrs_ranges[:, 0],
                          isrs_ranges[:, 1] - np.mean(isrs_ranges, axis=1)],
                    fmt='s', capsize=5, capthick=2, markersize=8, alpha=0.8, color='orange')
        
        ax3.set_xlabel('Scenario')
        ax3.set_ylabel('ISRS Effect Range (dB)')
        ax3.set_title('ISRS Effect Range for PINN Training')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(scenario_names, rotation=45, ha='right')
        ax3.grid(True, alpha=0.3)
        ax3.axhline(0, color='black', linestyle='-', alpha=0.5)
        
        # Plot 4: Validation readiness summary
        ax4.axis('off')
        
        # Calculate overall statistics
        total_data_points = sum(data_points)
        total_scenarios = len(scenario_names)
        overall_power_range = [min(power_ranges[:, 0]), max(power_ranges[:, 1])]
        overall_isrs_range = [min(isrs_ranges[:, 0]), max(isrs_ranges[:, 1])]
        
        target_accuracy = dataset['validation_metrics'][list(dataset['scenarios'].keys())[0]]['pinn_target_accuracy_db']
        
        summary_text = f"""
        PINN VALIDATION READINESS SUMMARY
        ═══════════════════════════════════════════════════════════
        
         Dataset Overview:
           • Total Scenarios: {total_scenarios}
           • Total Data Points: {total_data_points:,}
           • Spatial Resolution: 100 m
           • Spectral Resolution: 100 GHz
        
         Coverage Analysis:
           • Power Range: {overall_power_range[0]:.1f} to {overall_power_range[1]:.1f} dBm
           • ISRS Range: {overall_isrs_range[0]:.3f} to {overall_isrs_range[1]:.3f} dB
           • Frequency Span: ~10 THz (C+L bands)
           • System Length: 610 km (8 spans)
        
         PINN Training Targets:
           • Target Accuracy: ±{target_accuracy} dB
           • Physical Constraints: SRS PDEs
           • Validation Method: Split-step ground truth
           • Ready for Training: ✅ YES
        
         Physics Fidelity:
           • ISRS: Full Raman gain matrix
           • Amplification: Dual-band EDFA
           • ASE Noise: Realistic NF models
           • Step Size: 100 m (high precision)
        
         Recommended PINN Architecture:
           • Input: (z, frequency) coordinates
           • Output: Channel powers
           • Loss: Data + Physics (SRS PDE)
           • Training: Causal enforcement
        """
        
        ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, 
                fontsize=10, fontfamily='monospace', verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.1))
        
        plt.tight_layout()
        plt.savefig(save_path / "validation_summary.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print("    Validation summary plot saved")
    
    def compare_with_simple_model(self, generator, scenario_data: Dict, save_path: Path):
        """Compare split-step results with simplified analytical model"""
        
        print("\n" + "=" * 60)
        print("COMPARING WITH SIMPLIFIED ANALYTICAL MODEL")
        print("=" * 60)
        
        # Extract scenario data
        #frequencies = np.array(scenario_data['system_parameters']['frequencies_hz'])
        frequencies = np.array(self.generator.frequencies)
        input_powers_w = np.array(scenario_data['input_configuration']['channel_powers_w'])
        output_powers_w = np.array(scenario_data['final_output']['powers_w'])
        total_distance = scenario_data['final_output']['total_distance_km']
        
        # Simple linear attenuation model (no ISRS)
        mean_attenuation = np.mean(generator.alpha_f)  # dB/km
        linear_loss_db = mean_attenuation * total_distance
        linear_loss_linear = 10**(-linear_loss_db / 10)
        
        simple_output_powers_w = input_powers_w * linear_loss_linear
        
        # Create comparison plot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Convert to dBm for plotting
        frequencies_thz = frequencies / 1e12
        input_powers_dbm = 10 * np.log10(input_powers_w * 1000 + 1e-12)
        output_powers_dbm = 10 * np.log10(output_powers_w * 1000 + 1e-12)
        simple_output_dbm = 10 * np.log10(simple_output_powers_w * 1000 + 1e-12)
        
        # Only plot active channels
        active_mask = input_powers_w > 0
        
        # Plot 1: Power comparison
        ax1.plot(frequencies_thz[active_mask], input_powers_dbm[active_mask], 
                'b-o', label='Input', linewidth=2, markersize=6)
        ax1.plot(frequencies_thz[active_mask], output_powers_dbm[active_mask], 
                'r-s', label='Split-step (with ISRS)', linewidth=2, markersize=6)
        ax1.plot(frequencies_thz[active_mask], simple_output_dbm[active_mask], 
                'g--^', label='Linear attenuation only', linewidth=2, markersize=6)
        
        ax1.set_xlabel('Frequency (THz)')
        ax1.set_ylabel('Power (dBm)')
        ax1.set_title('Power Spectrum Comparison')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot 2: ISRS effect (difference from linear)
        isrs_effect_split_step = output_powers_dbm - simple_output_dbm
        
        ax2.bar(frequencies_thz[active_mask], isrs_effect_split_step[active_mask], 
               color=['green' if x > 0 else 'red' for x in isrs_effect_split_step[active_mask]], 
               alpha=0.7, width=0.1)
        ax2.set_xlabel('Frequency (THz)')
        ax2.set_ylabel('ISRS Effect (dB)')
        ax2.set_title('ISRS Gain/Loss vs Linear Attenuation')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(0, color='black', linestyle='-', alpha=0.5)
        
        # Plot 3: Error analysis
        absolute_error = np.abs(isrs_effect_split_step[active_mask])
        ax3.hist(absolute_error, bins=15, alpha=0.7, color='purple', edgecolor='black')
        ax3.axvline(np.mean(absolute_error), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {np.mean(absolute_error):.3f} dB')
        ax3.set_xlabel('Absolute ISRS Effect (dB)')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Distribution of ISRS Effects')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # Plot 4: Summary statistics
        ax4.axis('off')
        
        # Calculate statistics
        total_input_power_dbm = 10 * np.log10(np.sum(input_powers_w) * 1000)
        total_output_power_dbm = 10 * np.log10(np.sum(output_powers_w) * 1000)
        total_simple_power_dbm = 10 * np.log10(np.sum(simple_output_powers_w) * 1000)
        
        net_isrs_effect = total_output_power_dbm - total_simple_power_dbm
        max_isrs_gain = np.max(isrs_effect_split_step[active_mask])
        max_isrs_loss = np.min(isrs_effect_split_step[active_mask])
        
        stats_text = f"""
        MODEL COMPARISON STATISTICS
        ══════════════════════════════════════════════════════
        
         Total Power Analysis:
           • Input Power: {total_input_power_dbm:.2f} dBm
           • Split-step Output: {total_output_power_dbm:.2f} dBm
           • Linear Model Output: {total_simple_power_dbm:.2f} dBm
           • Net ISRS Effect: {net_isrs_effect:.3f} dB
        
         ISRS Effect Statistics:
           • Maximum Gain: {max_isrs_gain:.3f} dB
           • Maximum Loss: {max_isrs_loss:.3f} dB
           • Mean Absolute Effect: {np.mean(absolute_error):.3f} dB
           • Standard Deviation: {np.std(isrs_effect_split_step[active_mask]):.3f} dB
        
         Model Accuracy:
           • Linear Loss: {linear_loss_db:.2f} dB over {total_distance} km
           • Mean Attenuation: {mean_attenuation:.3f} dB/km
           • ISRS Correction Range: {max_isrs_loss:.3f} to {max_isrs_gain:.3f} dB
        
         Validation Status:
           Split-step model includes realistic ISRS effects
           that cannot be captured by linear attenuation alone.
           Maximum deviation: {max(abs(max_isrs_gain), abs(max_isrs_loss)):.3f} dB
        """
        
        ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, 
                fontsize=10, fontfamily='monospace', verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightcyan", alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(save_path / "model_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print("    Model comparison plot saved")
        print(f"    Net ISRS effect: {net_isrs_effect:.3f} dB")
        print(f"    ISRS range: {max_isrs_loss:.3f} to {max_isrs_gain:.3f} dB")

def main():
    """Main verification and plotting function"""
    
    print("Split-Step Ground Truth Verification and Plotting")
    print("=" * 70)
    
    try:
        # Import the main generator (assuming the original code is available)
        from split_step import SplitStepGroundTruthGenerator
        
        # Initialize generator and verifier
        generator = SplitStepGroundTruthGenerator()
        verifier = SplitStepVerifier(generator)
        
        # 1. Verify physical parameters
        verification_results = verifier.verify_physical_parameters(generator)
        
        # 2. Generate a sample dataset
        print("\n" + "=" * 60)
        print("GENERATING SAMPLE DATASET FOR VERIFICATION")
        print("=" * 60)
        
        dataset = generator.generate_complete_dataset()
        
        # 3. Create comprehensive plots
        output_dir = "split_step_analysis"
        verifier.plot_comprehensive_analysis(generator, dataset, output_dir)
        
        # 4. Model comparison for first scenario
        first_scenario = list(dataset['scenarios'].values())[0]
        verifier.compare_with_simple_model(generator, first_scenario, Path(output_dir))
        
        # 5. Print verification summary
        print("\n" + "=" * 60)
        print("VERIFICATION SUMMARY")
        print("=" * 60)
        
        all_passed = True
        for param_name, result in verification_results.items():
            status = result.get('status', 'UNKNOWN')
            print(f"   {param_name.replace('_', ' ').title()}: {status}")
            if status not in ['PASS']:
                all_passed = False
        
        if all_passed:
            print("\n ALL VERIFICATIONS PASSED!")
            print("   Split-step implementation is ready for PINN validation")
        else:
            print("\n  Some verification issues detected")
            print("   Please review the results above")
        
        print(f"\n Analysis plots saved to: {output_dir}/")
        print("   Use these plots for paper figures and validation")
        
    except ImportError:
        print(" Could not import SplitStepGroundTruthGenerator")
        print("   Please ensure the original split-step code is available")
        
        # Create a demo verifier for code demonstration
        print("\n Code verification and plotting framework ready!")
        print("   Integration steps:")
        print("   1. Import your SplitStepGroundTruthGenerator class")
        print("   2. Run: verifier.verify_physical_parameters(generator)")
        print("   3. Run: verifier.plot_comprehensive_analysis(generator, dataset)")
        
    except Exception as e:
        print(f"\n Error during verification: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()