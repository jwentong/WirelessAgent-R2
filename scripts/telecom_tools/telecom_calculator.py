# -*- coding: utf-8 -*-
"""
Telecommunications Calculator Module

Provides precise mathematical calculations for telecommunications problems,
including special functions that LLMs often get wrong:

1. Error functions: erfc(x), Q(x), erf(x)
2. Bessel functions: J_n(x), I_n(x), Y_n(x)
3. Marcum Q-function: Q_M(a, b)
4. Common telecom formulas: Shannon capacity, BER calculations, etc.

This module is designed to be used as a tool by ToolAgent for precise
numerical calculations.
"""

import math
from typing import Union, Tuple, Optional, Dict, Any
from scipy import special
from scipy import integrate
import numpy as np


class TelecomCalculator:
    """
    Precise calculator for telecommunications formulas.
    
    All methods return results with explanations for transparency.
    """
    
    # ==================== Error Functions ====================
    
    @staticmethod
    def erfc(x: float) -> Dict[str, Any]:
        """
        Complementary error function: erfc(x) = 1 - erf(x)
        
        Used in BER calculations for coherent detection.
        
        Args:
            x: Input value
            
        Returns:
            Dict with 'value', 'formula', 'explanation'
        """
        result = special.erfc(x)
        return {
            'value': result,
            'formula': f'erfc({x})',
            'explanation': f'Complementary error function: erfc({x}) = {result:.10e}',
            'related': 'Used in BPSK BER = 0.5 * erfc(sqrt(Eb/N0))'
        }
    
    @staticmethod
    def erf(x: float) -> Dict[str, Any]:
        """
        Error function: erf(x) = (2/sqrt(π)) * ∫₀ˣ exp(-t²) dt
        """
        result = special.erf(x)
        return {
            'value': result,
            'formula': f'erf({x})',
            'explanation': f'Error function: erf({x}) = {result:.10e}'
        }
    
    @staticmethod
    def Q_function(x: float) -> Dict[str, Any]:
        """
        Q-function: Q(x) = 0.5 * erfc(x / sqrt(2))
        
        The Gaussian Q-function gives the tail probability of standard normal.
        
        Args:
            x: Input value
            
        Returns:
            Dict with value and explanation
        """
        result = 0.5 * special.erfc(x / math.sqrt(2))
        return {
            'value': result,
            'formula': f'Q({x}) = 0.5 * erfc({x}/√2)',
            'explanation': f'Gaussian Q-function: Q({x}) = {result:.10e}',
            'related': 'Q(x) = P(Z > x) for standard normal Z'
        }
    
    @staticmethod
    def Q_inverse(p: float) -> Dict[str, Any]:
        """
        Inverse Q-function: given probability p, find x such that Q(x) = p
        
        Args:
            p: Probability value (0 < p < 1)
            
        Returns:
            x such that Q(x) = p
        """
        if p <= 0 or p >= 1:
            return {'value': None, 'error': 'p must be between 0 and 1'}
        
        # Q(x) = 0.5 * erfc(x/sqrt(2))
        # erfc(x/sqrt(2)) = 2p
        # x/sqrt(2) = erfc_inv(2p)
        # x = sqrt(2) * erfc_inv(2p)
        result = math.sqrt(2) * special.erfcinv(2 * p)
        return {
            'value': result,
            'formula': f'Q⁻¹({p})',
            'explanation': f'Inverse Q-function: Q⁻¹({p}) = {result:.6f}, meaning Q({result:.6f}) = {p}'
        }
    
    # ==================== Bessel Functions ====================
    
    @staticmethod
    def bessel_J(n: int, x: float) -> Dict[str, Any]:
        """
        Bessel function of the first kind: J_n(x)
        
        Used in FM modulation (sideband amplitudes) and Rician fading.
        
        Args:
            n: Order (integer)
            x: Argument
        """
        result = special.jv(n, x)
        return {
            'value': result,
            'formula': f'J_{n}({x})',
            'explanation': f'Bessel function first kind: J_{n}({x}) = {result:.10f}',
            'applications': [
                f'FM: Carrier amplitude ∝ J_0(β), sideband amplitudes ∝ J_n(β)',
                f'Rician fading: involves J_0 in pdf expression'
            ]
        }
    
    @staticmethod
    def bessel_I(n: int, x: float) -> Dict[str, Any]:
        """
        Modified Bessel function of the first kind: I_n(x)
        
        Used in Rician fading and Rice distribution.
        
        Args:
            n: Order (integer)
            x: Argument
        """
        result = special.iv(n, x)
        return {
            'value': result,
            'formula': f'I_{n}({x})',
            'explanation': f'Modified Bessel function first kind: I_{n}({x}) = {result:.10e}',
            'applications': [
                'Rician fading pdf: p(r) = (r/σ²) * exp(-(r²+s²)/(2σ²)) * I_0(rs/σ²)',
                'Marcum Q-function calculation'
            ]
        }
    
    @staticmethod
    def bessel_Y(n: int, x: float) -> Dict[str, Any]:
        """
        Bessel function of the second kind: Y_n(x)
        """
        result = special.yv(n, x)
        return {
            'value': result,
            'formula': f'Y_{n}({x})',
            'explanation': f'Bessel function second kind: Y_{n}({x}) = {result:.10e}'
        }
    
    # ==================== Marcum Q-Function ====================
    
    @staticmethod
    def marcum_Q(a: float, b: float, M: int = 1) -> Dict[str, Any]:
        """
        Marcum Q-function: Q_M(a, b)
        
        Used in Rician fading outage probability and detection theory.
        
        The generalized Marcum Q-function is:
        Q_M(a, b) = ∫_b^∞ x * (x/a)^(M-1) * exp(-(x² + a²)/2) * I_{M-1}(ax) dx
        
        For M=1 (standard Marcum Q):
        Q_1(a, b) = ∫_b^∞ x * exp(-(x² + a²)/2) * I_0(ax) dx
        
        Args:
            a: Non-centrality parameter (√(2K) for Rician K-factor)
            b: Threshold parameter
            M: Order (default 1)
        """
        # Try scipy's implementation first (available in newer scipy versions)
        try:
            result = special.marcumq(M, a, b)
        except AttributeError:
            # Fallback: numerical integration for Marcum Q_1
            if M == 1:
                # Q_1(a,b) = ∫_b^∞ x * exp(-(x² + a²)/2) * I_0(ax) dx
                def integrand(x):
                    return x * np.exp(-(x**2 + a**2) / 2) * special.iv(0, a * x)
                
                # Integrate from b to infinity (use large upper limit)
                upper_limit = max(b + 20, a + 20, 50)  # Ensure enough integration range
                result, _ = integrate.quad(integrand, b, upper_limit)
            else:
                # For higher orders, use recursive series approximation
                # This is an approximation - for production, consider more sophisticated methods
                result = TelecomCalculator._marcum_q_series(a, b, M)
        
        return {
            'value': result,
            'formula': f'Q_{M}({a}, {b})',
            'explanation': f'Marcum Q-function: Q_{M}({a}, {b}) = {result:.10e}',
            'applications': [
                'Rician fading: Outage probability = 1 - Q_1(√(2K), γ_th/σ)',
                'Detection theory: Detection probability calculations'
            ],
            'notes': [
                f'For Rician K-factor: a = √(2K) where K = LOS_power/scattered_power',
                f'Outage probability at threshold γ_th: P_out = 1 - Q_1(a, b)'
            ]
        }
    
    @staticmethod
    def _marcum_q_series(a: float, b: float, M: int, max_terms: int = 100) -> float:
        """
        Series expansion for Marcum Q-function.
        
        Uses the series: Q_M(a,b) = exp(-(a²+b²)/2) * Σ (a/b)^n * I_n(ab)
        """
        if b == 0:
            return 1.0
        
        total = 0.0
        ab = a * b
        
        for n in range(max_terms):
            term = (a / b) ** (n + M - 1) * special.iv(n + M - 1, ab)
            total += term
            if abs(term) < 1e-15:  # Convergence check
                break
        
        result = np.exp(-(a**2 + b**2) / 2) * total
        return min(max(result, 0.0), 1.0)  # Clamp to [0, 1]
    
    # ==================== BER Calculations ====================
    
    @staticmethod
    def ber_bpsk_coherent(Eb_N0_dB: float) -> Dict[str, Any]:
        """
        BER for coherent BPSK: P_b = 0.5 * erfc(√(Eb/N0))
        
        Args:
            Eb_N0_dB: Energy per bit to noise ratio in dB
        """
        Eb_N0_linear = 10 ** (Eb_N0_dB / 10)
        ber = 0.5 * special.erfc(math.sqrt(Eb_N0_linear))
        return {
            'value': ber,
            'Eb_N0_linear': Eb_N0_linear,
            'formula': f'P_b = 0.5 * erfc(√{Eb_N0_linear:.4f}) = 0.5 * erfc({math.sqrt(Eb_N0_linear):.4f})',
            'explanation': f'BPSK BER at Eb/N0 = {Eb_N0_dB} dB: P_b = {ber:.6e}'
        }
    
    @staticmethod
    def ber_bfsk_coherent(Eb_N0_dB: float) -> Dict[str, Any]:
        """
        BER for coherent BFSK: P_b = 0.5 * erfc(√(Eb/(2*N0)))
        
        Note: BFSK has 3dB penalty compared to BPSK!
        """
        Eb_N0_linear = 10 ** (Eb_N0_dB / 10)
        ber = 0.5 * special.erfc(math.sqrt(Eb_N0_linear / 2))
        return {
            'value': ber,
            'Eb_N0_linear': Eb_N0_linear,
            'formula': f'P_b = 0.5 * erfc(√({Eb_N0_linear:.4f}/2))',
            'explanation': f'BFSK BER at Eb/N0 = {Eb_N0_dB} dB: P_b = {ber:.6e}',
            'note': 'BFSK has 3dB penalty vs BPSK (divide Eb/N0 by 2 in erfc argument)'
        }
    
    @staticmethod
    def ber_bfsk_noncoherent(Eb_N0_dB: float) -> Dict[str, Any]:
        """
        BER for non-coherent BFSK: P_b = 0.5 * exp(-Eb/(2*N0))
        """
        Eb_N0_linear = 10 ** (Eb_N0_dB / 10)
        ber = 0.5 * math.exp(-Eb_N0_linear / 2)
        return {
            'value': ber,
            'Eb_N0_linear': Eb_N0_linear,
            'formula': f'P_b = 0.5 * exp(-{Eb_N0_linear:.4f}/2)',
            'explanation': f'Non-coherent BFSK BER at Eb/N0 = {Eb_N0_dB} dB: P_b = {ber:.6e}'
        }
    
    @staticmethod
    def ber_dpsk(Eb_N0_dB: float) -> Dict[str, Any]:
        """
        BER for DPSK: P_b = 0.5 * exp(-Eb/N0)
        """
        Eb_N0_linear = 10 ** (Eb_N0_dB / 10)
        ber = 0.5 * math.exp(-Eb_N0_linear)
        return {
            'value': ber,
            'formula': f'P_b = 0.5 * exp(-{Eb_N0_linear:.4f})',
            'explanation': f'DPSK BER at Eb/N0 = {Eb_N0_dB} dB: P_b = {ber:.6e}'
        }
    
    # ==================== Channel Capacity ====================
    
    @staticmethod
    def shannon_capacity(bandwidth_Hz: float, snr_dB: float) -> Dict[str, Any]:
        """
        Shannon channel capacity: C = B * log2(1 + SNR)
        
        Args:
            bandwidth_Hz: Channel bandwidth in Hz
            snr_dB: Signal-to-noise ratio in dB
        """
        snr_linear = 10 ** (snr_dB / 10)
        capacity_bps = bandwidth_Hz * math.log2(1 + snr_linear)
        return {
            'value': capacity_bps,
            'value_kbps': capacity_bps / 1000,
            'value_Mbps': capacity_bps / 1e6,
            'snr_linear': snr_linear,
            'formula': f'C = {bandwidth_Hz} * log2(1 + {snr_linear:.4f})',
            'explanation': f'Shannon capacity: C = {capacity_bps:.2f} bps = {capacity_bps/1000:.2f} kbps'
        }
    
    # ==================== Fading Channel ====================
    
    @staticmethod
    def rayleigh_outage_probability(gamma_threshold: float, gamma_avg: float) -> Dict[str, Any]:
        """
        Rayleigh fading outage probability: P_out = 1 - exp(-γ_th/γ_avg)
        
        Args:
            gamma_threshold: Threshold SNR (linear, not dB)
            gamma_avg: Average SNR (linear, not dB)
        """
        p_out = 1 - math.exp(-gamma_threshold / gamma_avg)
        return {
            'value': p_out,
            'formula': f'P_out = 1 - exp(-{gamma_threshold}/{gamma_avg})',
            'explanation': f'Rayleigh outage probability: P_out = {p_out:.6e}'
        }
    
    @staticmethod
    def rayleigh_level_crossing_rate(rho: float, f_D: float) -> Dict[str, Any]:
        """
        Rayleigh fading level crossing rate: N_R = √(2π) * f_D * ρ * exp(-ρ²)
        
        Args:
            rho: Normalized threshold ρ = R/R_rms (threshold/RMS level)
            f_D: Maximum Doppler frequency in Hz
        """
        N_R = math.sqrt(2 * math.pi) * f_D * rho * math.exp(-rho**2)
        return {
            'value': N_R,
            'formula': f'N_R = √(2π) * {f_D} * {rho} * exp(-{rho}²)',
            'explanation': f'Level crossing rate: N_R = {N_R:.4f} crossings/second'
        }
    
    @staticmethod
    def rayleigh_average_fade_duration(rho: float, f_D: float) -> Dict[str, Any]:
        """
        Average fade duration: τ = (exp(ρ²) - 1) / (ρ * f_D * √(2π))
        
        Args:
            rho: Normalized threshold ρ = R/R_rms
            f_D: Maximum Doppler frequency in Hz
        """
        if rho == 0:
            return {'value': float('inf'), 'error': 'ρ cannot be zero'}
        
        tau = (math.exp(rho**2) - 1) / (rho * f_D * math.sqrt(2 * math.pi))
        return {
            'value': tau,
            'formula': f'τ = (exp({rho}²) - 1) / ({rho} * {f_D} * √(2π))',
            'explanation': f'Average fade duration: τ = {tau:.6e} seconds'
        }
    
    @staticmethod
    def rician_outage_probability(K: float, gamma_threshold: float, gamma_avg: float) -> Dict[str, Any]:
        """
        Rician fading outage probability using Marcum Q-function.
        
        P_out = 1 - Q_1(√(2K), √(2(K+1)γ_th/γ_avg))
        
        Args:
            K: Rician K-factor (ratio of LOS to scattered power)
            gamma_threshold: Threshold SNR (linear)
            gamma_avg: Average SNR (linear)
        """
        a = math.sqrt(2 * K)
        b = math.sqrt(2 * (K + 1) * gamma_threshold / gamma_avg)
        
        # Use our marcum_Q method which has fallback implementation
        marcum_result = TelecomCalculator.marcum_Q(a, b, M=1)
        marcum_q = marcum_result['value']
        p_out = 1 - marcum_q
        
        return {
            'value': p_out,
            'marcum_Q': marcum_q,
            'a': a,
            'b': b,
            'formula': f'P_out = 1 - Q_1({a:.4f}, {b:.4f})',
            'explanation': f'Rician outage (K={K}): P_out = {p_out:.6e}'
        }
    
    # ==================== FM Modulation ====================
    
    @staticmethod
    def fm_bessel_coefficients(beta: float, n_max: int = 10) -> Dict[str, Any]:
        """
        FM Bessel coefficients for carrier and sidebands.
        
        Carrier amplitude: J_0(β)
        n-th sideband amplitude: J_n(β)
        
        Args:
            beta: Modulation index
            n_max: Maximum sideband order to compute
        """
        coefficients = {}
        for n in range(n_max + 1):
            coefficients[f'J_{n}'] = special.jv(n, beta)
        
        # Find first zero crossings
        carrier_power = coefficients['J_0'] ** 2
        total_power = sum(c**2 for c in coefficients.values())
        
        return {
            'coefficients': coefficients,
            'carrier_power_fraction': carrier_power,
            'beta': beta,
            'explanation': f'FM with β={beta}: J_0={coefficients["J_0"]:.6f}, J_1={coefficients["J_1"]:.6f}',
            'note': 'First sideband zero at β ≈ 3.832 (J_1(β) = 0)'
        }
    
    @staticmethod
    def fm_carson_bandwidth(delta_f: float, f_m: float) -> Dict[str, Any]:
        """
        Carson's rule for FM bandwidth: B ≈ 2(Δf + f_m)
        
        Args:
            delta_f: Peak frequency deviation in Hz
            f_m: Maximum modulating frequency in Hz
        """
        beta = delta_f / f_m
        bandwidth = 2 * (delta_f + f_m)
        
        return {
            'value': bandwidth,
            'beta': beta,
            'formula': f'B = 2 * ({delta_f} + {f_m}) = {bandwidth}',
            'explanation': f"Carson's rule: B = {bandwidth} Hz, β = {beta:.2f}"
        }


# Convenience function for external use
def calculate(operation: str, **kwargs) -> Dict[str, Any]:
    """
    Universal calculator interface for ToolAgent.
    
    Args:
        operation: Name of the calculation to perform
        **kwargs: Parameters for the calculation
        
    Returns:
        Result dictionary with value and explanation
    
    Example:
        calculate('Q_function', x=4.0)
        calculate('ber_bpsk_coherent', Eb_N0_dB=10)
        calculate('marcum_Q', a=2.0, b=1.5)
    """
    calc = TelecomCalculator()
    
    operations = {
        'erfc': calc.erfc,
        'erf': calc.erf,
        'Q_function': calc.Q_function,
        'Q_inverse': calc.Q_inverse,
        'bessel_J': calc.bessel_J,
        'bessel_I': calc.bessel_I,
        'bessel_Y': calc.bessel_Y,
        'marcum_Q': calc.marcum_Q,
        'ber_bpsk_coherent': calc.ber_bpsk_coherent,
        'ber_bfsk_coherent': calc.ber_bfsk_coherent,
        'ber_bfsk_noncoherent': calc.ber_bfsk_noncoherent,
        'ber_dpsk': calc.ber_dpsk,
        'shannon_capacity': calc.shannon_capacity,
        'rayleigh_outage_probability': calc.rayleigh_outage_probability,
        'rayleigh_level_crossing_rate': calc.rayleigh_level_crossing_rate,
        'rayleigh_average_fade_duration': calc.rayleigh_average_fade_duration,
        'rician_outage_probability': calc.rician_outage_probability,
        'fm_bessel_coefficients': calc.fm_bessel_coefficients,
        'fm_carson_bandwidth': calc.fm_carson_bandwidth,
    }
    
    if operation not in operations:
        return {
            'error': f'Unknown operation: {operation}',
            'available_operations': list(operations.keys())
        }
    
    try:
        return operations[operation](**kwargs)
    except Exception as e:
        return {'error': str(e), 'operation': operation, 'kwargs': kwargs}


# Test
if __name__ == "__main__":
    print("=" * 60)
    print("Telecom Calculator Test")
    print("=" * 60)
    
    # Test Q-function
    result = calculate('Q_function', x=4.0)
    print(f"\nQ(4.0) = {result['value']:.10e}")
    print(f"  {result['explanation']}")
    
    # Test BPSK BER
    result = calculate('ber_bpsk_coherent', Eb_N0_dB=10)
    print(f"\nBPSK BER at 10dB = {result['value']:.6e}")
    
    # Test Marcum Q
    result = calculate('marcum_Q', a=2.0, b=1.5)
    print(f"\nMarcum Q_1(2.0, 1.5) = {result['value']:.6f}")
    
    # Test Bessel
    result = calculate('bessel_J', n=0, x=3.832)
    print(f"\nJ_0(3.832) = {result['value']:.6f} (should be near 0)")
    
    # Test Rician outage
    result = calculate('rician_outage_probability', K=3, gamma_threshold=1, gamma_avg=10)
    print(f"\nRician outage (K=3, γ_th=1, γ_avg=10): P_out = {result['value']:.6e}")
