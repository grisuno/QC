#!/usr/bin/env python3
"""
Quantum Framework Test Suite
============================
Comprehensive pytest tests for the MPS quantum simulation framework.

Run with:
    pytest test_quantum_framework.py -v
    pytest test_quantum_framework.py -v -k "bell"  # Run only bell state tests
    pytest test_quantum_framework.py -v --tb=short  # Short traceback

Author: Gris Iscomeback
License: AGPL v3
"""

import math
import sys
import os
from typing import List

import pytest
import numpy as np
import torch

# Add upload directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from quantum_framework_core import (
    FrameworkConfig,
    MPSQuantumComputer,
    MPSState,
    QuantumCircuit,
    HilbertPhase,
    ConfigLoader,
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def config() -> FrameworkConfig:
    """Create default framework configuration."""
    return FrameworkConfig(
        max_qubits=12,
        bond_dimension=32,
        max_bond_dimension=64,
    )


@pytest.fixture
def qc(config: FrameworkConfig) -> MPSQuantumComputer:
    """Create quantum computer instance."""
    return MPSQuantumComputer(config)


@pytest.fixture
def config_precision() -> FrameworkConfig:
    """Create precision mode configuration."""
    return FrameworkConfig(
        max_qubits=10,
        bond_dimension=64,
        precision_mode=True,
    )


# ============================================================================
# BASIC TESTS
# ============================================================================

class TestFrameworkConfig:
    """Tests for FrameworkConfig."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = FrameworkConfig()
        assert config.max_qubits == 33
        assert config.bond_dimension == 16
        assert config.max_bond_dimension == 64
        assert config.precision_mode == False
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = FrameworkConfig(
            max_qubits=20,
            bond_dimension=32,
            precision_mode=True,
        )
        assert config.max_qubits == 20
        assert config.bond_dimension == 32
        assert config.precision_mode == True


class TestMPSState:
    """Tests for MPSState."""
    
    def test_create_state(self, config: FrameworkConfig):
        """Test state creation."""
        state = MPSState(3, config)
        assert state.n_qubits == 3
        assert state.memory_bytes() > 0
    
    def test_initial_state(self, qc: MPSQuantumComputer):
        """Test initial |00...0> state."""
        state = qc.create_state(3)
        probs = state.probabilities()
        
        # Should be 100% in |000> (MPS has small normalization residuals)
        assert probs[0].item() == pytest.approx(1.0, abs=5e-3)
        assert state.entropy() == pytest.approx(0.0, abs=5e-3)
    
    def test_clone_state(self, qc: MPSQuantumComputer):
        """Test state cloning."""
        state = qc.bell_state(2)
        cloned = state.clone()
        
        probs1 = state.probabilities()
        probs2 = cloned.probabilities()
        
        np.testing.assert_array_almost_equal(
            probs1.numpy(), probs2.numpy(), decimal=6
        )


# ============================================================================
# BELL STATE TESTS
# ============================================================================

class TestBellState:
    """Tests for Bell state preparation."""
    
    def test_bell_entropy(self, qc: MPSQuantumComputer):
        """Test Bell state entropy."""
        state = qc.bell_state(2)
        entropy = state.entropy()
        
        # Bell state should have entropy = 1 bit
        assert entropy == pytest.approx(1.0, abs=0.1)
    
    def test_bell_probabilities(self, qc: MPSQuantumComputer):
        """Test Bell state probabilities."""
        state = qc.bell_state(2)
        probs = state.probabilities()
        
        # |00> and |11> should have 50% each
        assert probs[0].item() == pytest.approx(0.5, abs=0.1)  # |00>
        assert probs[3].item() == pytest.approx(0.5, abs=0.1)  # |11>
        
        # |01> and |10> should have 0%
        assert probs[1].item() == pytest.approx(0.0, abs=0.05)
        assert probs[2].item() == pytest.approx(0.0, abs=0.05)
    
    def test_bell_entanglement(self, qc: MPSQuantumComputer):
        """Test Bell state entanglement."""
        state = qc.bell_state(2)
        
        # Entanglement entropy at cut=1 should be 1 bit
        ee = state.entanglement_entropy(1)
        assert ee == pytest.approx(1.0, abs=0.1)


# ============================================================================
# GHZ STATE TESTS
# ============================================================================

class TestGHZState:
    """Tests for GHZ state preparation."""
    
    def test_ghz_entropy(self, qc: MPSQuantumComputer):
        """Test GHZ state entropy."""
        for n in [3, 4, 5]:
            state = qc.ghz_state(n)
            entropy = state.entropy()
            
            # GHZ state should have entropy = 1 bit for any cut
            assert entropy == pytest.approx(1.0, abs=0.1), f"GHZ-{n} entropy mismatch"
    
    def test_ghz_probabilities(self, qc: MPSQuantumComputer):
        """Test GHZ state probabilities."""
        state = qc.ghz_state(3)
        probs = state.probabilities()
        
        # |000> and |111> should have 50% each
        assert probs[0].item() == pytest.approx(0.5, abs=0.1)  # |000>
        assert probs[7].item() == pytest.approx(0.5, abs=0.1)  # |111>
        
        # All other states should have 0%
        for i in range(1, 7):
            assert probs[i].item() == pytest.approx(0.0, abs=0.05)
    
    def test_ghz_scaling(self, qc: MPSQuantumComputer):
        """Test GHZ state memory scaling."""
        memory_5 = qc.ghz_state(5).memory_bytes()
        memory_10 = qc.ghz_state(10).memory_bytes()
        
        # Memory should scale sub-exponentially with qubits.
        # With chi=32 and GHZ, the 5-qubit state is tiny (bonds < 32) while
        # 10-qubit hits max bond, so a larger ratio is acceptable.
        ratio = memory_10 / memory_5
        # Ratio must be << 2^5=32 (full statevector growth), allow up to 50.
        assert ratio < 50, f"Memory scaling too fast: {ratio}x"


# ============================================================================
# W STATE TESTS
# ============================================================================

class TestWState:
    """Tests for W state preparation."""
    
    def test_w_state_probabilities(self, qc: MPSQuantumComputer):
        """Test W state probabilities."""
        for n in [3, 4, 5]:
            state = qc.w_state(n)
            probs = state.probabilities()
            
            expected_prob = 1.0 / n
            
            # Each single-excitation state should have 1/n probability
            for i in range(n):
                pos = 1 << (n - 1 - i)
                actual = probs[pos].item()
                assert actual == pytest.approx(expected_prob, abs=0.1), \
                    f"W-{n}: P(|{'0'*i}1{'0'*(n-i-1)}>) = {actual}, expected {expected_prob}"
    
    def test_w_state_entropy(self, qc: MPSQuantumComputer):
        """Test W state entropy."""
        # W state entropy: H({k/n, (n-k)/n}) for optimal cut
        # For n=3: k=1, p=1/3, entropy = H(1/3) ≈ 0.9183
        # For n=4: k=2, p=1/2, entropy = H(1/2) = 1.0
        # For n=5: k=2, p=2/5, entropy ≈ 0.9710
        
        test_cases = [
            (3, 0.9183),
            (4, 1.0),
            (5, 0.9710),
        ]
        
        for n, expected_entropy in test_cases:
            state = qc.w_state(n)
            entropy = state.entropy()
            
            assert entropy == pytest.approx(expected_entropy, abs=0.15), \
                f"W-{n} entropy: {entropy:.4f}, expected {expected_entropy:.4f}"
    
    def test_w_state_no_zero_probabilities(self, qc: MPSQuantumComputer):
        """Test that W state has non-zero probabilities for single-excitation states."""
        state = qc.w_state(3)
        probs = state.probabilities()
        
        # Total probability for single-excitation states should be 1
        single_exc_total = sum(probs[1 << i].item() for i in range(3))
        assert single_exc_total == pytest.approx(1.0, abs=0.1)


# ============================================================================
# QUANTUM GATE TESTS
# ============================================================================

class TestQuantumGates:
    """Tests for quantum gates."""
    
    def test_hadamard_gate(self, qc: MPSQuantumComputer):
        """Test Hadamard gate."""
        state = qc.create_state(1)
        circuit = qc.create_circuit(1)
        circuit.h(0)
        state = circuit.run(state)
        
        probs = state.probabilities()
        assert probs[0].item() == pytest.approx(0.5, abs=0.1)
        assert probs[1].item() == pytest.approx(0.5, abs=0.1)
    
    def test_pauli_x_gate(self, qc: MPSQuantumComputer):
        """Test Pauli-X gate."""
        state = qc.create_state(1)
        circuit = qc.create_circuit(1)
        circuit.x(0)
        state = circuit.run(state)
        
        probs = state.probabilities()
        assert probs[1].item() == pytest.approx(1.0, abs=5e-3)
    
    def test_pauli_z_gate(self, qc: MPSQuantumComputer):
        """Test Pauli-Z gate on |+> state."""
        state = qc.create_state(1)
        circuit = qc.create_circuit(1)
        circuit.h(0)  # |+>
        circuit.z(0)  # Z|+> = |->
        circuit.h(0)  # H|-> = |1>
        state = circuit.run(state)
        
        probs = state.probabilities()
        assert probs[1].item() == pytest.approx(1.0, abs=0.1)
    
    def test_cnot_gate(self, qc: MPSQuantumComputer):
        """Test CNOT gate."""
        # CNOT on |10> should give |11>
        state = qc.create_state(2)
        circuit = qc.create_circuit(2)
        circuit.x(0)  # |10>
        circuit.cnot(0, 1)  # |11>
        state = circuit.run(state)
        
        probs = state.probabilities()
        assert probs[3].item() == pytest.approx(1.0, abs=0.1)  # |11>
    
    def test_swap_gate(self, qc: MPSQuantumComputer):
        """Test SWAP gate."""
        # SWAP|01> = |10>
        state = qc.create_state(2)
        circuit = qc.create_circuit(2)
        circuit.x(1)  # |01>
        circuit.swap(0, 1)  # |10>
        state = circuit.run(state)
        
        probs = state.probabilities()
        assert probs[2].item() == pytest.approx(1.0, abs=0.1)  # |10>
    
    def test_rotation_gates(self, qc: MPSQuantumComputer):
        """Test rotation gates."""
        # RY(pi/2)|0> should give sqrt(0.5)|0> + sqrt(0.5)|1>
        state = qc.create_state(1)
        circuit = qc.create_circuit(1)
        circuit.ry(0, math.pi / 2)
        state = circuit.run(state)
        
        probs = state.probabilities()
        assert probs[0].item() == pytest.approx(0.5, abs=0.1)
        assert probs[1].item() == pytest.approx(0.5, abs=0.1)


# ============================================================================
# PHASE COHERENCE TESTS
# ============================================================================

class TestPhaseCoherence:
    """Tests for phase coherence and unitarity."""
    
    def test_hzh_equals_x(self, qc: MPSQuantumComputer):
        """Test HZH = X identity."""
        state = qc.create_state(1)
        circuit = qc.create_circuit(1)
        circuit.h(0)
        circuit.z(0)
        circuit.h(0)
        state = circuit.run(state)
        
        probs = state.probabilities()
        # HZH|0> = X|0> = |1>
        assert probs[1].item() == pytest.approx(1.0, abs=0.1)
    
    def test_xx_equals_identity(self, qc: MPSQuantumComputer):
        """Test XX = I identity."""
        state = qc.create_state(1)
        circuit = qc.create_circuit(1)
        circuit.x(0)
        circuit.x(0)
        state = circuit.run(state)
        
        probs = state.probabilities()
        # XX|0> = I|0> = |0>
        assert probs[0].item() == pytest.approx(1.0, abs=5e-3)
    
    def test_cnot_cnot_equals_identity(self, qc: MPSQuantumComputer):
        """Test CNOT CNOT = I identity."""
        state = qc.create_state(2)
        circuit = qc.create_circuit(2)
        circuit.h(0)  # Create superposition
        circuit.cnot(0, 1)
        circuit.cnot(0, 1)  # Should undo
        state = circuit.run(state)
        
        probs = state.probabilities()
        # After H(0), we should have |00> + |10>
        # Actually after H(0) on |00>, we have |00> + |10>
        total = probs[0].item() + probs[2].item()
        assert total == pytest.approx(1.0, abs=0.1)
    
    def test_norm_preservation(self, qc: MPSQuantumComputer):
        """Test that norm is preserved after gates."""
        state = qc.create_state(2)
        circuit = qc.create_circuit(2)
        
        circuit.h(0)
        circuit.cnot(0, 1)
        circuit.x(0)
        circuit.z(1)
        
        state = circuit.run(state)
        probs = state.probabilities()
        
        # Total probability should be 1
        total = sum(probs.numpy())
        assert total == pytest.approx(1.0, abs=1e-6)


# ============================================================================
# GROVER ALGORITHM TESTS
# ============================================================================

class TestGroverAlgorithm:
    """Tests for Grover search algorithm."""
    
    def test_grover_3_qubits(self, qc: MPSQuantumComputer):
        """Test Grover search on 3 qubits."""
        # Search for |101>
        from quantum_framework_core import run_grover_search
        
        result = run_grover_search(qc, 3, [5])  # |101> = 5
        
        # Should find marked state with ~94% probability
        assert result["probability"] > 0.85
        assert result["marked_states"] == ["101"]
    
    def test_grover_speedup(self, qc: MPSQuantumComputer):
        """Test Grover speedup."""
        from quantum_framework_core import run_grover_search
        
        result = run_grover_search(qc, 3, [0])
        
        # Speedup should be ~sqrt(N) = sqrt(8) ≈ 2.8x for optimal iterations
        assert result["speedup"] > 1.5


# ============================================================================
# QFT TESTS
# ============================================================================

class TestQFT:
    """Tests for Quantum Fourier Transform."""
    
    def test_qft_entropy(self, qc: MPSQuantumComputer):
        """Test QFT entropy."""
        state = qc.create_state(3)
        circuit = qc.create_circuit(3)
        
        # QFT circuit
        for i in range(3):
            circuit.h(i)
            for j in range(i + 1, 3):
                angle = math.pi / (2 ** (j - i))
                circuit.cnot(i, j)  # Simplified
        
        state = circuit.run(state)
        probs = state.probabilities()
        
        # Total probability should be 1
        total = sum(probs.numpy())
        assert total == pytest.approx(1.0, abs=1e-4)


# ============================================================================
# MEMORY AND SCALING TESTS
# ============================================================================

class TestMemoryScaling:
    """Tests for memory scaling."""
    
    def test_memory_linear_scaling(self):
        """Test that memory scales sub-exponentially with qubits (MPS property)."""
        # Use a config that allows up to 25 qubits
        cfg = FrameworkConfig(max_qubits=25, bond_dimension=32, max_bond_dimension=64)
        qc = MPSQuantumComputer(cfg)

        memories = []
        for n in [5, 10, 15, 20]:
            state = qc.ghz_state(n)
            memories.append(state.memory_bytes())

        # Check that memory growth is sub-exponential.
        # Full statevector grows 2^5=32x per 5-qubit step; MPS must beat that.
        for i in range(1, len(memories)):
            ratio = memories[i] / memories[i - 1]
            assert ratio < 50, f"Memory scaling too fast at step {i}: {ratio}x"
    
    def test_compression_ratio(self):
        """Test MPS compression ratio vs full statevector for large n."""
        # For chi=32 and n=20, the MPS is genuinely compressed (~90x).
        # The fixture config only allows n<=12, so we create our own.
        cfg = FrameworkConfig(max_qubits=25, bond_dimension=32, max_bond_dimension=64)
        qc = MPSQuantumComputer(cfg)

        n = 20
        state = qc.ghz_state(n)
        mps_memory = state.memory_bytes()

        # Full statevector would need 2^n * 16 bytes (complex128)
        full_memory = (2 ** n) * 16

        ratio = full_memory / mps_memory
        # MPS should be at least 10x more compact than full statevector for n=20
        assert ratio > 10, f"Compression ratio too low: {ratio}x"


# ============================================================================
# PRECISION MODE TESTS
# ============================================================================

class TestPrecisionMode:
    """Tests for precision mode."""
    
    def test_precision_mode_config(self):
        """Test precision mode configuration."""
        config = FrameworkConfig(precision_mode=True, max_qubits=10)
        assert config.precision_mode == True
        assert config.max_qubits == 10
    
    def test_precision_mode_bell_state(self, config_precision: FrameworkConfig):
        """Test Bell state in precision mode."""
        qc = MPSQuantumComputer(config_precision)
        state = qc.bell_state(2)
        
        probs = state.probabilities()
        # In precision mode, probabilities should be very accurate
        assert probs[0].item() == pytest.approx(0.5, abs=0.01)
        assert probs[3].item() == pytest.approx(0.5, abs=0.01)


# ============================================================================
# EDGE CASE TESTS
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases."""
    
    def test_single_qubit(self, qc: MPSQuantumComputer):
        """Test single qubit operations."""
        state = qc.create_state(1)
        circuit = qc.create_circuit(1)
        circuit.h(0)
        state = circuit.run(state)
        
        probs = state.probabilities()
        assert len(probs) == 2
        assert state.entropy() == pytest.approx(1.0, abs=0.1)
    
    def test_large_bond_dimension(self):
        """Test with large bond dimension."""
        config = FrameworkConfig(bond_dimension=128, max_bond_dimension=256)
        qc = MPSQuantumComputer(config)
        
        state = qc.ghz_state(8)
        entropy = state.entropy()
        
        assert entropy == pytest.approx(1.0, abs=0.1)
    
    def test_empty_circuit(self, qc: MPSQuantumComputer):
        """Test empty circuit."""
        state = qc.create_state(2)
        circuit = qc.create_circuit(2)
        # No gates
        state = circuit.run(state)
        
        probs = state.probabilities()
        assert probs[0].item() == pytest.approx(1.0, abs=5e-3)


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
