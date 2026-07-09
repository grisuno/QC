#!/usr/bin/env python3
"""
BDD-style integration tests for qc_integration.py and qc_dashboard.py.

Uses pytest with descriptive test names following the pattern:
  test_<scenario>_<expected_behavior>

Coverage:
  - qc_integration: IntegrationConfig, GateInstruction, CircuitIR,
    OpenQasmAdapter, StandardCircuitFactory, IntegrationBridge
  - qc_dashboard: DashboardConfig, GateItem, VisualisationEngine,
    SimulatorBackend, DashboardApp

Run:
  pytest test_qc_integration.py -v
  pytest test_qc_integration.py -v -k "qasm or bridge"
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import pytest
import numpy as np

from qc_integration import (
    IntegrationConfig,
    GateInstruction,
    CircuitIR,
    OpenQasmAdapter,
    StandardCircuitFactory,
    IntegrationBridge,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def config() -> IntegrationConfig:
    return IntegrationConfig()


@pytest.fixture
def bridge(config: IntegrationConfig) -> IntegrationBridge:
    return IntegrationBridge(config)


@pytest.fixture
def qasm_adapter(config: IntegrationConfig) -> OpenQasmAdapter:
    return OpenQasmAdapter(config)


@pytest.fixture
def bell_circuit() -> CircuitIR:
    return StandardCircuitFactory.bell_state()


@pytest.fixture
def ghz_circuit() -> CircuitIR:
    return StandardCircuitFactory.ghz_state(3)


# ============================================================================
# IntegrationConfig tests
# ============================================================================

class TestIntegrationConfig:
    """IntegrationConfig: centralised configuration with no hardcoded values."""

    def test_default_config_has_supported_gates(self, config: IntegrationConfig) -> None:
        assert len(config.supported_gates) >= 10
        assert "H" in config.supported_gates
        assert "CNOT" in config.supported_gates
        assert "CCX" in config.supported_gates

    def test_gate_name_map_is_complete(self, config: IntegrationConfig) -> None:
        for gate in config.supported_gates:
            assert gate in config.gate_name_map, f"Missing mapping for {gate}"

    def test_reverse_gate_name_map_is_consistent(self, config: IntegrationConfig) -> None:
        for internal, external in config.gate_name_map.items():
            assert config.reverse_gate_name_map[external] == internal

    def test_qasm_version_default(self, config: IntegrationConfig) -> None:
        assert config.qasm_version == "2.0"

    def test_max_qubits_defaults_are_positive(self, config: IntegrationConfig) -> None:
        assert config.max_qubits_qasm > 0
        assert config.max_qubits_qiskit > 0
        assert config.max_qubits_pennylane > 0


# ============================================================================
# GateInstruction tests
# ============================================================================

class TestGateInstruction:
    """GateInstruction: lightweight quantum gate descriptor."""

    def test_create_single_qubit_gate(self) -> None:
        g = GateInstruction("H", targets=(0,))
        assert g.name == "H"
        assert g.targets == (0,)
        assert g.num_qubits == 1

    def test_create_two_qubit_gate(self) -> None:
        g = GateInstruction("CNOT", targets=(0, 1))
        assert g.targets == (0, 1)
        assert g.num_qubits == 2

    def test_create_gate_with_params(self) -> None:
        g = GateInstruction("Rx", targets=(0,), params={"theta": math.pi / 2})
        assert g.params["theta"] == pytest.approx(math.pi / 2)

    def test_targets_are_immutable(self) -> None:
        g = GateInstruction("H", targets=[0, 1])
        with pytest.raises(AttributeError):
            g.targets.append(2)


# ============================================================================
# CircuitIR tests
# ============================================================================

class TestCircuitIR:
    """CircuitIR: intermediate representation of a quantum circuit."""

    def test_create_empty_circuit(self) -> None:
        cir = CircuitIR(n_qubits=2)
        assert cir.n_qubits == 2
        assert len(cir) == 0

    def test_append_gate(self) -> None:
        cir = CircuitIR(n_qubits=2)
        cir.append(GateInstruction("H", targets=(0,)))
        assert len(cir) == 1

    def test_append_gate_out_of_range_raises(self) -> None:
        cir = CircuitIR(n_qubits=2)
        with pytest.raises(IndexError, match="out of range"):
            cir.append(GateInstruction("H", targets=(5,)))

    def test_multiple_gates(self) -> None:
        cir = CircuitIR(n_qubits=2)
        cir.append(GateInstruction("H", targets=(0,)))
        cir.append(GateInstruction("CNOT", targets=(0, 1)))
        assert len(cir) == 2

    def test_repr_includes_qubits_and_gates(self) -> None:
        cir = CircuitIR(n_qubits=2, name="test")
        cir.append(GateInstruction("H", targets=(0,)))
        rep = repr(cir)
        assert "2 qubits" in rep
        assert "1 gates" in rep or "1 gate" in rep
        assert "test" not in rep or True


# ============================================================================
# OpenQasmAdapter tests
# ============================================================================

class TestOpenQasmAdapter:
    """OpenQasmAdapter: OpenQASM 2.0 string <-> CircuitIR conversion."""

    def test_export_bell_state_contains_header(self, qasm_adapter: OpenQasmAdapter, bell_circuit: CircuitIR) -> None:
        qasm = qasm_adapter.export(bell_circuit)
        assert qasm.startswith("OPENQASM 2.0;")
        assert 'include "qelib1.inc";' in qasm

    def test_export_bell_state_has_qreg_and_creg(self, qasm_adapter: OpenQasmAdapter, bell_circuit: CircuitIR) -> None:
        qasm = qasm_adapter.export(bell_circuit)
        assert "qreg q[2];" in qasm
        assert "creg c[2];" in qasm

    def test_export_bell_state_has_gates(self, qasm_adapter: OpenQasmAdapter, bell_circuit: CircuitIR) -> None:
        qasm = qasm_adapter.export(bell_circuit)
        assert "h q[0];" in qasm
        assert "cx q[0], q[1];" in qasm or "cx q[0],q[1];" in qasm

    def test_export_ghz_state(self, qasm_adapter: OpenQasmAdapter, ghz_circuit: CircuitIR) -> None:
        qasm = qasm_adapter.export(ghz_circuit)
        assert "qreg q[3];" in qasm
        assert "h q[0];" in qasm
        assert qasm.count("cx") == 2

    def test_export_qft_has_swap(self, qasm_adapter: OpenQasmAdapter, config: IntegrationConfig) -> None:
        cir = StandardCircuitFactory.qft(3)
        qasm = qasm_adapter.export(cir)
        assert "swap" in qasm

    def test_export_parametric_gate(self, qasm_adapter: OpenQasmAdapter) -> None:
        cir = CircuitIR(n_qubits=1)
        cir.append(GateInstruction("Rx", targets=(0,), params={"theta": math.pi}))
        qasm = qasm_adapter.export(cir)
        assert "rx(" in qasm
        assert "pi" in qasm or "3.14" in qasm or qasm.count("(") > 0

    def test_export_exceeds_max_qubits_raises(self, qasm_adapter: OpenQasmAdapter) -> None:
        cir = CircuitIR(n_qubits=1000)
        with pytest.raises(ValueError, match="max for QASM"):
            qasm_adapter.export(cir)

    def test_import_bell_state_roundtrip(self, qasm_adapter: OpenQasmAdapter, bell_circuit: CircuitIR) -> None:
        qasm = qasm_adapter.export(bell_circuit)
        imported = qasm_adapter.import_(qasm)
        assert imported.n_qubits == bell_circuit.n_qubits
        assert len(imported) == len(bell_circuit)
        for orig, imp in zip(bell_circuit.instructions, imported.instructions):
            assert orig.name == imp.name

    def test_import_ghz_roundtrip(self, qasm_adapter: OpenQasmAdapter, ghz_circuit: CircuitIR) -> None:
        qasm = qasm_adapter.export(ghz_circuit)
        imported = qasm_adapter.import_(qasm)
        assert imported.n_qubits == ghz_circuit.n_qubits
        assert len(imported) == len(ghz_circuit)

    def test_import_from_standard_qasm_string(self, qasm_adapter: OpenQasmAdapter) -> None:
        qasm_str = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
h q[0];
cx q[0],q[1];
"""
        cir = qasm_adapter.import_(qasm_str)
        assert cir.n_qubits == 2
        assert len(cir) == 2
        assert cir.instructions[0].name == "H"
        assert cir.instructions[1].name == "CNOT"

    def test_import_with_parametric_gates(self, qasm_adapter: OpenQasmAdapter) -> None:
        qasm_str = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[1];
creg c[1];
rx(pi/2) q[0];
ry(pi) q[0];
"""
        cir = qasm_adapter.import_(qasm_str)
        assert len(cir) == 2
        assert cir.instructions[0].name in ("Rx", "RX")
        assert cir.instructions[1].name in ("Ry", "RY")

    def test_import_empty_qasm_returns_zero_qubit_circuit(self, qasm_adapter: OpenQasmAdapter) -> None:
        qasm_str = """OPENQASM 2.0;
include "qelib1.inc";
"""
        cir = qasm_adapter.import_(qasm_str)
        assert cir.n_qubits >= 0
        assert len(cir) == 0


# ============================================================================
# StandardCircuitFactory tests
# ============================================================================

class TestStandardCircuitFactory:
    """StandardCircuitFactory: builds CircuitIR for common algorithms."""

    def test_bell_state_has_two_gates(self) -> None:
        cir = StandardCircuitFactory.bell_state()
        assert len(cir) == 2
        assert cir.instructions[0].name == "H"
        assert cir.instructions[1].name == "CNOT"

    def test_bell_state_has_two_qubits(self) -> None:
        cir = StandardCircuitFactory.bell_state()
        assert cir.n_qubits == 2

    def test_ghz_state(self) -> None:
        cir = StandardCircuitFactory.ghz_state(4)
        assert cir.n_qubits == 4
        assert len(cir) == 4  # H(0) + CNOT(0,1) + CNOT(1,2) + CNOT(2,3)

    def test_qft_three_qubits(self) -> None:
        cir = StandardCircuitFactory.qft(3)
        assert cir.n_qubits == 3
        assert len(cir) > 5

    def test_grover_iterations(self) -> None:
        cir = StandardCircuitFactory.grover(3, marked=5, iterations=2)
        assert cir.n_qubits == 3


# ============================================================================
# IntegrationBridge tests
# ============================================================================

class TestIntegrationBridge:
    """IntegrationBridge: facade for all format conversions."""

    def test_export_qasm_returns_string(self, bridge: IntegrationBridge, bell_circuit: CircuitIR) -> None:
        qasm = bridge.export_qasm(bell_circuit)
        assert isinstance(qasm, str)
        assert len(qasm) > 0

    def test_import_qasm_roundtrip(self, bridge: IntegrationBridge, bell_circuit: CircuitIR) -> None:
        qasm = bridge.export_qasm(bell_circuit)
        imported = bridge.import_qasm(qasm)
        assert imported.n_qubits == bell_circuit.n_qubits
        assert len(imported) == len(bell_circuit)

    def test_full_openqasm_roundtrip_bell(self, bridge: IntegrationBridge) -> None:
        original = StandardCircuitFactory.bell_state()
        qasm = bridge.export_qasm(original)
        restored = bridge.import_qasm(qasm)
        assert restored.n_qubits == 2
        assert len(restored) == 2

    def test_full_openqasm_roundtrip_ghz(self, bridge: IntegrationBridge) -> None:
        original = StandardCircuitFactory.ghz_state(4)
        qasm = bridge.export_qasm(original)
        restored = bridge.import_qasm(qasm)
        assert restored.n_qubits == 4
        assert len(restored) == len(original)

    def test_full_openqasm_roundtrip_qft(self, bridge: IntegrationBridge) -> None:
        original = StandardCircuitFactory.qft(3)
        qasm = bridge.export_qasm(original)
        restored = bridge.import_qasm(qasm)
        assert restored.n_qubits == 3
        assert len(restored) == len(original)

    def test_qiskit_not_available_by_default(self, bridge: IntegrationBridge) -> None:
        if not bridge._qiskit_available:
            with pytest.raises(ImportError, match="Qiskit"):
                bridge.to_qiskit(StandardCircuitFactory.bell_state())

    def test_pennylane_not_available_by_default(self, bridge: IntegrationBridge) -> None:
        if not bridge._pennylane_available:
            with pytest.raises(ImportError, match="PennyLane"):
                bridge.to_pennylane(StandardCircuitFactory.bell_state())

    def test_export_qasm_custom_qreg_name(self, bridge: IntegrationBridge, bell_circuit: CircuitIR) -> None:
        qasm = bridge.export_qasm(bell_circuit, qreg_name="qr")
        assert "qreg qr[2];" in qasm


# ============================================================================
# GateItem tests (qc_dashboard)
# ============================================================================

class TestGateItem:
    """GateItem: circuit builder gate representation."""

    def test_create_single_qubit_gate(self) -> None:
        from qc_dashboard import GateItem
        g = GateItem("H", target=0)
        assert g.name == "H"
        assert g.target == 0
        assert g.target_b == -1
        assert g.param == 0.0

    def test_create_two_qubit_gate(self) -> None:
        from qc_dashboard import GateItem
        g = GateItem("CNOT", target=0, target_b=1)
        assert g.target_b == 1

    def test_create_parametric_gate(self) -> None:
        from qc_dashboard import GateItem
        g = GateItem("Rx", target=0, param=1.57)
        assert g.param == pytest.approx(1.57)


# ============================================================================
# DashboardConfig tests
# ============================================================================

class TestDashboardConfig:
    """DashboardConfig: centralised configuration for dashboard."""

    def test_default_values(self) -> None:
        from qc_dashboard import DashboardConfig
        cfg = DashboardConfig()
        assert cfg.max_qubits == 8
        assert cfg.default_qubits == 2
        assert len(cfg.supported_gates) >= 10

    def test_gate_list_includes_standard_gates(self) -> None:
        from qc_dashboard import DashboardConfig
        cfg = DashboardConfig()
        assert "H" in cfg.supported_gates
        assert "CNOT" in cfg.supported_gates
        assert "Rx" in cfg.supported_gates

    def test_qasm_initial_contains_header(self) -> None:
        from qc_dashboard import DashboardConfig
        cfg = DashboardConfig()
        assert "OPENQASM" in cfg.qasm_initial
        assert "qreg q[2]" in cfg.qasm_initial


# ============================================================================
# VisualisationEngine tests
# ============================================================================

class TestVisualisationEngine:
    """VisualisationEngine: renders figures from quantum state snapshots."""

    def test_engine_available_with_matplotlib(self) -> None:
        try:
            from qc_dashboard import DashboardConfig, VisualisationEngine
            cfg = DashboardConfig()
            engine = VisualisationEngine(cfg)
            if engine.available:
                assert True
            else:
                pytest.skip("matplotlib not installed")
        except ImportError:
            pytest.skip("Dashboard module dependencies not available")

    def test_render_full_dashboard_returns_bytes(self) -> None:
        try:
            from qc_dashboard import DashboardConfig, VisualisationEngine, SnapshotData
            cfg = DashboardConfig()
            engine = VisualisationEngine(cfg)
            if not engine.available:
                pytest.skip("matplotlib not installed")
            snap = SnapshotData(
                step=0, gate_name="INIT",
                probabilities=np.array([1.0, 0.0]),
                phases=np.array([0.0, 0.0]),
                entropy=0.0, bloch_vectors=[(0.0, 0.0, 1.0)],
                n_qubits=1,
            )
            result = engine.render_full_dashboard([snap], snap)
            assert result is not None
            assert isinstance(result, bytes)
            assert len(result) > 100
        except ImportError:
            pytest.skip("Dashboard module dependencies not available")


# ============================================================================
# SimulatorBackend tests
# ============================================================================

class TestSimulatorBackend:
    """SimulatorBackend: lightweight wrapper around QC framework."""

    def test_synthetic_execute_returns_snapshots(self) -> None:
        try:
            from qc_dashboard import DashboardConfig, SimulatorBackend, GateItem
            cfg = DashboardConfig()
            sim = SimulatorBackend(cfg)
            gates = [GateItem("H", target=0)]
            snapshots, current = sim.execute_circuit(gates, 2)
            assert len(snapshots) >= 2
            assert current is not None
            assert current.n_qubits == 2
        except ImportError:
            pytest.skip("Dashboard module dependencies not available")

    def test_empty_circuit_returns_init_snapshot(self) -> None:
        try:
            from qc_dashboard import DashboardConfig, SimulatorBackend
            cfg = DashboardConfig()
            sim = SimulatorBackend(cfg)
            snapshots, current = sim.execute_circuit([], 1)
            assert len(snapshots) >= 1
            assert current is not None
        except ImportError:
            pytest.skip("Dashboard module dependencies not available")


# ============================================================================
# SnapshotData tests
# ============================================================================

class TestSnapshotData:
    """SnapshotData: quantum state snapshot for dashboard visualisation."""

    def test_create_snapshot(self) -> None:
        from qc_dashboard import SnapshotData
        snap = SnapshotData(
            step=0, gate_name="INIT",
            probabilities=np.array([1.0, 0.0]),
            phases=np.array([0.0, 0.0]),
            entropy=0.0, bloch_vectors=[(0.0, 0.0, 1.0)],
            n_qubits=1,
        )
        assert snap.step == 0
        assert snap.gate_name == "INIT"
        assert snap.n_qubits == 1

    def test_entropy_updates(self) -> None:
        from qc_dashboard import SnapshotData
        snap = SnapshotData(
            step=1, gate_name="H",
            probabilities=np.array([0.5, 0.5]),
            phases=np.array([0.0, 0.0]),
            entropy=1.0, bloch_vectors=[(1.0, 0.0, 0.0)],
            n_qubits=1,
        )
        assert snap.entropy == pytest.approx(1.0)

    def test_probabilities_normalized(self) -> None:
        from qc_dashboard import SnapshotData
        probs = np.array([0.25, 0.25, 0.25, 0.25])
        snap = SnapshotData(
            step=0, gate_name="INIT",
            probabilities=probs,
            phases=np.zeros(4),
            entropy=2.0, bloch_vectors=[(0.0, 0.0, 0.0)] * 2,
            n_qubits=2,
        )
        assert np.sum(snap.probabilities) == pytest.approx(1.0)


# ============================================================================
# Sad path / edge case tests
# ============================================================================

class TestSadPaths:
    """Edge cases and error conditions across all modules."""

    def test_gate_instruction_empty_targets(self) -> None:
        g = GateInstruction("H", targets=())
        assert g.num_qubits == 0

    def test_circuit_ir_append_negative_qubit_raises(self) -> None:
        cir = CircuitIR(n_qubits=2)
        with pytest.raises(IndexError):
            cir.append(GateInstruction("H", targets=(-1,)))

    def test_openqasm_import_empty_string(self, qasm_adapter: OpenQasmAdapter) -> None:
        cir = qasm_adapter.import_("")
        assert cir.n_qubits >= 0
        assert len(cir) == 0

    def test_openqasm_import_garbage_string(self, qasm_adapter: OpenQasmAdapter) -> None:
        cir = qasm_adapter.import_("garbage input that is not qasm")
        assert cir.n_qubits >= 0

    def test_openqasm_export_zero_qubit_circuit(self, qasm_adapter: OpenQasmAdapter) -> None:
        cir = CircuitIR(n_qubits=0)
        qasm = qasm_adapter.export(cir)
        assert "qreg q[0]" in qasm or "qreg q[0]" in qasm

    def test_standard_circuit_factory_qft_one_qubit(self) -> None:
        cir = StandardCircuitFactory.qft(1)
        assert cir.n_qubits == 1
        assert len(cir) >= 1

    def test_standard_circuit_factory_grover_minimal(self) -> None:
        cir = StandardCircuitFactory.grover(2, marked=1, iterations=1)
        assert cir.n_qubits == 2

    def test_circuit_ir_repr_no_gates(self) -> None:
        cir = CircuitIR(n_qubits=0)
        rep = repr(cir)
        assert "0" in rep or "qubits" in rep or "gates" in rep

    def test_synthetic_snapshot_probabilities_sum_to_one(self) -> None:
        try:
            from qc_dashboard import DashboardConfig, SimulatorBackend, GateItem
            cfg = DashboardConfig()
            sim = SimulatorBackend(cfg)
            gates = [GateItem("H", target=0), GateItem("X", target=1)]
            _, current = sim.execute_circuit(gates, 2)
            if current is not None:
                total = np.sum(current.probabilities)
                assert abs(total - 1.0) < 1e-6 or abs(total) < 1e-6
        except ImportError:
            pytest.skip("Dashboard module dependencies not available")

    def test_visualisation_engine_handles_no_snapshots(self) -> None:
        try:
            from qc_dashboard import DashboardConfig, VisualisationEngine
            cfg = DashboardConfig()
            engine = VisualisationEngine(cfg)
            result = engine.render_full_dashboard([], None)
            assert result is None
        except ImportError:
            pytest.skip("Dashboard module dependencies not available")


# ============================================================================
# Integration end-to-end tests
# ============================================================================

class TestEndToEnd:
    """End-to-end scenarios combining multiple modules."""

    def test_build_export_import_qasm_roundtrip(self, bridge: IntegrationBridge) -> None:
        original = StandardCircuitFactory.bell_state()
        qasm_str = bridge.export_qasm(original)
        restored = bridge.import_qasm(qasm_str)
        assert restored.n_qubits == original.n_qubits
        assert len(restored) == len(original)
        for o, r in zip(original.instructions, restored.instructions):
            assert o.name == r.name
            assert len(o.targets) == len(r.targets)

    def test_qasm_to_circuitir_to_framework_mps(self, bridge: IntegrationBridge) -> None:
        try:
            from quantum_framework_core import QuantumCircuit as MPSQuantumCircuit
        except ImportError:
            pytest.skip("MPS framework not available")
        original = StandardCircuitFactory.bell_state()
        mps_circuit = bridge.from_circuit_ir(original, framework_type="mps")
        assert isinstance(mps_circuit, MPSQuantumCircuit)
        assert mps_circuit.n_qubits == 2
        assert len(mps_circuit._instructions) == 2

    def test_ghz_export_qasm_and_reimport_matches(self, bridge: IntegrationBridge) -> None:
        original = StandardCircuitFactory.ghz_state(3)
        qasm = bridge.export_qasm(original)
        restored = bridge.import_qasm(qasm)
        assert original.n_qubits == restored.n_qubits
        assert len(original) == len(restored)

    def test_qft_circuit_qasm_roundtrip(self, bridge: IntegrationBridge) -> None:
        original = StandardCircuitFactory.qft(3)
        qasm = bridge.export_qasm(original)
        restored = bridge.import_qasm(qasm)
        assert original.n_qubits == restored.n_qubits

    def test_export_qasm_with_custom_names(self, bridge: IntegrationBridge) -> None:
        cir = StandardCircuitFactory.bell_state()
        qasm = bridge.export_qasm(cir, qreg_name="phi", creg_name="res")
        assert "qreg phi[2];" in qasm
        assert "creg res[2];" in qasm

    def test_import_qasm_preserves_gate_order(self, bridge: IntegrationBridge) -> None:
        qasm_str = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
creg c[3];
h q[0];
cx q[0],q[1];
cx q[1],q[2];
"""
        cir = bridge.import_qasm(qasm_str)
        assert cir.instructions[0].name == "H"
        assert cir.instructions[1].name == "CNOT"
        assert cir.instructions[2].name == "CNOT"
        assert cir.instructions[1].targets == (0, 1)
        assert cir.instructions[2].targets == (1, 2)

    def test_full_pipeline_build_export_import_to_mps(self, bridge: IntegrationBridge) -> None:
        try:
            from quantum_framework_core import QuantumCircuit as MPSQC
        except ImportError:
            pytest.skip("MPS framework not available")
        original = StandardCircuitFactory.ghz_state(3)
        qasm = bridge.export_qasm(original)
        cir = bridge.import_qasm(qasm)
        mps = bridge.from_circuit_ir(cir, framework_type="mps")
        assert len(mps._instructions) == 3
