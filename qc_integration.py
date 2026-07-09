#!/usr/bin/env python3
"""
QC Integration Bridge - OpenQASM, Qiskit, and PennyLane interoperability layer.

Provides bidirectional conversion between the QC quantum circuit representation
and standard quantum computing formats:
  - OpenQASM 2.0 (export/import)
  - Qiskit QuantumCircuit (export/import when qiskit is installed)
  - PennyLane QNode / tape (export/import when pennylane is installed)

Architecture
------------
  IntegrationConfig    -- centralised configuration (no hardcoded values)
  IQCAdapter          -- abstract interface for each target format
  OpenQasmAdapter     -- OpenQASM 2.0 string <-> internal circuit
  QiskitAdapter       -- Qiskit QuantumCircuit <-> internal circuit
  PennyLaneAdapter    -- PennyLane operations <-> internal circuit
  IntegrationBridge   -- facade that delegates to the correct adapter

Usage
-----
  from qc_integration import IntegrationBridge, IntegrationConfig

  config = IntegrationConfig()
  bridge = IntegrationBridge(config)

  # Export to OpenQASM
  qasm_str = bridge.export_qasm(circuit)

  # Import from OpenQASM
  circuit = bridge.import_qasm(qasm_str)

  # Convert to Qiskit (requires qiskit)
  qc_qiskit = bridge.to_qiskit(circuit)

  # Convert from Qiskit
  circuit = bridge.from_qiskit(qc_qiskit)

  # Convert to PennyLane (requires pennylane)
  tape = bridge.to_pennylane(circuit)

Gate mapping
------------
  H       -> OpenQASM h, Qiskit h, PennyLane Hadamard
  X       -> OpenQASM x, Qiskit x, PennyLane PauliX
  Y       -> OpenQASM y, Qiskit y, PennyLane PauliY
  Z       -> OpenQASM z, Qiskit z, PennyLane PauliZ
  S       -> OpenQASM s, Qiskit s, PennyLane S
  T       -> OpenQASM t, Qiskit t, PennyLane T
  Rx      -> OpenQASM rx, Qiskit rx, PennyLane RX
  Ry      -> OpenQASM ry, Qiskit ry, PennyLane RY
  Rz      -> OpenQASM rz, Qiskit rz, PennyLane RZ
  CNOT    -> OpenQASM cx, Qiskit cx, PennyLane CNOT
  CZ      -> OpenQASM cz, Qiskit cz, PennyLane CZ
  SWAP    -> OpenQASM swap, Qiskit swap, PennyLane SWAP
  CCX     -> OpenQASM ccx, Qiskit ccx, PennyLane Toffoli
"""

from __future__ import annotations

import logging
import math
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple


_LOG = logging.getLogger("QCIntegration")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class IntegrationConfig:
    """Centralised configuration for the integration bridge.

    All tunable parameters live here -- no hardcoded values in logic.
    """

    qasm_version: str = "2.0"
    qasm_include_stdgates: bool = True
    max_qubits_qasm: int = 256
    max_qubits_qiskit: int = 256
    max_qubits_pennylane: int = 256
    indent_qasm: str = "  "
    parameter_precision: int = 15
    gate_name_map: Dict[str, str] = field(
        default_factory=lambda: {
            "H": "h",
            "X": "x",
            "Y": "y",
            "Z": "z",
            "S": "s",
            "T": "t",
            "Rx": "rx",
            "Ry": "ry",
            "Rz": "rz",
            "CNOT": "cx",
            "CZ": "cz",
            "SWAP": "swap",
            "CCX": "ccx",
        }
    )
    reverse_gate_name_map: Dict[str, str] = field(init=False)
    supported_gates: Tuple[str, ...] = (
        "H", "X", "Y", "Z", "S", "T",
        "Rx", "Ry", "Rz",
        "CNOT", "CZ", "SWAP", "CCX",
    )
    pennylane_device: str = "default.qubit"

    def __post_init__(self) -> None:
        reverse: Dict[str, str] = {}
        preferred: Dict[str, str] = {"cx": "CNOT"}
        for k, v in self.gate_name_map.items():
            if v in reverse and v not in preferred:
                continue
            reverse[v] = preferred.get(v, k)
        object.__setattr__(self, "reverse_gate_name_map", reverse)


# ---------------------------------------------------------------------------
# Internal circuit representation (lightweight, framework-agnostic)
# ---------------------------------------------------------------------------

@dataclass
class GateInstruction:
    """A single quantum gate instruction."""

    name: str
    targets: Tuple[int, ...]
    params: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.targets = tuple(self.targets)

    @property
    def num_qubits(self) -> int:
        return len(self.targets)


@dataclass
class CircuitIR:
    """Intermediate representation of a quantum circuit.

    This is the lingua franca between the QC framework and external formats.
    """

    n_qubits: int
    instructions: List[GateInstruction] = field(default_factory=list)
    name: str = "qcir"

    def append(self, gate: GateInstruction) -> None:
        for t in gate.targets:
            if t < 0 or t >= self.n_qubits:
                raise IndexError(
                    f"Target qubit {t} out of range for {self.n_qubits} qubits"
                )
        self.instructions.append(gate)

    def __len__(self) -> int:
        return len(self.instructions)

    def __repr__(self) -> str:
        lines = [f"CircuitIR({self.n_qubits} qubits, {len(self)} gates)"]
        for g in self.instructions:
            base = f"  {g.name} q{', q'.join(str(t) for t in g.targets)}"
            if g.params:
                base += f" ({g.params})"
            lines.append(base)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Abstract adapter
# ---------------------------------------------------------------------------

class IQCAdapter(ABC):
    """Interface for a format-specific adapter (Interface Segregation)."""

    @abstractmethod
    def export(self, circuit: CircuitIR, **kwargs: Any) -> Any:
        """Export a CircuitIR to the target format."""

    @abstractmethod
    def import_(self, data: Any, **kwargs: Any) -> CircuitIR:
        """Import from the target format into a CircuitIR."""


# ---------------------------------------------------------------------------
# OpenQASM 2.0 adapter
# ---------------------------------------------------------------------------

_QASM_STDGATES = """
gate u3(theta,phi,lambda) q { U(theta,phi,lambda) q; }
gate u2(phi,lambda) q { U(pi/2,phi,lambda) q; }
gate u1(lambda) q { U(0,0,lambda) q; }
gate cx c,t { CX c,t; }
gate id a { U(0,0,0) a; }
gate u0(gamma) q { U(0,0,0) q; }
gate x a { u3(pi,0,pi) a; }
gate y a { u3(pi,pi/2,pi/2) a; }
gate z a { u1(pi) a; }
gate h a { u2(0,pi) a; }
gate s a { u1(pi/2) a; }
gate sdg a { u1(-pi/2) a; }
gate t a { u1(pi/4) a; }
gate tdg a { u1(-pi/4) a; }
gate rx(theta) a { u3(theta, -pi/2, pi/2) a; }
gate ry(theta) a { u3(theta, 0, 0) a; }
gate rz(phi) a { u3(0, 0, phi) a; }
gate cz a,b { h b; cx a,b; h b; }
gate cy a,b { sdg b; cx a,b; s b; }
gate swap a,b { cx a,b; cx b,a; cx a,b; }
gate ch a,b { h b; sdg b; cx a,b; h b; t b; cx a,b; t b; h b; s b; x b; s a; }
gate ccx a,b,c { h c; cx b,c; tdg c; cx a,c; t c; cx b,c; tdg c; cx a,c; t c; h c; t b; cx a,b; t a; tdg b; cx a,b; }
gate crz(theta) a,b { u1(theta/2) b; cx a,b; u1(-theta/2) b; cx a,b; }
gate cu1(theta) a,b { u1(theta/2) a; cx a,b; u1(-theta/2) b; cx a,b; u1(theta/2) b; }
gate cu3(theta,phi,lambda) a,b { u1((lambda-phi)/2) a; u1(-(lambda+phi)/2) b; cx a,b; u3(-theta,-(phi+lambda)/2,0) b; cx a,b; u3(theta,phi,lambda) b; }
"""


class OpenQasmAdapter(IQCAdapter):
    """Adapter for OpenQASM 2.0 format."""

    _RE_QREG = re.compile(r"qreg\s+(\w+)\[(\d+)\]\s*;")
    _RE_CREG = re.compile(r"creg\s+(\w+)\[(\d+)\]\s*;")
    _RE_GATE = re.compile(
        r"(?P<name>[a-z][a-z0-9_]*)\s*"
        r"(?:\((?P<params>[^)]*)\))?\s*"
        r"(?P<qubits>(?:[a-z]\w*(?:\[(\d+)\])?\s*(?:,\s*)?)+)\s*;"
    )
    _RE_QUBIT_REF = re.compile(r"([a-z]\w*)(?:\[(\d+)\])?")

    def __init__(self, config: IntegrationConfig) -> None:
        self._config = config
        self._gate_map = dict(config.gate_name_map)
        self._reverse_map = dict(config.reverse_gate_name_map)

    def export(self, circuit: CircuitIR, **kwargs: Any) -> str:
        if circuit.n_qubits > self._config.max_qubits_qasm:
            raise ValueError(
                f"Circuit has {circuit.n_qubits} qubits, "
                f"max for QASM is {self._config.max_qubits_qasm}"
            )
        lines = [
            f"OPENQASM {self._config.qasm_version};",
            'include "qelib1.inc";' if self._config.qasm_include_stdgates else "",
        ]
        qreg_name = kwargs.get("qreg_name", "q")
        lines.append(f"qreg {qreg_name}[{circuit.n_qubits}];")
        creg_name = kwargs.get("creg_name", "c")
        lines.append(f"creg {creg_name}[{circuit.n_qubits}];")

        for instr in circuit.instructions:
            qasm_name = self._gate_map.get(instr.name, instr.name.lower())
            qubit_str = ", ".join(
                f"{qreg_name}[{t}]" for t in instr.targets
            )
            if instr.params:
                param_str = ", ".join(
                    f"{v:.{self._config.parameter_precision}g}"
                    if isinstance(v, float)
                    else str(v)
                    for v in instr.params.values()
                )
                line = f"{qasm_name}({param_str}) {qubit_str};"
            else:
                line = f"{qasm_name} {qubit_str};"
            lines.append(line)

        return "\n".join(lines)

    def import_(self, data: str, **kwargs: Any) -> CircuitIR:
        """Parse an OpenQASM 2.0 string into a CircuitIR."""
        qreg_name = "q"
        n_qubits = 0
        instructions: List[GateInstruction] = []

        for line in data.splitlines():
            line = line.strip()
            if not line or line.startswith("//") or line.startswith("OPENQASM") or line.startswith("include"):
                continue

            qreg_match = self._RE_QREG.match(line)
            if qreg_match:
                qreg_name = qreg_match.group(1)
                n_qubits = max(n_qubits, int(qreg_match.group(2)))
                continue

            if line.startswith("creg"):
                continue

            gate_match = self._RE_GATE.match(line)
            if gate_match:
                raw_name = gate_match.group("name")
                raw_params = gate_match.group("params")
                qubits_part = gate_match.group("qubits")

                internal_name = self._reverse_map.get(raw_name, raw_name.upper())

                params: Dict[str, float] = {}
                if raw_params is not None and raw_params.strip():
                    val_strs = [s.strip() for s in raw_params.split(",")]
                    param_names = self._param_names_for_gate(raw_name)
                    for i, vs in enumerate(val_strs):
                        try:
                            val = float(vs.replace("pi", str(math.pi)))
                        except ValueError:
                            val = 0.0
                        if i < len(param_names):
                            params[param_names[i]] = val
                        else:
                            params[f"p{i}"] = val

                targets: List[int] = []
                for ref in self._RE_QUBIT_REF.finditer(qubits_part):
                    idx = int(ref.group(2)) if ref.group(2) is not None else 0
                    targets.append(idx)

                if targets:
                    instructions.append(
                        GateInstruction(
                            name=internal_name,
                            targets=tuple(targets),
                            params=params,
                        )
                    )

        result = CircuitIR(n_qubits=n_qubits, name=kwargs.get("name", "imported"))
        for inst in instructions:
            result.append(inst)
        return result

    @staticmethod
    def _param_names_for_gate(name: str) -> List[str]:
        gate_param_map: Dict[str, List[str]] = {
            "rx": ["theta"],
            "ry": ["theta"],
            "rz": ["phi"],
            "u1": ["lambda"],
            "u2": ["phi", "lambda"],
            "u3": ["theta", "phi", "lambda"],
            "crz": ["theta"],
            "cu1": ["theta"],
            "cu3": ["theta", "phi", "lambda"],
        }
        return gate_param_map.get(name, [])


# ---------------------------------------------------------------------------
# Qiskit adapter (optional dependency)
# ---------------------------------------------------------------------------

class QiskitAdapter(IQCAdapter):
    """Adapter for Qiskit QuantumCircuit format."""

    def __init__(self, config: IntegrationConfig) -> None:
        self._config = config

    def export(self, circuit: CircuitIR, **kwargs: Any) -> Any:
        qiskit = self._ensure_qiskit()
        qc = qiskit.QuantumCircuit(circuit.n_qubits, name=circuit.name)
        gate_method_map = self._build_qiskit_method_map(qc)
        for instr in circuit.instructions:
            method = gate_method_map.get(instr.name)
            if method is None:
                raise ValueError(f"Unsupported gate for Qiskit: {instr.name}")
            if instr.params:
                vals = list(instr.params.values())
                method(*instr.targets, *vals)
            else:
                method(*instr.targets)
        return qc

    def import_(self, data: Any, **kwargs: Any) -> CircuitIR:
        qiskit = self._ensure_qiskit()
        if isinstance(data, qiskit.QuantumCircuit):
            qc = data
        else:
            raise TypeError(f"Expected QuantumCircuit, got {type(data)}")

        result = CircuitIR(n_qubits=qc.num_qubits, name=qc.name or "imported_qiskit")
        reverse_gate_map = self._build_reverse_gate_map()

        for instruction, qargs, _cargs in qc.data:
            gate_name = instruction.name
            internal_name = reverse_gate_map.get(gate_name, gate_name.upper())
            targets = tuple(qc.find_bit(q).index for q in qargs)
            params: Dict[str, float] = {}
            if instruction.params:
                for i, p in enumerate(instruction.params):
                    params[f"p{i}"] = float(p)
            result.append(
                GateInstruction(name=internal_name, targets=targets, params=params)
            )
        return result

    @staticmethod
    def _ensure_qiskit() -> Any:
        try:
            import qiskit
            return qiskit
        except ImportError:
            raise ImportError(
                "Qiskit is required for QiskitAdapter. "
                "Install it with: pip install qiskit"
            )

    @staticmethod
    def _build_qiskit_method_map(qc: Any) -> Dict[str, Any]:
        return {
            "H": qc.h,
            "X": qc.x,
            "Y": qc.y,
            "Z": qc.z,
            "S": qc.s,
            "T": qc.t,
            "Rx": lambda q, theta: qc.rx(theta, q),
            "Ry": lambda q, theta: qc.ry(theta, q),
            "Rz": lambda q, phi: qc.rz(phi, q),
            "CNOT": qc.cx,
            "CX": qc.cx,
            "CZ": qc.cz,
            "SWAP": qc.swap,
            "CCX": qc.ccx,
        }

    @staticmethod
    def _build_reverse_gate_map() -> Dict[str, str]:
        return {
            "h": "H",
            "x": "X",
            "y": "Y",
            "z": "Z",
            "s": "S",
            "t": "T",
            "rx": "Rx",
            "ry": "Ry",
            "rz": "Rz",
            "cx": "CNOT",
            "cz": "CZ",
            "swap": "SWAP",
            "ccx": "CCX",
        }


# ---------------------------------------------------------------------------
# PennyLane adapter (optional dependency)
# ---------------------------------------------------------------------------

class PennyLaneAdapter(IQCAdapter):
    """Adapter for PennyLane format."""

    def __init__(self, config: IntegrationConfig) -> None:
        self._config = config

    def export(self, circuit: CircuitIR, **kwargs: Any) -> Any:
        pl = self._ensure_pennylane()
        dev = pl.device(self._config.pennylane_device, wires=circuit.n_qubits)

        gate_ops = self._build_gate_ops(pl)

        def circuit_fn(*input_params: Any) -> Any:
            param_iter = iter(input_params)
            for instr in circuit.instructions:
                op = gate_ops.get(instr.name)
                if op is None:
                    raise ValueError(f"Unsupported gate for PennyLane: {instr.name}")
                if instr.params:
                    vals = [float(v) for v in instr.params.values()]
                    op(*instr.targets, *vals)
                else:
                    op(*instr.targets)
            return [pl.expval(pl.Z(i)) for i in range(circuit.n_qubits)]

        qnode = pl.QNode(circuit_fn, dev)
        return qnode

    def import_(self, data: Any, **kwargs: Any) -> CircuitIR:
        pl = self._ensure_pennylane()
        if isinstance(data, pl.tape.QuantumTape):
            tape = data
        elif hasattr(data, "tape"):
            tape = data.tape
        else:
            raise TypeError(f"Cannot import from {type(data)}")

        result = CircuitIR(n_qubits=tape.num_wires, name="imported_pennylane")
        reverse_map = self._build_reverse_ops(pl)
        for op in tape.operations:
            internal_name = reverse_map.get(type(op).__name__, op.name.upper())
            wires = [w.label for w in op.wires]
            params: Dict[str, float] = {}
            for i, p in enumerate(op.parameters):
                params[f"p{i}"] = float(p)
            result.append(
                GateInstruction(name=internal_name, targets=tuple(wires), params=params)
            )
        return result

    @staticmethod
    def _ensure_pennylane() -> Any:
        try:
            import pennylane
            return pennylane
        except ImportError:
            raise ImportError(
                "PennyLane is required for PennyLaneAdapter. "
                "Install it with: pip install pennylane"
            )

    @staticmethod
    def _build_gate_ops(pl: Any) -> Dict[str, Any]:
        return {
            "H": pl.Hadamard,
            "X": pl.PauliX,
            "Y": pl.PauliY,
            "Z": pl.PauliZ,
            "S": pl.S,
            "T": pl.T,
            "Rx": pl.RX,
            "Ry": pl.RY,
            "Rz": pl.RZ,
            "CNOT": pl.CNOT,
            "CZ": pl.CZ,
            "SWAP": pl.SWAP,
            "CCX": pl.Toffoli,
        }

    @staticmethod
    def _build_reverse_ops(pl: Any) -> Dict[str, str]:
        return {
            "Hadamard": "H",
            "PauliX": "X",
            "PauliY": "Y",
            "PauliZ": "Z",
            "S": "S",
            "T": "T",
            "RX": "Rx",
            "RY": "Ry",
            "RZ": "Rz",
            "CNOT": "CNOT",
            "CZ": "CZ",
            "SWAP": "SWAP",
            "Toffoli": "CCX",
        }


# ---------------------------------------------------------------------------
# Framework adapter (converts between CircuitIR and QC framework circuits)
# ---------------------------------------------------------------------------

class FrameworkAdapter:
    """Converts between CircuitIR and the actual QC framework circuit types.

    Supports both:
      - quantum_framework_core.MPSQuantumComputer / QuantumCircuit (MPS)
      - quantum_computer.QuantumComputer / QuantumCircuit (statevector)
    """

    def __init__(self, config: IntegrationConfig) -> None:
        self._config = config

    def to_circuit_ir(
        self,
        circuit: Any,
        framework_type: str = "auto",
    ) -> CircuitIR:
        """Extract a CircuitIR from a framework circuit object."""
        if framework_type == "auto":
            framework_type = self._detect_framework(circuit)

        if framework_type == "mps":
            return self._from_mps_circuit(circuit)
        elif framework_type == "statevector":
            return self._from_sv_circuit(circuit)
        else:
            raise ValueError(f"Unknown framework type: {framework_type}")

    def from_circuit_ir(
        self,
        cir: CircuitIR,
        framework_type: str = "mps",
    ) -> Any:
        """Build a framework circuit object from a CircuitIR."""
        if framework_type == "mps":
            return self._to_mps_circuit(cir)
        elif framework_type == "statevector":
            return self._to_sv_circuit(cir)
        else:
            raise ValueError(f"Unknown framework type: {framework_type}")

    @staticmethod
    def _detect_framework(circuit: Any) -> str:
        mod = type(circuit).__module__
        if "quantum_framework_core" in mod:
            return "mps"
        if "quantum_computer" in mod:
            return "statevector"
        return "mps"

    @staticmethod
    def _from_mps_circuit(circuit: Any) -> CircuitIR:
        from quantum_framework_core import QuantumCircuit as MPSQuantumCircuit
        if not isinstance(circuit, MPSQuantumCircuit):
            raise TypeError(f"Expected MPS QuantumCircuit, got {type(circuit)}")
        result = CircuitIR(n_qubits=circuit.n_qubits, name="from_mps")
        for inst in circuit._instructions:
            result.append(
                GateInstruction(
                    name=inst.gate_name,
                    targets=tuple(inst.targets),
                    params=dict(inst.params) if inst.params else {},
                )
            )
        return result

    @staticmethod
    def _to_mps_circuit(cir: CircuitIR) -> Any:
        from quantum_framework_core import QuantumCircuit as MPSQuantumCircuit
        qc = MPSQuantumCircuit(cir.n_qubits)
        for inst in cir.instructions:
            qc._append(inst.name, list(inst.targets), dict(inst.params) if inst.params else None)
        return qc

    @staticmethod
    def _from_sv_circuit(circuit: Any) -> CircuitIR:
        from quantum_computer import QuantumCircuit as SVQuantumCircuit
        if not isinstance(circuit, SVQuantumCircuit):
            raise TypeError(f"Expected statevector QuantumCircuit, got {type(circuit)}")
        result = CircuitIR(n_qubits=circuit.n_qubits, name="from_statevector")
        for inst in circuit._instructions:
            result.append(
                GateInstruction(
                    name=inst.gate_name,
                    targets=tuple(inst.targets),
                    params=dict(inst.params) if inst.params else {},
                )
            )
        return result

    @staticmethod
    def _to_sv_circuit(cir: CircuitIR) -> Any:
        from quantum_computer import QuantumCircuit as SVQuantumCircuit
        qc = SVQuantumCircuit(cir.n_qubits)
        for inst in cir.instructions:
            qc._append(inst.name, list(inst.targets), dict(inst.params) if inst.params else None)
        return qc


# ---------------------------------------------------------------------------
# Standard circuits factory
# ---------------------------------------------------------------------------

class StandardCircuitFactory:
    """Build CircuitIR instances for common quantum algorithms."""

    @staticmethod
    def bell_state() -> CircuitIR:
        cir = CircuitIR(n_qubits=2, name="bell_state")
        cir.append(GateInstruction("H", targets=(0,)))
        cir.append(GateInstruction("CNOT", targets=(0, 1)))
        return cir

    @staticmethod
    def ghz_state(n_qubits: int) -> CircuitIR:
        cir = CircuitIR(n_qubits=n_qubits, name="ghz_state")
        cir.append(GateInstruction("H", targets=(0,)))
        for i in range(n_qubits - 1):
            cir.append(GateInstruction("CNOT", targets=(i, i + 1)))
        return cir

    @staticmethod
    def qft(n_qubits: int) -> CircuitIR:
        cir = CircuitIR(n_qubits=n_qubits, name="qft")
        for i in range(n_qubits):
            cir.append(GateInstruction("H", targets=(i,)))
            for j in range(i + 1, n_qubits):
                angle = math.pi / (2 ** (j - i))
                cir.append(GateInstruction("Rz", targets=(j,), params={"theta": angle}))
                cir.append(GateInstruction("CNOT", targets=(i, j)))
                cir.append(GateInstruction("Rz", targets=(j,), params={"theta": -angle}))
                cir.append(GateInstruction("CNOT", targets=(i, j)))
        for i in range(n_qubits // 2):
            cir.append(GateInstruction("SWAP", targets=(i, n_qubits - 1 - i)))
        return cir

    @staticmethod
    def w_state(n_qubits: int) -> CircuitIR:
        cir = CircuitIR(n_qubits=n_qubits, name="w_state")
        cir.append(GateInstruction("H", targets=(0,)))
        for i in range(1, n_qubits):
            cir.append(GateInstruction("CNOT", targets=(0, i)))
        return cir

    @staticmethod
    def grover(n_qubits: int, marked: int = 5, iterations: int = 2) -> CircuitIR:
        cir = CircuitIR(n_qubits=n_qubits, name="grover")
        for i in range(n_qubits):
            cir.append(GateInstruction("H", targets=(i,)))
        for _ in range(iterations):
            for i in range(n_qubits):
                if not (marked >> (n_qubits - 1 - i)) & 1:
                    cir.append(GateInstruction("X", targets=(i,)))
            cir.append(GateInstruction("MCZ", targets=tuple(range(n_qubits))))
            for i in range(n_qubits):
                if not (marked >> (n_qubits - 1 - i)) & 1:
                    cir.append(GateInstruction("X", targets=(i,)))
            for i in range(n_qubits):
                cir.append(GateInstruction("H", targets=(i,)))
                cir.append(GateInstruction("X", targets=(i,)))
            cir.append(GateInstruction("MCZ", targets=tuple(range(n_qubits))))
            for i in range(n_qubits):
                cir.append(GateInstruction("X", targets=(i,)))
                cir.append(GateInstruction("H", targets=(i,)))
        return cir


# ---------------------------------------------------------------------------
# Integration Bridge (facade)
# ---------------------------------------------------------------------------

class IntegrationBridge:
    """Facade that exposes all format conversions through a single API.

    Usage
    -----
        bridge = IntegrationBridge()
        cir = StandardCircuitFactory.bell_state()

        # OpenQASM
        qasm_str = bridge.export_qasm(cir)
        assert isinstance(qasm_str, str)

        cir_restored = bridge.import_qasm(qasm_str)
        assert cir_restored.n_qubits == cir.n_qubits

        # Qiskit (if installed)
        if bridge._qiskit_available:
            qc_qiskit = bridge.to_qiskit(cir)
            cir_back = bridge.from_qiskit(qc_qiskit)

        # PennyLane (if installed)
        if bridge._pennylane_available:
            qnode = bridge.to_pennylane(cir)
            result = qnode()
    """

    def __init__(self, config: Optional[IntegrationConfig] = None) -> None:
        self._config = config or IntegrationConfig()
        self._qasm_adapter = OpenQasmAdapter(self._config)
        self._qiskit_adapter: Optional[QiskitAdapter] = None
        self._pennylane_adapter: Optional[PennyLaneAdapter] = None
        self._framework_adapter = FrameworkAdapter(self._config)
        self._qiskit_available: bool = False
        self._pennylane_available: bool = False
        self._init_optional_adapters()

    def _init_optional_adapters(self) -> None:
        try:
            self._qiskit_adapter = QiskitAdapter(self._config)
            self._qiskit_available = True
        except ImportError:
            self._qiskit_adapter = None
            self._qiskit_available = False

        try:
            self._pennylane_adapter = PennyLaneAdapter(self._config)
            self._pennylane_available = True
        except ImportError:
            self._pennylane_adapter = None
            self._pennylane_available = False

    # -- OpenQASM --

    def export_qasm(self, circuit: CircuitIR, **kwargs: Any) -> str:
        return self._qasm_adapter.export(circuit, **kwargs)

    def import_qasm(self, qasm_str: str, **kwargs: Any) -> CircuitIR:
        return self._qasm_adapter.import_(qasm_str, **kwargs)

    # -- Qiskit --

    def to_qiskit(self, circuit: CircuitIR, **kwargs: Any) -> Any:
        if self._qiskit_adapter is None:
            raise ImportError("Qiskit is not available. Install with: pip install qiskit")
        return self._qiskit_adapter.export(circuit, **kwargs)

    def from_qiskit(self, qiskit_circuit: Any, **kwargs: Any) -> CircuitIR:
        if self._qiskit_adapter is None:
            raise ImportError("Qiskit is not available. Install with: pip install qiskit")
        return self._qiskit_adapter.import_(qiskit_circuit, **kwargs)

    # -- PennyLane --

    def to_pennylane(self, circuit: CircuitIR, **kwargs: Any) -> Any:
        if self._pennylane_adapter is None:
            raise ImportError("PennyLane is not available. Install with: pip install pennylane")
        return self._pennylane_adapter.export(circuit, **kwargs)

    def from_pennylane(self, pennylane_data: Any, **kwargs: Any) -> CircuitIR:
        if self._pennylane_adapter is None:
            raise ImportError("PennyLane is not available. Install with: pip install pennylane")
        return self._pennylane_adapter.import_(pennylane_data, **kwargs)

    # -- Framework conversion --

    def to_circuit_ir(
        self,
        circuit: Any,
        framework_type: str = "auto",
    ) -> CircuitIR:
        return self._framework_adapter.to_circuit_ir(circuit, framework_type)

    def from_circuit_ir(
        self,
        cir: CircuitIR,
        framework_type: str = "mps",
    ) -> Any:
        return self._framework_adapter.from_circuit_ir(cir, framework_type)

    # -- Round-trip helpers --

    def export_qasm_from_framework(
        self,
        circuit: Any,
        framework_type: str = "auto",
        **kwargs: Any,
    ) -> str:
        cir = self.to_circuit_ir(circuit, framework_type)
        return self.export_qasm(cir, **kwargs)

    def import_qasm_to_framework(
        self,
        qasm_str: str,
        framework_type: str = "mps",
        **kwargs: Any,
    ) -> Any:
        cir = self.import_qasm(qasm_str, **kwargs)
        return self.from_circuit_ir(cir, framework_type)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Command-line interface for the integration bridge."""
    import argparse

    parser = argparse.ArgumentParser(
        description="QC Integration Bridge - export/import circuits",
    )
    parser.add_argument("action", choices=["export-qasm", "import-qasm", "list-gates"])
    parser.add_argument("--circuit", "-c", default="bell", help="Standard circuit name")
    parser.add_argument("--qubits", "-q", type=int, default=3, help="Number of qubits")
    parser.add_argument("--output", "-o", default="", help="Output file path")

    args = parser.parse_args()

    config = IntegrationConfig()
    bridge = IntegrationBridge(config)
    factory = StandardCircuitFactory()

    circuit_map = {
        "bell": factory.bell_state,
        "ghz": lambda: factory.ghz_state(args.qubits),
        "qft": lambda: factory.qft(args.qubits),
        "w_state": lambda: factory.w_state(args.qubits),
        "grover": lambda: factory.grover(args.qubits),
    }

    if args.action == "list-gates":
        print("Supported gates:", ", ".join(sorted(config.supported_gates)))
        print("Gate mapping:", config.gate_name_map)
        print("Available circuits:", ", ".join(sorted(circuit_map)))
        return

    if args.action == "export-qasm":
        builder = circuit_map.get(args.circuit)
        if builder is None:
            print(f"Unknown circuit: {args.circuit}")
            print("Available circuits:", ", ".join(sorted(circuit_map)))
            return
        cir = builder()
        qasm_str = bridge.export_qasm(cir)
        if args.output:
            with open(args.output, "w") as f:
                f.write(qasm_str)
            print(f"QASM written to {args.output}")
        else:
            print(qasm_str)
        return

    if args.action == "import-qasm":
        if args.output and not args.output.startswith("-"):
            with open(args.output) as f:
                qasm_str = f.read()
        else:
            qasm_str = sys.stdin.read()
        cir = bridge.import_qasm(qasm_str)
        print(f"Imported circuit: {cir.n_qubits} qubits, {len(cir)} gates")
        for inst in cir.instructions:
            print(f"  {inst}")
        return


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
