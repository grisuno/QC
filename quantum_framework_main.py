#!/usr/bin/env python3
"""
Quantum Simulation Framework - Main Entry Point
================================================
Main entry point for the quantum simulation framework with
command-line interface and interactive menu support.

Usage:
    python quantum_framework_main.py [OPTIONS]

Options:
    --config PATH       Path to TOML configuration file
    --interactive       Launch interactive menu mode
    --all               Run ALL experiments automatically (for debugging)
    --experiment NAME   Run specific experiment by name
    --benchmark         Run scaling benchmark
    --max-qubits N      Maximum qubits for benchmark
    --molecule NAME     Specify molecule for experiments
    --orbital NAME      Specify orbital for visualization
    --qubits N          Number of qubits for experiments
    --output DIR        Output directory
    --verbose           Enable verbose logging
    --help              Show help message

Author: Gris Iscomeback
License: AGPL v3
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Optional

import torch

from quantum_framework_core import (
    FrameworkConfig,
    ConfigLoader,
    MPSQuantumComputer,
    MPSState,
    run_scaling_benchmark,
    _LOG,
)

from quantum_framework_menu import run_interactive_menu, run_all_experiments


def setup_logging(verbose: bool = False) -> None:
    """Configure logging level based on verbosity."""
    level = logging.DEBUG if verbose else logging.INFO
    
    for name in ["QuantumFramework", "MPSQuantumComputer"]:
        logger = logging.getLogger(name)
        logger.setLevel(level)


def run_benchmark(args: argparse.Namespace, config: FrameworkConfig) -> None:
    """Run scaling benchmark."""
    max_qubits = args.max_qubits or config.max_qubits
    
    _LOG.info("="*60)
    _LOG.info("QUANTUM FRAMEWORK - SCALING BENCHMARK")
    _LOG.info("="*60)
    _LOG.info("Configuration:")
    _LOG.info("  Max qubits: %d", max_qubits)
    _LOG.info("  Bond dimension: %d", config.bond_dimension)
    _LOG.info("  Device: %s", config.device)
    _LOG.info("")
    
    results = run_scaling_benchmark(config, max_qubits)
    
    _LOG.info("")
    _LOG.info("="*60)
    _LOG.info("BENCHMARK RESULTS")
    _LOG.info("="*60)
    
    print(f"\n{'Qubits':>8} {'Memory (KB)':>14} {'Time (ms)':>12} {'Compression':>15}")
    print("-" * 55)
    
    for i, n in enumerate(results["qubits"]):
        mem_kb = results["memory_kb"][i]
        time_ms = results["time_seconds"][i] * 1000
        ratio = results["compression_ratio"][i]
        print(f"{n:>8} {mem_kb:>14.2f} {time_ms:>12.2f} {ratio:>15.2e}")
    
    if results["qubits"]:
        last_n = results["qubits"][-1]
        last_ratio = results["compression_ratio"][-1]
        print(f"\nMaximum achieved: {last_n} qubits")
        print(f"Final compression ratio: {last_ratio:.2e}x")
        
        theoretical_direct = 2 ** last_n * 2 * config.grid_size ** 2 * 8
        print(f"Theoretical direct memory: {theoretical_direct / (1024**3):.2f} GB")
        print(f"Actual MPS memory: {results['memory_kb'][-1]:.2f} KB")


def run_experiment(args: argparse.Namespace, config: FrameworkConfig, 
                   config_loader: ConfigLoader) -> None:
    """Run a specific experiment by name."""
    experiment_name = args.experiment
    
    exp_data = config_loader.get_experiment(experiment_name)
    if exp_data is None:
        _LOG.error("Experiment not found: %s", experiment_name)
        print(f"\nAvailable experiments:")
        for name, data in config_loader.experiments.items():
            print(f"  - {data['name']} ({data['category']})")
        return
    
    _LOG.info("="*60)
    _LOG.info("RUNNING EXPERIMENT: %s", exp_data['name'])
    _LOG.info("="*60)
    _LOG.info("Description: %s", exp_data['description'])
    _LOG.info("Category: %s", exp_data['category'])
    
    n_qubits = args.qubits or exp_data['default_qubits']
    qc = MPSQuantumComputer(config)
    
    if experiment_name == "bell_state":
        state = qc.bell_state(n_qubits)
        probs = state.probabilities()
        entropy = state.entropy()
        
        print(f"\nBell State Results:")
        print(f"  Qubits: {n_qubits}")
        print(f"  Entropy: {entropy:.4f} bits (theoretical: 1.0000)")
        print(f"  Probabilities:")
        for i, p in enumerate(probs[:4]):
            print(f"    |{format(i, '02b')}>: {p:.4f}")
        print(f"  Memory: {state.memory_bytes()} bytes")
        
    elif experiment_name == "ghz_state":
        state = qc.ghz_state(n_qubits)
        probs = state.probabilities()
        entropy = state.entropy()
        
        print(f"\nGHZ State Results:")
        print(f"  Qubits: {n_qubits}")
        print(f"  Entropy: {entropy:.4f} bits (theoretical: 1.0000)")
        print(f"  Most probable: |{state.most_probable_bitstring()}>")
        print(f"  Memory: {state.memory_bytes() / 1024:.2f} KB")
        print(f"  Phase: {qc.detect_phase(state).name}")
        
    elif experiment_name == "w_state":
        state = qc.w_state(n_qubits)
        probs = state.probabilities()
        entropy = state.entropy()
        theoretical_entropy = torch.log2(torch.tensor(n_qubits, dtype=torch.float64))
        
        print(f"\nW State Results:")
        print(f"  Qubits: {n_qubits}")
        print(f"  Entropy: {entropy:.4f} bits (theoretical: {theoretical_entropy:.4f})")
        print(f"  Memory: {state.memory_bytes() / 1024:.2f} KB")
        
    elif experiment_name == "scaling_benchmark":
        max_q = args.max_qubits or config.max_qubits
        results = run_scaling_benchmark(config, min(max_q, config.max_qubits))
        
        print(f"\nScaling Benchmark Results:")
        print(f"  Max qubits achieved: {results['qubits'][-1]}")
        print(f"  Final compression: {results['compression_ratio'][-1]:.2e}x")
        
    else:
        print(f"\nExperiment '{experiment_name}' implementation pending")
        print(f"  Category: {exp_data['category']}")
        print(f"  Default qubits: {exp_data['default_qubits']}")


def run_molecular_simulation(args: argparse.Namespace, config: FrameworkConfig,
                             config_loader: ConfigLoader) -> None:
    """Run molecular simulation."""
    molecule_name = args.molecule
    
    mol = config_loader.get_molecule(molecule_name)
    if mol is None:
        _LOG.error("Molecule not found: %s", molecule_name)
        print(f"\nAvailable molecules:")
        for name, data in config_loader.molecules.items():
            if data.n_qubits <= config.max_qubits:
                print(f"  - {data.name} ({data.formula}, {data.n_qubits} qubits)")
        return
    
    _LOG.info("="*60)
    _LOG.info("MOLECULAR SIMULATION: %s", mol.name)
    _LOG.info("="*60)
    
    print(f"\nMolecule: {mol.name} ({mol.formula})")
    print(f"  Description: {mol.description}")
    print(f"  Electrons: {mol.n_electrons}")
    print(f"  Orbitals: {mol.n_orbitals}")
    print(f"  Qubits: {mol.n_qubits}")
    print(f"  Basis: {mol.basis}")
    print(f"  Bond length: {mol.bond_length_angstrom:.4f} A")
    print(f"\nReference energies:")
    print(f"  HF Energy: {mol.hf_energy_hartree:.8f} Ha")
    print(f"  FCI Energy: {mol.fci_energy_hartree:.8f} Ha")
    print(f"  Correlation energy: {mol.hf_energy_hartree - mol.fci_energy_hartree:.6f} Ha")
    
    if mol.n_qubits <= config.max_qubits:
        print(f"\n  This molecule can be simulated with {mol.n_qubits} qubits")
        print(f"  MPS memory requirement: ~{mol.n_qubits * config.bond_dimension**2 * 2 * 8 / 1024:.1f} KB")
    else:
        print(f"\n  Warning: {mol.n_qubits} qubits exceeds max {config.max_qubits}")


def run_orbital_visualization(args: argparse.Namespace, config: FrameworkConfig,
                              config_loader: ConfigLoader) -> None:
    """Run orbital visualization."""
    orbital_name = args.orbital
    
    orb = config_loader.get_orbital(orbital_name)
    if orb is None:
        _LOG.error("Orbital not found: %s", orbital_name)
        print(f"\nAvailable orbitals:")
        for name, data in config_loader.orbitals.items():
            print(f"  - {data.name} (n={data.n}, l={data.l}, m={data.m})")
        return
    
    _LOG.info("="*60)
    _LOG.info("ORBITAL VISUALIZATION: %s", orb.name)
    _LOG.info("="*60)
    
    print(f"\nOrbital: {orb.name}")
    print(f"  Description: {orb.description}")
    print(f"  Quantum numbers: n={orb.n}, l={orb.l}, m={orb.m}")
    print(f"\nOrbital type: ", end="")
    
    orbital_types = ["s", "p", "d", "f", "g"]
    if orb.l < len(orbital_types):
        print(f"{orb.n}{orbital_types[orb.l]}")
    else:
        print(f"{orb.n}l={orb.l}")
    
    print(f"\nVisualization requires matplotlib and scipy")
    print(f"  Use interactive menu for full visualization")


def print_info(config: FrameworkConfig, config_loader: ConfigLoader) -> None:
    """Print framework information."""
    print("""
================================================================================
                    QUANTUM SIMULATION FRAMEWORK
================================================================================

A comprehensive toolkit for quantum computing simulations using Matrix Product
State (MPS) representation. Achieves sub-exponential memory scaling O(n*chi^2)
instead of O(2^n) for full statevector representation.

KEY FEATURES:
  - Scalable quantum simulation up to 33+ qubits
  - MPS-based memory-efficient representation
  - Multiple physics backends (Hamiltonian, Schrodinger, Dirac)
  - Quantum gate library with MPS-optimized operations
  - Molecular simulations with VQE support
  - Hydrogen orbital visualization
  - Relativistic quantum mechanics (Dirac equation)
  - QED effects calculations (Lamb shift, anomalous magnetic moment)
  - Interactive menu system

CONFIGURATION:
""")
    print(f"  Grid size: {config.grid_size}")
    print(f"  Device: {config.device}")
    print(f"  Max qubits: {config.max_qubits}")
    print(f"  Bond dimension: {config.bond_dimension}")
    print(f"  Speed of light: {config.c_light}")
    print(f"  Fine structure constant: {config.alpha_fs}")
    
    print(f"""
AVAILABLE RESOURCES:
  Atoms: {len(config_loader.atoms)}
  Molecules: {len(config_loader.molecules)}
  Orbitals: {len(config_loader.orbitals)}
  Experiments: {len(config_loader.experiments)}

USAGE:
  python quantum_framework_main.py --interactive          # Launch interactive menu
  python quantum_framework_main.py --all                  # Run ALL experiments (debug)
  python quantum_framework_main.py --benchmark            # Run scaling benchmark
  python quantum_framework_main.py --experiment bell_state  # Run specific experiment
  python quantum_framework_main.py --molecule H2          # Show molecule info
  python quantum_framework_main.py --orbital 1s           # Show orbital info

================================================================================
""")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Quantum Simulation Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="",
        help="Path to TOML configuration file"
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Launch interactive menu mode"
    )
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="Run ALL experiments automatically (for debugging)"
    )
    parser.add_argument(
        "--experiment", "-e",
        type=str,
        default="",
        help="Run specific experiment by name"
    )
    parser.add_argument(
        "--benchmark", "-b",
        action="store_true",
        help="Run scaling benchmark"
    )
    parser.add_argument(
        "--max-qubits",
        type=int,
        default=None,
        help="Maximum qubits for benchmark"
    )
    parser.add_argument(
        "--molecule", "-m",
        type=str,
        default="",
        help="Specify molecule for experiments"
    )
    parser.add_argument(
        "--orbital", "-o",
        type=str,
        default="",
        help="Specify orbital for visualization"
    )
    parser.add_argument(
        "--qubits", "-q",
        type=int,
        default=None,
        help="Number of qubits for experiments"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Output directory"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Show framework information"
    )
    
    args = parser.parse_args()
    
    setup_logging(args.verbose)
    
    config_path = args.config or os.path.join(
        os.path.dirname(__file__),
        "quantum_framework_config.toml"
    )
    
    config = FrameworkConfig.from_toml(config_path) if os.path.exists(config_path) else FrameworkConfig()
    
    if args.output:
        config.output_dir = args.output
    
    config_loader = ConfigLoader(config_path if os.path.exists(config_path) else None)
    
    if args.info:
        print_info(config, config_loader)
        return
    
    if args.all:
        run_all_experiments(config, config_loader)
        return
    
    if args.interactive:
        run_interactive_menu(config, config_loader)
        return
    
    if args.benchmark:
        run_benchmark(args, config)
        return
    
    if args.experiment:
        run_experiment(args, config, config_loader)
        return
    
    if args.molecule:
        run_molecular_simulation(args, config, config_loader)
        return
    
    if args.orbital:
        run_orbital_visualization(args, config, config_loader)
        return
    
    print_info(config, config_loader)


if __name__ == "__main__":
    main()
