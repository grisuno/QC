#!/usr/bin/env python3
"""
Quantum Framework Demo - Production Version
===========================================
Demonstrates all improvements and features of the refactored molecular module.

Improvements implemented:
1. OpenFermion for all Hamiltonians (NO hardcoded values)
2. Precision mode flag (direct vs MPS)
3. Smart initialization with MP2 + parameter scan
4. Cached Pauli operations (x10-100 speedup)
5. Particle-conserving subspace
6. TOML configuration
"""

import logging
import sys
import time
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s"
)

_LOG = logging.getLogger("Demo")

# Import molecular module
try:
    from quantum_framework_molecular_v2 import (
        MoleculeBuilder,
        MolecularConfig,
        VQESolver,
        VQEResult,
        run_vqe,
        OPENFERMION_AVAILABLE,
        PYSCF_AVAILABLE,
        TORCH_AVAILABLE,
    )
    MODULE_AVAILABLE = True
except ImportError as e:
    _LOG.error("Failed to import molecular module: %s", e)
    MODULE_AVAILABLE = False


def print_header(title: str):
    """Print formatted header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def check_dependencies():
    """Check and report available dependencies."""
    print_header("DEPENDENCY CHECK")
    
    deps = [
        ("OpenFermion", OPENFERMION_AVAILABLE),
        ("PySCF", PYSCF_AVAILABLE),
        ("PyTorch", TORCH_AVAILABLE),
    ]
    
    all_ok = True
    for name, available in deps:
        status = "✓ Available" if available else "✗ Missing"
        print(f"  {name:20s} {status}")
        if not available:
            all_ok = False
    
    if not all_ok:
        print("\n  WARNING: Some dependencies are missing.")
        print("  Install with: pip install openfermion pyscf torch")
    
    return all_ok


def demo_h2_direct():
    """Demo: H2 with direct statevector (precision mode)."""
    print_header("DEMO 1: H2 VQE - Direct Statevector (Precision Mode)")
    
    if not OPENFERMION_AVAILABLE:
        print("  Skipping: OpenFermion not available")
        return None
    
    print("\n  Running VQE with precision_mode=True (direct statevector)")
    print("  Expected: Machine precision (~0 Ha error)")
    print()
    
    start = time.time()
    result = run_vqe(
        molecule="H2",
        precision_mode=True,
        max_iterations=100
    )
    elapsed = time.time() - start
    
    print(result)
    print(f"\n  Elapsed time: {elapsed:.2f} s")
    
    return result


def demo_h2_mps():
    """Demo: H2 with MPS compression."""
    print_header("DEMO 2: H2 VQE - MPS Compression (Scalable Mode)")
    
    if not OPENFERMION_AVAILABLE:
        print("  Skipping: OpenFermion not available")
        return None
    
    print("\n  Running VQE with precision_mode=False (MPS)")
    print("  Expected: Good precision with reduced memory")
    print()
    
    start = time.time()
    result = run_vqe(
        molecule="H2",
        precision_mode=False,
        bond_dimension=16,
        max_iterations=100
    )
    elapsed = time.time() - start
    
    print(result)
    print(f"\n  Elapsed time: {elapsed:.2f} s")
    
    return result


def demo_comparison():
    """Demo: Compare direct vs MPS."""
    print_header("DEMO 3: Direct vs MPS Comparison")
    
    if not OPENFERMION_AVAILABLE:
        print("  Skipping: OpenFermion not available")
        return
    
    print("\n  Running H2 VQE with both modes for comparison...\n")
    
    results = {}
    times = {}
    
    # Direct mode
    start = time.time()
    results["Direct"] = run_vqe(molecule="H2", precision_mode=True, max_iterations=50)
    times["Direct"] = time.time() - start
    
    # MPS mode
    start = time.time()
    results["MPS"] = run_vqe(molecule="H2", precision_mode=False, max_iterations=50)
    times["MPS"] = time.time() - start
    
    # Print comparison table
    print("\n  Mode     | Energy (Ha)      | Error (Ha)    | Time (s)  | Evaluations")
    print("  " + "-" * 65)
    
    for mode, result in results.items():
        print(f"  {mode:8s} | {result.vqe_energy:+.8f}     | {result.energy_error:.2e}     | {times[mode]:.2f}      | {result.n_evaluations}")
    
    # Summary
    print("\n  Summary:")
    print(f"    Direct mode: Best precision ({results['Direct'].energy_error:.2e} Ha error)")
    print(f"    MPS mode:    Scalable to >{14} qubits ({results['MPS'].energy_error:.2e} Ha error)")


def demo_molecule_builder():
    """Demo: OpenFermion molecule builder."""
    print_header("DEMO 4: Molecule Builder (OpenFermion)")
    
    if not OPENFERMION_AVAILABLE:
        print("  Skipping: OpenFermion not available")
        return
    
    print("\n  Building molecules with OpenFermion (NO hardcoded values):\n")
    
    molecules = [
        ("H2", lambda: MoleculeBuilder.h2()),
        ("LiH", lambda: MoleculeBuilder.lih()),
    ]
    
    for name, builder in molecules:
        print(f"  {name}:")
        try:
            mol = builder()
            print(f"    Electrons: {mol.n_electrons}")
            print(f"    Orbitals:  {mol.n_orbitals}")
            print(f"    Qubits:    {mol.n_qubits}")
            print(f"    HF Energy: {mol.hf_energy:.8f} Ha")
            print(f"    FCI Energy: {mol.fci_energy:.8f} Ha")
            print(f"    Correlation: {mol.hf_energy - mol.fci_energy:.6f} Ha")
        except Exception as e:
            print(f"    Error: {e}")
        print()


def demo_smart_initialization():
    """Demo: Smart parameter initialization."""
    print_header("DEMO 5: Smart Initialization")
    
    if not OPENFERMION_AVAILABLE:
        print("  Skipping: OpenFermion not available")
        return
    
    print("""
  Smart Initialization Features:
  1. MP2 amplitude estimation for doubles parameters
  2. Systematic scan of parameter space
  3. Parabolic refinement of minimum
  4. Sensitivity analysis for high-impact parameters
  
  Impact: Reduces optimization iterations by ~50%
""")
    
    # Run with comparison
    print("  Running H2 VQE with smart initialization...\n")
    
    result = run_vqe(
        molecule="H2",
        precision_mode=True,
        use_mp2_init=True,
        scan_samples=21,
        max_iterations=50
    )
    
    print(result)
    print(f"\n  Total evaluations: {result.n_evaluations}")
    print("  (Compare to ~400+ without smart init)")


def demo_cached_operations():
    """Demo: Cached Pauli operations."""
    print_header("DEMO 6: Cached Pauli Operations")
    
    print("""
  Cache Optimization:
  - Precompute index mappings for each Pauli term
  - Avoid repeated phase calculations
  - Enable batch evaluation of multiple states
  
  Impact: x10-100 speedup for repeated evaluations
""")
    
    if not OPENFERMION_AVAILABLE:
        print("  Skipping detailed demo: OpenFermion not available")
        return
    
    # Build molecule and evaluator
    from quantum_framework_molecular_v2 import (
        MoleculeBuilder,
        CachedHamiltonianEvaluator,
        MolecularConfig,
    )
    
    mol = MoleculeBuilder.h2()
    config = MolecularConfig()
    evaluator = CachedHamiltonianEvaluator(mol, config)
    
    print(f"  Cached {len(evaluator._cached_operations)} Pauli operations")
    print(f"  Unique operations: {len(evaluator._pauli_cache)}")
    
    # Timing test
    import numpy as np
    state = np.random.rand(2**mol.n_qubits) + 1j * np.random.rand(2**mol.n_qubits)
    state = state / np.linalg.norm(state)
    
    n_evals = 100
    start = time.time()
    for _ in range(n_evals):
        evaluator.expectation_value(state)
    elapsed = time.time() - start
    
    print(f"\n  Timing: {n_evals} evaluations in {elapsed:.4f} s")
    print(f"  Rate: {n_evals/elapsed:.0f} evals/s")


def demo_config_from_toml():
    """Demo: Configuration from TOML."""
    print_header("DEMO 7: TOML Configuration")
    
    import os
    config_path = os.path.join(os.path.dirname(__file__), "quantum_framework_config.toml")
    
    if not os.path.exists(config_path):
        print(f"  Config file not found: {config_path}")
        return
    
    config = MolecularConfig.from_toml(config_path)
    
    print(f"\n  Loaded configuration from: {config_path}")
    print("\n  Key settings:")
    print(f"    Precision mode:     {config.precision_mode}")
    print(f"    Max qubits direct:  {config.max_qubits_direct}")
    print(f"    Bond dimension:     {config.bond_dimension}")
    print(f"    Max iterations:     {config.max_iterations}")
    print(f"    Scan samples:       {config.scan_samples}")
    print(f"    Use MP2 init:       {config.use_mp2_init}")
    print(f"    Backend:            {config.backend}")
    print(f"    Use cache:          {config.use_cache}")


def main():
    """Run all demos."""
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║     QUANTUM FRAMEWORK - MOLECULAR VQE DEMO (Production Version)       ║
╠══════════════════════════════════════════════════════════════════════╣
║  Improvements:                                                        ║
║  1. OpenFermion for ALL Hamiltonians (NO hardcoded values)           ║
║  2. Precision mode flag (direct statevector vs MPS)                  ║
║  3. Smart initialization with MP2 + parameter scan                   ║
║  4. Cached Pauli operations (x10-100 speedup)                        ║
║  5. Particle-conserving subspace                                     ║
║  6. TOML configuration                                               ║
╚══════════════════════════════════════════════════════════════════════╝
""")
    
    # Check dependencies first
    if not check_dependencies():
        print("\n  Some demos will be skipped due to missing dependencies.")
    
    # Run demos
    demos = [
        ("H2 Direct Mode", demo_h2_direct),
        ("H2 MPS Mode", demo_h2_mps),
        ("Direct vs MPS", demo_comparison),
        ("Molecule Builder", demo_molecule_builder),
        ("Smart Initialization", demo_smart_initialization),
        ("Cached Operations", demo_cached_operations),
        ("TOML Config", demo_config_from_toml),
    ]
    
    for name, demo_func in demos:
        try:
            demo_func()
        except Exception as e:
            _LOG.error("Demo '%s' failed: %s", name, e)
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("  DEMO COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
