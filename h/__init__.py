"""H: Hamiltonian operator network on the 2D torus, trained so the checkpoint actually computes H.

Same architecture and state_dict layout as the project's HamiltonianBackboneNet, so the resulting .pth loads
as `hamiltonian.pth` in quantum_computer.QuantumComputer. Training follows the Maxwell protocol: GOE/GUE
imaginary-ratio search, batch-size prospecting, seed mining, long training with an adaptive lambda pressure.
"""

from .config import HConfig
from .model import HamiltonianTorusNet

__all__ = ["HConfig", "HamiltonianTorusNet"]
