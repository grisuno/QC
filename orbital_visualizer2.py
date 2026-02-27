#!/usr/bin/env python3
"""
Hydrogen Orbital Visualizer
============================
HIGH RESOLUTION visualization with Hamiltonian NN.
FIXED: Large image size, proper point sizes, high quality output.
"""

import numpy as np
from scipy.special import sph_harm, factorial, genlaguerre
import matplotlib.pyplot as plt
from matplotlib import cm
import torch
import os
import sys
import warnings
from typing import Dict, Tuple, Any

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from schrodinger_crystal_fixed2 import (
        Config as BaseConfig,
        HamiltonianBackbone,
        HamiltonianInferenceEngine,
        LoggerFactory,
    )
    FROM_SCHRODINGER = True
except ImportError as e:
    print(f"Warning: Could not import from schrodinger_crystal_fixed2: {e}")
    FROM_SCHRODINGER = False

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# =============================================================================
# CONFIG - LARGE IMAGE, NOT TINY 16x16
# =============================================================================
class Config:
    # Model config (16x16 is for the NN, NOT for visualization)
    GRID_SIZE = 16
    HIDDEN_DIM = 32
    NUM_SPECTRAL_LAYERS = 2
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # CHECKPOINT
    CHECKPOINT_PATH = 'checkpoint_phase3_training_epoch_4994_20260216_142858.pth'
    BACKBONE_CHECKPOINT_PATH = 'weights/latest.pth'
    BACKBONE_ENABLED = True
    NORMALIZATION_EPS = 1e-8
    
    # Monte Carlo
    MONTE_CARLO_BATCH_SIZE = 100000
    MONTE_CARLO_MAX_PARTICLES = 2000000
    MONTE_CARLO_MIN_PARTICLES = 5000
    ORBITAL_R_MAX_FACTOR = 4
    ORBITAL_R_MAX_OFFSET = 10
    ORBITAL_PROBABILITY_SAFETY_FACTOR = 1.05
    ORBITAL_GRID_SEARCH_R = 300
    ORBITAL_GRID_SEARCH_THETA = 150
    ORBITAL_GRID_SEARCH_PHI = 150
    
    # VISUALIZATION - LARGE IMAGES!
    FIGURE_DPI = 150
    FIGURE_SIZE_X = 24  # inches
    FIGURE_SIZE_Y = 18  # inches
    # Result: 3600 x 2700 pixels - LARGE!
    
    HISTOGRAM_BINS = 300  # More bins = more detail
    SCATTER_SIZE_MIN = 1.0
    SCATTER_SIZE_MAX = 8.0
    MAX_PLOTLY_POINTS = 100000
    
    HAMILTONIAN_EVOLUTION_STEPS = 5
    HAMILTONIAN_DT = 0.005


class WavefunctionCalculator:
    """Calculates hydrogen atom wavefunctions."""
    
    @staticmethod
    def radial_wavefunction(n: int, l: int, r: np.ndarray) -> np.ndarray:
        if l >= n or l < 0:
            return np.zeros_like(r)
        norm = np.sqrt((2.0 / n)**3 * factorial(n - l - 1) / (2 * n * factorial(n + l)))
        rho = 2.0 * r / n
        laguerre = genlaguerre(n - l - 1, 2 * l + 1)(rho)
        R = norm * np.power(rho, l) * laguerre * np.exp(-rho / 2)
        return np.nan_to_num(R, nan=0.0, posinf=0.0, neginf=0.0)
    
    @staticmethod
    def spherical_harmonic_real(l: int, m: int, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
        Y = sph_harm(abs(m), l, phi, theta)
        if m == 0:
            return Y.real
        elif m > 0:
            return np.sqrt(2) * Y.real * ((-1)**m)
        else:
            return np.sqrt(2) * Y.imag * ((-1)**abs(m))
    
    @staticmethod
    def psi_on_grid(n: int, l: int, m: int, grid_size: int = 16) -> torch.Tensor:
        x = np.linspace(0, 2*np.pi, grid_size)
        y = np.linspace(0, 2*np.pi, grid_size)
        X, Y = np.meshgrid(x, y, indexing='ij')
        cx, cy = np.pi, np.pi
        r = np.sqrt((X - cx)**2 + (Y - cy)**2) * (n * 2) / np.pi
        phi = np.arctan2(Y - cy, X - cx)
        theta = np.ones_like(r) * np.pi / 2
        
        psi_grid = np.zeros((grid_size, grid_size))
        for i in range(grid_size):
            for j in range(grid_size):
                R = WavefunctionCalculator.radial_wavefunction(n, l, np.array([r[i, j]]))[0]
                Y_harm = WavefunctionCalculator.spherical_harmonic_real(l, m, np.array([theta[i, j]]), np.array([phi[i, j]]))[0]
                psi_grid[i, j] = np.real(R * Y_harm)
        
        return torch.tensor(psi_grid, dtype=torch.float32)


class HamiltonianNNProcessor:
    """Uses YOUR TRAINED MODEL for calculations."""
    
    def __init__(self, engine):
        self.engine = engine
        self.device = Config.DEVICE
    
    def is_model_loaded(self) -> bool:
        return self.engine is not None and self.engine.backbone is not None
    
    def compute_expected_energy(self, n: int, l: int, m: int) -> Dict[str, float]:
        if not self.is_model_loaded():
            return {'energy_nn': None, 'energy_analytical': -0.5 / (n * n)}
        
        psi_grid = WavefunctionCalculator.psi_on_grid(n, l, m, Config.GRID_SIZE)
        
        with torch.no_grad():
            H_psi = self.engine.apply_hamiltonian(psi_grid.to(self.device))
        
        psi_flat = psi_grid.flatten().to(self.device)
        H_psi_flat = H_psi.flatten()
        
        expectation = torch.dot(psi_flat, H_psi_flat).item()
        norm = torch.dot(psi_flat, psi_flat).item()
        energy_nn = expectation / (norm + 1e-10)
        energy_analytical = -0.5 / (n * n)
        
        return {
            'energy_nn': energy_nn,
            'energy_analytical': energy_analytical,
            'energy_difference': abs(energy_nn - energy_analytical)
        }


class MonteCarloSampler:
    """Monte Carlo sampling for orbital visualization."""
    
    def __init__(self, hamiltonian_processor=None):
        self.hamiltonian_processor = hamiltonian_processor
    
    def find_max_probability(self, n: int, l: int, m: int) -> Tuple[float, float, float, float]:
        r_max = Config.ORBITAL_R_MAX_FACTOR * n * n + Config.ORBITAL_R_MAX_OFFSET
        
        r_vals = np.linspace(0.01, r_max, Config.ORBITAL_GRID_SEARCH_R)
        theta_vals = np.linspace(0.01, np.pi - 0.01, Config.ORBITAL_GRID_SEARCH_THETA)
        phi_vals = np.linspace(0, 2*np.pi, Config.ORBITAL_GRID_SEARCH_PHI)
        
        max_prob = 0.0
        best = (0.0, 0.0, 0.0)
        
        R_grid = WavefunctionCalculator.radial_wavefunction(n, l, r_vals)
        
        for theta in theta_vals:
            sin_theta = np.sin(theta)
            if sin_theta < 0.01:
                continue
            for j, r in enumerate(r_vals):
                R = R_grid[j]
                if np.abs(R) < 1e-10:
                    continue
                for phi in phi_vals:
                    Y = WavefunctionCalculator.spherical_harmonic_real(l, m, np.array([theta]), np.array([phi]))[0]
                    prob = np.abs(R * Y)**2 * r**2 * sin_theta
                    if prob > max_prob:
                        max_prob = prob
                        best = (r, theta, phi)
        
        return max_prob, best[0], best[1], best[2]
    
    def sample(self, n: int, l: int, m: int, num_samples: int) -> Dict[str, Any]:
        num_samples = max(Config.MONTE_CARLO_MIN_PARTICLES, min(Config.MONTE_CARLO_MAX_PARTICLES, num_samples))
        
        print(f"\nMonte Carlo: n={n}, l={l}, m={m}, target={num_samples:,} particles")
        
        # Compute energy with YOUR MODEL
        energy_info = None
        if self.hamiltonian_processor and self.hamiltonian_processor.is_model_loaded():
            print("Computing energy using YOUR TRAINED MODEL...")
            energy_info = self.hamiltonian_processor.compute_expected_energy(n, l, m)
            print(f"  E_NN: {energy_info['energy_nn']:.6f}")
            print(f"  E_analytical: {energy_info['energy_analytical']:.6f}")
        
        P_max, _, _, _ = self.find_max_probability(n, l, m)
        print(f"P_max = {P_max:.8f}")
        
        if P_max < 1e-15:
            P_max = 1e-10
        
        r_max = Config.ORBITAL_R_MAX_FACTOR * n * n + Config.ORBITAL_R_MAX_OFFSET
        P_threshold = P_max * Config.ORBITAL_PROBABILITY_SAFETY_FACTOR
        
        points_x, points_y, points_z = [], [], []
        points_prob, points_phase = [], []
        total_attempts = 0
        
        while len(points_x) < num_samples and total_attempts < num_samples * 150:
            total_attempts += Config.MONTE_CARLO_BATCH_SIZE
            
            r_batch = r_max * (np.random.uniform(0, 1, Config.MONTE_CARLO_BATCH_SIZE) ** (1/3))
            theta_batch = np.arccos(1 - 2 * np.random.uniform(0, 1, Config.MONTE_CARLO_BATCH_SIZE))
            phi_batch = np.random.uniform(0, 2*np.pi, Config.MONTE_CARLO_BATCH_SIZE)
            
            R_batch = WavefunctionCalculator.radial_wavefunction(n, l, r_batch)
            Y_batch = WavefunctionCalculator.spherical_harmonic_real(l, m, theta_batch, phi_batch)
            psi_batch = R_batch * Y_batch
            prob_batch = np.abs(psi_batch)**2
            prob_vol_batch = prob_batch * r_batch**2 * np.sin(theta_batch)
            
            u_batch = np.random.uniform(0, P_threshold, Config.MONTE_CARLO_BATCH_SIZE)
            accepted = u_batch < prob_vol_batch
            
            r_acc = r_batch[accepted]
            theta_acc = theta_batch[accepted]
            phi_acc = phi_batch[accepted]
            
            sin_t = np.sin(theta_acc)
            points_x.extend((r_acc * sin_t * np.cos(phi_acc)).tolist())
            points_y.extend((r_acc * sin_t * np.sin(phi_acc)).tolist())
            points_z.extend((r_acc * np.cos(theta_acc)).tolist())
            points_prob.extend(prob_batch[accepted].tolist())
            points_phase.extend(np.real(psi_batch[accepted]).tolist())
        
        points_x = np.array(points_x[:num_samples])
        points_y = np.array(points_y[:num_samples])
        points_z = np.array(points_z[:num_samples])
        points_prob = np.array(points_prob[:num_samples])
        points_phase = np.array(points_phase[:num_samples])
        
        efficiency = len(points_x) / total_attempts * 100
        print(f"Accepted: {len(points_x):,} / {total_attempts:,} ({efficiency:.2f}%)")
        
        return {
            'x': points_x, 'y': points_y, 'z': points_z,
            'prob': points_prob, 'phase': points_phase,
            'n': n, 'l': l, 'm': m, 'r_max': r_max,
            'efficiency': efficiency, 'energy_info': energy_info
        }


class OrbitalVisualizer:
    """HIGH RESOLUTION visualization - NOT 16x16!"""
    
    def visualize(self, data: Dict, save_path: str = None, hamiltonian_processor=None):
        X, Y, Z = data['x'], data['y'], data['z']
        probs, phases = data['prob'], data['phase']
        n, l, m = data['n'], data['l'], data['m']
        energy_info = data.get('energy_info')
        
        max_prob = np.max(probs) if np.max(probs) > 0 else 1.0
        prob_norm = probs / max_prob
        
        # LARGE FIGURE - 24x18 inches at 150 DPI = 3600x2700 pixels
        fig = plt.figure(figsize=(Config.FIGURE_SIZE_X, Config.FIGURE_SIZE_Y), 
                        dpi=Config.FIGURE_DPI)
        fig.patch.set_facecolor('#000008')
        
        # 2x2 grid
        ax1 = fig.add_subplot(221, projection='3d')
        ax2 = fig.add_subplot(222)
        ax3 = fig.add_subplot(223)
        ax4 = fig.add_subplot(224)
        
        for ax in [ax2, ax3, ax4]:
            ax.set_facecolor('#000008')
        ax1.set_facecolor('#000008')
        
        # 3D SCATTER - LARGER POINTS
        colors_rgba = np.zeros((len(X), 4))
        pos_mask = phases >= 0
        
        colors_rgba[pos_mask] = [1.0, 0.3, 0.0, 0.5]
        colors_rgba[~pos_mask] = [0.0, 0.5, 1.0, 0.5]
        
        # LARGER SIZES based on probability
        sizes = Config.SCATTER_SIZE_MIN + prob_norm * (Config.SCATTER_SIZE_MAX - Config.SCATTER_SIZE_MIN)
        ax1.scatter(X, Y, Z, c=colors_rgba, s=sizes, alpha=0.5, depthshade=True)
        
        orbital_type = ['s', 'p', 'd', 'f', 'g'][l] if l < 5 else f'l={l}'
        ax1.set_title(f'Orbital {n}{orbital_type} (n={n}, l={l}, m={m})\n{len(X):,} particles', 
                     color='white', fontsize=14, fontweight='bold')
        ax1.set_xlabel('x (a0)', color='white', fontsize=12)
        ax1.set_ylabel('y (a0)', color='white', fontsize=12)
        ax1.set_zlabel('z (a0)', color='white', fontsize=12)
        ax1.tick_params(colors='white')
        ax1.xaxis.pane.fill = False
        ax1.yaxis.pane.fill = False
        ax1.zaxis.pane.fill = False
        
        # XY PROJECTION - MORE BINS = MORE DETAIL
        H, xe, ye = np.histogram2d(X, Y, bins=Config.HISTOGRAM_BINS, weights=probs)
        im2 = ax2.imshow(H.T**0.3, extent=[xe[0], xe[-1], ye[0], ye[-1]], 
                        origin='lower', cmap='inferno', aspect='equal', 
                        interpolation='gaussian')
        ax2.set_title('XY Projection (top view)', color='white', fontsize=14)
        ax2.set_xlabel('x (a0)', color='white', fontsize=12)
        ax2.set_ylabel('y (a0)', color='white', fontsize=12)
        ax2.tick_params(colors='white')
        plt.colorbar(im2, ax=ax2, shrink=0.8)
        
        # XZ PROJECTION
        H_xz, xxe, zze = np.histogram2d(X, Z, bins=Config.HISTOGRAM_BINS, weights=probs)
        im3 = ax3.imshow(H_xz.T**0.3, extent=[xxe[0], xxe[-1], zze[0], zze[-1]],
                        origin='lower', cmap='viridis', aspect='equal', 
                        interpolation='gaussian')
        ax3.set_title('XZ Projection (side view)', color='white', fontsize=14)
        ax3.set_xlabel('x (a0)', color='white', fontsize=12)
        ax3.set_ylabel('z (a0)', color='white', fontsize=12)
        ax3.tick_params(colors='white')
        plt.colorbar(im3, ax=ax3, shrink=0.8)
        
        # INFO PANEL
        ax4.axis('off')
        r_vals = np.sqrt(X**2 + Y**2 + Z**2)
        
        model_status = "LOADED" if hamiltonian_processor and hamiltonian_processor.is_model_loaded() else "ANALYTICAL"
        
        energy_text = ""
        if energy_info and energy_info.get('energy_nn') is not None:
            energy_text = f"""
ENERGY (YOUR MODEL)
  E_NN:          {energy_info['energy_nn']:.6f}
  E_analytical:  {energy_info['energy_analytical']:.6f}
  Difference:    {energy_info['energy_difference']:.6f}
"""
        
        info = f"""
{'='*50}
HYDROGEN ORBITAL VISUALIZER
{'='*50}

Orbital: n={n}, l={l}, m={m}
Type: {n}{orbital_type}

PARTICLES
  Total:      {len(X):>12,}
  Efficiency: {data['efficiency']:>12.2f}%

STATISTICS
  r_mean: {np.mean(r_vals):>10.3f} a0
  r_std:  {np.std(r_vals):>10.3f} a0
  r_max:  {np.max(r_vals):>10.3f} a0
{energy_text}
HAMILTONIAN NN
  Status: {model_status}
  Checkpoint: {Config.CHECKPOINT_PATH}

COLOR
  Red/Orange:  Positive phase (+)
  Blue/Cyan:   Negative phase (-)
{'='*50}
"""
        
        ax4.text(0.05, 0.95, info, transform=ax4.transAxes,
                fontfamily='monospace', fontsize=11, color='white',
                verticalalignment='top')
        
        plt.tight_layout()
        
        if save_path:
            print(f"Saving: {save_path}")
            plt.savefig(save_path, dpi=Config.FIGURE_DPI, facecolor='#000008', 
                       bbox_inches='tight')
            print(f"Saved: {Config.FIGURE_SIZE_X*Config.FIGURE_DPI}x{Config.FIGURE_SIZE_Y*Config.FIGURE_DPI} pixels")
        
        plt.show()
        
        # Plotly
        if PLOTLY_AVAILABLE:
            self._plotly(X, Y, Z, prob_norm, phases, n, l, m)
    
    def _plotly(self, X, Y, Z, prob_norm, phases, n, l, m):
        max_pts = min(Config.MAX_PLOTLY_POINTS, len(X))
        if len(X) > max_pts:
            idx = np.random.choice(len(X), max_pts, replace=False)
            X, Y, Z = X[idx], Y[idx], Z[idx]
            prob_norm, phases = prob_norm[idx], phases[idx]
        
        colors = np.where(phases >= 0, prob_norm, -prob_norm)
        
        fig = go.Figure(data=go.Scatter3d(
            x=X, y=Y, z=Z, mode='markers',
            marker=dict(size=2, color=colors,
                       colorscale=[[0, '#0066ff'], [0.5, '#111111'], [1, '#ff3300']],
                       opacity=0.5)
        ))
        fig.update_layout(
            title=f'Orbital n={n}, l={l}, m={m}',
            scene=dict(bgcolor='#050510'),
            paper_bgcolor='#050510', font=dict(color='white'),
            width=1200, height=900
        )
        fig.show()


ORBITALS = {
    '1s': (1, 0, 0), '2s': (2, 0, 0),
    '2p_z': (2, 1, 0), '2p_x': (2, 1, 1), '2p_y': (2, 1, -1),
    '3s': (3, 0, 0), '3p_z': (3, 1, 0), '3p_x': (3, 1, 1), '3p_y': (3, 1, -1),
    '3d_z2': (3, 2, 0), '3d_xz': (3, 2, 1), '3d_yz': (3, 2, -1),
    '3d_xy': (3, 2, 2), '3d_x2-y2': (3, 2, -2),
    '4s': (4, 0, 0), '4p_z': (4, 1, 0), '4d_z2': (4, 2, 0), 
    '4f_z3': (4, 3, 0), '4f_xz2': (4, 3, 1), '4f_yz2': (4, 3, -1),
    '4f_xyz': (4, 3, 2), '4f_z(x2-y2)': (4, 3, -2),
    '5s': (5, 0, 0), '5p_z': (5, 1, 0), '5d_z2': (5, 2, 0),
}


def main():
    print("\n" + "="*50)
    print("HYDROGEN ORBITAL VISUALIZER")
    print(f"Image size: {Config.FIGURE_SIZE_X*Config.FIGURE_DPI}x{Config.FIGURE_SIZE_Y*Config.FIGURE_DPI} pixels")
    print("="*50)
    
    hamiltonian_processor = None
    engine = None
    
    if FROM_SCHRODINGER:
        try:
            engine = HamiltonianInferenceEngine(Config)
            if engine.backbone:
                hamiltonian_processor = HamiltonianNNProcessor(engine)
                print(f"YOUR MODEL: {Config.CHECKPOINT_PATH}")
        except Exception as e:
            print(f"No model: {e}")
    
    sampler = MonteCarloSampler(hamiltonian_processor)
    visualizer = OrbitalVisualizer()
    
    print(f"\nOrbitals: {list(ORBITALS.keys())}")
    
    while True:
        try:
            choice = input("\nOrbital (or 'q'): ").strip().lower()
            if choice == 'q':
                break
            if choice not in ORBITALS:
                continue
            
            n, l, m = ORBITALS[choice]
            
            num = input(f"Particles [default=100000]: ").strip()
            num_samples = int(num) if num else 100000
            num_samples = max(Config.MONTE_CARLO_MIN_PARTICLES, 
                            min(Config.MONTE_CARLO_MAX_PARTICLES, num_samples))
            
            data = sampler.sample(n, l, m, num_samples)
            
            save = input("Save? [y/N]: ").strip().lower() == 'y'
            save_path = f"orbital_{choice}_{num_samples}.png" if save else None
            
            visualizer.visualize(data, save_path, hamiltonian_processor)
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()


if __name__ == '__main__':
    main()
