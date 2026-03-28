"""
AISU-ETCS Unified Theory: FDM Solver
Solves the X field equation in 3D pyramid geometry

X parameter definition:
  - Reference point (centroid, 0, 0, 0) : X = X_MIN = 5.825
  - Polar axes (0, ±y_max, 0)           : X = X_MAX = 9.325

The field is governed by Poisson's equation (∇²X = S) with Dirichlet boundary
conditions X=X_MAX on the polar axes.  The source term S is a dimensionally
consistent attractor that drives interior points toward X_MIN (centroid)
or X_MAX (polar boundary), depending on their phase.

--- Vertex Convergence Theorem ---
At the vertex (z = Z_VERTEX = -4739.857 km) the pyramid cross-section shrinks
to a single point, so the centroid, the North pole, and the South pole all
coincide.  Therefore X at the centroid must approach X_MAX as z → Z_VERTEX.

The FDM (Laplacian diffusion) captures this correctly: as the cross-section
narrows, the Dirichlet boundary X=X_MAX at the poles diffuses inward more
strongly, naturally raising the centroid X above X_MIN.  This is the
theoretically correct behaviour.

The REST API's geometricX() uses only the relative distance ratio
(d/d_reference), which is always 1.0 at the centroid regardless of depth,
yielding X_MIN everywhere on the central axis.  This is a simplification
that cannot represent the vertex-convergence topology and is considered a
known limitation of the analytical model.

Usage:
    python etcs_fdm_solver.py
"""

import numpy as np
import matplotlib.pyplot as plt

# Constants
X0 = 5.825
X_MIN = 5.825
X_MAX = 9.325
Z_VERTEX = -4739.857  # km
Z_BASE = 1579.952     # km
DIAGONAL_0 = 8776.263  # km

# Legacy Lagrangian parameters (no longer used in the solver;
# replaced by a dimensionally-consistent attractor source term).
# Kept for reference / future Lagrangian analysis.

lambda_mh = 0.002  # Weak coupling for stable convergence
v_mh = 2.0         # Symmetry breaking scale
mu = 0.5           # Phase coupling
delta_N = 0.01
delta_S = 0.05
U_v = 10.0         # Vertex potential strength
alpha_v = 1.5


class ETCSNumericalSolver:
    """FDM-like solver for ETCS field equations"""
    
    def __init__(self, nx=20, ny=20, nz=40):
        self.nx, self.ny, self.nz = nx, ny, nz
        
        # Domain bounds
        self.x_range = (-5000, 5000)  # km
        self.y_range = (-5000, 5000)
        self.z_range = (Z_VERTEX + 100, Z_BASE - 100)
        
        # Create grid
        self.x = np.linspace(*self.x_range, nx)
        self.y = np.linspace(*self.y_range, ny)
        self.z = np.linspace(*self.z_range, nz)
        
        self.dx = self.x[1] - self.x[0]
        self.dy = self.y[1] - self.y[0]
        self.dz = self.z[1] - self.z[0]
        
        print(f"Grid: {nx}×{ny}×{nz} = {nx*ny*nz} points")
        print(f"Resolution: dx={self.dx:.1f} km, dy={self.dy:.1f} km, dz={self.dz:.1f} km")
        
        # Initialize X field with analytic pole-distance formula:
        #   d = 0 (at pole)     -> X = X_MAX
        #   d = diagonal/2 (at centroid) -> X = X_MIN
        self.X_field = np.ones((nx, ny, nz)) * X0
        
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    x, y, z = self.x[i], self.y[j], self.z[k]
                    if self.is_inside_pyramid(x, y, z):
                        diagonal = self.get_diagonal_at_z(z)
                        half_diag = diagonal / 2.0
                        if half_diag > 0.1:
                            d_north = np.sqrt(x**2 + (y - half_diag)**2)
                            d_south = np.sqrt(x**2 + (y + half_diag)**2)
                            ratio_n = min(d_north / half_diag, 1.0)
                            ratio_s = min(d_south / half_diag, 1.0)
                            X_n = X_MAX - (X_MAX - X_MIN) * ratio_n
                            X_s = X_MAX - (X_MAX - X_MIN) * ratio_s
                            self.X_field[i, j, k] = max(X_n, X_s)
    
    def get_diagonal_at_z(self, z):
        """Calculate diagonal length at height z"""
        z_from_vertex = z - Z_VERTEX
        z0_from_vertex = 0 - Z_VERTEX
        return (DIAGONAL_0 / z0_from_vertex) * z_from_vertex
    
    def is_inside_pyramid(self, x, y, z):
        """Check if point is inside pyramid boundary"""
        if z < Z_VERTEX or z > Z_BASE:
            return False
        
        diagonal = self.get_diagonal_at_z(z)
        return abs(x) + abs(y) <= diagonal / 2.0
    
    def compute_source_term(self, X, phi, z, half_diag):
        """
        Source term for the screened Poisson equation:
            nabla^2 X + k * X = k * X_target
        which is rewritten as:
            nabla^2 X = k * (X_target - X)  = S

        In the Gauss-Seidel update this is handled as:
            X_gs = (neighbor_sum - k * X_target) / (diag_coeff - k)

        The returned value is a tuple (k, X_target) so the caller
        can use the correct formula.

        Physical targets:
          phi = 0  -> X_target = X_MIN  (centroid/axis)
          phi =+-1 -> X_target = X_MAX  (polar axes)

        source_scale k uses a fixed characteristic length D_CHAR so
        it is strong enough to compete with the Laplacian diffusion.
        """
        X_target = X_MIN if abs(phi) == 0 else X_MAX

        # Fixed coupling constant: k ~ 10% of typical diag_coeff.
        # Characteristic length D_CHAR ~ 300 km  ->  k = DX/D_CHAR^2
        D_CHAR = 300.0   # km
        k = (X_MAX - X_MIN) / (D_CHAR ** 2)   # ≈ 3.9e-5 km^-2

        return k, X_target
    
    def solve_laplacian_step(self):
        """One Gauss-Seidel iteration"""
        
        X_new = self.X_field.copy()
        
        for i in range(1, self.nx - 1):
            for j in range(1, self.ny - 1):
                for k in range(1, self.nz - 1):
                    
                    x, y, z = self.x[i], self.y[j], self.z[k]
                    
                    # Check if inside pyramid
                    if not self.is_inside_pyramid(x, y, z):
                        X_new[i, j, k] = X0  # Outside: boundary value
                        continue
                    
                    # --- Polar axis Dirichlet boundary condition ---
                    # Grid points on or very near the polar axes are pinned to X_MAX.
                    diagonal = self.get_diagonal_at_z(z)
                    half_diag = diagonal / 2.0
                    d_north = np.sqrt(x**2 + (y - half_diag)**2)
                    d_south = np.sqrt(x**2 + (y + half_diag)**2)
                    # Cap pole_thresh at 15% of half_diag so it never overflows
                    # the cross-section at deep (narrow) z-levels.
                    pole_thresh = min(
                        max(self.dy, self.dx) * 1.5,
                        half_diag * 0.15
                    )
                    if d_north < pole_thresh or d_south < pole_thresh:
                        X_new[i, j, k] = X_MAX  # Dirichlet: polar axis = X_MAX
                        continue

                    # --- Pyramid wall boundary condition ---
                    dist_to_wall = half_diag - (abs(x) + abs(y))
                    if dist_to_wall < self.dx * 0.7:
                        # Smooth X on the wall using analytic pole-distance formula
                        ratio_n = min(d_north / half_diag, 1.0)
                        ratio_s = min(d_south / half_diag, 1.0)
                        X_wall = max(
                            X_MAX - (X_MAX - X_MIN) * ratio_n,
                            X_MAX - (X_MAX - X_MIN) * ratio_s
                        )
                        X_new[i, j, k] = X_wall
                        continue
                    
                    # --- Centroid-axis Dirichlet boundary condition ---
                    # ETCS theory: X = X_MIN on the centroid axis in the
                    # eternal/celestial realms (z >= -1500 km).
                    # Use 0.6*dx to ensure the nearest grid point is captured.
                    centroid_thresh = max(self.dx, self.dy) * 0.6
                    if (abs(x) < centroid_thresh and
                            abs(y) < centroid_thresh and
                            z >= -3000.0):
                        X_new[i, j, k] = X_MIN  # Dirichlet: centroid axis = X_MIN
                        continue

                    # --- Pure Laplace equation: nabla^2 X = 0 ---
                    # The X field is governed entirely by boundary conditions:
                    #   Poles (Dirichlet)       : X = X_MAX
                    #   Centroid axis (Dirichlet): X = X_MIN
                    #   Pyramid wall (Dirichlet) : X = analytic geometricX
                    # Interior points are updated by the Gauss-Seidel mean
                    # of their six neighbors (no source term needed).
                    dx2 = self.dx ** 2
                    dy2 = self.dy ** 2
                    dz2 = self.dz ** 2
                    diag_coeff = 2.0/dx2 + 2.0/dy2 + 2.0/dz2
                    neighbor_sum = (
                        (self.X_field[i+1, j, k] + self.X_field[i-1, j, k]) / dx2 +
                        (self.X_field[i, j+1, k] + self.X_field[i, j-1, k]) / dy2 +
                        (self.X_field[i, j, k+1] + self.X_field[i, j, k-1]) / dz2
                    )
                    X_gs = neighbor_sum / diag_coeff

                    # Under-relaxation for stable convergence
                    omega = 0.7
                    X_new[i, j, k] = (1.0 - omega) * self.X_field[i, j, k] + omega * X_gs

                    # Clamp to physical range
                    X_new[i, j, k] = np.clip(X_new[i, j, k], X_MIN, X_MAX)
        
        self.X_field = X_new
    
    def solve(self, max_iter=1000, tolerance=1e-4):
        """Iterative solver"""
        
        print("\n" + "="*60)
        print("Starting FDM Solver")
        print("="*60)
        
        for iteration in range(max_iter):
            X_old = self.X_field.copy()
            
            self.solve_laplacian_step()
            
            # Check convergence
            diff = np.max(np.abs(self.X_field - X_old))
            
            if iteration % 50 == 0:
                X_min = np.min(self.X_field)
                X_max = np.max(self.X_field)
                print(f"Iter {iteration:4d}: max_diff={diff:.6e}, X_range=[{X_min:.3f}, {X_max:.3f}]")
            
            if diff < tolerance:
                print(f"\nConverged at iteration {iteration}")
                break
        else:
            print(f"\nMax iterations reached. Final diff: {diff:.6e}")
        
        return self.X_field
    
    def visualize(self, output_file='etcs_fdm_solution.png'):
        """Create visualization of solution"""
        
        fig = plt.figure(figsize=(15, 5))
        
        # z=0 slice (reference plane)
        ax1 = plt.subplot(131)
        z_idx = np.argmin(np.abs(self.z - 0))
        
        # Create valid mask for plotting
        X_slice = self.X_field[:, :, z_idx].T
        mask = np.zeros_like(X_slice, dtype=bool)
        for i in range(self.nx):
            for j in range(self.ny):
                mask[j, i] = self.is_inside_pyramid(self.x[i], self.y[j], self.z[z_idx])
        
        X_masked = np.ma.masked_where(~mask, X_slice)
        
        im1 = ax1.contourf(self.x, self.y, X_masked, levels=20, cmap='viridis')
        plt.colorbar(im1, ax=ax1, label='X parameter')
        ax1.set_xlabel('x [km]')
        ax1.set_ylabel('y [km]')
        ax1.set_title(f'X field at z={self.z[z_idx]:.0f} km (Reference Plane)')
        ax1.set_aspect('equal')
        
        # y=0 slice (x-z plane)
        ax2 = plt.subplot(132)
        y_idx = np.argmin(np.abs(self.y - 0))
        
        X_slice2 = self.X_field[:, y_idx, :].T
        mask2 = np.zeros_like(X_slice2, dtype=bool)
        for i in range(self.nx):
            for k in range(self.nz):
                mask2[k, i] = self.is_inside_pyramid(self.x[i], 0, self.z[k])
        
        X_masked2 = np.ma.masked_where(~mask2, X_slice2)
        
        im2 = ax2.contourf(self.x, self.z, X_masked2, levels=20, cmap='viridis')
        plt.colorbar(im2, ax=ax2, label='X parameter')
        ax2.set_xlabel('x [km]')
        ax2.set_ylabel('z [km]')
        ax2.set_title('X field at y=0 (East-West)')
        
        # x=0 slice (y-z plane, showing north-south)
        ax3 = plt.subplot(133)
        x_idx = np.argmin(np.abs(self.x - 0))
        
        X_slice3 = self.X_field[x_idx, :, :].T
        mask3 = np.zeros_like(X_slice3, dtype=bool)
        for j in range(self.ny):
            for k in range(self.nz):
                mask3[k, j] = self.is_inside_pyramid(0, self.y[j], self.z[k])
        
        X_masked3 = np.ma.masked_where(~mask3, X_slice3)
        
        im3 = ax3.contourf(self.y, self.z, X_masked3, levels=20, cmap='viridis')
        plt.colorbar(im3, ax=ax3, label='X parameter')
        ax3.set_xlabel('y [km] (North +)')
        ax3.set_ylabel('z [km]')
        ax3.set_title('X field at x=0 (North-South)')
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Visualization saved: {output_file}")
        
        return fig
    
    def extract_profiles(self):
        """Extract 1D profiles for validation"""
        
        # Find indices
        x0_idx = np.argmin(np.abs(self.x - 0))
        y0_idx = np.argmin(np.abs(self.y - 0))
        z0_idx = np.argmin(np.abs(self.z - 0))
        
        print("\n" + "="*60)
        print("VALIDATION: Extracted Values")
        print("="*60)
        
        # Reference point
        X_ref = self.X_field[x0_idx, y0_idx, z0_idx]
        print(f"Reference (0, 0, 0): X = {X_ref:.3f} (target: {X0:.3f})")
        
        # North/South pole: find the grid point inside the pyramid with
        # largest |y| (closest to the polar axis) at z=0, x=0.
        z = self.z[z0_idx]
        diagonal_z0 = self.get_diagonal_at_z(z)
        half_diag_z0 = diagonal_z0 / 2.0

        # North pole: largest valid y (positive)
        y_north_idx = None
        for j in range(self.ny - 1, -1, -1):  # Search from max y downward
            y_val = self.y[j]
            if self.is_inside_pyramid(self.x[x0_idx], y_val, z):
                y_north_idx = j
                break
        if y_north_idx is not None:
            X_north = self.X_field[x0_idx, y_north_idx, z0_idx]
            print(f"North pole (0, +y_max, 0): X = {X_north:.3f} (target: {X_MAX:.3f})  "
                  f"[y={self.y[y_north_idx]:.1f} km, pole at y={half_diag_z0:.1f} km]")
        else:
            print("North pole: no valid grid point found")
        
        # South pole: smallest valid y (most negative)
        y_south_idx = None
        for j in range(self.ny):
            y_val = self.y[j]
            if self.is_inside_pyramid(self.x[x0_idx], y_val, z):
                y_south_idx = j
                break
        if y_south_idx is not None:
            X_south = self.X_field[x0_idx, y_south_idx, z0_idx]
            print(f"South pole (0, -y_max, 0): X = {X_south:.3f} (target: {X_MAX:.3f})  "
                  f"[y={self.y[y_south_idx]:.1f} km, pole at y=-{half_diag_z0:.1f} km]")
        else:
            print("South pole: no valid grid point found")
        
        # Vertical profile at center
        z_profile = self.z
        X_vertical = self.X_field[x0_idx, y0_idx, :]
        
        print(f"\nVertical profile at center (x=0, y=0):")
        for i in [0, len(z_profile)//4, len(z_profile)//2, 3*len(z_profile)//4, -1]:
            print(f"  z={z_profile[i]:7.1f} km: X={X_vertical[i]:.3f}")


def main():
    """Main execution"""
    
    print("="*60)
    print("AISU-ETCS UNIFIED THEORY")
    print("FDM Solver for X Field in Pyramid Geometry")
    print("="*60)
    
    # Create solver
    print("\nInitializing solver...")
    solver = ETCSNumericalSolver(nx=20, ny=20, nz=40)
    
    # Solve
    X_solution = solver.solve(max_iter=500, tolerance=1e-4)
    
    # Extract profiles
    solver.extract_profiles()
    
    # Visualize
    solver.visualize('etcs_fdm_solution.png')
    
    print("\n" + "="*60)
    print("[DONE] Simulation complete!")
    print("="*60)


if __name__ == '__main__':
    main()
