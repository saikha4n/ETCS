#!/usr/bin/env python3
"""
Generate all figures for AISU-ETCS unified theory paper
Usage: python3 generate_paper_figures.py

Fixes vs original:
  - FDM solver is run ONCE and shared across all figures (consistency)
  - Unicode symbols removed from print() for Windows cp932 compatibility
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import rcParams

# Publication settings
rcParams['font.size'] = 11
rcParams['font.family'] = 'serif'
rcParams['figure.dpi'] = 300
rcParams['savefig.dpi'] = 300
rcParams['savefig.bbox'] = 'tight'

# Import FDM solver
try:
    from etcs_fdm_solver import ETCSNumericalSolver, X0, X_MIN, X_MAX, Z_VERTEX, DIAGONAL_0
    from etcs_validation import ProperTimeCalculator, ETCSValidator
    HAVE_FDM = True
except ImportError:
    HAVE_FDM = False
    print("Warning: FDM solver not found. Only Figure 1 will be generated.")


# ---------------------------------------------------------------------------
# Shared FDM solve (run once, reuse across all figures)
# ---------------------------------------------------------------------------

def run_solver():
    """Run the FDM solver once and return the solver object."""
    print("Running FDM solver (30x30x60)...")
    solver = ETCSNumericalSolver(nx=30, ny=30, nz=60)
    solver.solve(max_iter=2000, tolerance=1e-5)
    print("  FDM solve complete.")
    return solver


# ---------------------------------------------------------------------------
# Figure 1: Attractor Source Term Model
# ---------------------------------------------------------------------------

def figure1_source_term(solver):
    """
    Figure 1: Actual attractor source term used in the FDM solver.

    The source term for the screened Poisson equation is:
        S(X, phi) = k * (X_target - X)
    where X_target = X_MIN (phi=0) or X_MAX (phi=+-1),
    and k = (X_MAX - X_MIN) / D_CHAR^2  with D_CHAR = 300 km.

    This replaces the deprecated Mexican Hat potential that is no longer
    used in the solver implementation.
    """
    print("Generating Figure 1: Attractor Source Term...")

    D_CHAR = 300.0
    k = (X_MAX - X_MIN) / D_CHAR**2

    X_vals = np.linspace(X_MIN - 0.3, X_MAX + 0.3, 500)

    # Source term = k * (X_target - X) for each phase
    S_north = k * (X_MAX - X_vals)   # phi=+1: target X_MAX
    S_axis  = k * (X_MIN - X_vals)   # phi= 0: target X_MIN
    S_south = k * (X_MAX - X_vals)   # phi=-1: target X_MAX (same as north)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)

    # Left: Source term S(X)
    ax = axes[0]
    ax.plot(X_vals, S_north, 'b-', linewidth=2, label=r'$\phi = \pm1$ (Polar)')
    ax.plot(X_vals, S_axis,  'k-', linewidth=2, label=r'$\phi = 0$ (Axis)')
    ax.axhline(0, color='gray', linewidth=0.8, linestyle='--')
    ax.axvline(X_MIN, color='gray', linewidth=0.8, linestyle=':', alpha=0.7)
    ax.axvline(X_MAX, color='gray', linewidth=0.8, linestyle=':', alpha=0.7)
    ax.text(X_MIN, ax.get_ylim()[0] if ax.get_ylim()[0] != 0 else -k*0.5,
            r'$X_{\min}$', ha='center', va='top', fontsize=9, color='gray')
    ax.text(X_MAX, k*0.05,
            r'$X_{\max}$', ha='center', va='bottom', fontsize=9, color='gray')
    ax.set_xlabel(r'$X$ parameter')
    ax.set_ylabel(r'Source term $S(X,\phi)$ [km$^{-2}$]')
    ax.set_title('Attractor Source Term\n'
                 r'($\nabla^2 X = k\,(X_{\rm target}-X)$, $k \approx 3.9\times10^{-5}$ km$^{-2}$)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Right: FDM X field at z=0 (y-profile through x=0)
    ax2 = axes[1]
    i_c = np.argmin(np.abs(solver.x - 0))
    k_ref = np.argmin(np.abs(solver.z - 0))
    z_val = solver.z[k_ref]

    diag = DIAGONAL_0 * (z_val - Z_VERTEX) / (0.0 - Z_VERTEX)
    half_diag = diag / 2.0

    y_vals = solver.y
    X_line = solver.X_field[i_c, :, k_ref].copy().astype(float)
    # Mask points outside the pyramid
    for jj, yy in enumerate(y_vals):
        if not solver.is_inside_pyramid(solver.x[i_c], yy, z_val):
            X_line[jj] = np.nan

    ax2.plot(y_vals / 1000, X_line, 'b-', linewidth=2.5, label='FDM Solution')
    ax2.axhline(X_MIN, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax2.axhline(X_MAX, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax2.text(0, X_MIN - 0.12, r'$X_{\min}=5.825$', ha='center', fontsize=9)
    ax2.text(0, X_MAX + 0.12, r'$X_{\max}=9.325$', ha='center', fontsize=9)
    ax2.axvline( half_diag / 1000, color='red', linestyle='--', linewidth=1, alpha=0.6)
    ax2.axvline(-half_diag / 1000, color='red', linestyle='--', linewidth=1, alpha=0.6)
    ax2.plot(0, X_MIN, 'ko', markersize=7, label='Reference point')
    ax2.set_xlabel(r'$y$ [Mm] (North $+$, South $-$)')
    ax2.set_ylabel(r'$X$ parameter')
    ax2.set_title('FDM Solution: X Field at $z=0$')
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(5.5, 9.7)

    plt.tight_layout()
    plt.savefig('figure1_attractor_model.pdf')
    plt.savefig('figure1_attractor_model.png')
    print("  [OK] Saved: figure1_attractor_model.pdf/png")
    plt.close()


# ---------------------------------------------------------------------------
# Figure 2: Vertical proper time profile
# ---------------------------------------------------------------------------

def figure2_vertical_profile(solver):
    """Figure 2: Vertical proper time profile at center (x=0, y=0)."""
    print("Generating Figure 2: Vertical Profile...")

    tau_calc = ProperTimeCalculator()
    i_c = np.argmin(np.abs(solver.x - 0))
    j_c = np.argmin(np.abs(solver.y - 0))

    z_profile = solver.z
    X_profile = solver.X_field[i_c, j_c, :]

    # tau at centroid: phi=0 -> tau_h=1, result = tau_vertical(z)
    tau_profile = [tau_calc.compute_tau_ratio(float(X), 0, float(z))
                   for X, z in zip(X_profile, z_profile)]

    # Mask z values outside pyramid (outside = X=X0 sentinel; also exclude near-vertex)
    valid = []
    for idx, z in enumerate(z_profile):
        inside = solver.is_inside_pyramid(0.0, 0.0, z)
        valid.append(inside)
    valid = np.array(valid)

    tau_arr = np.array(tau_profile, dtype=float)
    z_arr   = np.array(z_profile,   dtype=float)
    tau_arr[~valid] = np.nan

    # REST API reference points
    z_ref   = [0, -1000, -2000, -4000]
    tau_ref = [1.000000, 0.932066, 0.849893, 0.577087]

    fig, ax = plt.subplots(figsize=(9, 4))
    fig.subplots_adjust(left=0.08, right=0.97, top=0.90, bottom=0.14)

    # Plot with z on x-axis (surface z=0 on left, deeper on right) and tau on y-axis
    ax.plot(z_arr / 1000, tau_arr, 'b-', linewidth=2.5, label='FDM Solution')
    ax.plot([z / 1000 for z in z_ref], tau_ref,
            'ro', markersize=8, label='REST API Reference', zorder=10)

    ax.axvline(x=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.text(0.05, 0.62, r'$z=0$' '\n(surface)', fontsize=8, color='gray',
            ha='left', va='bottom')

    ax.axvline(x=-13.986 / 1000, color='purple', linestyle='--', linewidth=1, alpha=0.5)
    ax.text(-0.07, 0.595, 'Nivlkut', fontsize=8, color='purple', ha='right')

    ax.axhline(y=1.0, color='gray', linestyle=':', linewidth=1, alpha=0.4)

    ax.set_xlabel(r'Height $z$ [Mm]')
    ax.set_ylabel(r'Proper Time Ratio $\tau$')
    ax.set_title(r'Vertical Profile at Centre ($x=0$, $y=0$)')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    # x-axis: surface (z=0) on LEFT, deeper to the right (inverted)
    ax.set_xlim(0.15, -4.5)   # xlim reversed: left=surface, right=deep
    ax.set_ylim(0.52, 1.08)

    plt.savefig('figure2_vertical_profile.pdf')
    plt.savefig('figure2_vertical_profile.png')
    print("  [OK] Saved: figure2_vertical_profile.pdf/png")
    plt.close()


# ---------------------------------------------------------------------------
# Figure 3: Horizontal X-field profile (y-scan at z=0, x=0)
# ---------------------------------------------------------------------------

def figure3_horizontal_profile(solver):
    """Figure 3: X field along the N-S axis at z=0, masked outside pyramid."""
    print("Generating Figure 3: Horizontal Profile...")

    i_c   = np.argmin(np.abs(solver.x - 0))
    k_ref = np.argmin(np.abs(solver.z - 0))
    z_val = solver.z[k_ref]

    diag     = DIAGONAL_0 * (z_val - Z_VERTEX) / (0.0 - Z_VERTEX)
    half_diag = diag / 2.0

    y_vals = solver.y
    X_line = solver.X_field[i_c, :, k_ref].copy().astype(float)

    # Mask points outside the pyramid boundary
    for jj, yy in enumerate(y_vals):
        if not solver.is_inside_pyramid(solver.x[i_c], yy, z_val):
            X_line[jj] = np.nan

    # Analytic API formula for comparison
    y_analytic = np.linspace(-half_diag, half_diag, 1000)
    d_north = np.abs(y_analytic - half_diag)
    d_south = np.abs(y_analytic + half_diag)
    d_ref   = half_diag
    X_from_north = X_MAX - (X_MAX - X_MIN) * np.clip(d_north / d_ref, 0, 1)
    X_from_south = X_MAX - (X_MAX - X_MIN) * np.clip(d_south / d_ref, 0, 1)
    X_analytic   = np.maximum(X_from_north, X_from_south)

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(y_analytic / 1000, X_analytic, 'r--', linewidth=1.8,
            label='API geometricX() (analytic)', alpha=0.8)
    ax.plot(y_vals / 1000, X_line, 'b-', linewidth=2.5, label='FDM Solution')

    ax.axhline(X_MIN, color='gray', linestyle=':', linewidth=1, alpha=0.6)
    ax.axhline(X_MAX, color='gray', linestyle=':', linewidth=1, alpha=0.6)
    ax.text(0, X_MIN - 0.12, r'$X_{\min}=5.825$', ha='center', fontsize=9)
    ax.text(0, X_MAX + 0.12, r'$X_{\max}=9.325$', ha='center', fontsize=9)

    ax.axvline( half_diag / 1000, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax.axvline(-half_diag / 1000, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax.text( half_diag / 1000, 9.5, 'N Pole', ha='center', fontsize=9, color='red')
    ax.text(-half_diag / 1000, 9.5, 'S Pole', ha='center', fontsize=9, color='red')

    ax.plot(0, X_MIN, 'ko', markersize=8, label='Reference point', zorder=10)

    ax.set_xlabel(r'$y$ [Mm] (North $+$, South $-$)')
    ax.set_ylabel(r'$X$ parameter')
    ax.set_title(r'X Field: N-S profile at Reference Height ($z=0$, $x=0$)')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(5.5, 9.9)

    plt.tight_layout()
    plt.savefig('figure3_horizontal_profile.pdf')
    plt.savefig('figure3_horizontal_profile.png')
    print("  [OK] Saved: figure3_horizontal_profile.pdf/png")
    plt.close()


# ---------------------------------------------------------------------------
# Figure 4: Validation error distribution
# ---------------------------------------------------------------------------

def figure4_error_distribution(solver):
    """Figure 4: Validation error distribution (reuses existing solver)."""
    print("Generating Figure 4: Error Distribution...")

    validator = ETCSValidator()
    results   = validator.validate(solver)

    names      = [r['name'] for r in results]
    X_errors   = [r['X_error_pct'] for r in results]
    tau_errors = [r['tau_error_pct'] for r in results]

    fig, ax = plt.subplots(figsize=(6, 5))

    scatter = ax.scatter(X_errors, tau_errors, s=120, alpha=0.8,
                         c='purple', edgecolors='black', linewidth=1, zorder=5)

    for i, name in enumerate(names):
        ax.annotate(name, (X_errors[i], tau_errors[i]),
                    textcoords="offset points", xytext=(6, 4),
                    fontsize=8, ha='left')

    ax.axhline(y=1.0, color='orange', linestyle='--', linewidth=1,
               alpha=0.7, label='1% threshold')
    ax.axvline(x=1.0, color='orange', linestyle='--', linewidth=1, alpha=0.7)

    ax.set_xlabel(r'$X$ Field Error [%]')
    ax.set_ylabel(r'$\tau$ Ratio Error [%]')
    ax.set_title('Validation: FDM vs REST API Reference')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    x_hi = max(max(X_errors) * 1.3, 1.5)
    y_hi = max(max(tau_errors) * 1.3, 1.5)
    ax.set_xlim(-0.1, x_hi)
    ax.set_ylim(-0.05, y_hi)

    mean_X   = np.mean(X_errors)
    mean_tau = np.mean(tau_errors)
    ax.text(0.02, 0.98,
            f'Mean error:\n  X: {mean_X:.3f}%\n  tau: {mean_tau:.3f}%',
            transform=ax.transAxes, fontsize=9,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.6))

    plt.tight_layout()
    plt.savefig('figure4_error_distribution.pdf')
    plt.savefig('figure4_error_distribution.png')
    print("  [OK] Saved: figure4_error_distribution.pdf/png")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("AISU-ETCS Paper Figure Generation")
    print("=" * 60)

    if not HAVE_FDM:
        print("\n[!] FDM solver not found.")
        print("Place etcs_fdm_solver.py and etcs_validation.py")
        print("in the same directory to generate all figures.")
        return

    # Run solver ONCE - shared by all figures
    solver = run_solver()

    figure1_source_term(solver)
    figure2_vertical_profile(solver)
    figure3_horizontal_profile(solver)
    figure4_error_distribution(solver)

    print("\n" + "=" * 60)
    print("[OK] Figure generation complete!")
    print("=" * 60)
    print("\nGenerated files:")
    print("  figure1_attractor_model.pdf/png")
    print("  figure2_vertical_profile.pdf/png")
    print("  figure3_horizontal_profile.pdf/png")
    print("  figure4_error_distribution.pdf/png")


if __name__ == '__main__':
    main()
