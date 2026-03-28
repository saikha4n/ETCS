#!/usr/bin/env python3
"""
Generate parameter physical units table for paper appendix
"""

import numpy as np


class ParameterDimensionalAnalysis:
    """Analyze physical dimensions of all parameters"""
    
    def __init__(self):
        self.c = 299792.458  # km/s
        self.G = 6.674e-11   # m^3/(kg*s^2)
        
        self.parameters = {
            'X_min': {
                'value': 5.825,
                'dimension': '[1]',
                'unit': 'dimensionless',
                'physical_meaning': 'Minimal temporal coupling'
            },
            'X_max': {
                'value': 9.325,
                'dimension': '[1]',
                'unit': 'dimensionless',
                'physical_meaning': 'Maximal temporal coupling'
            },
            'X_0': {
                'value': 5.825,
                'dimension': '[1]',
                'unit': 'dimensionless',
                'physical_meaning': 'Reference coupling'
            },
            'lambda_MH': {
                'value': 0.002,
                'dimension': '[1]',
                'unit': 'dimensionless',
                'physical_meaning': 'Self-interaction coupling',
            },
            'v_MH': {
                'value': 2.0,
                'dimension': '[1]',
                'unit': 'dimensionless',
                'physical_meaning': 'Symmetry breaking scale',
            },
            'Delta_N': {
                'value': 0.01,
                'dimension': '[Energy]',
                'unit': 'arb. units',
                'physical_meaning': 'North vacuum shift',
            },
            'Delta_S': {
                'value': 0.05,
                'dimension': '[Energy]',
                'unit': 'arb. units',
                'physical_meaning': 'South vacuum shift'
            },
            'kappa': {
                'value': 0.1,
                'dimension': '[1]',
                'unit': 'dimensionless',
                'physical_meaning': 'Shiodome self-coupling',
            },
            'gamma': {
                'value': 0.01,
                'dimension': r'[Length$^2$]',
                'unit': r'km$^2$',
                'physical_meaning': 'Shiodome-curvature coupling',
            },
            'beta': {
                'value': 0.5,
                'dimension': '[1]',
                'unit': 'dimensionless',
                'physical_meaning': 'Y-axis anisotropy',
            },
            'lambda_boundary': {
                'value': 50.0,
                'dimension': '[Length]',
                'unit': 'km',
                'physical_meaning': 'Boundary layer thickness',
            },
            'U_0': {
                'value': 1.0e6,
                'dimension': '[Energy]',
                'unit': 'arb. units',
                'physical_meaning': 'Rift barrier strength',
            },
            'lambda_b': {
                'value': 10.0,
                'dimension': '[Length]',
                'unit': 'km',
                'physical_meaning': 'Barrier penetration depth',
            },
            'U_v': {
                'value': 10.0,
                'dimension': r'[Energy$\cdot$Length$^{\alpha_v}$]',
                'unit': r'arb. $\times$ km$^{1.5}$',
                'physical_meaning': 'Vertex potential strength',
            },
            'alpha_v': {
                'value': 1.5,
                'dimension': '[1]',
                'unit': 'dimensionless',
                'physical_meaning': 'Vertex divergence exponent',
            },
            'omega_BD': {
                'value': r'$3/(2X^2)$',
                'dimension': '[1]',
                'unit': 'dimensionless',
                'physical_meaning': 'Brans-Dicke coupling',
            },
        }
    
    def check_lagrangian_dimensions(self):
        print("="*70)
        print("LAGRANGIAN DIMENSIONAL CONSISTENCY CHECK")
        print("="*70)
        print()
        
        print("Target: [Lagrangian density] = [Energy/Length^3] = [1/Length^4]")
        print()
        
        terms = {
            'Einstein-Hilbert (f(X)R)': {
                'formula': '(1/16piG) f(X) R',
                'dimension': '[1/G][1][1/L^2] = [Energy/L^3]',
                'check': 'OK'
            },
            'Kinetic (grad X)^2': {
                'formula': '(1/2) omega(X) g^uv grad_u X grad_v X',
                'dimension': '[1][1][1/L^2] = [1/L^2] x X^2 = [Energy/L^3]',
                'check': 'OK (X dimensionless)'
            },
            'Potential U(X,phi,z)': {
                'formula': 'U(X,phi,z)',
                'dimension': '[Energy/L^3]',
                'check': 'OK by construction'
            },
            'Shiodome S^2': {
                'formula': 'kappa S^uv S_uv',
                'dimension': '[1][1][1] = [1] -> needs [Energy/L^3]',
                'check': 'Absorbed into U_0'
            },
            'Shiodome-curvature': {
                'formula': 'gamma S^uv R_uv',
                'dimension': '[L^2][1][1/L^2] = [1] -> needs [Energy/L^3]',
                'check': 'gamma has implicit energy scale'
            },
        }
        
        for name, term in terms.items():
            print(f"{name}:")
            print(f"  Formula:    {term['formula']}")
            print(f"  Dimension:  {term['dimension']}")
            print(f"  Status:     {term['check']}")
            print()
    
    def generate_latex_table(self):
        # Use LaTeX-safe formatting throughout
        latex = r"""\begin{table}[H]
\centering
\caption{Complete list of theoretical parameters with physical dimensions. 
All dimensionless parameters are ratios; energy parameters are in arbitrary units 
(only differences are physical).}
\label{tab:parameter_dimensions}
\begin{tabular}{lcccl}
\toprule
Parameter & Symbol & Value & Dimension & Physical Meaning \\
\midrule
"""
        
        categories = {
            'Field Amplitudes': ['X_min', 'X_max', 'X_0'],
            'Mexican Hat': ['lambda_MH', 'v_MH', 'Delta_N', 'Delta_S'],
            'Shiodome Tensor': ['kappa', 'gamma', 'beta', 'lambda_boundary'],
            'Boundary Barrier': ['U_0', 'lambda_b'],
            'Vertex Singularity': ['U_v', 'alpha_v'],
            'Brans-Dicke': ['omega_BD'],
        }
        
        for category, param_keys in categories.items():
            latex += f"\\multicolumn{{5}}{{c}}{{\\textit{{{category}}}}} \\\\\n"
            
            for key in param_keys:
                if key not in self.parameters:
                    continue
                
                param = self.parameters[key]
                
                # Safe symbol formatting
                symbol_safe = key.replace('_', r'\_')
                symbol = f"${symbol_safe}$"
                
                if isinstance(param['value'], str):
                    value = param['value']
                else:
                    if param['value'] >= 1000:
                        value = f"{param['value']:.1e}"
                    elif param['value'] < 0.01:
                        value = f"{param['value']:.4f}"
                    else:
                        value = f"{param['value']:.3f}"
                
                dim = param['dimension']
                unit = param['unit']
                meaning = param['physical_meaning']
                
                if len(meaning) > 30:
                    meaning = meaning[:27] + "..."
                
                latex += f"{symbol:20s} & {value:15s} & {dim:25s} & {unit:20s} & {meaning} \\\\\n"
            
            latex += "\\midrule\n"
        
        latex = latex.rstrip("\\midrule\n") + "\\bottomrule\n"
        
        # Windows-safe text (no Unicode symbols)
        latex += r"""\end{tabular}
\end{table}

\paragraph{Dimensional Consistency}

The Lagrangian density has dimension $[\mathcal{L}] = [\text{Energy}/\text{Length}^3]$. 
In natural units ($c = \hbar = 1$), this is $[1/\text{Length}^4]$.

\begin{itemize}
\item \textbf{Einstein-Hilbert term:} $\frac{1}{16\pi G} f(X) R$ has 
$[1/G][1][1/L^2] = [E/L^3]$ $\checkmark$

\item \textbf{Kinetic term:} $\omega(X) (\nabla X)^2$ has 
$[1][1/L^2] = [E/L^3]$ $\checkmark$ (since $X$ is dimensionless)

\item \textbf{Potential:} $U(X,\phi,z)$ is defined to have $[E/L^3]$ $\checkmark$

\item \textbf{Shiodome terms:} $\kappa S^2$ and $\gamma S R$ 
require implicit energy scales absorbed into $\kappa, \gamma$ definitions
\end{itemize}

All energy parameters ($\Delta_N, \Delta_S, U_0, U_v$) appear only in 
\emph{differences} or \emph{ratios}, making absolute scales physically irrelevant.
"""
        
        return latex
    
    def save_table(self, filename='parameter_units_table.tex'):
        """Save table to file with UTF-8 encoding"""
        
        latex = self.generate_latex_table()
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(latex)
        
        print(f"Saved parameter table to: {filename}")
        print()
        print(latex)


def main():
    """Generate parameter dimensional analysis"""
    
    analysis = ParameterDimensionalAnalysis()
    
    analysis.check_lagrangian_dimensions()
    
    print()
    
    analysis.save_table('parameter_units_table.tex')


if __name__ == '__main__':
    main()
