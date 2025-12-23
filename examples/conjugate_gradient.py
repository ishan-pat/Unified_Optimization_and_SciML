"""Conjugate Gradient example for solving linear systems."""

import jax.numpy as jnp
from unified_opt import ConjugateGradient, PreconditionedConjugateGradient
from unified_opt.core.linear_solver import MatrixFreeOperator
from unified_opt.preconditioners import DiagonalPreconditioner


def main():
    # Create a positive definite matrix
    n = 100
    np = __import__('numpy', fromlist=[''])
    A_dense = np.random.randn(n, n)
    A_dense = A_dense.T @ A_dense + np.eye(n) * 0.1  # Make it positive definite
    
    # Convert to matrix-free operator
    def apply_A(x):
        return jnp.array(A_dense @ np.array(x))
    
    operator = MatrixFreeOperator(apply_A)
    
    # Right-hand side
    x_true = jnp.ones(n)
    b = apply_A(x_true)
    
    # Solve with CG
    print("Solving A @ x = b with Conjugate Gradient...")
    cg = ConjugateGradient(max_iterations=n, tol=1e-6)
    x_cg, info_cg = cg.solve(operator, b, x0=None)
    
    print(f"\nCG Results:")
    print(f"  Iterations: {info_cg['iterations']}")
    print(f"  Converged: {info_cg['converged']}")
    print(f"  Residual norm: {info_cg.get('residual_norm', 'N/A')}")
    print(f"  Error: {jnp.linalg.norm(x_cg - x_true):.2e}")
    
    # Solve with Preconditioned CG using diagonal preconditioner
    print("\nSolving with Preconditioned CG (diagonal preconditioner)...")
    diag_precond = DiagonalPreconditioner(jnp.diag(jnp.array(A_dense)))
    pcg = PreconditionedConjugateGradient(
        preconditioner=diag_precond,
        max_iterations=n,
        tol=1e-6
    )
    x_pcg, info_pcg = pcg.solve(operator, b, x0=None)
    
    print(f"\nPCG Results:")
    print(f"  Iterations: {info_pcg['iterations']}")
    print(f"  Converged: {info_pcg['converged']}")
    print(f"  Residual norm: {info_pcg.get('residual_norm', 'N/A')}")
    print(f"  Error: {jnp.linalg.norm(x_pcg - x_true):.2e}")


if __name__ == "__main__":
    main()

