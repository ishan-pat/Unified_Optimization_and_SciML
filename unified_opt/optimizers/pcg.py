"""Preconditioned Conjugate Gradient (PCG) method."""

from typing import Any, Dict, Callable
import jax.numpy as jnp
from unified_opt.core.linear_solver import LinearSolver, MatrixFreeOperator
from unified_opt.preconditioners.identity import IdentityPreconditioner
from unified_opt.optimizers.cg import ConjugateGradient
from unified_opt.core.geometry import Geometry


class PreconditionedConjugateGradient(ConjugateGradient):
    """
    Preconditioned Conjugate Gradient method.
    
    Solves A @ x = b with preconditioner M^{-1}.
    """
    
    def __init__(
        self,
        preconditioner: Any = None,
        max_iterations: int = 1000,
        tol: float = 1e-6,
        **kwargs
    ):
        """
        Initialize PCG solver.
        
        Args:
            preconditioner: A preconditioner with an apply(x) method.
                          Default: IdentityPreconditioner (no preconditioning)
            max_iterations: Maximum PCG iterations
            tol: Convergence tolerance
            **kwargs: Additional arguments passed to ConjugateGradient
        """
        super().__init__(max_iterations=max_iterations, tol=tol, **kwargs)
        self.preconditioner = preconditioner or IdentityPreconditioner()
    
    def solve(
        self,
        operator: MatrixFreeOperator | Callable[[jnp.ndarray], jnp.ndarray],
        b: jnp.ndarray,
        x0: jnp.ndarray | None = None,
        **kwargs: Any
    ) -> tuple[jnp.ndarray, Dict[str, Any]]:
        """
        Solve A @ x = b using preconditioned CG.
        
        Args:
            operator: Matrix-free operator
            b: Right-hand side
            x0: Initial guess
            **kwargs: Additional options
            
        Returns:
            Tuple of (solution, info_dict)
        """
        # Convert operator to callable if needed
        if isinstance(operator, MatrixFreeOperator):
            A = operator.apply
        else:
            A = operator
        
        # Initialize
        if x0 is None:
            x = jnp.zeros_like(b)
        else:
            x = x0
        
        r = b - A(x)
        z = self.preconditioner.apply(r)
        p = z.copy()
        rzold = jnp.dot(r, z)
        
        info = {'iterations': 0, 'converged': False}
        
        for i in range(self.max_iterations):
            Ap = A(p)
            alpha = rzold / (jnp.dot(p, Ap) + 1e-12)
            x = x + alpha * p
            r = r - alpha * Ap
            rznew = jnp.dot(r, self.preconditioner.apply(r))
            
            info['iterations'] = i + 1
            
            # Check convergence
            if jnp.sqrt(jnp.dot(r, r)) < self.tol:
                info['converged'] = True
                info['residual_norm'] = float(jnp.sqrt(jnp.dot(r, r)))
                break
            
            z = self.preconditioner.apply(r)
            p = z + (rznew / rzold) * p
            rzold = rznew
        
        if not info['converged']:
            info['residual_norm'] = float(jnp.sqrt(jnp.dot(r, r)))
        
        return x, info

