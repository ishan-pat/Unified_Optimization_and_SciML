"""Conjugate Gradient (CG) method for linear systems and optimization."""

from typing import Any, Dict, Callable
import jax.numpy as jnp
from unified_opt.core.objective import Objective
from unified_opt.core.linear_solver import LinearSolver, MatrixFreeOperator
from unified_opt.optimizers.base import BaseOptimizer


class ConjugateGradient(LinearSolver, BaseOptimizer):
    """
    Conjugate Gradient method.
    
    Can be used as:
    1. A linear solver for A @ x = b (positive definite systems)
    2. An optimizer for quadratic objectives
    """
    
    def __init__(
        self,
        max_iterations: int = 1000,
        tol: float = 1e-6,
        **kwargs
    ):
        """
        Initialize CG solver/optimizer.
        
        Args:
            max_iterations: Maximum CG iterations
            tol: Convergence tolerance for residual norm
            **kwargs: Additional arguments passed to BaseOptimizer
        """
        super().__init__(**kwargs)
        self.max_iterations = max_iterations
        self.tol = tol
    
    def solve(
        self,
        operator: MatrixFreeOperator | Callable[[jnp.ndarray], jnp.ndarray],
        b: jnp.ndarray,
        x0: jnp.ndarray | None = None,
        **kwargs: Any
    ) -> tuple[jnp.ndarray, Dict[str, Any]]:
        """
        Solve A @ x = b using CG.
        
        Args:
            operator: Matrix-free operator or callable
            b: Right-hand side
            x0: Initial guess (default: zeros)
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
        p = r.copy()
        rsold = jnp.dot(r, r)
        
        info = {'iterations': 0, 'converged': False}
        
        for i in range(self.max_iterations):
            Ap = A(p)
            alpha = rsold / (jnp.dot(p, Ap) + 1e-12)
            x = x + alpha * p
            r = r - alpha * Ap
            rsnew = jnp.dot(r, r)
            
            info['iterations'] = i + 1
            
            # Check convergence
            if jnp.sqrt(rsnew) < self.tol:
                info['converged'] = True
                info['residual_norm'] = float(jnp.sqrt(rsnew))
                break
            
            p = r + (rsnew / rsold) * p
            rsold = rsnew
        
        if not info['converged']:
            info['residual_norm'] = float(jnp.sqrt(rsold))
        
        return x, info
    
    def _step(
        self,
        x: jnp.ndarray,
        objective: Objective,
        state: Dict[str, Any],
        **kwargs: Any
    ) -> tuple[jnp.ndarray, Dict[str, Any]]:
        """
        CG step for quadratic optimization.
        
        For quadratic objectives f(x) = 0.5 * x^T A x - b^T x,
        CG is equivalent to solving A @ x = b.
        """
        # For general objectives, we use CG as a line search direction finder
        # This is a simplified version - in practice, you'd want more sophisticated CG
        
        if 'p' not in state or 'r' not in state or 'iteration' not in state:
            # Initialize CG state
            grad_x = objective.gradient(x)
            state['r'] = grad_x
            state['p'] = -grad_x
            state['iteration'] = 0
            state['rsold'] = jnp.dot(grad_x, grad_x)
        
        r = state['r']
        p = state['p']
        rsold = state['rsold']
        
        # Approximate Hessian-vector product using finite differences
        # (In practice, you'd use automatic differentiation)
        eps = 1e-5
        grad_x_plus = objective.gradient(x + eps * p)
        Ap = (grad_x_plus - r) / eps
        
        alpha = rsold / (jnp.dot(p, Ap) + 1e-12)
        x_new = x + alpha * p
        
        # Compute new gradient
        r_new = objective.gradient(x_new)
        rsnew = jnp.dot(r_new, r_new)
        
        # Update direction
        p_new = -r_new + (rsnew / rsold) * p
        
        state['r'] = r_new
        state['p'] = p_new
        state['rsold'] = rsnew
        state['iteration'] = state['iteration'] + 1
        
        return x_new, state

