"""Linear solver abstraction for subproblems."""

from __future__ import annotations

from typing_extensions import Protocol
from typing import Callable, Any, Dict
import jax.numpy as jnp


class MatrixFreeOperator:
    """
    A matrix-free linear operator A: R^n -> R^n.
    
    Instead of storing a matrix, we only need a function that computes A @ x.
    This enables efficient handling of large-scale problems.
    """
    
    def __init__(self, apply: Callable[[jnp.ndarray], jnp.ndarray]):
        """
        Initialize a matrix-free operator.
        
        Args:
            apply: A function that computes A @ x for any vector x
        """
        self.apply = apply
    
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Apply the operator to x."""
        return self.apply(x)


class LinearSolver(Protocol):
    """
    Solves linear systems of the form A @ x = b.
    
    The solver operates on matrix-free operators, meaning it only needs
    to be able to compute A @ x, not store the full matrix.
    """
    
    def solve(
        self,
        operator: MatrixFreeOperator | Callable[[jnp.ndarray], jnp.ndarray],
        b: jnp.ndarray,
        x0: jnp.ndarray | None = None,
        **kwargs: Any
    ) -> tuple[jnp.ndarray, Dict[str, Any]]:
        """
        Solve A @ x = b.
        
        Args:
            operator: A matrix-free operator or callable that computes A @ x
            b: Right-hand side vector
            x0: Initial guess (optional)
            **kwargs: Additional solver-specific options
            
        Returns:
            A tuple of (solution, info_dict)
        """
        ...

