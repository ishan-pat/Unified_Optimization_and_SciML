"""Jacobi preconditioner."""

import jax.numpy as jnp
from typing import Callable
from unified_opt.preconditioners.diagonal import DiagonalPreconditioner


class JacobiPreconditioner(DiagonalPreconditioner):
    """
    Jacobi preconditioner.
    
    For a matrix A, the Jacobi preconditioner is M = diag(A).
    This is a special case of the diagonal preconditioner where
    the diagonal comes from the matrix being preconditioned.
    """
    
    def __init__(self, matrix_diagonal: jnp.ndarray | Callable[[], jnp.ndarray]):
        """
        Initialize Jacobi preconditioner.
        
        Args:
            matrix_diagonal: Diagonal elements of the matrix A, or a function
                           that returns them
        """
        super().__init__(matrix_diagonal)

