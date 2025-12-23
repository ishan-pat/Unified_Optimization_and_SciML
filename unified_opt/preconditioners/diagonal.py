"""Diagonal preconditioner."""

import jax.numpy as jnp
from typing import Callable


class DiagonalPreconditioner:
    """
    Diagonal preconditioner M^{-1} = diag(d)^{-1}.
    
    The preconditioner is defined by a diagonal matrix (or vector of diagonal elements).
    """
    
    def __init__(self, diagonal: jnp.ndarray | Callable[[jnp.ndarray], jnp.ndarray]):
        """
        Initialize diagonal preconditioner.
        
        Args:
            diagonal: Either a vector of diagonal elements, or a function
                     that computes the diagonal from a vector (e.g., diagonal of A @ x / x)
        """
        if callable(diagonal):
            self._diagonal_fn = diagonal
            self._diagonal = None
        else:
            self._diagonal = jnp.asarray(diagonal)
            self._diagonal_fn = None
    
    def apply(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Apply diagonal preconditioner: diag(d)^{-1} @ x.
        
        Args:
            x: Input vector
            
        Returns:
            Preconditioned vector
        """
        if self._diagonal_fn is not None:
            # Compute diagonal on the fly
            diag = self._diagonal_fn(x)
        else:
            diag = self._diagonal
        
        # Avoid division by zero
        diag = jnp.where(jnp.abs(diag) > 1e-12, diag, 1.0)
        return x / diag

