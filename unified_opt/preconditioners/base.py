"""Base preconditioner class."""

from typing_extensions import Protocol
import jax.numpy as jnp


class Preconditioner(Protocol):
    """
    A preconditioner M^{-1} for linear systems.
    
    Applies the preconditioner to a vector: z = M^{-1} @ r
    """
    
    def apply(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Apply the preconditioner: M^{-1} @ x.
        
        Args:
            x: Input vector
            
        Returns:
            Preconditioned vector
        """
        ...

