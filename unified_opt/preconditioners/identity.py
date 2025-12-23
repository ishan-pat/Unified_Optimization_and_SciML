"""Identity preconditioner (no preconditioning)."""

import jax.numpy as jnp


class IdentityPreconditioner:
    """
    Identity preconditioner (M^{-1} = I).
    
    This is equivalent to no preconditioning.
    """
    
    def apply(self, x: jnp.ndarray) -> jnp.ndarray:
        """Apply identity: returns x unchanged."""
        return x

