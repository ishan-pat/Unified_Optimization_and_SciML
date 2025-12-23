"""Update rule abstraction for optimization algorithms."""

from typing_extensions import Protocol
from typing import Any, Dict
import jax.numpy as jnp
from unified_opt.core.objective import Objective


class UpdateRule(Protocol):
    """
    Defines how parameters are updated in an optimization step.
    
    An update rule takes the current state, objective information, and
    produces the next parameter value.
    """
    
    def update(
        self,
        x: jnp.ndarray,
        objective: Objective,
        state: Dict[str, Any] | None = None,
    ) -> tuple[jnp.ndarray, Dict[str, Any]]:
        """
        Compute the next parameter value.
        
        Args:
            x: Current parameter vector
            objective: The objective function
            state: Optional state dictionary for the update rule
            
        Returns:
            A tuple of (next_x, new_state)
        """
        ...

