"""Stopping rule abstraction for optimization termination."""

from __future__ import annotations

from typing_extensions import Protocol
from typing import Any, Dict
import jax.numpy as jnp
from unified_opt.core.objective import Objective


class StoppingRule(Protocol):
    """
    Determines when to stop optimization.
    
    A stopping rule checks convergence criteria and decides whether
    the optimization should continue or terminate.
    """
    
    def should_stop(
        self,
        x: jnp.ndarray,
        objective: Objective,
        iteration: int,
        history: Dict[str, Any] | None = None,
    ) -> tuple[bool, Dict[str, Any]]:
        """
        Determine if optimization should stop.
        
        Args:
            x: Current parameter vector
            objective: The objective function
            iteration: Current iteration number
            history: Optional history of previous iterations
            
        Returns:
            A tuple of (should_stop, info_dict)
            info_dict should contain at least 'reason' if should_stop is True
        """
        ...

