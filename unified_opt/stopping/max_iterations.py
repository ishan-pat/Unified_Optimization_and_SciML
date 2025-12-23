"""Maximum iterations stopping criterion."""

from typing import Any, Dict
import jax.numpy as jnp
from unified_opt.core.objective import Objective
from unified_opt.core.stopping_rule import StoppingRule


class MaxIterationsStopping(StoppingRule):
    """
    Stop when maximum number of iterations is reached.
    """
    
    def __init__(self, max_iterations: int):
        """
        Initialize max iterations stopping rule.
        
        Args:
            max_iterations: Maximum number of iterations allowed
        """
        self.max_iterations = max_iterations
    
    def should_stop(
        self,
        x: jnp.ndarray,
        objective: Objective,
        iteration: int,
        history: Dict[str, Any] | None = None,
    ) -> tuple[bool, Dict[str, Any]]:
        """Check if maximum iterations reached."""
        should_stop = iteration >= self.max_iterations - 1
        info = {}
        if should_stop:
            info['reason'] = 'max_iterations'
        return should_stop, info

