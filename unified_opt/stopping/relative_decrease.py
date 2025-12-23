"""Relative objective decrease stopping criterion."""

from typing import Any, Dict
import jax.numpy as jnp
from unified_opt.core.objective import Objective
from unified_opt.core.stopping_rule import StoppingRule


class RelativeDecreaseStopping(StoppingRule):
    """
    Stop when relative decrease in objective is below threshold.
    
    Convergence: |f(x_k) - f(x_{k-n})| / |f(x_{k-n})| < threshold
    """
    
    def __init__(self, threshold: float = 1e-6, window: int = 10):
        """
        Initialize relative decrease stopping rule.
        
        Args:
            threshold: Relative decrease threshold
            window: Number of iterations to look back for comparison
        """
        self.threshold = threshold
        self.window = window
    
    def should_stop(
        self,
        x: jnp.ndarray,
        objective: Objective,
        iteration: int,
        history: Dict[str, Any] | None = None,
    ) -> tuple[bool, Dict[str, Any]]:
        """Check if relative decrease is below threshold."""
        current_value = float(objective.value(x))
        info = {'current_value': current_value}
        
        # Need history to compare
        if history is None or 'history' not in history or len(history['history']) < self.window + 1:
            return False, info
        
        hist = history['history']
        past_value = hist[-self.window]['value']
        
        # Avoid division by zero
        if abs(past_value) < 1e-12:
            relative_decrease = abs(current_value - past_value)
        else:
            relative_decrease = abs(current_value - past_value) / abs(past_value)
        
        should_stop = relative_decrease < self.threshold
        info['relative_decrease'] = relative_decrease
        info['past_value'] = past_value
        
        if should_stop:
            info['reason'] = 'relative_decrease_converged'
        
        return should_stop, info

