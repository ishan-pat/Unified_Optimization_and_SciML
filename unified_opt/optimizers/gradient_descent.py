"""Gradient Descent optimizer."""

from typing import Any, Dict
import jax.numpy as jnp
from unified_opt.core.objective import Objective
from unified_opt.optimizers.base import BaseOptimizer


class GradientDescent(BaseOptimizer):
    """
    Standard gradient descent optimizer.
    
    Update rule: x_{k+1} = x_k - learning_rate * grad(f)(x_k)
    """
    
    def __init__(self, learning_rate: float = 0.01, **kwargs):
        """
        Initialize gradient descent optimizer.
        
        Args:
            learning_rate: Step size for gradient updates
            **kwargs: Additional arguments passed to BaseOptimizer
        """
        super().__init__(**kwargs)
        self.learning_rate = learning_rate
    
    def _step(
        self,
        x: jnp.ndarray,
        objective: Objective,
        state: Dict[str, Any],
        **kwargs: Any
    ) -> tuple[jnp.ndarray, Dict[str, Any]]:
        """Perform one gradient descent step."""
        grad_x = objective.gradient(x)
        x_new = x - self.learning_rate * grad_x
        return x_new, state

