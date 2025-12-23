"""Stochastic Gradient Descent (SGD) optimizer."""

from typing import Any, Dict, Callable
import jax.numpy as jnp
from unified_opt.core.objective import Objective
from unified_opt.optimizers.base import BaseOptimizer


class SGD(BaseOptimizer):
    """
    Stochastic Gradient Descent optimizer.
    
    Uses a mini-batch gradient estimate instead of full gradient.
    Update rule: x_{k+1} = x_k - learning_rate * grad_batch(f)(x_k)
    """
    
    def __init__(
        self,
        learning_rate: float = 0.01,
        batch_gradient_fn: Callable[[jnp.ndarray, Any], jnp.ndarray] | None = None,
        **kwargs
    ):
        """
        Initialize SGD optimizer.
        
        Args:
            learning_rate: Step size for gradient updates
            batch_gradient_fn: Function that computes batch gradient.
                             Should take (x, batch_data) and return gradient.
                             If None, uses full gradient (degenerates to GD).
            **kwargs: Additional arguments passed to BaseOptimizer
        """
        super().__init__(**kwargs)
        self.learning_rate = learning_rate
        self.batch_gradient_fn = batch_gradient_fn
    
    def _step(
        self,
        x: jnp.ndarray,
        objective: Objective,
        state: Dict[str, Any],
        batch_data: Any = None,
        **kwargs: Any
    ) -> tuple[jnp.ndarray, Dict[str, Any]]:
        """Perform one SGD step."""
        if self.batch_gradient_fn is not None and batch_data is not None:
            grad_x = self.batch_gradient_fn(x, batch_data)
        else:
            # Fall back to full gradient
            grad_x = objective.gradient(x)
        
        x_new = x - self.learning_rate * grad_x
        return x_new, state

