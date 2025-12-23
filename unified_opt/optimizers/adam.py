"""Adam optimizer."""

from typing import Any, Dict
import jax.numpy as jnp
from unified_opt.core.objective import Objective
from unified_opt.optimizers.base import BaseOptimizer


class Adam(BaseOptimizer):
    """
    Adam (Adaptive Moment Estimation) optimizer.
    
    Combines momentum and adaptive learning rates.
    """
    
    def __init__(
        self,
        learning_rate: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        **kwargs
    ):
        """
        Initialize Adam optimizer.
        
        Args:
            learning_rate: Initial learning rate
            beta1: Exponential decay rate for first moment estimates
            beta2: Exponential decay rate for second moment estimates
            epsilon: Small constant for numerical stability
            **kwargs: Additional arguments passed to BaseOptimizer
        """
        super().__init__(**kwargs)
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
    
    def _step(
        self,
        x: jnp.ndarray,
        objective: Objective,
        state: Dict[str, Any],
        **kwargs: Any
    ) -> tuple[jnp.ndarray, Dict[str, Any]]:
        """Perform one Adam step."""
        # Initialize state if needed
        if 'm' not in state:
            state['m'] = jnp.zeros_like(x)
        if 'v' not in state:
            state['v'] = jnp.zeros_like(x)
        if 't' not in state:
            state['t'] = 0
        
        # Increment iteration counter
        state['t'] = state['t'] + 1
        t = state['t']
        
        # Compute gradient
        grad_x = objective.gradient(x)
        
        # Update biased first moment estimate
        m = state['m']
        m = self.beta1 * m + (1 - self.beta1) * grad_x
        
        # Update biased second raw moment estimate
        v = state['v']
        v = self.beta2 * v + (1 - self.beta2) * (grad_x ** 2)
        
        # Compute bias-corrected first moment estimate
        m_hat = m / (1 - self.beta1 ** t)
        
        # Compute bias-corrected second raw moment estimate
        v_hat = v / (1 - self.beta2 ** t)
        
        # Update parameters
        x_new = x - self.learning_rate * m_hat / (jnp.sqrt(v_hat) + self.epsilon)
        
        # Update state
        state['m'] = m
        state['v'] = v
        
        return x_new, state

