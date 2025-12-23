"""Gradient norm stopping criterion."""

from typing import Any, Dict
import jax.numpy as jnp
from unified_opt.core.objective import Objective
from unified_opt.core.geometry import EuclideanGeometry, Geometry
from unified_opt.core.stopping_rule import StoppingRule


class GradientNormStopping(StoppingRule):
    """
    Stop when gradient norm falls below a threshold.
    """
    
    def __init__(self, threshold: float = 1e-6, geometry: Geometry | None = None):
        """
        Initialize gradient norm stopping rule.
        
        Args:
            threshold: Gradient norm threshold for convergence
            geometry: Geometry for computing norm (default: Euclidean)
        """
        self.threshold = threshold
        self.geometry = geometry or EuclideanGeometry()
    
    def should_stop(
        self,
        x: jnp.ndarray,
        objective: Objective,
        iteration: int,
        history: Dict[str, Any] | None = None,
    ) -> tuple[bool, Dict[str, Any]]:
        """Check if gradient norm is below threshold."""
        grad_x = objective.gradient(x)
        grad_norm = self.geometry.norm(grad_x)
        
        should_stop = grad_norm < self.threshold
        info = {'gradient_norm': float(grad_norm)}
        
        if should_stop:
            info['reason'] = 'gradient_norm_converged'
        
        return should_stop, info

