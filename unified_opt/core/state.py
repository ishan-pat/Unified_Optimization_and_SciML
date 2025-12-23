"""
Explicit optimization state and dynamics model.

This module provides a structured representation of optimization state,
enabling analysis of dynamics, Lyapunov functions, and research-level insights.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import jax.numpy as jnp


@dataclass
class OptimizationState:
    """
    Explicit representation of optimization algorithm state.
    
    This enables:
    - Inspection of algorithm dynamics
    - Computation of Lyapunov-like quantities
    - Research-level analysis
    """
    
    # Primary state
    x: jnp.ndarray  # Current parameter vector
    iteration: int = 0
    
    # First-order information
    gradient: jnp.ndarray | None = None
    gradient_norm: float | None = None
    
    # Momentum/velocity (for methods that use it)
    velocity: jnp.ndarray | None = None
    
    # Second-order information
    hessian: jnp.ndarray | None = None
    curvature_estimate: float | None = None  # Approximate condition number
    
    # Objective values
    objective_value: float | jnp.ndarray | None = None
    
    # Energy/Lyapunov quantities
    energy: float | None = None  # For energy-based methods
    lyapunov: float | None = None  # Lyapunov function value
    
    # Algorithm-specific state
    algorithm_state: dict[str, Any] = field(default_factory=dict)
    
    def update(
        self,
        x: jnp.ndarray | None = None,
        objective: Any | None = None,
        **kwargs: Any
    ) -> OptimizationState:
        """
        Update state from new information.
        
        Args:
            x: New parameter vector
            objective: Objective function (to compute gradient/value if needed)
            **kwargs: Additional state updates
        """
        # Update position
        if x is not None:
            self.x = x
        
        self.iteration += 1
        
        # Compute gradient if objective provided
        if objective is not None:
            self.gradient = objective.gradient(self.x)
            self.gradient_norm = float(jnp.linalg.norm(self.gradient))
            self.objective_value = objective.value(self.x)
        
        # Update other fields
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                self.algorithm_state[key] = value
        
        return self
    
    def compute_curvature(self, hessian: jnp.ndarray | None = None) -> float | None:
        """
        Estimate condition number / curvature.
        
        Args:
            hessian: Hessian matrix (if available)
            
        Returns:
            Condition number estimate
        """
        if hessian is not None:
            self.hessian = hessian
            eigenvals = jnp.linalg.eigvalsh(hessian)
            eigenvals = eigenvals[eigenvals > 1e-10]  # Filter near-zero
            if len(eigenvals) > 0:
                self.curvature_estimate = float(eigenvals[-1] / eigenvals[0])
                return self.curvature_estimate
        
        return None
    
    def compute_lyapunov(self, reference: jnp.ndarray | None = None) -> float | None:
        """
        Compute Lyapunov function: V(x) = ||x - x*||² or similar.
        
        Useful for stability analysis.
        
        Args:
            reference: Reference point (typically optimum or initial point)
        """
        if reference is not None:
            self.lyapunov = float(jnp.linalg.norm(self.x - reference) ** 2)
            return self.lyapunov
        
        # Fallback: use gradient norm as proxy
        if self.gradient_norm is not None:
            self.lyapunov = self.gradient_norm ** 2
            return self.lyapunov
        
        return None
    
    def to_dict(self) -> dict[str, Any]:
        """Convert state to dictionary for logging/analysis."""
        return {
            'iteration': self.iteration,
            'objective_value': float(self.objective_value) if self.objective_value is not None else None,
            'gradient_norm': self.gradient_norm,
            'curvature_estimate': self.curvature_estimate,
            'lyapunov': self.lyapunov,
            'velocity_norm': float(jnp.linalg.norm(self.velocity)) if self.velocity is not None else None,
            **self.algorithm_state,
        }

