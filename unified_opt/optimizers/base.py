"""Base optimizer class."""

from typing import Any, Dict, Optional
import jax.numpy as jnp
from unified_opt.core.objective import Objective
from unified_opt.core.geometry import Geometry, EuclideanGeometry
from unified_opt.core.stopping_rule import StoppingRule
from unified_opt.stopping.max_iterations import MaxIterationsStopping


class OptimizerResult:
    """Result of an optimization run."""
    
    def __init__(
        self,
        x: jnp.ndarray,
        converged: bool,
        iterations: int,
        final_value: float | jnp.ndarray,
        info: Dict[str, Any],
    ):
        self.x = x
        self.converged = converged
        self.iterations = iterations
        self.final_value = final_value
        self.info = info


class BaseOptimizer:
    """Base class for optimizers."""
    
    def __init__(
        self,
        geometry: Optional[Geometry] = None,
        stopping_rule: Optional[StoppingRule] = None,
    ):
        """
        Initialize the optimizer.
        
        Args:
            geometry: The geometry of the optimization space
            stopping_rule: Custom stopping rule (default: max iterations)
        """
        self.geometry = geometry or EuclideanGeometry()
        self.stopping_rule = stopping_rule
    
    def optimize(
        self,
        objective: Objective,
        x0: jnp.ndarray,
        max_iterations: int = 1000,
        **kwargs: Any
    ) -> OptimizerResult:
        """
        Run the optimization algorithm.
        
        Args:
            objective: The objective function to minimize
            x0: Initial parameter vector
            max_iterations: Maximum number of iterations
            **kwargs: Additional optimizer-specific arguments
            
        Returns:
            OptimizerResult containing the solution and metadata
        """
        # Use provided stopping rule or default to max iterations
        stopping_rule = self.stopping_rule or MaxIterationsStopping(max_iterations)
        
        x = x0
        state = {}
        history = []
        
        for iteration in range(max_iterations):
            # Perform one optimization step
            x, state = self._step(x, objective, state, **kwargs)
            
            # Update history
            value = objective.value(x)
            grad_norm = self.geometry.norm(objective.gradient(x))
            history.append({
                'iteration': iteration,
                'value': float(value),
                'gradient_norm': float(grad_norm),
                'x': x,
            })
            
            # Check stopping criteria
            should_stop, stop_info = stopping_rule.should_stop(
                x, objective, iteration, {'history': history}
            )
            
            if should_stop:
                return OptimizerResult(
                    x=x,
                    converged=True,
                    iterations=iteration + 1,
                    final_value=value,
                    info={'stopping_reason': stop_info.get('reason', 'unknown'), 'history': history},
                )
        
        # Max iterations reached
        final_value = objective.value(x)
        return OptimizerResult(
            x=x,
            converged=False,
            iterations=max_iterations,
            final_value=final_value,
            info={'stopping_reason': 'max_iterations', 'history': history},
        )
    
    def _step(
        self,
        x: jnp.ndarray,
        objective: Objective,
        state: Dict[str, Any],
        **kwargs: Any
    ) -> tuple[jnp.ndarray, Dict[str, Any]]:
        """
        Perform one optimization step.
        
        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement _step")

