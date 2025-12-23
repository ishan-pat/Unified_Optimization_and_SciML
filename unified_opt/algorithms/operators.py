"""
Algorithm operators as first-class mathematical objects.

This module provides composable optimization operators that can be combined
using the + operator to build algorithms:

    algo = Gradient() + Momentum(beta=0.9) + StepSize(0.01)

This represents optimization algorithms as mathematical compositions,
not opaque black boxes.
"""

from __future__ import annotations

from typing import Protocol, Any, Callable
from abc import ABC, abstractmethod
import jax.numpy as jnp
from unified_opt.core.objective import Objective


class AlgorithmOperator(Protocol):
    """
    A mathematical operator in an optimization algorithm.
    
    Operators can be composed: op1 + op2 creates a composite operator.
    """
    
    def apply(
        self,
        x: jnp.ndarray,
        objective: Objective,
        state: Any,
    ) -> tuple[jnp.ndarray, Any]:
        """
        Apply the operator to produce a direction or update.
        
        Args:
            x: Current parameter vector
            objective: The objective function
            state: Current algorithm state
            
        Returns:
            Tuple of (direction/update, new_state)
        """
        ...
    
    def __add__(self, other: AlgorithmOperator) -> CompositeOperator:
        """Compose this operator with another."""
        return CompositeOperator([self, other])


class CompositeOperator:
    """
    Composition of multiple algorithm operators.
    
    Applies operators sequentially, with state flowing through.
    """
    
    def __init__(self, operators: list[AlgorithmOperator]):
        """
        Initialize composite operator.
        
        Args:
            operators: List of operators to apply in sequence
        """
        self.operators = operators
    
    def apply(
        self,
        x: jnp.ndarray,
        objective: Objective,
        state: Any,
    ) -> tuple[jnp.ndarray, Any]:
        """Apply all operators in sequence."""
        direction = None
        for op in self.operators:
            if direction is None:
                direction, state = op.apply(x, objective, state)
            else:
                # Subsequent operators transform the direction
                direction, state = op.apply(x, objective, state, direction)
        
        return direction, state
    
    def __add__(self, other: AlgorithmOperator) -> CompositeOperator:
        """Add another operator to the composition."""
        return CompositeOperator(self.operators + [other])
    
    def __repr__(self) -> str:
        return " + ".join(str(op) for op in self.operators)


class Gradient:
    """
    Gradient operator: computes ∇f(x).
    
    This is the fundamental first-order information operator.
    """
    
    def apply(
        self,
        x: jnp.ndarray,
        objective: Objective,
        state: Any,
        direction: jnp.ndarray | None = None,
    ) -> tuple[jnp.ndarray, Any]:
        """
        Compute gradient.
        
        If direction is provided, this transforms it (used in compositions).
        """
        grad_x = objective.gradient(x)
        return grad_x, state
    
    def __add__(self, other: AlgorithmOperator) -> CompositeOperator:
        return CompositeOperator([self, other])
    
    def __repr__(self) -> str:
        return "Gradient()"


class Momentum:
    """
    Momentum operator: applies exponential moving average to direction.
    
    Mathematically: v_{k+1} = β * v_k + (1 - β) * d_k
    """
    
    def __init__(self, beta: float = 0.9):
        """
        Initialize momentum operator.
        
        Args:
            beta: Momentum coefficient (typically 0.9-0.99)
        """
        self.beta = beta
    
    def apply(
        self,
        x: jnp.ndarray,
        objective: Objective,
        state: Any,
        direction: jnp.ndarray | None = None,
    ) -> tuple[jnp.ndarray, Any]:
        """
        Apply momentum to the direction.
        
        Args:
            direction: Input direction (typically from Gradient operator)
        """
        if direction is None:
            raise ValueError("Momentum requires a direction input")
        
        # Initialize velocity if needed
        if state is None:
            state = {}
        if 'velocity' not in state:
            state['velocity'] = jnp.zeros_like(x)
        
        # Update velocity with exponential moving average
        state['velocity'] = self.beta * state['velocity'] + (1 - self.beta) * direction
        
        return state['velocity'], state
    
    def __repr__(self) -> str:
        return f"Momentum(beta={self.beta})"


class StepSize:
    """
    Step size operator: scales direction by learning rate.
    
    Can use fixed step size or adaptive schedules.
    """
    
    def __init__(self, learning_rate: float | Callable[[int], float] = 0.01):
        """
        Initialize step size operator.
        
        Args:
            learning_rate: Either a fixed float or a callable f(iteration) -> float
        """
        self.learning_rate = learning_rate
    
    def apply(
        self,
        x: jnp.ndarray,
        objective: Objective,
        state: Any,
        direction: jnp.ndarray | None = None,
    ) -> tuple[jnp.ndarray, Any]:
        """Scale direction by step size."""
        if direction is None:
            raise ValueError("StepSize requires a direction input")
        
        if state is None:
            state = {}
        
        # Get current iteration
        iteration = state.get('iteration', 0)
        
        # Get step size (adaptive or fixed)
        if callable(self.learning_rate):
            lr = self.learning_rate(iteration)
        else:
            lr = self.learning_rate
        
        scaled_direction = -lr * direction  # Negative for descent
        
        return scaled_direction, state
    
    def __repr__(self) -> str:
        if callable(self.learning_rate):
            return "StepSize(Schedule)"
        return f"StepSize({self.learning_rate})"


class AdaptiveStepSize:
    """
    Adaptive step size using curvature or line search.
    
    More sophisticated than fixed step size.
    """
    
    def __init__(self, method: str = "armijo", alpha: float = 0.1, beta: float = 0.5):
        """
        Initialize adaptive step size.
        
        Args:
            method: "armijo", "goldstein", or "wolfe"
            alpha: Armijo parameter
            beta: Backtracking factor
        """
        self.method = method
        self.alpha = alpha
        self.beta = beta
    
    def apply(
        self,
        x: jnp.ndarray,
        objective: Objective,
        state: Any,
        direction: jnp.ndarray | None = None,
    ) -> tuple[jnp.ndarray, Any]:
        """Compute adaptive step size."""
        if direction is None:
            raise ValueError("AdaptiveStepSize requires a direction input")
        
        # Simple Armijo backtracking
        if self.method == "armijo":
            f_x = objective.value(x)
            grad_x = objective.gradient(x)
            lr = 1.0
            
            # Backtrack until Armijo condition satisfied
            for _ in range(20):
                x_new = x + lr * direction
                f_new = objective.value(x_new)
                
                # Armijo condition: f(x + αd) ≤ f(x) + c₁α∇f(x)ᵀd
                if f_new <= f_x + self.alpha * lr * jnp.dot(grad_x, direction):
                    break
                lr *= self.beta
            
            return -lr * direction, state
        
        raise ValueError(f"Unknown method: {self.method}")
    
    def __repr__(self) -> str:
        return f"AdaptiveStepSize(method={self.method})"


# Note: Gradient, Momentum, StepSize already implement __add__ directly

