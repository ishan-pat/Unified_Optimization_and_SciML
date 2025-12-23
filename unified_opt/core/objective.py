"""Objective function abstraction."""

from __future__ import annotations

from typing import Callable, Any
import jax.numpy as jnp
from jax import grad, jit
from typing_extensions import Protocol


class Objective:
    """
    Represents an objective function to be optimized.
    
    The objective encapsulates a function f: R^n -> R and provides
    automatic gradient computation via JAX.
    """
    
    def __init__(self, func: Callable[[jnp.ndarray], float | jnp.ndarray], jit_compile: bool = True):
        """
        Initialize an objective function.
        
        Args:
            func: A callable that takes a JAX array and returns a scalar (or array)
            jit_compile: Whether to JIT compile the function and gradient
        """
        self._func = func
        self._grad_func = grad(func)
        
        if jit_compile:
            self._func = jit(func)
            self._grad_func = jit(self._grad_func)
    
    def __call__(self, x: jnp.ndarray) -> float | jnp.ndarray:
        """Evaluate the objective at x."""
        return self._func(x)
    
    def value(self, x: jnp.ndarray) -> float | jnp.ndarray:
        """Evaluate the objective at x."""
        return self.__call__(x)
    
    def gradient(self, x: jnp.ndarray) -> jnp.ndarray:
        """Compute the gradient at x."""
        return self._grad_func(x)
    
    def value_and_gradient(self, x: jnp.ndarray) -> tuple[float | jnp.ndarray, jnp.ndarray]:
        """Compute both value and gradient at x (more efficient than separate calls)."""
        value = self.value(x)
        grad_val = self.gradient(x)
        return value, grad_val

