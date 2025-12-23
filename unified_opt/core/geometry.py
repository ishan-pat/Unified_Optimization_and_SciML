"""Geometry abstraction for optimization spaces."""

from __future__ import annotations

from typing_extensions import Protocol
import jax.numpy as jnp
from abc import ABC, abstractmethod


class Geometry(Protocol):
    """
    Defines the geometry of the optimization space.
    
    Geometry determines how we measure distances, project onto constraints,
    and compute inner products.
    """
    
    def inner_product(self, x: jnp.ndarray, y: jnp.ndarray) -> float | jnp.ndarray:
        """Compute the inner product <x, y> in this geometry."""
        ...
    
    def norm(self, x: jnp.ndarray) -> float | jnp.ndarray:
        """Compute the norm ||x|| in this geometry."""
        ...
    
    def project(self, x: jnp.ndarray) -> jnp.ndarray:
        """Project x onto the feasible set (identity for unconstrained)."""
        ...


class EuclideanGeometry:
    """
    Standard Euclidean geometry (L2 inner product).
    
    This is the default geometry for most optimization problems.
    """
    
    def inner_product(self, x: jnp.ndarray, y: jnp.ndarray) -> float | jnp.ndarray:
        """Compute the standard Euclidean inner product."""
        return jnp.dot(x, y)
    
    def norm(self, x: jnp.ndarray) -> float | jnp.ndarray:
        """Compute the L2 norm."""
        return jnp.linalg.norm(x)
    
    def project(self, x: jnp.ndarray) -> jnp.ndarray:
        """Identity projection (no constraints)."""
        return x

