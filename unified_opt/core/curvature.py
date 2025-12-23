"""
Curvature as a first-class concept.

This unifies first- and second-order methods:
- GD = identity curvature
- Newton = exact Hessian
- Quasi-Newton = low-rank curvature
- CG = implicit curvature solve
"""

from __future__ import annotations

from typing import Protocol, Callable
from abc import ABC, abstractmethod
import jax.numpy as jnp
from jax import grad, hessian


class CurvatureModel(Protocol):
    """
    A curvature model approximating or computing Hessian information.
    
    This is the bridge between first- and second-order methods.
    """
    
    def matvec(self, v: jnp.ndarray, x: jnp.ndarray, objective: Any) -> jnp.ndarray:
        """
        Compute H @ v where H approximates the Hessian.
        
        Args:
            v: Vector to multiply
            x: Current point
            objective: Objective function
            
        Returns:
            H @ v (Hessian-vector product)
        """
        ...
    
    def solve(self, b: jnp.ndarray, x: jnp.ndarray, objective: Any) -> jnp.ndarray:
        """
        Solve H @ s = b for step direction.
        
        Args:
            b: Right-hand side (typically -gradient)
            x: Current point
            objective: Objective function
            
        Returns:
            Step direction s
        """
        ...


class IdentityCurvature:
    """
    Identity curvature: H = I.
    
    This is equivalent to gradient descent (no curvature information).
    """
    
    def matvec(self, v: jnp.ndarray, x: jnp.ndarray, objective: Any) -> jnp.ndarray:
        """H @ v = I @ v = v"""
        return v
    
    def solve(self, b: jnp.ndarray, x: jnp.ndarray, objective: Any) -> jnp.ndarray:
        """Solve I @ s = b => s = b"""
        return b


class ExactHessian:
    """
    Exact Hessian: H = ∇²f(x).
    
    This is Newton's method (exact curvature).
    """
    
    def __init__(self):
        """Initialize exact Hessian model."""
        self._hessian_fn_cache = None
    
    def matvec(self, v: jnp.ndarray, x: jnp.ndarray, objective: Any) -> jnp.ndarray:
        """Compute ∇²f(x) @ v using forward-over-reverse autodiff."""
        # Use HVP (Hessian-vector product) trick
        grad_fn = grad(objective.value)
        
        def hvp(v):
            return grad(lambda x: jnp.dot(grad_fn(x), v))(x)
        
        return hvp(v)
    
    def solve(self, b: jnp.ndarray, x: jnp.ndarray, objective: Any) -> jnp.ndarray:
        """Solve ∇²f(x) @ s = b."""
        # Compute full Hessian (expensive but exact)
        H = hessian(objective.value)(x)
        # Add regularization for numerical stability
        H_reg = H + 1e-8 * jnp.eye(len(x))
        return jnp.linalg.solve(H_reg, b)


class DiagonalCurvature:
    """
    Diagonal Hessian approximation: H ≈ diag(∇²f(x)).
    
    This is a cheap approximation used in many adaptive methods.
    """
    
    def __init__(self, epsilon: float = 1e-8):
        """
        Initialize diagonal curvature model.
        
        Args:
            epsilon: Regularization for diagonal elements
        """
        self.epsilon = epsilon
    
    def matvec(self, v: jnp.ndarray, x: jnp.ndarray, objective: Any) -> jnp.ndarray:
        """Compute diag(H) @ v (element-wise multiplication)."""
        # Approximate diagonal via finite differences or autodiff
        diag = self._get_diagonal(x, objective)
        return diag * v
    
    def solve(self, b: jnp.ndarray, x: jnp.ndarray, objective: Any) -> jnp.ndarray:
        """Solve diag(H) @ s = b => s = b / diag(H)."""
        diag = self._get_diagonal(x, objective)
        diag = jnp.maximum(diag, self.epsilon)  # Regularize
        return b / diag
    
    def _get_diagonal(self, x: jnp.ndarray, objective: Any) -> jnp.ndarray:
        """Compute diagonal of Hessian."""
        # Use diagonal Hessian computation
        grad_fn = grad(objective.value)
        eps = 1e-5
        
        # Finite difference approximation of diagonal
        diag = jnp.zeros_like(x)
        for i in range(len(x)):
            e = jnp.zeros_like(x)
            e = e.at[i].set(1.0)
            grad_plus = grad_fn(x + eps * e)
            grad_minus = grad_fn(x - eps * e)
            diag = diag.at[i].set((grad_plus[i] - grad_minus[i]) / (2 * eps))
        
        return diag


class ImplicitCurvature:
    """
    Implicit curvature via iterative solves (e.g., CG).
    
    For large-scale problems where explicit Hessian is infeasible.
    """
    
    def __init__(self, solver: Callable | None = None, max_iterations: int = 100):
        """
        Initialize implicit curvature model.
        
        Args:
            solver: Linear solver function (default: CG)
            max_iterations: Maximum solver iterations
        """
        self.solver = solver
        self.max_iterations = max_iterations
    
    def matvec(self, v: jnp.ndarray, x: jnp.ndarray, objective: Any) -> jnp.ndarray:
        """Compute H @ v implicitly (via HVP)."""
        grad_fn = grad(objective.value)
        
        def hvp(v_inner):
            return grad(lambda x_inner: jnp.dot(grad_fn(x_inner), v_inner))(x)
        
        return hvp(v)
    
    def solve(self, b: jnp.ndarray, x: jnp.ndarray, objective: Any) -> jnp.ndarray:
        """Solve H @ s = b using iterative solver (e.g., CG)."""
        # Use CG to solve H @ s = b implicitly
        from unified_opt.optimizers.cg import ConjugateGradient
        
        def hessian_matvec(v):
            return self.matvec(v, x, objective)
        
        cg = ConjugateGradient(max_iterations=self.max_iterations)
        operator = type('Operator', (), {'apply': hessian_matvec})()
        s, _ = cg.solve(operator, b)
        
        return s


class LowRankCurvature:
    """
    Low-rank Hessian approximation (L-BFGS style).
    
    Maintains a limited memory approximation of the Hessian.
    """
    
    def __init__(self, memory_size: int = 10):
        """
        Initialize low-rank curvature model.
        
        Args:
            memory_size: Number of correction pairs to store
        """
        self.memory_size = memory_size
        self.s_history = []  # s_k = x_{k+1} - x_k
        self.y_history = []  # y_k = g_{k+1} - g_k
    
    def matvec(self, v: jnp.ndarray, x: jnp.ndarray, objective: Any) -> jnp.ndarray:
        """Compute H @ v using L-BFGS approximation."""
        # L-BFGS matvec (two-loop recursion)
        if len(self.s_history) == 0:
            return v  # Identity if no history
        
        q = v
        alpha = []
        
        # First loop
        for i in range(len(self.s_history) - 1, -1, -1):
            s, y = self.s_history[i], self.y_history[i]
            rho = 1.0 / (jnp.dot(s, y) + 1e-10)
            alpha_i = rho * jnp.dot(s, q)
            alpha.append(alpha_i)
            q = q - alpha_i * y
        
        # Initial approximation (use most recent)
        if len(self.s_history) > 0:
            s, y = self.s_history[-1], self.y_history[-1]
            gamma = jnp.dot(s, y) / (jnp.dot(y, y) + 1e-10)
            r = gamma * q
        else:
            r = q
        
        # Second loop
        for i, (s, y) in enumerate(zip(self.s_history, self.y_history)):
            rho = 1.0 / (jnp.dot(s, y) + 1e-10)
            beta = rho * jnp.dot(y, r)
            r = r + s * (alpha[-1-i] - beta)
        
        return r
    
    def update(self, s: jnp.ndarray, y: jnp.ndarray):
        """
        Update curvature approximation with new (s, y) pair.
        
        Args:
            s: Step: x_{k+1} - x_k
            y: Gradient change: g_{k+1} - g_k
        """
        self.s_history.append(s)
        self.y_history.append(y)
        
        # Limit memory
        if len(self.s_history) > self.memory_size:
            self.s_history.pop(0)
            self.y_history.pop(0)
    
    def solve(self, b: jnp.ndarray, x: jnp.ndarray, objective: Any) -> jnp.ndarray:
        """Solve H @ s = b."""
        return -self.matvec(-b, x, objective)  # Approximate inverse

