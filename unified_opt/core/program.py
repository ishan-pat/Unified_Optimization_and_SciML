"""
Optimization Program abstraction.

Treats optimization as an executable program, not just a function call.
This enables:
- Replay and analysis
- Differentiation through optimization
- Benchmarking as data
- Research-level introspection
"""

from __future__ import annotations

from typing import Any, Callable
from dataclasses import dataclass, field
import jax.numpy as jnp
from jax import jit, vmap
from unified_opt.core.objective import Objective
from unified_opt.core.geometry import Geometry, EuclideanGeometry
from unified_opt.core.stopping_rule import StoppingRule
from unified_opt.core.state import OptimizationState
from unified_opt.algorithms.operators import AlgorithmOperator, CompositeOperator, Gradient, StepSize
from unified_opt.stopping.max_iterations import MaxIterationsStopping


@dataclass
class OptimizationTrace:
    """
    Complete trace of an optimization run.
    
    Enables replay, analysis, and differentiation.
    """
    
    states: list[OptimizationState] = field(default_factory=list)
    iterations: int = 0
    converged: bool = False
    stopping_reason: str = "unknown"
    
    def final_state(self) -> OptimizationState | None:
        """Get the final state."""
        return self.states[-1] if self.states else None
    
    def trajectory(self) -> jnp.ndarray:
        """Get trajectory of parameter vectors."""
        if not self.states:
            return jnp.array([])
        return jnp.stack([state.x for state in self.states])
    
    def objective_values(self) -> jnp.ndarray:
        """Get trajectory of objective values."""
        if not self.states:
            return jnp.array([])
        values = [s.objective_value for s in self.states if s.objective_value is not None]
        return jnp.array(values)
    
    def gradient_norms(self) -> jnp.ndarray:
        """Get trajectory of gradient norms."""
        if not self.states:
            return jnp.array([])
        norms = [s.gradient_norm for s in self.states if s.gradient_norm is not None]
        return jnp.array(norms)


class OptimizationProgram:
    """
    An optimization program: a complete specification of an optimization problem.
    
    This is the elite abstraction that enables:
    - Replay and analysis
    - Differentiation through optimization
    - Benchmarking as data
    """
    
    def __init__(
        self,
        objective: Objective,
        algorithm: AlgorithmOperator | CompositeOperator | None = None,
        geometry: Geometry | None = None,
        termination: StoppingRule | None = None,
        initial_state: OptimizationState | None = None,
    ):
        """
        Initialize optimization program.
        
        Args:
            objective: Objective function
            algorithm: Algorithm as composition of operators (default: Gradient + StepSize)
            geometry: Geometry of optimization space
            termination: Stopping criterion
            initial_state: Initial state (optional)
        """
        self.objective = objective
        self.algorithm = algorithm or (Gradient() + StepSize(0.01))
        self.geometry = geometry or EuclideanGeometry()
        self.termination = termination
        self.initial_state = initial_state
    
    def execute(
        self,
        x0: jnp.ndarray,
        max_iterations: int = 1000,
        track_state: bool = True,
        jit_compile: bool = False,
    ) -> OptimizationTrace:
        """
        Execute the optimization program.
        
        Args:
            x0: Initial parameter vector
            max_iterations: Maximum iterations
            track_state: Whether to track full state history
            jit_compile: Whether to JIT compile the step function
            
        Returns:
            OptimizationTrace with complete execution history
        """
        # Initialize state
        state = self.initial_state or OptimizationState(x=x0)
        state.objective_value = self.objective.value(x0)
        state.gradient = self.objective.gradient(x0)
        state.gradient_norm = float(self.geometry.norm(state.gradient))
        
        # Initialize algorithm state
        algo_state = {}
        
        # Initialize termination
        termination = self.termination or MaxIterationsStopping(max_iterations)
        
        # Create step function
        def step(x, s, iteration):
            # Apply algorithm operator
            # Note: algorithm.apply returns (direction, new_state)
            direction, s_new = self.algorithm.apply(x, self.objective, s)
            
            # Update parameters
            x_new = x + direction
            
            # Update state
            new_state = OptimizationState(x=x_new, iteration=iteration)
            new_state.update(objective=self.objective)
            new_state.algorithm_state = s_new
            
            return x_new, s_new, new_state
        
        # Optionally JIT compile
        if jit_compile:
            step = jit(step)
        
        # Execution loop
        trace = OptimizationTrace()
        trace.states.append(state)
        
        x = x0
        
        for iteration in range(max_iterations):
            # Execute one step
            x, algo_state, new_state = step(x, algo_state, iteration + 1)
            
            # Track state
            if track_state:
                trace.states.append(new_state)
            
            # Check termination
            should_stop, stop_info = termination.should_stop(
                x, self.objective, iteration,
                {'history': [s.to_dict() for s in trace.states]}
            )
            
            if should_stop:
                trace.converged = True
                trace.stopping_reason = stop_info.get('reason', 'unknown')
                trace.iterations = iteration + 1
                if track_state:
                    trace.states.append(new_state)
                break
        
        if not trace.converged:
            trace.iterations = max_iterations
            trace.stopping_reason = 'max_iterations'
        
        return trace
    
    def differentiate(
        self,
        x0: jnp.ndarray,
        outer_objective: Callable[[jnp.ndarray], float],
        max_iterations: int = 100,
        method: str = "unroll",
    ) -> jnp.ndarray:
        """
        Differentiate through optimization.
        
        Computes: ∇_{x0} outer_objective(optimize(inner_problem, x0))
        
        This is the key feature for bilevel optimization and meta-learning.
        
        Args:
            x0: Initial parameter vector
            outer_objective: Outer objective that depends on optimization result
            max_iterations: Maximum iterations for inner optimization
            method: "unroll" (truncate) or "implicit" (implicit differentiation)
            
        Returns:
            Gradient of outer objective w.r.t. x0
        """
        from jax import grad
        
        if method == "unroll":
            # Unroll optimization and differentiate
            def optimize_and_evaluate(x0_inner):
                trace = self.execute(x0_inner, max_iterations=max_iterations, track_state=False)
                x_final = trace.final_state().x if trace.final_state() else x0_inner
                return outer_objective(x_final)
            
            return grad(optimize_and_evaluate)(x0)
        
        elif method == "implicit":
            # Implicit differentiation via fixed-point
            raise NotImplementedError("Implicit differentiation coming soon")
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def replay(self, trace: OptimizationTrace) -> OptimizationTrace:
        """
        Replay an optimization trace.
        
        Useful for debugging and analysis.
        """
        # For now, just return the trace
        # In future, could validate or modify
        return trace
    
    def benchmark(self, x0: jnp.ndarray, max_iterations: int = 1000) -> dict[str, Any]:
        """
        Benchmark the optimization program.
        
        Returns structured data for analysis.
        """
        trace = self.execute(x0, max_iterations=max_iterations)
        
        return {
            'converged': trace.converged,
            'iterations': trace.iterations,
            'final_value': trace.final_state().objective_value if trace.final_state() else None,
            'final_gradient_norm': trace.final_state().gradient_norm if trace.final_state() else None,
            'trajectory_length': len(trace.states),
            'objective_values': trace.objective_values().tolist() if len(trace.objective_values()) > 0 else [],
            'gradient_norms': trace.gradient_norms().tolist() if len(trace.gradient_norms()) > 0 else [],
        }

