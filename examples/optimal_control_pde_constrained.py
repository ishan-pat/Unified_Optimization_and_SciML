"""
Optimal Control with PDE Constraints using Differentiable Optimization

Traditional approach: Use adjoint methods or penalty methods for
PDE-constrained optimization problems.

Our approach: Treat PDE solver as differentiable optimization layer,
enabling end-to-end gradient computation through the PDE solve.

Problem: Optimal control of heat equation
         min_u J(y, u)  s.t.  PDE(y, u) = 0
"""

import jax.numpy as jnp
from jax import grad, vmap
from jax import random

from unified_opt import Objective
from unified_opt.core.program import OptimizationProgram
from unified_opt.algorithms.operators import Gradient, StepSize
from unified_opt.core.curvature import ImplicitCurvature
from unified_opt.optimizers.cg import ConjugateGradient


def main():
    """
    Optimal control of 1D heat equation.
    
    PDE: y_t - y_xx = u(x, t)  (with boundary conditions)
    Objective: min_u ∫(y - y_target)² + λ∫u²
    
    Traditional: Adjoint method or penalty
    Our approach: Differentiate through PDE solver
    """
    
    print("=" * 70)
    print("Optimal Control with PDE Constraints")
    print("=" * 70)
    print("\nDemonstrating differentiable PDE solving for optimal control.\n")
    
    # Discretization
    N_x = 20  # Spatial grid points
    N_t = 10  # Time steps
    dx = 1.0 / N_x
    dt = 0.1
    
    # Target state
    x_grid = jnp.linspace(0, 1, N_x)
    y_target = jnp.sin(jnp.pi * x_grid)  # Desired final state
    
    # Control variable (what we optimize)
    key = random.PRNGKey(42)
    u_init = random.normal(key, (N_t, N_x)) * 0.1
    
    def heat_equation_solver(u_control):
        """
        Solve heat equation: y_t - y_xx = u using implicit Euler.
        
        Traditional: Black-box solver, requires adjoint method
        Our approach: Solve as optimization problem, fully differentiable
        """
        # Initial condition
        y = jnp.zeros(N_x)
        
        # Solve at each time step
        for t in range(N_t):
            u_t = u_control[t]
            
            # Implicit Euler: (y_new - y_old)/dt - y_xx_new = u
            # Rearranged: y_new - dt * y_xx_new = y_old + dt * u
            
            # Define objective: minimize ||residual||²
            # where residual = (y_new - y_old)/dt - y_xx_new - u
            def residual_objective(y_new):
                # Laplacian approximation (finite difference)
                y_xx = (jnp.roll(y_new, 1) - 2*y_new + jnp.roll(y_new, -1)) / (dx**2)
                # Fix boundaries (Dirichlet: y(0) = y(1) = 0)
                y_xx = y_xx.at[0].set(0)
                y_xx = y_xx.at[-1].set(0)
                
                residual = (y_new - y) / dt - y_xx - u_t
                return jnp.sum(residual ** 2)
            
            # Solve using optimization (instead of linear solve)
            obj = Objective(residual_objective)
            algo = Gradient() + StepSize(0.1)
            program = OptimizationProgram(objective=obj, algorithm=algo)
            
            trace = program.execute(y, max_iterations=50)
            y = trace.final_state().x if trace.final_state() else y
            
            # Enforce boundary conditions
            y = y.at[0].set(0)
            y = y.at[-1].set(0)
        
        return y
    
    def objective_function(u_control):
        """
        Objective: J(y, u) = ||y(T) - y_target||² + λ||u||²
        """
        # Solve PDE
        y_final = heat_equation_solver(u_control)
        
        # Tracking term
        tracking = jnp.sum((y_final - y_target) ** 2)
        
        # Regularization
        lambda_reg = 0.01
        regularization = lambda_reg * jnp.sum(u_control ** 2)
        
        return tracking + regularization
    
    print("Solving optimal control problem...")
    print("PDE: Heat equation with distributed control")
    print("Objective: Track target state while minimizing control effort\n")
    
    # Optimize control
    obj = Objective(objective_function)
    algo = Gradient() + Momentum(beta=0.9) + StepSize(0.001)
    program = OptimizationProgram(objective=obj, algorithm=algo)
    
    trace = program.execute(u_init.flatten(), max_iterations=100)
    
    u_opt = trace.final_state().x if trace.final_state() else u_init.flatten()
    u_opt = u_opt.reshape((N_t, N_x))
    
    # Compute final state
    y_final = heat_equation_solver(u_opt)
    error = jnp.sum((y_final - y_target) ** 2)
    
    print(f"Final tracking error: {error:.6f}")
    print(f"Control effort: {jnp.sum(u_opt ** 2):.6f}")
    
    print("\n✓ Key innovation: PDE solver is differentiable optimization layer.")
    print("  Gradients flow through PDE solve automatically.")
    print("  No need for adjoint methods or manual gradient computation.")
    print("  Enables optimal control of complex PDEs end-to-end.")


if __name__ == "__main__":
    main()

