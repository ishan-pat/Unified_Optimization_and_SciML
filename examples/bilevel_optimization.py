"""
Bilevel Optimization Example

This demonstrates differentiable optimization - a key capability for:
- Hyperparameter optimization
- Meta-learning
- Implicit neural layers
- Adversarial training

Problem: Find hyperparameters λ that minimize outer objective,
         where inner variables x* are optimal for inner objective:
         
    min_λ  g(x*(λ), λ)
    s.t.   x*(λ) = argmin_x f(x, λ)
"""

import jax.numpy as jnp
from jax import grad
from unified_opt import Objective
from unified_opt.core.program import OptimizationProgram
from unified_opt.algorithms.operators import Gradient, Momentum, StepSize


def main():
    """Bilevel optimization: hyperparameter tuning."""
    
    print("=" * 60)
    print("Bilevel Optimization Example")
    print("=" * 60)
    print("\nProblem: Find regularization parameter λ that minimizes")
    print("         validation loss, where model weights are optimal")
    print("         for training loss with regularization.\n")
    
    # Inner problem: L2-regularized least squares
    # f(x, λ) = ||Ax - b||² + λ||x||²
    A_train = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    b_train = jnp.array([1.0, 2.0, 3.0])
    
    A_val = jnp.array([[1.0, 1.0]])
    b_val = jnp.array([0.5])
    
    def inner_objective(x, lambda_reg):
        """Inner objective: training loss with regularization."""
        residual = A_train @ x - b_train
        return jnp.sum(residual ** 2) + lambda_reg * jnp.sum(x ** 2)
    
    def outer_objective(x, lambda_reg):
        """Outer objective: validation loss."""
        residual = A_val @ x - b_val
        return jnp.sum(residual ** 2)
    
    # Initial guess for hyperparameter
    lambda_init = 0.1
    x_init = jnp.array([0.0, 0.0])
    
    print(f"Initial regularization: λ = {lambda_init}")
    print(f"Initial model weights: x = {x_init}\n")
    
    # Solve inner problem for given λ
    def solve_inner(lambda_reg):
        """Solve inner optimization for given hyperparameter."""
        def obj(x):
            return inner_objective(x, lambda_reg)
        
        objective = Objective(obj)
        algorithm = Gradient() + Momentum(beta=0.9) + StepSize(0.01)
        program = OptimizationProgram(objective=objective, algorithm=algorithm)
        
        trace = program.execute(x_init, max_iterations=200)
        return trace.final_state().x if trace.final_state() else x_init
    
    # Compute gradient of outer objective w.r.t. λ
    def outer_loss(lambda_reg):
        """Outer loss as function of hyperparameter."""
        x_opt = solve_inner(lambda_reg)
        return outer_objective(x_opt, lambda_reg)
    
    # Differentiate through optimization
    print("Computing gradient via unrolling optimization...")
    grad_fn = grad(outer_loss)
    
    # Optimize hyperparameter
    print("\nOptimizing hyperparameter λ:\n")
    lambda_reg = lambda_init
    lr_lambda = 0.01
    
    for iteration in range(50):
        # Compute gradient
        grad_lambda = grad_fn(lambda_reg)
        
        # Update hyperparameter
        lambda_reg_new = lambda_reg - lr_lambda * grad_lambda
        
        # Compute loss
        loss_val = outer_loss(lambda_reg)
        loss_val_new = outer_loss(lambda_reg_new)
        
        if loss_val_new < loss_val:
            lambda_reg = lambda_reg_new
            lr_lambda *= 1.1  # Increase step size
        else:
            lr_lambda *= 0.5  # Decrease step size
        
        if iteration % 10 == 0:
            x_opt = solve_inner(lambda_reg)
            print(f"  Iter {iteration:3d}: λ = {lambda_reg:.4f}, "
                  f"val_loss = {loss_val:.6f}, x* = [{x_opt[0]:.4f}, {x_opt[1]:.4f}]")
    
    # Final solution
    x_final = solve_inner(lambda_reg)
    final_val_loss = outer_objective(x_final, lambda_reg)
    
    print("\n" + "=" * 60)
    print("Results:")
    print("=" * 60)
    print(f"Optimal regularization: λ* = {lambda_reg:.6f}")
    print(f"Optimal model weights: x* = [{x_final[0]:.6f}, {x_final[1]:.6f}]")
    print(f"Final validation loss: {final_val_loss:.6f}")
    print("\n✓ This demonstrates differentiation through optimization,")
    print("  enabling hyperparameter optimization and meta-learning!")


if __name__ == "__main__":
    main()

