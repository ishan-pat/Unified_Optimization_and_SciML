"""Simple gradient descent example."""

import jax.numpy as jnp
from unified_opt import Objective, GradientDescent


def main():
    # Define a simple quadratic objective
    def f(x):
        return jnp.sum((x - 1.0) ** 2)
    
    # Create objective
    objective = Objective(f)
    
    # Initial guess
    x0 = jnp.array([0.0, 0.0, 0.0])
    
    # Create optimizer
    optimizer = GradientDescent(learning_rate=0.1)
    
    # Optimize
    print("Running gradient descent...")
    result = optimizer.optimize(objective, x0, max_iterations=100)
    
    print(f"\nSolution: {result.x}")
    print(f"Final value: {result.final_value}")
    print(f"Converged: {result.converged}")
    print(f"Iterations: {result.iterations}")
    print(f"Stopping reason: {result.info['stopping_reason']}")


if __name__ == "__main__":
    main()

