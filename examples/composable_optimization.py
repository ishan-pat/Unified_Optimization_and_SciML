"""Example showing composability of optimization components."""

import jax.numpy as jnp
from unified_opt import (
    Objective,
    GradientDescent,
    GradientNormStopping,
    RelativeDecreaseStopping,
    CompositeStopping,
    MaxIterationsStopping
)


def main():
    """Demonstrate swapping components without rewriting algorithms."""
    
    # Define objective
    def f(x):
        return jnp.sum((x - 2.0) ** 4) + 0.1 * jnp.sum(x ** 2)
    
    objective = Objective(f)
    x0 = jnp.array([0.0, 0.0, 0.0])
    
    print("=== Example 1: Gradient descent with gradient norm stopping ===\n")
    stopping1 = GradientNormStopping(threshold=1e-4)
    optimizer1 = GradientDescent(learning_rate=0.01, stopping_rule=stopping1)
    result1 = optimizer1.optimize(objective, x0, max_iterations=1000)
    print(f"Converged: {result1.converged}, Iterations: {result1.iterations}")
    print(f"Solution: {result1.x}\n")
    
    print("=== Example 2: Same optimizer, different stopping rule ===\n")
    stopping2 = RelativeDecreaseStopping(threshold=1e-6, window=5)
    optimizer2 = GradientDescent(learning_rate=0.01, stopping_rule=stopping2)
    result2 = optimizer2.optimize(objective, x0, max_iterations=1000)
    print(f"Converged: {result2.converged}, Iterations: {result2.iterations}")
    print(f"Solution: {result2.x}\n")
    
    print("=== Example 3: Composite stopping (gradient norm OR relative decrease) ===\n")
    stopping3 = CompositeStopping(
        rules=[
            GradientNormStopping(threshold=1e-4),
            RelativeDecreaseStopping(threshold=1e-6, window=5),
        ],
        operator='OR'
    )
    optimizer3 = GradientDescent(learning_rate=0.01, stopping_rule=stopping3)
    result3 = optimizer3.optimize(objective, x0, max_iterations=1000)
    print(f"Converged: {result3.converged}, Iterations: {result3.iterations}")
    print(f"Solution: {result3.x}\n")
    
    print("=== Key Insight: We swapped stopping criteria without touching optimizer code! ===")


if __name__ == "__main__":
    main()

