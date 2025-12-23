"""Adam optimizer example."""

import jax.numpy as jnp
from unified_opt import Objective, Adam, GradientNormStopping, CompositeStopping, MaxIterationsStopping


def main():
    # Define a non-convex objective (Rosenbrock function)
    def rosenbrock(x):
        a, b = 1.0, 100.0
        return (a - x[0]) ** 2 + b * (x[1] - x[0] ** 2) ** 2
    
    # Create objective
    objective = Objective(rosenbrock)
    
    # Initial guess (difficult starting point)
    x0 = jnp.array([-1.0, 1.0])
    
    # Create stopping criteria: gradient norm OR max iterations
    stopping = CompositeStopping(
        rules=[
            GradientNormStopping(threshold=1e-4),
            MaxIterationsStopping(max_iterations=1000)
        ],
        operator='OR'
    )
    
    # Create Adam optimizer
    optimizer = Adam(learning_rate=0.001, stopping_rule=stopping)
    
    # Optimize
    print("Running Adam on Rosenbrock function...")
    result = optimizer.optimize(objective, x0, max_iterations=1000)
    
    print(f"\nSolution: {result.x}")
    print(f"Expected: [1.0, 1.0]")
    print(f"Final value: {result.final_value}")
    print(f"Converged: {result.converged}")
    print(f"Iterations: {result.iterations}")
    print(f"Stopping reason: {result.info['stopping_reason']}")
    
    # Show convergence history
    if result.info.get('history'):
        values = [h['value'] for h in result.info['history']]
        print(f"\nObjective values (first 5): {values[:5]}")
        print(f"Objective values (last 5): {values[-5:]}")


if __name__ == "__main__":
    main()

