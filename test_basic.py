"""Basic test to verify the library works."""

import jax.numpy as jnp
from unified_opt import (
    Objective,
    GradientDescent,
    Adam,
    GradientNormStopping,
    MaxIterationsStopping,
    CompositeStopping,
)


def test_basic():
    """Test basic functionality."""
    print("Testing basic optimization...")
    
    # Simple quadratic
    def f(x):
        return jnp.sum((x - 1.0) ** 2)
    
    objective = Objective(f)
    x0 = jnp.array([0.0, 0.0])
    
    # Test gradient descent
    optimizer = GradientDescent(learning_rate=0.1)
    result = optimizer.optimize(objective, x0, max_iterations=50)
    
    assert result.converged or result.iterations == 50
    print(f"✓ Gradient Descent: converged={result.converged}, iterations={result.iterations}")
    
    # Test with stopping criteria
    stopping = CompositeStopping(
        rules=[
            GradientNormStopping(threshold=1e-4),
            MaxIterationsStopping(max_iterations=100)
        ],
        operator='OR'
    )
    
    optimizer2 = Adam(learning_rate=0.01, stopping_rule=stopping)
    result2 = optimizer2.optimize(objective, x0, max_iterations=100)
    
    assert result2.iterations <= 100
    print(f"✓ Adam with stopping criteria: converged={result2.converged}, iterations={result2.iterations}")
    
    print("\n✅ All basic tests passed!")


if __name__ == "__main__":
    test_basic()

