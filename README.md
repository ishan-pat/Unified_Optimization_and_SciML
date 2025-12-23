# Unified Optimization and Scientific ML

A modular, composable optimization framework for scientific computing and machine learning that treats optimization algorithms as structured compositions of mathematical primitives.

## Philosophy

Instead of exposing optimizers as opaque functions (e.g. `optimize(f)`), this library exposes the internal structure of optimization itself—objective geometry, update rules, linear solvers, preconditioners, and stopping criteria—as first-class, composable components.

## Key Features

- **Modular Design**: Swap components independently (preconditioners, solvers, stopping criteria)
- **Matrix-Free**: All solvers operate on implicit linear operators
- **JAX-Compatible**: Fully differentiable, GPU/TPU ready
- **Composable**: Build complex algorithms from simple primitives
- **Research-Friendly**: Prototype new algorithms quickly

## Installation

```bash
pip install -e .
```

## Quick Start

```python
import jax.numpy as jnp
from unified_opt import GradientDescent, Objective

# Define objective
def f(x):
    return jnp.sum((x - 1.0) ** 2)

objective = Objective(f)
x0 = jnp.array([0.0, 0.0])

# Create optimizer
optimizer = GradientDescent(learning_rate=0.1)

# Optimize
result = optimizer.optimize(objective, x0, max_iterations=100)
print(f"Solution: {result.x}")
print(f"Converged: {result.converged}")
```

## Installation

```bash
# Clone the repository
cd Unified_Optimization_and_SciML

# Install in development mode
pip install -e .

# Or install dependencies manually
pip install jax jaxlib numpy typing-extensions
```

## Components

### Core Abstractions

- **Objective**: Defines the function being optimized
- **Geometry**: Defines the space and constraints
- **UpdateRule**: Defines how parameters change
- **LinearSolver**: Solves subproblems
- **StoppingRule**: Defines termination

### Supported Algorithms

- Gradient Descent
- SGD
- Adam
- Conjugate Gradient (CG)
- Preconditioned CG (PCG)

### Preconditioners

- Identity
- Jacobi
- Diagonal

### Stopping Criteria

- Gradient norm
- Relative objective decrease
- Max iterations
- Composite logical conditions

## Documentation

See `examples/` for detailed usage examples.

## License

MIT

