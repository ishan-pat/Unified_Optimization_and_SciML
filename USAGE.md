# Usage Guide

## Core Concepts

This library treats optimization as a composition of five core components:

1. **Objective**: The function to minimize
2. **Geometry**: The space and metric (default: Euclidean)
3. **UpdateRule**: How parameters change each iteration
4. **LinearSolver**: For solving subproblems (e.g., in CG methods)
5. **StoppingRule**: When to terminate optimization

## Basic Usage

### Simple Optimization

```python
import jax.numpy as jnp
from unified_opt import Objective, GradientDescent

# Define and wrap objective
def f(x):
    return jnp.sum((x - 1.0) ** 2)

objective = Objective(f)

# Create optimizer
optimizer = GradientDescent(learning_rate=0.1)

# Optimize
result = optimizer.optimize(
    objective, 
    x0=jnp.array([0.0, 0.0]),
    max_iterations=100
)

print(f"Solution: {result.x}")
print(f"Converged: {result.converged}")
print(f"Iterations: {result.iterations}")
```

### Custom Stopping Criteria

```python
from unified_opt import (
    GradientDescent,
    GradientNormStopping,
    RelativeDecreaseStopping,
    CompositeStopping
)

# Combine multiple stopping criteria
stopping = CompositeStopping(
    rules=[
        GradientNormStopping(threshold=1e-4),
        RelativeDecreaseStopping(threshold=1e-6, window=10)
    ],
    operator='OR'  # Stop if ANY condition is met
)

optimizer = GradientDescent(
    learning_rate=0.01,
    stopping_rule=stopping
)
```

### Using Different Optimizers

#### Adam

```python
from unified_opt import Adam

optimizer = Adam(
    learning_rate=0.001,
    beta1=0.9,
    beta2=0.999
)
```

#### Conjugate Gradient (as linear solver)

```python
from unified_opt import ConjugateGradient
from unified_opt.core.linear_solver import MatrixFreeOperator

# Define matrix-free operator
def apply_A(x):
    # Some linear operator A
    return A_times_x(x)

operator = MatrixFreeOperator(apply_A)
b = jnp.array([1.0, 2.0, 3.0])

# Solve A @ x = b
cg = ConjugateGradient(max_iterations=100, tol=1e-6)
x, info = cg.solve(operator, b)

print(f"Solution: {x}")
print(f"Converged: {info['converged']}")
```

#### Preconditioned CG

```python
from unified_opt import PreconditionedConjugateGradient
from unified_opt.preconditioners import DiagonalPreconditioner

# Create preconditioner
preconditioner = DiagonalPreconditioner(diagonal_vector)

# Solve with preconditioning
pcg = PreconditionedConjugateGradient(
    preconditioner=preconditioner,
    max_iterations=100,
    tol=1e-6
)
x, info = pcg.solve(operator, b)
```

## Composing Components

### Swap Preconditioners

```python
from unified_opt import (
    PreconditionedConjugateGradient,
    IdentityPreconditioner,
    JacobiPreconditioner,
    DiagonalPreconditioner
)

# No preconditioning
pcg1 = PreconditionedConjugateGradient(
    preconditioner=IdentityPreconditioner()
)

# Jacobi preconditioner
pcg2 = PreconditionedConjugateGradient(
    preconditioner=JacobiPreconditioner(matrix_diagonal)
)

# Custom diagonal preconditioner
pcg3 = PreconditionedConjugateGradient(
    preconditioner=DiagonalPreconditioner(custom_diagonal)
)
```

### Swap Stopping Rules

```python
from unified_opt import GradientDescent, GradientNormStopping

# Same optimizer, different stopping rules
optimizer1 = GradientDescent(
    learning_rate=0.01,
    stopping_rule=GradientNormStopping(threshold=1e-4)
)

optimizer2 = GradientDescent(
    learning_rate=0.01,
    stopping_rule=GradientNormStopping(threshold=1e-6)  # More strict
)
```

## Matrix-Free Computation

All solvers work with implicit operators:

```python
from unified_opt.core.linear_solver import MatrixFreeOperator
from jax import grad

# Define objective
def objective(x):
    return 0.5 * x.T @ A @ x - b.T @ x

# Matrix-free Hessian operator (for CG on quadratic objectives)
def hessian_operator(x):
    return grad(grad(objective))(x)

operator = MatrixFreeOperator(hessian_operator)

# Solve using matrix-free operator
cg = ConjugateGradient()
x, info = cg.solve(operator, b)
```

## Advanced: Creating Custom Components

### Custom Stopping Rule

```python
from unified_opt.core.stopping_rule import StoppingRule
from unified_opt.core.objective import Objective

class CustomStopping(StoppingRule):
    def should_stop(self, x, objective, iteration, history=None):
        # Your custom logic
        value = objective.value(x)
        should_stop = value < 1e-10
        return should_stop, {'reason': 'custom_criterion'}
```

### Custom Preconditioner

```python
from unified_opt.preconditioners.base import Preconditioner

class CustomPreconditioner:
    def apply(self, x):
        # Apply M^{-1} @ x
        return preconditioned_vector
```

## GPU Support

Since the library uses JAX, GPU support is automatic:

```python
import jax

# Use GPU if available
x0 = jax.numpy.array([0.0, 0.0])

# Operations automatically run on GPU
result = optimizer.optimize(objective, x0)
```

## Examples

See the `examples/` directory for complete working examples:

- `simple_gradient_descent.py`: Basic gradient descent
- `adam_optimizer.py`: Adam on Rosenbrock function
- `conjugate_gradient.py`: CG for linear systems
- `composable_optimization.py`: Demonstrating component swapping

Run examples:

```bash
python examples/simple_gradient_descent.py
```

