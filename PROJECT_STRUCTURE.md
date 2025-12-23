# Project Structure

This document describes the organization of the Unified Optimization and Scientific ML library.

## Directory Layout

```
Unified_Optimization_and_SciML/
├── unified_opt/              # Main package
│   ├── __init__.py           # Package exports
│   ├── core/                 # Core abstractions
│   │   ├── objective.py      # Objective function abstraction
│   │   ├── geometry.py       # Geometry (space/metric) abstraction
│   │   ├── update_rule.py    # Update rule protocol
│   │   ├── linear_solver.py  # Linear solver protocol
│   │   └── stopping_rule.py  # Stopping criteria protocol
│   ├── optimizers/           # Optimization algorithms
│   │   ├── base.py           # Base optimizer class
│   │   ├── gradient_descent.py
│   │   ├── sgd.py
│   │   ├── adam.py
│   │   ├── cg.py             # Conjugate Gradient
│   │   └── pcg.py            # Preconditioned CG
│   ├── preconditioners/      # Preconditioners for linear solvers
│   │   ├── base.py
│   │   ├── identity.py
│   │   ├── jacobi.py
│   │   └── diagonal.py
│   └── stopping/             # Stopping criteria implementations
│       ├── max_iterations.py
│       ├── gradient_norm.py
│       ├── relative_decrease.py
│       └── composite.py      # Logical combinations
├── examples/                 # Usage examples
│   ├── simple_gradient_descent.py
│   ├── adam_optimizer.py
│   ├── conjugate_gradient.py
│   └── composable_optimization.py
├── pyproject.toml            # Project configuration
├── README.md                 # Main documentation
├── USAGE.md                  # Detailed usage guide
├── test_basic.py             # Basic tests
└── .gitignore
```

## Core Components

### 1. Objective (`core/objective.py`)
- Wraps functions for optimization
- Automatic gradient computation via JAX
- JIT compilation support

### 2. Geometry (`core/geometry.py`)
- Defines optimization space and metric
- Default: EuclideanGeometry (L2 norm)
- Extensible for other geometries (Riemannian, etc.)

### 3. UpdateRule (`core/update_rule.py`)
- Protocol for parameter updates
- Implemented by each optimizer

### 4. LinearSolver (`core/linear_solver.py`)
- Protocol for solving A @ x = b
- Matrix-free operators supported
- Used by CG/PCG methods

### 5. StoppingRule (`core/stopping_rule.py`)
- Protocol for termination criteria
- Composable with logical operators

## Optimizers

All optimizers inherit from `BaseOptimizer` and implement `_step()`:

1. **GradientDescent**: Standard gradient descent
2. **SGD**: Stochastic gradient descent with batch support
3. **Adam**: Adaptive moment estimation
4. **ConjugateGradient**: CG for linear systems and optimization
5. **PreconditionedConjugateGradient**: PCG with pluggable preconditioners

## Preconditioners

1. **IdentityPreconditioner**: No preconditioning (M = I)
2. **DiagonalPreconditioner**: Diagonal matrix preconditioner
3. **JacobiPreconditioner**: Jacobi preconditioner (diag(A))

## Stopping Criteria

1. **MaxIterationsStopping**: Stop after N iterations
2. **GradientNormStopping**: Stop when ||grad|| < threshold
3. **RelativeDecreaseStopping**: Stop when relative change < threshold
4. **CompositeStopping**: Combine rules with AND/OR logic

## Design Principles

1. **Composability**: All components are swappable
2. **Matrix-Free**: No explicit matrices required
3. **JAX-Native**: Automatic differentiation and JIT
4. **Extensible**: Easy to add new algorithms/components
5. **Type-Safe**: Uses protocols for clear interfaces

## Extension Points

To add new functionality:

1. **New Optimizer**: Inherit from `BaseOptimizer`, implement `_step()`
2. **New Preconditioner**: Implement `apply(x)` method
3. **New Stopping Rule**: Implement `StoppingRule` protocol
4. **New Geometry**: Implement `Geometry` protocol

See examples in respective directories for patterns.

