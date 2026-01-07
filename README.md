# Unified Optimization and Scientific ML

**Optimization as program composition — a unifying framework for first- and second-order methods**

> **This is not another optimizer library.** This is a **new intellectual layer** that treats optimization algorithms as structured compositions of mathematical primitives, enabling research-level introspection, differentiable optimization, and algorithmic innovation.

## Core Intellectual Claim

**Expose the mathematical structure of optimization itself.**

Instead of black-box optimizers (`optimizer.optimize(f)`), we provide:

- **Algorithms as composable operators**: `Gradient() + Momentum(0.9) + StepSize(0.01)`
- **Curvature as first-class concept**: Unifies GD, Newton, Quasi-Newton, CG
- **Explicit state & dynamics**: Inspect Lyapunov functions, energy, convergence
- **Optimization programs**: Executable, replayable, differentiable specifications
- **Differentiable optimization**: Bilevel optimization, meta-learning, implicit layers

This bridges the gap between **how optimization is described in papers** and **how it's implemented in code**.

---

## Capabilities

### 1. Algorithms as First-Class Composable Objects

```python
from unified_opt.algorithms.operators import Gradient, Momentum, StepSize

# Compose algorithms like mathematical expressions
algo = Gradient() + Momentum(beta=0.9) + StepSize(0.01)

# Algorithms are inspectable, transformable objects
# Not opaque black boxes
```

This matches how optimization is described in research papers: as compositions of operators, not monolithic solvers.

### 2. Curvature as Unifying Concept

```python
from unified_opt.core.curvature import IdentityCurvature, ExactHessian, ImplicitCurvature, LowRankCurvature

# All methods unified via curvature model:
# - GD = IdentityCurvature
# - Newton = ExactHessian
# - CG = ImplicitCurvature
# - L-BFGS = LowRankCurvature

# Swap curvature models to change algorithm class
```

This is **PhD-level structure** — curvature is the mathematical bridge between first- and second-order methods.

### 3. Explicit State & Dynamics

```python
from unified_opt.core.state import OptimizationState

# State exposes dynamics for research analysis
state.velocity          # Momentum state
state.curvature_estimate # Condition number
state.lyapunov          # Stability analysis
state.gradient_norm     # Convergence metrics

# Enables Lyapunov-style analysis, energy tracking, etc.
```

Turn your library into a **research instrument**, not just a solver.

### 4. Optimization Programs (Not Runs)

```python
from unified_opt.core.program import OptimizationProgram

# Optimization as executable program
program = OptimizationProgram(
    objective=obj,
    algorithm=Gradient() + StepSize(0.01),
    termination=GradientNormStopping(1e-6)
)

# Execute, replay, analyze
trace = program.execute(x0)
program.replay(trace)
benchmark_data = program.benchmark(x0)

# Enables differentiation through optimization
grad = program.differentiate(x0, outer_objective)
```

Enables **replay**, **analysis**, and **differentiation through optimization**.

### 5. Differentiable Optimization

```python
# Bilevel optimization: optimize hyperparameters
from unified_opt.differentiable.implicit import bilevel_optimization

x_opt, lambda_opt = bilevel_optimization(
    inner_objective=lambda x, λ: loss(x) + λ * regularizer(x),
    outer_objective=lambda x, λ: validation_loss(x),
    x0=x_init,
    hyperparams=lambda_init
)

# Gradients flow through optimization
# Enables meta-learning, hyperparameter tuning, etc.
```

**This is catnip for MIT / Cornell faculty** — enables bilevel optimization, meta-learning, implicit neural layers.

---

## Quick Start

### Installation

```bash
pip install jax jaxlib numpy typing-extensions
pip install -e .
```

### Basic Usage (Traditional API)

```python
import jax.numpy as jnp
from unified_opt import Objective, GradientDescent

def f(x):
    return jnp.sum((x - 1.0) ** 2)

objective = Objective(f)
optimizer = GradientDescent(learning_rate=0.1)
result = optimizer.optimize(objective, jnp.array([0.0, 0.0]))

print(f"Solution: {result.x}")
```

### Advanced Usage (Program Composition)

```python
from unified_opt.core.program import OptimizationProgram
from unified_opt.algorithms.operators import Gradient, Momentum, StepSize

# Compose algorithm
algo = Gradient() + Momentum(0.9) + StepSize(0.01)

# Create program
program = OptimizationProgram(objective=obj, algorithm=algo)

# Execute with full trace
trace = program.execute(x0)

# Differentiate through optimization
grad = program.differentiate(x0, outer_objective_fn)
```

---

## Research Examples

### Bilevel Optimization

```bash
python examples/bilevel_optimization.py
```

Hyperparameter optimization where outer objective depends on inner optimization solution.

### Implicit Neural Layers

```bash
python examples/implicit_neural_layer.py
```

Use optimization as a differentiable layer in neural networks (equilibrium networks).

---

## Architecture

### Core Abstractions

| Component | Responsibility |
|-----------|---------------|
| **Objective** | Function wrapper with automatic gradients |
| **Geometry** | Space and metric (Euclidean, Riemannian, etc.) |
| **AlgorithmOperator** | Composable mathematical operators (Gradient, Momentum, etc.) |
| **CurvatureModel** | Unifies first- and second-order methods |
| **OptimizationProgram** | Executable, differentiable optimization specification |
| **OptimizationState** | Explicit state with dynamics tracking |

### Algorithm Hierarchy

```
OptimizationProgram
  ├── Objective
  ├── Algorithm (composition of operators)
  │   ├── Gradient
  │   ├── Momentum
  │   ├── StepSize
  │   └── CurvatureModel
  ├── Geometry
  └── Termination
```

---

## What Makes This Different

### What Other Libraries Do

- Expose optimizers as opaque functions
- Hide algorithm structure
- No differentiation through optimization
- No state introspection
- Curvature is implicit or missing

### What This Library Does

- **Exposes algorithm structure** as composable operators
- **Unifies methods** via curvature abstraction
- **Enables differentiation** through optimization
- **Exposes dynamics** for research analysis
- **Treats optimization as programs**, not functions

---

## Status

This is a **research-grade framework** designed for:

- **Researchers**: Prototype new algorithms quickly
- **Practitioners**: Deploy mathematically correct solvers
- **Students**: Learn optimization theory with executable code

**Not yet production-hardened** — this is optimized for intellectual clarity and research productivity.

---

## 📖 Documentation

- **[USAGE.md](USAGE.md)**: Detailed usage guide
- **[PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)**: Architecture deep-dive
- **[examples/](examples/)**: Research-grade examples

---

## Positioning

**"A framework for first- and second-order optimization that exposes the mathematical structure of algorithms, enabling research-level introspection and differentiable optimization."**

---

## License

MIT

---

## Citation

If you use this library in research:

```bibtex
@software{unified_optimization_sciml,
  title = {Unified Optimization and Scientific ML},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/ishan-pat/Unified_Optimization_and_SciML}
}
```
