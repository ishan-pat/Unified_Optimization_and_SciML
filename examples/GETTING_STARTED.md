# Getting Started with Research Examples

Quick guide for researchers and practitioners wanting to use these examples.

## Prerequisites

```bash
pip install jax jaxlib numpy typing-extensions
pip install -e .
```

## Running Examples

Each example can be run independently:

```bash
# Physics-Informed Neural Networks
python examples/physics_informed_neural_networks.py

# Optimal Control
python examples/optimal_control_pde_constrained.py

# Adversarial Training
python examples/adversarial_robust_training.py

# Meta-Learning
python examples/meta_learning_maml.py

# Large-Scale Inverse Problems
python examples/large_scale_inverse_problem.py
```

## Understanding the Examples

### Structure
Each example follows this pattern:
1. **Problem setup**: Define the optimization problem
2. **Traditional approach**: Show how it's typically done
3. **Our approach**: Demonstrate our framework's advantages
4. **Comparison**: Highlight key differences

### Key Concepts

**Optimization Programs**: Replace `optimizer.optimize()` with `program.execute()`
- Enables differentiation through optimization
- Provides complete execution traces
- Supports replay and analysis

**Composable Algorithms**: Build algorithms from operators
```python
algo = Gradient() + Momentum(0.9) + StepSize(0.01)
```

**Matrix-Free Operators**: For large-scale problems
```python
operator = MatrixFreeOperator(lambda x: A_times_x(x))
```

**Differentiable Optimization**: Gradients flow through solves
```python
grad = program.differentiate(x0, outer_objective)
```

## Extending Examples

### Modify Problem Parameters
Most examples use synthetic data. Replace with real data:
- Medical imaging data for large-scale example
- Real PDE problems for optimal control
- Actual neural networks for adversarial training

### Change Algorithms
Swap optimization algorithms easily:
```python
# Change from GD to CG
algo = Gradient() + StepSize(0.01)  # Before
algo = ConjugateGradient()  # After
```

### Add Preconditioners
For large-scale problems:
```python
precond = DiagonalPreconditioner(diagonal_vector)
pcg = PreconditionedConjugateGradient(preconditioner=precond)
```

## Common Patterns

### Bilevel Optimization
```python
def solve_inner(lambda_param):
    obj = Objective(lambda x: inner_loss(x, lambda_param))
    program = OptimizationProgram(objective=obj)
    return program.execute(x0).x

def outer_loss(lambda_param):
    x_opt = solve_inner(lambda_param)
    return outer_objective(x_opt, lambda_param)

grad_fn = grad(outer_loss)
```

### Constraint Satisfaction
```python
def constraint_layer(params, x):
    obj = Objective(lambda u: constraint_violation(u, params))
    program = OptimizationProgram(objective=obj)
    return program.execute(x).x  # Exact satisfaction!
```

### Matrix-Free Solver
```python
def A_times_x(x):
    # Your matrix-free operator
    return result

operator = MatrixFreeOperator(A_times_x)
cg = ConjugateGradient()
x_opt, info = cg.solve(operator, b)
```

## Tips for Research

1. **Start small**: Run examples with reduced problem sizes first
2. **Understand the trace**: Use `OptimizationTrace` for analysis
3. **Experiment with algorithms**: Try different operator compositions
4. **Monitor convergence**: Check `trace.final_state().gradient_norm`
5. **Scale gradually**: Increase problem size after verifying correctness

## Troubleshooting

**Memory issues**: Use matrix-free operators, reduce problem size
**Slow convergence**: Try different algorithms or preconditioners
**Gradient errors**: Check that objectives are differentiable
**Constraint violations**: Ensure optimization layer is working correctly

## Next Steps

1. Read the main [README.md](README.md) for overview
2. See [COMPARISON.md](COMPARISON.md) for detailed comparisons
3. Check [APPLICATIONS.md](APPLICATIONS.md) for use cases
4. Modify examples for your specific problem
5. Contribute back improvements!

