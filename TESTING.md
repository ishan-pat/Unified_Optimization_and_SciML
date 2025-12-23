# Testing Guide

## Prerequisites

First, install the required dependencies:

```bash
# Install dependencies
pip install jax jaxlib numpy typing-extensions

# Optional: Install pytest for structured testing
pip install pytest pytest-cov
```

Or install from the project:

```bash
pip install -e .
```

## Running Tests

### Method 1: Direct Execution (Simplest)

Run the basic test script directly:

```bash
python3 test_basic.py
```

Or:

```bash
python test_basic.py
```

### Method 2: Using pytest

If pytest is installed, you can run:

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest test_basic.py

# Run with coverage report
pytest --cov=unified_opt --cov-report=html
```

### Method 3: Using pytest with discovery

Pytest will automatically discover test files:

```bash
# Run all tests matching test_*.py or *_test.py
pytest test_*.py

# Run with verbose output and show print statements
pytest -v -s
```

## Running Examples as Tests

You can also run the example scripts to verify functionality:

```bash
# Run individual examples
python3 examples/simple_gradient_descent.py
python3 examples/adam_optimizer.py
python3 examples/conjugate_gradient.py
python3 examples/composable_optimization.py

# Run all examples
for file in examples/*.py; do
    if [ -f "$file" ] && [ "$(basename "$file")" != "__init__.py" ]; then
        echo "Running $file..."
        python3 "$file"
        echo ""
    fi
done
```

## Expected Output

When running `test_basic.py`, you should see:

```
Testing basic optimization...
✓ Gradient Descent: converged=True, iterations=...
✓ Adam with stopping criteria: converged=True, iterations=...

✅ All basic tests passed!
```

## Troubleshooting

### Import Errors

If you see `ModuleNotFoundError: No module named 'jax'`:
- Install dependencies: `pip install jax jaxlib numpy typing-extensions`

### JAX Installation Issues

On macOS with Apple Silicon:
```bash
pip install --upgrade pip
pip install "jax[cpu]"
```

For GPU support:
```bash
pip install "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

## Adding New Tests

Create test files with the naming convention `test_*.py`:

```python
# test_my_feature.py
import jax.numpy as jnp
from unified_opt import Objective, GradientDescent

def test_my_feature():
    def f(x):
        return jnp.sum(x ** 2)
    
    objective = Objective(f)
    optimizer = GradientDescent(learning_rate=0.1)
    result = optimizer.optimize(objective, jnp.array([1.0, 1.0]))
    
    assert result.converged
```

