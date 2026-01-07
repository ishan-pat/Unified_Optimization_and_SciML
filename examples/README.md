# Research-Grade Examples

These examples demonstrate how this library approaches optimization in research and practice.

## Key Themes

Each example shows a fundamental shift from traditional frameworks:

1. **Optimization as differentiable layers** (not just penalty methods)
2. **Exact constraint satisfaction** (not approximate)
3. **Matrix-free methods** (not explicit matrices)
4. **Bilevel optimization** (not alternating loops)
5. **End-to-end differentiation** (not manual gradients)

---

## Examples

### 1. Physics-Informed Neural Networks (PINNs)
**File:** `physics_informed_neural_networks.py`

**Traditional approach:** Penalty methods with carefully tuned weights
**Our approach:** Optimization as differentiable layer with exact constraint satisfaction

**Why it matters:**
- PINNs require satisfying PDE constraints exactly
- Penalty methods don't guarantee constraint satisfaction
- Our approach ensures physics is satisfied exactly, not approximately

**Research impact:** Enables reliable physics-informed ML where constraint violations are physically meaningless.

---

### 2. Adversarial Robust Training
**File:** `adversarial_robust_training.py`

**Traditional approach:** Alternating inner (attack) / outer (defense) loops
**Our approach:** Bilevel optimization with differentiable inner optimization

**Why it matters:**
- Traditional adversarial training is slow and suboptimal
- Doesn't find true worst-case perturbations
- Our approach computes optimal adversarial examples exactly

**Research impact:** Enables faster, more effective adversarial training with theoretical guarantees.

---

### 3. Optimal Control with PDE Constraints
**File:** `optimal_control_pde_constrained.py`

**Traditional approach:** Adjoint methods or penalty methods
**Our approach:** Differentiate through PDE solver as optimization layer

**Why it matters:**
- Adjoint methods require manual gradient computation
- Penalty methods don't satisfy constraints exactly
- Our approach: gradients flow through PDE solve automatically

**Research impact:** Enables optimal control of complex PDEs without manual adjoint computation.

---

### 4. Meta-Learning (MAML)
**File:** `meta_learning_maml.py`

**Traditional approach:** First-order approximation (ignores second-order terms)
**Our approach:** Exact gradients through inner optimization

**Why it matters:**
- Traditional MAML uses approximations
- Our approach computes exact gradients
- Works with any inner optimization algorithm (CG, L-BFGS, etc.)

**Research impact:** Enables better meta-learning with exact gradients and flexible inner algorithms.

---

### 5. Large-Scale Inverse Problems
**File:** `large_scale_inverse_problem.py`

**Traditional approach:** Form explicit matrices (O(n²) memory, O(n³) time)
**Our approach:** Matrix-free Krylov methods with composable preconditioners

**Why it matters:**
- Explicit matrices: 1M×1M = 1TB memory (impossible!)
- Our approach: O(n) memory, works on consumer hardware
- Same code scales from small to billion-scale problems

**Research impact:** Enables billion-scale inverse problems (MRI, CT, seismic) on standard hardware.

---

## How These Change Traditional Frameworks

### Before (Traditional)
- **Constraints:** Penalty methods (approximate)
- **Adversarial training:** Alternating loops (slow, suboptimal)
- **PDE-constrained:** Adjoint methods (manual, error-prone)
- **Meta-learning:** Approximations (inaccurate)
- **Large-scale:** Explicit matrices (memory-limited)

### After (This Framework)
- **Constraints:** Exact satisfaction via optimization layers
- **Adversarial training:** Bilevel optimization (optimal, fast)
- **PDE-constrained:** Automatic differentiation through PDE solve
- **Meta-learning:** Exact gradients through inner optimization
- **Large-scale:** Matrix-free methods (billion-scale on consumer hardware)

---

## Research Applications

These examples enable new research directions:

1. **Physics-Informed ML:** Exact constraint satisfaction
2. **Adversarial ML:** Optimal adversarial examples
3. **Optimal Control:** Automatic gradient computation
4. **Meta-Learning:** Exact gradients with flexible algorithms
5. **Large-Scale Optimization:** Billion-scale problems

---

## Key Innovations

1. **Optimization as differentiable layer:** Not just a function call
2. **Exact vs approximate:** Constraints satisfied exactly
3. **Matrix-free:** O(n) memory instead of O(n²)
4. **Bilevel optimization:** End-to-end differentiation
5. **Composable:** Swap algorithms without rewriting code


