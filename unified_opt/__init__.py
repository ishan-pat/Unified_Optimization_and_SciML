"""
Unified Optimization and Scientific ML

A modular, composable optimization framework for scientific computing and machine learning.
"""

# Core abstractions
from unified_opt.core.objective import Objective
from unified_opt.core.geometry import Geometry, EuclideanGeometry
from unified_opt.core.update_rule import UpdateRule
from unified_opt.core.linear_solver import LinearSolver, MatrixFreeOperator
from unified_opt.core.stopping_rule import StoppingRule

# Elite features: Algorithms as composable operators
from unified_opt.algorithms.operators import (
    AlgorithmOperator,
    CompositeOperator,
    Gradient,
    Momentum,
    StepSize,
    AdaptiveStepSize,
)

# Elite features: State & dynamics
from unified_opt.core.state import OptimizationState

# Elite features: Curvature as first-class concept
from unified_opt.core.curvature import (
    CurvatureModel,
    IdentityCurvature,
    ExactHessian,
    DiagonalCurvature,
    ImplicitCurvature,
    LowRankCurvature,
)

# Elite features: Optimization programs
from unified_opt.core.program import OptimizationProgram, OptimizationTrace

# Elite features: Differentiable optimization
from unified_opt.differentiable.implicit import (
    implicit_gradient,
    fixed_point_optimization_gradient,
    bilevel_optimization,
)

# Traditional optimizers (backward compatible)
from unified_opt.optimizers.gradient_descent import GradientDescent
from unified_opt.optimizers.sgd import SGD
from unified_opt.optimizers.adam import Adam
from unified_opt.optimizers.cg import ConjugateGradient
from unified_opt.optimizers.pcg import PreconditionedConjugateGradient

# Preconditioners
from unified_opt.preconditioners.identity import IdentityPreconditioner
from unified_opt.preconditioners.jacobi import JacobiPreconditioner
from unified_opt.preconditioners.diagonal import DiagonalPreconditioner

# Stopping criteria
from unified_opt.stopping.gradient_norm import GradientNormStopping
from unified_opt.stopping.relative_decrease import RelativeDecreaseStopping
from unified_opt.stopping.max_iterations import MaxIterationsStopping
from unified_opt.stopping.composite import CompositeStopping

__version__ = "0.2.0"

__all__ = [
    # Core abstractions
    "Objective",
    "Geometry",
    "EuclideanGeometry",
    "UpdateRule",
    "LinearSolver",
    "MatrixFreeOperator",
    "StoppingRule",
    # Elite: Algorithm operators
    "AlgorithmOperator",
    "CompositeOperator",
    "Gradient",
    "Momentum",
    "StepSize",
    "AdaptiveStepSize",
    # Elite: State & dynamics
    "OptimizationState",
    # Elite: Curvature models
    "CurvatureModel",
    "IdentityCurvature",
    "ExactHessian",
    "DiagonalCurvature",
    "ImplicitCurvature",
    "LowRankCurvature",
    # Elite: Optimization programs
    "OptimizationProgram",
    "OptimizationTrace",
    # Elite: Differentiable optimization
    "implicit_gradient",
    "fixed_point_optimization_gradient",
    "bilevel_optimization",
    # Traditional optimizers (backward compatible)
    "GradientDescent",
    "SGD",
    "Adam",
    "ConjugateGradient",
    "PreconditionedConjugateGradient",
    # Preconditioners
    "IdentityPreconditioner",
    "JacobiPreconditioner",
    "DiagonalPreconditioner",
    # Stopping criteria
    "GradientNormStopping",
    "RelativeDecreaseStopping",
    "MaxIterationsStopping",
    "CompositeStopping",
]

