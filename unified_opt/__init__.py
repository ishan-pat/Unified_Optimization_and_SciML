"""
Unified Optimization and Scientific ML

A modular, composable optimization framework for scientific computing and machine learning.
"""

from unified_opt.core.objective import Objective
from unified_opt.core.geometry import Geometry, EuclideanGeometry
from unified_opt.core.update_rule import UpdateRule
from unified_opt.core.linear_solver import LinearSolver, MatrixFreeOperator
from unified_opt.core.stopping_rule import StoppingRule

from unified_opt.optimizers.gradient_descent import GradientDescent
from unified_opt.optimizers.sgd import SGD
from unified_opt.optimizers.adam import Adam
from unified_opt.optimizers.cg import ConjugateGradient
from unified_opt.optimizers.pcg import PreconditionedConjugateGradient

from unified_opt.preconditioners.identity import IdentityPreconditioner
from unified_opt.preconditioners.jacobi import JacobiPreconditioner
from unified_opt.preconditioners.diagonal import DiagonalPreconditioner

from unified_opt.stopping.gradient_norm import GradientNormStopping
from unified_opt.stopping.relative_decrease import RelativeDecreaseStopping
from unified_opt.stopping.max_iterations import MaxIterationsStopping
from unified_opt.stopping.composite import CompositeStopping

__version__ = "0.1.0"

__all__ = [
    # Core abstractions
    "Objective",
    "Geometry",
    "EuclideanGeometry",
    "UpdateRule",
    "LinearSolver",
    "MatrixFreeOperator",
    "StoppingRule",
    # Optimizers
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

