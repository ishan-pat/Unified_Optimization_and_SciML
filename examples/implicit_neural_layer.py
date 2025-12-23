"""
Implicit Neural Layer Example

Demonstrates using optimization as a differentiable layer in a neural network.
This enables:
- Deeper implicit models
- Equilibrium networks
- Optimization-based architectures

The optimization solver becomes part of the computational graph.
"""

import jax.numpy as jnp
from jax import grad, vmap, random
from unified_opt import Objective
from unified_opt.core.program import OptimizationProgram
from unified_opt.algorithms.operators import Gradient, StepSize
from unified_opt.core.curvature import ImplicitCurvature


def main():
    """
    Implicit layer: solve optimization problem as part of forward pass.
    
    Architecture:
        Input -> Feature extractor -> Implicit optimization -> Output
        
    The implicit layer solves: min_z ||Wz - features||² + λ||z||²
    """
    
    print("=" * 60)
    print("Implicit Neural Layer Example")
    print("=" * 60)
    print("\nUsing optimization as a differentiable layer in a neural network.\n")
    
    # Setup
    key = random.PRNGKey(42)
    batch_size = 4
    feature_dim = 10
    latent_dim = 5
    
    # Random weights (would be learned in real training)
    W = random.normal(key, (feature_dim, latent_dim)) * 0.1
    
    # Random input features
    key, subkey = random.split(key)
    features = random.normal(subkey, (batch_size, feature_dim))
    
    def implicit_layer(features_batch, W_matrix, lambda_reg=0.01):
        """
        Implicit layer: solves optimization for each sample in batch.
        
        For each sample, solves: min_z ||Wz - f||² + λ||z||²
        """
        def solve_for_sample(f):
            """Solve optimization for single sample."""
            def objective(z):
                residual = W_matrix @ z - f
                return jnp.sum(residual ** 2) + lambda_reg * jnp.sum(z ** 2)
            
            obj = Objective(objective)
            algo = Gradient() + StepSize(0.1)
            program = OptimizationProgram(objective=obj, algorithm=algo)
            
            z0 = jnp.zeros(latent_dim)
            trace = program.execute(z0, max_iterations=50)
            
            return trace.final_state().x if trace.final_state() else z0
        
        # Solve for all samples in batch
        return vmap(solve_for_sample)(features_batch)
    
    # Forward pass
    print("Forward pass through implicit layer...")
    latent_codes = implicit_layer(features, W)
    
    print(f"\nInput features shape: {features.shape}")
    print(f"Latent codes shape: {latent_codes.shape}")
    print(f"Sample latent code: {latent_codes[0]}\n")
    
    # Define loss (reconstruction)
    def loss_fn(W_param, features_data):
        """Loss function for training."""
        latent = implicit_layer(features_data, W_param)
        reconstructed = W_param @ latent.T
        return jnp.mean((reconstructed.T - features_data) ** 2)
    
    # Gradient computation through implicit layer
    print("Computing gradient through implicit optimization...")
    grad_W = grad(loss_fn)(W, features)
    
    print(f"Gradient shape: {grad_W.shape}")
    print(f"Gradient norm: {jnp.linalg.norm(grad_W):.6f}\n")
    
    print("✓ The optimization solver is fully differentiable!")
    print("  Gradients flow through the implicit layer.")
    print("  This enables end-to-end training of optimization-based architectures.")


if __name__ == "__main__":
    main()

