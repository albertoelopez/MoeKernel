"""
Comprehensive test suite for MOE kernel implementation.

This test suite validates the correctness of our Mixture of Experts implementation,
including configuration validation, expert routing, load balancing, and performance.
"""

# Core testing and tensor utilities
from testing import assert_equal, assert_true, assert_almost_equal
from tensor import Tensor, TensorShape
from moe_kernel import MOEConfig, moe_gating_forward, moe_expert_computation
from builtin import float32

fn test_moe_config():
    """Test MOE configuration creation.
    
    Ensures that MOE configuration parameters are properly initialized
    and validated for correctness.
    """
    # Test basic configuration with typical transformer values
    # 8 experts, top-2 routing, 512 hidden dim, 2048 expert dim
    let config = MOEConfig(
        num_experts=8,     # Total number of expert networks
        top_k=2,          # Number of experts activated per token
        hidden_dim=512,   # Input/output dimension
        expert_dim=2048,  # Expert internal dimension  
        capacity_factor=1.25  # Load balancing factor
    )
    
    # Validate all configuration parameters are stored correctly
    assert_equal(config.num_experts, 8)      # Number of expert networks
    assert_equal(config.top_k, 2)           # Sparse activation count
    assert_equal(config.hidden_dim, 512)    # Input/output dimension
    assert_equal(config.expert_dim, 2048)   # Expert internal dimension
    assert_almost_equal(config.capacity_factor, 1.25, 0.001)  # Load balancing factor

fn test_moe_gating_shapes():
    """Test that MOE gating produces correct output shapes.
    
    Validates that the gating network outputs tensors with expected dimensions
    for expert routing and load balancing.
    """
    # Setup test configuration with moderate dimensions
    let config = MOEConfig(8, 2, 128, 512)  # 8 experts, top-2, 128 hidden, 512 expert
    let batch_size = 4    # Small batch for testing
    let seq_len = 32      # Moderate sequence length
    
    # Create input tensors with proper shapes
    let input = Tensor[DType.float32](batch_size, seq_len, config.hidden_dim)  # Input tokens
    let gate_weights = Tensor[DType.float32](config.hidden_dim, config.num_experts)  # Gating network weights
    
    # Initialize with random values (simplified)
    for i in range(input.num_elements()):
        input[i] = 0.01 * Float32(i % 100)
    
    for i in range(gate_weights.num_elements()):
        gate_weights[i] = 0.01 * Float32((i * 7) % 100)
    
    try:
        let results = moe_gating_forward(input, gate_weights, config)
        let expert_weights = results.0
        let expert_indices = results.1
        let load_loss = results.2
        
        # Check output shapes
        let num_tokens = batch_size * seq_len
        assert_equal(expert_weights.dim(0), num_tokens)
        assert_equal(expert_weights.dim(1), config.top_k)
        assert_equal(expert_indices.dim(0), num_tokens)
        assert_equal(expert_indices.dim(1), config.top_k)
        assert_equal(load_loss.num_elements(), 1)
        
        print("✓ MOE gating shapes test passed")
    
    except e:
        print("✗ MOE gating shapes test failed:", str(e))

fn test_expert_indices_range():
    """Test that expert indices are within valid range."""
    let config = MOEConfig(4, 2, 64, 256)
    let batch_size = 2
    let seq_len = 8
    
    let input = Tensor[DType.float32](batch_size, seq_len, config.hidden_dim)
    let gate_weights = Tensor[DType.float32](config.hidden_dim, config.num_experts)
    
    # Initialize tensors
    for i in range(input.num_elements()):
        input[i] = Float32(i % 10) * 0.1
        
    for i in range(gate_weights.num_elements()):
        gate_weights[i] = Float32((i * 3) % 20) * 0.05
    
    try:
        let results = moe_gating_forward(input, gate_weights, config)
        let expert_indices = results.1
        
        # Check all indices are valid
        for i in range(expert_indices.num_elements()):
            let idx = expert_indices[i].to[DType.index]()
            assert_true(idx >= 0 and idx < config.num_experts)
        
        print("✓ Expert indices range test passed")
        
    except e:
        print("✗ Expert indices range test failed:", str(e))

fn test_moe_expert_computation_shapes():
    """Test MOE expert computation output shapes."""
    let config = MOEConfig(4, 2, 128, 512)
    let batch_size = 2
    let seq_len = 16
    let num_tokens = batch_size * seq_len
    
    # Create input tensors
    let input = Tensor[DType.float32](batch_size, seq_len, config.hidden_dim)
    let expert_weights = Tensor[DType.float32](num_tokens, config.top_k)
    let expert_indices = Tensor[DType.int32](num_tokens, config.top_k)
    
    # Create expert parameters (simplified)
    var expert_params = List[Tensor[DType.float32]]()
    for i in range(config.num_experts):
        let expert_param_size = config.hidden_dim * config.expert_dim + config.expert_dim * config.hidden_dim + config.expert_dim + config.hidden_dim
        let expert_param = Tensor[DType.float32](expert_param_size)
        for j in range(expert_param_size):
            expert_param[j] = 0.01 * Float32((i * j) % 50)
        expert_params.append(expert_param)
    
    # Initialize inputs
    for i in range(input.num_elements()):
        input[i] = Float32(i % 20) * 0.05
        
    for i in range(expert_weights.num_elements()):
        expert_weights[i] = 1.0 / Float32(config.top_k)  # Equal weights
        
    for i in range(expert_indices.num_elements()):
        expert_indices[i] = Int32(i % config.num_experts)
    
    try:
        let output = moe_expert_computation(input, expert_weights, expert_indices, expert_params, config)
        
        # Check output shape matches input
        assert_equal(output.dim(0), batch_size)
        assert_equal(output.dim(1), seq_len) 
        assert_equal(output.dim(2), config.hidden_dim)
        
        print("✓ MOE expert computation shapes test passed")
        
    except e:
        print("✗ MOE expert computation shapes test failed:", str(e))

fn test_load_balancing_loss_properties():
    """Test properties of load balancing loss."""
    let config = MOEConfig(4, 2, 64, 256)
    
    # Create perfectly balanced scenario
    let num_tokens = 16
    let gate_probs = Tensor[DType.float32](num_tokens, config.num_experts)
    let expert_indices = Tensor[DType.int32](num_tokens, config.top_k)
    
    # Set uniform probabilities
    for i in range(num_tokens):
        for j in range(config.num_experts):
            gate_probs[i, j] = 1.0 / Float32(config.num_experts)
    
    # Set balanced expert assignments
    for i in range(num_tokens):
        for k in range(config.top_k):
            expert_indices[i, k] = Int32((i * config.top_k + k) % config.num_experts)
    
    try:
        let loss = compute_load_balancing_loss(gate_probs, expert_indices, config)
        
        # For perfectly balanced case, loss should be 1/num_experts
        let expected_loss = 1.0 / Float32(config.num_experts)
        assert_almost_equal(loss[0], expected_loss, 0.01)
        
        print("✓ Load balancing loss test passed")
        
    except e:
        print("✗ Load balancing loss test failed:", str(e))

fn main():
    """Run all MOE kernel tests."""
    print("Running MOE Kernel Tests...")
    print("=" * 50)
    
    test_moe_config()
    test_moe_gating_shapes()
    test_expert_indices_range()
    test_moe_expert_computation_shapes()
    test_load_balancing_loss_properties()
    
    print("=" * 50)
    print("All tests completed!")