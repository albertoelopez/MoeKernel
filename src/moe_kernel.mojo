"""
Mixture of Experts (MOE) kernel implementation in Mojo.

This module implements efficient MOE routing and computation kernels optimized for GPU execution.
Key features:
- Top-k expert routing with learned gating
- Sparse expert computation 
- Load balancing across experts
- Batched processing for efficiency
"""

# Core tensor operations for MOE computations
from tensor import Tensor, TensorSpec, TensorShape
# Parallelization utilities for efficient computation
from algorithm import parallelize
# Math utilities for capacity calculations
from math import ceil, log2
# Memory management for tensor initialization
from memory import memset_zero, Allocator
# Built-in numeric types
from builtin import int32, float32
# Collections for memory pool management
from collections import List

# Type aliases for consistent data types throughout MOE implementation
alias FLOAT_TYPE = DType.float32  # Primary floating point type for weights and activations
alias INT_TYPE = DType.int32      # Integer type for indices and counts

struct MOEMemoryPool:
    """Memory pool for efficient tensor allocation and reuse in MOE operations."""
    var float_buffers: List[Tensor[FLOAT_TYPE]]     # Pool of reusable float tensors
    var int_buffers: List[Tensor[INT_TYPE]]         # Pool of reusable int tensors
    var buffer_sizes: List[Int]                     # Track sizes of pooled buffers
    var max_pool_size: Int                          # Maximum number of buffers to pool
    var total_allocations: Int                      # Statistics: total allocations made
    var cache_hits: Int                             # Statistics: successful buffer reuses
    
    fn __init__(inout self, max_pool_size: Int = 32):
        """Initialize memory pool with specified maximum pool size."""
        self.float_buffers = List[Tensor[FLOAT_TYPE]]()
        self.int_buffers = List[Tensor[INT_TYPE]]()
        self.buffer_sizes = List[Int]()
        self.max_pool_size = max_pool_size
        self.total_allocations = 0
        self.cache_hits = 0
    
    fn get_float_tensor(inout self, shape: TensorShape) raises -> Tensor[FLOAT_TYPE]:
        """Get a float tensor from the pool or allocate a new one."""
        self.total_allocations += 1
        let required_size = shape.num_elements()
        
        # Search for a suitable buffer in the pool
        for i in range(len(self.float_buffers)):
            if self.buffer_sizes[i] >= required_size:
                self.cache_hits += 1
                let buffer = self.float_buffers.pop(i)
                _ = self.buffer_sizes.pop(i)
                # Reshape to required shape and zero out
                let reshaped = buffer.reshape(shape)
                memset_zero(reshaped.data(), reshaped.num_elements())
                return reshaped
        
        # No suitable buffer found, allocate new one
        return Tensor[FLOAT_TYPE](shape)
    
    fn get_int_tensor(inout self, shape: TensorShape) raises -> Tensor[INT_TYPE]:
        """Get an int tensor from the pool or allocate a new one."""
        self.total_allocations += 1
        let required_size = shape.num_elements()
        
        # Search for a suitable buffer in the pool
        for i in range(len(self.int_buffers)):
            if self.buffer_sizes[i] >= required_size:
                self.cache_hits += 1
                let buffer = self.int_buffers.pop(i)
                _ = self.buffer_sizes.pop(i)
                # Reshape to required shape and zero out
                let reshaped = buffer.reshape(shape)
                memset_zero(reshaped.data(), reshaped.num_elements() * sizeof[INT_TYPE]())
                return reshaped
        
        # No suitable buffer found, allocate new one
        return Tensor[INT_TYPE](shape)
    
    fn return_float_tensor(inout self, tensor: Tensor[FLOAT_TYPE]):
        """Return a float tensor to the pool for reuse."""
        if len(self.float_buffers) < self.max_pool_size:
            self.float_buffers.append(tensor)
            self.buffer_sizes.append(tensor.num_elements())
    
    fn return_int_tensor(inout self, tensor: Tensor[INT_TYPE]):
        """Return an int tensor to the pool for reuse.""" 
        if len(self.int_buffers) < self.max_pool_size:
            self.int_buffers.append(tensor)
            self.buffer_sizes.append(tensor.num_elements())
    
    fn get_cache_hit_rate(self) -> Float32:
        """Get the cache hit rate for performance monitoring."""
        if self.total_allocations == 0:
            return 0.0
        return Float32(self.cache_hits) / Float32(self.total_allocations)
    
    fn clear_pool(inout self):
        """Clear all pooled buffers to free memory."""
        self.float_buffers.clear()
        self.int_buffers.clear()
        self.buffer_sizes.clear()
        self.cache_hits = 0
        self.total_allocations = 0

struct MOEConfig:
    """Configuration for Mixture of Experts."""
    var num_experts: Int      # Total number of expert networks available
    var top_k: Int           # Number of experts to activate per token (sparse activation)
    var hidden_dim: Int      # Input/output dimension of the MOE layer
    var expert_dim: Int      # Internal dimension of each expert network
    var capacity_factor: Float32  # Multiplier for expert capacity (handles load imbalance)
    
    fn __init__(inout self, num_experts: Int, top_k: Int, hidden_dim: Int, expert_dim: Int, capacity_factor: Float32 = 1.25):
        """Initialize MOE configuration with validation."""
        # Store core MOE parameters
        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden_dim = hidden_dim
        self.expert_dim = expert_dim
        # Capacity factor > 1.0 allows for load balancing slack
        self.capacity_factor = capacity_factor

fn moe_gating_forward[
    NUM_EXPERTS: Int = -1,
    TOP_K: Int = -1,
    HIDDEN_DIM: Int = -1
](
    input: Tensor[FLOAT_TYPE],
    gate_weights: Tensor[FLOAT_TYPE],
    config: MOEConfig
) raises -> (Tensor[FLOAT_TYPE], Tensor[INT_TYPE], Tensor[FLOAT_TYPE]):
    """
    MOE gating function that routes tokens to top-k experts.
    
    This is the core routing mechanism that decides which experts should process
    each token. It computes gating scores and selects top-k experts for sparse activation.
    
    Args:
        input: Input tokens [batch_size, seq_len, hidden_dim]
        gate_weights: Gating network weights [hidden_dim, num_experts]
        config: MOE configuration
        
    Returns:
        Tuple of (expert_weights, expert_indices, load_balancing_loss)
    """
    # Extract input dimensions for routing calculations
    let batch_size = input.dim(0)    # Number of sequences in batch
    let seq_len = input.dim(1)       # Length of each sequence
    let num_tokens = batch_size * seq_len  # Total tokens to route
    
    # Use compile-time constants when available for optimization
    alias num_experts = NUM_EXPERTS if NUM_EXPERTS > 0 else config.num_experts
    alias top_k = TOP_K if TOP_K > 0 else config.top_k
    alias hidden_dim = HIDDEN_DIM if HIDDEN_DIM > 0 else config.hidden_dim
    
    # Compute gating scores: [num_tokens, num_experts]
    # This determines how well each token matches each expert's specialization
    let reshaped_input = input.reshape(num_tokens, hidden_dim)
    let gate_logits = reshaped_input @ gate_weights  # Linear projection to expert scores
    
    # Apply softmax to get probabilities
    let gate_probs = softmax(gate_logits, dim=1)
    
    # Top-k selection with compile-time optimization
    let expert_weights = Tensor[FLOAT_TYPE](num_tokens, top_k)
    let expert_indices = Tensor[INT_TYPE](num_tokens, top_k)
    
    # Find top-k experts for each token with compile-time loop unrolling
    for token_idx in range(num_tokens):
        let token_probs = gate_probs[token_idx]
        let top_k_results = top_k_selection(token_probs, top_k)
        
        # Unroll top-k loop when k is known at compile time
        @parameter
        if TOP_K > 0:
            @unroll
            for k in range(TOP_K):
                expert_weights[token_idx, k] = top_k_results.values[k]
                expert_indices[token_idx, k] = top_k_results.indices[k]
        else:
            for k in range(top_k):
                expert_weights[token_idx, k] = top_k_results.values[k]
                expert_indices[token_idx, k] = top_k_results.indices[k]
    
    # Compute load balancing loss
    let load_balancing_loss = compute_load_balancing_loss(gate_probs, expert_indices, config)
    
    return expert_weights, expert_indices, load_balancing_loss

fn moe_gating_forward_pooled[
    NUM_EXPERTS: Int = -1,
    TOP_K: Int = -1,
    HIDDEN_DIM: Int = -1
](
    input: Tensor[FLOAT_TYPE],
    gate_weights: Tensor[FLOAT_TYPE],
    config: MOEConfig,
    inout memory_pool: MOEMemoryPool
) raises -> (Tensor[FLOAT_TYPE], Tensor[INT_TYPE], Tensor[FLOAT_TYPE]):
    """
    Memory pool-aware MOE gating function for reduced allocation overhead.
    """
    # Extract input dimensions for routing calculations
    let batch_size = input.dim(0)
    let seq_len = input.dim(1)
    let num_tokens = batch_size * seq_len
    
    # Use compile-time constants when available for optimization
    alias num_experts = NUM_EXPERTS if NUM_EXPERTS > 0 else config.num_experts
    alias top_k = TOP_K if TOP_K > 0 else config.top_k
    alias hidden_dim = HIDDEN_DIM if HIDDEN_DIM > 0 else config.hidden_dim
    
    # Get temporary tensors from memory pool
    let reshaped_input = input.reshape(num_tokens, hidden_dim)
    let gate_logits = memory_pool.get_float_tensor(TensorShape(num_tokens, num_experts))
    
    # Compute gating scores using pooled memory
    # gate_logits = reshaped_input @ gate_weights (simplified for this implementation)
    for i in range(num_tokens):
        for j in range(num_experts):
            var sum = Float32(0.0)
            for k in range(hidden_dim):
                sum += reshaped_input[i, k] * gate_weights[k, j]
            gate_logits[i, j] = sum
    
    # Apply softmax to get probabilities
    let gate_probs = softmax(gate_logits, dim=1)
    
    # Get output tensors from memory pool
    let expert_weights = memory_pool.get_float_tensor(TensorShape(num_tokens, top_k))
    let expert_indices = memory_pool.get_int_tensor(TensorShape(num_tokens, top_k))
    
    # Find top-k experts for each token
    for token_idx in range(num_tokens):
        let token_probs = gate_probs[token_idx]
        let top_k_results = top_k_selection(token_probs, top_k)
        
        for k in range(top_k):
            expert_weights[token_idx, k] = top_k_results.values[k]
            expert_indices[token_idx, k] = top_k_results.indices[k]
    
    # Compute load balancing loss
    let load_balancing_loss = compute_load_balancing_loss(gate_probs, expert_indices, config)
    
    # Return gate_logits to pool for reuse
    memory_pool.return_float_tensor(gate_logits)
    
    return expert_weights, expert_indices, load_balancing_loss

fn moe_expert_computation[
    NUM_EXPERTS: Int = -1,
    TOP_K: Int = -1,
    HIDDEN_DIM: Int = -1,
    EXPERT_DIM: Int = -1
](
    input: Tensor[FLOAT_TYPE],
    expert_weights: Tensor[FLOAT_TYPE], 
    expert_indices: Tensor[INT_TYPE],
    expert_params: List[Tensor[FLOAT_TYPE]],
    config: MOEConfig
) raises -> Tensor[FLOAT_TYPE]:
    """
    Compute outputs from selected experts with sparse routing.
    
    Args:
        input: Input tokens [batch_size, seq_len, hidden_dim]
        expert_weights: Weights for selected experts [num_tokens, top_k]
        expert_indices: Indices of selected experts [num_tokens, top_k]
        expert_params: List of expert parameter tensors
        config: MOE configuration
        
    Returns:
        Combined expert outputs [batch_size, seq_len, hidden_dim]
    """
    let batch_size = input.dim(0)
    let seq_len = input.dim(1)
    let num_tokens = batch_size * seq_len
    
    # Use compile-time constants for optimization
    alias num_experts = NUM_EXPERTS if NUM_EXPERTS > 0 else config.num_experts
    alias top_k = TOP_K if TOP_K > 0 else config.top_k
    alias hidden_dim = HIDDEN_DIM if HIDDEN_DIM > 0 else config.hidden_dim
    alias expert_dim = EXPERT_DIM if EXPERT_DIM > 0 else config.expert_dim
    
    let reshaped_input = input.reshape(num_tokens, hidden_dim)
    let output = Tensor[FLOAT_TYPE](num_tokens, hidden_dim)
    memset_zero(output.data(), output.num_elements())
    
    # Group tokens by expert for batched computation
    let expert_token_groups = group_tokens_by_expert(expert_indices, num_experts)
    
    # Process each expert with compile-time loop optimization
    @parameter
    if NUM_EXPERTS > 0:
        @unroll
        for expert_id in range(NUM_EXPERTS):
            let token_list = expert_token_groups[expert_id]
            if len(token_list) == 0:
                continue
                
            # Batch tokens for this expert
            let expert_input = gather_tokens(reshaped_input, token_list)
            
            # Apply expert transformation with compile-time dims
            let expert_output = apply_expert_ffn[hidden_dim, expert_dim](expert_input, expert_params[expert_id])
            
            # Scatter results back with proper weighting
            scatter_weighted_results(output, expert_output, token_list, expert_weights, expert_indices, expert_id)
    else:
        for expert_id in range(num_experts):
            let token_list = expert_token_groups[expert_id]
            if len(token_list) == 0:
                continue
                
            # Batch tokens for this expert
            let expert_input = gather_tokens(reshaped_input, token_list)
            
            # Apply expert transformation
            let expert_output = apply_expert_ffn(expert_input, expert_params[expert_id], config)
            
            # Scatter results back with proper weighting
            scatter_weighted_results(output, expert_output, token_list, expert_weights, expert_indices, expert_id)
    
    return output.reshape(batch_size, seq_len, config.hidden_dim)

fn apply_expert_ffn[
    HIDDEN_DIM: Int = -1,
    EXPERT_DIM: Int = -1
](
    input: Tensor[FLOAT_TYPE],
    expert_weights: Tensor[FLOAT_TYPE]
) raises -> Tensor[FLOAT_TYPE]:
    """Apply feed-forward network for a single expert with compile-time optimization."""
    let num_tokens = input.dim(0)
    
    # Use compile-time constants when available
    alias hidden_dim = HIDDEN_DIM if HIDDEN_DIM > 0 else input.dim(1)
    alias expert_dim = EXPERT_DIM if EXPERT_DIM > 0 else (expert_weights.num_elements() // (2 * hidden_dim + 2))
    
    # Assuming expert has: input -> hidden -> output structure
    # W1: [hidden_dim, expert_dim], W2: [expert_dim, hidden_dim]
    let w1_offset = 0
    let w2_offset = hidden_dim * expert_dim
    let b1_offset = w2_offset + expert_dim * hidden_dim
    let b2_offset = b1_offset + expert_dim
    
    let w1 = expert_weights[w1_offset:w1_offset + hidden_dim * expert_dim].reshape(hidden_dim, expert_dim)
    let w2 = expert_weights[w2_offset:w2_offset + expert_dim * hidden_dim].reshape(expert_dim, hidden_dim)
    let b1 = expert_weights[b1_offset:b1_offset + expert_dim].reshape(1, expert_dim)
    let b2 = expert_weights[b2_offset:b2_offset + hidden_dim].reshape(1, hidden_dim)
    
    # Forward pass: input -> W1 -> ReLU -> W2 -> output
    let hidden = (input @ w1) + b1
    let activated = relu(hidden)
    let output = (activated @ w2) + b2
    
    return output

# Keep backward compatibility version
fn apply_expert_ffn(
    input: Tensor[FLOAT_TYPE],
    expert_weights: Tensor[FLOAT_TYPE],
    config: MOEConfig
) raises -> Tensor[FLOAT_TYPE]:
    """Apply feed-forward network for a single expert."""
    let num_tokens = input.dim(0)
    
    # Assuming expert has: input -> hidden -> output structure
    # W1: [hidden_dim, expert_dim], W2: [expert_dim, hidden_dim]
    let w1_offset = 0
    let w2_offset = config.hidden_dim * config.expert_dim
    let b1_offset = w2_offset + config.expert_dim * config.hidden_dim
    let b2_offset = b1_offset + config.expert_dim
    
    let w1 = expert_weights[w1_offset:w1_offset + config.hidden_dim * config.expert_dim].reshape(config.hidden_dim, config.expert_dim)
    let w2 = expert_weights[w2_offset:w2_offset + config.expert_dim * config.hidden_dim].reshape(config.expert_dim, config.hidden_dim)
    let b1 = expert_weights[b1_offset:b1_offset + config.expert_dim].reshape(1, config.expert_dim)
    let b2 = expert_weights[b2_offset:b2_offset + config.hidden_dim].reshape(1, config.hidden_dim)
    
    # Forward pass: input -> W1 -> ReLU -> W2 -> output
    let hidden = (input @ w1) + b1
    let activated = relu(hidden)
    let output = (activated @ w2) + b2
    
    return output

fn compute_load_balancing_loss(
    gate_probs: Tensor[FLOAT_TYPE],
    expert_indices: Tensor[INT_TYPE], 
    config: MOEConfig
) raises -> Tensor[FLOAT_TYPE]:
    """Compute load balancing loss to encourage uniform expert utilization."""
    let num_tokens = gate_probs.dim(0)
    
    # Compute fraction of tokens assigned to each expert
    let expert_counts = Tensor[FLOAT_TYPE](config.num_experts)
    memset_zero(expert_counts.data(), expert_counts.num_elements())
    
    for token_idx in range(num_tokens):
        for k in range(config.top_k):
            let expert_idx = expert_indices[token_idx, k].to[DType.index]()
            expert_counts[expert_idx] += 1.0
    
    # Normalize by total assignments
    let total_assignments = Float32(num_tokens * config.top_k)
    for i in range(config.num_experts):
        expert_counts[i] = expert_counts[i] / total_assignments
    
    # Compute average gate probability for each expert
    let avg_gate_probs = Tensor[FLOAT_TYPE](config.num_experts)
    memset_zero(avg_gate_probs.data(), avg_gate_probs.num_elements())
    
    for expert_idx in range(config.num_experts):
        var sum_prob = Float32(0.0)
        for token_idx in range(num_tokens):
            sum_prob += gate_probs[token_idx, expert_idx]
        avg_gate_probs[expert_idx] = sum_prob / Float32(num_tokens)
    
    # Load balancing loss: sum of products of usage fractions and gate probabilities
    var loss = Float32(0.0)
    for i in range(config.num_experts):
        loss += expert_counts[i] * avg_gate_probs[i]
    
    return Tensor[FLOAT_TYPE](TensorShape(1), loss * Float32(config.num_experts))

# Helper functions with SIMD optimizations
from algorithm import vectorize
from sys.info import simdwidthof
from math import exp, log, max

fn softmax(input: Tensor[FLOAT_TYPE], dim: Int) raises -> Tensor[FLOAT_TYPE]:
    """Apply softmax along specified dimension with SIMD optimization."""
    let result = Tensor[FLOAT_TYPE](input.shape())
    
    if dim == 1:  # Apply softmax along last dimension
        let batch_size = input.dim(0)
        let feature_size = input.dim(1)
        
        for batch_idx in range(batch_size):
            # Find max value for numerical stability
            var max_val = input[batch_idx, 0]
            for i in range(1, feature_size):
                if input[batch_idx, i] > max_val:
                    max_val = input[batch_idx, i]
            
            # Compute exponentials and sum
            var sum_exp = Float32(0.0)
            for i in range(feature_size):
                let val = input[batch_idx, i] - max_val
                let exp_val = exp(val)
                result[batch_idx, i] = exp_val
                sum_exp += exp_val
            
            # Normalize by sum with vectorized division
            @parameter
            fn normalize_simd[simd_width: Int](idx: Int):
                if idx < feature_size:
                    result[batch_idx, idx] = result[batch_idx, idx] / sum_exp
            
            vectorize[normalize_simd, 4](feature_size)
    
    return result

fn top_k_selection(probs: Tensor[FLOAT_TYPE], k: Int) raises -> TopKResult:
    """Select top-k elements and their indices."""
    # Implementation would use efficient top-k selection
    pass

fn group_tokens_by_expert(expert_indices: Tensor[INT_TYPE], num_experts: Int) raises -> List[List[Int]]:
    """Group token indices by their assigned experts."""
    # Implementation would create efficient grouping
    pass

fn gather_tokens(input: Tensor[FLOAT_TYPE], token_indices: List[Int]) raises -> Tensor[FLOAT_TYPE]:
    """Gather tokens at specified indices."""
    # Implementation would efficiently gather tokens
    pass

fn scatter_weighted_results(
    output: Tensor[FLOAT_TYPE],
    expert_output: Tensor[FLOAT_TYPE], 
    token_indices: List[Int],
    expert_weights: Tensor[FLOAT_TYPE],
    expert_indices: Tensor[INT_TYPE],
    expert_id: Int
):
    """Scatter expert results back to output with proper weighting."""
    # Implementation would efficiently scatter with weights
    pass

fn relu(input: Tensor[FLOAT_TYPE]) raises -> Tensor[FLOAT_TYPE]:
    """Apply ReLU activation function with SIMD optimization."""
    let result = Tensor[FLOAT_TYPE](input.shape())
    
    @parameter
    fn vectorized_relu[simd_width: Int](idx: Int):
        if idx < input.num_elements():
            let val = input.load[width=1](idx)
            let relu_val = max(val, Float32(0.0))
            result.store[width=1](idx, relu_val)
    
    vectorize[vectorized_relu, 4](input.num_elements())
    return result

struct TopKResult:
    var values: Tensor[FLOAT_TYPE]
    var indices: Tensor[INT_TYPE]
    
    fn __init__(inout self, values: Tensor[FLOAT_TYPE], indices: Tensor[INT_TYPE]):
        self.values = values
        self.indices = indices