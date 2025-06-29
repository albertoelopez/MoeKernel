# MOE Kernel API Reference

This document provides a comprehensive reference for the MOE (Mixture of Experts) kernel implementation.

## Table of Contents

1. [Core Types](#core-types)
2. [Configuration](#configuration)
3. [Main Functions](#main-functions)
4. [Helper Functions](#helper-functions)
5. [Usage Examples](#usage-examples)
6. [Error Handling](#error-handling)

## Core Types

### FLOAT_TYPE
```mojo
alias FLOAT_TYPE = DType.float32
```
Primary floating-point type used throughout the MOE implementation. Uses 32-bit floats for optimal performance and memory usage.

### INT_TYPE
```mojo
alias INT_TYPE = DType.int32
```
Integer type for indices, counts, and configuration parameters.

## Configuration

### MOEConfig
```mojo
struct MOEConfig:
    var num_experts: Int
    var top_k: Int
    var hidden_dim: Int
    var expert_dim: Int
    var capacity_factor: Float32
```

Configuration structure for MOE parameters.

#### Fields

- **`num_experts: Int`** - Total number of expert networks available
  - Valid range: 2 to 128
  - Recommended: 4, 8, 16, 32 for optimal performance

- **`top_k: Int`** - Number of experts activated per token
  - Valid range: 1 to `num_experts`
  - Recommended: 2, 4, 8 (must be ≤ `num_experts`)

- **`hidden_dim: Int`** - Input/output dimension size
  - Valid range: 64 to 4096
  - Must match model's hidden dimension

- **`expert_dim: Int`** - Expert internal dimension (feed-forward size)
  - Valid range: 256 to 16384
  - Typically 4× `hidden_dim` for optimal capacity

- **`capacity_factor: Float32`** - Load balancing parameter
  - Valid range: 1.0 to 2.0
  - Default: 1.25
  - Higher values allow more tokens per expert

#### Constructor
```mojo
fn __init__(inout self, num_experts: Int, top_k: Int, hidden_dim: Int, expert_dim: Int, capacity_factor: Float32 = 1.25)
```

**Parameters:**
- All struct fields as described above
- `capacity_factor` has a default value of 1.25

**Example:**
```mojo
let config = MOEConfig(
    num_experts=8,
    top_k=2,
    hidden_dim=512,
    expert_dim=2048
)
```

## Main Functions

### moe_gating_forward

```mojo
fn moe_gating_forward(
    input: Tensor[FLOAT_TYPE],
    gate_weights: Tensor[FLOAT_TYPE],
    config: MOEConfig
) raises -> (Tensor[FLOAT_TYPE], Tensor[INT_TYPE], Tensor[FLOAT_TYPE])
```

Performs MOE expert routing using a learned gating network.

#### Parameters

- **`input: Tensor[FLOAT_TYPE]`** - Input token embeddings
  - Shape: `[batch_size, seq_len, hidden_dim]` or `[num_tokens, hidden_dim]`
  - Contains token representations to be routed

- **`gate_weights: Tensor[FLOAT_TYPE]`** - Gating network weights
  - Shape: `[hidden_dim, num_experts]`
  - Learned parameters for expert selection

- **`config: MOEConfig`** - MOE configuration parameters

#### Returns

Returns a tuple containing:

1. **`expert_weights: Tensor[FLOAT_TYPE]`** - Expert activation weights
   - Shape: `[num_tokens, top_k]`
   - Normalized weights for selected experts

2. **`expert_indices: Tensor[INT_TYPE]`** - Selected expert indices
   - Shape: `[num_tokens, top_k]`
   - Indices of chosen experts for each token

3. **`load_balancing_loss: Tensor[FLOAT_TYPE]`** - Load balancing loss
   - Shape: `[1]`
   - Auxiliary loss to encourage uniform expert utilization

#### Raises

- `Error` if input/gate weight dimensions are incompatible
- `Error` if configuration parameters are invalid

#### Example

```mojo
let input = Tensor[FLOAT_TYPE](batch_size, seq_len, config.hidden_dim)
let gate_weights = Tensor[FLOAT_TYPE](config.hidden_dim, config.num_experts)

let (expert_weights, expert_indices, load_loss) = moe_gating_forward(
    input, gate_weights, config
)
```

### moe_expert_computation

```mojo
fn moe_expert_computation(
    input: Tensor[FLOAT_TYPE],
    expert_weights: Tensor[FLOAT_TYPE], 
    expert_indices: Tensor[INT_TYPE],
    expert_params: List[Tensor[FLOAT_TYPE]],
    config: MOEConfig
) raises -> Tensor[FLOAT_TYPE]
```

Computes outputs from selected experts using sparse activation.

#### Parameters

- **`input: Tensor[FLOAT_TYPE]`** - Input token embeddings
  - Shape: `[batch_size, seq_len, hidden_dim]`
  - Same input used for gating

- **`expert_weights: Tensor[FLOAT_TYPE]`** - Expert activation weights
  - Shape: `[num_tokens, top_k]`
  - Output from `moe_gating_forward`

- **`expert_indices: Tensor[INT_TYPE]`** - Selected expert indices
  - Shape: `[num_tokens, top_k]`
  - Output from `moe_gating_forward`

- **`expert_params: List[Tensor[FLOAT_TYPE]]`** - Expert parameters
  - Length: `num_experts`
  - Each tensor contains flattened parameters for one expert

- **`config: MOEConfig`** - MOE configuration parameters

#### Returns

- **`output: Tensor[FLOAT_TYPE]`** - Final MOE output
  - Shape: `[batch_size, seq_len, hidden_dim]`
  - Weighted combination of expert outputs

#### Expert Parameter Format

Each expert's parameters are stored as a flattened tensor with the following layout:
```
[W1_flat, W2_flat, b1_flat, b2_flat]
```

Where:
- `W1`: `[hidden_dim, expert_dim]` - Input projection
- `W2`: `[expert_dim, hidden_dim]` - Output projection
- `b1`: `[expert_dim]` - Hidden bias
- `b2`: `[hidden_dim]` - Output bias

#### Example

```mojo
# Create expert parameters
var expert_params = List[Tensor[FLOAT_TYPE]]()
for i in range(config.num_experts):
    let param_size = config.hidden_dim * config.expert_dim * 2 + config.expert_dim + config.hidden_dim
    let params = Tensor[FLOAT_TYPE](param_size)
    # Initialize parameters...
    expert_params.append(params)

let output = moe_expert_computation(
    input, expert_weights, expert_indices, expert_params, config
)
```

## Helper Functions

### compute_load_balancing_loss

```mojo
fn compute_load_balancing_loss(
    gate_probs: Tensor[FLOAT_TYPE],
    expert_indices: Tensor[INT_TYPE], 
    config: MOEConfig
) raises -> Tensor[FLOAT_TYPE]
```

Computes load balancing loss to encourage uniform expert utilization.

#### Parameters

- **`gate_probs: Tensor[FLOAT_TYPE]`** - Gating probabilities
  - Shape: `[num_tokens, num_experts]`
  - Softmax outputs from gating network

- **`expert_indices: Tensor[INT_TYPE]`** - Expert assignments
  - Shape: `[num_tokens, top_k]`
  - Selected expert indices for each token

- **`config: MOEConfig`** - MOE configuration

#### Returns

- **`loss: Tensor[FLOAT_TYPE]`** - Load balancing loss value
  - Shape: `[1]`
  - Higher values indicate more imbalanced expert usage

### apply_expert_ffn

```mojo
fn apply_expert_ffn(
    input: Tensor[FLOAT_TYPE],
    expert_weights: Tensor[FLOAT_TYPE],
    config: MOEConfig
) raises -> Tensor[FLOAT_TYPE]
```

Applies a single expert's feed-forward network.

#### Parameters

- **`input: Tensor[FLOAT_TYPE]`** - Batched input tokens
  - Shape: `[batch_size, hidden_dim]`
  - Tokens assigned to this expert

- **`expert_weights: Tensor[FLOAT_TYPE]`** - Expert parameters
  - Flattened parameter tensor for one expert

- **`config: MOEConfig`** - Configuration parameters

#### Returns

- **`output: Tensor[FLOAT_TYPE]`** - Expert output
  - Shape: `[batch_size, hidden_dim]`
  - Processed tokens from this expert

## Usage Examples

### Complete MOE Forward Pass

```mojo
from modular_hack.src.moe_kernel import *

fn complete_moe_example() raises:
    # Configuration
    let config = MOEConfig(8, 2, 512, 2048)
    
    # Input data
    let batch_size = 4
    let seq_len = 32
    let input = Tensor[FLOAT_TYPE](batch_size, seq_len, config.hidden_dim)
    let gate_weights = Tensor[FLOAT_TYPE](config.hidden_dim, config.num_experts)
    
    # Initialize input and weights...
    
    # Expert parameters
    var expert_params = List[Tensor[FLOAT_TYPE]]()
    for i in range(config.num_experts):
        let param_size = config.hidden_dim * config.expert_dim * 2 + config.expert_dim + config.hidden_dim
        let params = Tensor[FLOAT_TYPE](param_size)
        # Initialize expert parameters...
        expert_params.append(params)
    
    # Forward pass
    let (expert_weights, expert_indices, load_loss) = moe_gating_forward(
        input, gate_weights, config
    )
    
    let output = moe_expert_computation(
        input, expert_weights, expert_indices, expert_params, config
    )
    
    print("MOE forward pass completed!")
    print("Load balancing loss:", load_loss[0])
```

### Configuration Validation

```mojo
fn validate_config(config: MOEConfig) -> Bool:
    """Validate MOE configuration parameters."""
    if config.top_k > config.num_experts:
        print("Error: top_k cannot exceed num_experts")
        return False
    
    if config.hidden_dim <= 0 or config.expert_dim <= 0:
        print("Error: dimensions must be positive")
        return False
    
    if config.capacity_factor < 1.0:
        print("Error: capacity_factor must be >= 1.0")
        return False
    
    return True
```

### Performance Monitoring

```mojo
fn benchmark_moe_performance(config: MOEConfig, num_runs: Int = 100):
    """Benchmark MOE performance for given configuration."""
    let num_tokens = 64
    
    # Setup...
    let start_time = perf_counter_ns()
    
    for _ in range(num_runs):
        # MOE forward pass...
        pass
    
    let end_time = perf_counter_ns()
    let avg_time = Float64(end_time - start_time) / Float64(num_runs) / 1_000_000.0
    
    print("Average time per forward pass:", avg_time, "ms")
    print("Throughput:", Float64(num_tokens) / avg_time * 1000.0, "tokens/sec")
```

## Error Handling

### Common Errors

1. **Dimension Mismatch**
   ```mojo
   # Error: input.dim(1) != gate_weights.dim(0)
   Error("Dimension mismatch: input hidden_dim != gate_weights input_dim")
   ```

2. **Invalid Configuration**
   ```mojo
   # Error: top_k > num_experts
   Error("Invalid config: top_k must be <= num_experts")
   ```

3. **Memory Allocation**
   ```mojo
   # Error: insufficient memory for expert parameters
   Error("Failed to allocate memory for expert parameters")
   ```

### Best Practices

1. **Always validate configuration** before use
2. **Check tensor shapes** match expected dimensions
3. **Handle memory allocation failures** gracefully
4. **Monitor load balancing loss** during training
5. **Verify expert utilization** is balanced

## Performance Tips

1. **Batch Size**: Use larger batch sizes for better GPU utilization
2. **Expert Count**: Powers of 2 (4, 8, 16, 32) often perform better
3. **Memory Layout**: Ensure contiguous memory allocation for parameters
4. **Top-K Selection**: Keep k small (2-8) for optimal efficiency
5. **Load Balancing**: Monitor and adjust capacity_factor as needed

## Version Compatibility

This API is designed for:
- **Mojo**: Latest nightly builds
- **MAX**: Compatible with MAX Graph integration
- **Hardware**: Optimized for NVIDIA GPUs, supports CPU fallback

For the latest updates and compatibility information, see the main [README](../README.md).