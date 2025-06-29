"""
Benchmark suite for MOE kernel performance evaluation.
"""

from time import now
from tensor import Tensor, TensorShape
from moe_kernel import MOEConfig, moe_gating_forward, moe_expert_computation
from algorithm import parallelize
from builtin import float32, int32

struct BenchmarkConfig:
    """Configuration for benchmarking MOE kernels."""
    var batch_size: Int
    var seq_len: Int
    var num_experts: Int
    var top_k: Int
    var hidden_dim: Int
    var expert_dim: Int
    var num_runs: Int
    var warmup_runs: Int
    
    fn __init__(inout self, batch_size: Int, seq_len: Int, num_experts: Int, 
                top_k: Int, hidden_dim: Int, expert_dim: Int, 
                num_runs: Int = 100, warmup_runs: Int = 10):
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden_dim = hidden_dim
        self.expert_dim = expert_dim
        self.num_runs = num_runs
        self.warmup_runs = warmup_runs

fn benchmark_moe_gating(config: BenchmarkConfig) raises -> Float64:
    """Benchmark MOE gating performance."""
    let moe_config = MOEConfig(
        config.num_experts,
        config.top_k, 
        config.hidden_dim,
        config.expert_dim
    )
    
    # Create input tensors
    let input = Tensor[DType.float32](config.batch_size, config.seq_len, config.hidden_dim)
    let gate_weights = Tensor[DType.float32](config.hidden_dim, config.num_experts)
    
    # Initialize with random-like values
    for i in range(input.num_elements()):
        input[i] = Float32((i * 7) % 1000) * 0.001 - 0.5
        
    for i in range(gate_weights.num_elements()):
        gate_weights[i] = Float32((i * 13) % 1000) * 0.001 - 0.5
    
    # Warmup runs
    for _ in range(config.warmup_runs):
        _ = moe_gating_forward(input, gate_weights, moe_config)
    
    # Benchmark runs
    let start_time = now()
    
    for _ in range(config.num_runs):
        _ = moe_gating_forward(input, gate_weights, moe_config)
    
    let end_time = now()
    let total_time = end_time - start_time
    let avg_time_ns = total_time / config.num_runs
    
    return Float64(avg_time_ns) / 1_000_000.0  # Convert to milliseconds

fn benchmark_moe_expert_computation(config: BenchmarkConfig) raises -> Float64:
    """Benchmark MOE expert computation performance."""
    let moe_config = MOEConfig(
        config.num_experts,
        config.top_k,
        config.hidden_dim, 
        config.expert_dim
    )
    
    let num_tokens = config.batch_size * config.seq_len
    
    # Create input tensors
    let input = Tensor[DType.float32](config.batch_size, config.seq_len, config.hidden_dim)
    let expert_weights = Tensor[DType.float32](num_tokens, config.top_k)
    let expert_indices = Tensor[DType.int32](num_tokens, config.top_k)
    
    # Create expert parameters
    var expert_params = List[Tensor[DType.float32]]()
    for i in range(config.num_experts):
        let param_size = config.hidden_dim * config.expert_dim + config.expert_dim * config.hidden_dim + config.expert_dim + config.hidden_dim
        let expert_param = Tensor[DType.float32](param_size)
        for j in range(param_size):
            expert_param[j] = Float32((i * j * 17) % 1000) * 0.001 - 0.5
        expert_params.append(expert_param)
    
    # Initialize inputs
    for i in range(input.num_elements()):
        input[i] = Float32((i * 11) % 1000) * 0.001 - 0.5
        
    for i in range(expert_weights.num_elements()):
        expert_weights[i] = 1.0 / Float32(config.top_k)
        
    for i in range(expert_indices.num_elements()):
        expert_indices[i] = Int32(i % config.num_experts)
    
    # Warmup runs
    for _ in range(config.warmup_runs):
        _ = moe_expert_computation(input, expert_weights, expert_indices, expert_params, moe_config)
    
    # Benchmark runs
    let start_time = now()
    
    for _ in range(config.num_runs):
        _ = moe_expert_computation(input, expert_weights, expert_indices, expert_params, moe_config)
    
    let end_time = now()
    let total_time = end_time - start_time
    let avg_time_ns = total_time / config.num_runs
    
    return Float64(avg_time_ns) / 1_000_000.0  # Convert to milliseconds

fn benchmark_full_moe_forward(config: BenchmarkConfig) raises -> Float64:
    """Benchmark full MOE forward pass."""
    let moe_config = MOEConfig(
        config.num_experts,
        config.top_k,
        config.hidden_dim,
        config.expert_dim
    )
    
    let num_tokens = config.batch_size * config.seq_len
    
    # Create input tensors
    let input = Tensor[DType.float32](config.batch_size, config.seq_len, config.hidden_dim)
    let gate_weights = Tensor[DType.float32](config.hidden_dim, config.num_experts)
    
    # Create expert parameters
    var expert_params = List[Tensor[DType.float32]]()
    for i in range(config.num_experts):
        let param_size = config.hidden_dim * config.expert_dim + config.expert_dim * config.hidden_dim + config.expert_dim + config.hidden_dim
        let expert_param = Tensor[DType.float32](param_size)
        for j in range(param_size):
            expert_param[j] = Float32((i * j * 19) % 1000) * 0.001 - 0.5
        expert_params.append(expert_param)
    
    # Initialize inputs
    for i in range(input.num_elements()):
        input[i] = Float32((i * 23) % 1000) * 0.001 - 0.5
        
    for i in range(gate_weights.num_elements()):
        gate_weights[i] = Float32((i * 29) % 1000) * 0.001 - 0.5
    
    # Warmup runs
    for _ in range(config.warmup_runs):
        let gating_results = moe_gating_forward(input, gate_weights, moe_config)
        _ = moe_expert_computation(input, gating_results.0, gating_results.1, expert_params, moe_config)
    
    # Benchmark runs
    let start_time = now()
    
    for _ in range(config.num_runs):
        let gating_results = moe_gating_forward(input, gate_weights, moe_config)
        _ = moe_expert_computation(input, gating_results.0, gating_results.1, expert_params, moe_config)
    
    let end_time = now()
    let total_time = end_time - start_time
    let avg_time_ns = total_time / config.num_runs
    
    return Float64(avg_time_ns) / 1_000_000.0  # Convert to milliseconds

fn run_benchmark_suite():
    """Run comprehensive benchmark suite."""
    print("MOE Kernel Benchmark Suite")
    print("=" * 60)
    
    # Define benchmark configurations
    let configs = List[BenchmarkConfig](
        # Small scale
        BenchmarkConfig(2, 32, 4, 2, 128, 512),
        # Medium scale  
        BenchmarkConfig(4, 64, 8, 2, 256, 1024),
        # Large scale
        BenchmarkConfig(8, 128, 16, 4, 512, 2048),
        # Extra large scale
        BenchmarkConfig(16, 256, 32, 8, 1024, 4096)
    )
    
    for i in range(len(configs)):
        let config = configs[i]
        let num_tokens = config.batch_size * config.seq_len
        let scale_name: String
        
        if i == 0:
            scale_name = "Small"
        elif i == 1:
            scale_name = "Medium"
        elif i == 2:
            scale_name = "Large"
        else:
            scale_name = "Extra Large"
        
        print(f"\n{scale_name} Scale Configuration:")
        print(f"  Batch Size: {config.batch_size}, Seq Len: {config.seq_len}")
        print(f"  Num Experts: {config.num_experts}, Top-K: {config.top_k}")
        print(f"  Hidden Dim: {config.hidden_dim}, Expert Dim: {config.expert_dim}")
        print(f"  Total Tokens: {num_tokens}")
        
        try:
            # Benchmark gating
            let gating_time = benchmark_moe_gating(config)
            print(f"  Gating Time: {gating_time:.3f} ms")
            
            # Benchmark expert computation
            let expert_time = benchmark_moe_expert_computation(config)
            print(f"  Expert Computation Time: {expert_time:.3f} ms")
            
            # Benchmark full forward pass
            let full_time = benchmark_full_moe_forward(config)
            print(f"  Full Forward Time: {full_time:.3f} ms")
            
            # Calculate throughput
            let tokens_per_ms = Float64(num_tokens) / full_time
            let tokens_per_sec = tokens_per_ms * 1000.0
            print(f"  Throughput: {tokens_per_sec:.0f} tokens/sec")
            
        except e:
            print(f"  Error running benchmark: {str(e)}")
    
    print("\n" + "=" * 60)
    print("Benchmark suite completed!")

fn main():
    """Main benchmark entry point."""
    run_benchmark_suite()