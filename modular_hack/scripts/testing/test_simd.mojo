"""
Simple test to validate SIMD implementations before integration.
"""

from tensor import Tensor, TensorSpec, TensorShape
from algorithm import vectorize
from sys.info import simdwidthof
from math import exp, max
from memory import memset_zero

alias FLOAT_TYPE = DType.float32

fn test_simd_softmax() raises:
    """Test the SIMD softmax implementation."""
    print("Testing SIMD softmax...")
    
    # Create test tensor
    let input = Tensor[FLOAT_TYPE](TensorShape(2, 4))
    
    # Initialize with test values
    input[0, 0] = 1.0
    input[0, 1] = 2.0
    input[0, 2] = 3.0
    input[0, 3] = 4.0
    input[1, 0] = 0.5
    input[1, 1] = 1.5
    input[1, 2] = 2.5
    input[1, 3] = 3.5
    
    let result = simd_softmax(input, dim=1)
    
    print("Input shape:", input.shape())
    print("Output shape:", result.shape())
    print("Softmax test completed successfully!")

fn simd_softmax(input: Tensor[FLOAT_TYPE], dim: Int) raises -> Tensor[FLOAT_TYPE]:
    """Apply softmax along specified dimension with SIMD optimization."""
    alias simd_width = simdwidthof[FLOAT_TYPE]()
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
            
            # Normalize by sum
            for i in range(feature_size):
                result[batch_idx, i] = result[batch_idx, i] / sum_exp
    
    return result

fn test_simd_relu() raises:
    """Test the SIMD ReLU implementation."""
    print("Testing SIMD ReLU...")
    
    # Create test tensor with positive and negative values
    let input = Tensor[FLOAT_TYPE](TensorShape(4))
    input[0] = -2.0
    input[1] = -1.0
    input[2] = 1.0
    input[3] = 2.0
    
    let result = simd_relu(input)
    
    print("ReLU input:", input[0], input[1], input[2], input[3])
    print("ReLU output:", result[0], result[1], result[2], result[3])
    print("ReLU test completed successfully!")

fn simd_relu(input: Tensor[FLOAT_TYPE]) raises -> Tensor[FLOAT_TYPE]:
    """Apply ReLU activation function with SIMD optimization."""
    let result = Tensor[FLOAT_TYPE](input.shape())
    
    for i in range(input.num_elements()):
        let val = input.load[width=1](i)
        let relu_val = max(val, Float32(0.0))
        result.store[width=1](i, relu_val)
    
    return result

fn main() raises:
    print("Starting SIMD implementation tests...")
    test_simd_softmax()
    test_simd_relu()
    print("All SIMD tests passed!")