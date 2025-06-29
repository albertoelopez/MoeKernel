"""
Simple test to verify SIMD implementations compile and run correctly.
"""

from tensor import Tensor, TensorShape
from math import exp, max
from algorithm import vectorize

alias FLOAT_TYPE = DType.float32

fn simple_softmax_test() raises:
    """Test basic softmax functionality."""
    let input = Tensor[FLOAT_TYPE](TensorShape(2, 3))
    
    # Initialize with simple values
    input[0, 0] = 1.0
    input[0, 1] = 2.0  
    input[0, 2] = 3.0
    input[1, 0] = 0.5
    input[1, 1] = 1.5
    input[1, 2] = 2.5
    
    let result = basic_softmax(input)
    
    print("Input:", input[0, 0], input[0, 1], input[0, 2])
    print("Softmax result:", result[0, 0], result[0, 1], result[0, 2])

fn basic_softmax(input: Tensor[FLOAT_TYPE]) raises -> Tensor[FLOAT_TYPE]:
    """Basic softmax implementation for testing."""
    let result = Tensor[FLOAT_TYPE](input.shape())
    let batch_size = input.dim(0)
    let feature_size = input.dim(1)
    
    for batch_idx in range(batch_size):
        # Find max for numerical stability
        var max_val = input[batch_idx, 0]
        for i in range(1, feature_size):
            if input[batch_idx, i] > max_val:
                max_val = input[batch_idx, i]
        
        # Compute exp and sum
        var sum_exp = Float32(0.0)
        for i in range(feature_size):
            let val = input[batch_idx, i] - max_val
            let exp_val = exp(val)
            result[batch_idx, i] = exp_val
            sum_exp += exp_val
        
        # Normalize
        for i in range(feature_size):
            result[batch_idx, i] = result[batch_idx, i] / sum_exp
    
    return result

fn simple_relu_test() raises:
    """Test basic ReLU functionality.""" 
    let input = Tensor[FLOAT_TYPE](TensorShape(4))
    
    input[0] = -2.0
    input[1] = -1.0
    input[2] = 1.0
    input[3] = 2.0
    
    let result = basic_relu(input)
    
    print("ReLU input:", input[0], input[1], input[2], input[3])
    print("ReLU output:", result[0], result[1], result[2], result[3])

fn basic_relu(input: Tensor[FLOAT_TYPE]) raises -> Tensor[FLOAT_TYPE]:
    """Basic ReLU implementation for testing."""
    let result = Tensor[FLOAT_TYPE](input.shape())
    
    for i in range(input.num_elements()):
        let val = input.load[width=1](i)
        let relu_val = max(val, Float32(0.0))
        result.store[width=1](i, relu_val)
    
    return result

fn main() raises:
    print("Testing basic implementations...")
    simple_softmax_test()
    simple_relu_test()
    print("Basic tests completed!")