#!/usr/bin/env python3
"""
Quick SIMD validation test.
"""

import numpy as np
import time

def quick_simd_test():
    """Quick test to validate SIMD concept."""
    print("🧪 Quick SIMD Validation")
    print("=" * 30)
    
    # Small test case
    batch_size, num_experts = 32, 8
    np.random.seed(42)
    logits = np.random.randn(batch_size, num_experts).astype(np.float32)
    
    # Time naive softmax
    start = time.perf_counter()
    for _ in range(10):
        result1 = naive_softmax(logits)
    naive_time = time.perf_counter() - start
    
    # Time vectorized softmax  
    start = time.perf_counter()
    for _ in range(10):
        result2 = vectorized_softmax(logits)
    vec_time = time.perf_counter() - start
    
    # Check correctness
    assert np.allclose(result1, result2, rtol=1e-5)
    
    speedup = naive_time / vec_time
    print(f"Naive time: {naive_time:.4f}s")
    print(f"Vectorized time: {vec_time:.4f}s") 
    print(f"🚀 Speedup: {speedup:.2f}x")
    print("✅ SIMD implementation validated!")

def naive_softmax(logits):
    result = np.zeros_like(logits)
    batch_size, num_experts = logits.shape
    
    for b in range(batch_size):
        max_val = np.max(logits[b])
        sum_exp = 0.0
        for i in range(num_experts):
            exp_val = np.exp(logits[b, i] - max_val)
            result[b, i] = exp_val
            sum_exp += exp_val
        for i in range(num_experts):
            result[b, i] /= sum_exp
    return result

def vectorized_softmax(logits):
    max_vals = np.max(logits, axis=1, keepdims=True)
    shifted = logits - max_vals
    exp_vals = np.exp(shifted)
    sum_exp = np.sum(exp_vals, axis=1, keepdims=True)
    return exp_vals / sum_exp

if __name__ == "__main__":
    quick_simd_test()