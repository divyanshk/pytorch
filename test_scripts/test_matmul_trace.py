#!/usr/bin/env python3
"""
Test script to trace PyTorch internals in eager mode.
Tests: Scaled dot product attention (SDPA) with backward pass

Instrumentation layers (7 total):
1. ATen Dispatcher (Dispatcher.h) - Shows dispatch key routing
2. Autograd function registration (functions/utils.h) - Shows backward node creation
3. Memory allocation/deallocation (CPUAllocator.cpp) - Shows tensor memory management
4. C++ implementation (attention.cpp, LinearAlgebra.cpp) - High-level operators
5. BLAS layer (CPUBlas.cpp) - Low-level matrix operations
6. CPU kernel dispatch (DispatchStub.cpp) - AVX2/AVX512 vectorization selection
7. Autograd backward execution (engine.cpp) - Backward pass execution

NOTE: Parallelism is disabled for clean trace output.
"""

import os

# ==============================================================================
# ENABLE DISPATCHER TRACING - Shows which dispatch keys are invoked
# ==============================================================================
os.environ['TORCH_SHOW_DISPATCH_TRACE'] = '1'

# ==============================================================================
# DISABLE PARALLELISM - Must be set BEFORE importing torch!
# This prevents garbled output from multi-threaded execution
# ==============================================================================
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'  # macOS Accelerate
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import torch
import torch.nn.functional as F

# Set PyTorch's internal thread counts
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

print("=" * 80)
print("PyTorch Internals Trace: SDPA with Backward Pass")
print("=" * 80)
print(f"\nSettings:")
print(f"  Parallelism: DISABLED (single-threaded for clean output)")
print(f"  OMP_NUM_THREADS: {os.environ.get('OMP_NUM_THREADS')}")
print(f"  torch.get_num_threads(): {torch.get_num_threads()}")
print(f"\nInstrumentation (7 layers):")
print(f"  ✓ ATen Dispatcher - dispatch key routing")
print(f"  ✓ Autograd registration - backward node creation")
print(f"  ✓ Memory allocation - tensor memory lifecycle")
print(f"  ✓ C++ implementations - high-level operators")
print(f"  ✓ BLAS layer - low-level matrix ops")
print(f"  ✓ CPU kernel dispatch - vectorization selection")
print(f"  ✓ Autograd execution - backward pass")
print("=" * 80)

# ============================================================================
# Test: Scaled Dot Product Attention with Backward Pass
# ============================================================================

print("\n" + "=" * 80)
print("SCALED DOT PRODUCT ATTENTION (FORWARD + BACKWARD)")
print("=" * 80)

# Create tensors for attention
# Shape: (batch, num_heads, seq_len, head_dim)
batch_size = 2
num_heads = 4
seq_len = 8
head_dim = 16

query = torch.randn(batch_size, num_heads, seq_len, head_dim, requires_grad=True)
key = torch.randn(batch_size, num_heads, seq_len, head_dim, requires_grad=True)
value = torch.randn(batch_size, num_heads, seq_len, head_dim, requires_grad=True)

print(f"\nInput tensors:")
print(f"  query.shape: {query.shape}, dtype: {query.dtype}, requires_grad: {query.requires_grad}")
print(f"  key.shape: {key.shape}, dtype: {key.dtype}, requires_grad: {key.requires_grad}")
print(f"  value.shape: {value.shape}, dtype: {value.dtype}, requires_grad: {value.requires_grad}")

print(f"\n{'=' * 80}")
print("FORWARD PASS")
print("=" * 80)
print(f"\nOperation: out = F.scaled_dot_product_attention(Q, K, V)")
print("-" * 80)

# Perform scaled dot product attention
out = F.scaled_dot_product_attention(query, key, value)

print(f"\nOutput tensor:")
print(f"  out.shape: {out.shape}, dtype: {out.dtype}, requires_grad: {out.requires_grad}")
print(f"  out.grad_fn: {out.grad_fn}")

print(f"\n{'=' * 80}")
print("BACKWARD PASS")
print("=" * 80)
print(f"\nOperation: loss = out.sum(); loss.backward()")
print("-" * 80)

# Compute loss and backpropagate
loss = out.sum()
loss.backward()

print(f"\nGradients computed:")
print(f"  query.grad.shape: {query.grad.shape}")
print(f"  key.grad.shape: {key.grad.shape}")
print(f"  value.grad.shape: {value.grad.shape}")

# ============================================================================
# Summary
# ============================================================================

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"""
Complete trace showing all 7 instrumented layers:

1. FORWARD PASS:
   ✓ Dispatcher - routing through AutogradCPU → CPU dispatch keys
   ✓ Autograd registration - backward nodes attached to output tensors
   ✓ Memory allocation - tensor buffers allocated for Q, K, V, outputs
   ✓ Flash Attention backend selection
   ✓ 16 GEMM calls (8 heads × 2 matmuls per head)
   ✓ CPU kernel dispatch (AVX2/AVX512/DEFAULT selection)
   ✓ BLAS library calls (sgemm_)

2. BACKWARD PASS:
   ✓ Dispatcher - routing backward operations
   ✓ Autograd execution - engine traversing computation graph
   ✓ Backward functions - gradient computation for each operation
   ✓ Memory allocation - gradient buffers allocated
   ✓ More GEMM calls for computing gradients (dQ, dK, dV)
   ✓ Memory deallocation - temporary buffers freed

Expected output: Interleaved traces from all 7 layers showing the complete
execution flow from Python → Dispatcher → Autograd → Memory → C++ → BLAS → Kernels

Total operations: ~32+ GEMMs (16 forward + 16+ backward)
""")

print("=" * 80)
print("Test completed successfully!")
print("=" * 80)
