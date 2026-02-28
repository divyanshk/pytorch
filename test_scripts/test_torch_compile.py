#!/usr/bin/env python3
"""
Test script to trace PyTorch internals with torch.compile enabled.
Tests: Scaled dot product attention (SDPA) with backward pass in compiled mode

This shows the additional compilation layers:
- Dynamo: Graph capture and tracing
- AOTAutograd: Ahead-of-time autograd graph generation
- Inductor: Code generation and optimization
- Triton: GPU kernel generation (CPU fallback on Mac)

Compare with test_matmul_trace.py to see eager vs compiled execution.
"""

import os
import sys

# ==============================================================================
# TORCH_LOGS - Enable detailed logging for torch.compile internals
# Reference: https://docs.pytorch.org/tutorials/recipes/torch_logs.html
# ==============================================================================

# Enable comprehensive logging for all torch.compile components
os.environ['TORCH_LOGS'] = '+dynamo,+aot,+inductor,+output_code,+graph_breaks,+guards,+recompiles'

# Alternative: Enable specific components only
# os.environ['TORCH_LOGS'] = '+dynamo'  # Just Dynamo (graph capture)
# os.environ['TORCH_LOGS'] = '+aot'     # Just AOTAutograd
# os.environ['TORCH_LOGS'] = '+inductor' # Just Inductor (code gen)

# ==============================================================================
# TORCH_COMPILE_DEBUG - Save generated code and graphs to disk
# Reference: https://pytorch.org/docs/stable/torch.compiler_debug.html
# ==============================================================================

# Save all compilation artifacts to torch_compile_debug/ directory
os.environ['TORCH_COMPILE_DEBUG'] = '1'

# Alternative debug levels:
# os.environ['TORCH_COMPILE_DEBUG'] = '0'  # Disabled (default)
# os.environ['TORCH_COMPILE_DEBUG'] = '1'  # Basic debugging info
# os.environ['TORCH_COMPILE_DEBUG'] = '2'  # Verbose debugging

# ==============================================================================
# Additional useful environment variables
# ==============================================================================

# Disable parallelism for clean trace (same as eager mode test)
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'  # macOS Accelerate
os.environ['NUMEXPR_NUM_THREADS'] = '1'

# Disable CUDA (we're testing CPU compilation)
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import torch
import torch.nn.functional as F

# Set PyTorch internal thread counts
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

print("=" * 80)
print("PyTorch Internals Trace: SDPA with torch.compile")
print("=" * 80)
print(f"\nSettings:")
print(f"  Compilation: ENABLED (torch.compile)")
print(f"  Backend: inductor (default)")
print(f"  Parallelism: DISABLED (single-threaded)")
print(f"  OMP_NUM_THREADS: {os.environ.get('OMP_NUM_THREADS')}")
print(f"  torch.get_num_threads(): {torch.get_num_threads()}")
print(f"\nEnvironment Variables:")
print(f"  TORCH_LOGS: {os.environ.get('TORCH_LOGS')}")
print(f"  TORCH_COMPILE_DEBUG: {os.environ.get('TORCH_COMPILE_DEBUG')}")
print(f"\nInstrumentation:")
print(f"  ✓ Eager mode layers (Dispatcher, Autograd, Memory, BLAS, etc.)")
print(f"  ✓ Compilation layers (Dynamo, AOTAutograd, Inductor)")
print(f"  ✓ Generated code saved to: torch_compile_debug/")
print("=" * 80)

# ==============================================================================
# Define the SDPA operation as a separate function for compilation
# ==============================================================================

def sdpa_forward(query, key, value):
    """
    Wrapper function for scaled dot product attention.
    This function will be compiled by torch.compile.
    """
    return F.scaled_dot_product_attention(query, key, value)

# ==============================================================================
# Compile the function
# ==============================================================================

print("\n" + "=" * 80)
print("COMPILING FUNCTION")
print("=" * 80)
print("\nCompiling sdpa_forward with torch.compile(backend='inductor')...")
print("This will trigger:")
print("  1. Dynamo - Captures Python bytecode into FX graph")
print("  2. AOTAutograd - Generates forward + backward graphs")
print("  3. Inductor - Optimizes and generates CPU/Triton kernels")
print("-" * 80)

# Compile the function
# backend='inductor' is the default - generates optimized code
# mode='default' balances compilation time vs runtime performance
# fullgraph=False allows graph breaks (more flexible but may be slower)
compiled_sdpa = torch.compile(
    sdpa_forward,
    backend='inductor',  # Use TorchInductor code generator
    mode='default',      # Optimization mode: 'default', 'reduce-overhead', 'max-autotune'
    fullgraph=False,     # Allow graph breaks
)

print("\n✓ Function compiled successfully!")
print("  Note: Actual compilation happens on first call (lazy compilation)")

# ==============================================================================
# Test: Scaled Dot Product Attention with Backward Pass
# ==============================================================================

print("\n" + "=" * 80)
print("SCALED DOT PRODUCT ATTENTION - COMPILED MODE (FORWARD + BACKWARD)")
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
print("FORWARD PASS - FIRST CALL (TRIGGERS COMPILATION)")
print("=" * 80)
print(f"\nOperation: out = compiled_sdpa(Q, K, V)")
print("This is where compilation actually happens!")
print("-" * 80)

# First call triggers compilation
# You'll see Dynamo, AOTAutograd, and Inductor traces here
out = compiled_sdpa(query, key, value)

print(f"\n✓ Compilation complete!")
print(f"\nOutput tensor:")
print(f"  out.shape: {out.shape}, dtype: {out.dtype}, requires_grad: {out.requires_grad}")
print(f"  out.grad_fn: {out.grad_fn}")

print(f"\n{'=' * 80}")
print("BACKWARD PASS")
print("=" * 80)
print(f"\nOperation: loss = out.sum(); loss.backward()")
print("-" * 80)

# Compute loss and backpropagate
# This will use the compiled backward graph
loss = out.sum()
loss.backward()

print(f"\n✓ Backward pass complete!")
print(f"\nGradients computed:")
print(f"  query.grad.shape: {query.grad.shape}")
print(f"  key.grad.shape: {key.grad.shape}")
print(f"  value.grad.shape: {value.grad.shape}")

# ==============================================================================
# Second call - No recompilation (should be fast)
# ==============================================================================

print(f"\n{'=' * 80}")
print("FORWARD PASS - SECOND CALL (NO RECOMPILATION)")
print("=" * 80)
print(f"\nCalling compiled_sdpa again with same input shapes...")
print("This should reuse the compiled code (no recompilation).")
print("-" * 80)

# Create new tensors with same shape
query2 = torch.randn(batch_size, num_heads, seq_len, head_dim, requires_grad=True)
key2 = torch.randn(batch_size, num_heads, seq_len, head_dim, requires_grad=True)
value2 = torch.randn(batch_size, num_heads, seq_len, head_dim, requires_grad=True)

out2 = compiled_sdpa(query2, key2, value2)
loss2 = out2.sum()
loss2.backward()

print(f"\n✓ Second call complete (used cached compiled code)")

# ==============================================================================
# Summary
# ==============================================================================

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"""
torch.compile adds several new layers to the execution stack:

COMPILATION PHASE (First call only):
╔════════════════════════════════════════════════════════════════╗
║ 1. Dynamo (TorchDynamo)                                         ║
║    - Intercepts Python bytecode                                 ║
║    - Traces execution into FX graph                             ║
║    - Handles dynamic shapes, control flow, graph breaks         ║
║    Location: torch/_dynamo/                                     ║
╠════════════════════════════════════════════════════════════════╣
║ 2. AOTAutograd (Ahead-of-Time Autograd)                        ║
║    - Takes FX graph from Dynamo                                 ║
║    - Generates separate forward and backward graphs             ║
║    - Partitions into forward + backward                         ║
║    Location: torch/_functorch/_aot_autograd/                   ║
╠════════════════════════════════════════════════════════════════╣
║ 3. Inductor (TorchInductor)                                    ║
║    - Receives forward/backward graphs from AOTAutograd          ║
║    - Applies optimization passes (fusion, memory planning)      ║
║    - Generates optimized code:                                  ║
║      * CPU: C++ code with vectorization                         ║
║      * GPU: Triton kernels                                      ║
║    - Compiles generated code                                    ║
║    Location: torch/_inductor/                                   ║
╚════════════════════════════════════════════════════════════════╝

EXECUTION PHASE (All calls):
  - Runs compiled code instead of eager operations
  - Still goes through Dispatcher for ops not compiled
  - Gradients computed using compiled backward graph

SAVED ARTIFACTS:
  Check torch_compile_debug/ directory for:
  - FX graphs (from Dynamo)
  - Forward/backward graphs (from AOTAutograd)
  - Generated code (from Inductor)
  - Optimization logs

EAGER vs COMPILED:
  Eager mode:  Python → Dispatcher → C++ → BLAS → Kernels
  Compiled:    Python → [Dynamo → AOTAutograd → Inductor] → Compiled Code
               (compilation cached after first call)

Next steps:
1. Check torch_compile_debug/ for generated code
2. Compare performance: eager vs compiled
3. Instrument Dynamo, AOTAutograd, Inductor for deeper tracing
""")

print("=" * 80)
print("Test completed successfully!")
print("=" * 80)
