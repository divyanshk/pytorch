# PyTorch SDPA Execution Flow - Key Layers Summary

This document distills the 1000+ line trace into the essential execution path.

## Overview: What Happens When You Call SDPA

```python
out = F.scaled_dot_product_attention(query, key, value)
loss = out.sum()
loss.backward()
```

## The 7 Layers in Action

### FORWARD PASS

```
Python Call: F.scaled_dot_product_attention(Q, K, V)
     ↓
┌────────────────────────────────────────────────────────────────┐
│ Layer 1: ATen Dispatcher                                        │
│ ────────────────────────────────────────────────────────────── │
│ • Operator: aten::scaled_dot_product_attention                 │
│ • Dispatch Keys: {CPU, AutogradCPU}                            │
│ • Routes to: AutogradCPU kernel (wraps with gradient tracking) │
└────────────────────────────────────────────────────────────────┘
     ↓
┌────────────────────────────────────────────────────────────────┐
│ Layer 4: C++ Implementation (attention.cpp)                     │
│ ────────────────────────────────────────────────────────────── │
│ • Input shapes: [2, 4, 8, 16] (batch, heads, seq, dim)        │
│ • Backend selection: Flash Attention (memory-efficient)        │
│ • Calls: _scaled_dot_product_flash_attention_for_cpu           │
└────────────────────────────────────────────────────────────────┘
     ↓
┌────────────────────────────────────────────────────────────────┐
│ Layer 3: Memory Allocation (CPUAllocator.cpp)                  │
│ ────────────────────────────────────────────────────────────── │
│ • Allocate output buffer: 4096 bytes                           │
│ • Allocate temp buffers: ~1KB for intermediate results         │
└────────────────────────────────────────────────────────────────┘
     ↓
┌────────────────────────────────────────────────────────────────┐
│ Layer 5: BLAS Operations (CPUBlas.cpp)                         │
│ ────────────────────────────────────────────────────────────── │
│ • 16 GEMM calls (8 attention heads × 2 matmuls each)           │
│                                                                 │
│ Per head (repeated 8 times):                                   │
│   1. Q @ K^T  → attention scores (8×16) @ (16×8) = (8×8)      │
│   2. attn @ V → output (8×8) @ (8×16) = (8×16)                │
│                                                                 │
│ Uses external BLAS library: sgemm_ (Accelerate on macOS)       │
└────────────────────────────────────────────────────────────────┘
     ↓
┌────────────────────────────────────────────────────────────────┐
│ Layer 6: CPU Kernel Dispatch (DispatchStub.cpp)               │
│ ────────────────────────────────────────────────────────────── │
│ • CPU Capability: DEFAULT (ARM architecture, no AVX2/AVX512)   │
│ • Selects: Scalar fallback kernels                             │
└────────────────────────────────────────────────────────────────┘
     ↓
┌────────────────────────────────────────────────────────────────┐
│ Layer 2: Autograd Registration (functions/utils.h)             │
│ ────────────────────────────────────────────────────────────── │
│ • Backward function: ScaledDotProductFlashAttentionForCpuBackward0 │
│ • Attached to output tensor [2, 4, 8, 16]                      │
│ • Stores saved tensors (Q, K, V, attn) for gradient computation│
└────────────────────────────────────────────────────────────────┘
     ↓
Return output tensor with grad_fn attached
```

---

### BACKWARD PASS

```
Python Call: loss.backward()
     ↓
┌────────────────────────────────────────────────────────────────┐
│ Layer 7: Autograd Engine (engine.cpp)                          │
│ ────────────────────────────────────────────────────────────── │
│ • Starting backward pass from: SumBackward0                    │
│ • Traverses computation graph in reverse topological order     │
└────────────────────────────────────────────────────────────────┘
     ↓
┌────────────────────────────────────────────────────────────────┐
│ Execute: SumBackward0                                           │
│ ────────────────────────────────────────────────────────────── │
│ • Input gradient: scalar 1.0                                   │
│ • Output gradient: broadcast to [2, 4, 8, 16]                  │
│ • Passes grad_output to next node                              │
└────────────────────────────────────────────────────────────────┘
     ↓
┌────────────────────────────────────────────────────────────────┐
│ Execute: ScaledDotProductFlashAttentionForCpuBackward0         │
│ ────────────────────────────────────────────────────────────── │
│ • Receives: grad_output [2, 4, 8, 16]                          │
│ • Retrieves saved tensors: Q, K, V, attn_scores                │
│ • Computes: grad_query, grad_key, grad_value                   │
└────────────────────────────────────────────────────────────────┘
     ↓
┌────────────────────────────────────────────────────────────────┐
│ Layer 3: Memory Allocation (CPUAllocator.cpp)                  │
│ ────────────────────────────────────────────────────────────── │
│ • Allocate dQ (grad_query): 4096 bytes                         │
│ • Allocate dK (grad_key): 4096 bytes                           │
│ • Allocate dV (grad_value): 4096 bytes                         │
│ • Allocate temp buffers: ~1KB                                  │
└────────────────────────────────────────────────────────────────┘
     ↓
┌────────────────────────────────────────────────────────────────┐
│ Layer 5: BLAS Operations (CPUBlas.cpp)                         │
│ ────────────────────────────────────────────────────────────── │
│ • 48 GEMM calls (8 heads × 6 matmuls each)                     │
│                                                                 │
│ Per head gradient computation (repeated 8 times):              │
│   1. d_attn @ K     → partial dQ (with scale α=0.25)          │
│   2. Q^T @ d_attn   → dK                                       │
│   3. attn^T @ d_out → dV                                       │
│   4. Accumulation ops (beta=1) to aggregate gradients          │
│   + Additional matmuls for attention score gradients           │
│                                                                 │
│ Note: alpha=0.25 = 1/sqrt(16) is the attention scaling factor │
└────────────────────────────────────────────────────────────────┘
     ↓
┌────────────────────────────────────────────────────────────────┐
│ Execute: AccumulateGrad (×3 for Q, K, V)                       │
│ ────────────────────────────────────────────────────────────── │
│ • Allocates .grad buffers if not exist                         │
│ • Copies computed gradients into query.grad                    │
│ • Copies computed gradients into key.grad                      │
│ • Copies computed gradients into value.grad                    │
└────────────────────────────────────────────────────────────────┘
     ↓
┌────────────────────────────────────────────────────────────────┐
│ Layer 3: Memory Deallocation (CPUAllocator.cpp)               │
│ ────────────────────────────────────────────────────────────── │
│ • Free temporary buffers                                       │
│ • Keep gradient buffers (.grad) for user access                │
└────────────────────────────────────────────────────────────────┘
     ↓
Backward pass complete! Gradients available in Q.grad, K.grad, V.grad
```

---

## Key Statistics from Trace

| Metric | Forward Pass | Backward Pass | Total |
|--------|-------------|---------------|-------|
| **GEMM Operations** | 16 | 48 | 64 |
| **Dispatcher Calls** | ~30 | ~50 | ~80 |
| **Memory Allocations** | ~6 | ~7 | ~13 |
| **Memory Deallocations** | ~2 | ~8 | ~10 |
| **Autograd Registrations** | 2 | 0 | 2 |
| **Backward Functions Executed** | 0 | 4 | 4 |

## The Complete Stack Visualization

```
┌─────────────────────────────────────────────────────────────┐
│ Python: torch.nn.functional                                  │
│   F.scaled_dot_product_attention(Q, K, V)                   │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ Layer 1: ATen Dispatcher (Dispatcher.h)                      │
│   Routes operator → dispatch key → kernel                   │
│   Dispatch Keys: BackendSelect → AutogradCPU → CPU          │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ Layer 2: Autograd Registration (functions/utils.h)          │
│   set_history(output, grad_fn)                              │
│   Attaches backward node to output tensor                   │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ Layer 3: Memory Allocation (CPUAllocator.cpp)               │
│   Allocates tensor storage                                  │
│   Tracks memory lifecycle                                   │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ Layer 4: C++ Implementation (attention.cpp)                 │
│   High-level algorithm logic                                │
│   Backend selection (Flash vs Math fallback)                │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ Layer 5: BLAS Layer (CPUBlas.cpp)                           │
│   Low-level matrix multiplication                           │
│   Calls external library (MKL/OpenBLAS/Accelerate)          │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ Layer 6: CPU Kernel Dispatch (DispatchStub.cpp)            │
│   Selects vectorized kernels (AVX2/AVX512/DEFAULT)          │
│   CPU capability detection                                  │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ Hardware: CPU executes SIMD instructions                     │
│   ARM NEON (on Mac M1/M2)                                   │
│   x86 AVX2/AVX512 (on Intel/AMD)                            │
└─────────────────────────────────────────────────────────────┘
                          ↓
                  ← Results flow back up ←
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ Layer 7: Autograd Engine (engine.cpp)                       │
│   Executes backward pass when .backward() called            │
│   Traverses computation graph                               │
│   Calls each backward function in reverse order             │
└─────────────────────────────────────────────────────────────┘
```

## Simplified Mental Model

Think of PyTorch execution as passing through these checkpoints:

1. **Python API** → "What operation do you want?"
2. **Dispatcher** → "Which implementation should handle this?"
3. **Autograd** → "Should we track gradients?" (if yes, attach backward node)
4. **Memory** → "Allocate space for results"
5. **C++ Logic** → "How do we break this into primitives?"
6. **BLAS** → "Do the actual math (matrix multiplications)"
7. **Kernels** → "Use SIMD instructions for speed"

Then on backward:
1. **Autograd Engine** → "Execute backward functions in reverse"
2. Repeat steps 4-7 to compute gradients
3. **AccumulateGrad** → "Write gradients into .grad attributes"

## Why So Many Operations?

**Forward pass (16 GEMMs):**
- 2 batches × 4 heads = 8 attention heads
- Each head: `Q@K^T` + `attn@V` = 2 GEMMs
- Total: 8 × 2 = **16 GEMMs**

**Backward pass (48 GEMMs):**
- For each head, compute gradients wrt Q, K, V, and attention scores
- Each gradient requires ~6 matrix operations (including accumulations)
- Total: 8 × 6 = **48 GEMMs**

**Why 3× more work in backward?**
- Chain rule: need to differentiate through multiple operations
- Must compute gradients for 3 inputs (Q, K, V)
- Attention mechanism has multiplicative interactions → more gradient terms

## Key Takeaways

1. **Everything goes through the dispatcher** - it's the traffic controller
2. **Autograd wraps operations transparently** - you don't see it in Python
3. **Memory management is explicit** - every allocation/deallocation is tracked
4. **Operations decompose into primitives** - SDPA → matmuls → BLAS → SIMD
5. **Backward pass is more expensive** - 3× the GEMMs of forward pass
6. **Gradients flow backwards** - from loss → intermediate nodes → inputs

This is how modern deep learning frameworks work under the hood!
