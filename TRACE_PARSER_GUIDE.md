# Trace Parser Guide

The `parse_trace.py` script makes your trace logs readable by summarizing and organizing the output.

## Quick Start

```bash
# Basic usage - shows stats + condensed view
python parse_trace.py log.out

# Statistics only
python parse_trace.py log.out --stats

# Condensed view (groups similar operations)
python parse_trace.py log.out --condensed

# Timeline view (chronological)
python parse_trace.py log.out --timeline

# Filter by layer
python parse_trace.py log.out --layer BLAS
python parse_trace.py log.out --layer Dispatcher
python parse_trace.py log.out --layer Memory
```

## Output Modes

### 1. Statistics Mode (`--stats`)

Shows quantitative summary:

```
TRACE STATISTICS
================================================================================

Events by Layer:
  Dispatcher          :  121 events
  BLAS                :   54 events
  Memory_Free         :    8 events
  Memory_Alloc        :    3 events
  Autograd_Execute    :    2 events

Operation Details:
  GEMM calls: 54
  GEMM dimensions (top 5):
    M=16, N=8, K=8: 30 times
    M=8, N=8, K=16: 24 times

  Memory allocations: 3
  Total allocated: 288 bytes (0.00 MB)
  Potential leaks: -5 allocations

  Top dispatched operations:
    aten::transpose.int: 20 times
    aten::as_strided: 18 times
    ...
```

**Use this to**:
- Get overall picture of what happened
- Count operations at each layer
- Identify most common operations
- Check for memory leaks

### 2. Condensed Mode (`--condensed`)

Groups similar operations together:

```
CONDENSED TRACE (Grouped by Operation)
================================================================================

FORWARD PASS:
--------------------------------------------------------------------------------

[BLAS] GEMM<float> (×16)
  M: 8
  N: 8
  K: 16
  alpha: 1
  beta: 0
  transA: T
  transB: N

[Dispatcher] aten::scaled_dot_product_attention (×1)
  keyset: DispatchKeySet(CPU, AutogradCPU)
  selected: AutogradCPU
```

**Use this to**:
- See the big picture without repetition
- Understand the flow: Forward → Backward
- Identify patterns (e.g., "16 GEMMs with same dimensions")
- Compare forward vs backward passes

### 3. Timeline Mode (`--timeline`)

Shows chronological execution order:

```
EXECUTION TIMELINE
================================================================================

FORWARD PASS
────────────────────────────────────────────────────────────────────────────

  1. Line    2 | [Dispatcher] aten::randn (keyset=DispatchKeySet(BackendSelect), selected=BackendSelect)
  2. Line    8 | [Dispatcher] aten::empty.memory_format (keyset=DispatchKeySet(BackendSelect), selected=BackendSelect)
  3. Line   14 | [Memory] Allocate (size=4096, mb=0.00390625, addr=0x124b71600)
  ...

────────────────────────────────────────────────────────────────────────────
BACKWARD PASS BEGINS
────────────────────────────────────────────────────────────────────────────

 87. Line  389 | [Autograd] Start backward (root=SumBackward0)
 88. Line  391 | [Autograd] Execute SumBackward0 ()
 ...
```

**Use this to**:
- See exact execution order
- Find where backward pass starts
- Trace specific operations through the stack
- Debug unexpected behavior

### 4. Filter Mode (`--layer <name>`)

Shows only events from one layer:

```bash
python parse_trace.py log.out --layer BLAS
```

```
FILTERED: BLAS layer (54 events)
================================================================================
  1. Line  199 | [BLAS] GEMM<float> (M=8, N=8, K=16, alpha=1, beta=0, transA=T, transB=N)
  2. Line  205 | [BLAS] GEMM<float> (M=16, N=8, K=8, alpha=1, beta=0, transA=N, transB=N)
  ...
```

**Available layers**:
- `Dispatcher` - ATen dispatcher routing
- `BLAS` - Matrix multiplication calls
- `Memory` - Allocations and deallocations
- `Autograd` - Backward function registration/execution
- `Kernel` - CPU kernel selection (AVX2/AVX512/DEFAULT)
- `Dynamo` - torch.compile graph capture
- `AOTAutograd` - Forward/backward partitioning
- `Inductor` - Code generation

## Common Workflows

### Understanding What Happened

```bash
# 1. Get statistics first
python parse_trace.py log.out --stats

# 2. See grouped operations
python parse_trace.py log.out --condensed

# 3. Dive into specific layer if needed
python parse_trace.py log.out --layer BLAS
```

### Debugging an Issue

```bash
# 1. See timeline to find when issue occurs
python parse_trace.py log.out --timeline --max 200

# 2. Filter to specific layer
python parse_trace.py log.out --layer Dispatcher

# 3. Check original log at specific line numbers
vim log.out +389  # Jump to line 389
```

### Comparing Forward vs Backward

```bash
# Condensed mode automatically separates forward and backward
python parse_trace.py log.out --condensed

# You'll see:
# - Forward pass operations
# - "BACKWARD PASS BEGINS" separator
# - Backward pass operations
```

### Finding Performance Bottlenecks

```bash
# 1. Count GEMMs
python parse_trace.py log.out --stats
# Look for: "GEMM calls: 54"

# 2. See GEMM patterns
python parse_trace.py log.out --layer BLAS
# Look for repeated patterns

# 3. Check memory allocations
python parse_trace.py log.out --stats
# Look for: "Total allocated: X MB"
```

## Tips

### Reduce Timeline Output

```bash
# Show only first 50 events
python parse_trace.py log.out --timeline --max 50

# Show first 100 events (default)
python parse_trace.py log.out --timeline
```

### Redirect to File

```bash
# Save condensed view to file
python parse_trace.py log.out --condensed > summary.txt

# Save statistics
python parse_trace.py log.out --stats > stats.txt
```

### Grep for Specific Operations

```bash
# Find all aten::matmul operations
python parse_trace.py log.out --layer Dispatcher | grep matmul

# Find all 8x8 GEMMs
python parse_trace.py log.out --layer BLAS | grep "M=8, N=8"

# Count backward functions
python parse_trace.py log.out --condensed | grep "Execute" | wc -l
```

## What the Parser Tracks

The parser recognizes these trace patterns:

| Layer | What It Shows |
|-------|--------------|
| **Dispatcher** | Operator routing, dispatch keys, kernel selection |
| **Autograd** | Backward function registration, execution, graph traversal |
| **Memory** | Tensor allocations, deallocations, addresses, sizes |
| **BLAS** | Matrix multiplication (M, N, K dimensions, transpose flags) |
| **C++ Impl** | High-level C++ implementations (SDPA, matmul, etc.) |
| **Kernel Dispatch** | CPU vectorization selection (AVX2/AVX512/DEFAULT) |
| **Dynamo** | torch.compile graph capture |
| **AOTAutograd** | Forward/backward graph partitioning |
| **Inductor** | Code generation and optimization |

## Example: Analyzing Your log.out

From your trace, the parser found:

```
Events by Layer:
  Dispatcher: 121 events  ← Lots of small operations
  BLAS: 54 events         ← 54 matrix multiplications
  Memory: 11 events       ← Only 11 allocations (efficient!)
  Autograd: 2 events      ← 2 backward functions executed

GEMM calls: 54
  - 24 calls: M=8, N=8, K=16 (Q@K^T pattern)
  - 30 calls: M=16, N=8, K=8 (attn@V pattern)

Memory:
  - 3 allocations
  - 8 deallocations
  - Total: 288 bytes (tiny!)
```

**Key insights**:
1. **16 forward GEMMs + 38 backward GEMMs = 54 total**
2. **Two patterns**: 8×8×16 (attention scores) and 16×8×8 (output)
3. **Minimal memory**: Only 288 bytes tracked (small example)
4. **Most operations**: Dispatcher routing (121 calls - lots of small ops)

## Customizing the Parser

You can easily add more patterns to `parse_trace.py`:

```python
# In the __init__ method, add new patterns
self.patterns = {
    # ... existing patterns ...
    'YourLayer': re.compile(r'\[TRACE\] Your pattern here.*\n.*details: (.+)'),
}

# Then add handling in _create_event method
elif layer == 'YourLayer':
    return TraceEvent(line_num, 'Your Layer', 'Operation', {
        'detail': match.group(1)
    })
```

## Next Steps

1. **Understand the condensed view** - This shows the big picture
2. **Use stats to count things** - How many GEMMs? Allocations?
3. **Filter by layer** - Deep dive into specific layers
4. **Compare eager vs compiled** - Run both tests and compare stats
5. **Add your own patterns** - Extend the parser for new instrumentation

---

**Remember**: The parser is a tool to make sense of verbose logs. The original log.out still has complete details if you need them!
