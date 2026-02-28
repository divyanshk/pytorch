# torch.compile Instrumentation Summary

This document describes the instrumentation added to trace torch.compile execution.

## Test Script
- **File**: `test_torch_compile.py`
- **Operation**: Scaled Dot Product Attention (SDPA) with backward pass
- **Mode**: Compiled mode with `torch.compile(backend='inductor')`
- **Comparison**: Compare with `test_matmul_trace.py` (eager mode)

## torch.compile Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│ Python: torch.compile(fn, backend='inductor')                   │
└─────────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────────┐
│ LAZY COMPILATION: First call to compiled function triggers      │
│ the compilation pipeline below                                  │
└─────────────────────────────────────────────────────────────────┘
                          ↓
        ╔═════════════════════════════════════════════╗
        ║  COMPILATION PIPELINE (runs once per shape) ║
        ╚═════════════════════════════════════════════╝
                          ↓
┌─────────────────────────────────────────────────────────────────┐
│ Stage 1: Dynamo (TorchDynamo)                                   │
│ Location: torch/_dynamo/                                        │
│ ────────────────────────────────────────────────────────────────│
│ • Intercepts Python bytecode via frame evaluation hook          │
│ • Traces execution into FX graph (symbolic tracing)             │
│ • Handles dynamic shapes, guards, graph breaks                  │
│ • Outputs: FX graph representing the Python function            │
└─────────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────────┐
│ Stage 2: AOTAutograd (Ahead-of-Time Autograd)                  │
│ Location: torch/_functorch/_aot_autograd/                      │
│ ────────────────────────────────────────────────────────────────│
│ • Takes FX graph from Dynamo                                    │
│ • Creates "joint" forward-backward graph                        │
│ • Partitions into separate forward and backward graphs          │
│ • Applies decompositions and transformations                    │
│ • Outputs: Forward graph + Backward graph                       │
└─────────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────────┐
│ Stage 3: Inductor (TorchInductor)                              │
│ Location: torch/_inductor/                                      │
│ ────────────────────────────────────────────────────────────────│
│ • Receives forward/backward graphs from AOTAutograd             │
│ • Applies optimization passes:                                  │
│   - Operator fusion (combine multiple ops into one kernel)      │
│   - Memory planning (minimize allocations)                      │
│   - Layout optimization (choose best memory layout)             │
│ • Generates code:                                               │
│   - CPU: C++ code with vectorization (AVX2/AVX512)             │
│   - GPU: Triton kernels (Python-like GPU programming)          │
│ • Compiles generated code into executable                       │
│ • Outputs: Compiled function ready to execute                   │
└─────────────────────────────────────────────────────────────────┘
                          ↓
        ╔═════════════════════════════════════════════╗
        ║  EXECUTION PHASE (runs every call)          ║
        ╚═════════════════════════════════════════════╝
                          ↓
┌─────────────────────────────────────────────────────────────────┐
│ Compiled Code Execution                                          │
│ ────────────────────────────────────────────────────────────────│
│ • Runs optimized C++/Triton kernels                             │
│ • Bypasses Python overhead                                      │
│ • Fused operations reduce memory traffic                        │
│ • Still goes through Dispatcher for non-compiled ops            │
└─────────────────────────────────────────────────────────────────┘
```

## Instrumented Files (3 Total)

### 1. Dynamo - Graph Capture
**File**: `torch/_dynamo/convert_frame.py`
**Function**: `_compile` (line 1442)
**Purpose**: Captures Python bytecode and converts to FX graph

**Instrumentation added** (lines 1465-1469):
```python
print("\n[TRACE] Dynamo: Starting graph capture (torch/_dynamo/convert_frame.py:1442)")
print(f"  Function: {code.co_name}")
print(f"  Filename: {code.co_filename}:{code.co_firstlineno}")
print(f"  Compile ID: {compile_id}")
```

**What it shows**:
- Which Python function is being compiled
- Source file and line number
- Unique compilation ID for tracking

**Example output**:
```
[TRACE] Dynamo: Starting graph capture (torch/_dynamo/convert_frame.py:1442)
  Function: sdpa_forward
  Filename: /path/to/test_torch_compile.py:42
  Compile ID: CompileId(123)
```

### 2. AOTAutograd - Forward/Backward Graph Generation
**File**: `torch/_functorch/_aot_autograd/graph_compile.py`
**Function**: `aot_stage2_autograd` (line 2148)
**Purpose**: Partitions joint graph into forward and backward

**Instrumentation added** (lines 2157-2160):
```python
print("\n[TRACE] AOTAutograd: Generating forward/backward graphs (torch/_functorch/_aot_autograd/graph_compile.py:2148)")
print(f"  Joint graph nodes: {len(aot_graph_capture.graph_module.graph.nodes)}")
print(f"  → Will partition into separate forward and backward graphs")
```

**What it shows**:
- Size of the joint (combined forward+backward) graph
- Indication that partitioning will occur

**Example output**:
```
[TRACE] AOTAutograd: Generating forward/backward graphs (torch/_functorch/_aot_autograd/graph_compile.py:2148)
  Joint graph nodes: 47
  → Will partition into separate forward and backward graphs
```

### 3. Inductor - Code Generation
**File**: `torch/_inductor/compile_fx.py`
**Function**: `compile_fx` (line 2483)
**Purpose**: Optimizes graph and generates executable code

**Instrumentation added** (lines 2502-2506):
```python
print("\n[TRACE] Inductor: Starting code generation (torch/_inductor/compile_fx.py:2483)")
print(f"  FX graph nodes: {len(model_.graph.nodes)}")
print(f"  Example inputs: {len(example_inputs_)}")
print(f"  → Will optimize and generate C++/Triton code")
```

**What it shows**:
- Size of FX graph to be compiled
- Number of example inputs for shape inference
- Target code generation (C++ for CPU, Triton for GPU)

**Example output**:
```
[TRACE] Inductor: Starting code generation (torch/_inductor/compile_fx.py:2483)
  FX graph nodes: 25
  Example inputs: 3
  → Will optimize and generate C++/Triton code
```

## Environment Variables for Debugging

### TORCH_LOGS
Enables detailed logging from torch.compile components.

**Usage in test script**:
```python
os.environ['TORCH_LOGS'] = '+dynamo,+aot,+inductor,+output_code,+graph_breaks,+guards,+recompiles'
```

**Components**:
- `+dynamo` - Dynamo graph capture logs
- `+aot` - AOTAutograd partitioning logs
- `+inductor` - Inductor optimization logs
- `+output_code` - Shows generated code
- `+graph_breaks` - Shows where graphs are broken
- `+guards` - Shows shape guards and specializations
- `+recompiles` - Shows when recompilation occurs

**Reference**: https://pytorch.org/tutorials/recipes/torch_logs.html

### TORCH_COMPILE_DEBUG
Saves compilation artifacts to disk.

**Usage in test script**:
```python
os.environ['TORCH_COMPILE_DEBUG'] = '1'
```

**Values**:
- `'0'` - Disabled (default)
- `'1'` - Basic debugging (saves graphs and code)
- `'2'` - Verbose debugging (includes intermediate steps)

**Output directory**: `torch_compile_debug/`

**Saved artifacts**:
- FX graphs (GraphViz .dot files)
- Generated C++ or Triton code
- Optimization logs
- Performance metrics

**Reference**: https://pytorch.org/docs/stable/torch.compiler_debug.html

## Execution Flow Comparison

### Eager Mode (test_matmul_trace.py)
```
Python
  ↓
Dispatcher → routes to kernel
  ↓
Autograd → registers backward function
  ↓
Memory Allocation
  ↓
C++ Implementation
  ↓
BLAS (16 GEMMs for SDPA)
  ↓
CPU Kernels (DEFAULT/AVX2/AVX512)
  ↓
Hardware execution
```

### Compiled Mode (test_torch_compile.py)
```
Python (first call only)
  ↓
╔══════════════════════════════════╗
║ COMPILATION (lazy, cached)       ║
╠══════════════════════════════════╣
║ Dynamo: bytecode → FX graph     ║
║   ↓                              ║
║ AOTAutograd: joint → fwd+bwd     ║
║   ↓                              ║
║ Inductor: optimize + codegen     ║
╚══════════════════════════════════╝
  ↓
Compiled function (all calls)
  ↓
Optimized kernels (fused operations)
  ↓
Reduced memory traffic
  ↓
Hardware execution
```

## Key Differences: Eager vs Compiled

| Aspect | Eager Mode | Compiled Mode |
|--------|-----------|---------------|
| **Execution** | Interprets ops one-by-one | Compiles entire graph upfront |
| **Overhead** | Python + dispatcher per op | Compilation once, fast execution |
| **Optimization** | None (runs ops as-is) | Fusion, memory planning, etc. |
| **GEMMs** | 16 separate calls | May fuse into fewer kernels |
| **Memory** | Allocates per-op | Planned allocations |
| **Debugging** | Easy (matches Python) | Harder (compiled code) |
| **First run** | Fast | Slow (compilation) |
| **Subsequent runs** | Same speed | Much faster |

## What to Expect in Output

### First Call (Compilation)
You'll see traces in this order:
1. **Dynamo** trace - function being captured
2. **AOTAutograd** trace - graph partitioning
3. **Inductor** trace - code generation
4. TORCH_LOGS output (if enabled):
   - Dynamo: bytecode analysis, guards
   - AOTAutograd: forward/backward split
   - Inductor: optimization passes, generated code
5. Eager mode traces (for ops not compiled)
   - Dispatcher, Memory, BLAS, etc. (if any ops fall back)

### Second Call (Cached)
You'll see:
- No compilation traces (reuses cached code)
- Minimal output (just execution)
- Much faster runtime

### Saved Artifacts (torch_compile_debug/)
Check this directory for:
```
torch_compile_debug/
├── run_<timestamp>/
│   ├── torchdynamo/
│   │   └── debug.log          # Dynamo logs
│   ├── aot_autograd/
│   │   ├── forward_graph.txt  # Forward FX graph
│   │   └── backward_graph.txt # Backward FX graph
│   └── torchinductor/
│       ├── model.cpp          # Generated C++ code (CPU)
│       ├── model.py           # Generated Triton code (GPU)
│       └── output_code.txt    # Final compiled code
```

## Performance Expectations

For SDPA with shape `[2, 4, 8, 16]`:
- **Eager mode**: ~16 GEMM calls, many small ops
- **Compiled mode**: Potentially fused into fewer, larger kernels
- **Speedup**: 1-3x for small examples, 2-10x for larger models
- **Compilation time**: 5-30 seconds (first call only)

## How to Run

1. **Rebuild PyTorch** (required for instrumentation):
   ```bash
   python setup.py develop
   ```

2. **Run eager mode test** (baseline):
   ```bash
   python test_matmul_trace.py > log_eager.out 2>&1
   ```

3. **Run compiled mode test**:
   ```bash
   python test_torch_compile.py > log_compiled.out 2>&1
   ```

4. **Compare outputs**:
   ```bash
   diff log_eager.out log_compiled.out
   ```

5. **Check compilation artifacts**:
   ```bash
   ls -R torch_compile_debug/
   ```

## Next Steps

1. Run both tests and compare execution paths
2. Examine generated code in `torch_compile_debug/`
3. Optionally add more instrumentation:
   - Operator fusion passes in Inductor
   - Memory planning in Inductor
   - Guard checking in Dynamo
   - Triton kernel generation (if GPU available)
