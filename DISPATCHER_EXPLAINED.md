# The PyTorch Dispatcher: ATen, C10, and the Dispatch System

## The Big Picture

PyTorch is organized into layers, with the **Dispatcher** being the central routing system that connects everything.

```
┌─────────────────────────────────────────────────────────────┐
│ Python API (torch.nn, torch.functional)                     │
│   torch.matmul(a, b)                                        │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ ATen (A Tensor Library) - "A10"                             │
│   Location: aten/                                           │
│   Role: Defines ALL tensor operations                       │
│   - Operator schemas (what ops exist)                       │
│   - Implementations (CPU, CUDA, Meta, etc.)                 │
│   - The "vocabulary" of operations                          │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ DISPATCHER - The Central Router                             │
│   Location: c10/core/dispatch/ and aten/src/ATen/core/dispatch/
│   Role: Routes each operation to the right implementation   │
│   - Looks at dispatch keys                                  │
│   - Selects appropriate kernel                              │
│   - Handles autograd, profiling, etc.                       │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ C10 (Core 10) - Foundation Library                          │
│   Location: c10/                                            │
│   Role: Core utilities and primitives                       │
│   - Tensor metadata (sizes, strides, dtype, device)         │
│   - Memory allocators                                       │
│   - Dispatch key system                                     │
│   - Device abstraction                                      │
│   - NO implementations, just infrastructure                 │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ Actual Kernels (Implementations)                            │
│   - CPU kernels (aten/src/ATen/native/)                    │
│   - CUDA kernels (aten/src/ATen/native/cuda/)              │
│   - Meta kernels (fake execution for shape inference)      │
│   - Autograd kernels (generate backward graphs)            │
└─────────────────────────────────────────────────────────────┘
```

---

## What is ATen? (A10)

**ATen = "A Tensor Library"** (also jokingly "A10" because it's 10x better than some other library)

**Location**: `aten/` directory

**Role**: ATen is the **operator library** - it defines:
1. **What operations exist** (matmul, conv2d, add, relu, etc.)
2. **Their schemas** (what arguments they take)
3. **Multiple implementations** for different backends

**Example**:
```cpp
// ATen defines the operation
namespace at {
  Tensor matmul(const Tensor& self, const Tensor& other);
}

// And provides implementations
// aten/src/ATen/native/LinearAlgebra.cpp - CPU implementation
// aten/src/ATen/native/cuda/Blas.cu - CUDA implementation
```

**Key point**: ATen is the **"vocabulary"** of PyTorch operations. Every operation you call in Python (torch.add, F.relu, etc.) maps to an ATen operation.

---

## What is C10? (Core 10)

**C10 = "Core 10"** (core utilities, foundation layer)

**Location**: `c10/` directory

**Role**: C10 is the **infrastructure layer** - it provides:
1. **Tensor metadata** (TensorImpl, Storage, sizes, strides)
2. **Device abstraction** (CPU, CUDA, MPS, etc.)
3. **Memory allocators** (Allocator interface, CPUAllocator, CUDAAllocator)
4. **Dispatch keys** (the enum of all dispatch keys)
5. **Type system** (ScalarType - float, int, etc.)
6. **No actual operations!** Just the plumbing

**Example**:
```cpp
// c10 defines the Tensor structure
namespace c10 {
  class TensorImpl {
    Storage storage_;
    IntArrayRef sizes_;
    IntArrayRef strides_;
    ScalarType dtype_;
    Device device_;
    // ... metadata only, NO operations
  };

  enum class DispatchKey {
    CPU,
    CUDA,
    AutogradCPU,
    AutogradCUDA,
    // ... all possible dispatch keys
  };

  class Allocator {
    virtual DataPtr allocate(size_t bytes) = 0;
    // ... interface only
  };
}
```

**Key point**: C10 is the **foundation** - it has NO implementations of operations, just the infrastructure that ATen and the Dispatcher use.

---

## The Dispatcher: The Central Router

### What It Does

The Dispatcher is a **dynamic routing system** that:
1. Takes an operation call (e.g., `torch.matmul(a, b)`)
2. Looks at the **dispatch keys** (CPU? CUDA? Autograd enabled? Tracing?)
3. Routes to the **appropriate kernel** for that combination

### How Dispatch Keys Work

Every tensor carries a **DispatchKeySet** - a set of dispatch keys that describe its context:

```cpp
// Your tensor on CPU with autograd enabled
DispatchKeySet = {AutogradCPU, CPU, BackendSelect}
                  ^            ^    ^
                  |            |    +-- Selects device backend
                  |            +------- Actual CPU implementation
                  +-------------------- Autograd wrapper (records for backward)
```

### Dispatch Key Priority Order

Keys are processed in **priority order** (highest to lowest):

```
Priority (highest to lowest):
1. AutogradCPU / AutogradCUDA     ← Wraps operation for gradient tracking
2. ADInplaceOrView                ← Tracks in-place ops and views
3. Tracer                         ← torch.jit.trace
4. Profiler                       ← Profiling hooks
5. BackendSelect                  ← Chooses CPU vs CUDA vs ...
6. CPU / CUDA / MPS / ...         ← Actual implementation
7. CompositeImplicitAutograd      ← Fallback decompositions
```

### Example: torch.matmul Dispatch

Let's trace `torch.matmul(a, b)` where `a` and `b` are CPU tensors with `requires_grad=True`:

```
Python: torch.matmul(a, b)
  ↓
ATen: at::matmul(a, b)  [defined in aten/]
  ↓
Dispatcher: Look at dispatch keys
  - a.device() = CPU
  - a.requires_grad() = True
  - DispatchKeySet = {AutogradCPU, CPU}
  ↓
Dispatcher: Select highest priority key
  - Highest: AutogradCPU
  ↓
Route to: AutogradCPU kernel
  Location: torch/csrc/autograd/generated/VariableType_0.cpp
  Action:
    1. Call the actual CPU kernel
    2. Register backward function (MmBackward0)
    3. Return result with grad_fn attached
  ↓
Inside AutogradCPU kernel:
  - Redispatch with lower priority keys: {CPU}
  - Dispatcher routes to: CPU kernel
  ↓
CPU kernel: aten/src/ATen/native/LinearAlgebra.cpp
  - _matmul_impl(a, b)
  - Calls BLAS gemm
  - Returns result
  ↓
Back to AutogradCPU:
  - Attach MmBackward0 to result
  - Return to Python
```

---

## How They Work Together

### Example: You call `torch.add(a, b)`

```
┌─────────────────────────────────────────────────────────────┐
│ 1. Python                                                    │
│    torch.add(a, b)                                          │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. ATen (defines the operation)                             │
│    at::add(const Tensor& self, const Tensor& other)        │
│    Schema: add(Tensor self, Tensor other) -> Tensor        │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. C10 (provides tensor metadata)                           │
│    - a.device() -> CPU                                      │
│    - a.dtype() -> Float                                     │
│    - a.requires_grad() -> True                              │
│    - Construct DispatchKeySet: {AutogradCPU, CPU}          │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ 4. Dispatcher (routes based on keys)                        │
│    Look up operator: "aten::add"                            │
│    Dispatch keys: {AutogradCPU, CPU}                        │
│    Highest priority: AutogradCPU                            │
│    → Route to AutogradCPU kernel                            │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ 5. AutogradCPU kernel (ATen)                                │
│    Location: torch/csrc/autograd/generated/VariableType.cpp │
│    Action:                                                   │
│      - Redispatch to CPU kernel (strip Autograd key)        │
│      - Register AddBackward0 for gradient                   │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ 6. CPU kernel (ATen)                                        │
│    Location: aten/src/ATen/native/BinaryOps.cpp            │
│    Action: Actual CPU implementation                        │
│      - Allocate output (via C10 allocator)                  │
│      - Perform element-wise addition                        │
│      - Return result                                        │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ 7. Back through the stack                                   │
│    - CPU kernel returns to AutogradCPU                      │
│    - AutogradCPU attaches grad_fn                           │
│    - Result returned to Python                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Code Locations

### C10 (Foundation)
```
c10/
├── core/
│   ├── TensorImpl.h          # Tensor metadata structure
│   ├── Device.h              # Device abstraction
│   ├── ScalarType.h          # Data types (float, int, etc.)
│   ├── Allocator.h           # Memory allocator interface
│   ├── CPUAllocator.cpp      # CPU memory allocator ← You instrumented this!
│   └── dispatch/
│       └── DispatchKeySet.h  # Dispatch key definitions
└── util/
    └── ArrayRef.h            # Array reference utilities
```

### ATen (Operations)
```
aten/
├── src/ATen/
│   ├── core/
│   │   └── dispatch/
│   │       └── Dispatcher.h/cpp  # The Dispatcher itself ← You instrumented this!
│   ├── native/
│   │   ├── LinearAlgebra.cpp     # CPU matmul implementation
│   │   ├── BinaryOps.cpp         # CPU add, mul, etc.
│   │   ├── Activation.cpp        # CPU relu, sigmoid, etc.
│   │   ├── CPUBlas.cpp           # BLAS wrappers ← You instrumented this!
│   │   └── cuda/
│   │       └── Blas.cu           # CUDA BLAS wrappers
│   └── templates/
│       └── ... (code generation templates)
└── gen.py  # Code generator for dispatch tables
```

### Autograd (Built on ATen + C10)
```
torch/csrc/autograd/
├── generated/
│   ├── VariableType_0.cpp    # AutogradCPU kernels (generated)
│   └── Functions.cpp         # Backward functions (generated)
├── engine.cpp                # Backward pass execution ← You instrumented this!
└── functions/
    └── utils.h               # set_history() ← You instrumented this!
```

---

## Why This Architecture?

### Separation of Concerns

1. **C10**: Core infrastructure
   - Can be used standalone
   - No operation implementations
   - Just data structures and interfaces

2. **ATen**: Operation library
   - Depends on C10
   - Defines operations
   - Multiple implementations per operation

3. **Dispatcher**: Dynamic routing
   - Connects operations to implementations
   - Handles cross-cutting concerns (autograd, profiling, tracing)
   - Single point of control

### Benefits

**Extensibility**: Add new backends without changing ATen
```cpp
// Register a new backend (e.g., MPS for Apple Silicon)
TORCH_LIBRARY_IMPL(aten, MPS, m) {
  m.impl("add", &add_mps_kernel);
}
// Dispatcher automatically routes CPU tensors → MPS kernels
```

**Composability**: Dispatch keys stack
```
{Profiler, AutogradCPU, CPU}
   ↓
Profiler wraps AutogradCPU wraps CPU implementation
```

**Single source of truth**: All routing logic in one place
- Don't need if/else in every operation
- Just register kernels for each dispatch key

---

## Dispatcher in Your Trace

From your `log.out`, you saw:

```
[TRACE] ATen Dispatcher (aten/src/ATen/core/dispatch/Dispatcher.h:776)
  Operator: aten::scaled_dot_product_attention
  Dispatch KeySet: DispatchKeySet(CPU, AutogradCPU)
  Highest priority key: AutogradCPU
  → Selected kernel for key: AutogradCPU
```

**What happened**:
1. **ATen**: Defined `scaled_dot_product_attention` operation
2. **C10**: Provided tensor metadata (device=CPU, requires_grad=True)
3. **C10**: Constructed DispatchKeySet = {CPU, AutogradCPU}
4. **Dispatcher**: Selected AutogradCPU (highest priority)
5. **ATen**: Routed to AutogradCPU kernel
6. **Autograd kernel**: Called CPU implementation + registered backward
7. **C10**: Allocated memory for result
8. **ATen**: CPU kernel executed actual computation

---

## The Dispatch Table

The Dispatcher maintains a table like this:

```
Operator: aten::add
┌──────────────────┬────────────────────────────────────┐
│ Dispatch Key     │ Kernel Function                    │
├──────────────────┼────────────────────────────────────┤
│ AutogradCPU      │ torch/csrc/autograd/.../add        │
│ CPU              │ aten/native/BinaryOps.cpp::add_cpu │
│ CUDA             │ aten/native/cuda/BinaryOps.cu::add │
│ MPS              │ aten/native/mps/BinaryOps.mm::add  │
│ Meta             │ aten/native/meta/BinaryOps.cpp     │
│ CompositeImplicit│ [decomposition to primitives]      │
└──────────────────┴────────────────────────────────────┘

Operator: aten::matmul
┌──────────────────┬────────────────────────────────────┐
│ AutogradCPU      │ VariableType.cpp::matmul           │
│ CPU              │ LinearAlgebra.cpp::matmul          │
│ CUDA             │ cuda/Blas.cu::matmul               │
│ ...              │ ...                                │
└──────────────────┴────────────────────────────────────┘
```

When you call an operation:
1. Dispatcher looks up the operator name
2. Checks the DispatchKeySet
3. Finds the highest priority key that has a registered kernel
4. Calls that kernel

---

## Dispatch Keys You've Seen

From your trace:

### BackendSelect
```
Dispatch KeySet: DispatchKeySet(BackendSelect)
```
- Initial routing layer
- Selects CPU vs CUDA vs MPS based on tensor device
- Then redispatches with specific device key

### AutogradCPU
```
Dispatch KeySet: DispatchKeySet(CPU, AutogradCPU)
Highest priority key: AutogradCPU
```
- Wraps operations for gradient tracking
- Registers backward functions
- Redispatches to CPU kernel for actual computation

### CPU
```
Dispatch KeySet: DispatchKeySet(CPU)
Highest priority key: CPU
```
- Actual CPU implementation
- No autograd wrapper
- Direct execution

### ADInplaceOrView
```
Dispatch KeySet: DispatchKeySet(CPU, ADInplaceOrView, AutogradCPU)
```
- Tracks in-place operations (x.add_())
- Tracks views (x.view(), x.transpose())
- Needed for autograd correctness

---

## Summary: The Three Layers

| Component | What | Where | Role |
|-----------|------|-------|------|
| **C10** | Core infrastructure | `c10/` | Foundation: tensors, devices, allocators, dispatch keys |
| **ATen** | Operator library | `aten/` | Operations: defines all ops, provides implementations |
| **Dispatcher** | Routing system | `aten/src/ATen/core/dispatch/` | Router: connects ops to kernels based on context |

**Think of it like**:
- **C10**: The roads and traffic lights
- **ATen**: The destinations and buildings
- **Dispatcher**: The GPS that routes you based on conditions

---

## Why "A10" and "C10"?

**Historical naming**:
- **A10**: "A Tensor library" - 10x better than the previous library (Torch7's TH)
- **C10**: "Core 10" - the core utilities, extracted to be reusable

Initially, PyTorch had everything in ATen. Over time:
1. Core utilities extracted → C10
2. Dispatch system formalized → Dispatcher
3. ATen became the operation library on top of C10

The "10" is a bit of PyTorch humor - it's "10x" better! 😄

---

## Key Takeaways

1. **C10 = Foundation** (no ops, just infrastructure)
2. **ATen = Operations** (defines all tensor operations)
3. **Dispatcher = Router** (connects ops to implementations)
4. **Dispatch keys** determine which implementation runs
5. **Priority order** allows wrapping (Autograd wraps CPU)
6. **All together** = PyTorch's flexible, extensible architecture

The Dispatcher is why PyTorch can:
- Support multiple backends (CPU, CUDA, MPS, etc.)
- Add autograd transparently
- Enable torch.compile
- Profile operations
- Trace for JIT
- All without modifying operation code!

It's the **central nervous system** of PyTorch.
