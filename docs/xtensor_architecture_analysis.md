# xtensor Architecture Analysis

## Overview

`xtensor` (v0.27.1) is a **header-only C++ library** for numerical analysis with multi-dimensional array expressions. It provides an extensible expression system enabling lazy broadcasting, a NumPy-inspired API that follows C++ standard library idioms, and tools for building upon the expression system. The library requires C++20 (as of v0.27.x).

> **Source:** `README.md:15–37`, `CMakeLists.txt:209` (`target_compile_features(xtensor INTERFACE cxx_std_20)`)

---

## 1. Repository Structure

The top-level layout is:

```
xtensor/
├── CMakeLists.txt                  # Root build system
├── xtensorConfig.cmake.in          # CMake package config template
├── xtensor.pc.in                   # pkg-config template
├── include/xtensor/                # All library headers (header-only)
│   ├── core/                       # Expression engine foundation
│   ├── containers/                 # Concrete container types & adaptors
│   ├── views/                      # View types (slicing, broadcasting, etc.)
│   ├── reducers/                   # Reduction operations
│   ├── generators/                 # Array builders & random generators
│   ├── io/                         # I/O: CSV, NPY, JSON, MIME
│   ├── optional/                   # Optional/missing-value support
│   ├── chunk/                      # Chunked array support
│   ├── misc/                       # Sorting, FFT, histograms, manipulation
│   └── utils/                      # SIMD abstraction, utilities, exceptions
├── test/                           # Unit tests (doctest framework)
├── benchmark/                      # Google Benchmark micro-benchmarks
├── docs/                           # Sphinx + Doxygen documentation
├── notebooks/                      # Jupyter notebook examples (xeus-cling)
├── cmake/                          # CMake modules (e.g., TBB finder)
├── share/ & etc/                   # xeus-cpp Doxygen tagfiles & configs
├── environment-dev.yml             # Conda dev dependencies
└── environment.yml                 # Conda runtime environment
```

> **Source:** Repository root listing; `CMakeLists.txt:126–200` (complete header list)

---

## 2. Build System & Packaging

### 2.1 Header-Only INTERFACE Library

The library is declared as a CMake `INTERFACE` target with no compiled object files:

```cmake
add_library(xtensor INTERFACE)
target_compile_features(xtensor INTERFACE cxx_std_20)
target_link_libraries(xtensor INTERFACE xtl)
```

> **Source:** `CMakeLists.txt:202–211`

### 2.2 Generated Single-Include Header

At configure time, CMake generates a **single umbrella header** `xtensor.hpp` that aggregates all public module headers (excluding optional JSON/MIME/NPY headers):

```cmake
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/xtensor.hpp" "${XTENSOR_SINGLE_INCLUDE}")
```

> **Source:** `CMakeLists.txt:329–345`

### 2.3 CMake Package Export

The installed CMake config (`xtensorConfig.cmake.in`) finds `xtl` as a hard dependency and conditionally links `xsimd` and `TBB`. It also exports convenience targets:

- `xtensor::optimize` — adds `-march=native` (or MSVC equivalents)
- `xtensor::use_xsimd` — links `xsimd` and defines `XTENSOR_USE_XSIMD`
- `xtensor::use_TBB` — defines `XTENSOR_USE_TBB`

> **Source:** `xtensorConfig.cmake.in:1–71`

---

## 3. Dependencies

### 3.1 Required: xtl (≥ 0.8.0)

`xtl` is the **sole hard dependency**, providing meta-programming utilities, closure types, type traits, optional sequences, and dynamic bitsets that xtensor uses pervasively.

> **Source:** `CMakeLists.txt:43–55`, `environment-dev.yml:6`

### 3.2 Optional: xsimd (≥ 13.2.0)

Enables SIMD-vectorized operations. When `XTENSOR_USE_XSIMD` is defined, the SIMD abstraction layer in `utils/xtensor_simd.hpp` delegates to `xsimd` types and operations. Without it, scalar fallbacks are compiled instead.

> **Source:** `CMakeLists.txt:62–86`, `include/xtensor/utils/xtensor_simd.hpp:19–84` (xsimd path), lines `86–203` (fallback path)

### 3.3 Optional: nlohmann_json (≥ 3.1.1)

Enables JSON serialization/deserialization of xexpressions via `to_json`/`from_json` in `io/xjson.hpp` and MIME rendering in `io/xmime.hpp`.

> **Source:** `CMakeLists.txt:57`, `include/xtensor/io/xjson.hpp:17`

### 3.4 Optional: Intel TBB / OpenMP

Alternative parallelization backends (mutually exclusive). Enabled via `XTENSOR_USE_TBB` or `XTENSOR_USE_OPENMP` CMake options.

> **Source:** `CMakeLists.txt:62–121`

### 3.5 Test/Benchmark Dependencies

- **doctest** — unit testing framework (`test/CMakeLists.txt:21`)
- **Google Benchmark** — micro-benchmark framework, optionally fetched via `FetchContent` (`benchmark/CMakeLists.txt:77–93`)

---

## 4. Core Library Architecture

### 4.1 Expression Engine (`core/`)

The expression system is the **heart of xtensor**, implementing lazy evaluation via CRTP (Curiously Recurring Template Pattern) expression templates.

#### 4.1.1 `xexpression<D>` — Base CRTP Class

All expression types derive from `xexpression<D>`, which provides `derived_cast()` for safe downcasting:

```cpp
template <class D>
class xexpression {
public:
    using derived_type = D;
    derived_type& derived_cast() & noexcept;
    const derived_type& derived_cast() const& noexcept;
    derived_type derived_cast() && noexcept;
};
```

> **Source:** `include/xtensor/core/xexpression.hpp:34–66`

The `is_xexpression<E>` trait enables SFINAE-based dispatch for any expression type.

> **Source:** `include/xtensor/core/xexpression.hpp:180–189`

#### 4.1.2 `xfunction<F, CT...>` — Lazy Function Node

Elementwise operations return `xfunction` objects rather than materialized arrays. These store the functor `F` and closure references to operands `CT...`, computing values only on access:

```cpp
template <class F, class... CT>
class xfunction : private xconst_iterable<xfunction<F, CT...>>,
                  public xsharable_expression<xfunction<F, CT...>>,
                  ...
```

> **Source:** `include/xtensor/core/xfunction.hpp:204–209` (class declaration), lines `116–131` (shape/stepper types)

#### 4.1.3 Operator Lifting (`xoperation.hpp`)

C++ arithmetic/logical operators (`+`, `-`, `*`, `/`, `<`, `==`, etc.) are lifted into lazy `xfunction` expressions using macro-generated functor structs with both scalar and SIMD apply methods:

```cpp
BINARY_OPERATOR_FUNCTOR(plus, +);
BINARY_OPERATOR_FUNCTOR(minus, -);
BINARY_OPERATOR_FUNCTOR(multiplies, *);
// ... etc.
```

Each operator creates an `xfunction` via `detail::make_xfunction<F>(e...)`.

> **Source:** `include/xtensor/core/xoperation.hpp:31–44` (functor macros), lines `68–82` (binary macro)

#### 4.1.4 Evaluation Strategy & `eval()`

Expressions support lazy vs. immediate evaluation. `xt::eval()` forces materialization — returning a reference to existing containers (no copy) or materializing lazy expressions into `xarray`/`xtensor`:

```cpp
template <class T>
inline auto eval(T&& t)
    -> std::enable_if_t<detail::is_container<std::decay_t<T>>::value, T&&>;
```

> **Source:** `include/xtensor/core/xeval.hpp:33–57`

#### 4.1.5 Mathematical Functions (`xmath.hpp`)

Standard math functions (`sin`, `cos`, `exp`, `pow`, etc.) are lifted from `std::` into `xt::math::` and wrapped in functor structs via macros (`XTENSOR_UNARY_MATH_FUNCTOR`, `XTENSOR_BINARY_MATH_FUNCTOR`) that provide both scalar and SIMD paths.

> **Source:** `include/xtensor/core/xmath.hpp:84–150` (functor macros), lines `152–218` (using declarations)

#### 4.1.6 Supporting Core Components

| File | Purpose |
|------|---------|
| `xlayout.hpp` | `layout_type` enum (`row_major`, `column_major`, `dynamic`, `any`) |
| `xshape.hpp` | Shape container utilities, `fixed_shape<N...>` |
| `xstrides.hpp` | Stride computation, index-to-offset conversion |
| `xiterable.hpp` / `xiterator.hpp` | N-D iteration machinery, steppers |
| `xaccessible.hpp` | Element access (`operator()`, `at`, `[]`, `element`) |
| `xsemantic.hpp` | Container/view semantic base classes (assign, move) |
| `xassign.hpp` | Assignment engine with overlap checking |
| `xnoalias.hpp` | No-alias assignment optimization |
| `xvectorize.hpp` | Scalar function → expression vectorization |
| `xexpression_traits.hpp` | Expression tag dispatch, temporary type deduction |
| `xtensor_forward.hpp` | Forward declarations resolving circular includes |
| `xtensor_config.hpp` | Version macros, default layout/allocator/container config |

> **Source:** `CMakeLists.txt:138–156` (header listing), respective header files

---

### 4.2 Container Types (`containers/`)

#### 4.2.1 `xcontainer<D>` — Dense Container Base

The CRTP base class for all dense multidimensional containers. Provides:
- Shape, strides, backstrides access
- `operator()`, `element()`, `data()` element access
- Linear iterators and SIMD load/store
- Broadcasting shape computation

```cpp
template <class D>
class xcontainer : public xcontiguous_iterable<D>,
                   private xaccessible<D>
```

> **Source:** `include/xtensor/containers/xcontainer.hpp:59–120`

`xstrided_container<D>` extends this with `resize()`, `reshape()`, and layout management.

> **Source:** `include/xtensor/containers/xcontainer.hpp` (further in the file)

#### 4.2.2 `xarray` — Dynamic-Rank Container

`xarray<T>` is an alias for `xarray_container` with dynamic shape:

```cpp
template <class T, layout_type L, class A, class SA>
using xarray = xarray_container<XTENSOR_DEFAULT_DATA_CONTAINER(T, A), L, ...>;
```

Shape dimensions are determined at runtime. This is analogous to NumPy's `ndarray`.

> **Source:** `include/xtensor/core/xtensor_forward.hpp:57–82`

#### 4.2.3 `xtensor<T, N>` — Fixed-Rank Container

`xtensor<T, N>` has a compile-time number of dimensions `N`, using `std::array` for shape/strides:

```cpp
template <class T, std::size_t N, layout_type L, class A>
using xtensor = xtensor_container<XTENSOR_DEFAULT_DATA_CONTAINER(T, A), N, L>;
```

> **Source:** `include/xtensor/core/xtensor_forward.hpp:116–137`

#### 4.2.4 `xtensor_fixed<T, xshape<N...>>` — Compile-Time Shape

Shape is fully known at compile time, enabling stack allocation and aggressive optimization:

```cpp
template <class T, class FSH, layout_type L, bool Sharable>
using xtensor_fixed = xfixed_container<T, FSH, L, Sharable>;
```

> **Source:** `include/xtensor/core/xtensor_forward.hpp:155–182`

#### 4.2.5 Adaptors (`xadapt.hpp`)

The `xt::adapt()` family wraps **external memory** (STL containers, raw pointers, C arrays) into tensor semantics without copying. Supports both `xarray_adaptor` and `xtensor_adaptor` variants with ownership policies (`no_ownership`, `acquire_ownership`):

```cpp
template <layout_type L, class C, class SC>
xarray_adaptor<...> adapt(C&& container, const SC& shape, layout_type l = L);
```

This is the primary integration point for language bindings (Python/Julia/R).

> **Source:** `include/xtensor/containers/xadapt.hpp:27–90`

#### 4.2.6 Optional/Missing Value Types

`xarray_optional` and `xtensor_optional` aliases use `xtl::xoptional_vector` for arrays with missing values, tagged with `xoptional_expression_tag`:

```cpp
template <class T, layout_type L, ...>
using xarray_optional = xarray_container<xtl::xoptional_vector<T, A, BC>, L, ..., xoptional_expression_tag>;
```

> **Source:** `include/xtensor/core/xtensor_forward.hpp:91–111`, `include/xtensor/core/xtensor_forward.hpp:194–200`

#### 4.2.7 Other Container Components

| File | Purpose |
|------|---------|
| `xscalar.hpp` | Wraps scalars as 0-D expressions for broadcasting |
| `xstorage.hpp` | `uvector` (uninitialized vector), `svector` (small-buffer-optimized vector) |
| `xbuffer_adaptor.hpp` | Non-owning buffer adaptor for raw pointers |

> **Source:** `CMakeLists.txt:130–137`

---

### 4.3 Views (`views/`)

Views provide **non-owning windows** into expressions without copying data.

| File | Purpose |
|------|---------|
| `xview.hpp` | Primary N-D view with integer/slice indexing (`xt::view(a, 1, xt::all())`) |
| `xstrided_view.hpp` / `xstrided_view_base.hpp` | Strided view with dynamic slice vectors |
| `xdynamic_view.hpp` | Dynamic (runtime-determined) slice view |
| `xbroadcast.hpp` | Broadcasting expression to a target shape |
| `xindex_view.hpp` | View indexed by a set of indices (fancy indexing) |
| `xmasked_view.hpp` | Boolean-mask-based view |
| `xfunctor_view.hpp` | Apply functor to each element of underlying expression |
| `xoffset_view.hpp` | Offset into underlying storage |
| `xrepeat.hpp` | Repeat elements along an axis |
| `xslice.hpp` | Slice types: `xrange`, `xstepped_range`, `xall`, `xkeep_slice`, `xdrop_slice` |
| `xview_utils.hpp` | View helper utilities |
| `xaxis_iterator.hpp` / `xaxis_slice_iterator.hpp` | Iteration along a specific axis |
| `index_mapper.hpp` | Index mapping utilities |

> **Source:** `CMakeLists.txt:186–199`, respective header files

---

### 4.4 Reducers (`reducers/`)

| File | Purpose |
|------|---------|
| `xreducer.hpp` | `xt::reduce()` — general reduction along axes with lazy/immediate strategies, `keep_dims`, and `initial` value support |
| `xaccumulator.hpp` | `xt::accumulate()` — cumulative operations along an axis |
| `xnorm.hpp` | Norm computations (L1, L2, Linf, etc.) |
| `xblockwise_reducer.hpp` / `xblockwise_reducer_functors.hpp` | Block-wise reduction for chunked arrays |

> **Source:** `CMakeLists.txt:178–182`, `include/xtensor/reducers/xreducer.hpp:1–40`

---

### 4.5 Generators (`generators/`)

| File | Purpose |
|------|---------|
| `xbuilder.hpp` | `ones()`, `zeros()`, `empty()`, `full_like()`, `arange()`, `linspace()`, `eye()`, `diag()`, etc. |
| `xgenerator.hpp` | `xgenerator` expression template for lazily-computed arrays |
| `xrandom.hpp` | Random number generation: `rand()`, `randn()`, `randint()`, `choice()`, `shuffle()`, `permutation()` |

> **Source:** `CMakeLists.txt:157–159`, `include/xtensor/generators/xbuilder.hpp:37–55`

---

### 4.6 I/O (`io/`)

| File | Purpose |
|------|---------|
| `xcsv.hpp` | `load_csv()` / `dump_csv()` — CSV file I/O returning `xtensor_container<..., 2>` |
| `xnpy.hpp` | `load_npy()` / `dump_npy()` — NumPy `.npy` binary format support |
| `xjson.hpp` | `to_json()` / `from_json()` — nlohmann_json integration (optional, requires `nlohmann_json`) |
| `xio.hpp` | Stream output (`operator<<`) and pretty-printing |
| `xmime.hpp` | HTML table MIME output for Jupyter (optional, requires `nlohmann_json`) |
| `xinfo.hpp` | Expression info display (shape, strides, layout) |

> **Source:** `CMakeLists.txt:160–165`, respective header files

---

### 4.7 Miscellaneous (`misc/`)

| File | Purpose |
|------|---------|
| `xsort.hpp` | `sort()`, `argsort()`, `argmin()`, `argmax()`, `partition()`, `argpartition()`, `unique()` |
| `xmanipulation.hpp` | `reshape()`, `ravel()`, `flatten()`, `transpose()`, `moveaxis()`, `swapaxes()`, `squeeze()`, `expand_dims()`, `atleast_Nd()`, `split()`, `trim_zeros()`, `roll()`, `rot90()`, `flip()`, `repeat()`, `concatenate()`, `stack()`, `tile()` |
| `xcomplex.hpp` | Complex number operations (`real()`, `imag()`, `conj()`, `abs()`, `arg()`, `norm()`) |
| `xfft.hpp` | FFT operations |
| `xhistogram.hpp` | `histogram()`, `bincount()`, `digitize()` |
| `xpad.hpp` | Array padding (`pad()`) |
| `xset_operation.hpp` | Set operations (`isin()`, `in1d()`, `searchsorted()`) |
| `xexpression_holder.hpp` | Type-erased expression holder |

> **Source:** `CMakeLists.txt:166–173`

---

### 4.8 Optional Value Support (`optional/`)

| File | Purpose |
|------|---------|
| `xoptional.hpp` | Optional expression wrapper for missing value handling |
| `xoptional_assembly.hpp` | Assembly of value + flag arrays |
| `xoptional_assembly_base.hpp` | Base class for optional assemblies |
| `xoptional_assembly_storage.hpp` | Storage backend for optional assemblies |

> **Source:** `CMakeLists.txt:174–177`

---

### 4.9 Chunked Arrays (`chunk/`)

| File | Purpose |
|------|---------|
| `xchunked_array.hpp` | Chunked array container (tiled storage) |
| `xchunked_assign.hpp` | Assignment logic for chunked arrays |
| `xchunked_view.hpp` | View into a chunk of a chunked array |

> **Source:** `CMakeLists.txt:127–129`

---

### 4.10 Utilities (`utils/`)

| File | Purpose |
|------|---------|
| `xtensor_simd.hpp` | SIMD abstraction layer — delegates to xsimd when available, provides scalar fallbacks otherwise. Defines `xt_simd::simd_type<T>`, `aligned_mode`, `load_as()`, `store_as()`, etc. |
| `xutils.hpp` | General metaprogramming utilities (`nested_initializer_list_t`, `promote_shape_t`, etc.) |
| `xexception.hpp` | Custom exception types and assertion macros |

> **Source:** `CMakeLists.txt:183–185`, `include/xtensor/utils/xtensor_simd.hpp:19–84`

---

## 5. Configuration & Compile-Time Customization

The library behavior is extensively configurable via macros defined in `core/xtensor_config.hpp`:

| Macro | Default | Purpose |
|-------|---------|---------|
| `XTENSOR_DEFAULT_LAYOUT` | `row_major` | Default memory layout for containers |
| `XTENSOR_DEFAULT_TRAVERSAL` | `row_major` | Default iteration order |
| `XTENSOR_DEFAULT_DATA_CONTAINER(T, A)` | `uvector<T, A>` | Underlying flat storage type |
| `XTENSOR_DEFAULT_ALLOCATOR(T)` | `std::allocator<T>` (or `xsimd::aligned_allocator` if SIMD) | Memory allocator |
| `XTENSOR_DEFAULT_SHAPE_CONTAINER(T, EA, SA)` | `svector<size_type, 4, SA, true>` | Shape/strides container |
| `XTENSOR_DISABLE_EXCEPTIONS` | (not set) | Replace `throw` with `std::abort()` |
| `XTENSOR_ENABLE_ASSERT` | OFF | Runtime bound checking |
| `XTENSOR_CHECK_DIMENSION` | OFF | Dimension mismatch checking |

> **Source:** `include/xtensor/core/xtensor_config.hpp:13–119`, `CMakeLists.txt:213–239`

---

## 6. Language Bindings & Interop Architecture

The adaptor system (`xadapt.hpp`, `xbuffer_adaptor.hpp`) is the primary integration point for foreign language bindings. Adaptors wrap external memory buffers into xtensor expressions **without copying**, enabling:

- **xtensor-python**: Process NumPy arrays in-place via Python buffer protocol
- **xtensor-julia**: Operate on Julia arrays
- **xtensor-r**: Operate on R arrays

These bindings are maintained in **separate repositories** and are not part of this codebase.

> **Source:** `README.md:24–33`, `include/xtensor/containers/xadapt.hpp:27–29`

---

## 7. Testing & Benchmarking

### 7.1 Tests (`test/`)

Tests use the **doctest** framework. Each test file covers a specific component (e.g., `test_xarray.cpp`, `test_xview.cpp`, `test_xreducer.cpp`). The test CMake builds individual per-file executables plus a combined `test_xtensor_lib` executable. JSON-related tests are conditionally included:

```cmake
if(nlohmann_json_FOUND)
    list(APPEND XTENSOR_TESTS test_xjson.cpp)
    ...
endif()
```

> **Source:** `test/CMakeLists.txt:21–289`

### 7.2 Benchmarks (`benchmark/`)

Google Benchmark-based micro-benchmarks covering assignments, views, containers, math, reducers, builders, random, and STL interop. Can fetch Google Benchmark via `FetchContent`.

> **Source:** `benchmark/CMakeLists.txt:77–147`

---

## 8. Key Design Patterns

### 8.1 CRTP Expression Templates

Every expression type inherits from `xexpression<Derived>`. This enables static polymorphism — all template functions accept `xexpression<E>&` and use `e.derived_cast()` for zero-overhead dispatch.

### 8.2 Lazy Evaluation

Operations like `a + b * sin(c)` return `xfunction` objects that store references to operands and compute values on-demand. No intermediate arrays are allocated until `eval()` or assignment to a container.

### 8.3 Extension Points

The `extension` namespace in each module (`xarray_container_base`, `xfunction_base`, `xview_base`, etc.) provides tag-dispatched customization points, allowing optional expression tags (e.g., `xoptional_expression_tag`) to inject additional behavior.

### 8.4 SIMD Abstraction

The `xt_simd` namespace provides a compile-time switchable SIMD layer. Functors define both `operator()` (scalar) and `simd_apply()` (batched) methods. The container stepper and assignment engine select the appropriate path based on alignment and SIMD availability.

---

## 9. Dependency Graph (Simplified)

```
┌─────────────────────────────────────────────────────────┐
│                      User Code                          │
└────────┬────────────────────────┬───────────────────────┘
         │                        │
    ┌────▼────┐            ┌──────▼──────┐
    │ xarray  │            │  xtensor    │
    │ xtensor │            │  _fixed     │
    │ adapt() │            │             │
    └────┬────┘            └──────┬──────┘
         │                        │
    ┌────▼────────────────────────▼────┐
    │       xcontainer / xstrided_     │
    │         container (CRTP)         │
    └────────────┬────────────────────┘
                 │
    ┌────────────▼────────────────────┐
    │     xexpression<D> (CRTP)       │
    │  xfunction / xview / xreducer   │
    │  xgenerator / xbroadcast / ...  │
    └────────────┬────────────────────┘
                 │
    ┌────────────▼────────────────────┐
    │   xtl (metaprogramming, types)  │
    │   xsimd (optional SIMD accel.)  │
    │   nlohmann_json (optional I/O)  │
    │   TBB / OpenMP (optional par.)  │
    └─────────────────────────────────┘
```

---

## 10. Summary

| Aspect | Detail |
|--------|--------|
| **Language** | C++20 (header-only) |
| **Version** | 0.27.1 |
| **License** | BSD 3-Clause |
| **Build System** | CMake (≥ 3.15) |
| **Core Pattern** | CRTP expression templates with lazy evaluation |
| **Required Dep** | xtl ≥ 0.8.0 |
| **Optional Deps** | xsimd ≥ 13.2.0, nlohmann_json ≥ 3.1.1, Intel TBB, OpenMP |
| **Container Types** | `xarray` (dynamic rank), `xtensor` (fixed rank), `xtensor_fixed` (compile-time shape) |
| **Key Features** | NumPy-like broadcasting, lazy evaluation, SIMD acceleration, adaptors for foreign memory, optional/missing values, chunked arrays |
| **External Bindings** | xtensor-python, xtensor-julia, xtensor-r (separate repos) |
| **Test Framework** | doctest |
| **Benchmark Framework** | Google Benchmark |
