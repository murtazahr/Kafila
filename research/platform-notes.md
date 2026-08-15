# Platform notes

Cross-platform constraints found by running the same code on Apple Metal and on
CUDA. These are engineering facts rather than research questions — they belong
in `open-questions.md` only if they turn into something to study. They are
recorded here because each one passed on one platform for days before failing on
the other, and the next person to hit them should not have to rediscover the
cause from a stack trace.

The cluster as of this writing: an M3 Pro (Metal, unified memory, 18 GB) and an
NVIDIA L40S-24Q (CUDA 12.8, driver 570, discrete VRAM, 24 GB, vGPU).

---

## mlx-c omits the 16-bit float accessors on CUDA

`mlx/c/array.h` guards `mlx_array_data_float16` and `mlx_array_data_bfloat16`
behind `#ifdef HAS_FLOAT16` and `#ifdef HAS_BFLOAT16`. Apple clang defines
those; gcc 11.4 does not. The result:

| | Metal build | CUDA build |
|---|---|---|
| `mlx_array_data_*` exported | 14 | 12 |
| `mlx_array_data_bfloat16` | present | **absent** |
| `mlx_array_data_float16` | present | **absent** |

The symbols are resolved through `dlopen` into function pointers, so calling one
that was never defined does not fail to link — it jumps to address zero and
takes the process down with `PC=0x0`. bf16 is the dtype activations use, so this
hit exactly the transport path the whole split depends on.

**What to do:** read raw bytes through an unsigned integer view of the same
width (`mlx_view` to uint8/16/32/64, all of which exist in both builds) rather
than the accessor matching the array's own dtype. `x/mlxrunner/mlx/bytes.go`
does this. It also removes the need for a C shim to cast `__bf16`, which cgo
cannot represent.

## MLX streams are thread-local on CUDA

Evaluating an array from a different OS thread than the one that created the
stream fails on CUDA with:

    mlx: There is no Stream(gpu, 0) in current thread

Metal tolerates the hop. This is what `x/internal/mlxthread` exists to prevent —
the runner funnels all MLX work through one locked OS thread — but anything
reaching MLX outside that path has to observe the same rule. In particular
`t.Run` starts a new goroutine, so a subtest that evaluates arrays will fail on
CUDA while passing on Metal. Tests need `runtime.LockOSThread`, and any future
shard agent must keep its MLX work on the pinned thread.

## Concurrent MLX processes contend, and tests go flaky

`go test ./x/...` runs package binaries in parallel, so several processes
initialize MLX and drive the same GPU at once. Roughly one run in three then
fails somewhere unrelated to what changed, with:

    mlx: There is no Stream(gpu, 0) in current thread

The failure moves between packages from run to run, which is the signature of
contention rather than a bug in whichever test reported it. Adding another
MLX-using package makes it more likely, since it adds another concurrent
process.

Run the suite with `-p 1` when the result has to be trusted:

```sh
go test ./x/... -count=1 -p 1
```

This is the same underlying limitation that stops two shards sharing a process
(see the pipeline notes): MLX keeps global state that several independent
models stepping on each other will disturb. Across processes it is intermittent
rather than a hard deadlock, which makes it more annoying to diagnose and no
less real.

## Wired-memory limits do not exist on CUDA

`configureWiredMemory()` runs during model load and calls
`mlx.MaxRecommendedWorkingSetSize()`, which is a unified-memory concept. On CUDA
it reports *"max recommended working set size unavailable"*. Upstream's
`TestSetWiredLimitRejectsOversizeWithoutChangingLimit` fails there for this
reason; the load path itself only warns. Not ours to fix, but it means one
upstream test is expected to fail on any CUDA node.

## Some custom kernels have no CUDA source

At load the runner logs:

    custom GPU kernel backend disabled kernel=gated_delta_states backend=cuda reason="no source"

`gated_delta_states` backs Mamba-style recurrent layers, so it affects
architectures like `nemotron_h` and `qwen3_5`, not dense models such as qwen3.
Worth remembering when those architectures reach the test matrix — a CUDA node
may fall back to a slower path or refuse the model.

## Ollama's "CUDA 13+" requirement is not MLX's

`docs/development.md` states CUDA 13+ for the MLX engine on Linux, and the
build preset is named `cuda_v13`. Neither is a real floor. MLX's own CMake gates
are all feature switches with fallbacks — below 12.8 it drops fp4 quantization,
below 12.9 it uses an older batched GEMM — and the only hard error is the
opposite of a minimum:

```cmake
if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "13.1" AND CUDAToolkit_VERSION VERSION_LESS "13.2")
  message(FATAL_ERROR "CUDA Toolkit 13.1 is not supported.")
```

Ollama adds no version check of its own. A CUDA 12.8 node builds and runs
correctly.

The one thing that genuinely needs care is the architecture list: the default
`mlx_cuda_v13_linux` preset asks for `sm_110/120/121`, which do require CUDA 13.
Setting `MLX_CUDA_ARCHITECTURES` routes to the `mlx_cuda_v13_user_arch` preset
and avoids them. For the L40S:

```sh
cmake -B build -S . -G Ninja \
  -DOLLAMA_MLX_BACKENDS=cuda_v13 \
  -DMLX_CUDA_ARCHITECTURES=89-real \
  -DCUDNN_INCLUDE_PATH=... -DCUDNN_LIBRARY_PATH=... -DLAPACK_INCLUDE_DIRS=...
```

## Building a node without root

Every dependency can go in a user prefix, which matters on managed machines
where sudo is not available:

- **Go** — official tarball to `~/.local/go`.
- **CMake, Ninja** — `pip install --user cmake ninja`.
- **cuDNN 9** — `apt-get download libcudnn9-cuda-12 libcudnn9-headers-cuda-12`
  then `dpkg-deb -x` into a prefix. Note `libcudnn9-dev-cuda-12` is a 13 kB
  metapackage; the headers are in `libcudnn9-headers-cuda-12`.
- **LAPACKE** — `liblapacke-dev`, extracted the same way. `libopenblas-dev`
  provides the libraries, so BLAS and LAPACK resolve and only `lapacke.h` is
  missing; the failure is `LAPACK_INCLUDE_DIRS-NOTFOUND` at MLX's generate step,
  well after configure appears to be going fine.

MLX's `FindCUDNN.cmake` honours `CUDNN_INCLUDE_PATH` and `CUDNN_LIBRARY_PATH`,
and the build copies cuDNN into the runtime payload, so the finished binary does
not depend on the prefix staying put.

## No registry tag serves safetensors

Twelve tags checked — qwen3 (0.6b/1.7b/4b), qwen3.5, gemma3, llama3.2, qwen3-vl,
granite4 — all GGUF, which routes to llama-server rather than MLX. Reaching the
MLX path means importing a HuggingFace checkpoint by hand with `ollama create
--experimental`; the unflagged path goes to GGUF conversion, which does not even
support bare `Qwen3ForCausalLM`. This is a per-node step on every machine.
