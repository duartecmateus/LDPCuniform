# Metal port of the alg3 decoder

macOS-only port of the `alg3` (LU) pruning decoder, used for the parallel
measurements. It performs the same cancellation search as the CPU version, with
the syndrome/column intersections evaluated on the GPU.

**Status: uploaded as-is.** It is not part of the default build, needs Xcode
(Objective-C + the Metal toolchain), and has not been re-verified against the
current source in `decoders/alg3_lu/`. Treat it as a starting point rather than
a supported target; the CPU tree is the reference.

## Build

```bash
cd gpu/metal
make          # compiles kernel.metal -> build/kernel.metallib, then links build/test
make bench    # BP-versus-pruning comparison harness
```

The Makefile looks for `metal`/`metallib` through `xcrun` and falls back to the
Metal toolchain that ships inside the OS cryptex, so it works on a machine with
Xcode command line tools but no full Xcode SDK.

Frameworks linked: `Foundation`, `Metal`, `AppKit` (`-fobjc-arc`).

## Layout

| File | Role |
|---|---|
| `kernel.metal` | GPU kernels (syndrome intersection counting, column support) |
| `metal_bridge.m` / `.h` | Objective-C bridge: device setup, buffer upload, dispatch |
| `alg2.c`, `alg2_gpu_prune.c` | Driver and the GPU-side pruning loop |
| `alg3prun{256,bin,gf16,gf4}.c` | Field-specialised solvers, as on the CPU side |
| `sparse_utils.c`, `gf{4,16,256}.c` | Shared loading and finite-field arithmetic |
| `test.c`, `bench_bp_vs_prune.c` | Entry points produced by the Makefile |

Matrices are not shipped here; generate them with `genmatrix/genU` or reuse the
ones in `../../decoders/alg3_lu/matrices/`.
