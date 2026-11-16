# LDPC decoders for burst errors

Reference implementation of a family of LDPC decoders and of the *uniform*
parity-check matrix construction they rely on. The material here accompanies

> D. Mateus and R. Chaves, *A Parallel LDPC Decoder for Error Reconciliation in
> Quantum Key Distribution* (manuscript, to be submitted).

and builds on the earlier decoder of

> D. Mateus and R. Chaves, *A Fast Parallel Decoder for LDPC Codes Suitable for
> Burst Errors*, IEEE Information Theory Workshop (ITW), 2025.

If you use this code, please cite both.

The decoders are not belief propagation. Error correction is cast as **solving a
linear system over a finite field**: the syndrome identifies which columns can
contain errors, the columns that provably do not are pinned to zero by appending
identity rows, and the resulting square system is solved by Gaussian elimination
(LU). The construction is tuned so that this works below the code's *efficient
threshold*, and the same problem lies in **NC** when the solution is obtained
with Csanky's inversion (see Bibliography), which is what makes the scheme
interesting on massively parallel hardware.

Everything here works over `GF(2)`, `GF(4)`, `GF(16)` and `GF(256)`.

## Contents

| Path | What it is |
|---|---|
| `decoders/alg2_pruning/` | Sequential pruning decoders, one per field, plus the binary augmentation decoder, the BP baselines (SPA and Min-Sum) and a head-to-head `compare` harness |
| `decoders/alg3_lu/` | The same decoding strategy rewritten around a standalone LU solver: the fastest CPU variant, all four fields, plus the collected comparison table |
| `decoders/alg4_prefilter/` | `GF(4)`/`GF(16)` consistency pre-filter: ranks candidate columns by how consistently their error values agree across rows, then solves; optional row augmentation |
| `genmatrix/` | `genU` (uniform random sparse matrix generator, all fields) and `genpeg` (binary PEG, girth-optimised, for the BP baselines) |
| `results/` | Collected benchmark output (CSV) and plotting script |
| `gpu/metal/` | Metal port of the `alg3` decoder (macOS). **Provided as-is: not part of the default build, not verified against the current CPU sources** |
| `C/`, `Python/` | The 2025 prototype of the original decoder (serial C + threaded Python). Kept for reference; the trees above supersede it |

## Build

Requires a C99 compiler and `make`. No external dependencies.

```bash
make -C decoders/alg3_lu          # -> test_lu
make -C decoders/alg2_pruning     # -> 6 binaries in build/
make -C decoders/alg4_prefilter   # -> test_alg4prefilt   (uses sources from alg3_lu)
make -C genmatrix                 # -> build/genU
make -C genmatrix build/genpeg
```

`decoders/alg4_prefilter` compiles the linear-algebra helpers from
`decoders/alg3_lu`, so build that one first.

## Run

Every decoder reads a parity-check matrix in the sparse format written by
`genU`: a 4-line header `m n c field` followed by one line per column holding
`c` entries of the form `row:value`. Test matrices live in each decoder's
`matrices/` directory.

`decoders/alg3_lu` — arguments `<matrix> <field> <error_pct> [threshold_v] [max_weight] [max_tries]`,
with `field` given as `0=GF(2)`, `4=GF(4)`, `16=GF(16)`, `1=GF(256)`:

```bash
cd decoders/alg3_lu
./test_lu matrices/H_GF16_1600x4000_c6_R060.txt 16 5 4 200 20   # PASS  132 ms
./test_lu matrices/H_GF256_BER2_2000.txt        1  2 4 200 20   # PASS  1.6 ms
./test_lu matrices/H_GF4_n1000_R0.60.txt        4  2 4 200 20   # PASS  1.6 ms
./test_lu matrices/H_bin_BER1_Bprime_2000.txt   0  1 3 50 20    # PASS  1.6 ms
```

Timings are what these commands print on an M-series Mac, single-threaded, on
random (unoptimised) matrices.

`decoders/alg2_pruning` — one binary per field, arguments
`<matrix> <err_pct> <threshold_v> <max_error_weight> <max_tries> <csv> [seed]`:

```bash
cd decoders/alg2_pruning
./build/testprunebin matrices/H_bin_BER1_Bprime_2000.txt 1 2 50 20 out.csv
./build/testprunegf16 matrices/H_GF16_n500_1pct.txt 1 2 50 20 out.csv
```

`decoders/alg4_prefilter` — arguments `<matrix> <field> <ber_list> <trials>`:

```bash
cd decoders/alg4_prefilter
./test_alg4prefilt matrices/H_GF4_n1000_c5_R055.txt 4 1,2 10
```

Every binary prints its own `Usage:` line when called with no arguments; that
line is the authoritative form. `make test` in `decoders/alg3_lu` prints the
canonical example invocations.

## Generating matrices

```bash
cd genmatrix && make && make build/genpeg

# uniform matrix: <m> <n> <column_weight> <field> <out>   (0=GF2 1=GF256 2=GF16 3=GF4)
./build/genU 60 500 6 2 H_gf16_60x500.txt

# binary PEG with girth optimisation: <m> <n> <c> <seed> <out>
./build/genpeg 400 1000 5 12345 H_peg_bin_R060_n1000.txt
```

## Results

Timings come from the benchmark runs shipped in `results/` and
`decoders/alg3_lu/comparison_results.csv`, measured single-threaded on Apple
silicon. Re-measure on your own hardware.

Symbol channel, equal information size (250 GF(256) symbols ≡ 500 GF(16) ≡
1000 GF(4) ≡ 2000 bits), pruning/LU decoder versus PEG + BP:

| Symbol error | Field | Best rate | Time | PEG-BP baseline |
|---|---|---|---|---|
| 1% | GF(256) | R = 0.88 | 0.04 ms | R = 0.80, 16.8 ms |
| 2% | GF(256) | R = 0.80 | 0.04 ms | R = 0.75, 12.0 ms |
| 3% | GF(256) | R = 0.70 | 0.09 ms | R = 0.60, 7.1 ms |
| 4% | GF(256) | R = 0.65 | 0.09 ms | R = 0.60, 8.1 ms |

On the binary symmetric channel the same machinery reaches R = 0.65 at 1% and
degrades above 2%, while PEG-BP reaches R = 0.80 at 1% and R = 0.60 at 4%:
binary pruning is competitive but not better. The advantage of this approach is
on symbol/burst channels and in non-binary fields, where no BP baseline of
comparable rate exists.

## Channel models — read this before comparing numbers

| Decoder | Channel | What an "error" is |
|---|---|---|
| GF(256), GF(16), GF(4) pruning | burst / q-ary symmetric | a whole symbol flips |
| GF(2) pruning, PEG-BP | binary symmetric (BSC) | one bit flips |

Comparing a `GF(256)` decoder with a binary BP decoder at the same *numeric*
error rate is not a fair comparison: at 1% bit error a 2000-bit block holds about
20 flips spread over roughly 19 of 250 symbols, i.e. about 7.6% symbol error.
Match the information size and compare like with like.

## Status

- CPU decoders: build and run in all four fields (verified with clang on macOS).
- `gpu/metal/`: used for the parallel measurements, uploaded unchanged. Needs
  Xcode, is not wired into the default build, and has not been re-verified
  against the current CPU sources. Treat it as a starting point.
- CUDA: not included. The PRAM/NC model the algorithm assumes does not map
  directly onto CUDA, so a port is not a translation.
- Decoding is stochastic: a run that fails at one `threshold_v` can succeed at
  another, so re-run before concluding that a matrix is out of reach.

## Caveats that cost us time

- Field arguments are encoded (`0/1/2/3`) in the pruning decoders but natural
  (`4/16/256`) in some harnesses. Check each binary's `Usage:` line.
- BP success means the decoded codeword is the all-zero word, *not* that the
  recovered vector equals the error vector.
- `genU`'s threshold parameter is the **maximum column intersection**, not the
  decoder's pruning threshold.
- Matrix files are fixture data; the same name can appear in more than one
  directory with different parameters. Use the `matrices/` copy of the decoder
  you are running.

## Bibliography

- L. Csanky, *Fast parallel matrix inversion algorithms*, SIAM J. Comput. 5
  (1976), 618-623.
- A. Schönhage, *Fast parallel computation of characteristic polynomials by
  Leverrier's power sum method adapted to fields of finite characteristic*,
  ISSAC 1986, 124-128.
- A. Borodin, J. von zur Gathen, J. E. Hopcroft, *Fast parallel matrix and GCD
  computations*, Information and Control 52 (1982), 241-256.
- X.-Y. Hu, E. Eleftheriou, D.-M. Arnold, *Regular and irregular progressive
  edge-growth Tanner graphs*, IEEE Trans. Inf. Theory 51 (2005), 386-398.
- D. J. C. MacKay, LDPC code database,
  <http://www.inference.org.uk/mackay/codes/data.html>.

## License

No license file yet; all rights reserved by the authors until one is added.
