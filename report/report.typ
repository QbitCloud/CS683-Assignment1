#import "template.typ": *

#show: report.with(
  course: "CS683 — Advanced Computer Architecture",
  title: "Hardware-Conscious Performance Engineering",
  subtitle: "Programming Assignment 1: 2D Convolution and SGEMM",
  authors: (
    (name: "TODO Name", roll: "TODO Roll No."),
  ),
  date: "September 2026",
)

== Experimental setup

#restable(
  (auto, 1fr),
  ([Component], [Configuration]),
  cellalign: left + horizon,
  [CPU], [13th Gen Intel Core i5-13420H  \
          4 P-cores + 4 E-cores],
  [L1d cache], [48 KiB per P-core ],
  [L2 cache], [1.25 MiB per P-core ],
  [L3 cache], [12 MiB],
  [Memory], [16 GiB],
  [Vector ISA], [SSE4.2, AVX, AVX2, FMA, AVX-VNNI — *no AVX-512*],
)


= Task 1: 2D convolution

Baseline: `conv_naive`


== Task 1A: Loop unrolling and reordering


== Task 1B: Tiling


== Task 1C: SIMD


== Task 1D: Sabka saath sabka vikaas


== Results

#restable(
  (auto, auto, auto, auto),
  ([Stage], [Time (ms)], [GFLOP/s], [Speedup vs naive]),
  [naive], [], [], [1.00×],
  [reorder], [], [], [],
  [unroll], [], [], [],
  [tile], [], [], [],
  [simd], [], [], [],
  [optimized], [], [], [],
)
#v(-0.6em)
#align(center)[#text(size: 9pt, fill: muted)[Table: 2048 × 2048 image, K = 3.]]

#plotsoon[Speedup over `conv_naive` versus image dimension, at fixed $K = 3$.]

#plotsoon[Speedup over `conv_naive` versus kernel size $K$, at a fixed image size.]

== Discussion


= Task 2: Matrix multiplication (SGEMM)

Baseline: `matmul_naive`


== Task 2A: Software prefetching


== Task 2B: SIMD (Single Instructor Multiple Deadlines)


== Task 2C: Software prefetching + SIMD


== Results

#restable(
  (auto, auto, auto, auto),
  ([Stage], [Time (ms)], [GFLOP/s], [Speedup vs naive]),
  [naive], [], [], [1.00×],
  [simd], [], [], [],
  [prefetch], [], [], [],
  [optimized], [], [], [],
)
#v(-0.6em)
#align(center)[#text(size: 9pt, fill: muted)[Table: $M = N = K = 1024$.]]

#plotsoon[Speedup over `matmul_naive` versus matrix size, $M = N = K$ from 128 to 2048.]

#plotsoon[Speedup of `matmul_prefetch` versus prefetch distance.]

#plotsoon[Speedup of `matmul_prefetch` for the `_MM_HINT_T0` / `T1` / `T2` / `NTA` locality hints.]

#plotsoon[Speedup versus SIMD width: 128-bit SSE, 256-bit AVX2, 512-bit AVX-512.]

== Integration with llama.cpp


== Discussion


= Conclusion

