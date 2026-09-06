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
  [CPU], [13th Gen Intel Core i5-13420H, topology is 4 P-cores + 4 E-cores],
  [L1d cache], [48 KiB per P-core ],
  [L2 cache], [1.25 MiB per P-core ],
  [L3 cache], [12 MiB],
  [Memory], [16 GiB],
  [Vector ISA], [SSE, AVX, AVX2, FMA — *no AVX-512*],
)


= Task 1: 2D convolution

Baseline: `conv_naive`


== Task 1A: Loop unrolling and reordering


== Task 1B: Tiling
The size of L1d cache in P-core is 48 KiB, and its size in E-Core is 32 KiB, we have used the performant cores through out our assignment. \ 
For the naive variant of Tiling we got an MPKI of 1.099

#restable(
  (auto, auto),
  ([Metric], [Count]),
  cellalign: left + horizon,
  [Instructions], [4,377,401,359],
  [Cycles], [920,331,028],
  [L1d cache loads], [894,269,791],
  [L1d cache load misses], [4,812,940],
  [MPKI], [1.099],
)
#v(-0.6em)
#align(center)[#text(size: 9pt, fill: muted)[Table: Performance statistics of naive 2D convolution.]]

Here follows our findings on the effect of tile sizes for different size matrices, we found that very narrow rectangular tiles gave the best and nearly identical results. For the graded case of 2048 x 2048 we noticed that 2 x 512 performed the best in terms of MPKI.

#restable(
  (auto, auto, auto, auto, auto),
  ([Tile size ($H × W$)], [$N = 256$], [$N = 512$], [$N = 1024$], [$N = 2048$]),
  [2 × 512],  [1.620], [1.431], [1.200], [1.246],
  [3 × 512],  [1.619], [1.437], [1.274], [1.430],
  [5 × 512],  [1.623], [1.441], [1.326], [1.392],
  [6 × 512],  [1.635], [1.422], [1.786], [1.584],
  [64 × 512], [6.982], [7.274], [7.520], [7.499],
  [64 × 64],  [5.905], [7.662], [8.229], [8.317],
)
#v(-0.6em)
#align(center)[#text(size: 9pt, fill: muted)[Table: MPKI of Tiled code over various matrix sizes under different tile sizes ]]


#restable(
  (auto, auto, auto, auto, auto),
  ([Tile size ($H × W$)], [$N = 256$], [$N = 512$], [$N = 1024$], [$N = 2048$]),
  [2 × 512],  [1.23×], [1.43×], [1.24×], [1.31×],
  [3 × 512],  [1.85×], [1.40×], [1.33×], [1.19×],
  [5 × 512],  [1.68×], [1.54×], [1.28×], [1.23×],
  [6 × 512],  [1.72×], [1.26×], [1.32×], [1.23×],
  [64 × 512], [1.44×], [1.38×], [1.17×], [1.08×],
  [64 × 64],  [1.10×], [1.03×], [1.15×], [1.00×],
)
#v(-0.6em)
#align(center)[#text(size: 9pt, fill: muted)[Table: Speedup of Tiled code over naive code at different tile and matrix sizes.]]




#plot("figures/task1b_mpki_vs_size.svg",
  [L1-D MPKI versus image size for each tile size. Note the broken axis: the
   $"TILE"_H = 64$ controls sit roughly 5× above every candidate that fits in the
   48 KiB L1.])

#plot("figures/task1b_speedup_vs_size.svg",
  [Speedup over `conv_naive` versus image size for each tile size.])


== Task 1C: SIMD
The number of Instructions in naive 2D convolution is 4,374,266,223 \ 
The number of Instructions in SIMD version for 2048 x 2048 matrix with Kernel of dimension 3 x 3 is 4,906,638,833 \
For this particular configuration of inputs we got a speedup of 7.93 \

#restable(
  (auto, auto, auto, auto, auto, auto),
  ([Matrix size], [$K$], [Speedup], [Instructions], [L1d misses], [MPKI]),
  [256],  [3], [8.24×], [79,908,283],     [133,260],    [1.668],
  [256],  [5], [9.59×], [168,636,071],    [139,416],    [0.827],
  [512],  [3], [6.28×], [309,934,501],    [393,169],    [1.269],
  [512],  [5], [9.02×], [664,888,948],    [466,413],    [0.701],
  [1024], [3], [7.52×], [1,229,221,723],  [1,550,748],  [1.262],
  [1024], [5], [9.16×], [2,648,557,208],  [1,928,404],  [0.728],
  [2048], [3], [7.93×], [4,906,638,833],  [6,841,282],  [1.394],
  [2048], [5], [9.10×], [10,585,776,892], [17,458,830], [1.649],
)
#v(-0.6em)
#align(center)[#text(size: 9pt, fill: muted)[Table: For 256 Bit SIMD]]

#restable(
  (auto, auto, auto, auto, auto, auto),
  ([Matrix size], [$K$], [Speedup], [Instructions], [L1d misses], [MPKI]),
  [256],  [3], [5.48×], [87,724,922],     [121,413],    [1.384],
  [256],  [5], [4.86×], [185,825,985],    [132,613],    [0.714],
  [512],  [3], [5.62×], [340,930,560],    [415,973],    [1.220],
  [512],  [5], [5.02×], [733,174,587],    [468,340],    [0.639],
  [1024], [3], [4.34×], [1,354,016,801],  [1,640,039],  [1.211],
  [1024], [5], [4.72×], [2,922,516,815],  [1,958,168],  [0.670],
  [2048], [3], [4.24×], [5,403,258,689],  [7,360,450],  [1.362],
  [2048], [5], [4.81×], [11,679,133,866], [16,389,588], [1.403],
)
#v(-0.6em)
#align(center)[#text(size: 9pt, fill: muted)[Table: For 128 Bit SIMD]]

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

#align(center)[
  #table(
    columns: (auto, auto, auto, auto, auto, auto, auto),
    align: center + horizon,
    stroke: 0.5pt,
    inset: 8pt,

    table.header(
      table.cell(rowspan: 3, fill: luma(240))[*Matrix Size*],
      table.cell(colspan: 4, fill: luma(240))[*SIMD*],
      table.cell(colspan: 2, rowspan: 2, fill: luma(240))[*Without SIMD*],

      table.cell(colspan: 2, fill: luma(240))[*SIMD width 128*],
      table.cell(colspan: 2, fill: luma(240))[*SIMD width 256*],

      [*Instructions (M)*], [*Execution Time*],
      [*Instructions (M)*], [*Execution Time*],
      [*Instructions (M)*], [*Execution Time*],
    ),

    [256],
    [659.68],     [0.765 ms],
    [645.32],     [0.478 ms],
    [620.38],     [8.07 ms],

    [512],
    [5,194.07],   [7.714 ms],
    [5,057.22],   [5.724 ms],
    [4,888.86],   [55.58 ms],

    [1024],
    [41,170.41],  [76.117 ms],
    [40,104.55],  [59.397 ms],
    [38,886.06],  [487.66 ms],

    [2048],
    [328,261.32], [1291.929 ms],
    [320,231.07], [1103.235 ms],
    [310,339.38], [4838.61 ms],
  )
]
#v(-0.6em)
#align(center)[#text(size: 9pt, fill: muted)[Table: Matrix Sizes vs Instruction size & Execution time for SIMD and non SIMD]]

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

