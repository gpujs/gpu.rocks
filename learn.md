# Learn GPGPU in your browser — GPU.js Learn

A free hands-on GPGPU course built on gpu.js: write real kernels in your browser, run them on your own GPU, and learn ideas that transfer to CUDA and WebGPU.

## GPGPU 101

From zero to your first thousand threads

- [Hello, Kernel](https://gpu.rocks/learn/hello-kernel-f1399353.md) — What a kernel is, what a thread is, and why this.thread.x replaces your for-loop. (5 tasks)
- [Data In, Data Out](https://gpu.rocks/learn/data-in-data-out-42b68d01.md) — Feeding arrays and images into kernels, shaping 1D/2D/3D output, and reading results back. (6 tasks)
- [Pipelines & Textures](https://gpu.rocks/learn/pipelines-and-textures-9f4aeaa5.md) — Chaining kernels so data stays on the GPU — the single biggest real-world speedup. (5 tasks)
- [Measuring Speed Honestly](https://gpu.rocks/learn/measuring-speed-honestly-b9188894.md) — Warm-up, transfer costs, and precision — when the GPU wins, and when the CPU quietly beats it. (4 tasks)

## Parallel Primitives

The handful of patterns everything else is built from

- [Thinking in Parallel](https://gpu.rocks/learn/thinking-in-parallel-c3876efb.md) — Map and gather patterns, why kernels write only their own cell, and how to design around it. (6 tasks)
- [Reductions](https://gpu.rocks/learn/reductions-3dadc130.md) — Sum, min, max and mean over millions of values — the ladder pattern every platform uses. (6 tasks)
- [Prefix Sums (Scan)](https://gpu.rocks/learn/prefix-sum-351cfa41.md) — Running totals in parallel — the doubling ladder, exclusive scans, and the offsets every variable-sized output depends on. (6 tasks)
- [Stream Compaction](https://gpu.rocks/learn/stream-compaction-0aed2e43.md) — Filtering on a GPU: flag what survives, scan to find out where it lands, then gather it into a packed array. (5 tasks)
- [Histograms & Binning](https://gpu.rocks/learn/histograms-and-binning-dfb254f4.md) — Counting values into bins with no atomics — the scatter that has to become a gather. (5 tasks)
- [Top-K Selection](https://gpu.rocks/learn/top-k-selection-1ba56df3.md) — The ten largest of a million values: rank by counting, gather the winners, or bisect for a cutoff — and when each one wins. (5 tasks)
- [Bitonic Sort](https://gpu.rocks/learn/bitonic-sort-84e0728e.md) — More comparisons than quicksort, and far faster on a GPU — because the whole comparison schedule is fixed before the data arrives. (5 tasks)
- [Radix Sort](https://gpu.rocks/learn/radix-sort-fd3ff796.md) — A histogram, a scan and a gather assembled into the sort production GPU libraries actually run. (6 tasks)

## Math & Simulation

Heavy math, thousands of threads at once

- [Matrix Multiply](https://gpu.rocks/learn/matrix-multiply-972e080b.md) — The canonical GPGPU workload: from naive triple loop to a kernel that scales. (5 tasks)
- [Monte Carlo Methods](https://gpu.rocks/learn/monte-carlo-methods-9ea19810.md) — Estimate π, price an option, integrate the un-integrable — with a million random samples. (4 tasks)
- [N-Body Gravity](https://gpu.rocks/learn/n-body-gravity-5de47751.md) — Every particle pulls on every other: an O(n²) problem the GPU eats for breakfast. (5 tasks)

## Computer Vision

Teaching a GPU to look at pictures, not just draw them

- [Colour Spaces](https://gpu.rocks/learn/colour-spaces-8d79c6af.md) — Leaving RGB: perceptual luminance, the hue wheel, and why a channel that wraps breaks ordinary arithmetic. (5 tasks)
- [Convolution & Filters](https://gpu.rocks/learn/convolution-and-filters-66933805.md) — Sliding-window math on signals and images: blur, sharpen, edge detection. (5 tasks)
- [Thresholding & Morphology](https://gpu.rocks/learn/thresholding-and-morphology-670eaafa.md) — Turning grey pixels into a clean binary mask: global and adaptive thresholds, then erosion and dilation as a neighbourhood min and max. (6 tasks)
- [The Canny Edge Pipeline](https://gpu.rocks/learn/canny-edges-6901c51a.md) — The edge detector every vision library ships, one kernel per stage — blur, gradient, thinning, thresholds, hysteresis — then chained with pipeline: true. (6 tasks)
- [Template Matching](https://gpu.rocks/learn/template-matching-f57b4bed.md) — Finding a patch in a picture — and why a raw difference score is fooled by a light switch. (5 tasks)
- [Optical Flow](https://gpu.rocks/learn/optical-flow-e85c6dfa.md) — Per-pixel motion between two frames: the aperture problem, a 2×2 least-squares solve per thread, and knowing when not to believe the answer. (5 tasks)
- [Video Filters](https://gpu.rocks/learn/video-filters-4d39e404.md) — Sixteen milliseconds a frame, and state that survives between them: temporal filtering, motion masks and a background model. (6 tasks)

## Signal Processing

Time in, frequency out — and the algorithm that made it practical

- [Sampling & Aliasing](https://gpu.rocks/learn/sampling-and-aliasing-ad14836c.md) — One thread per sample: build a signal, watch a tone come back as the wrong one, and rebuild what fell between. (5 tasks)
- [The DFT, Honestly](https://gpu.rocks/learn/the-dft-7b1e3f9b.md) — One thread per frequency bin, each summing over every sample — the honest O(n²) transform, complex arithmetic and all. (5 tasks)
- [The FFT Butterfly](https://gpu.rocks/learn/fft-butterfly-d4375da7.md) — Split the sum by parity and the transform collapses from n² terms to log₂n passes of a two-line butterfly — the same multi-pass gather every ladder in this course uses. (6 tasks)
- [Windowing & Spectral Leakage](https://gpu.rocks/learn/windowing-f563138d.md) — Why the same tone looks clean or filthy depending only on how many samples you took — and what a window costs to fix it. (5 tasks)
- [Filtering in the Frequency Domain](https://gpu.rocks/learn/frequency-filtering-8c225e10.md) — Convolution becomes multiplication — the trade that makes the FFT worth its complexity, plus the ringing, the wrap-around and the cross terms it hides. (5 tasks)
- [Spectrograms](https://gpu.rocks/learn/spectrograms-9ecd2295.md) — Slide a window along a signal and transform every slice — a picture of frequency over time, one thread per (frame, bin). (5 tasks)
- [Autocorrelation & Pitch](https://gpu.rocks/learn/autocorrelation-b159433f.md) — Finding the note in a sound by asking how well it resembles itself, shifted — and the octave error that catches every naive detector once. (5 tasks)

## Computational Graphics

Pictures computed, not drawn

- [Pixels from Scratch](https://gpu.rocks/learn/pixels-from-scratch-d2869039.md) — Graphical kernels and this.color(): gradients, patterns and plots, one thread per pixel. (4 tasks)
- [Escape-Time Fractals](https://gpu.rocks/learn/escape-time-fractals-0de4764c.md) — Mandelbrot and Julia sets with smooth coloring — infinite detail from a ten-line kernel. (5 tasks)
- [Cellular Automata](https://gpu.rocks/learn/cellular-automata-407c2c34.md) — Conway's Life and friends: feed a kernel's output back in and watch worlds evolve. (5 tasks)
- [Reaction–Diffusion](https://gpu.rocks/learn/reaction-diffusion-bc3d0b34.md) — Two chemicals, two equations, and suddenly: coral, fingerprints, leopard spots. (4 tasks)
- [Ray-Marched Metaballs](https://gpu.rocks/learn/ray-marched-metaballs-8b1282bd.md) — Signed distance fields and soft shadows — a real-time 3D scene with no triangles at all. (6 tasks)

---

Interactive version: https://gpu.rocks/learn
