// Module 1.2 — Data In, Data Out (the golden content module).
//
// Six tasks: 1D arrays in → 2D output shapes → the mockup's graphical
// grayscale → reading results back into JS → images as plain data → a
// two-kernel pipeline that combines all of it.
//
// Kernel-authoring rules (contract): no closures inside kernel functions,
// only numbers / nested number arrays as inputs, this.thread.* for indexing,
// image convention image[y][x] = [r, g, b, a] with channels 0–1, graphical
// kernels use this.color(). Every task passes in CPU mode.

import { ARRAY_LAYOUT } from '../layoutNote.js';
import { plainToImageData, quantizePixel } from '../../engine/utils.js';

const LUM = [0.299, 0.587, 0.114];

function luminanceOf(pixel) {
  return LUM[0] * pixel[0] + LUM[1] * pixel[1] + LUM[2] * pixel[2];
}

// Deterministic 64-value input for task 1 (shared by inputs() and tests).
function makeSignal(utils, seed = 4201) {
  const rand = utils.seededRandom(seed);
  const data = new Array(64);
  for (let i = 0; i < 64; i++) data[i] = Math.round(rand() * 1000) / 100; // 0–10, 2 dp
  return data;
}

// Swapping this.thread.x and this.thread.y reads the transpose of the image —
// the single most common mistake in these tasks, and invisible to tests that
// only check greyness or averages. When the value under test matches the
// transposed pixel, say so instead of just reporting two numbers.
// `last` is the highest index on each axis; the pixel under test is at column
// `col` of the first returned row, which is row 0 or row `last` depending on
// the backend's row order — so check the transpose of both candidates.
function transposeHint(got, img, last, col) {
  const swapped = [img[col][0], img[col][last]];
  const hit = swapped.some(pixel => Math.abs(got - luminanceOf(pixel)) <= 2 / 255);
  return hit
    ? `that value is the luminance of the transposed pixel — looks like this.thread.x and ` +
      `this.thread.y are swapped. Rows come first: image[this.thread.y][this.thread.x].`
    : null;
}

// Numeric companion to transposeHint, for results that come back as plain 2D
// arrays: there the cell under test is known exactly, so a cell holding the
// value that belongs to cell [x][y] is that same swapped-index read. Cells on
// the diagonal (y === x) are their own transpose and can never show it, which
// is why the case lists below also probe off-diagonal cells.
function transposeCellHint(got, transposed, eps, y, x) {
  return Math.abs(got - transposed) <= eps
    ? `that is the value for cell [${x}][${y}] — this.thread.x and this.thread.y are ` +
      `swapped. Rows come first: image[this.thread.y][this.thread.x]`
    : null;
}

// Constant-color image helper for orientation-independent pixel tests. Same
// shape as the task's own input: an ImageData whose .plain/.at(x, y) answer
// image[y][x] = [r, g, b, a], with the pixel quantized to what 8-bit channels
// can actually hold (so expectations read off .at() are exact). A kernel built
// with an ImageData must never be re-invoked with a nested array — on WebGL2
// that silently paints wrong pixels — so every image a test passes is one.
function constantImage(size, pixel) {
  const row = new Array(size).fill(quantizePixel(pixel));
  return plainToImageData(new Array(size).fill(row));
}

// ---- near-miss diagnosis --------------------------------------------------
//
// The transpose hints above are one instance of a general idea: when a failing
// value is exactly what some specific mistake would produce, name that mistake
// instead of reporting two numbers. A probe pairs such a value with its
// sentence; diagnose() speaks only when the observed value matches a probe
// within the test's own tolerance AND the correct value does not — so an index
// where two candidates coincide (element 2, where x² and 2x are both 4) stays
// silent, as do observations matching probes that disagree with each other.
// A wrong diagnosis is worse than none.
function diagnose(got, expected, eps, probes) {
  const hits = probes
    .filter(p => Math.abs(got - p[0]) <= eps && Math.abs(expected - p[0]) > eps)
    .map(p => p[1]);
  return hits.length && hits.every(m => m === hits[0]) ? hits[0] : null;
}

// Where a candidate is computed from the thread index rather than from data,
// one matching cell is weak evidence — `x * 2` also equals `x + 1` at x = 1.
// These two forms therefore demand that a probe predict EVERY element (and
// differ from the right answer somewhere) before it may speak. Probe values are
// functions of the index; a missing cell makes the comparison NaN, which fails.
function diagnoseAll(count, got, expected, eps, probes) {
  const hits = probes
    .filter(([value]) => {
      let differs = false;
      for (let i = 0; i < count; i++) {
        if (!(Math.abs(got(i) - value(i)) <= eps)) return false;
        if (Math.abs(expected(i) - value(i)) > eps) differs = true;
      }
      return differs;
    })
    .map(p => p[1]);
  return hits.length && hits.every(m => m === hits[0]) ? hits[0] : null;
}

function diagnoseGrid(size, out, expected, eps, probes) {
  const hits = probes
    .filter(([value]) => {
      let differs = false;
      for (let y = 0; y < size; y++) {
        for (let x = 0; x < size; x++) {
          const c = value(y, x);
          if (!(out[y] && Math.abs(out[y][x] - c) <= eps)) return false;
          if (Math.abs(expected(y, x) - c) > eps) differs = true;
        }
      }
      return differs;
    })
    .map(p => p[1]);
  return hits.length && hits.every(m => m === hits[0]) ? hits[0] : null;
}

// Task 1: the doubling applied to the wrong thing, or not at all.
function doubleHint(out, arr) {
  return diagnoseAll(64, i => out[i], i => arr[i] * 2, 1e-3, [
    [i => arr[i], 'every cell is the element itself — the doubling never happened'],
    [i => 2 * i, 'every cell is twice the thread index, not twice the element — index the array with it: data[this.thread.x]'],
  ]);
}

// Task 2: the table is 1-based, so both coordinates need their + 1. All three
// ways of forgetting one collapse to 0 in cell [0][0] — sharing a single
// message is what lets the diagnosis speak there instead of cancelling itself.
function tableHint(out) {
  const missing = 'a + 1 is missing — this.thread.x and this.thread.y both start at 0, so cell [y][x] holds (x + 1) * (y + 1)';
  return diagnoseGrid(16, out, (y, x) => (x + 1) * (y + 1), 1e-3, [
    [(y, x) => x * y, missing],
    [(y, x) => (x + 1) * y, missing],
    [(y, x) => x * (y + 1), missing],
    [(y, x) => (x + 1) + (y + 1),
      'the coordinates were added, not multiplied — the cell is (x + 1) * (y + 1)'],
  ]);
}

// Task 4: the index and its double both look like x² somewhere (2 · 2 = 2², and
// 2x = x² at x = 2), so these have to hold across all 128 cells before either
// one is named.
function squareHint(out) {
  return diagnoseAll(128, i => out[i], i => i * i, 1e-2, [
    [i => i, 'you returned the thread index itself, not its square — every cell is exactly this.thread.x'],
    [i => 2 * i, 'every cell is twice the index, not the index squared — x * x, not x * 2'],
  ]);
}

// Tasks 5 and 6: the two grayscale recipes are easy to swap for each other,
// and a mean is easy to leave un-divided.
function meanProbes(pixel) {
  return [
    [luminanceOf(pixel), 'that is the weighted luminance — this map wants the plain average (r + g + b) / 3'],
    [pixel[0] + pixel[1] + pixel[2], 'the three channels were summed but never divided by 3'],
    [(pixel[0] + pixel[1] + pixel[2] + pixel[3]) / 4, 'alpha crept into the average — only r, g and b belong in it'],
  ];
}

function luminanceProbes(pixel) {
  return [
    [(pixel[0] + pixel[1] + pixel[2]) / 3, 'that is the plain channel average — luminance weights the channels 0.299 R + 0.587 G + 0.114 B'],
    [LUM[2] * pixel[0] + LUM[1] * pixel[1] + LUM[0] * pixel[2], 'the weights are in the wrong order — 0.299 belongs on red and 0.114 on blue'],
  ];
}

// Task 3 reads pixels back from a canvas whose row order is unknown, so a probe
// there has to be distinguishable from the correct luminance under BOTH
// orientations before it may speak.
function canvasHint(got, probes, correctA, correctB) {
  const eps = 2 / 255;
  const hits = probes
    .filter(p => Math.abs(got - p[0]) <= eps &&
      Math.abs(correctA - p[0]) > eps && Math.abs(correctB - p[0]) > eps)
    .map(p => p[1]);
  return hits.length && hits.every(m => m === hits[0]) ? hits[0] : null;
}

// Task 3: 0–255 channels handed to this.color() clamp every pixel to white.
function saturatedHint(got, correctA, correctB) {
  return got >= 254 / 255 && Math.max(correctA, correctB) < 0.9
    ? 'that pixel is clamped to full white — this.color() takes 0–1 channels and the image already is 0–1, so scaling by 255 saturates everything'
    : null;
}

export default {
  id: '1-2',
  track: 1,
  title: 'Data In, Data Out',
  blurb: 'Feeding arrays and images into kernels, shaping 1D/2D/3D output, and reading results back.',
  tasks: [
    // ------------------------------------------------------------------ 1
    {
      slug: 'pass-an-array',
      title: 'Pass an Array In',
      intro: `<p>Kernels don't reach out and grab data — data is <strong>handed to them</strong>
        as arguments, and every thread sees the same arguments. What differs between threads is
        exactly one thing: <code>this.thread.x</code>, the index of the output cell this thread owns.</p>
        <p>Here <code>data</code> is a 64-number array. The kernel below runs 64 times — once per
        output cell — and each run should pick out <em>its own</em> element.</p>`,
      goal: `<strong>Goal:</strong> make the kernel return <strong>double</strong> the element of
        <code>data</code> that belongs to this thread.`,
      requirements: [
        'Pass <code>data</code> into the kernel as an argument (already wired up)',
        'Index it with <code>this.thread.x</code> — no loops over the array',
        'Return the element multiplied by <code>2</code>',
      ],
      hints: [
        {
          title: 'Hint 1 — which element is mine?',
          body: `<p>With <code>output: [64]</code> there are 64 threads, numbered
            <code>this.thread.x</code> = 0…63. Thread 7 should read <code>data[7]</code>.</p>`,
        },
        {
          title: 'Hint 2 — the one-liner',
          body: `<p>The whole kernel body is a single statement:
            <code>return data[this.thread.x] * 2;</code></p>`,
        },
      ],
      transfer: `Arguments-in, index-by-thread-id is the universal GPGPU calling convention:
        CUDA kernels get device pointers plus <code>threadIdx</code>, WebGPU compute shaders get
        bound buffers plus <code>global_invocation_id</code>. Same shape, different spelling.`,
      starterCode: `// A kernel runs once per output cell — 64 cells here, 64 threads.
const gpu = new GPU({ mode });

const double = gpu.createKernel(function (data) {
  // TODO: return double the value that belongs to THIS thread.
  // Which element is yours? this.thread.x knows.
  return 0;
}, { output: [64] });

const result = double(data);
console.log(result);
`,
      solutionCode: `// A kernel runs once per output cell — 64 cells here, 64 threads.
const gpu = new GPU({ mode });

const double = gpu.createKernel(function (data) {
  return data[this.thread.x] * 2;
}, { output: [64] });

const result = double(data);
console.log(result);
`,
      inputs: utils => ({ data: makeSignal(utils) }),
      publicTests: [
        {
          name: 'kernel returns 64 values — one per thread',
          run: async ctx => {
            ctx.assert(ctx.kernels.length >= 1, 'no kernel was created — call gpu.createKernel()');
            const out = ctx.kernel(makeSignal(ctx.utils));
            ctx.assert(out && out.length === 64, `expected 64 output values, got ${out && out.length}`);
          },
        },
        {
          name: 'every element is doubled: <code>data[i] * 2</code>',
          run: async ctx => {
            const arr = new Array(64);
            for (let i = 0; i < 64; i++) arr[i] = i * 1.5 - 10;
            const out = ctx.kernel(arr);
            const hint = doubleHint(out, arr);
            for (let i = 0; i < 64; i++) {
              ctx.assertClose(out[i], arr[i] * 2, 1e-3, hint || `element ${i}`);
            }
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            const data = makeSignal(ctx.utils, 777);
            const out = ctx.kernel(data);
            ctx.assert(out.length === 64, 'expected 64 output values');
            const hint = doubleHint(out, data);
            for (let i = 0; i < 64; i++) {
              ctx.assertClose(out[i], data[i] * 2, 1e-3, hint || `element ${i}`);
            }
          },
        },
      ],
    },

    // ------------------------------------------------------------------ 2
    {
      slug: 'output-shapes',
      title: 'Shape the Output: 2D',
      intro: `<p><code>output</code> is not just a size — it's a <strong>shape</strong>.
        <code>output: [16]</code> launches a line of 16 threads; <code>output: [16, 16]</code>
        launches a 16×16 <em>grid</em> of 256 threads, and each one gets two coordinates:
        <code>this.thread.x</code> (column) and <code>this.thread.y</code> (row).</p>
        <p>The result comes back with the same shape: a 2D kernel returns an array of rows,
        indexed <code>result[y][x]</code>.</p>`,
      goal: `<strong>Goal:</strong> turn the kernel into a 16×16 grid that computes a
        multiplication table — cell <code>[y][x]</code> holds <code>(x + 1) * (y + 1)</code>.`,
      requirements: [
        'Change <code>output</code> to a 16×16 grid: <code>[16, 16]</code>',
        'Use both <code>this.thread.x</code> and <code>this.thread.y</code>',
        'Return <code>(x + 1) * (y + 1)</code> so row 1 counts 1…16, row 2 counts 2…32, …',
      ],
      hints: [
        {
          title: 'Hint 1 — reading the shape',
          body: `<p><code>output: [width, height]</code> — x runs over <code>width</code>,
            y over <code>height</code>. The returned value lands in <code>result[y][x]</code>.</p>`,
        },
      ],
      transfer: `2D and 3D launch grids are first-class everywhere: CUDA's <code>dim3</code>
        grid/block sizes, WebGPU's <code>workgroup_size</code> and dispatch dimensions. Choosing
        the launch shape to match the output shape is the same design move on every platform.`,
      starterCode: `// output: [width, height] — gpu.js hands you a whole grid of threads.
const gpu = new GPU({ mode });

const table = gpu.createKernel(function () {
  // TODO: use BOTH this.thread.x and this.thread.y
  // and return (x + 1) * (y + 1).
  return this.thread.x + 1;
}, {
  // TODO: make this a 16×16 grid, not a 16-cell line
  output: [16],
});

const result = table();
console.log('rows:', result.length);
console.log('row 0:', result[0]);
`,
      solutionCode: `// output: [width, height] — gpu.js hands you a whole grid of threads.
const gpu = new GPU({ mode });

const table = gpu.createKernel(function () {
  return (this.thread.x + 1) * (this.thread.y + 1);
}, {
  output: [16, 16],
});

const result = table();
console.log('rows:', result.length);
console.log('row 0:', result[0]);
`,
      publicTests: [
        {
          name: 'result is a 16×16 grid — 16 rows of 16 values',
          run: async ctx => {
            ctx.assert(ctx.kernels.length >= 1, 'no kernel was created — call gpu.createKernel()');
            const out = ctx.kernel();
            ctx.assert(out && out.length === 16, `expected 16 rows, got ${out && out.length}`);
            ctx.assert(
              out[0] && typeof out[0] !== 'number' && out[0].length === 16,
              'each row should hold 16 values — is your output still 1D?'
            );
          },
        },
        {
          name: 'cell [y][x] equals <code>(x + 1) * (y + 1)</code>',
          run: async ctx => {
            const out = ctx.kernel();
            const cases = [[0, 0, 1], [2, 3, 12], [7, 0, 8], [0, 7, 8], [15, 15, 256]];
            const hint = tableHint(out);
            for (const [y, x, expected] of cases) {
              ctx.assertClose(out[y][x], expected, 1e-3, hint || `cell [${y}][${x}]`);
            }
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            const out = ctx.kernel();
            const hint = tableHint(out);
            for (let y = 0; y < 16; y++) {
              for (let x = 0; x < 16; x++) {
                ctx.assertClose(out[y][x], (x + 1) * (y + 1), 1e-3, hint || `cell [${y}][${x}]`);
              }
            }
          },
        },
      ],
    },

    // ------------------------------------------------------------------ 3
    // The mockup's task, verbatim: 'Grayscale, the GPU way' (Task 3 of 6).
    {
      slug: 'grayscale',
      title: 'Grayscale, the GPU way',
      intro: `<p>On the CPU you'd loop over 262,144 pixels one by one. On the GPU, every pixel gets
        <strong>its own thread</strong> — the kernel body runs once per pixel, all at the same time.</p>
        ${ARRAY_LAYOUT}`,
      goal: `<strong>Goal:</strong> write a graphical kernel that converts <code>image</code> to
        grayscale using perceptual luminance.`,
      requirements: [
        'Create the kernel with <code>graphical: true</code> and <code>output: [512, 512]</code>',
        'Read the pixel for <em>this</em> thread from <code>image</code>',
        'Weight the channels <code>0.299 R + 0.587 G + 0.114 B</code>',
        'Write the result with <code>this.color()</code>',
      ],
      hints: [
        {
          title: 'Hint 1 — which pixel is mine?',
          body: `<p>Inside a kernel, <code>this.thread.x</code> and <code>this.thread.y</code> tell you which
            output cell this thread owns. Use them to index into <code>image</code>.</p>`,
        },
        {
          title: 'Hint 2 — reading a pixel',
          body: `<p><code>image[this.thread.y][this.thread.x]</code> gives you an <code>[r, g, b, a]</code>
            array with channels in the 0–1 range.</p>`,
        },
      ],
      transfer: `This is exactly a fragment shader in WebGPU/Metal, or a 2D thread block in CUDA and
        ROCm — one thread per output element.`,
      starterCode: `// One thread per pixel. No loops over pixels — ever.
const gpu = new GPU({ mode });

const grayscale = gpu.createKernel(function (image) {
  // TODO: read this thread's pixel from image, weight the channels
  // 0.299 R + 0.587 G + 0.114 B, and write it with this.color()
  this.color(1, 0, 1, 1);
}, {
  output: [512, 512],
  graphical: true,
});

grayscale(inputImage);
render(grayscale.canvas);
`,
      solutionCode: `// One thread per pixel. No loops over pixels — ever.
const gpu = new GPU({ mode });

const grayscale = gpu.createKernel(function (image) {
  const pixel = image[this.thread.y][this.thread.x];
  const l = 0.299 * pixel[0]
          + 0.587 * pixel[1]
          + 0.114 * pixel[2];
  this.color(l, l, l, 1);
}, {
  output: [512, 512],
  graphical: true,
});

grayscale(inputImage);
render(grayscale.canvas);
`,
      inputs: utils => ({ inputImage: utils.makeTestImage(512) }),
      publicTests: [
        {
          name: 'returns a graphical canvas of size <code>512×512</code>',
          run: async ctx => {
            ctx.assert(ctx.kernels.length >= 1, 'no kernel was created — call gpu.createKernel()');
            ctx.assert(ctx.canvas, 'no canvas — is the kernel graphical: true, and did you call render()?');
            ctx.assert(
              ctx.canvas.width === 512 && ctx.canvas.height === 512,
              `expected a 512×512 canvas, got ${ctx.canvas.width}×${ctx.canvas.height}`
            );
            const pixels = ctx.getPixels();
            ctx.assert(pixels.length === 512 * 512 * 4, 'pixel buffer should hold 512×512 RGBA values');
          },
        },
        {
          name: 'pixel (0,0) matches <code>0.299r + 0.587g + 0.114b</code> within ±1/255',
          run: async ctx => {
            const img = ctx.utils.makeTestImage(512).plain;
            const pixels = ctx.getPixels();
            const got = pixels[0] / 255;
            // getPixels row order can be top-down or bottom-up depending on
            // backend — accept the correct luminance for either orientation.
            const a = luminanceOf(img[0][0]);
            const b = luminanceOf(img[511][0]);
            const ok = Math.abs(got - a) <= 2 / 255 || Math.abs(got - b) <= 2 / 255;
            const recipe = canvasHint(
              got, [...luminanceProbes(img[0][0]), ...luminanceProbes(img[511][0])], a, b
            );
            ctx.assert(ok, transposeHint(got, img, 511, 0) || saturatedHint(got, a, b) || recipe ||
              `corner pixel should be its luminance (got ${got.toFixed(3)}, expected ≈${a.toFixed(3)})`);
          },
        },
        {
          name: 'output is monochrome — <code>r == g == b</code> for sampled pixels',
          run: async ctx => {
            const pixels = ctx.getPixels();
            for (let i = 0; i < pixels.length; i += 997 * 4) {
              const r = pixels[i];
              const g = pixels[i + 1];
              const b = pixels[i + 2];
              ctx.assert(
                Math.abs(r - g) <= 1 && Math.abs(g - b) <= 1,
                `pixel at byte ${i} is not gray: rgb(${r}, ${g}, ${b})`
              );
            }
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            // Constant-color image → every output pixel must be its luminance.
            const image = constantImage(512, [0.2, 0.4, 0.6, 1]);
            const expected = luminanceOf(image.at(0, 0)) * 255;
            ctx.kernel(image);
            const pixels = ctx.getPixels();
            for (let i = 0; i < pixels.length; i += 4999 * 4) {
              ctx.assertClose(pixels[i], expected, 2, `red at byte ${i}`);
              ctx.assertClose(pixels[i + 1], expected, 2, `green at byte ${i}`);
              ctx.assertClose(pixels[i + 2], expected, 2, `blue at byte ${i}`);
            }
          },
        },
        {
          name: 'private test #2',
          run: async ctx => {
            // Mean output luminance must match the input's mean luminance —
            // independent of row order.
            const img = ctx.utils.makeTestImage(512);
            ctx.kernel(img);
            const pixels = ctx.getPixels();
            let gotMean = 0;
            for (let i = 0; i < pixels.length; i += 4) gotMean += pixels[i];
            gotMean /= (pixels.length / 4) * 255;
            let expectedMean = 0;
            for (let y = 0; y < 512; y++) {
              for (let x = 0; x < 512; x++) expectedMean += luminanceOf(img.plain[y][x]);
            }
            expectedMean /= 512 * 512;
            ctx.assertClose(gotMean, expectedMean, 1.5 / 255, 'mean luminance');
          },
        },
      ],
    },

    // ------------------------------------------------------------------ 4
    {
      slug: 'read-it-back',
      title: 'Read the Results Back',
      intro: `<p>A kernel's return value doesn't stay on the GPU — invoking the kernel hands you the
        finished result as an ordinary (typed) array. From there it's plain JavaScript: loop over it,
        sum it, feed it to a chart, whatever you like.</p>
        <p>This round trip is the heartbeat of GPGPU: <strong>upload → compute in parallel →
        read back</strong>. Here the parallel part computes 128 squares; the read-back part totals them.</p>`,
      goal: `<strong>Goal:</strong> make the kernel return <code>x²</code> for each thread, then sum
        the returned array in plain JavaScript and log the total with <code>console.log</code>.`,
      requirements: [
        'Kernel returns <code>this.thread.x * this.thread.x</code> for all 128 threads',
        'Sum the returned <code>result</code> array in ordinary JavaScript — outside the kernel',
        'Log the total (it should come to <code>690880</code>)',
      ],
      hints: [
        {
          title: 'Hint 1 — what comes back?',
          body: `<p>With <code>output: [128]</code> the call <code>squares()</code> returns a
            <code>Float32Array</code> of 128 numbers. It's indexable and loopable like any array.</p>`,
        },
        {
          title: 'Hint 2 — the sum',
          body: `<p>A plain <code>for</code> loop after the kernel call:</p>
<pre><code>let total = 0;
for (let i = 0; i &lt; result.length; i++) {
  total += result[i];
}</code></pre>`,
        },
      ],
      transfer: `Read-back is never free: CUDA's <code>cudaMemcpy</code> device→host and WebGPU's
        <code>mapAsync</code> staging buffers exist for exactly this step — and minimizing round trips
        is rule one of real GPU performance (module 1.4 makes a whole meal of it).`,
      starterCode: `// Kernel output comes back to JavaScript as a typed array.
const gpu = new GPU({ mode });

const squares = gpu.createKernel(function () {
  // TODO: return this thread's index, squared
  return this.thread.x;
}, { output: [128] });

const result = squares();
console.log(result);

// TODO: total up \`result\` in plain JavaScript, then:
// console.log('sum of squares:', total);
`,
      solutionCode: `// Kernel output comes back to JavaScript as a typed array.
const gpu = new GPU({ mode });

const squares = gpu.createKernel(function () {
  return this.thread.x * this.thread.x;
}, { output: [128] });

const result = squares();
console.log(result);

let total = 0;
for (let i = 0; i < result.length; i++) total += result[i];
console.log('sum of squares:', total);
`,
      publicTests: [
        {
          name: 'kernel returns <code>x²</code> for each of 128 threads',
          run: async ctx => {
            ctx.assert(ctx.kernels.length >= 1, 'no kernel was created — call gpu.createKernel()');
            const out = ctx.kernel();
            ctx.assert(out && out.length === 128, `expected 128 values, got ${out && out.length}`);
            const hint = squareHint(out);
            for (let i = 0; i < 128; i++) {
              ctx.assertClose(out[i], i * i, 1e-2, hint || `element ${i}`);
            }
          },
        },
        {
          name: 'the total <code>690880</code> is computed and logged',
          run: async ctx => {
            const logged = ctx.logs.some(
              line => line.type === 'log' && line.text && line.text.includes('690880')
            );
            ctx.assert(
              logged,
              'log the sum with console.log — expected to see 690880 in the console output'
            );
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            const out = ctx.kernel();
            let total = 0;
            for (let i = 0; i < out.length; i++) total += out[i];
            // 8128 is 0 + 1 + … + 127: the indices themselves, unsquared.
            const hint = diagnose(total, 690880, 1, [
              [8128, 'that total is the sum of the indices themselves — the kernel is returning this.thread.x, not its square'],
              [2 * 8128, 'that total is the sum of twice each index — the kernel is doubling where it should square'],
            ]);
            ctx.assertClose(total, 690880, 1, hint || 'sum of the kernel output');
          },
        },
      ],
    },

    // ------------------------------------------------------------------ 5
    {
      slug: 'image-as-data',
      title: 'Images Are Just Arrays',
      intro: `<p>Task 3 painted pixels. But an image doesn't have to <em>stay</em> an image: in this
        course an image is a nested array — <code>photo[y][x]</code> is an <code>[r, g, b, a]</code>
        pixel with channels 0–1 — and a kernel can read it like any other array argument.</p>
        <p>Drop <code>graphical: true</code>, and the same per-pixel indexing produces
        <strong>numbers</strong> instead of colors: a measurement per pixel, ready for JavaScript.</p>
        ${ARRAY_LAYOUT}`,
      goal: `<strong>Goal:</strong> compute a 64×64 brightness map of <code>photo</code> — each cell
        the average of that pixel's red, green and blue channels.`,
      requirements: [
        'Keep the kernel numeric — no <code>graphical: true</code>, output <code>[64, 64]</code>',
        'Read this thread\'s pixel: <code>photo[this.thread.y][this.thread.x]</code>',
        'Return <code>(r + g + b) / 3</code>',
      ],
      hints: [
        {
          title: 'Hint 1 — same indexing as task 3',
          body: `<p>The pixel lookup is identical to the grayscale task — only the ending changes:
            <code>return</code> a number instead of calling <code>this.color()</code>.</p>`,
        },
        {
          title: 'Hint 2 — the average',
          body: `<pre><code>const pixel = photo[this.thread.y][this.thread.x];
return (pixel[0] + pixel[1] + pixel[2]) / 3;</code></pre>`,
        },
      ],
      transfer: `Treating an image as a data grid is how real pipelines work: computer-vision
        pre-processing, depth-map filtering, scientific imaging. In CUDA/WebGPU this is a compute
        pass sampling a texture and writing to a plain buffer.`,
      starterCode: `// An image is a nested array: photo[y][x] → [r, g, b, a], all 0–1.
const gpu = new GPU({ mode });

const brightness = gpu.createKernel(function (photo) {
  const pixel = photo[this.thread.y][this.thread.x];
  // TODO: return the average of the red, green and blue channels
  return pixel[0];
}, { output: [64, 64] });

const map = brightness(photo);
console.log('top-left brightness:', map[0][0]);
`,
      solutionCode: `// An image is a nested array: photo[y][x] → [r, g, b, a], all 0–1.
const gpu = new GPU({ mode });

const brightness = gpu.createKernel(function (photo) {
  const pixel = photo[this.thread.y][this.thread.x];
  return (pixel[0] + pixel[1] + pixel[2]) / 3;
}, { output: [64, 64] });

const map = brightness(photo);
console.log('top-left brightness:', map[0][0]);
`,
      inputs: utils => ({ photo: utils.makeTestImage(64) }),
      publicTests: [
        {
          name: 'produces a 64×64 brightness map',
          run: async ctx => {
            ctx.assert(ctx.kernels.length >= 1, 'no kernel was created — call gpu.createKernel()');
            const out = ctx.kernel(ctx.utils.makeTestImage(64));
            ctx.assert(out && out.length === 64, `expected 64 rows, got ${out && out.length}`);
            ctx.assert(out[0] && out[0].length === 64, 'each row should hold 64 values');
          },
        },
        {
          name: 'each cell averages the channels — <code>(r + g + b) / 3</code>',
          run: async ctx => {
            const img = ctx.utils.makeTestImage(64);
            const out = ctx.kernel(img);
            const plain = img.plain; // host-side view of the same pixels
            const cases = [[0, 0], [10, 3], [31, 40], [63, 63]];
            for (const [y, x] of cases) {
              const p = plain[y][x];
              const t = plain[x][y]; // the transposed cell's pixel
              const expected = (p[0] + p[1] + p[2]) / 3;
              const hint = transposeCellHint(out[y][x], (t[0] + t[1] + t[2]) / 3, 2e-3, y, x) ||
                diagnose(out[y][x], expected, 2e-3, meanProbes(p));
              ctx.assertClose(out[y][x], expected, 2e-3, hint || `cell [${y}][${x}]`);
            }
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            const img = ctx.utils.makeTestImage(64);
            const out = ctx.kernel(img);
            const plain = img.plain;
            for (let y = 0; y < 64; y++) {
              for (let x = 0; x < 64; x++) {
                const p = plain[y][x];
                const t = plain[x][y];
                const expected = (p[0] + p[1] + p[2]) / 3;
                const hint = transposeCellHint(out[y][x], (t[0] + t[1] + t[2]) / 3, 2e-3, y, x) ||
                  diagnose(out[y][x], expected, 2e-3, meanProbes(p));
                ctx.assertClose(out[y][x], expected, 2e-3, hint || `cell [${y}][${x}]`);
              }
            }
          },
        },
      ],
    },

    // ------------------------------------------------------------------ 6
    {
      slug: 'two-kernels',
      title: 'Put It Together: Two Kernels',
      intro: `<p>Everything from this module in one pipeline. Kernel one reads the
        <code>photo</code> and produces a 64×64 <strong>luminance map</strong> — pure numbers.
        That result comes back to JavaScript, and you pass it straight into kernel two, a
        <strong>graphical</strong> kernel that paints the map as a grayscale picture.</p>
        <p>Array in → numbers out → array in again → pixels out. Data flowing <em>through</em>
        kernels is the whole game (and module 1.4 shows how to keep that flow on the GPU).</p>
        ${ARRAY_LAYOUT}`,
      goal: `<strong>Goal:</strong> finish both kernels — <code>luminance</code> returns
        <code>0.299r + 0.587g + 0.114b</code> per pixel, and <code>paint</code> renders the map
        as gray pixels with <code>this.color()</code>.`,
      requirements: [
        'Numeric kernel: read <code>photo[this.thread.y][this.thread.x]</code>, return the weighted luminance',
        'Graphical kernel: read this thread\'s value from <code>map</code>',
        'Paint it gray: <code>this.color(l, l, l, 1)</code>',
        'Feed the first kernel\'s result into the second (already wired up)',
      ],
      hints: [
        {
          title: 'Hint 1 — the luminance pass',
          body: `<p>Same lookup as before, but return a number:</p>
<pre><code>return 0.299 * pixel[0] + 0.587 * pixel[1] + 0.114 * pixel[2];</code></pre>`,
        },
        {
          title: 'Hint 2 — the paint pass',
          body: `<p><code>map</code> is a plain 2D array of numbers, so</p>
<pre><code>const l = map[this.thread.y][this.thread.x];
this.color(l, l, l, 1);</code></pre>`,
        },
      ],
      transfer: `Multi-pass pipelines are the backbone of GPU work: render passes in graphics,
        kernel launch chains in CUDA, encoder passes in WebGPU. The handoff you just did through
        JavaScript is the slow version — pipelines (module 1.4) keep it on-device.`,
      starterCode: `// Kernel 1 turns the photo into numbers. Kernel 2 turns numbers into pixels.
const gpu = new GPU({ mode });

const luminance = gpu.createKernel(function (photo) {
  // TODO: return perceptual luminance — 0.299 R + 0.587 G + 0.114 B
  return 0;
}, { output: [64, 64] });

const paint = gpu.createKernel(function (map) {
  // TODO: read this thread's value from map and paint it gray
  this.color(1, 0, 1, 1);
}, { output: [64, 64], graphical: true });

const map = luminance(photo);
paint(map);
render(paint.canvas);
`,
      solutionCode: `// Kernel 1 turns the photo into numbers. Kernel 2 turns numbers into pixels.
const gpu = new GPU({ mode });

const luminance = gpu.createKernel(function (photo) {
  const pixel = photo[this.thread.y][this.thread.x];
  return 0.299 * pixel[0] + 0.587 * pixel[1] + 0.114 * pixel[2];
}, { output: [64, 64] });

const paint = gpu.createKernel(function (map) {
  const l = map[this.thread.y][this.thread.x];
  this.color(l, l, l, 1);
}, { output: [64, 64], graphical: true });

const map = luminance(photo);
paint(map);
render(paint.canvas);
`,
      inputs: utils => ({ photo: utils.makeTestImage(64) }),
      publicTests: [
        {
          name: 'two kernels: a numeric pass and a graphical pass',
          run: async ctx => {
            ctx.assert(ctx.kernels.length >= 2, `expected 2 kernels, found ${ctx.kernels.length}`);
            const graphical = ctx.kernels.find(k => k.kernel && k.kernel.graphical);
            const numeric = ctx.kernels.find(k => k.kernel && !k.kernel.graphical);
            ctx.assert(numeric, 'no numeric (non-graphical) kernel found');
            ctx.assert(graphical, 'no graphical kernel found');
          },
        },
        {
          name: 'luminance pass: cell [y][x] = <code>0.299r + 0.587g + 0.114b</code>',
          run: async ctx => {
            const numeric = ctx.kernels.find(k => k.kernel && !k.kernel.graphical);
            ctx.assert(numeric, 'no numeric kernel found');
            const img = ctx.utils.makeTestImage(64);
            const out = numeric(img);
            const plain = img.plain;
            const cases = [[0, 0], [5, 50], [33, 12], [63, 63]];
            for (const [y, x] of cases) {
              const hint = transposeCellHint(out[y][x], luminanceOf(plain[x][y]), 2e-3, y, x) ||
                diagnose(out[y][x], luminanceOf(plain[y][x]), 2e-3, luminanceProbes(plain[y][x]));
              ctx.assertClose(out[y][x], luminanceOf(plain[y][x]), 2e-3, hint || `cell [${y}][${x}]`);
            }
          },
        },
        {
          name: 'painted canvas is monochrome and <code>64×64</code>',
          run: async ctx => {
            ctx.assert(ctx.canvas, 'no canvas — did you call render(paint.canvas)?');
            ctx.assert(
              ctx.canvas.width === 64 && ctx.canvas.height === 64,
              `expected a 64×64 canvas, got ${ctx.canvas.width}×${ctx.canvas.height}`
            );
            const pixels = ctx.getPixels();
            for (let i = 0; i < pixels.length; i += 61 * 4) {
              const r = pixels[i];
              const g = pixels[i + 1];
              const b = pixels[i + 2];
              ctx.assert(
                Math.abs(r - g) <= 1 && Math.abs(g - b) <= 1,
                `pixel at byte ${i} is not gray: rgb(${r}, ${g}, ${b})`
              );
            }
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            // End-to-end on a constant image: every painted pixel must equal
            // the constant's luminance, whatever the row order.
            const numeric = ctx.kernels.find(k => k.kernel && !k.kernel.graphical);
            const graphical = ctx.kernels.find(k => k.kernel && k.kernel.graphical);
            ctx.assert(numeric && graphical, 'expected a numeric and a graphical kernel');
            const image = constantImage(64, [0.6, 0.2, 0.4, 1]);
            const expected = luminanceOf(image.at(0, 0)) * 255;
            const map = numeric(image);
            graphical(map);
            const pixels = graphical.getPixels();
            for (let i = 0; i < pixels.length; i += 149 * 4) {
              ctx.assertClose(pixels[i], expected, 2, `red at byte ${i}`);
              ctx.assertClose(pixels[i + 1], expected, 2, `green at byte ${i}`);
              ctx.assertClose(pixels[i + 2], expected, 2, `blue at byte ${i}`);
            }
          },
        },
      ],
    },
  ],
};
