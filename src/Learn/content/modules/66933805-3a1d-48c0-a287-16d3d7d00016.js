// Module: Convolution & Filters — uuid 66933805-3a1d-48c0-a287-16d3d7d00016 (short id 66933805).
// The file name is the uuid; identity lives in the exported object below,
// never in the path. Legacy id (pre-uuid URLs, localStorage migration): 2-3.
//
// Convolution & Filters.
//
// Five tasks: a hardcoded 3-tap smoothing kernel → a generic 1D convolution
// with the filter as data and its size as constants → a 2D box blur on an
// image → sharpening with negative weights on a luminance map → Sobel edge
// detection as the payoff (two directional convolutions, combined).
//
// Kernel-authoring rules (contract): no closures inside kernel functions,
// only numbers / nested number arrays as inputs, this.thread.* for indexing,
// this.constants.* for compile-time values (legal as loop bounds), image
// convention image[y][x] = [r, g, b, a] with channels 0–1. Every task passes
// in CPU mode.

import { ARRAY_LAYOUT } from '../layoutNote.js';
import { plainToImageData, quantizePixel } from '../../engine/utils.js';

// Swapping this.thread.x and this.thread.y reads the transpose of the image —
// the single most common mistake in the image tasks, and invisible to tests
// that only check flat colours or greyness. Numeric results come back as plain
// 2D arrays, so the cell under test is known exactly: a cell holding the value
// that belongs to cell [x][y] is that swapped read. Cells on the diagonal
// (y === x) are their own transpose and can never show it, which is why the
// case lists below also probe off-diagonal cells.
function transposeCellHint(got, transposed, eps, y, x) {
  return Math.abs(got - transposed) <= eps
    ? `that is the value for cell [${x}][${y}] — this.thread.x and this.thread.y are ` +
      `swapped. Rows come first: image[this.thread.y][this.thread.x]`
    : null;
}

// Canvas counterpart: a graphical kernel that swaps the two thread coordinates
// paints the whole picture transposed, so the value belonging at (row, col)
// turns up at (col, row) instead. getPixels row order can be top-down or
// bottom-up, so each caller decides — in whatever terms its own expectation is
// written — whether the observation matches the transposed picture under either
// order, and this words the diagnosis once.
function transposePixelHint(hit, row, col) {
  return hit
    ? `the picture is transposed — the value for row ${row}, col ${col} turned up at ` +
      `row ${col}, col ${row}. this.thread.x and this.thread.y are swapped; rows come ` +
      `first: image[this.thread.y][this.thread.x].`
    : null;
}

// ---- near-miss diagnosis --------------------------------------------------
//
// The transpose hints above are one instance of a general idea: when a failing
// value is exactly what some specific mistake would produce, name that mistake
// instead of reporting two numbers. A probe pairs such a value with its
// sentence; diagnose() speaks only when the observed value matches a probe
// within the test's own tolerance AND the correct value does not — so samples
// where a candidate happens to equal the right answer stay silent, as do
// observations matching probes that disagree with each other. A wrong diagnosis
// is worse than none.
function diagnose(got, expected, eps, probes) {
  const hits = probes
    .filter(p => Math.abs(got - p[0]) <= eps && Math.abs(expected - p[0]) > eps)
    .map(p => p[1]);
  return hits.length && hits.every(m => m === hits[0]) ? hits[0] : null;
}

// Task 1: the 3-tap filter left unweighted, mis-weighted, or not applied.
function smoothProbes(signal, i) {
  const n = signal.length;
  const clamp = j => Math.max(0, Math.min(n - 1, j));
  const l = signal[clamp(i - 1)];
  const c = signal[i];
  const r = signal[clamp(i + 1)];
  return [
    [(l + c + r) / 3, 'that is the plain 3-tap mean — this filter weights the taps 0.25 / 0.5 / 0.25'],
    [c, 'that is the sample itself — the weighted average of its neighborhood never happened'],
    [0.25 * (l + c + r), 'the center tap carries 0.5, not 0.25 — the three weights have to sum to 1'],
  ];
}

// Task 1, the ends: a neighbor index that was never clamped reads past the
// signal and comes back as a non-number.
function unclampedHint(got) {
  return Number.isFinite(got)
    ? null
    : 'that sample read outside the signal — clamp the neighbor indexes into 0…127 before reading';
}

// Task 2: both ways of misplacing the window (dropping the radius, or adding it
// instead of subtracting it) slide the taps off center, so they share a
// message. The shifted references are built once per test by the caller.
function tapProbes(shiftedRefs, i) {
  const shifted = 'the window is not centered on this thread — tap i belongs at x + i − this.constants.radius';
  return shiftedRefs.map(ref => [ref[i], shifted]);
}

// Task 3: painting the nine-sample sum instead of its mean clamps every channel
// to white, which a not-nearly-white expectation gives away.
function undividedHint(pixels, row, col, expected) {
  const i = (row * 128 + col) * 4;
  const white = pixels[i] >= 253 && pixels[i + 1] >= 253 && pixels[i + 2] >= 253;
  return white && Math.max(expected[0], expected[1], expected[2]) < 0.9
    ? 'every channel is clamped to white — that is the sum of the nine samples; divide each one by 9'
    : null;
}

// Task 4: the sharpen cross with the wrong center weight or the wrong signs.
function sharpenProbes(gray, y, x) {
  const size = gray.length;
  const clamp = i => Math.max(0, Math.min(size - 1, i));
  const c = gray[y][x];
  const around = gray[y][clamp(x - 1)] + gray[y][clamp(x + 1)] +
    gray[clamp(y - 1)][x] + gray[clamp(y + 1)][x];
  return [
    [4 * c - around, 'the center weight is 4, not 5 — the five weights have to sum to 1 so flat areas pass through unchanged'],
    [5 * c + around, 'the four neighbors are being added — a sharpen subtracts them: 5·center − left − right − up − down'],
    [c, 'that is the value unchanged — none of the five weights reached the return value'],
    [around / 4, 'that is the average of the four neighbors — the 5·center term is missing'],
  ];
}

// Deterministic noisy signal for the 1D tasks (shared by inputs() and tests).
function makeSignal(utils, seed = 2301) {
  const rand = utils.seededRandom(seed);
  const data = new Array(128);
  for (let i = 0; i < 128; i++) {
    data[i] = Math.round((Math.sin(i / 6) * 3 + rand() * 4) * 100) / 100;
  }
  return data;
}

// CPU reference: 1D convolution with clamp-to-edge indexing.
function convolve1dRef(signal, filter, radius) {
  const n = signal.length;
  const out = new Array(n);
  for (let x = 0; x < n; x++) {
    let sum = 0;
    for (let i = 0; i < filter.length; i++) {
      let tap = x + i - radius;
      if (tap < 0) tap = 0;
      if (tap > n - 1) tap = n - 1;
      sum += filter[i] * signal[tap];
    }
    out[x] = sum;
  }
  return out;
}

// CPU reference: 3×3 box blur with clamp-to-edge, [y][x] = [r, g, b] 0–1.
// `image` is a course image (an ImageData); .plain is its host-side [y][x] view.
function boxBlurRef(image) {
  const pixels = image.plain;
  const size = pixels.length;
  const out = new Array(size);
  for (let y = 0; y < size; y++) {
    const row = new Array(size);
    for (let x = 0; x < size; x++) {
      let r = 0;
      let g = 0;
      let b = 0;
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          const sy = Math.min(size - 1, Math.max(0, y + dy));
          const sx = Math.min(size - 1, Math.max(0, x + dx));
          const p = pixels[sy][sx];
          r += p[0];
          g += p[1];
          b += p[2];
        }
      }
      row[x] = [r / 9, g / 9, b / 9];
    }
    out[y] = row;
  }
  return out;
}

// Luminance map of a course image (an ImageData) — plain nested arrays of
// numbers, which is what the numeric tasks take as input.
function grayOf(image) {
  const pixels = image.plain;
  const size = pixels.length;
  const out = new Array(size);
  for (let y = 0; y < size; y++) {
    const row = new Array(size);
    for (let x = 0; x < size; x++) {
      const p = pixels[y][x];
      row[x] = 0.299 * p[0] + 0.587 * p[1] + 0.114 * p[2];
    }
    out[y] = row;
  }
  return out;
}

// Deterministic random 2D grayscale map (private tests, task 4).
function randomGray(utils, size, seed) {
  const rand = utils.seededRandom(seed);
  const out = new Array(size);
  for (let y = 0; y < size; y++) {
    const row = new Array(size);
    for (let x = 0; x < size; x++) row[x] = Math.round(rand() * 1000) / 1000;
    out[y] = row;
  }
  return out;
}

// CPU reference: 5×center − the 4 direct neighbors, clamp-to-edge.
function sharpenRef(gray) {
  const size = gray.length;
  const out = new Array(size);
  for (let y = 0; y < size; y++) {
    const row = new Array(size);
    const up = Math.max(0, y - 1);
    const down = Math.min(size - 1, y + 1);
    for (let x = 0; x < size; x++) {
      const left = Math.max(0, x - 1);
      const right = Math.min(size - 1, x + 1);
      row[x] = 5 * gray[y][x] - gray[y][left] - gray[y][right] - gray[up][x] - gray[down][x];
    }
    out[y] = row;
  }
  return out;
}

// The three test-image helpers below all return the same thing the tasks' own
// inputs() do: an ImageData, the one image shape a graphical kernel can run on
// the GPU, whose .plain/.at(x, y) answer image[y][x] = [r, g, b, a] host-side.
// Their pixels are quantized to what 8-bit channels can hold, so an expectation
// read off .at() is exactly what the kernel sees. A kernel built with an
// ImageData must never be re-invoked with a nested array — on WebGL2 that
// silently paints wrong pixels, with no error and no warning.

// Constant-color image (orientation-independent pixel tests).
function constantImage(size, pixel) {
  const row = new Array(size).fill(quantizePixel(pixel));
  return plainToImageData(new Array(size).fill(row));
}

// Gray image split into a dark left half and a light right half.
function verticalSplitImage(size, darkValue, lightValue) {
  const dark = quantizePixel([darkValue, darkValue, darkValue, 1]);
  const light = quantizePixel([lightValue, lightValue, lightValue, 1]);
  const image = new Array(size);
  for (let y = 0; y < size; y++) {
    const row = new Array(size);
    for (let x = 0; x < size; x++) row[x] = x < size / 2 ? dark : light;
    image[y] = row;
  }
  return plainToImageData(image);
}

// Gray image split into a dark top half and a light bottom half.
function horizontalSplitImage(size, darkValue, lightValue) {
  const dark = quantizePixel([darkValue, darkValue, darkValue, 1]);
  const light = quantizePixel([lightValue, lightValue, lightValue, 1]);
  const image = new Array(size);
  for (let y = 0; y < size; y++) {
    image[y] = new Array(size).fill(y < size / 2 ? dark : light);
  }
  return plainToImageData(image);
}

export default {
  uuid: '66933805-3a1d-48c0-a287-16d3d7d00016',
  version: 1,
  slug: 'convolution-and-filters',
  legacyId: '2-3',
  title: 'Convolution & Filters',
  blurb: 'Sliding-window math on signals and images: blur, sharpen, edge detection.',
  tasks: [
    // ------------------------------------------------------------------ 1
    {
      slug: 'smooth-a-signal',
      title: 'Slide a Window: 1D Convolution',
      intro: `<p>A <strong>convolution</strong> slides a small window of weights along a signal:
        each output sample is a weighted average of the input around it. With weights
        <code>[0.25, 0.5, 0.25]</code> the window <em>smooths</em> — every sample leans toward
        its neighbors and jitter cancels out.</p>
        <p>On the GPU nothing actually slides. Every output sample gets its own thread, and each
        thread reads its <em>own</em> three inputs, all at the same time. The only wrinkle is the
        ends: sample 0 has no left neighbor, so we <strong>clamp</strong> — reuse the nearest
        in-bounds sample instead of reading past the edge.</p>`,
      goal: `<strong>Goal:</strong> smooth the 128-sample <code>signal</code> — each output is
        <code>0.25·left + 0.5·center + 0.25·right</code>, with indexes clamped at both ends.`,
      requirements: [
        'Read this thread\'s neighbors: <code>signal[x - 1]</code> and <code>signal[x + 1]</code>',
        'Clamp the indexes — below <code>0</code> becomes <code>0</code>, above <code>127</code> becomes <code>127</code>',
        'Return <code>0.25·left + 0.5·center + 0.25·right</code>',
      ],
      hints: [
        {
          title: 'Hint 1 — nothing slides',
          body: `<p>Thread <code>x</code> only ever touches <code>signal[x - 1]</code>,
            <code>signal[x]</code> and <code>signal[x + 1]</code>. Three reads, one weighted sum,
            done — the "sliding" is 128 threads doing this at once.</p>`,
        },
        {
          title: 'Hint 2 — clamping with an if',
          body: `<pre><code>let left = x - 1;
if (left &lt; 0) left = 0;</code></pre>
<p>and the mirror image
            for <code>right</code> against <code>127</code>. Plain <code>if</code> statements work
            fine inside kernels.</p>`,
        },
        {
          title: 'Hint 3 — the whole body',
          body: `<pre><code>let left = x - 1;
if (left &lt; 0) left = 0;
let right = x + 1;
if (right &gt; 127) right = 127;
return 0.25 * signal[left] + 0.5 * signal[x] + 0.25 * signal[right];</code></pre>`,
        },
      ],
      transfer: `Neighborhood reads like this are called <em>stencil</em> patterns in CUDA and
        ROCm — the classic optimization is staging the window in shared memory. A WebGPU compute
        shader does the same thing with neighboring buffer reads inside a workgroup.`,
      starterCode: `// Convolution: each output sample is a weighted average of its neighborhood.
const gpu = new GPU({ mode });

const smooth = gpu.createKernel(function (signal) {
  const x = this.thread.x;
  // TODO: return 0.25 * left + 0.5 * center + 0.25 * right,
  // clamping the neighbor indexes so x = 0 and x = 127 stay in bounds.
  return signal[x];
}, { output: [128] });

const result = await smooth(signal);
console.log('before:', signal[63], ' after:', result[63]);
`,
      solutionCode: `// Convolution: each output sample is a weighted average of its neighborhood.
const gpu = new GPU({ mode });

const smooth = gpu.createKernel(function (signal) {
  const x = this.thread.x;
  let left = x - 1;
  if (left < 0) left = 0;
  let right = x + 1;
  if (right > 127) right = 127;
  return 0.25 * signal[left] + 0.5 * signal[x] + 0.25 * signal[right];
}, { output: [128] });

const result = await smooth(signal);
console.log('before:', signal[63], ' after:', result[63]);
`,
      inputs: utils => ({ signal: makeSignal(utils) }),
      publicTests: [
        {
          name: 'returns 128 samples, each a <code>[0.25, 0.5, 0.25]</code> weighted average',
          run: async ctx => {
            ctx.assert(ctx.kernels.length >= 1, 'no kernel was created — call gpu.createKernel()');
            const signal = makeSignal(ctx.utils);
            const out = await ctx.kernel(signal);
            ctx.assert(out && out.length === 128, `expected 128 output samples, got ${out && out.length}`);
            const ref = convolve1dRef(signal, [0.25, 0.5, 0.25], 1);
            for (const i of [1, 17, 42, 63, 100, 126]) {
              const hint = diagnose(out[i], ref[i], 1e-3, smoothProbes(signal, i));
              ctx.assertClose(out[i], ref[i], 1e-3, hint || `sample ${i}`);
            }
          },
        },
        {
          name: 'edges clamp: sample 0 is <code>0.75·s[0] + 0.25·s[1]</code>',
          run: async ctx => {
            const s = new Array(128);
            for (let i = 0; i < 128; i++) s[i] = ((i * 37) % 23) - 11;
            const out = await ctx.kernel(s);
            const ref = convolve1dRef(s, [0.25, 0.5, 0.25], 1);
            const edgeHint = i => unclampedHint(out[i]) ||
              diagnose(out[i], ref[i], 1e-3, smoothProbes(s, i));
            ctx.assertClose(out[0], 0.75 * s[0] + 0.25 * s[1], 1e-3, edgeHint(0) || 'sample 0');
            ctx.assertClose(out[127], 0.25 * s[126] + 0.75 * s[127], 1e-3, edgeHint(127) || 'sample 127');
            for (let i = 0; i < 128; i++) {
              ctx.assertClose(out[i], ref[i], 1e-3, edgeHint(i) || `sample ${i}`);
            }
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            const signal = makeSignal(ctx.utils, 909);
            const out = await ctx.kernel(signal);
            const ref = convolve1dRef(signal, [0.25, 0.5, 0.25], 1);
            ctx.assert(out.length === 128, 'expected 128 output samples');
            for (let i = 0; i < 128; i++) {
              const hint = unclampedHint(out[i]) ||
                diagnose(out[i], ref[i], 1e-3, smoothProbes(signal, i));
              ctx.assertClose(out[i], ref[i], 1e-3, hint || `sample ${i}`);
            }
          },
        },
      ],
    },

    // ------------------------------------------------------------------ 2
    {
      slug: 'filter-as-data',
      title: 'Any Filter, One Kernel',
      intro: `<p>Hardcoded weights mean writing a new kernel for every filter. The fix: pass the
        <code>filter</code> in as an ordinary array argument and loop over its taps. But a GPU
        loop wants bounds it can see <em>at compile time</em> — and that is exactly what
        <code>this.constants</code> is for: values baked into the kernel when it compiles,
        perfectly legal as loop bounds.</p>
        <p>This kernel is built with <code>constants: { size: 5, radius: 2 }</code>. Tap
        <code>i</code> of the filter lines up with input sample
        <code>x + i - radius</code> — clamp that index like before and accumulate
        <code>filter[i] * signal[tap]</code>.</p>`,
      goal: `<strong>Goal:</strong> finish the generic convolution — loop over
        <code>this.constants.size</code> taps, clamp each tap index, and return the accumulated
        weighted sum. One kernel, any 5-tap filter.`,
      requirements: [
        'Loop <code>for (let i = 0; i &lt; this.constants.size; i++)</code> — a constant is a legal bound',
        'Tap index: <code>x + i - this.constants.radius</code>, clamped to <code>0…127</code>',
        'Accumulate <code>filter[i] * signal[tap]</code> into <code>sum</code> and return it',
      ],
      hints: [
        {
          title: 'Hint 1 — why constants?',
          body: `<p>Kernel arguments change per call; constants are frozen into the compiled
            kernel. That is why <code>this.constants.size</code> can bound a loop when a plain
            argument could not.</p>`,
        },
        {
          title: 'Hint 2 — the loop body',
          body: `<pre><code>let tap = x + i - this.constants.radius;
if (tap &lt; 0) tap = 0;
if (tap &gt; 127) tap = 127;
sum += filter[i] * signal[tap];</code></pre>`,
        },
      ],
      transfer: `Baked-in constants are a first-class idea everywhere: WGSL has
        pipeline-overridable constants, CUDA kernels take template parameters and
        <code>__constant__</code> memory, Metal has function constants — all so the compiler
        knows your loop bounds and can unroll the filter loop.`,
      starterCode: `// One kernel, any 5-tap filter: weights come in as data, size as constants.
const gpu = new GPU({ mode });

const convolve = gpu.createKernel(function (signal, filter) {
  const x = this.thread.x;
  let sum = 0;
  // TODO: loop i from 0 to this.constants.size,
  //   tap index = x + i - this.constants.radius (clamped to 0…127),
  //   accumulate filter[i] * signal[tap].
  return sum;
}, {
  output: [128],
  constants: { size: 5, radius: 2 },
});

const gauss = [0.06, 0.24, 0.4, 0.24, 0.06];
const result = await convolve(signal, gauss);
console.log('smoothed sample 64:', result[64]);
`,
      solutionCode: `// One kernel, any 5-tap filter: weights come in as data, size as constants.
const gpu = new GPU({ mode });

const convolve = gpu.createKernel(function (signal, filter) {
  const x = this.thread.x;
  let sum = 0;
  for (let i = 0; i < this.constants.size; i++) {
    let tap = x + i - this.constants.radius;
    if (tap < 0) tap = 0;
    if (tap > 127) tap = 127;
    sum += filter[i] * signal[tap];
  }
  return sum;
}, {
  output: [128],
  constants: { size: 5, radius: 2 },
});

const gauss = [0.06, 0.24, 0.4, 0.24, 0.06];
const result = await convolve(signal, gauss);
console.log('smoothed sample 64:', result[64]);
`,
      inputs: utils => ({ signal: makeSignal(utils) }),
      publicTests: [
        {
          name: 'the identity filter <code>[0, 0, 1, 0, 0]</code> returns the signal untouched',
          run: async ctx => {
            ctx.assert(ctx.kernels.length >= 1, 'no kernel was created — call gpu.createKernel()');
            const signal = makeSignal(ctx.utils);
            const identity = [0, 0, 1, 0, 0];
            const out = await ctx.kernel(signal, identity);
            ctx.assert(out && out.length === 128, `expected 128 output samples, got ${out && out.length}`);
            const shifted = [
              convolve1dRef(signal, identity, 0),
              convolve1dRef(signal, identity, -2),
            ];
            for (let i = 0; i < 128; i++) {
              const hint = diagnose(out[i], signal[i], 1e-3, tapProbes(shifted, i));
              ctx.assertClose(out[i], signal[i], 1e-3, hint || `sample ${i}`);
            }
          },
        },
        {
          name: 'a box filter matches the clamped-edge reference everywhere',
          run: async ctx => {
            const s = new Array(128);
            for (let i = 0; i < 128; i++) s[i] = ((i * 29) % 17) - 8;
            const box = [0.2, 0.2, 0.2, 0.2, 0.2];
            const out = await ctx.kernel(s, box);
            const ref = convolve1dRef(s, box, 2);
            const shifted = [convolve1dRef(s, box, 0), convolve1dRef(s, box, -2)];
            for (let i = 0; i < 128; i++) {
              const hint = diagnose(out[i], ref[i], 2e-3, tapProbes(shifted, i));
              ctx.assertClose(out[i], ref[i], 2e-3, hint || `sample ${i}`);
            }
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            const signal = makeSignal(ctx.utils, 777);
            const rand = ctx.utils.seededRandom(31);
            const filter = new Array(5);
            for (let i = 0; i < 5; i++) filter[i] = Math.round((rand() * 0.6 - 0.1) * 100) / 100;
            const out = await ctx.kernel(signal, filter);
            const ref = convolve1dRef(signal, filter, 2);
            const shifted = [convolve1dRef(signal, filter, 0), convolve1dRef(signal, filter, -2)];
            for (let i = 0; i < 128; i++) {
              const hint = diagnose(out[i], ref[i], 2e-3, tapProbes(shifted, i));
              ctx.assertClose(out[i], ref[i], 2e-3, hint || `sample ${i}`);
            }
          },
        },
      ],
    },

    // ------------------------------------------------------------------ 3
    {
      slug: 'box-blur',
      title: 'Box Blur: the Window Goes 2D',
      intro: `<p>Take the sliding window into two dimensions and you have image filtering. A
        <strong>3×3 box blur</strong> is the simplest case: every output pixel is the plain
        average of the 3×3 patch centered on it — nine reads, per color channel, per pixel.
        131,072 threads each do their nine reads at once.</p>
        <p>Same edge problem, now on four sides: clamp <em>both</em> coordinates into
        <code>0…this.constants.last</code> before indexing. Average red, green and blue
        separately and hand the result to <code>this.color()</code>.</p>
        ${ARRAY_LAYOUT}`,
      goal: `<strong>Goal:</strong> blur <code>inputImage</code> with a 3×3 box filter — each
        painted pixel is the average of its 3×3 neighborhood, edges clamped.`,
      requirements: [
        'Loop over the 3×3 neighborhood (a double <code>for</code> loop over <code>dy</code>, <code>dx</code>)',
        'Clamp both sample coordinates to <code>0…this.constants.last</code>',
        'Accumulate red, green and blue separately, then paint <code>this.color(r/9, g/9, b/9, 1)</code>',
      ],
      hints: [
        {
          title: 'Hint 1 — the neighborhood loop',
          body: `<p><code>for (let dy = 0; dy &lt; 3; dy++)</code> nested with
            <code>dx</code>, and the sample position is
            <code>this.thread.y + dy - 1</code>, <code>this.thread.x + dx - 1</code> — the
            <code>- 1</code> centers the window on this thread's pixel.</p>`,
        },
        {
          title: 'Hint 2 — clamp, then read',
          body: `<pre><code>let sy = this.thread.y + dy - 1;
if (sy &lt; 0) sy = 0;
if (sy &gt; this.constants.last) sy = this.constants.last;</code></pre>
<p>— same for
            <code>sx</code> — then <code>const pixel = image[sy][sx];</code> and add
            <code>pixel[0]</code>, <code>pixel[1]</code>, <code>pixel[2]</code> into three
            running sums.</p>`,
        },
        {
          title: 'Hint 3 — the finish',
          body: `<p>After the loops: <code>this.color(r / 9, g / 9, b / 9, 1);</code> —
            nine samples went in, so divide by nine on the way out.</p>`,
        },
      ],
      transfer: `Blur passes ship in every production toolkit — Metal Performance Shaders'
        <code>MPSImageBox</code>, NVIDIA's NPP filtering routines, WebGPU post-processing
        chains. The fast ones exploit that a box blur is <em>separable</em>: a horizontal pass
        then a vertical pass — six reads per pixel instead of nine.`,
      starterCode: `// Nine reads per pixel, averaged per channel. 131,072 threads at once.
const gpu = new GPU({ mode });

const blur = gpu.createKernel(function (image) {
  // TODO: average the 3×3 neighborhood around this pixel.
  // Clamp sample coordinates to 0…this.constants.last on both axes.
  const pixel = image[this.thread.y][this.thread.x];
  this.color(pixel[0], pixel[1], pixel[2], 1);
}, {
  output: [128, 128],
  graphical: true,
  constants: { last: 127 },
});

await blur(inputImage);
render(blur.canvas);
`,
      solutionCode: `// Nine reads per pixel, averaged per channel. 131,072 threads at once.
const gpu = new GPU({ mode });

const blur = gpu.createKernel(function (image) {
  let r = 0;
  let g = 0;
  let b = 0;
  for (let dy = 0; dy < 3; dy++) {
    for (let dx = 0; dx < 3; dx++) {
      let sy = this.thread.y + dy - 1;
      let sx = this.thread.x + dx - 1;
      if (sy < 0) sy = 0;
      if (sy > this.constants.last) sy = this.constants.last;
      if (sx < 0) sx = 0;
      if (sx > this.constants.last) sx = this.constants.last;
      const pixel = image[sy][sx];
      r += pixel[0];
      g += pixel[1];
      b += pixel[2];
    }
  }
  this.color(r / 9, g / 9, b / 9, 1);
}, {
  output: [128, 128],
  graphical: true,
  constants: { last: 127 },
});

await blur(inputImage);
render(blur.canvas);
`,
      inputs: utils => ({ inputImage: utils.makeTestImage(128) }),
      publicTests: [
        {
          name: 'produces a <code>128×128</code> graphical canvas',
          run: async ctx => {
            ctx.assert(ctx.kernels.length >= 1, 'no kernel was created — call gpu.createKernel()');
            ctx.assert(ctx.canvas, 'no canvas — is the kernel graphical: true, and did you call render()?');
            ctx.assert(
              ctx.canvas.width === 128 && ctx.canvas.height === 128,
              `expected a 128×128 canvas, got ${ctx.canvas.width}×${ctx.canvas.height}`
            );
            const pixels = ctx.getPixels();
            ctx.assert(pixels.length === 128 * 128 * 4, 'pixel buffer should hold 128×128 RGBA values');
          },
        },
        {
          name: 'blurring a flat color changes nothing',
          run: async ctx => {
            const image = constantImage(128, [0.3, 0.5, 0.7, 1]);
            const flat = image.at(0, 0);
            await ctx.kernel(image);
            const pixels = ctx.getPixels();
            for (let i = 0; i < pixels.length; i += 331 * 4) {
              ctx.assertClose(pixels[i], flat[0] * 255, 2, `red at byte ${i}`);
              ctx.assertClose(pixels[i + 1], flat[1] * 255, 2, `green at byte ${i}`);
              ctx.assertClose(pixels[i + 2], flat[2] * 255, 2, `blue at byte ${i}`);
            }
          },
        },
        {
          name: 'each pixel is the average of its 3×3 neighborhood',
          run: async ctx => {
            const img = ctx.utils.makeTestImage(128);
            await ctx.kernel(img);
            const pixels = ctx.getPixels();
            const ref = boxBlurRef(img);
            // getPixels row order can be top-down or bottom-up depending on
            // backend — accept the correct average for either orientation.
            const matches = (row, col, expected) => {
              const i = (row * 128 + col) * 4;
              return (
                Math.abs(pixels[i] - expected[0] * 255) <= 3 &&
                Math.abs(pixels[i + 1] - expected[1] * 255) <= 3 &&
                Math.abs(pixels[i + 2] - expected[2] * 255) <= 3
              );
            };
            for (const row of [3, 17, 40, 64, 90, 121]) {
              for (const col of [5, 33, 64, 101, 124]) {
                // a box blur commutes with the transpose, so a swapped read
                // paints the average of the neighborhood around [col][row]
                const swapped =
                  matches(row, col, ref[col][row]) || matches(row, col, ref[col][127 - row]);
                ctx.assert(
                  matches(row, col, ref[row][col]) || matches(row, col, ref[127 - row][col]),
                  transposePixelHint(swapped, row, col) ||
                    undividedHint(pixels, row, col, ref[row][col]) ||
                    `pixel at row ${row}, col ${col} is not the 3×3 average of its neighborhood`
                );
              }
            }
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            // Dark-left / light-right split: blurred columns have exact,
            // row-independent values — orientation cannot hide a mistake.
            await ctx.kernel(verticalSplitImage(128, 0.2, 0.8));
            const pixels = ctx.getPixels();
            const expectedAt = col => {
              if (col <= 62) return 0.2;
              if (col === 63) return 0.4; // (0.2 + 0.2 + 0.8) / 3
              if (col === 64) return 0.6; // (0.2 + 0.8 + 0.8) / 3
              return 0.8;
            };
            for (const row of [8, 60, 119]) {
              for (const col of [0, 20, 63, 64, 90, 127]) {
                const i = (row * 128 + col) * 4;
                const expected = expectedAt(col) * 255;
                // The unblurred image is 0.2 left of the seam and 0.8 right of
                // it, so a passthrough shows up as the raw half at columns 63/64.
                const raw = (col < 64 ? 0.2 : 0.8) * 255;
                const hint = diagnose(pixels[i], expected, 2, [
                  [raw, 'that is the original pixel — the 3×3 average never happened, so the seam did not soften'],
                ]);
                ctx.assertClose(pixels[i], expected, 2, hint || `red at row ${row}, col ${col}`);
                ctx.assertClose(pixels[i + 1], expected, 2, hint || `green at row ${row}, col ${col}`);
                ctx.assertClose(pixels[i + 2], expected, 2, hint || `blue at row ${row}, col ${col}`);
              }
            }
          },
        },
      ],
    },

    // ------------------------------------------------------------------ 4
    {
      slug: 'sharpen',
      title: 'Sharpen: Negative Weights',
      intro: `<p>Filters are not all averages. Give the window <strong>negative weights</strong>
        and it starts measuring <em>differences</em>. The classic sharpen filter is a cross:
        <code>5</code> at the center, <code>−1</code> at each direct neighbor. Where the image is
        flat, the terms cancel to exactly the original value; where it changes, the difference
        gets amplified — edges pop.</p>
        <p>Sharpened values can overshoot right out of the 0–1 range, so this task computes on a
        numeric <strong>luminance map</strong> (<code>gray[y][x]</code>, one number per pixel)
        and returns raw numbers you can inspect — no color clamping hiding the math.</p>
        ${ARRAY_LAYOUT}`,
      goal: `<strong>Goal:</strong> sharpen the 96×96 <code>gray</code> map — each cell becomes
        <code>5·center − left − right − up − down</code>, with neighbor indexes clamped.`,
      requirements: [
        'Clamp all four neighbor indexes to <code>0…this.constants.last</code>',
        'Return <code>5 * gray[y][x]</code> minus the four clamped neighbor samples',
        'Keep the kernel numeric — no <code>graphical: true</code>, values may leave 0–1',
      ],
      hints: [
        {
          title: 'Hint 1 — why 5 and −1?',
          body: `<p>The weights sum to 1, so flat regions pass through unchanged:
            <code>5c − 4c = c</code>. Everything the filter adds comes purely from
            center-vs-neighbor <em>differences</em>.</p>`,
        },
        {
          title: 'Hint 2 — four clamps, one return',
          body: `<pre><code>let left = x - 1;
if (left &lt; 0) left = 0;</code></pre>
<p>— repeat for
            <code>right</code>, <code>up</code>, <code>down</code> against
            <code>this.constants.last</code>, then a single return with the five terms:</p>
<pre><code>return 5 * gray[y][x] - gray[y][left] - gray[y][right]
  - gray[up][x] - gray[down][x];</code></pre>`,
        },
      ],
      transfer: `A convolution with learned weights is a CNN layer — cuDNN (CUDA) and MIOpen
        (ROCm) are entire libraries for running this exact multiply-accumulate window fast.
        Your sharpen filter is the same arithmetic with the weights picked by hand instead of
        by gradient descent.`,
      starterCode: `// Sharpen = identity + edge boost: 5×center − the 4 direct neighbors.
const gpu = new GPU({ mode });

const sharpen = gpu.createKernel(function (gray) {
  const x = this.thread.x;
  const y = this.thread.y;
  // TODO: clamp left/right/up/down to 0…this.constants.last, then
  // return 5 * center − left − right − up − down.
  return gray[y][x];
}, {
  output: [96, 96],
  constants: { last: 95 },
});

const result = await sharpen(gray);
console.log('center before:', gray[48][48], ' after:', result[48][48]);
`,
      solutionCode: `// Sharpen = identity + edge boost: 5×center − the 4 direct neighbors.
const gpu = new GPU({ mode });

const sharpen = gpu.createKernel(function (gray) {
  const x = this.thread.x;
  const y = this.thread.y;
  let left = x - 1;
  if (left < 0) left = 0;
  let right = x + 1;
  if (right > this.constants.last) right = this.constants.last;
  let up = y - 1;
  if (up < 0) up = 0;
  let down = y + 1;
  if (down > this.constants.last) down = this.constants.last;
  return 5 * gray[y][x] - gray[y][left] - gray[y][right] - gray[up][x] - gray[down][x];
}, {
  output: [96, 96],
  constants: { last: 95 },
});

const result = await sharpen(gray);
console.log('center before:', gray[48][48], ' after:', result[48][48]);
`,
      inputs: utils => ({ gray: grayOf(utils.makeTestImage(96)) }),
      publicTests: [
        {
          name: 'flat regions are a fixed point — sharpening a constant map returns it unchanged',
          run: async ctx => {
            ctx.assert(ctx.kernels.length >= 1, 'no kernel was created — call gpu.createKernel()');
            const flat = new Array(96).fill(new Array(96).fill(0.5));
            const out = await ctx.kernel(flat);
            ctx.assert(out && out.length === 96, `expected 96 rows, got ${out && out.length}`);
            ctx.assert(out[0] && out[0].length === 96, 'each row should hold 96 values');
            for (let y = 0; y < 96; y += 7) {
              for (let x = 0; x < 96; x += 7) {
                const hint = diagnose(out[y][x], 0.5, 1e-3, sharpenProbes(flat, y, x));
                ctx.assertClose(out[y][x], 0.5, 1e-3, hint || `cell [${y}][${x}]`);
              }
            }
          },
        },
        {
          name: 'cell [y][x] equals <code>5·center − left − right − up − down</code>',
          run: async ctx => {
            const gray = grayOf(ctx.utils.makeTestImage(96));
            const out = await ctx.kernel(gray);
            const ref = sharpenRef(gray);
            const cases = [[0, 0], [0, 48], [11, 60], [48, 48], [77, 3], [95, 95]];
            for (const [y, x] of cases) {
              // the sharpen cross is symmetric in x and y, so it commutes with
              // the transpose: a swapped read lands ref[x][y] in this cell
              const hint = transposeCellHint(out[y][x], ref[x][y], 2e-3, y, x) ||
                diagnose(out[y][x], ref[y][x], 2e-3, sharpenProbes(gray, y, x));
              ctx.assertClose(out[y][x], ref[y][x], 2e-3, hint || `cell [${y}][${x}]`);
            }
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            const gray = randomGray(ctx.utils, 96, 4242);
            const out = await ctx.kernel(gray);
            const ref = sharpenRef(gray);
            for (let y = 0; y < 96; y++) {
              for (let x = 0; x < 96; x++) {
                const hint = transposeCellHint(out[y][x], ref[x][y], 2e-3, y, x) ||
                  diagnose(out[y][x], ref[y][x], 2e-3, sharpenProbes(gray, y, x));
                ctx.assertClose(out[y][x], ref[y][x], 2e-3, hint || `cell [${y}][${x}]`);
              }
            }
          },
        },
      ],
    },

    // ------------------------------------------------------------------ 5
    {
      slug: 'sobel',
      title: 'Sobel Edge Detection',
      intro: `<p>The payoff: run <strong>two convolutions at once</strong>. Sobel's
        <code>Gx</code> filter responds to horizontal change, <code>Gy</code> to vertical change,
        and the length of that gradient vector — <code>√(gx² + gy²)</code> — is how
        <em>edge-like</em> the pixel is, whatever the edge's direction.</p>
        <p>This is a two-kernel pipeline like the finale of <strong>Data In, Data Out</strong>: a
        numeric pass turns the image
        into a luminance map (written for you), then the Sobel pass reads each map cell's eight
        neighbors, applies both weight grids, and paints the magnitude. Border pixels have no
        full neighborhood, so the starter already paints them black — your work lives in the
        <code>else</code> branch.</p>
        ${ARRAY_LAYOUT}`,
      goal: `<strong>Goal:</strong> finish the Sobel kernel — read the 3×3 neighborhood of
        <code>gray</code>, compute <code>gx</code> and <code>gy</code> with the weights shown in
        the starter, and paint <code>Math.sqrt(gx * gx + gy * gy)</code> as a gray value.`,
      requirements: [
        'Read the eight neighbors of <code>gray[y][x]</code> (no clamping needed — the border branch already ran)',
        'Apply both weight grids: <code>gx</code> from the right column minus the left, <code>gy</code> from the bottom row minus the top',
        'Paint the magnitude <code>Math.sqrt(gx * gx + gy * gy)</code> as gray via <code>this.color(m, m, m, 1)</code>',
      ],
      hints: [
        {
          title: 'Hint 1 — name the neighborhood',
          body: `<p>Pull the nine cells into locals first —
            <code>const tl = gray[y - 1][x - 1];</code> through
            <code>const br = gray[y + 1][x + 1];</code> — then the two weighted sums are easy to
            read off the grids.</p>`,
        },
        {
          title: 'Hint 2 — the two sums',
          body: `<pre><code>const gx = (tr + 2 * mr + br) - (tl + 2 * ml + bl);</code></pre>
<p>— right column minus left column, middle counted double. <code>gy</code> is the same
            with rows: <code>(bl + 2 * bm + br) - (tl + 2 * tm + tr)</code>.</p>`,
        },
        {
          title: 'Hint 3 — the finish',
          body: `<pre><code>const m = Math.sqrt(gx * gx + gy * gy);
this.color(m, m, m, 1);</code></pre>
<p>— flat areas give 0 (black), sharp edges overshoot 1 and clamp to white.</p>`,
        },
      ],
      transfer: `Sobel is the hello-world of GPU vision: it opens the OpenCL and CUDA imaging
        tutorials, camera ISPs run it in silicon, and edge maps feed feature detectors
        everywhere. Fusing two directional filters into one pass is exactly how you would write
        it in WGSL or Metal, too.`,
      starterCode: `// Two directional convolutions, one kernel, magnitude out.
const gpu = new GPU({ mode });

// Pass 1 — luminance map (Data In, Data Out déjà vu; already done for you).
const luminance = gpu.createKernel(function (image) {
  const pixel = image[this.thread.y][this.thread.x];
  return 0.299 * pixel[0] + 0.587 * pixel[1] + 0.114 * pixel[2];
}, { output: [128, 128] });

// Pass 2 — Sobel. Gx and Gy weigh the same 3×3 neighborhood:
//
//        Gx              Gy
//    -1   0  +1      -1  -2  -1
//    -2   0  +2       0   0   0
//    -1   0  +1      +1  +2  +1
//
const sobel = gpu.createKernel(function (gray) {
  const x = this.thread.x;
  const y = this.thread.y;
  if (x === 0 || y === 0 || x === this.constants.last || y === this.constants.last) {
    this.color(0, 0, 0, 1); // border: no full neighborhood — paint it black
  } else {
    // TODO: read the 8 neighbors, compute gx and gy with the grids above,
    // then paint the magnitude Math.sqrt(gx * gx + gy * gy).
    const l = gray[y][x];
    this.color(l, l, l, 1);
  }
}, {
  output: [128, 128],
  graphical: true,
  constants: { last: 127 },
});

const grayMap = await luminance(inputImage);
await sobel(grayMap);
render(sobel.canvas);
`,
      solutionCode: `// Two directional convolutions, one kernel, magnitude out.
const gpu = new GPU({ mode });

// Pass 1 — luminance map.
const luminance = gpu.createKernel(function (image) {
  const pixel = image[this.thread.y][this.thread.x];
  return 0.299 * pixel[0] + 0.587 * pixel[1] + 0.114 * pixel[2];
}, { output: [128, 128] });

// Pass 2 — Sobel magnitude.
const sobel = gpu.createKernel(function (gray) {
  const x = this.thread.x;
  const y = this.thread.y;
  if (x === 0 || y === 0 || x === this.constants.last || y === this.constants.last) {
    this.color(0, 0, 0, 1);
  } else {
    const tl = gray[y - 1][x - 1];
    const tm = gray[y - 1][x];
    const tr = gray[y - 1][x + 1];
    const ml = gray[y][x - 1];
    const mr = gray[y][x + 1];
    const bl = gray[y + 1][x - 1];
    const bm = gray[y + 1][x];
    const br = gray[y + 1][x + 1];
    const gx = (tr + 2 * mr + br) - (tl + 2 * ml + bl);
    const gy = (bl + 2 * bm + br) - (tl + 2 * tm + tr);
    const m = Math.sqrt(gx * gx + gy * gy);
    this.color(m, m, m, 1);
  }
}, {
  output: [128, 128],
  graphical: true,
  constants: { last: 127 },
});

const grayMap = await luminance(inputImage);
await sobel(grayMap);
render(sobel.canvas);
`,
      inputs: utils => ({ inputImage: utils.makeTestImage(128) }),
      publicTests: [
        {
          name: 'a numeric luminance pass feeding a graphical Sobel pass',
          run: async ctx => {
            ctx.assert(ctx.kernels.length >= 2, `expected 2 kernels, found ${ctx.kernels.length}`);
            const numeric = ctx.kernels.find(k => k.kernel && !k.kernel.graphical);
            const graphical = ctx.kernels.find(k => k.kernel && k.kernel.graphical);
            ctx.assert(numeric, 'no numeric (non-graphical) kernel found');
            ctx.assert(graphical, 'no graphical kernel found');
            ctx.assert(ctx.canvas, 'no canvas — did you call render(sobel.canvas)?');
            ctx.assert(
              ctx.canvas.width === 128 && ctx.canvas.height === 128,
              `expected a 128×128 canvas, got ${ctx.canvas.width}×${ctx.canvas.height}`
            );
          },
        },
        {
          name: 'a flat image has no edges — constant in, all black out',
          run: async ctx => {
            const numeric = ctx.kernels.find(k => k.kernel && !k.kernel.graphical);
            const graphical = ctx.kernels.find(k => k.kernel && k.kernel.graphical);
            ctx.assert(numeric && graphical, 'expected a numeric and a graphical kernel');
            await graphical(await numeric(constantImage(128, [0.4, 0.6, 0.2, 1])));
            const pixels = graphical.getPixels();
            for (let i = 0; i < pixels.length; i += 251 * 4) {
              ctx.assert(
                pixels[i] <= 1 && pixels[i + 1] <= 1 && pixels[i + 2] <= 1,
                `pixel at byte ${i} should be black, got rgb(${pixels[i]}, ${pixels[i + 1]}, ${pixels[i + 2]})`
              );
              ctx.assert(pixels[i + 3] === 255, `alpha at byte ${i} should be 255`);
            }
          },
        },
        {
          name: 'a vertical brightness step lights up exactly the step columns',
          run: async ctx => {
            const numeric = ctx.kernels.find(k => k.kernel && !k.kernel.graphical);
            const graphical = ctx.kernels.find(k => k.kernel && k.kernel.graphical);
            ctx.assert(numeric && graphical, 'expected a numeric and a graphical kernel');
            await graphical(await numeric(verticalSplitImage(128, 0.1, 0.9)));
            const pixels = graphical.getPixels();
            const red = (r, c) => pixels[(r * 128 + c) * 4];
            // Transposed, the picture answers to the row where it should answer
            // to the column: what belongs at (row, col) sits at (col, row) —
            // or at (127 - col, row) when the row order is bottom-up.
            const mirrorHolds = (row, col, ok) => ok(red(col, row)) || ok(red(127 - col, row));
            // Column positions survive any vertical flip of the row order.
            for (const row of [10, 64, 100]) {
              for (const col of [63, 64]) {
                const i = (row * 128 + col) * 4;
                const swapped = mirrorHolds(row, col, v => v >= 253);
                ctx.assert(
                  pixels[i] >= 253,
                  transposePixelHint(swapped, row, col) ||
                    `the step at col ${col} should saturate white, got ${pixels[i]} (row ${row})`
                );
              }
              for (const col of [30, 96]) {
                const i = (row * 128 + col) * 4;
                const swapped = mirrorHolds(row, col, v => v <= 1);
                ctx.assert(
                  pixels[i] <= 1,
                  transposePixelHint(swapped, row, col) ||
                    `flat area at col ${col} should be black, got ${pixels[i]} (row ${row})`
                );
              }
            }
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            // Horizontal step: rows 63/64 must saturate, far rows stay black —
            // and the expected picture is symmetric under a vertical flip.
            const numeric = ctx.kernels.find(k => k.kernel && !k.kernel.graphical);
            const graphical = ctx.kernels.find(k => k.kernel && k.kernel.graphical);
            ctx.assert(numeric && graphical, 'expected a numeric and a graphical kernel');
            await graphical(await numeric(horizontalSplitImage(128, 0.15, 0.85)));
            const pixels = graphical.getPixels();
            const red = (r, c) => pixels[(r * 128 + c) * 4];
            // A transposed paint answers to the column instead of the row: what
            // belongs at (row, col) sits at (col, row), or at (127 - col, row)
            // when the row order is bottom-up.
            const mirrorHolds = (row, col, ok) => ok(red(col, row)) || ok(red(127 - col, row));
            for (const col of [10, 64, 120]) {
              for (const row of [63, 64]) {
                const i = (row * 128 + col) * 4;
                const swapped = mirrorHolds(row, col, v => v >= 253);
                ctx.assert(
                  pixels[i] >= 253,
                  transposePixelHint(swapped, row, col) ||
                    `the step at row ${row} should saturate white, got ${pixels[i]} (col ${col})`
                );
              }
              for (const row of [20, 100]) {
                const i = (row * 128 + col) * 4;
                const swapped = mirrorHolds(row, col, v => v <= 1);
                ctx.assert(
                  pixels[i] <= 1,
                  transposePixelHint(swapped, row, col) ||
                    `flat area at row ${row} should be black, got ${pixels[i]} (col ${col})`
                );
              }
            }
          },
        },
      ],
    },
  ],
};
