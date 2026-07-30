// Module 1.4 — Pipelines & Textures.
//
// Five tasks: pipeline: true and what a texture is → chaining two kernels
// through a texture → refactoring readbacks out of a three-stage chain →
// immutable textures for iterative feedback loops → a photo-to-screen
// pipeline with zero readbacks.
//
// Backend facts these tasks rely on (verified against the gpu.js source):
// GL pipeline kernels return Texture objects with .toArray(); the CPU
// backend returns plain arrays, so mode-safe reads use the same guard
// gpu.js uses internally: `v.toArray ? v.toArray() : v`. Feeding a
// non-immutable pipeline kernel its own output throws "Source and
// destination … are the same. Use immutable = true" on BOTH backends.

import { ARRAY_LAYOUT } from '../layoutNote.js';
import { plainToImageData, quantizePixel } from '../../engine/utils.js';

const LUM = [0.299, 0.587, 0.114];

function luminanceOf(pixel) {
  return LUM[0] * pixel[0] + LUM[1] * pixel[1] + LUM[2] * pixel[2];
}

// Swapping this.thread.x and this.thread.y reads the transpose of the image —
// the single most common mistake in the image tasks, and invisible to tests
// that only check greyness or averages. These results come back as plain 2D
// arrays, so the cell under test is known exactly: a cell holding the value
// that belongs to cell [x][y] is that swapped read. Cells on the diagonal
// (y === x) are their own transpose and can never show it, which is why the
// case lists below also probe off-diagonal cells.
function transposeCellHint(got, transposed, eps, y, x) {
  return Math.abs(got - transposed) <= eps
    ? `that is the value for cell [${x}][${y}] — this.thread.x and this.thread.y are ` +
      `swapped. Rows come first: image[this.thread.y][this.thread.x]`
    : null;
}

// Mode-safe read of a pipeline result: Texture on GL, plain array on CPU.
function toArr(value) {
  return value && typeof value.toArray === 'function' ? value.toArray() : value;
}

// Deterministic 256-sample signal in 0–1 (task 1).
function makeSignal01(utils, seed = 1701) {
  const rand = utils.seededRandom(seed);
  const data = new Array(256);
  for (let i = 0; i < 256; i++) data[i] = Math.round(rand() * 100) / 100;
  return data;
}

// Deterministic 256-sample raw signal in 0–10 (task 3).
function makeRawSignal(utils, seed = 2026) {
  const rand = utils.seededRandom(seed);
  const data = new Array(256);
  for (let i = 0; i < 256; i++) data[i] = Math.round(rand() * 1000) / 100;
  return data;
}

// All-zero field with one hot cell (task 4).
function makeSpike(size, at, magnitude) {
  const field = new Array(size).fill(0);
  field[at] = magnitude;
  return field;
}

// Constant-color image for orientation-independent pixel tests. Same shape as
// the task's own input: an ImageData whose .plain/.at(x, y) answer
// image[y][x] = [r, g, b, a], with the pixel quantized to what 8-bit channels can
// hold (so expectations read off .at() are exact). Never hand a kernel built with
// an ImageData a nested array instead — on WebGL2 that silently reads garbage.
function constantImage(size, pixel) {
  const row = new Array(size).fill(quantizePixel(pixel));
  return plainToImageData(new Array(size).fill(row));
}

// Task 3 reference: normalize (/10) → gamma (v²) → 3-tap smooth.
function refChain3(signal) {
  const g = signal.map(v => {
    const n = v / 10;
    return n * n;
  });
  const out = new Array(g.length);
  for (let i = 0; i < g.length; i++) {
    const l = Math.max(i - 1, 0);
    const r = Math.min(i + 1, g.length - 1);
    out[i] = (g[l] + g[i] + g[r]) / 3;
  }
  return out;
}

// Task 4 reference: 12-step 1D diffusion, edges held fixed.
function refDiffuse(field, steps) {
  let prev = field.slice();
  const n = prev.length;
  for (let s = 0; s < steps; s++) {
    const next = prev.slice();
    for (let x = 1; x < n - 1; x++) {
      next[x] = 0.25 * prev[x - 1] + 0.5 * prev[x] + 0.25 * prev[x + 1];
    }
    prev = next;
  }
  return prev;
}

// Task 5 references: luminance map, then clamped 3×3 box blur. `image` is a
// course image (an ImageData); .plain is its host-side image[y][x] view.
function refLuminanceMap(image) {
  return image.plain.map(row => row.map(luminanceOf));
}

function refBlur3(map) {
  const size = map.length;
  const out = new Array(size);
  for (let y = 0; y < size; y++) {
    out[y] = new Array(size);
    for (let x = 0; x < size; x++) {
      let sum = 0;
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          const yy = Math.min(Math.max(y + dy, 0), size - 1);
          const xx = Math.min(Math.max(x + dx, 0), size - 1);
          sum += map[yy][xx];
        }
      }
      out[y][x] = sum / 9;
    }
  }
  return out;
}

function contrastOf(l) {
  return Math.min(Math.max((l - 0.5) * 2 + 0.5, 0), 1);
}

// ---- near-miss diagnosis --------------------------------------------------
//
// The transpose hint above is one instance of a general idea: when a failing
// value is exactly what some specific mistake would produce, name that mistake
// instead of reporting two numbers. A probe pairs such a value with its
// sentence; diagnose() speaks only when the observed value matches a probe
// within the test's own tolerance AND the correct value does not — so cells
// where a candidate happens to equal the right answer (a stretch that never
// left 0–1, where clamping changes nothing) stay silent, as do observations
// matching probes that disagree with each other. A wrong diagnosis is worse
// than none.
function diagnose(got, expected, eps, probes) {
  const hits = probes
    .filter(p => Math.abs(got - p[0]) <= eps && Math.abs(expected - p[0]) > eps)
    .map(p => p[1]);
  return hits.length && hits.every(m => m === hits[0]) ? hits[0] : null;
}

// Task 2: the contrast stretch un-clamped, un-centred, or never applied.
function contrastProbes(l) {
  return [
    [(l - 0.5) * 2 + 0.5,
      'that is the stretch without its clamp — Math.min / Math.max keep the result inside 0–1'],
    [l,
      'that is the luminance unchanged — the stretch never reached the return value'],
    [Math.min(Math.max(l * 2, 0), 1),
      'the midpoint is missing — subtract 0.5 before doubling and add it back afterwards'],
  ];
}

// Task 5: a blur pass that does nothing, or one that never divides by 9. A pass
// that forgets to clamp reads off the map and comes back as a non-number.
function blurHint(got, ref, unblurred, eps, y, x) {
  if (!Number.isFinite(got)) {
    return 'that cell read past the edge of the map — clamp yy and xx into 0…63 before indexing';
  }
  return diagnose(got, ref[y][x], eps, [
    [unblurred[y][x], 'that is the unblurred luminance — the 3×3 average never happened'],
    [9 * ref[y][x], 'that is the sum of the nine samples — a mean divides by 9'],
  ]);
}

export default {
  id: '1-4',
  track: 1,
  title: 'Pipelines & Textures',
  blurb: 'Chaining kernels so data stays on the GPU — the single biggest real-world speedup.',
  tasks: [
    // ------------------------------------------------------------------ 1
    {
      slug: 'pipeline-on',
      title: 'Flip On the Pipeline',
      intro: `<p>Until now, every kernel call ended the same way: the GPU finished computing,
        then the whole result was <strong>downloaded back to JavaScript</strong> as a typed array.
        That download is the expensive part — for a 512×512 grid it's a megabyte crossing the
        bus on every single call.</p>
        <p><code>pipeline: true</code> changes the ending. The kernel still runs the same, but the
        result <em>stays in GPU memory</em>, and what you get back is a <strong>texture</strong> —
        a lightweight handle to data that never left the card. Log one and you'll see an object,
        not numbers. When you actually want the values, you ask for the download explicitly with
        <code>.toArray()</code>.</p>
        <p>One backend wrinkle to know: the CPU backend has no textures, so there a pipeline
        kernel hands back a plain array. Mode-safe code uses the same guard gpu.js uses
        internally: <code>result.toArray ? result.toArray() : result</code>.</p>`,
      goal: `<strong>Goal:</strong> make the <code>boost</code> kernel a pipeline kernel, then
        download its result explicitly and log the first sample.`,
      requirements: [
        'Add <code>pipeline: true</code> to the kernel settings',
        'Log the raw result — see what a texture looks like in the console',
        'Download the values with <code>.toArray()</code>, using the mode-safe guard',
        `Log the first value as <code>console.log('first sample:', values[0])</code>`,
      ],
      hints: [
        {
          title: 'Hint 1 — where does the flag go?',
          body: `<p><code>pipeline: true</code> sits in the settings object, right next to
            <code>output</code>. Nothing about the kernel function itself changes.</p>`,
        },
        {
          title: 'Hint 2 — the mode-safe download',
          body: `<pre><code>const values = result.toArray ? result.toArray() : result;</code></pre>
<p>On the GL backend this calls <code>toArray()</code>; on the CPU backend
            <code>result</code> is already an array and passes through untouched.</p>`,
        },
      ],
      transfer: `A gpu.js texture is the same idea as a <code>GPUBuffer</code> you never map in
        WebGPU, or device memory behind a pointer in CUDA and ROCm: the data has an address on
        the card, and JavaScript only holds the ticket stub. <code>.toArray()</code> is the
        explicit "map it back to the host" step.`,
      starterCode: `// Run this as-is first: the kernel returns plain numbers, which means
// every call ships the whole result back to JavaScript. Let's stop that.
const gpu = new GPU({ mode });

const boost = gpu.createKernel(function (signal) {
  return Math.min(signal[this.thread.x] * 1.5, 1);
}, {
  output: [256],
  // TODO: keep the result on the GPU
});

const result = boost(signal);
console.log(result);

// TODO: \`result\` is about to become a texture. Download the values
// explicitly (mode-safe: result.toArray ? result.toArray() : result)
// and log the first one as:  console.log('first sample:', values[0]);
`,
      solutionCode: `// pipeline: true — the result stays in GPU memory as a texture.
const gpu = new GPU({ mode });

const boost = gpu.createKernel(function (signal) {
  return Math.min(signal[this.thread.x] * 1.5, 1);
}, {
  output: [256],
  pipeline: true,
});

const result = boost(signal);
console.log(result); // GL backend: a Texture object — no numbers in sight

// The explicit download. CPU backend already returns a plain array,
// so guard the call — the same trick gpu.js uses internally.
const values = result.toArray ? result.toArray() : result;
console.log('first sample:', values[0]);
`,
      inputs: utils => ({ signal: makeSignal01(utils) }),
      publicTests: [
        {
          name: 'kernel is created with <code>pipeline: true</code>',
          run: async ctx => {
            ctx.assert(ctx.kernels.length >= 1, 'no kernel was created — call gpu.createKernel()');
            ctx.assert(
              ctx.kernel.kernel && ctx.kernel.kernel.pipeline === true,
              'the kernel is not a pipeline kernel — add pipeline: true to its settings'
            );
          },
        },
        {
          name: 'boosted values read back correctly through <code>toArray()</code>',
          run: async ctx => {
            const arr = new Array(256);
            for (let i = 0; i < 256; i++) arr[i] = (i % 100) / 100;
            const out = toArr(ctx.kernel(arr));
            ctx.assert(out && out.length === 256, `expected 256 values, got ${out && out.length}`);
            for (let i = 0; i < 256; i++) {
              ctx.assertClose(out[i], Math.min(arr[i] * 1.5, 1), 2e-3, `sample ${i}`);
            }
          },
        },
        {
          name: `the downloaded first sample is logged as 'first sample:'`,
          run: async ctx => {
            const logged = ctx.logs.some(
              line => line.type === 'log' && line.text && line.text.includes('first sample')
            );
            ctx.assert(
              logged,
              `expected a console.log('first sample:', values[0]) after the download`
            );
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            const signal = makeSignal01(ctx.utils, 8842);
            const out = toArr(ctx.kernel(signal));
            ctx.assert(out.length === 256, 'expected 256 values');
            for (let i = 0; i < 256; i++) {
              ctx.assertClose(out[i], Math.min(signal[i] * 1.5, 1), 2e-3, `sample ${i}`);
            }
          },
        },
      ],
    },

    // ------------------------------------------------------------------ 2
    {
      slug: 'chain-two-kernels',
      title: 'Chain Kernels, Skip the Round Trip',
      intro: `<p>Here's the payoff of textures: a texture returned by one kernel can be passed
        <strong>straight into the next kernel</strong> as an argument. gpu.js binds the texture
        as the input — no download, no re-upload, no JavaScript in the middle. The data makes
        the whole trip without ever leaving the card.</p>
        <p>In module 1.2 you chained two kernels through JavaScript: the luminance map came
        back as arrays, then went up again for the second pass. Same chain below — except this
        time <code>luminance</code> is a pipeline kernel, and the second pass eats its texture
        directly.</p>
        ${ARRAY_LAYOUT}`,
      goal: `<strong>Goal:</strong> finish the <code>contrast</code> kernel — stretch each
        luminance value around the midpoint with <code>(l − 0.5) × 2 + 0.5</code>, clamped
        to 0–1 — and keep the texture handoff intact.`,
      requirements: [
        'Keep <code>luminance</code> a pipeline kernel — its result never touches JavaScript',
        'Pass the returned texture directly into <code>contrast</code> (already wired up)',
        'In <code>contrast</code>, return <code>(l - 0.5) * 2 + 0.5</code> clamped with <code>Math.min</code> / <code>Math.max</code>',
      ],
      hints: [
        {
          title: 'Hint 1 — textures index like arrays',
          body: `<p>Inside <code>contrast</code>, the texture argument reads exactly like the
            2D arrays you already know: <code>map[this.thread.y][this.thread.x]</code>.
            The kernel doesn't care where the data lives.</p>`,
        },
        {
          title: 'Hint 2 — the clamp',
          body: `<pre><code>return Math.min(Math.max((l - 0.5) * 2 + 0.5, 0), 1);</code></pre>`,
        },
      ],
      transfer: `Handing a texture from kernel to kernel is what CUDA does when consecutive
        launches read and write the same device pointers, and what a WebGPU compute pass does
        when one dispatch's storage buffer becomes the next dispatch's binding. On Metal it's
        two encoders sharing an <code>MTLBuffer</code>. Nobody copies to the CPU in between.`,
      starterCode: `const gpu = new GPU({ mode });

// Pass 1 — luminance map, kept on the GPU as a texture.
const luminance = gpu.createKernel(function (photo) {
  const pixel = photo[this.thread.y][this.thread.x];
  return 0.299 * pixel[0] + 0.587 * pixel[1] + 0.114 * pixel[2];
}, { output: [64, 64], pipeline: true });

// Pass 2 — contrast stretch. Final stage, so it returns plain numbers.
const contrast = gpu.createKernel(function (map) {
  const l = map[this.thread.y][this.thread.x];
  // TODO: stretch around the midpoint — (l - 0.5) * 2 + 0.5 —
  // clamped to 0–1 with Math.min / Math.max
  return l;
}, { output: [64, 64] });

const mapTexture = luminance(photo); // a texture — still on the GPU
const result = contrast(mapTexture); // and straight back in it goes
console.log('center cell:', result[32][32]);
`,
      solutionCode: `const gpu = new GPU({ mode });

// Pass 1 — luminance map, kept on the GPU as a texture.
const luminance = gpu.createKernel(function (photo) {
  const pixel = photo[this.thread.y][this.thread.x];
  return 0.299 * pixel[0] + 0.587 * pixel[1] + 0.114 * pixel[2];
}, { output: [64, 64], pipeline: true });

// Pass 2 — contrast stretch. Final stage, so it returns plain numbers.
const contrast = gpu.createKernel(function (map) {
  const l = map[this.thread.y][this.thread.x];
  return Math.min(Math.max((l - 0.5) * 2 + 0.5, 0), 1);
}, { output: [64, 64] });

const mapTexture = luminance(photo); // a texture — still on the GPU
const result = contrast(mapTexture); // and straight back in it goes
console.log('center cell:', result[32][32]);
`,
      inputs: utils => ({ photo: utils.makeTestImage(64) }),
      publicTests: [
        {
          name: 'two kernels: a pipeline pass feeding a plain pass',
          run: async ctx => {
            ctx.assert(ctx.kernels.length >= 2, `expected 2 kernels, found ${ctx.kernels.length}`);
            const piped = ctx.kernels.find(k => k.kernel && k.kernel.pipeline);
            const plain = ctx.kernels.find(k => k.kernel && !k.kernel.pipeline);
            ctx.assert(piped, 'no pipeline kernel found — keep pipeline: true on luminance');
            ctx.assert(plain, 'no plain kernel found — contrast should NOT be a pipeline kernel');
            if (ctx.resolvedMode === 'gpu') {
              ctx.assert(
                plain.lastArgs && plain.lastArgs[0] && typeof plain.lastArgs[0].toArray === 'function',
                'contrast should be fed the texture itself — no .toArray() in between'
              );
            }
          },
        },
        {
          name: 'chained result: clamped <code>(l - 0.5) * 2 + 0.5</code> per cell',
          run: async ctx => {
            const piped = ctx.kernels.find(k => k.kernel && k.kernel.pipeline);
            const plain = ctx.kernels.find(k => k.kernel && !k.kernel.pipeline);
            ctx.assert(piped && plain, 'expected a pipeline kernel and a plain kernel');
            const img = ctx.utils.makeTestImage(64);
            const out = plain(piped(img));
            const pixels = img.plain; // host-side view of the same image
            const cases = [[0, 0], [7, 41], [32, 32], [63, 63]];
            for (const [y, x] of cases) {
              const l = luminanceOf(pixels[y][x]);
              const hint = transposeCellHint(
                out[y][x], contrastOf(luminanceOf(pixels[x][y])), 3e-3, y, x
              ) || diagnose(out[y][x], contrastOf(l), 3e-3, contrastProbes(l));
              ctx.assertClose(
                out[y][x], contrastOf(l), 3e-3, hint || `cell [${y}][${x}]`
              );
            }
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            const piped = ctx.kernels.find(k => k.kernel && k.kernel.pipeline);
            const plain = ctx.kernels.find(k => k.kernel && !k.kernel.pipeline);
            ctx.assert(piped && plain, 'expected a pipeline kernel and a plain kernel');
            const image = constantImage(64, [0.8, 0.3, 0.5, 1]);
            const luminance = luminanceOf(image.at(0, 0));
            const expected = contrastOf(luminance);
            const out = plain(piped(image));
            const probes = contrastProbes(luminance);
            for (let y = 0; y < 64; y++) {
              for (let x = 0; x < 64; x++) {
                const hint = diagnose(out[y][x], expected, 3e-3, probes);
                ctx.assertClose(out[y][x], expected, 3e-3, hint || `cell [${y}][${x}]`);
              }
            }
          },
        },
      ],
    },

    // ------------------------------------------------------------------ 3
    {
      slug: 'tollbooth',
      title: 'toArray() Is a Tollbooth',
      intro: `<p>Here's the mental model that makes GPU code fast: computation on the card is
        nearly free — it's the <strong>transfers</strong> that cost. Every kernel that is
        <em>not</em> <code>pipeline: true</code> ends with an implicit download, and passing
        that array to the next kernel triggers a re-upload. A three-stage chain without
        pipelines pays the toll <strong>four times</strong> for one result.</p>
        <p>The starter below is a fully working three-stage audio chain — normalize, gamma,
        smooth — and every hop goes through JavaScript. Your job isn't to fix the math.
        It's to fix the traffic: intermediates become pipeline kernels, and only the
        <em>final</em> stage returns plain numbers. The chain call itself shouldn't change
        by a single character.</p>`,
      goal: `<strong>Goal:</strong> refactor the chain so stages 1 and 2 keep their results on
        the GPU, the final stage returns numbers, and the output is bit-for-bit the same idea —
        just without the round trips.`,
      requirements: [
        'Make <code>normalize</code> and <code>gamma</code> pipeline kernels',
        'Leave <code>smooth</code> as a plain kernel — the one download you actually want',
        'Do not change the chain: <code>smooth(gamma(normalize(signal)))</code> stays as-is',
      ],
      hints: [
        {
          title: 'Hint 1 — where is the readback hiding?',
          body: `<p>There's no <code>.toArray()</code> in the starter, but the readbacks are
            still there: a non-pipeline kernel's <em>return value</em> is the readback.
            Count them: normalize downloads, gamma re-uploads and downloads, smooth re-uploads.</p>`,
        },
        {
          title: 'Hint 2 — a two-line diff',
          body: `<p>Add <code>pipeline: true</code> to the settings of <code>normalize</code>
            and <code>gamma</code>. That's the entire refactor — the chain line already does
            the right thing once textures flow through it.</p>`,
        },
      ],
      transfer: `Profile any real CUDA or ROCm app and the widest bars are often
        <code>cudaMemcpy</code> DtoH/HtoD, not kernels; in WebGPU the same toll is
        <code>mapAsync</code> plus staging-buffer copies. "Keep data resident, read back once
        at the end" is performance rule number one on every GPU platform.`,
      starterCode: `const gpu = new GPU({ mode });

// Stage 1 — scale the raw 0–10 signal down to 0–1.
const normalize = gpu.createKernel(function (signal) {
  return signal[this.thread.x] / 10;
}, { output: [256] }); // TODO: this intermediate should stay on the GPU

// Stage 2 — gamma curve to tame the loud parts.
const gamma = gpu.createKernel(function (v) {
  return v[this.thread.x] * v[this.thread.x];
}, { output: [256] }); // TODO: so should this one

// Stage 3 — 3-tap smoothing. Final stage: plain numbers out, on purpose.
const smooth = gpu.createKernel(function (v) {
  let left = this.thread.x - 1;
  let right = this.thread.x + 1;
  if (left < 0) left = 0;
  if (right > 255) right = 255;
  return (v[left] + v[this.thread.x] + v[right]) / 3;
}, { output: [256] });

// This chain is CORRECT — and slow. Each non-pipeline return is a full
// GPU → JS download, and the next call re-uploads it. Four transfers.
const out = smooth(gamma(normalize(signal)));
console.log('smoothed[0]:', out[0]);
`,
      solutionCode: `const gpu = new GPU({ mode });

// Stage 1 — scale the raw 0–10 signal down to 0–1.
const normalize = gpu.createKernel(function (signal) {
  return signal[this.thread.x] / 10;
}, { output: [256], pipeline: true });

// Stage 2 — gamma curve to tame the loud parts.
const gamma = gpu.createKernel(function (v) {
  return v[this.thread.x] * v[this.thread.x];
}, { output: [256], pipeline: true });

// Stage 3 — 3-tap smoothing. Final stage: plain numbers out, on purpose.
const smooth = gpu.createKernel(function (v) {
  let left = this.thread.x - 1;
  let right = this.thread.x + 1;
  if (left < 0) left = 0;
  if (right > 255) right = 255;
  return (v[left] + v[this.thread.x] + v[right]) / 3;
}, { output: [256] });

// Identical chain, one transfer in, one out. The code didn't change —
// the data's home address did.
const out = smooth(gamma(normalize(signal)));
console.log('smoothed[0]:', out[0]);
`,
      inputs: utils => ({ signal: makeRawSignal(utils) }),
      publicTests: [
        {
          name: 'stages 1–2 are pipeline kernels; the final stage is not',
          run: async ctx => {
            ctx.assert(ctx.kernels.length >= 3, `expected 3 kernels, found ${ctx.kernels.length}`);
            const [a, b, c] = ctx.kernels;
            ctx.assert(a.kernel && a.kernel.pipeline === true, 'normalize should have pipeline: true');
            ctx.assert(b.kernel && b.kernel.pipeline === true, 'gamma should have pipeline: true');
            ctx.assert(c.kernel && !c.kernel.pipeline, 'smooth should stay a plain kernel — its return IS the readback you want');
            if (ctx.resolvedMode === 'gpu') {
              ctx.assert(
                b.lastArgs && b.lastArgs[0] && typeof b.lastArgs[0].toArray === 'function',
                'gamma should receive a texture from normalize, not an array'
              );
            }
          },
        },
        {
          name: 'the numbers survive the refactor — chain output is unchanged',
          run: async ctx => {
            const [a, b, c] = ctx.kernels;
            const signal = makeRawSignal(ctx.utils);
            const out = c(b(a(signal)));
            const ref = refChain3(signal);
            ctx.assert(out && out.length === 256, `expected 256 values, got ${out && out.length}`);
            for (let i = 0; i < 256; i++) {
              ctx.assertClose(out[i], ref[i], 3e-3, `sample ${i}`);
            }
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            const [a, b, c] = ctx.kernels;
            const signal = makeRawSignal(ctx.utils, 5150);
            const out = c(b(a(signal)));
            const ref = refChain3(signal);
            for (let i = 0; i < 256; i++) {
              ctx.assertClose(out[i], ref[i], 3e-3, `sample ${i}`);
            }
          },
        },
      ],
    },

    // ------------------------------------------------------------------ 4
    {
      slug: 'iterate-immutable',
      title: 'Feedback Loops: immutable Textures',
      intro: `<p>Simulations don't run once — they <strong>step</strong>: the output of step
        <em>n</em> is the input of step <em>n</em>+1. With pipelines that means feeding a
        kernel its own texture back. Try it naively and gpu.js stops you cold:
        <em>"Source and destination … are the same. Use immutable = true"</em> — the kernel
        would be reading the very texture it's writing to.</p>
        <p><code>immutable: true</code> is the fix: each call renders to a <em>fresh</em>
        texture instead of recycling one, so last step's output is safe to read while this
        step writes. (In long-running sims you'd call <code>texture.delete()</code> on old
        steps to recycle their memory — at 128 cells here, we'll let them slide.)</p>
        <p>Below is a 1D heat field: 128 cells, all cold except one hot spike. One diffusion
        step moves each cell toward its neighbours. Twelve steps stay entirely on the GPU —
        one upload at the start, one download at the end.</p>`,
      goal: `<strong>Goal:</strong> make the feedback loop legal — the <code>step</code> kernel
        needs <code>immutable: true</code> — and run 12 diffusion steps without the heat ever
        visiting JavaScript.`,
      requirements: [
        'Add <code>immutable: true</code> to the <code>step</code> kernel (keep <code>pipeline: true</code>)',
        'Keep the loop feeding <code>step</code>\'s output straight back in — no readbacks inside it',
        'After 12 steps, download once and log the peak at cell 64',
      ],
      hints: [
        {
          title: 'Hint 1 — read the error message',
          body: `<p>Run the starter as-is. The error names both the crime and the sentence:
            the kernel's input and output are the same storage, and <code>immutable = true</code>
            is the fix. gpu.js error messages are unusually honest.</p>`,
        },
        {
          title: 'Hint 2 — why upload() exists',
          body: `<p>The tiny <code>upload</code> kernel copies the seed array into a texture
            once, so <code>step</code> always sees texture inputs from its very first call.
            Keeping argument types stable means the kernel compiles exactly once.</p>`,
        },
        {
          title: 'Hint 3 — the one-word diff',
          body: `<p>In <code>step</code>'s settings:</p>
<pre><code>{ output: [128], pipeline: true, immutable: true }</code></pre>
<p>The loop is already correct.</p>`,
        },
      ],
      transfer: `Every GPU API solves read-write hazards the same way gpu.js just made you do:
        ping-pong buffering. WebGPU compute passes swap two storage buffers each dispatch,
        CUDA stencil codes swap <code>in</code>/<code>out</code> device pointers, Metal
        simulations flip between two textures. <code>immutable: true</code> is ping-ponging
        with the bookkeeping done for you.`,
      starterCode: `const gpu = new GPU({ mode });

// Upload pass — copies the seed array into a texture, once.
const upload = gpu.createKernel(function (seed) {
  return seed[this.thread.x];
}, { output: [128], pipeline: true });

// One diffusion step: each cell relaxes toward its neighbours.
// Edge cells hold their value.
const step = gpu.createKernel(function (heat) {
  const x = this.thread.x;
  if (x === 0 || x === 127) {
    return heat[x];
  }
  return 0.25 * heat[x - 1] + 0.5 * heat[x] + 0.25 * heat[x + 1];
}, {
  output: [128],
  pipeline: true,
  // TODO: this kernel reads its own previous output — run it and
  // let the error message tell you the missing setting.
});

let state = upload(field);
for (let i = 0; i < 12; i++) {
  state = step(state); // output straight back in — a feedback loop
}

const heat = state.toArray ? state.toArray() : state;
console.log('peak after 12 steps:', heat[64]);
`,
      solutionCode: `const gpu = new GPU({ mode });

// Upload pass — copies the seed array into a texture, once.
const upload = gpu.createKernel(function (seed) {
  return seed[this.thread.x];
}, { output: [128], pipeline: true });

// One diffusion step: each cell relaxes toward its neighbours.
// Edge cells hold their value.
const step = gpu.createKernel(function (heat) {
  const x = this.thread.x;
  if (x === 0 || x === 127) {
    return heat[x];
  }
  return 0.25 * heat[x - 1] + 0.5 * heat[x] + 0.25 * heat[x + 1];
}, {
  output: [128],
  pipeline: true,
  immutable: true, // fresh output texture per call — feedback is now safe
});

let state = upload(field);
for (let i = 0; i < 12; i++) {
  state = step(state); // output straight back in — a feedback loop
}

const heat = state.toArray ? state.toArray() : state;
console.log('peak after 12 steps:', heat[64]);
`,
      inputs: () => ({ field: makeSpike(128, 64, 1) }),
      publicTests: [
        {
          name: 'the stepping kernel opts into <code>immutable: true</code>',
          run: async ctx => {
            ctx.assert(ctx.kernels.length >= 2, `expected 2 kernels, found ${ctx.kernels.length}`);
            const step = ctx.kernels.find(k => k.kernel && k.kernel.immutable);
            ctx.assert(
              step,
              'no immutable kernel found — the feedback loop needs immutable: true on step'
            );
            ctx.assert(step.kernel.pipeline === true, 'step should keep pipeline: true too');
          },
        },
        {
          name: 'twelve steps match a reference diffusion of the spike',
          run: async ctx => {
            const upload = ctx.kernels.find(k => k.kernel && k.kernel.pipeline && !k.kernel.immutable);
            const step = ctx.kernels.find(k => k.kernel && k.kernel.immutable);
            ctx.assert(upload && step, 'expected an upload kernel and an immutable step kernel');
            const seed = makeSpike(128, 64, 1);
            let state = upload(seed);
            for (let i = 0; i < 12; i++) state = step(state);
            const heat = toArr(state);
            const ref = refDiffuse(seed, 12);
            for (let x = 0; x < 128; x++) {
              ctx.assertClose(heat[x], ref[x], 2e-3, `cell ${x}`);
            }
          },
        },
        {
          name: 'heat is conserved — the field still sums to 1.0',
          run: async ctx => {
            const upload = ctx.kernels.find(k => k.kernel && k.kernel.pipeline && !k.kernel.immutable);
            const step = ctx.kernels.find(k => k.kernel && k.kernel.immutable);
            ctx.assert(upload && step, 'expected an upload kernel and an immutable step kernel');
            let state = upload(makeSpike(128, 64, 1));
            for (let i = 0; i < 12; i++) state = step(state);
            const heat = toArr(state);
            let sum = 0;
            for (let x = 0; x < 128; x++) sum += heat[x];
            ctx.assertClose(sum, 1, 1e-2, 'total heat in the field');
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            // Different spike: position 40, magnitude 0.75 — hardcoding the
            // public expectations won't survive this.
            const upload = ctx.kernels.find(k => k.kernel && k.kernel.pipeline && !k.kernel.immutable);
            const step = ctx.kernels.find(k => k.kernel && k.kernel.immutable);
            ctx.assert(upload && step, 'expected an upload kernel and an immutable step kernel');
            const seed = makeSpike(128, 40, 0.75);
            let state = upload(seed);
            for (let i = 0; i < 12; i++) state = step(state);
            const heat = toArr(state);
            const ref = refDiffuse(seed, 12);
            let sum = 0;
            for (let x = 0; x < 128; x++) {
              ctx.assertClose(heat[x], ref[x], 2e-3, `cell ${x}`);
              sum += heat[x];
            }
            ctx.assertClose(sum, 0.75, 1e-2, 'total heat in the field');
          },
        },
      ],
    },

    // ------------------------------------------------------------------ 5
    {
      slug: 'photo-to-screen',
      title: 'The Payoff: Photo to Screen, Zero Readbacks',
      intro: `<p>Time to cash in the whole module. In module 1.2's finale, a two-kernel chain
        hauled the luminance map down to JavaScript and back up again — two transfers it didn't
        need. This pipeline does more work with <em>fewer</em> transfers: photo →
        <strong>luminance</strong> → <strong>3×3 blur</strong> → <strong>painted canvas</strong>,
        and after the photo is uploaded, nothing comes back. The graphical kernel eats the blur
        texture and writes pixels; readbacks: zero.</p>
        <p>The missing piece is the blur. Each cell averages its 3×3 neighbourhood — two little
        loops over <code>dy</code>/<code>dx</code>, indices clamped to 0…63 so the edges don't
        read out of bounds. When it works, hit <strong>Benchmark</strong> and watch what
        keeping data on the card does to the gap.</p>
        ${ARRAY_LAYOUT}`,
      goal: `<strong>Goal:</strong> implement the 3×3 box blur so the full three-pass pipeline —
        two texture passes and a graphical finale — runs with zero readbacks.`,
      requirements: [
        'Blur: average the 3×3 neighbourhood, clamping indices to 0…63 at the edges',
        'Both <code>luminance</code> and <code>blur</code> stay <code>pipeline: true</code>',
        'The graphical pass is fed the blur <em>texture</em> — nothing is downloaded',
        'Render the result with <code>render(paint.canvas)</code>',
      ],
      hints: [
        {
          title: 'Hint 1 — the neighbourhood loops',
          body: `<p>Two nested loops with fixed bounds are fine in a kernel:
            <code>for (let dy = -1; dy &lt;= 1; dy++)</code> and the same for <code>dx</code>.
            Accumulate into a <code>sum</code>, return <code>sum / 9</code>.</p>`,
        },
        {
          title: 'Hint 2 — clamping the edges',
          body: `<p>Compute <code>let yy = this.thread.y + dy;</code> then push it back in
            range:</p>
<pre><code>if (yy &lt; 0) yy = 0;
if (yy &gt; 63) yy = 63;</code></pre>
<p>Same for <code>xx</code>. Corner cells just count some neighbours twice.</p>`,
        },
        {
          title: 'Hint 3 — the whole body',
          body: `<p><code>let sum = 0;</code> then inside the loops
            <code>sum += map[yy][xx];</code> and finally <code>return sum / 9;</code> —
            the clamped <code>yy</code>/<code>xx</code> from hint 2 do the rest.</p>`,
        },
      ],
      transfer: `You just built what engine programmers call a render graph: named passes,
        explicit dependencies, all resources resident on the GPU — the architecture behind
        Frostbite's frame graph, CUDA Graphs' pre-recorded launch chains, and a Metal command
        buffer full of encoder passes. Real engines are this task with more boxes.`,
      starterCode: `const gpu = new GPU({ mode });

// Pass 1 — luminance map. You've written this one twice already.
const luminance = gpu.createKernel(function (photo) {
  const pixel = photo[this.thread.y][this.thread.x];
  return 0.299 * pixel[0] + 0.587 * pixel[1] + 0.114 * pixel[2];
}, { output: [64, 64], pipeline: true });

// Pass 2 — 3×3 box blur. Currently a do-nothing passthrough.
const blur = gpu.createKernel(function (map) {
  // TODO: average the 3×3 neighbourhood around this cell.
  // Clamp indices to 0…63 so edges don't read out of bounds.
  return map[this.thread.y][this.thread.x];
}, { output: [64, 64], pipeline: true });

// Pass 3 — paint the blurred map. Texture in, pixels out.
const paint = gpu.createKernel(function (map) {
  const l = map[this.thread.y][this.thread.x];
  this.color(l, l, l, 1);
}, { output: [64, 64], graphical: true });

// The whole pipeline: after \`photo\` goes up, nothing comes back down.
paint(blur(luminance(photo)));
render(paint.canvas);
`,
      solutionCode: `const gpu = new GPU({ mode });

// Pass 1 — luminance map. You've written this one twice already.
const luminance = gpu.createKernel(function (photo) {
  const pixel = photo[this.thread.y][this.thread.x];
  return 0.299 * pixel[0] + 0.587 * pixel[1] + 0.114 * pixel[2];
}, { output: [64, 64], pipeline: true });

// Pass 2 — 3×3 box blur over the luminance texture.
const blur = gpu.createKernel(function (map) {
  let sum = 0;
  for (let dy = -1; dy <= 1; dy++) {
    for (let dx = -1; dx <= 1; dx++) {
      let yy = this.thread.y + dy;
      let xx = this.thread.x + dx;
      if (yy < 0) yy = 0;
      if (yy > 63) yy = 63;
      if (xx < 0) xx = 0;
      if (xx > 63) xx = 63;
      sum += map[yy][xx];
    }
  }
  return sum / 9;
}, { output: [64, 64], pipeline: true });

// Pass 3 — paint the blurred map. Texture in, pixels out.
const paint = gpu.createKernel(function (map) {
  const l = map[this.thread.y][this.thread.x];
  this.color(l, l, l, 1);
}, { output: [64, 64], graphical: true });

// The whole pipeline: after \`photo\` goes up, nothing comes back down.
paint(blur(luminance(photo)));
render(paint.canvas);
`,
      inputs: utils => ({ photo: utils.makeTestImage(64) }),
      publicTests: [
        {
          name: 'three passes: two texture kernels feeding a graphical finale',
          run: async ctx => {
            ctx.assert(ctx.kernels.length >= 3, `expected 3 kernels, found ${ctx.kernels.length}`);
            const [lum, blur, paint] = ctx.kernels;
            ctx.assert(lum.kernel && lum.kernel.pipeline === true, 'luminance should have pipeline: true');
            ctx.assert(blur.kernel && blur.kernel.pipeline === true, 'blur should have pipeline: true');
            ctx.assert(paint.kernel && paint.kernel.graphical, 'the third kernel should be graphical');
            ctx.assert(ctx.canvas, 'no canvas — did you call render(paint.canvas)?');
            ctx.assert(
              ctx.canvas.width === 64 && ctx.canvas.height === 64,
              `expected a 64×64 canvas, got ${ctx.canvas.width}×${ctx.canvas.height}`
            );
            if (ctx.resolvedMode === 'gpu') {
              ctx.assert(
                paint.lastArgs && paint.lastArgs[0] && typeof paint.lastArgs[0].toArray === 'function',
                'paint should be fed the blur texture directly — zero readbacks'
              );
            }
          },
        },
        {
          name: 'blur pass: each cell is the mean of its 3×3 neighbourhood',
          run: async ctx => {
            const [lum, blur] = ctx.kernels;
            const img = ctx.utils.makeTestImage(64);
            const out = toArr(blur(lum(img)));
            const map = refLuminanceMap(img);
            const ref = refBlur3(map);
            // interior, all four edges, and a corner
            const cases = [[32, 32], [0, 20], [63, 20], [20, 0], [20, 63], [0, 0], [10, 47]];
            for (const [y, x] of cases) {
              // a box blur commutes with the transpose, so a swapped read in
              // either pass lands the blurred cell [x][y] here
              const hint = transposeCellHint(out[y][x], ref[x][y], 3e-3, y, x) ||
                blurHint(out[y][x], ref, map, 3e-3, y, x);
              ctx.assertClose(out[y][x], ref[y][x], 3e-3, hint || `cell [${y}][${x}]`);
            }
          },
        },
        {
          name: 'painted canvas is monochrome',
          run: async ctx => {
            const pixels = ctx.getPixels();
            ctx.assert(pixels.length === 64 * 64 * 4, 'pixel buffer should hold 64×64 RGBA values');
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
            // Blur of a constant field is the same constant — end-to-end on a
            // constant image every painted pixel must be its luminance,
            // whatever the row order.
            const [lum, blur, paint] = ctx.kernels;
            const image = constantImage(64, [0.35, 0.65, 0.15, 1]);
            const expected = luminanceOf(image.at(0, 0)) * 255;
            paint(blur(lum(image)));
            const pixels = paint.getPixels();
            for (let i = 0; i < pixels.length; i += 149 * 4) {
              ctx.assertClose(pixels[i], expected, 2, `red at byte ${i}`);
              ctx.assertClose(pixels[i + 1], expected, 2, `green at byte ${i}`);
              ctx.assertClose(pixels[i + 2], expected, 2, `blue at byte ${i}`);
            }
          },
        },
        {
          name: 'private test #2',
          run: async ctx => {
            // Fresh image through the numeric passes: full-field check of the
            // blurred luminance, independent of the canvas.
            const [lum, blur] = ctx.kernels;
            const img = ctx.utils.makeTestImage(64);
            const out = toArr(blur(lum(img)));
            const map = refLuminanceMap(img);
            const ref = refBlur3(map);
            for (let y = 0; y < 64; y++) {
              for (let x = 0; x < 64; x++) {
                const hint = transposeCellHint(out[y][x], ref[x][y], 4e-3, y, x) ||
                  blurHint(out[y][x], ref, map, 4e-3, y, x);
                ctx.assertClose(out[y][x], ref[y][x], 4e-3, hint || `cell [${y}][${x}]`);
              }
            }
          },
        },
      ],
    },
  ],
};
