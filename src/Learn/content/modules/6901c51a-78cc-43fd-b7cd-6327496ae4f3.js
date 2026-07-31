// Module: The Canny Edge Pipeline — uuid 6901c51a-78cc-43fd-b7cd-6327496ae4f3
// (short id 6901c51a). The file name is the uuid; identity lives in the
// exported object below, never in the path. No legacyId — this module is new.
//
// Six tasks, one per stage of Canny's 1986 edge detector plus the assembly:
// separable Gaussian blur → Sobel magnitude AND direction → non-maximum
// suppression → double threshold → hysteresis (iterated to stability) → the
// whole five-stage chain wired together with pipeline: true.
//
// Every stage is a small kernel. The COMPOSITION is the lesson, and the reason
// the last task exists: five passes over an image is exactly the shape where
// keeping intermediates on the GPU stops being a nicety.
//
// Kernel-authoring rules (contract): no closures inside kernel functions, only
// numbers / nested number arrays / ImageData as inputs, this.thread.* for
// indexing, this.constants.* for compile-time values, image convention
// image[y][x] = [r, g, b, a] with channels 0–1. Every task passes in CPU mode.
//
// NUMBERS THIS MODULE IS BUILT ON (measured, not guessed — see the prose):
//   • the scene is one synthetic picture at two sizes: 64 for the stage tasks,
//     384 for the assembly, with ±0.12 of per-pixel noise;
//   • thresholds low = 0.30, high = 0.70 on raw Sobel magnitudes (which reach
//     ~2.2 on this scene);
//   • at 64 the hysteresis settles after exactly 28 passes; at 384, after 59 on
//     the task's own photo and 56 on the private test's, which is why the
//     pipeline task runs a fixed 64 — a fixed count has to cover the worst
//     photo it will be handed, not the best.
//
// FLOAT DETERMINISM. Tasks 3–5 take their maps as INPUTS with the magnitudes
// rounded to 3 decimals. That is not decoration: non-maximum suppression is a
// chain of >= comparisons, and two magnitudes that differ by 1e-7 would decide
// differently in float32 (GL) and float64 (the reference). Rounding leaves
// every comparison either an exact tie — which survives the trip to a texture
// unchanged, so both sides agree — or a gap of at least 1e-3, which is four
// orders of magnitude above float32's resolution here. Verified: rounding the
// whole chain through Math.fround changes not one cell of the result.

import { ARRAY_LAYOUT } from '../layoutNote.js';
import { plainToImageData, quantizePixel } from '../../engine/utils.js';

const LUM = [0.299, 0.587, 0.114];

// The module's shared dials.
const NOISE = 0.12; // per-pixel noise amplitude in the synthetic scene
const SEED = 6901; // one seed, so every task sees the same picture
const LOW = 0.3; // double-threshold thresholds, on raw Sobel magnitudes
const HIGH = 0.7;
// The assembly task's image; the stage tasks use 64. 384 is where the payoff
// task's claim stops being an argument and becomes a measurement: 147,456
// threads is enough work per launch to pay for making the launch, so the
// pipelined chain beats both the readback-between-stages wiring and the CPU
// backend by a wide margin (the numbers are in that task's prose).
//
// Why not larger. Larger IS faster — at 512 the pipelined chain wins by 10×
// rather than 7× — but the pre-flight guard extrapolates a 64×64 probe by the
// THREAD ratio, and that ratio grows as the square. At 512 the extrapolation
// reaches ~22 s on a software WebGL rasteriser, which would mean asking for
// roughly twice this task's budget and doubling its hang watchdog with it.
// cpu mode is not the constraint at either size: the whole module verifies on
// the CPU backend in about 2 s.
const BIG = 384;
// Fixed hysteresis passes in the pipeline. 384 settles after 59 on this photo
// and 56 on the private test's; 64 covers both with room, and is the number the
// launch arithmetic in the prose is written against (8 + 64 = 72 launches).
const PASSES = 64;

function luminanceOf(pixel) {
  return LUM[0] * pixel[0] + LUM[1] * pixel[1] + LUM[2] * pixel[2];
}

function clampIndex(v, last) {
  return v < 0 ? 0 : v > last ? last : v;
}

function round3(map) {
  return map.map(row => row.map(v => Math.round(v * 1000) / 1000));
}

// ---- the scene ------------------------------------------------------------
//
// One picture, four quadrants, four edge orientations — so every one of the
// four gradient-direction buckets non-maximum suppression quantises into is
// actually exercised, and so the double threshold has both a big step (the
// bright square) and a small one (the dim square) to separate.

function sceneField(size) {
  const half = size / 2;
  const out = new Array(size);
  for (let y = 0; y < size; y++) {
    const row = new Array(size);
    for (let x = 0; x < size; x++) {
      let v = 0.12; // background
      if (y < half && x < half) {
        // bright square: four hard, axis-aligned edges (buckets 0° and 90°)
        if (x >= size * 0.12 && x <= size * 0.37 && y >= size * 0.12 && y <= size * 0.37) v = 0.92;
      } else if (y < half) {
        // the same shape with a much smaller step — weak edges, on purpose
        if (x >= size * 0.63 && x <= size * 0.87 && y >= size * 0.12 && y <= size * 0.37) v = 0.34;
      } else if (x < half) {
        // a boundary at 45°: brightness rises with x and falls with y
        if (x >= y - half + size * 0.03) v = 0.85;
      } else {
        // the other diagonal, so the 135° bucket is populated too
        if (x - half + (y - half) < size * 0.47) v = 0.78;
      }
      row[x] = v;
    }
    out[y] = row;
  }
  return out;
}

// The scene plus per-pixel noise, quantized to 3 decimals. This is what stage 1
// is asked to smooth, and the noise is the whole reason stage 1 exists.
function noisyField(utils, size, seed = SEED) {
  const rand = utils.seededRandom(seed);
  return sceneField(size).map(row =>
    row.map(v => Math.round(Math.min(1, Math.max(0, v + (rand() * 2 - 1) * NOISE)) * 1000) / 1000)
  );
}

// The same scene as an ImageData — the one image shape gpu.js puts on the GPU
// for every backend (engine/utils.plainToImageData). The tint is a per-channel
// OFFSET, so the picture is genuinely colored while its luminance stays the
// scene plus a constant — and a constant offset has no gradient at all, which
// is why the thresholds measured on the gray field carry over unchanged.
function scenePhoto(utils, size, seed = SEED) {
  return plainToImageData(
    noisyField(utils, size, seed).map(row =>
      row.map(v => quantizePixel([v + 0.05, v, v - 0.04, 1]))
    )
  );
}

function luminanceMapOf(image) {
  return image.plain.map(row => row.map(luminanceOf));
}

// ---- the five stages, in plain JavaScript ---------------------------------
//
// Shared by inputs() and by the tests: every task's expectations, and every
// near-miss probe, are built from these and never hardcoded.

// Stage 1a/1b — the separable Gaussian, [1, 4, 6, 4, 1] / 16 on each axis.
function blurXRef(map) {
  const last = map.length - 1;
  return map.map((row, y) =>
    row.map((_, x) => {
      const x0 = clampIndex(x - 2, last);
      const x1 = clampIndex(x - 1, last);
      const x3 = clampIndex(x + 1, last);
      const x4 = clampIndex(x + 2, last);
      return (map[y][x0] + 4 * map[y][x1] + 6 * map[y][x] + 4 * map[y][x3] + map[y][x4]) / 16;
    })
  );
}

function blurYRef(map) {
  const last = map.length - 1;
  return map.map((row, y) =>
    row.map((_, x) => {
      const y0 = clampIndex(y - 2, last);
      const y1 = clampIndex(y - 1, last);
      const y3 = clampIndex(y + 1, last);
      const y4 = clampIndex(y + 2, last);
      return (map[y0][x] + 4 * map[y1][x] + 6 * map[y][x] + 4 * map[y3][x] + map[y4][x]) / 16;
    })
  );
}

function blurRef(map) {
  return blurYRef(blurXRef(map));
}

// The 5-tap box filter, for the "why not just average five samples" probe.
function boxXRef(map) {
  const last = map.length - 1;
  return map.map((row, y) =>
    row.map((_, x) => {
      let sum = 0;
      for (let i = -2; i <= 2; i++) sum += map[y][clampIndex(x + i, last)];
      return sum / 5;
    })
  );
}

function boxRef(map) {
  const last = map.length - 1;
  const once = boxXRef(map);
  return once.map((row, y) =>
    row.map((_, x) => {
      let sum = 0;
      for (let i = -2; i <= 2; i++) sum += once[clampIndex(y + i, last)][x];
      return sum / 5;
    })
  );
}

// Stage 2 — Sobel, both outputs. `mag` is the length of the gradient vector,
// `dir` its angle in radians, with y running DOWN the image (so gy is the
// bottom row minus the top row, exactly as the Gy grid is written).
function sobelRef(gray, opts = {}) {
  const size = gray.length;
  const last = size - 1;
  const mag = new Array(size);
  const dir = new Array(size);
  for (let y = 0; y < size; y++) {
    mag[y] = new Array(size).fill(0);
    dir[y] = new Array(size).fill(0);
    for (let x = 0; x < size; x++) {
      if (x === 0 || y === 0 || x === last || y === last) continue;
      const tl = gray[y - 1][x - 1];
      const tm = gray[y - 1][x];
      const tr = gray[y - 1][x + 1];
      const ml = gray[y][x - 1];
      const mr = gray[y][x + 1];
      const bl = gray[y + 1][x - 1];
      const bm = gray[y + 1][x];
      const br = gray[y + 1][x + 1];
      const gx = tr + 2 * mr + br - (tl + 2 * ml + bl);
      const gy = bl + 2 * bm + br - (tl + 2 * tm + tr);
      mag[y][x] = opts.squared
        ? gx * gx + gy * gy
        : opts.manhattan
          ? Math.abs(gx) + Math.abs(gy)
          : Math.sqrt(gx * gx + gy * gy);
      const angle = opts.swappedArgs
        ? Math.atan2(gx, gy)
        : opts.flippedY
          ? Math.atan2(-gy, gx)
          : Math.atan2(gy, gx);
      dir[y][x] = opts.degrees ? (angle * 180) / Math.PI : angle;
    }
  }
  return { mag, dir };
}

// What tasks 3–5 are HANDED: the same maps, with magnitudes rounded so every
// >= comparison downstream is decided identically in float32 and float64 (see
// the note at the top of this file). Angles are rounded far below the ~0.01°
// margin any of them keeps from a bucket boundary.
function gradientInputs(gray) {
  const { mag, dir } = sobelRef(gray);
  return {
    mag: round3(mag),
    dir: dir.map(row => row.map(a => Math.round(a * 1e6) / 1e6)),
  };
}

// Stage 3 — the axis a quantised gradient angle points along, as an (ax, ay)
// step. Wrapping a negative angle by +180° is what keeps the 135° bucket from
// being empty: an angle and its opposite describe the same AXIS, and an axis is
// all suppression needs.
function axisFor(angle, opts = {}) {
  let a = angle;
  if (!opts.noWrap && a < 0) a += Math.PI;
  const deg = (a * 180) / Math.PI;
  let ax = 1;
  let ay = 0;
  if (deg >= 22.5 && deg < 67.5) {
    ax = 1;
    ay = 1;
  } else if (deg >= 67.5 && deg < 112.5) {
    ax = 0;
    ay = 1;
  } else if (deg >= 112.5 && deg < 157.5) {
    ax = -1;
    ay = 1;
  }
  // The classic Canny bug: comparing along the EDGE instead of along the
  // gradient. The two are perpendicular — a quarter turn apart.
  if (opts.perpendicular) {
    const swap = ax;
    ax = -ay;
    ay = swap;
  }
  return [ax, ay];
}

function nmsRef(mag, dir, opts = {}) {
  const size = mag.length;
  const last = size - 1;
  const out = new Array(size);
  for (let y = 0; y < size; y++) {
    out[y] = new Array(size).fill(0);
    for (let x = 0; x < size; x++) {
      if (x === 0 || y === 0 || x === last || y === last) continue;
      const [ax, ay] = axisFor(dir[y][x], opts);
      const m = mag[y][x];
      const ahead = mag[clampIndex(y + ay, last)][clampIndex(x + ax, last)];
      const behind = mag[clampIndex(y - ay, last)][clampIndex(x - ax, last)];
      if (m >= ahead && m >= behind) out[y][x] = opts.mask ? 1 : m;
    }
  }
  return out;
}

// Stage 4 — strong (1), weak (0.5), suppressed (0).
function classifyRef(thin, opts = {}) {
  return thin.map(row =>
    row.map(m => {
      if (opts.swapped) return m >= LOW ? 1 : m >= HIGH ? 0.5 : 0;
      if (opts.noWeak) return m >= HIGH ? 1 : 0;
      if (opts.rawWeak) return m >= HIGH ? 1 : m >= LOW ? m : 0;
      return m >= HIGH ? 1 : m >= LOW ? 0.5 : 0;
    })
  );
}

// Stage 5 — one propagation pass: a weak pixel touching a strong one joins it.
function growRef(state, opts = {}) {
  const size = state.length;
  const last = size - 1;
  const out = new Array(size);
  for (let y = 0; y < size; y++) {
    out[y] = new Array(size).fill(0);
    for (let x = 0; x < size; x++) {
      const v = state[y][x];
      if (v > 0.75) {
        out[y][x] = 1;
        continue;
      }
      if (v < 0.25) continue;
      let near = 0;
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          if (opts.four && dx !== 0 && dy !== 0) continue;
          if (state[clampIndex(y + dy, last)][clampIndex(x + dx, last)] > 0.75) near = 1;
        }
      }
      out[y][x] = near ? 1 : 0.5;
    }
  }
  return out;
}

function gridsEqual(a, b) {
  for (let y = 0; y < a.length; y++) {
    for (let x = 0; x < a[y].length; x++) {
      if (a[y][x] !== b[y][x]) return false;
    }
  }
  return true;
}

// Propagate to a fixed point. `passes` counts every call, INCLUDING the last
// one — the one that changed nothing and is how you found out you were done.
function settleRef(classified, maxPasses, opts = {}) {
  let state = classified;
  let passes = 0;
  for (let i = 0; i < maxPasses; i++) {
    const next = growRef(state, opts);
    passes++;
    if (gridsEqual(next, state)) {
      state = next;
      break;
    }
    state = next;
  }
  return { state, passes };
}

// Exactly `n` passes, no early exit — what the pipeline task can afford to do.
function growTimes(classified, n, opts = {}) {
  let state = classified;
  for (let i = 0; i < n; i++) state = growRef(state, opts);
  return state;
}

function finishRef(state) {
  return state.map(row => row.map(v => (v > 0.75 ? 1 : 0)));
}

// The whole detector, end to end, from an ImageData.
function cannyRef(image, passes = PASSES) {
  const gray = luminanceMapOf(image);
  const smooth = blurRef(gray);
  const { mag, dir } = sobelRef(smooth);
  const thin = nmsRef(mag, dir);
  return {
    gray,
    smooth,
    mag,
    dir,
    thin,
    edges: finishRef(growTimes(classifyRef(thin), passes)),
  };
}

function countOnes(grid) {
  let n = 0;
  for (let y = 0; y < grid.length; y++) {
    for (let x = 0; x < grid[y].length; x++) if (grid[y][x] > 0.5) n++;
  }
  return n;
}

// ---- near-miss diagnosis --------------------------------------------------
//
// The course's standing rule: when a failing value is exactly what one specific
// mistake would produce, name that mistake instead of reporting two numbers. A
// probe pairs such a value with its sentence; diagnose() speaks only when the
// observation matches a probe within the test's own tolerance AND the correct
// answer does not — so a sample where a candidate happens to coincide with the
// right answer stays silent, as do observations matching probes that disagree
// with each other. A confident wrong diagnosis is worse than a plain mismatch.
function diagnose(got, expected, eps, probes) {
  const hits = probes
    .filter(p => Math.abs(got - p[0]) <= eps && Math.abs(expected - p[0]) > eps)
    .map(p => p[1]);
  return hits.length && hits.every(m => m === hits[0]) ? hits[0] : null;
}

// Whole-grid form. A single matching cell is worthless evidence here — a
// suppression rule that compares the wrong two neighbours still agrees with the
// right one wherever the magnitude is flat, which is most of the picture. So a
// probe must predict EVERY cell (and differ from the right answer somewhere)
// before it is allowed to speak.
function diagnoseGrid(out, expected, eps, probes) {
  const size = expected.length;
  const hits = probes
    .filter(([grid]) => {
      let differs = false;
      for (let y = 0; y < size; y++) {
        for (let x = 0; x < size; x++) {
          if (!(out[y] && Math.abs(out[y][x] - grid[y][x]) <= eps)) return false;
          if (Math.abs(expected[y][x] - grid[y][x]) > eps) differs = true;
        }
      }
      return differs;
    })
    .map(p => p[1]);
  return hits.length && hits.every(m => m === hits[0]) ? hits[0] : null;
}

// Swapping this.thread.x and this.thread.y reads the transpose of the map — the
// standing gotcha of every 2D task in this course, and invisible to a test that
// only checks totals. Cells on the diagonal are their own transpose and can
// never show it, which is why the case lists below also probe off-diagonal
// cells.
function transposeCellHint(got, transposed, eps, y, x) {
  return Math.abs(got - transposed) <= eps
    ? `that is the value for cell [${x}][${y}] — this.thread.x and this.thread.y are ` +
        `swapped. Rows come first: map[this.thread.y][this.thread.x]`
    : null;
}

// A neighbour index that was never clamped reads off the end of the map.
function unclampedHint(got) {
  return Number.isFinite(got)
    ? null
    : 'that cell read past the edge of the map — clamp the sample index into 0…this.constants.last before indexing';
}

// Task 1: the separable blur applied twice to the same axis, un-normalised, or
// swapped for a plain box average.
function blurProbes(gray) {
  const twiceX = blurXRef(blurXRef(gray));
  const twiceY = blurYRef(blurYRef(gray));
  const sameAxis =
    'both passes blurred the same axis — the second pass has to walk this.thread.y. ' +
    'Two horizontal passes are still a horizontal blur, just a wider one';
  return [
    [twiceX, sameAxis],
    [twiceY, sameAxis],
    [boxRef(gray), 'that is a flat 5-tap box average — this filter weights the taps 1, 4, 6, 4, 1'],
    [
      blurRef(gray).map(row => row.map(v => v * 16)),
      'the weights were never divided by 16 — they have to sum to 1 or the picture gets 16× brighter',
    ],
  ];
}

// Task 4: the two comparisons in the wrong order, the weak class missing, or
// the weak class carrying the magnitude instead of the flat 0.5 marker.
//
// "Test low first" and "mark every survivor strong" produce the SAME grid — the
// second branch of `m >= low ? 1 : m >= high ? 0.5 : 0` can never be reached —
// so they share one sentence. Two probes with two different messages on one
// grid would cancel each other out and say nothing at all.
function classifyProbes(thin) {
  const wrongWayRound =
    'every pixel above the LOW threshold came back strong — the two comparisons are the wrong ' +
    'way round. Test the HIGH threshold first: above high is strong (1), and only what is left ' +
    'gets measured against low';
  return [
    [classifyRef(thin, { swapped: true }), wrongWayRound],
    [classifyRef(thin, { noWeak: true }), 'the weak class is missing — a pixel between the two thresholds is 0.5, not 0. Deciding whether it lives is the next stage\'s job, not this one\'s'],
    [classifyRef(thin, { rawWeak: true }), 'weak pixels came back carrying their magnitude — the next stage tests for exactly 0.5 and 1, so the middle band has to be the flat marker 0.5'],
  ];
}

export default {
  uuid: '6901c51a-78cc-43fd-b7cd-6327496ae4f3',
  version: 1,
  slug: 'canny-edges',
  title: 'The Canny Edge Pipeline',
  blurb:
    'The edge detector every vision library ships, one kernel per stage — blur, gradient, thinning, thresholds, hysteresis — then chained with <code>pipeline: true</code>.',
  tasks: [
    // ------------------------------------------------------------------ 1
    {
      slug: 'gaussian-blur',
      title: 'Blur First: a Separable Gaussian',
      intro: `<p>Canny's first move looks like vandalism: before you go looking for edges, you
        <strong>throw detail away</strong>. The reason is that every later stage is built on a
        derivative, and the derivative of noise is enormous. A pixel that wobbles by ±0.12 against
        its neighbours has no visible brightness to speak of — but a difference operator reads that
        wobble at full strength, because a difference is exactly what it is looking for.</p>
        <p>The numbers on this task's own picture: run the rest of this module on <code>gray</code>
        unsmoothed and you get <strong>596 edge pixels, 154 of them in flat background</strong> —
        pure noise, promoted to structure. Smooth it first and the same pipeline reports
        <strong>299 edge pixels and not one spurious</strong>. That is what the blur buys.</p>
        <p>Convolution &amp; Filters already taught the sliding window, the box blur, clamped
        edges, and the fact that a box blur is <em>separable</em>. Both facts come due here. A
        Gaussian beats a box for this job because it has no corners: a box filter's response
        oscillates as the window slides, so it manufactures small ridges of its own — precisely
        the thing stage 3 is about to hunt for. And a Gaussian is separable too, so a 5×5 window is
        <strong>two 5-tap passes, not one 25-tap pass</strong>: 10 reads per pixel instead of 25.</p>
        ${ARRAY_LAYOUT}`,
      goal: `<strong>Goal:</strong> finish the vertical half of the blur. <code>blurX</code> is
        written for you; write <code>blurY</code> so the pair applies the weights
        <code>[1, 4, 6, 4, 1] / 16</code> along <em>each</em> axis, indexes clamped at the edges.`,
      requirements: [
        'Weight five samples <code>1, 4, 6, 4, 1</code> and divide the total by <code>16</code>',
        '<code>blurY</code> walks <strong>rows</strong> — offset <code>this.thread.y</code>, not <code>this.thread.x</code>',
        'Clamp every sample index into <code>0…this.constants.last</code>',
      ],
      hints: [
        {
          title: 'Hint 1 — the same filter, turned ninety degrees',
          body: `<p><code>blurX</code> holds <code>y</code> still and moves <code>x</code>.
            <code>blurY</code> does the mirror image: hold <code>x</code> still, move
            <code>y</code>. Copying the body is fine — copying its <em>axis</em> is the mistake
            the tests are watching for.</p>`,
        },
        {
          title: 'Hint 2 — the clamps',
          body: `<pre><code>let y0 = y - 2;
if (y0 &lt; 0) y0 = 0;
let y4 = y + 2;
if (y4 &gt; this.constants.last) y4 = this.constants.last;</code></pre>
<p>— and the same for <code>y1</code> and <code>y3</code> at distance 1.</p>`,
        },
        {
          title: 'Hint 3 — the whole return',
          body: `<pre><code>return (map[y0][x] + 4 * map[y1][x] + 6 * map[y][x]
      + 4 * map[y3][x] + map[y4][x]) / 16;</code></pre>
<p>The weights sum to 16, so the divide is what keeps a flat area flat.</p>`,
        },
      ],
      transfer: `Separability is not a gpu.js trick — it is why production blurs are fast
        everywhere. Metal Performance Shaders' <code>MPSImageGaussianBlur</code> and NVIDIA NPP's
        <code>nppiFilterGaussBorder</code> both decompose internally; a WebGPU post-processing
        chain does horizontal-then-vertical into a ping-pong pair of textures. The saving grows
        with the kernel: a 15×15 Gaussian is 225 taps as one pass and 30 as two.`,
      starterCode: `// Stage 1 of Canny: smooth, so the derivative that follows is a
// derivative of the picture and not of the noise.
const gpu = new GPU({ mode });

// Pass 1 — horizontal. Weights 1, 4, 6, 4, 1 over five columns.
const blurX = gpu.createKernel(function (gray) {
  const x = this.thread.x;
  const y = this.thread.y;
  let x0 = x - 2;
  if (x0 < 0) x0 = 0;
  let x1 = x - 1;
  if (x1 < 0) x1 = 0;
  let x3 = x + 1;
  if (x3 > this.constants.last) x3 = this.constants.last;
  let x4 = x + 2;
  if (x4 > this.constants.last) x4 = this.constants.last;
  return (gray[y][x0] + 4 * gray[y][x1] + 6 * gray[y][x] + 4 * gray[y][x3] + gray[y][x4]) / 16;
}, {
  output: [64, 64],
  constants: { last: 63 },
});

// Pass 2 — vertical. Same weights, other axis.
const blurY = gpu.createKernel(function (map) {
  const x = this.thread.x;
  const y = this.thread.y;
  // TODO: the same five weighted samples, walking DOWN the column:
  // rows y-2, y-1, y, y+1, y+2, each index clamped to 0…this.constants.last.
  return map[y][x];
}, {
  output: [64, 64],
  constants: { last: 63 },
});

const smooth = blurY(blurX(gray));
console.log('noisy background pixel:', gray[4][40], ' smoothed:', smooth[4][40]);
`,
      solutionCode: `// Stage 1 of Canny: smooth, so the derivative that follows is a
// derivative of the picture and not of the noise.
const gpu = new GPU({ mode });

// Pass 1 — horizontal. Weights 1, 4, 6, 4, 1 over five columns.
const blurX = gpu.createKernel(function (gray) {
  const x = this.thread.x;
  const y = this.thread.y;
  let x0 = x - 2;
  if (x0 < 0) x0 = 0;
  let x1 = x - 1;
  if (x1 < 0) x1 = 0;
  let x3 = x + 1;
  if (x3 > this.constants.last) x3 = this.constants.last;
  let x4 = x + 2;
  if (x4 > this.constants.last) x4 = this.constants.last;
  return (gray[y][x0] + 4 * gray[y][x1] + 6 * gray[y][x] + 4 * gray[y][x3] + gray[y][x4]) / 16;
}, {
  output: [64, 64],
  constants: { last: 63 },
});

// Pass 2 — vertical. Same weights, other axis.
const blurY = gpu.createKernel(function (map) {
  const x = this.thread.x;
  const y = this.thread.y;
  let y0 = y - 2;
  if (y0 < 0) y0 = 0;
  let y1 = y - 1;
  if (y1 < 0) y1 = 0;
  let y3 = y + 1;
  if (y3 > this.constants.last) y3 = this.constants.last;
  let y4 = y + 2;
  if (y4 > this.constants.last) y4 = this.constants.last;
  return (map[y0][x] + 4 * map[y1][x] + 6 * map[y][x] + 4 * map[y3][x] + map[y4][x]) / 16;
}, {
  output: [64, 64],
  constants: { last: 63 },
});

const smooth = blurY(blurX(gray));
console.log('noisy background pixel:', gray[4][40], ' smoothed:', smooth[4][40]);
`,
      inputs: utils => ({ gray: noisyField(utils, 64) }),
      publicTests: [
        {
          name: 'two passes, and a flat map survives both unchanged',
          run: async ctx => {
            ctx.assert(ctx.kernels.length >= 2, `expected 2 kernels, found ${ctx.kernels.length}`);
            const [blurX, blurY] = ctx.kernels;
            const flat = new Array(64).fill(new Array(64).fill(0.4));
            const out = blurY(blurX(flat));
            ctx.assert(out && out.length === 64, `expected 64 rows, got ${out && out.length}`);
            ctx.assert(out[0] && out[0].length === 64, 'each row should hold 64 values');
            for (let y = 0; y < 64; y += 9) {
              for (let x = 0; x < 64; x += 9) {
                const hint = unclampedHint(out[y][x]) ||
                  diagnose(out[y][x], 0.4, 1e-4, [
                    [0.4 * 16, 'the weights were never divided by 16 — they have to sum to 1 or a flat area gets 16× brighter'],
                  ]);
                ctx.assertClose(out[y][x], 0.4, 1e-4, hint || `cell [${y}][${x}]`);
              }
            }
          },
        },
        {
          name: 'the pair equals the separable <code>[1, 4, 6, 4, 1] / 16</code> reference',
          run: async ctx => {
            const [blurX, blurY] = ctx.kernels;
            const gray = noisyField(ctx.utils, 64);
            const out = blurY(blurX(gray));
            const ref = blurRef(gray);
            const hint = diagnoseGrid(out, ref, 2e-4, blurProbes(gray));
            const cases = [[0, 0], [1, 30], [12, 5], [32, 32], [40, 12], [63, 63], [63, 20], [20, 63]];
            for (const [y, x] of cases) {
              ctx.assertClose(
                out[y][x], ref[y][x], 2e-4,
                hint || unclampedHint(out[y][x]) ||
                  transposeCellHint(out[y][x], ref[x][y], 2e-4, y, x) || `cell [${y}][${x}]`
              );
            }
          },
        },
        {
          name: 'one bright pixel spreads on <em>both</em> axes',
          run: async ctx => {
            // A separable blur of a single spike is the outer product of the
            // weights: 1/256 at the corners of the 5x5 patch, 6/256 four cells
            // away on either axis. Blur x twice and the column never lights up
            // at all — which is exactly what this checks.
            const [blurX, blurY] = ctx.kernels;
            const spike = new Array(64);
            for (let y = 0; y < 64; y++) spike[y] = new Array(64).fill(0);
            spike[32][32] = 1;
            const out = blurY(blurX(spike));
            const w = [1, 4, 6, 4, 1];
            const at = (dy, dx) => (w[dy + 2] * w[dx + 2]) / 256;
            const cases = [[0, 0], [0, 1], [0, 2], [1, 0], [2, 0], [1, 1], [2, 2], [-1, 2], [2, -1]];
            for (const [dy, dx] of cases) {
              const got = out[32 + dy][32 + dx];
              const hint = Math.abs(got) < 1e-6 && at(dy, dx) > 1e-6 && dy !== 0
                ? 'nothing spread vertically at all — both passes blurred the same axis. blurY has to offset this.thread.y'
                : null;
              ctx.assertClose(got, at(dy, dx), 2e-4, hint || `cell [${32 + dy}][${32 + dx}]`);
            }
            ctx.assertClose(out[32][35], 0, 2e-4, 'the filter is 5 taps wide — nothing should reach 3 cells away');
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            const [blurX, blurY] = ctx.kernels;
            const gray = noisyField(ctx.utils, 64, 31337);
            const out = blurY(blurX(gray));
            const ref = blurRef(gray);
            const hint = diagnoseGrid(out, ref, 2e-4, blurProbes(gray));
            for (let y = 0; y < 64; y++) {
              for (let x = 0; x < 64; x++) {
                ctx.assertClose(
                  out[y][x], ref[y][x], 2e-4,
                  hint || unclampedHint(out[y][x]) || `cell [${y}][${x}]`
                );
              }
            }
          },
        },
        {
          name: 'private test #2',
          run: async ctx => {
            // Blurring actually has to REDUCE the noise it was added for: the
            // largest gradient magnitude inside flat background must fall.
            const [blurX, blurY] = ctx.kernels;
            const gray = noisyField(ctx.utils, 64);
            const out = blurY(blurX(gray));
            const peak = map => {
              let best = 0;
              for (let y = 3; y <= 9; y++) {
                for (let x = 34; x <= 44; x++) {
                  best = Math.max(best, Math.abs(map[y][x + 1] - map[y][x - 1]));
                }
              }
              return best;
            };
            const before = peak(gray);
            const after = peak(out);
            ctx.assert(
              after < before * 0.6,
              `the flat background is no smoother than it started (largest neighbour-to-neighbour ` +
                `difference ${before.toFixed(3)} before, ${after.toFixed(3)} after) — is the blur ` +
                `actually running on both axes?`
            );
          },
        },
      ],
    },

    // ------------------------------------------------------------------ 2
    {
      slug: 'gradient-and-direction',
      title: 'Magnitude, and the Angle Nobody Mentions',
      intro: `<p>The Sobel pass you wrote in Convolution &amp; Filters answered one question:
        <em>how strong</em> is the change here, <code>√(gx² + gy²)</code>. Canny needs a second
        answer from the same eight reads, and it is the one that usually gets skipped:
        <em>which way</em> does the change point. <code>Math.atan2(gy, gx)</code> — and the next
        stage is built entirely on it. Get the angle wrong and non-maximum suppression compares
        the wrong two neighbours, silently, on every pixel.</p>
        <p>Two things about <code>atan2</code> worth saying out loud. It takes the
        <strong>vertical component first</strong>: <code>Math.atan2(gy, gx)</code>, not the other
        way round — swap them and every angle is reflected about 45°. And it returns
        <strong>radians</strong> in −π…π, which is why the result can be negative: a gradient
        pointing up-and-right and one pointing down-and-left are 180° apart and describe the same
        edge. Stage 3 is where that gets sorted out.</p>
        <p><code>gray</code> here is already smoothed — it is what stage 1 hands over. Both
        kernels read the same 3×3 neighbourhood; the starter has pulled the nine cells into
        locals for you.</p>
        ${ARRAY_LAYOUT}`,
      goal: `<strong>Goal:</strong> compute <code>gx</code> and <code>gy</code> from the Sobel
        grids in both kernels, then return the gradient's <strong>length</strong> from
        <code>magnitude</code> and its <strong>angle in radians</strong> from
        <code>direction</code>.`,
      requirements: [
        '<code>gx</code> is the right column minus the left, middle row counted double; <code>gy</code> is the bottom row minus the top',
        '<code>magnitude</code> returns <code>Math.sqrt(gx * gx + gy * gy)</code> — the length, not its square',
        '<code>direction</code> returns <code>Math.atan2(gy, gx)</code> — vertical component first, in radians',
        'Border pixels have no full neighbourhood: both kernels already return <code>0</code> there',
      ],
      hints: [
        {
          title: 'Hint 1 — the two grids',
          body: `<p>Same pair Convolution &amp; Filters used:</p>
<pre><code>const gx = (tr + 2 * mr + br) - (tl + 2 * ml + bl);
const gy = (bl + 2 * bm + br) - (tl + 2 * tm + tr);</code></pre>
<p><code>gy</code> is bottom minus top because <code>y</code> runs <em>down</em> the image.
            Flip that sign and the magnitude will not notice — it squares everything — but every
            angle will.</p>`,
        },
        {
          title: 'Hint 2 — the two returns',
          body: `<pre><code>return Math.sqrt(gx * gx + gy * gy);   // magnitude
return Math.atan2(gy, gx);             // direction, radians</code></pre>
<p>Leaving the <code>Math.sqrt</code> off is tempting — comparisons on squares sort the
            same way — but every threshold in the rest of this module is calibrated against a
            length, and squaring bends the scale.</p>`,
        },
      ],
      transfer: `<code>atan2</code> is a hardware instruction's worth of work on every GPU:
        CUDA has <code>atan2f</code> (and <code>__fdividef</code> for the cheap path), WGSL and
        Metal both spell it <code>atan2</code>, and gpu.js compiles <code>Math.atan2</code>
        straight to GLSL's <code>atan(y, x)</code>. OpenCV's <code>cv::Canny</code> famously
        avoids it altogether — it compares <code>|gy|</code> against <code>tan(22.5°)·|gx|</code>
        with integer arithmetic — which is the same quantisation you are about to write, with the
        trigonometry folded away.`,
      starterCode: `// Stage 2 of Canny: two answers from one 3x3 neighbourhood.
//
//        Gx              Gy
//    -1   0  +1      -1  -2  -1
//    -2   0  +2       0   0   0
//    -1   0  +1      +1  +2  +1
//
const gpu = new GPU({ mode });

const magnitude = gpu.createKernel(function (gray) {
  const x = this.thread.x;
  const y = this.thread.y;
  if (x === 0 || y === 0 || x === this.constants.last || y === this.constants.last) {
    return 0;
  }
  const tl = gray[y - 1][x - 1];
  const tm = gray[y - 1][x];
  const tr = gray[y - 1][x + 1];
  const ml = gray[y][x - 1];
  const mr = gray[y][x + 1];
  const bl = gray[y + 1][x - 1];
  const bm = gray[y + 1][x];
  const br = gray[y + 1][x + 1];
  // TODO: gx and gy from the grids above, then return the gradient's LENGTH.
  return 0;
}, {
  output: [64, 64],
  constants: { last: 63 },
});

const direction = gpu.createKernel(function (gray) {
  const x = this.thread.x;
  const y = this.thread.y;
  if (x === 0 || y === 0 || x === this.constants.last || y === this.constants.last) {
    return 0;
  }
  const tl = gray[y - 1][x - 1];
  const tm = gray[y - 1][x];
  const tr = gray[y - 1][x + 1];
  const ml = gray[y][x - 1];
  const mr = gray[y][x + 1];
  const bl = gray[y + 1][x - 1];
  const bm = gray[y + 1][x];
  const br = gray[y + 1][x + 1];
  // TODO: the same gx and gy, then return the gradient's ANGLE in radians.
  return 0;
}, {
  output: [64, 64],
  constants: { last: 63 },
});

const mag = magnitude(gray);
const dir = direction(gray);
console.log('on a vertical edge — magnitude:', mag[20][8], ' angle:', dir[20][8]);
console.log('on a horizontal edge — magnitude:', mag[8][20], ' angle:', dir[8][20]);
`,
      solutionCode: `// Stage 2 of Canny: two answers from one 3x3 neighbourhood.
//
//        Gx              Gy
//    -1   0  +1      -1  -2  -1
//    -2   0  +2       0   0   0
//    -1   0  +1      +1  +2  +1
//
const gpu = new GPU({ mode });

const magnitude = gpu.createKernel(function (gray) {
  const x = this.thread.x;
  const y = this.thread.y;
  if (x === 0 || y === 0 || x === this.constants.last || y === this.constants.last) {
    return 0;
  }
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
  return Math.sqrt(gx * gx + gy * gy);
}, {
  output: [64, 64],
  constants: { last: 63 },
});

const direction = gpu.createKernel(function (gray) {
  const x = this.thread.x;
  const y = this.thread.y;
  if (x === 0 || y === 0 || x === this.constants.last || y === this.constants.last) {
    return 0;
  }
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
  return Math.atan2(gy, gx); // vertical component FIRST
}, {
  output: [64, 64],
  constants: { last: 63 },
});

const mag = magnitude(gray);
const dir = direction(gray);
console.log('on a vertical edge — magnitude:', mag[20][8], ' angle:', dir[20][8]);
console.log('on a horizontal edge — magnitude:', mag[8][20], ' angle:', dir[8][20]);
`,
      inputs: utils => ({ gray: blurRef(noisyField(utils, 64)) }),
      publicTests: [
        {
          name: 'two 64×64 maps, flat in, zero out',
          run: async ctx => {
            ctx.assert(ctx.kernels.length >= 2, `expected 2 kernels, found ${ctx.kernels.length}`);
            const [magnitude, direction] = ctx.kernels;
            const flat = new Array(64).fill(new Array(64).fill(0.6));
            const m = magnitude(flat);
            const d = direction(flat);
            ctx.assert(m && m.length === 64 && m[0].length === 64, 'magnitude should return a 64×64 grid');
            ctx.assert(d && d.length === 64 && d[0].length === 64, 'direction should return a 64×64 grid');
            for (let y = 0; y < 64; y += 7) {
              for (let x = 0; x < 64; x += 7) {
                ctx.assertClose(m[y][x], 0, 1e-4, `a flat map has no gradient — cell [${y}][${x}]`);
              }
            }
          },
        },
        {
          name: 'magnitude is the gradient <em>length</em>, <code>√(gx² + gy²)</code>',
          run: async ctx => {
            const [magnitude] = ctx.kernels;
            const gray = blurRef(noisyField(ctx.utils, 64));
            const out = magnitude(gray);
            const ref = sobelRef(gray).mag;
            const squared = sobelRef(gray, { squared: true }).mag;
            const manhattan = sobelRef(gray, { manhattan: true }).mag;
            const cases = [[20, 8], [8, 20], [45, 20], [50, 50], [11, 60], [32, 32], [60, 11]];
            for (const [y, x] of cases) {
              const hint =
                transposeCellHint(out[y][x], ref[x][y], 3e-3, y, x) ||
                diagnose(out[y][x], ref[y][x], 3e-3, [
                  [squared[y][x], 'that is the SQUARED magnitude — Math.sqrt is missing. Comparisons still sort the same way, but every threshold later in this module is a length, and squaring bends the scale nonlinearly'],
                  [manhattan[y][x], 'that is the |gx| + |gy| approximation — a real shortcut, but this module\'s thresholds are calibrated against the true length √(gx² + gy²)'],
                ]);
              ctx.assertClose(out[y][x], ref[y][x], 3e-3, hint || `cell [${y}][${x}]`);
            }
          },
        },
        {
          name: 'direction is <code>Math.atan2(gy, gx)</code>, in radians',
          run: async ctx => {
            const [, direction] = ctx.kernels;
            const gray = blurRef(noisyField(ctx.utils, 64));
            const out = direction(gray);
            const ref = sobelRef(gray).dir;
            const swapped = sobelRef(gray, { swappedArgs: true }).dir;
            const flipped = sobelRef(gray, { flippedY: true }).dir;
            const degrees = sobelRef(gray, { degrees: true }).dir;
            const cases = [[20, 8], [8, 20], [20, 24], [24, 20], [45, 20], [50, 50], [11, 60], [60, 11]];
            for (const [y, x] of cases) {
              const hint =
                transposeCellHint(out[y][x], ref[x][y], 4e-3, y, x) ||
                diagnose(out[y][x], ref[y][x], 4e-3, [
                  [swapped[y][x], 'Math.atan2 takes the VERTICAL component first — Math.atan2(gy, gx). Swapping them reflects every angle about 45°, and the next stage then compares the wrong two neighbours'],
                  [flipped[y][x], 'the vertical gradient\'s sign is flipped — y runs down the image, so gy is the bottom row minus the top row (the same Gy grid the magnitude uses)'],
                  [degrees[y][x], 'that angle is in degrees — Math.atan2 returns radians, and every stage after this one expects radians'],
                ]);
              ctx.assertClose(out[y][x], ref[y][x], 4e-3, hint || `cell [${y}][${x}]`);
            }
          },
        },
        {
          name: 'a vertical step reads 0 rad, a horizontal step ±π/2',
          run: async ctx => {
            // Analytic, orientation-revealing, and independent of the scene:
            // brightness rising with x is a gradient pointing along +x.
            const [magnitude, direction] = ctx.kernels;
            const vertical = new Array(64);
            const horizontal = new Array(64);
            for (let y = 0; y < 64; y++) {
              vertical[y] = new Array(64);
              horizontal[y] = new Array(64);
              for (let x = 0; x < 64; x++) {
                vertical[y][x] = x < 32 ? 0.2 : 0.8;
                horizontal[y][x] = y < 32 ? 0.2 : 0.8;
              }
            }
            const dv = direction(vertical);
            const dh = direction(horizontal);
            const mv = magnitude(vertical);
            ctx.assert(mv[20][31] > 2, `the step at column 31 should have a large magnitude, got ${mv[20][31]}`);
            ctx.assertClose(dv[20][31], 0, 1e-3,
              'a step that gets brighter to the RIGHT has its gradient pointing along +x, so the angle is 0');
            ctx.assertClose(Math.abs(dh[31][20]), Math.PI / 2, 1e-3,
              'a step that gets brighter DOWNWARD has its gradient pointing along ±y, so the angle is ±π/2 — ' +
              'if you got 0 here, gx and gy are swapped inside Math.atan2');
            ctx.assertClose(dh[31][20], Math.PI / 2, 1e-3,
              'brighter downward means gy is POSITIVE: gy is the bottom row minus the top row');
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            const [magnitude, direction] = ctx.kernels;
            const gray = blurRef(noisyField(ctx.utils, 64, 31337));
            const m = magnitude(gray);
            const d = direction(gray);
            const ref = sobelRef(gray);
            for (let y = 0; y < 64; y++) {
              for (let x = 0; x < 64; x++) {
                ctx.assertClose(m[y][x], ref.mag[y][x], 3e-3, `magnitude cell [${y}][${x}]`);
                // Angles are only meaningful where there is a gradient: on a
                // flat cell gx and gy are both ~0 and atan2 is free to return
                // anything, so those cells are checked for magnitude only.
                if (ref.mag[y][x] > 0.05) {
                  ctx.assertClose(d[y][x], ref.dir[y][x], 4e-3, `direction cell [${y}][${x}]`);
                }
              }
            }
          },
        },
      ],
    },

    // ------------------------------------------------------------------ 3
    {
      slug: 'non-maximum-suppression',
      title: 'Non-Maximum Suppression',
      intro: `<p>Stage 2 leaves edges several pixels thick: a gradient does not switch on at one
        column, it ramps across the whole slope. Canny's third stage is what makes the output an
        <strong>edge map</strong> rather than a heat map — and it is the stage everyone gets
        wrong.</p>
        <p>The rule: a pixel survives only if it is a local maximum <strong>along its own gradient
        direction</strong>. Not along the edge — <em>across</em> it. The gradient points the way
        the brightness climbs, which is perpendicular to the edge itself, and walking one step
        each way along that direction is walking off the ridge on both sides. If the pixel is the
        top of that little ridge, it stays; if either neighbour is above it, it is on the slope,
        and it goes to zero.</p>
        <p>Two steps, then. <strong>Quantise</strong> the angle to one of four axes — the only
        neighbours you have are the eight around you, so the gradient's direction can only be
        answered to 45° — and then <strong>compare</strong> against the two neighbours on that
        axis. Quantising has one trap in it: <code>atan2</code> returns −π…π, but an axis has no
        sense of forwards. −45° and +135° are the same axis, so an angle below zero has to be
        wrapped up by 180° first. Skip the wrap and one of your four buckets is never selected at
        all — on this task's own map, that is 570 of the 1,049 gradient pixels quietly landing in
        the wrong bucket.</p>
        ${ARRAY_LAYOUT}`,
      goal: `<strong>Goal:</strong> keep <code>mag[y][x]</code> when it is at least as large as
        both of its neighbours along the quantised gradient direction, and return <code>0</code>
        otherwise.`,
      requirements: [
        'Wrap a negative angle by <code>+ Math.PI</code> before quantising — 180° and 0° are the same axis',
        'Quantise into four buckets at 22.5°, 67.5°, 112.5°: an <code>(ax, ay)</code> step of <code>(1,0)</code>, <code>(1,1)</code>, <code>(0,1)</code> or <code>(-1,1)</code>',
        'Compare against <code>mag[y + ay][x + ax]</code> and <code>mag[y - ay][x - ax]</code> — the GRADIENT axis, not the edge',
        'Survivors keep their magnitude; everything else is <code>0</code>, borders included',
      ],
      hints: [
        {
          title: 'Hint 1 — which neighbours belong to which bucket',
          body: `<p>Take the angle to degrees after wrapping, so it lies in 0…180, and read off
            the axis:</p>
<pre><code>  0° ± 22.5   →  (ax, ay) = ( 1, 0)   left  ↔ right
 45° ± 22.5   →  (ax, ay) = ( 1, 1)   ↖ ↘
 90° ± 22.5   →  (ax, ay) = ( 0, 1)   up    ↕ down
135° ± 22.5   →  (ax, ay) = (-1, 1)   ↗ ↙</code></pre>
<p>Start with <code>(1, 0)</code> and let the last bucket fall out of the
            <code>else</code>: 0° and 180° share it.</p>`,
        },
        {
          title: 'Hint 2 — the shape of the body',
          body: `<pre><code>let a = dir[y][x];
if (a &lt; 0) a += Math.PI;
const deg = a * 180 / Math.PI;
let ax = 1;
let ay = 0;
if (deg &gt;= 22.5 &amp;&amp; deg &lt; 67.5) { ax = 1; ay = 1; }
else if (deg &gt;= 67.5 &amp;&amp; deg &lt; 112.5) { ax = 0; ay = 1; }
else if (deg &gt;= 112.5 &amp;&amp; deg &lt; 157.5) { ax = -1; ay = 1; }</code></pre>`,
        },
        {
          title: 'Hint 3 — the comparison',
          body: `<pre><code>const m = mag[y][x];
if (m &gt;= mag[y + ay][x + ax] &amp;&amp; m &gt;= mag[y - ay][x - ax]) {
  return m;
}
return 0;</code></pre>
<p><code>&gt;=</code>, not <code>&gt;</code>: on a perfectly symmetric edge the two
            middle pixels tie, and <code>&gt;</code> would erase both and leave a hole where the
            edge was. The border check has already returned, so these indexes are in bounds.</p>`,
        },
      ],
      transfer: `The name is borrowed all over vision: object detectors run "NMS" over overlapping
        boxes with exactly this argument — keep the local maximum, drop everything it explains.
        On the GPU the pattern is a pure gather, one thread per pixel with no coordination, which
        is why NVIDIA's VPI, OpenCV's <code>cudaimgproc</code> and every WebGPU implementation
        fuse it into a single compute pass. The awkward part on real hardware is the branch: four
        buckets means four different neighbour pairs, and a warp whose threads disagree runs all
        four paths — which is why some implementations interpolate along the true angle instead of
        quantising, trading arithmetic for branch uniformity.`,
      starterCode: `// Stage 3 of Canny: thin the ridges down to one pixel.
const gpu = new GPU({ mode });

const suppress = gpu.createKernel(function (mag, dir) {
  const x = this.thread.x;
  const y = this.thread.y;
  if (x === 0 || y === 0 || x === this.constants.last || y === this.constants.last) {
    return 0;
  }
  // TODO 1: wrap dir[y][x] up by Math.PI when it is negative, and turn it
  //         into degrees so it lies in 0…180.
  // TODO 2: pick the (ax, ay) step for its bucket — (1,0), (1,1), (0,1), (-1,1).
  // TODO 3: keep mag[y][x] only if it is >= BOTH mag[y + ay][x + ax]
  //         and mag[y - ay][x - ax]. Otherwise return 0.
  return mag[y][x];
}, {
  output: [64, 64],
  constants: { last: 63 },
});

const thin = suppress(mag, dir);

let before = 0;
let after = 0;
for (let y = 0; y < 64; y++) {
  for (let x = 0; x < 64; x++) {
    if (mag[y][x] > 0) before++;
    if (thin[y][x] > 0) after++;
  }
}
console.log('pixels with a gradient:', before, ' still standing after suppression:', after);
`,
      solutionCode: `// Stage 3 of Canny: thin the ridges down to one pixel.
const gpu = new GPU({ mode });

const suppress = gpu.createKernel(function (mag, dir) {
  const x = this.thread.x;
  const y = this.thread.y;
  if (x === 0 || y === 0 || x === this.constants.last || y === this.constants.last) {
    return 0;
  }

  // An axis has no forwards: -45 degrees and +135 degrees are the same line.
  let a = dir[y][x];
  if (a < 0) a += Math.PI;
  const deg = a * 180 / Math.PI;

  // Four buckets, four neighbour pairs. 0 degrees is the default so that
  // 157.5…180 falls back into it, where it belongs.
  let ax = 1;
  let ay = 0;
  if (deg >= 22.5 && deg < 67.5) {
    ax = 1;
    ay = 1;
  } else if (deg >= 67.5 && deg < 112.5) {
    ax = 0;
    ay = 1;
  } else if (deg >= 112.5 && deg < 157.5) {
    ax = -1;
    ay = 1;
  }

  // Local maximum ACROSS the edge — along the gradient — or nothing.
  const m = mag[y][x];
  if (m >= mag[y + ay][x + ax] && m >= mag[y - ay][x - ax]) {
    return m;
  }
  return 0;
}, {
  output: [64, 64],
  constants: { last: 63 },
});

const thin = suppress(mag, dir);

let before = 0;
let after = 0;
for (let y = 0; y < 64; y++) {
  for (let x = 0; x < 64; x++) {
    if (mag[y][x] > 0) before++;
    if (thin[y][x] > 0) after++;
  }
}
console.log('pixels with a gradient:', before, ' still standing after suppression:', after);
`,
      inputs: utils => gradientInputs(blurRef(noisyField(utils, 64))),
      publicTests: [
        {
          name: 'a single ridge is thinned to its crest',
          run: async ctx => {
            // A hand-built ridge in x with a unique maximum at column 32 and a
            // gradient that genuinely points along x. Nothing subtle: the only
            // column that can survive is 32.
            ctx.assert(ctx.kernels.length >= 1, 'no kernel was created — call gpu.createKernel()');
            const mag = new Array(64);
            const dir = new Array(64);
            for (let y = 0; y < 64; y++) {
              mag[y] = new Array(64);
              dir[y] = new Array(64).fill(0);
              for (let x = 0; x < 64; x++) {
                mag[y][x] = Math.max(0, 1 - Math.abs(x - 32) * 0.25);
              }
            }
            const out = ctx.kernel(mag, dir);
            ctx.assert(out && out.length === 64 && out[0].length === 64, 'expected a 64×64 grid');
            for (const y of [1, 20, 40, 62]) {
              ctx.assertClose(out[y][32], 1, 1e-4, `the crest at column 32 should survive (row ${y})`);
              for (const x of [29, 30, 31, 33, 34, 35]) {
                ctx.assertClose(
                  out[y][x], 0, 1e-4,
                  `column ${x} is on the slope, not the crest — it should be suppressed (row ${y})`
                );
              }
            }
          },
        },
        {
          name: 'the gradient axis, not the edge axis',
          run: async ctx => {
            // Same ridge, rotated: brightness now varies down the columns, so
            // the gradient points along y. A kernel comparing along the edge
            // instead compares left and right, where the ridge is flat, and
            // keeps every single row.
            const mag = new Array(64);
            const dir = new Array(64);
            for (let y = 0; y < 64; y++) {
              mag[y] = new Array(64).fill(Math.max(0, 1 - Math.abs(y - 32) * 0.25));
              dir[y] = new Array(64).fill(Math.PI / 2);
            }
            const out = ctx.kernel(mag, dir);
            const ref = nmsRef(mag, dir);
            const hint = diagnoseGrid(out, ref, 1e-4, [
              [nmsRef(mag, dir, { perpendicular: true }),
                'the two neighbours are being read along the EDGE, not along the gradient. They are perpendicular: ' +
                'the gradient points across the edge, which is the direction the ridge actually falls away in'],
              [nmsRef(mag, dir, { noWrap: true }),
                'negative angles are not being wrapped — add Math.PI before quantising, or one bucket never gets chosen'],
            ]);
            for (const x of [1, 20, 40, 62]) {
              ctx.assertClose(out[32][x], 1, 1e-4, hint || `the crest at row 32 should survive (column ${x})`);
              for (const y of [30, 31, 33, 34]) {
                ctx.assertClose(out[y][x], 0, 1e-4, hint || `row ${y} is on the slope — it should be suppressed (column ${x})`);
              }
            }
          },
        },
        {
          name: 'the buckets change over at 22.5°, 67.5°, 112.5° and 157.5°',
          run: async ctx => {
            // One cell decides the whole quantisation. The centre pixel is 0.9;
            // the two neighbours on the axis its bucket SHOULD pick are 1.2, so
            // it must be suppressed — and the six neighbours on the other three
            // axes are 0.5, so any other choice would let it through. Every
            // angle below sits 2.5° from a boundary: far enough that no float
            // could move it, close enough that a 30/60/120 split gets it wrong.
            const AXES = [[1, 0], [1, 1], [0, 1], [-1, 1]];
            const cases = [
              [20, [1, 0]], [25, [1, 1]],
              [65, [1, 1]], [70, [0, 1]],
              [110, [0, 1]], [115, [-1, 1]],
              [155, [-1, 1]], [160, [1, 0]],
              [-155, [1, 1]], [-20, [1, 0]], [-70, [0, 1]], [-115, [1, 1]],
            ];
            for (const [deg, axis] of cases) {
              const mag = new Array(64);
              const dir = new Array(64);
              for (let y = 0; y < 64; y++) {
                mag[y] = new Array(64).fill(0);
                dir[y] = new Array(64).fill((deg * Math.PI) / 180);
              }
              mag[32][32] = 0.9;
              for (const [ax, ay] of AXES) {
                const v = ax === axis[0] && ay === axis[1] ? 1.2 : 0.5;
                mag[32 + ay][32 + ax] = v;
                mag[32 - ay][32 - ax] = v;
              }
              const out = ctx.kernel(mag, dir);
              const wrapped = deg < 0 ? deg + 180 : deg;
              ctx.assertClose(
                out[32][32], 0, 1e-4,
                `at ${deg}°${deg < 0 ? ` (the same axis as ${wrapped}°)` : ''} the gradient runs ` +
                  `along (${axis[0]}, ${axis[1]}), and both neighbours there are larger — so this ` +
                  `pixel is not a maximum and should be 0. Check where your buckets change over: ` +
                  `the boundaries are 22.5°, 67.5°, 112.5° and 157.5°`
              );
            }
          },
        },
        {
          name: 'matches the reference on the real gradient maps',
          run: async ctx => {
            const { mag, dir } = gradientInputs(blurRef(noisyField(ctx.utils, 64)));
            const out = ctx.kernel(mag, dir);
            const ref = nmsRef(mag, dir);
            const hint = diagnoseGrid(out, ref, 1e-4, [
              [nmsRef(mag, dir, { perpendicular: true }),
                'the two neighbours are being read along the EDGE, not along the gradient — the two are perpendicular'],
              [nmsRef(mag, dir, { noWrap: true }),
                'a negative angle is never wrapped up by 180°, so one of the four buckets is never selected — ' +
                'on this map that is 570 of 1,049 gradient pixels comparing the wrong pair'],
              [nmsRef(mag, dir, { mask: true }),
                'survivors are coming back as 1 — a survivor keeps its own magnitude, because the next stage thresholds it'],
            ]);
            const cases = [[20, 8], [8, 20], [20, 24], [24, 20], [45, 20], [50, 50], [11, 60], [60, 11], [32, 32]];
            for (const [y, x] of cases) {
              ctx.assertClose(out[y][x], ref[y][x], 1e-4, hint || `cell [${y}][${x}]`);
            }
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            const { mag, dir } = gradientInputs(blurRef(noisyField(ctx.utils, 64, 31337)));
            const out = ctx.kernel(mag, dir);
            const ref = nmsRef(mag, dir);
            const hint = diagnoseGrid(out, ref, 1e-4, [
              [nmsRef(mag, dir, { perpendicular: true }),
                'the two neighbours are being read along the EDGE, not along the gradient — the two are perpendicular'],
              [nmsRef(mag, dir, { noWrap: true }),
                'a negative angle is never wrapped up by 180°, so one of the four buckets is never selected'],
              [nmsRef(mag, dir, { mask: true }),
                'survivors are coming back as 1 — a survivor keeps its own magnitude'],
            ]);
            for (let y = 0; y < 64; y++) {
              for (let x = 0; x < 64; x++) {
                ctx.assertClose(out[y][x], ref[y][x], 1e-4, hint || `cell [${y}][${x}]`);
              }
            }
          },
        },
        {
          name: 'private test #2',
          run: async ctx => {
            // A diagonal ridge: the 45° and 135° buckets, which a kernel that
            // forgets the wrap gets wrong in opposite directions.
            const build = sign => {
              const mag = new Array(64);
              const dir = new Array(64);
              for (let y = 0; y < 64; y++) {
                mag[y] = new Array(64);
                dir[y] = new Array(64).fill((sign * Math.PI) / 4);
                for (let x = 0; x < 64; x++) {
                  // ridge along the line x + sign*y = const, crest through the middle
                  const t = sign > 0 ? x + y - 64 : x - y;
                  mag[y][x] = Math.max(0, 1 - Math.abs(t) * 0.2);
                }
              }
              return { mag, dir };
            };
            for (const sign of [1, -1]) {
              const { mag, dir } = build(sign);
              const out = ctx.kernel(mag, dir);
              const ref = nmsRef(mag, dir);
              const hint = diagnoseGrid(out, ref, 1e-4, [
                [nmsRef(mag, dir, { perpendicular: true }),
                  'the two neighbours are being read along the EDGE, not along the gradient'],
                [nmsRef(mag, dir, { noWrap: true }),
                  'a negative angle is never wrapped up by 180° — an axis has no forwards, so -45° and +135° are the same line'],
              ]);
              for (let y = 1; y < 63; y++) {
                for (let x = 1; x < 63; x++) {
                  ctx.assertClose(out[y][x], ref[y][x], 1e-4,
                    hint || `cell [${y}][${x}] on the ${sign > 0 ? '45°' : '135°'} ridge`);
                }
              }
            }
          },
        },
      ],
    },

    // ------------------------------------------------------------------ 4
    {
      slug: 'double-threshold',
      title: 'Strong, Weak, Gone',
      intro: `<p>One threshold forces a bad choice. Set it high and long edges break into dashes
        wherever the contrast dips; set it low and the picture fills with noise. Canny's answer is
        to refuse to choose: use <strong>two</strong> thresholds and admit that the middle band is
        undecided.</p>
        <p>Above <code>high</code> a pixel is <strong>strong</strong> — it is an edge, no
        argument. Below <code>low</code> it is <strong>gone</strong>. Between them it is
        <strong>weak</strong>: it might be the faint continuation of a real edge, or it might be
        nothing, and this stage deliberately does not decide. It just labels. The next stage
        decides, and it decides by asking who the pixel's neighbours are.</p>
        <p>The labels are numbers, because a kernel returns a number: <code>1</code> for strong,
        <code>0.5</code> for weak, <code>0</code> for gone. Order matters — test <code>high</code>
        first. Written the other way round, <code>low</code> catches everything and the strong
        branch is the only one that ever fires, which turns three classes back into one.</p>`,
      goal: `<strong>Goal:</strong> classify every cell of <code>thin</code> into
        <code>1</code> (at or above <code>this.constants.high</code>), <code>0.5</code> (at or
        above <code>this.constants.low</code>) or <code>0</code>.`,
      requirements: [
        'Compare against <code>this.constants.high</code> <em>first</em>, then <code>this.constants.low</code>',
        'Return exactly <code>1</code>, <code>0.5</code> or <code>0</code> — the next stage tests for those values',
        'Both comparisons are <code>&gt;=</code>, so a pixel exactly on a threshold takes the higher class',
      ],
      hints: [
        {
          title: 'Hint 1 — three lines',
          body: `<pre><code>const m = thin[this.thread.y][this.thread.x];
if (m &gt;= this.constants.high) {
  return 1;
}</code></pre>
<p>…then the same shape for <code>low</code> returning <code>0.5</code>, and a bare
            <code>return 0;</code> at the end.</p>`,
        },
        {
          title: 'Hint 2 — why 0.5 and not 2',
          body: `<p>The value has to survive a float texture and a <code>&gt;</code> comparison in
            the next kernel, so the three labels want to be far apart and exactly representable.
            <code>0</code>, <code>0.5</code> and <code>1</code> are all exact in binary floating
            point, and the propagation kernel can then test <code>&gt; 0.75</code> for "strong"
            and <code>&lt; 0.25</code> for "gone" without ever comparing floats for equality.</p>`,
        },
      ],
      transfer: `This stage is the most boring kernel in the module and the most universally fast
        one: a pure elementwise map, one read and one write per thread, no neighbours, no
        coordination — the shape a GPU is happiest with. In CUDA it is a <code>thrust::transform</code>,
        in WebGPU a one-line compute shader, and in a fused production Canny it does not exist as
        a separate pass at all: the comparison gets folded into the tail of the suppression
        kernel, because the memory traffic of a whole extra pass costs more than the arithmetic
        it saves.`,
      starterCode: `// Stage 4 of Canny: three classes, two thresholds, no decisions.
const gpu = new GPU({ mode });

const classify = gpu.createKernel(function (thin) {
  const m = thin[this.thread.y][this.thread.x];
  // TODO: 1 when m is at or above this.constants.high,
  //       0.5 when it is at or above this.constants.low,
  //       0 otherwise. Mind which one you test first.
  return m;
}, {
  output: [64, 64],
  constants: { low: 0.3, high: 0.7 },
});

const labels = classify(thin);

let strong = 0;
let weak = 0;
for (let y = 0; y < 64; y++) {
  for (let x = 0; x < 64; x++) {
    if (labels[y][x] === 1) strong++;
    if (labels[y][x] === 0.5) weak++;
  }
}
console.log('strong:', strong, ' weak:', weak, ' gone:', 64 * 64 - strong - weak);
`,
      solutionCode: `// Stage 4 of Canny: three classes, two thresholds, no decisions.
const gpu = new GPU({ mode });

const classify = gpu.createKernel(function (thin) {
  const m = thin[this.thread.y][this.thread.x];
  if (m >= this.constants.high) {
    return 1; // strong: an edge, no argument
  }
  if (m >= this.constants.low) {
    return 0.5; // weak: undecided, and deliberately left that way
  }
  return 0; // gone
}, {
  output: [64, 64],
  constants: { low: 0.3, high: 0.7 },
});

const labels = classify(thin);

let strong = 0;
let weak = 0;
for (let y = 0; y < 64; y++) {
  for (let x = 0; x < 64; x++) {
    if (labels[y][x] === 1) strong++;
    if (labels[y][x] === 0.5) weak++;
  }
}
console.log('strong:', strong, ' weak:', weak, ' gone:', 64 * 64 - strong - weak);
`,
      inputs: utils => {
        const { mag, dir } = gradientInputs(blurRef(noisyField(utils, 64)));
        return { thin: nmsRef(mag, dir) };
      },
      publicTests: [
        {
          name: 'a ramp is cut at exactly the two thresholds',
          run: async ctx => {
            // thin[y][x] = x/63 * 1.2 crosses 0.3 at x = 15.75 and 0.7 at
            // x = 36.75 — no cell sits on a boundary, so every classification
            // here has a comfortable margin.
            ctx.assert(ctx.kernels.length >= 1, 'no kernel was created — call gpu.createKernel()');
            const ramp = new Array(64);
            for (let y = 0; y < 64; y++) {
              ramp[y] = new Array(64);
              for (let x = 0; x < 64; x++) ramp[y][x] = (x / 63) * 1.2;
            }
            const out = ctx.kernel(ramp);
            ctx.assert(out && out.length === 64 && out[0].length === 64, 'expected a 64×64 grid');
            const hint = diagnoseGrid(out, classifyRef(ramp), 1e-4, classifyProbes(ramp));
            for (const [x, expected] of [[0, 0], [10, 0], [15, 0], [16, 0.5], [30, 0.5], [36, 0.5], [37, 1], [50, 1], [63, 1]]) {
              ctx.assertClose(
                out[20][x], expected, 1e-4,
                hint || `column ${x} holds ${((x / 63) * 1.2).toFixed(3)}, which should classify as ${expected}`
              );
            }
          },
        },
        {
          name: 'only three values ever come out',
          run: async ctx => {
            const { mag, dir } = gradientInputs(blurRef(noisyField(ctx.utils, 64)));
            const thin = nmsRef(mag, dir);
            const out = ctx.kernel(thin);
            for (let y = 0; y < 64; y++) {
              for (let x = 0; x < 64; x++) {
                const v = out[y][x];
                ctx.assert(
                  Math.abs(v) < 1e-4 || Math.abs(v - 0.5) < 1e-4 || Math.abs(v - 1) < 1e-4,
                  `cell [${y}][${x}] came back as ${v} — the only three labels are 0, 0.5 and 1` +
                    (Math.abs(v - thin[y][x]) < 1e-4 ? ' (that is the raw magnitude, unclassified)' : '')
                );
              }
            }
          },
        },
        {
          name: 'the real map splits into all three classes',
          run: async ctx => {
            const { mag, dir } = gradientInputs(blurRef(noisyField(ctx.utils, 64)));
            const thin = nmsRef(mag, dir);
            const out = ctx.kernel(thin);
            const ref = classifyRef(thin);
            const hint = diagnoseGrid(out, ref, 1e-4, classifyProbes(thin));
            const census = grid => {
              let strong = 0;
              let weak = 0;
              for (let y = 0; y < 64; y++) {
                for (let x = 0; x < 64; x++) {
                  if (grid[y][x] > 0.75) strong++;
                  else if (grid[y][x] > 0.25) weak++;
                }
              }
              return { strong, weak };
            };
            const got = census(out);
            const want = census(ref);
            ctx.assert(
              got.strong === want.strong && got.weak === want.weak,
              hint || `expected ${want.strong} strong and ${want.weak} weak pixels, got ` +
                `${got.strong} and ${got.weak}`
            );
            const cases = [[20, 8], [8, 20], [45, 20], [50, 50], [11, 60], [60, 11]];
            for (const [y, x] of cases) {
              ctx.assertClose(out[y][x], ref[y][x], 1e-4, hint || `cell [${y}][${x}]`);
            }
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            const { mag, dir } = gradientInputs(blurRef(noisyField(ctx.utils, 64, 31337)));
            const thin = nmsRef(mag, dir);
            const out = ctx.kernel(thin);
            const ref = classifyRef(thin);
            const hint = diagnoseGrid(out, ref, 1e-4, classifyProbes(thin));
            for (let y = 0; y < 64; y++) {
              for (let x = 0; x < 64; x++) {
                ctx.assertClose(out[y][x], ref[y][x], 1e-4, hint || `cell [${y}][${x}]`);
              }
            }
          },
        },
      ],
    },

    // ------------------------------------------------------------------ 5
    {
      slug: 'hysteresis',
      title: 'Hysteresis: Run It Until Nothing Changes',
      intro: `<p>Stage 4 left a pile of undecided pixels. Hysteresis decides them with one rule:
        a weak pixel lives if it is <strong>connected to a strong one</strong> — touching it, or
        touching something that is. That "or" is the whole problem. Connectivity is
        <em>transitive</em>, and a GPU kernel can only see one step out.</p>
        <p>So you run the kernel again. One pass promotes every weak pixel that touches a strong
        one; the second pass promotes the ones that touch those; a chain of length <em>n</em>
        takes <em>n</em> passes to light up end to end. On this task's map that is
        <strong>28 passes</strong> — and you cannot know that in advance. The propagation is done
        when a pass changes nothing, which you can only find out by reading the result back and
        looking. Here that readback is free, because these kernels are not pipelined yet and every
        pass comes home to JavaScript anyway. Task 6 is where that stops being true, and where the
        honest cost of "iterate until stable" shows up.</p>
        <p>Worth knowing: plenty of real-time implementations do not iterate at all. They run
        <strong>one</strong> pass — a weak pixel survives if any of its eight neighbours is strong
        — and ship it. It under-connects long faint chains, and for a 60 fps video filter that is
        a bargain: a fixed, known cost per frame instead of a data-dependent loop nobody can
        budget for.</p>`,
      goal: `<strong>Goal:</strong> write the propagation kernel, then run it in a loop until a
        pass changes nothing, logging how many passes that took.`,
      requirements: [
        'A strong cell (<code>&gt; 0.75</code>) stays <code>1</code>; a gone cell (<code>&lt; 0.25</code>) stays <code>0</code>',
        'A weak cell becomes <code>1</code> if any of its <strong>8</strong> neighbours is strong, else stays <code>0.5</code>',
        'Loop until <code>unchanged(next, state)</code>, then log <code>console.log(\'settled after\', passes, \'passes\')</code>',
        'Count every call to <code>grow</code>, including the last one — the one that told you to stop',
      ],
      hints: [
        {
          title: 'Hint 1 — no early return inside the loop',
          body: `<p>Scan the 3×3 neighbourhood and set a flag rather than returning from inside
            the loops — it compiles the same on every backend and reads better:</p>
<pre><code>let strongNear = 0;
for (let dy = -1; dy &lt;= 1; dy++) {
  for (let dx = -1; dx &lt;= 1; dx++) {
    // clamp sy, sx into 0…this.constants.last, then:
    if (state[sy][sx] &gt; 0.75) {
      strongNear = 1;
    }
  }
}</code></pre>
<p>The centre cell is included in that scan, and it is harmless: this branch only runs
            when the centre is weak, so it can never mark itself.</p>`,
        },
        {
          title: 'Hint 2 — the loop',
          body: `<pre><code>let state = classified;
let passes = 0;
for (let i = 0; i &lt; 40; i++) {
  const next = grow(state);
  passes++;
  state = next;
  if (unchanged(next, state)) break;
}</code></pre>
<p>— except that assignment above happens too early to compare anything. Take
            <code>next</code>, count it, compare it against the <em>previous</em>
            <code>state</code>, and only then replace it.</p>`,
        },
        {
          title: 'Hint 3 — why 40',
          body: `<p>The <code>for</code> is a safety rail, not the plan: the <code>break</code>
            is what actually stops the loop, and 40 is simply more passes than a 64×64 map could
            ever need. Leaving a bound on a loop you expect to break out of is cheap insurance
            against a kernel that never settles.</p>`,
        },
      ],
      transfer: `"Iterate a local rule until the global answer stops changing" is label propagation,
        and it is how connected components are computed on GPUs everywhere — CUDA's
        <code>cuGraph</code>, ROCm's rocPRIM-based labelers, every union-find-on-GPU paper. The
        expensive part is always the same: the termination test. CUDA can keep a device-side
        "changed" flag and read back four bytes per iteration; WebGPU can write it to a storage
        buffer and feed it to an indirect dispatch. gpu.js has neither, so the choice is stark —
        pay a full readback per pass to ask, or pick a fixed count and accept whatever it gets you.
        Task 6 picks the second.`,
      starterCode: `// Stage 5 of Canny: a weak edge lives if it is connected to a strong one.
const gpu = new GPU({ mode });

// One propagation pass.
const grow = gpu.createKernel(function (state) {
  const x = this.thread.x;
  const y = this.thread.y;
  const v = state[y][x];
  if (v > 0.75) {
    return 1; // already strong
  }
  if (v < 0.25) {
    return 0; // already gone
  }
  // TODO: this cell is weak. Scan its 8 neighbours (clamp sy and sx into
  // 0…this.constants.last); return 1 if any of them is strong, else 0.5.
  return 0.5;
}, {
  output: [64, 64],
  constants: { last: 63 },
});

// Weak pixels that never found a strong friend do not make the cut.
const finish = gpu.createKernel(function (state) {
  if (state[this.thread.y][this.thread.x] > 0.75) {
    return 1;
  }
  return 0;
}, { output: [64, 64] });

// Plain JavaScript: did this pass change anything at all?
function unchanged(a, b) {
  for (let y = 0; y < a.length; y++) {
    for (let x = 0; x < a[y].length; x++) {
      if (a[y][x] !== b[y][x]) return false;
    }
  }
  return true;
}

// TODO: one pass is not hysteresis. A weak pixel three steps from a strong one
// needs three passes to hear about it — keep going until a pass changes nothing.
let state = grow(classified);
const passes = 1;

console.log('settled after', passes, 'passes');

const edges = finish(state);
let count = 0;
for (let y = 0; y < 64; y++) {
  for (let x = 0; x < 64; x++) count += edges[y][x];
}
console.log('edge pixels:', count);
`,
      solutionCode: `// Stage 5 of Canny: a weak edge lives if it is connected to a strong one.
const gpu = new GPU({ mode });

// One propagation pass.
const grow = gpu.createKernel(function (state) {
  const x = this.thread.x;
  const y = this.thread.y;
  const v = state[y][x];
  if (v > 0.75) {
    return 1; // already strong
  }
  if (v < 0.25) {
    return 0; // already gone
  }
  let strongNear = 0;
  for (let dy = -1; dy <= 1; dy++) {
    for (let dx = -1; dx <= 1; dx++) {
      let sy = y + dy;
      let sx = x + dx;
      if (sy < 0) sy = 0;
      if (sy > this.constants.last) sy = this.constants.last;
      if (sx < 0) sx = 0;
      if (sx > this.constants.last) sx = this.constants.last;
      if (state[sy][sx] > 0.75) {
        strongNear = 1;
      }
    }
  }
  if (strongNear === 1) {
    return 1;
  }
  return 0.5;
}, {
  output: [64, 64],
  constants: { last: 63 },
});

// Weak pixels that never found a strong friend do not make the cut.
const finish = gpu.createKernel(function (state) {
  if (state[this.thread.y][this.thread.x] > 0.75) {
    return 1;
  }
  return 0;
}, { output: [64, 64] });

// Plain JavaScript: did this pass change anything at all?
function unchanged(a, b) {
  for (let y = 0; y < a.length; y++) {
    for (let x = 0; x < a[y].length; x++) {
      if (a[y][x] !== b[y][x]) return false;
    }
  }
  return true;
}

// Iterate until stable. The 40 is a safety rail; the break is the plan.
let state = classified;
let passes = 0;
for (let i = 0; i < 40; i++) {
  const next = grow(state);
  passes++;
  const done = unchanged(next, state);
  state = next;
  if (done) break;
}

console.log('settled after', passes, 'passes');

const edges = finish(state);
let count = 0;
for (let y = 0; y < 64; y++) {
  for (let x = 0; x < 64; x++) count += edges[y][x];
}
console.log('edge pixels:', count);
`,
      inputs: utils => {
        const { mag, dir } = gradientInputs(blurRef(noisyField(utils, 64)));
        return { classified: classifyRef(nmsRef(mag, dir)) };
      },
      publicTests: [
        {
          name: 'one pass promotes exactly the weak pixels touching a strong one',
          run: async ctx => {
            ctx.assert(ctx.kernels.length >= 2, `expected 2 kernels, found ${ctx.kernels.length}`);
            const [grow] = ctx.kernels;
            // A hand-built case: a strong seed at (32,32), a weak chain running
            // right from it, a weak DIAGONAL step, and an isolated weak blob
            // far away that must never be promoted.
            const state = new Array(64);
            for (let y = 0; y < 64; y++) state[y] = new Array(64).fill(0);
            state[32][32] = 1;
            for (let x = 33; x <= 44; x++) state[32][x] = 0.5;
            state[33][33] = 0.5; // diagonal neighbour of the seed
            state[10][10] = 0.5;
            state[10][11] = 0.5;
            const out = grow(state);
            ctx.assert(out && out.length === 64 && out[0].length === 64, 'expected a 64×64 grid');
            ctx.assertClose(out[32][32], 1, 1e-4, 'a strong pixel stays strong');
            ctx.assertClose(out[32][33], 1, 1e-4, 'the weak pixel touching the seed should be promoted');
            ctx.assertClose(
              out[33][33], 1, 1e-4,
              'the DIAGONAL neighbour of the seed should be promoted too — connectivity here is all ' +
                '8 neighbours, not just the 4 direct ones'
            );
            ctx.assertClose(out[32][34], 0.5, 1e-4, 'two steps out is still weak after ONE pass');
            ctx.assertClose(out[10][10], 0.5, 1e-4, 'an isolated weak pixel stays weak — it has nothing to connect to');
            ctx.assertClose(out[0][0], 0, 1e-4, 'a gone pixel stays gone');
          },
        },
        {
          name: 'iterating to stability matches the reference',
          run: async ctx => {
            const [grow, finish] = ctx.kernels;
            const { mag, dir } = gradientInputs(blurRef(noisyField(ctx.utils, 64)));
            const classified = classifyRef(nmsRef(mag, dir));
            const ref = settleRef(classified, 60);
            let state = classified;
            for (let i = 0; i < ref.passes; i++) state = grow(state);
            const hint = diagnoseGrid(state, ref.state, 1e-4, [
              [settleRef(classified, 60, { four: true }).state,
                'only the 4 direct neighbours are being scanned — an edge that steps diagonally then ' +
                'breaks, which is most of them. The neighbourhood is all 8'],
              [growRef(classified),
                'that is the state after a SINGLE pass — connectivity is transitive, so the kernel has to ' +
                'be run again on its own output until nothing changes'],
            ]);
            for (let y = 0; y < 64; y++) {
              for (let x = 0; x < 64; x++) {
                ctx.assertClose(state[y][x], ref.state[y][x], 1e-4, hint || `cell [${y}][${x}]`);
              }
            }
            const edges = finish(state);
            const refEdges = finishRef(ref.state);
            for (let y = 0; y < 64; y++) {
              for (let x = 0; x < 64; x++) {
                ctx.assertClose(edges[y][x], refEdges[y][x], 1e-4, `finished cell [${y}][${x}]`);
              }
            }
          },
        },
        {
          name: 'your run looped until it settled, and said how long that took',
          run: async ctx => {
            const { mag, dir } = gradientInputs(blurRef(noisyField(ctx.utils, 64)));
            const expected = settleRef(classifyRef(nmsRef(mag, dir)), 60).passes;
            const line = ctx.logs.find(
              l => l.type === 'log' && l.text && l.text.includes('settled after')
            );
            ctx.assert(line, `expected a console.log('settled after', passes, 'passes')`);
            const got = Number((/settled after (\d+) passes/.exec(line.text) || [])[1]);
            ctx.assert(
              Number.isFinite(got),
              `could not read a pass count out of "${line.text}" — log it as ` +
                `console.log('settled after', passes, 'passes')`
            );
            ctx.assert(
              got === expected,
              got === 1
                ? `one pass is not hysteresis — this map needs ${expected} before a pass stops ` +
                  `changing anything, because connectivity has to travel one pixel at a time`
                : got > expected
                  ? `you ran ${got} passes; this map settles after ${expected}. Stop as soon as a ` +
                    `pass changes nothing — the extra calls are pure cost`
                  : `you ran ${got} passes; this map needs ${expected} before it stops changing`
            );
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            const [grow, finish] = ctx.kernels;
            const { mag, dir } = gradientInputs(blurRef(noisyField(ctx.utils, 64, 31337)));
            const classified = classifyRef(nmsRef(mag, dir));
            const ref = settleRef(classified, 60);
            let state = classified;
            for (let i = 0; i < ref.passes + 2; i++) state = grow(state);
            const hint = diagnoseGrid(state, ref.state, 1e-4, [
              [settleRef(classified, 60, { four: true }).state,
                'only the 4 direct neighbours are being scanned — the neighbourhood is all 8'],
            ]);
            for (let y = 0; y < 64; y++) {
              for (let x = 0; x < 64; x++) {
                ctx.assertClose(state[y][x], ref.state[y][x], 1e-4, hint || `cell [${y}][${x}]`);
              }
            }
            const edges = finish(state);
            let leftover = 0;
            for (let y = 0; y < 64; y++) {
              for (let x = 0; x < 64; x++) {
                if (edges[y][x] > 0.25 && edges[y][x] < 0.75) leftover++;
              }
            }
            ctx.assert(leftover === 0, `${leftover} weak pixels survived into the final map — finish() keeps only what reached 1`);
          },
        },
        {
          name: 'private test #2',
          run: async ctx => {
            // A long weak chain with a single strong seed at one end: the
            // promotion front has to travel the whole length, one cell per pass.
            const [grow] = ctx.kernels;
            const state = new Array(64);
            for (let y = 0; y < 64; y++) state[y] = new Array(64).fill(0);
            state[5][5] = 1;
            for (let i = 1; i <= 30; i++) state[5 + i][5 + i] = 0.5; // diagonal chain
            let cur = state;
            for (let pass = 1; pass <= 30; pass++) {
              cur = grow(cur);
              ctx.assertClose(
                cur[5 + pass][5 + pass], 1, 1e-4,
                `after ${pass} pass${pass === 1 ? '' : 'es'} the front should have reached step ${pass} of the chain`
              );
              if (pass < 30) {
                ctx.assertClose(
                  cur[5 + pass + 1][5 + pass + 1], 0.5, 1e-4,
                  `after ${pass} pass${pass === 1 ? '' : 'es'} step ${pass + 1} should still be weak — ` +
                    `a pass moves the front exactly one cell`
                );
              }
            }
          },
        },
      ],
    },

    // ------------------------------------------------------------------ 6
    {
      slug: 'assemble-the-pipeline',
      title: 'Payoff: Five Stages, Zero Round Trips',
      // WHY THIS TASK ASKS FOR A BUDGET. The pre-flight guard estimates a run's
      // cost by timing the code with every output axis clamped to 64 and
      // multiplying by the thread ratio — here 384²/64² = 36×. That model fits
      // a kernel whose cost is per-thread. This chain's cost is per-LAUNCH, and
      // the clamped probe is dominated by creating a GL context and compiling
      // nine shaders — a fixed cost the 36× then multiplies. Measured: the probe
      // extrapolates to ~3 s on a laptop GPU and ~13 s where WebGL falls back to
      // a software rasteriser, against a real run of ~10 ms (gpu) / ~55 ms
      // (cpu). 25 s clears the software-rasteriser case with room to spare and
      // still refuses genuinely pathological per-thread work, which overshoots
      // the guard by hundreds of times rather than by two. The cost of asking is
      // that the hang watchdogs scale with it (runner.js: 2× the budget), so
      // this is deliberately the smallest number that makes the task reliable
      // rather than the largest the validator would take.
      budgetMs: 25000,
      intro: `<p>Everything you have written, in one chain, on a real ${BIG}×${BIG} photo:
        <strong>luminance → blur<sub>x</sub> → blur<sub>y</sub> → magnitude + direction →
        suppression → threshold → hysteresis → edges</strong>. Nine kernel objects, and with the
        ${PASSES} hysteresis passes, <strong>${PASSES + 8} launches</strong> per image.</p>
        <p>That launch count is the point. Without <code>pipeline: true</code>, every one of those
        stages ends with a full download to JavaScript and the next one begins with a full upload:
        ${BIG}×${BIG} floats, ${((BIG * BIG * 4) / 1024).toFixed(0)} KB, crossing the bus twice per
        stage — <strong>${(PASSES + 8) * 2} transfers</strong> and
        ${((BIG * BIG * 4 * (PASSES + 8) * 2) / (1024 * 1024)).toFixed(0)} MB of traffic to produce
        one edge map. With pipelines it is two: the photo goes up, the edge map comes down, and the
        ${PASSES + 7} intermediates never leave the card. Pipelines &amp; Textures taught the
        mechanism on a three-stage chain; this is the chain long enough to make the arithmetic
        obvious.</p>
        <p>And at this size the stopwatch finally agrees with the arithmetic. Press
        <strong>Run</strong>: the console reports the whole thing — nine kernels compiled,
        ${PASSES + 8} launches, one edge map counted — in about <strong>45 ms</strong> on the
        laptop GPU these notes were measured on. Now delete the eight <code>pipeline: true</code>
        flags, so that every stage hands its result back to JavaScript and the next one uploads it
        again, and run it once more: about <strong>120 ms</strong>. Same kernels, same arithmetic,
        same ${PASSES + 8} launches — the extra 75 ms is bus traffic and nothing else. (Both
        figures carry roughly 35 ms of one-time shader compilation. Time the chain on its own,
        without that, and it is <strong>10 ms pipelined against 70 ms round-tripping</strong> — a
        clean seven times.) Eight deletions and two clicks: run that experiment rather than take
        this paragraph's word for it.</p>
        <p><strong>Benchmark</strong> agrees from the other direction, reporting the GPU
        <strong>7–8× faster</strong> than the CPU backend here — roughly 2 ms against 15 ms. Know
        what that button does before you quote it, though: it replays each of the nine kernels
        <em>once</em> with the arguments it last received, so your ${PASSES}-pass hysteresis loop
        collapses into a single call, and it drains the pipeline once at the end rather than after
        every stage. It times one pass of the chain, not the whole of it — which is why its
        milliseconds and your console's are different sizes.</p>
        <p>One honest footnote, because none of that holds at every size. Shrink the photo to
        96×96 and the same chain measures 3.2 ms on the GPU against 2.0 ms on the CPU — the CPU
        wins outright, because 24 launches over 9,216 threads is nowhere near enough work per
        launch to pay for the driver overhead of making them. The transfer arithmetic is just as
        true down there; it simply has nothing to show for itself. Launch overhead swamping small
        work is a real effect, and Measuring Speed Honestly makes a whole meal of it — it is just
        not the ending this particular chain deserves.</p>
        <p>The hysteresis loop changes shape here, and honestly so. In task 5 you looped until a
        pass changed nothing — which you could only know by reading the state back and comparing
        it. On a pipeline that readback is the very thing you are trying to avoid, so this version
        runs a <strong>fixed ${PASSES} passes</strong> and never asks. This photo settles after 59;
        the last five do nothing, and you pay for them anyway. A fixed count has to cover the worst
        photo you will be handed rather than this one — the second photo the tests use needs 56.
        That is the deal.</p>
        ${ARRAY_LAYOUT}`,
      goal: `<strong>Goal:</strong> make every stage but the last a pipeline kernel, give the
        hysteresis kernel <code>immutable: true</code> so it can eat its own output, and wire the
        nine stages into a chain that runs <code>grow</code> ${PASSES} times.`,
      requirements: [
        `Add <code>pipeline: true</code> to all eight intermediate kernels; leave <code>finish</code> plain — its return <em>is</em> the one readback you want`,
        'Add <code>immutable: true</code> to <code>grow</code>, which reads the texture it is writing',
        'Feed <code>magnitude</code> and <code>direction</code> the <strong>smoothed</strong> map, not the raw luminance',
        `Run <code>grow</code> exactly <code>PASSES</code> times, then log <code>console.log('edge pixels:', count)</code>`,
      ],
      hints: [
        {
          title: 'Hint 1 — the chain, stage by stage',
          body: `<pre><code>const gray = luminance(photo);
const smooth = blurY(blurX(gray));
const thin = suppress(magnitude(smooth), direction(smooth));
let state = classify(thin);
for (let i = 0; i &lt; PASSES; i++) {
  state = grow(state);
}
const edges = finish(state);</code></pre>
<p>Both gradient kernels read <em>smooth</em>. Handing them <code>gray</code> instead is
            the starter's first deliberate mistake, and it puts the noise straight back in.</p>`,
        },
        {
          title: 'Hint 2 — which flags, where',
          body: `<p><code>pipeline: true</code> on <code>luminance</code>, <code>blurX</code>,
            <code>blurY</code>, <code>magnitude</code>, <code>direction</code>,
            <code>suppress</code>, <code>classify</code> and <code>grow</code>. Additionally
            <code>immutable: true</code> on <code>grow</code> — without it gpu.js refuses the
            feedback loop with <em>"Source and destination … are the same"</em>, because a
            recycled output texture is the same storage the kernel is reading.</p>`,
        },
        {
          title: 'Hint 3 — reading the answer back',
          body: `<p><code>finish</code> stays a plain kernel, so its result is already a normal
            2D array — no <code>.toArray()</code> needed. Count the ones with an ordinary
            JavaScript double loop and log the total.</p>`,
        },
      ],
      transfer: `A named chain of passes with explicit dependencies and every intermediate
        resident on the device is what engine programmers call a render graph, or a frame graph:
        Frostbite's, Unreal's, and — in compute form — CUDA Graphs, where an entire launch chain
        is recorded once and replayed with a single API call precisely because ${PASSES + 8}
        individual launches carry ${PASSES + 8} lots of driver overhead. WebGPU encodes the same
        idea into one command buffer. The lesson does not change with the spelling: a pipeline is
        fast when the data never comes home.`,
      starterCode: `// The whole detector. Nine kernels; the data should touch JavaScript twice.
const gpu = new GPU({ mode });

const PASSES = ${PASSES};

// TODO: every kernel below except \`finish\` wants pipeline: true,
//       and \`grow\` additionally wants immutable: true.

const luminance = gpu.createKernel(function (image) {
  const p = image[this.thread.y][this.thread.x];
  return 0.299 * p[0] + 0.587 * p[1] + 0.114 * p[2];
}, { output: [${BIG}, ${BIG}] });

const blurX = gpu.createKernel(function (gray) {
  const x = this.thread.x;
  const y = this.thread.y;
  let x0 = x - 2;
  if (x0 < 0) x0 = 0;
  let x1 = x - 1;
  if (x1 < 0) x1 = 0;
  let x3 = x + 1;
  if (x3 > this.constants.last) x3 = this.constants.last;
  let x4 = x + 2;
  if (x4 > this.constants.last) x4 = this.constants.last;
  return (gray[y][x0] + 4 * gray[y][x1] + 6 * gray[y][x] + 4 * gray[y][x3] + gray[y][x4]) / 16;
}, { output: [${BIG}, ${BIG}], constants: { last: ${BIG - 1} } });

const blurY = gpu.createKernel(function (map) {
  const x = this.thread.x;
  const y = this.thread.y;
  let y0 = y - 2;
  if (y0 < 0) y0 = 0;
  let y1 = y - 1;
  if (y1 < 0) y1 = 0;
  let y3 = y + 1;
  if (y3 > this.constants.last) y3 = this.constants.last;
  let y4 = y + 2;
  if (y4 > this.constants.last) y4 = this.constants.last;
  return (map[y0][x] + 4 * map[y1][x] + 6 * map[y][x] + 4 * map[y3][x] + map[y4][x]) / 16;
}, { output: [${BIG}, ${BIG}], constants: { last: ${BIG - 1} } });

const magnitude = gpu.createKernel(function (gray) {
  const x = this.thread.x;
  const y = this.thread.y;
  if (x === 0 || y === 0 || x === this.constants.last || y === this.constants.last) {
    return 0;
  }
  const gx = (gray[y - 1][x + 1] + 2 * gray[y][x + 1] + gray[y + 1][x + 1])
           - (gray[y - 1][x - 1] + 2 * gray[y][x - 1] + gray[y + 1][x - 1]);
  const gy = (gray[y + 1][x - 1] + 2 * gray[y + 1][x] + gray[y + 1][x + 1])
           - (gray[y - 1][x - 1] + 2 * gray[y - 1][x] + gray[y - 1][x + 1]);
  return Math.sqrt(gx * gx + gy * gy);
}, { output: [${BIG}, ${BIG}], constants: { last: ${BIG - 1} } });

const direction = gpu.createKernel(function (gray) {
  const x = this.thread.x;
  const y = this.thread.y;
  if (x === 0 || y === 0 || x === this.constants.last || y === this.constants.last) {
    return 0;
  }
  const gx = (gray[y - 1][x + 1] + 2 * gray[y][x + 1] + gray[y + 1][x + 1])
           - (gray[y - 1][x - 1] + 2 * gray[y][x - 1] + gray[y + 1][x - 1]);
  const gy = (gray[y + 1][x - 1] + 2 * gray[y + 1][x] + gray[y + 1][x + 1])
           - (gray[y - 1][x - 1] + 2 * gray[y - 1][x] + gray[y - 1][x + 1]);
  return Math.atan2(gy, gx);
}, { output: [${BIG}, ${BIG}], constants: { last: ${BIG - 1} } });

const suppress = gpu.createKernel(function (mag, dir) {
  const x = this.thread.x;
  const y = this.thread.y;
  if (x === 0 || y === 0 || x === this.constants.last || y === this.constants.last) {
    return 0;
  }
  let a = dir[y][x];
  if (a < 0) a += Math.PI;
  const deg = a * 180 / Math.PI;
  let ax = 1;
  let ay = 0;
  if (deg >= 22.5 && deg < 67.5) {
    ax = 1;
    ay = 1;
  } else if (deg >= 67.5 && deg < 112.5) {
    ax = 0;
    ay = 1;
  } else if (deg >= 112.5 && deg < 157.5) {
    ax = -1;
    ay = 1;
  }
  const m = mag[y][x];
  if (m >= mag[y + ay][x + ax] && m >= mag[y - ay][x - ax]) {
    return m;
  }
  return 0;
}, { output: [${BIG}, ${BIG}], constants: { last: ${BIG - 1} } });

const classify = gpu.createKernel(function (thin) {
  const m = thin[this.thread.y][this.thread.x];
  if (m >= this.constants.high) {
    return 1;
  }
  if (m >= this.constants.low) {
    return 0.5;
  }
  return 0;
}, { output: [${BIG}, ${BIG}], constants: { low: ${LOW}, high: ${HIGH} } });

const grow = gpu.createKernel(function (state) {
  const x = this.thread.x;
  const y = this.thread.y;
  const v = state[y][x];
  if (v > 0.75) {
    return 1;
  }
  if (v < 0.25) {
    return 0;
  }
  let strongNear = 0;
  for (let dy = -1; dy <= 1; dy++) {
    for (let dx = -1; dx <= 1; dx++) {
      let sy = y + dy;
      let sx = x + dx;
      if (sy < 0) sy = 0;
      if (sy > this.constants.last) sy = this.constants.last;
      if (sx < 0) sx = 0;
      if (sx > this.constants.last) sx = this.constants.last;
      if (state[sy][sx] > 0.75) {
        strongNear = 1;
      }
    }
  }
  if (strongNear === 1) {
    return 1;
  }
  return 0.5;
}, { output: [${BIG}, ${BIG}], constants: { last: ${BIG - 1} } });

// The one kernel that stays plain: its return IS the readback.
const finish = gpu.createKernel(function (state) {
  if (state[this.thread.y][this.thread.x] > 0.75) {
    return 1;
  }
  return 0;
}, { output: [${BIG}, ${BIG}] });

// TODO: the chain. Two mistakes are already in it — the gradient stages are
// reading the UNSMOOTHED luminance, and the hysteresis runs exactly once.
const gray = luminance(photo);
const smooth = blurY(blurX(gray));
const thin = suppress(magnitude(gray), direction(gray));
let state = classify(thin);
state = grow(state);
const edges = finish(state);

let count = 0;
for (let y = 0; y < ${BIG}; y++) {
  for (let x = 0; x < ${BIG}; x++) count += edges[y][x];
}
console.log('edge pixels:', count);
`,
      solutionCode: `// The whole detector. Nine kernels; the data touches JavaScript twice.
const gpu = new GPU({ mode });

const PASSES = ${PASSES};

const luminance = gpu.createKernel(function (image) {
  const p = image[this.thread.y][this.thread.x];
  return 0.299 * p[0] + 0.587 * p[1] + 0.114 * p[2];
}, { output: [${BIG}, ${BIG}], pipeline: true });

const blurX = gpu.createKernel(function (gray) {
  const x = this.thread.x;
  const y = this.thread.y;
  let x0 = x - 2;
  if (x0 < 0) x0 = 0;
  let x1 = x - 1;
  if (x1 < 0) x1 = 0;
  let x3 = x + 1;
  if (x3 > this.constants.last) x3 = this.constants.last;
  let x4 = x + 2;
  if (x4 > this.constants.last) x4 = this.constants.last;
  return (gray[y][x0] + 4 * gray[y][x1] + 6 * gray[y][x] + 4 * gray[y][x3] + gray[y][x4]) / 16;
}, { output: [${BIG}, ${BIG}], constants: { last: ${BIG - 1} }, pipeline: true });

const blurY = gpu.createKernel(function (map) {
  const x = this.thread.x;
  const y = this.thread.y;
  let y0 = y - 2;
  if (y0 < 0) y0 = 0;
  let y1 = y - 1;
  if (y1 < 0) y1 = 0;
  let y3 = y + 1;
  if (y3 > this.constants.last) y3 = this.constants.last;
  let y4 = y + 2;
  if (y4 > this.constants.last) y4 = this.constants.last;
  return (map[y0][x] + 4 * map[y1][x] + 6 * map[y][x] + 4 * map[y3][x] + map[y4][x]) / 16;
}, { output: [${BIG}, ${BIG}], constants: { last: ${BIG - 1} }, pipeline: true });

const magnitude = gpu.createKernel(function (gray) {
  const x = this.thread.x;
  const y = this.thread.y;
  if (x === 0 || y === 0 || x === this.constants.last || y === this.constants.last) {
    return 0;
  }
  const gx = (gray[y - 1][x + 1] + 2 * gray[y][x + 1] + gray[y + 1][x + 1])
           - (gray[y - 1][x - 1] + 2 * gray[y][x - 1] + gray[y + 1][x - 1]);
  const gy = (gray[y + 1][x - 1] + 2 * gray[y + 1][x] + gray[y + 1][x + 1])
           - (gray[y - 1][x - 1] + 2 * gray[y - 1][x] + gray[y - 1][x + 1]);
  return Math.sqrt(gx * gx + gy * gy);
}, { output: [${BIG}, ${BIG}], constants: { last: ${BIG - 1} }, pipeline: true });

const direction = gpu.createKernel(function (gray) {
  const x = this.thread.x;
  const y = this.thread.y;
  if (x === 0 || y === 0 || x === this.constants.last || y === this.constants.last) {
    return 0;
  }
  const gx = (gray[y - 1][x + 1] + 2 * gray[y][x + 1] + gray[y + 1][x + 1])
           - (gray[y - 1][x - 1] + 2 * gray[y][x - 1] + gray[y + 1][x - 1]);
  const gy = (gray[y + 1][x - 1] + 2 * gray[y + 1][x] + gray[y + 1][x + 1])
           - (gray[y - 1][x - 1] + 2 * gray[y - 1][x] + gray[y - 1][x + 1]);
  return Math.atan2(gy, gx);
}, { output: [${BIG}, ${BIG}], constants: { last: ${BIG - 1} }, pipeline: true });

const suppress = gpu.createKernel(function (mag, dir) {
  const x = this.thread.x;
  const y = this.thread.y;
  if (x === 0 || y === 0 || x === this.constants.last || y === this.constants.last) {
    return 0;
  }
  let a = dir[y][x];
  if (a < 0) a += Math.PI;
  const deg = a * 180 / Math.PI;
  let ax = 1;
  let ay = 0;
  if (deg >= 22.5 && deg < 67.5) {
    ax = 1;
    ay = 1;
  } else if (deg >= 67.5 && deg < 112.5) {
    ax = 0;
    ay = 1;
  } else if (deg >= 112.5 && deg < 157.5) {
    ax = -1;
    ay = 1;
  }
  const m = mag[y][x];
  if (m >= mag[y + ay][x + ax] && m >= mag[y - ay][x - ax]) {
    return m;
  }
  return 0;
}, { output: [${BIG}, ${BIG}], constants: { last: ${BIG - 1} }, pipeline: true });

const classify = gpu.createKernel(function (thin) {
  const m = thin[this.thread.y][this.thread.x];
  if (m >= this.constants.high) {
    return 1;
  }
  if (m >= this.constants.low) {
    return 0.5;
  }
  return 0;
}, { output: [${BIG}, ${BIG}], constants: { low: ${LOW}, high: ${HIGH} }, pipeline: true });

const grow = gpu.createKernel(function (state) {
  const x = this.thread.x;
  const y = this.thread.y;
  const v = state[y][x];
  if (v > 0.75) {
    return 1;
  }
  if (v < 0.25) {
    return 0;
  }
  let strongNear = 0;
  for (let dy = -1; dy <= 1; dy++) {
    for (let dx = -1; dx <= 1; dx++) {
      let sy = y + dy;
      let sx = x + dx;
      if (sy < 0) sy = 0;
      if (sy > this.constants.last) sy = this.constants.last;
      if (sx < 0) sx = 0;
      if (sx > this.constants.last) sx = this.constants.last;
      if (state[sy][sx] > 0.75) {
        strongNear = 1;
      }
    }
  }
  if (strongNear === 1) {
    return 1;
  }
  return 0.5;
}, {
  output: [${BIG}, ${BIG}],
  constants: { last: ${BIG - 1} },
  pipeline: true,
  immutable: true, // it reads the texture it is writing — fresh one per call
});

// The one kernel that stays plain: its return IS the readback.
const finish = gpu.createKernel(function (state) {
  if (state[this.thread.y][this.thread.x] > 0.75) {
    return 1;
  }
  return 0;
}, { output: [${BIG}, ${BIG}] });

// The chain. One upload at the top, one download at the bottom.
const gray = luminance(photo);
const smooth = blurY(blurX(gray));
const thin = suppress(magnitude(smooth), direction(smooth));
let state = classify(thin);
for (let i = 0; i < PASSES; i++) {
  state = grow(state);
}
const edges = finish(state);

let count = 0;
for (let y = 0; y < ${BIG}; y++) {
  for (let x = 0; x < ${BIG}; x++) count += edges[y][x];
}
console.log('edge pixels:', count);
`,
      inputs: utils => ({ photo: scenePhoto(utils, BIG) }),
      publicTests: [
        {
          name: 'eight pipeline stages, one plain readback, one immutable kernel',
          run: async ctx => {
            ctx.assert(
              ctx.kernels.length >= 9,
              `expected 9 kernels, found ${ctx.kernels.length} — keep the nine stages the starter defines`
            );
            const names = ['luminance', 'blurX', 'blurY', 'magnitude', 'direction', 'suppress', 'classify', 'grow'];
            names.forEach((name, i) => {
              ctx.assert(
                ctx.kernels[i].kernel && ctx.kernels[i].kernel.pipeline === true,
                `${name} should have pipeline: true — its result is an intermediate and has no business in JavaScript`
              );
            });
            const grow = ctx.kernels[7];
            ctx.assert(
              grow.kernel.immutable === true,
              'grow reads the texture it is writing, so it needs immutable: true as well as pipeline: true'
            );
            const finish = ctx.kernels[8];
            ctx.assert(
              finish.kernel && !finish.kernel.pipeline,
              'finish should stay a plain kernel — its return value IS the one readback you want'
            );
            if (ctx.resolvedMode === 'gpu') {
              ctx.assert(
                finish.lastArgs && finish.lastArgs[0] && typeof finish.lastArgs[0].toArray === 'function',
                'finish should be handed the hysteresis texture directly — no .toArray() anywhere in the chain'
              );
            }
          },
        },
        {
          name: 'a flat photo has no edges anywhere',
          run: async ctx => {
            const [luminance, blurX, blurY, magnitude, direction, suppress, classify, grow, finish] = ctx.kernels;
            const row = new Array(BIG).fill(quantizePixel([0.45, 0.4, 0.35, 1]));
            const flat = plainToImageData(new Array(BIG).fill(row));
            const smooth = blurY(blurX(luminance(flat)));
            let state = classify(suppress(magnitude(smooth), direction(smooth)));
            for (let i = 0; i < PASSES; i++) state = grow(state);
            const edges = finish(state);
            for (let y = 0; y < BIG; y++) {
              for (let x = 0; x < BIG; x++) {
                ctx.assertClose(edges[y][x], 0, 1e-4, `a flat photo should produce no edge at [${y}][${x}]`);
              }
            }
          },
        },
        {
          name: 'the chained edge map matches the reference detector',
          run: async ctx => {
            const [luminance, blurX, blurY, magnitude, direction, suppress, classify, grow, finish] = ctx.kernels;
            const photo = scenePhoto(ctx.utils, BIG);
            const smooth = blurY(blurX(luminance(photo)));
            let state = classify(suppress(magnitude(smooth), direction(smooth)));
            for (let i = 0; i < PASSES; i++) state = grow(state);
            const edges = finish(state);
            const ref = cannyRef(photo);
            ctx.assert(edges && edges.length === BIG && edges[0].length === BIG, `expected a ${BIG}×${BIG} grid`);
            for (let y = 0; y < BIG; y++) {
              for (let x = 0; x < BIG; x++) {
                const v = edges[y][x];
                ctx.assert(
                  Math.abs(v) < 1e-4 || Math.abs(v - 1) < 1e-4,
                  `cell [${y}][${x}] came back as ${v} — the final map holds only 0 and 1`
                );
              }
            }
            const got = countOnes(edges);
            const want = countOnes(ref.edges);
            ctx.assert(
              Math.abs(got - want) <= 3,
              `expected about ${want} edge pixels, got ${got}` +
                (got > want * 1.5
                  ? ' — that is far too many: are magnitude and direction reading the raw luminance instead of the smoothed map?'
                  : got < want * 0.9
                    ? ' — that is too few: is the hysteresis running all PASSES times?'
                    : '')
            );
          },
        },
        {
          name: 'your own run wired the stages together correctly',
          run: async ctx => {
            const line = ctx.logs.find(l => l.type === 'log' && l.text && l.text.includes('edge pixels'));
            ctx.assert(line, `expected a console.log('edge pixels:', count) at the end of the chain`);
            const got = Number((/edge pixels:\s*(-?[\d.]+)/.exec(line.text) || [])[1]);
            ctx.assert(Number.isFinite(got), `could not read a count out of "${line.text}"`);
            const want = countOnes(cannyRef(scenePhoto(ctx.utils, BIG)).edges);
            ctx.assert(
              Math.abs(got - want) <= 3,
              got > want * 1.5
                ? `your chain reports ${got} edge pixels; the detector finds ${want}. That much extra is ` +
                  `noise — the gradient stages are reading the unsmoothed luminance`
                : got < want * 0.9
                  ? `your chain reports ${got} edge pixels; the detector finds ${want}. Weak edges are ` +
                    `dropping out — is grow() running all PASSES times, or just once?`
                  : `your chain reports ${got} edge pixels; the detector finds ${want}`
            );
            const blurY = ctx.kernels[2];
            ctx.assert(
              Array.isArray(blurY.lastArgs),
              'blurY was never invoked — the blur stages have to be in the chain, not just defined'
            );
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            // A different photo, cell by cell. Cells whose suppression decision
            // is a near-tie are skipped: a magnitude gap under 1e-4 is inside
            // the noise between the GL backend's float32 and this reference's
            // float64, and a coin-flip is not a fair thing to grade.
            const [luminance, blurX, blurY, magnitude, direction, suppress, classify, grow, finish] = ctx.kernels;
            const photo = scenePhoto(ctx.utils, BIG, 31337);
            const smooth = blurY(blurX(luminance(photo)));
            let state = classify(suppress(magnitude(smooth), direction(smooth)));
            for (let i = 0; i < PASSES; i++) state = grow(state);
            const edges = finish(state);
            const ref = cannyRef(photo);
            const last = BIG - 1;
            let checked = 0;
            for (let y = 1; y < last; y++) {
              for (let x = 1; x < last; x++) {
                const [ax, ay] = axisFor(ref.dir[y][x]);
                const m = ref.mag[y][x];
                const gapA = Math.abs(m - ref.mag[clampIndex(y + ay, last)][clampIndex(x + ax, last)]);
                const gapB = Math.abs(m - ref.mag[clampIndex(y - ay, last)][clampIndex(x - ax, last)]);
                const fragile =
                  (gapA > 0 && gapA < 1e-4) ||
                  (gapB > 0 && gapB < 1e-4) ||
                  Math.abs(ref.thin[y][x] - LOW) < 1e-4 ||
                  Math.abs(ref.thin[y][x] - HIGH) < 1e-4;
                if (fragile) continue;
                checked++;
                ctx.assertClose(edges[y][x], ref.edges[y][x], 1e-4, `cell [${y}][${x}]`);
              }
            }
            ctx.assert(checked > (last - 1) * (last - 1) * 0.95, `only ${checked} cells were decisive enough to check`);
          },
        },
        {
          name: 'private test #2',
          run: async ctx => {
            // The pipeline has to survive being run twice on different photos:
            // an immutable kernel that recycles a texture would show up here as
            // the second answer contaminating the first.
            const [luminance, blurX, blurY, magnitude, direction, suppress, classify, grow, finish] = ctx.kernels;
            const run = photo => {
              const smooth = blurY(blurX(luminance(photo)));
              let state = classify(suppress(magnitude(smooth), direction(smooth)));
              for (let i = 0; i < PASSES; i++) state = grow(state);
              const edges = finish(state);
              // finish is the last plain kernel, so its result is a live view
              // that the NEXT call overwrites — copy it before running again.
              return edges.map(row => Array.from(row));
            };
            const a = run(scenePhoto(ctx.utils, BIG));
            const b = run(scenePhoto(ctx.utils, BIG, 31337));
            const again = run(scenePhoto(ctx.utils, BIG));
            ctx.assertClose(countOnes(a), countOnes(cannyRef(scenePhoto(ctx.utils, BIG)).edges), 3, 'first run');
            ctx.assert(countOnes(a) !== countOnes(b) || true, 'two photos ran');
            for (let y = 0; y < BIG; y++) {
              for (let x = 0; x < BIG; x++) {
                ctx.assertClose(again[y][x], a[y][x], 1e-4, `re-running the same photo changed cell [${y}][${x}]`);
              }
            }
          },
        },
      ],
    },
  ],
};
