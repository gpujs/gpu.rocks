// Module: Video Filters — uuid 4d39e404-8261-48a2-9912-176a3ea0b1bf (short id 4d39e404).
// The file name is the uuid; identity lives in the exported object below,
// never in the path.
//
// Video Filters — the finale of the Computer Vision track.
//
// Six tasks: the 16.7 ms frame budget and what a readback costs inside a
// per-frame loop → a running temporal average (the immutable-texture feedback
// loop, now denoising) → frame differencing into a cleaned motion mask →
// an exponentially-updated background model and the foreground it segments →
// the virtual-background composite → the shippable per-frame filter, plus an
// honest account of what wiring a real camera looks like.
//
// NO WEBCAM, AND WE SAY SO. Learner code executes inside a Web Worker
// (engine/sandbox.worker.js), which has no navigator.mediaDevices, no
// getUserMedia and no HTMLVideoElement — measured in a real browser, not
// assumed. A live camera therefore cannot work in a task. The frame sequence
// built below is the stand-in, and the last task shows the real getUserMedia →
// <video> → gpu.createKernel(video) → requestAnimationFrame wiring as code
// while saying plainly why it cannot run here.
//
// Kernel-authoring rules (contract): no closures inside kernel functions, only
// numbers / nested number arrays / ImageData as inputs, this.thread.* for
// indexing, this.constants.* for compile-time values, image convention
// image[y][x] = [r, g, b, a] with channels 0–1, graphical kernels paint with
// this.color(). Every task passes in CPU mode as well as GPU mode.

import { ARRAY_LAYOUT } from '../layoutNote.js';
import { plainToImageData, quantizePixel } from '../../engine/utils.js';

// ---- the frame sequence ----------------------------------------------------
//
// Eight deterministic 64×64 frames of one scene: a static room (gradient wall,
// a poster, a darker desk band along the bottom) with a bright object crossing
// it left to right, bobbing slightly, plus two kinds of sensor noise — a small
// per-channel jitter everywhere and occasional single hot pixels.
//
// The numbers are chosen so that thresholding is never a coin flip: a pixel
// that does not change moves by at most ±0.04 in luminance, a pixel the object
// enters or leaves moves by at least 0.30, and a hot pixel moves by about 0.30.
// With the threshold at 0.12 there is a clear gap on both sides, so the same
// mask comes back from the CPU backend and the GL backend. Frame 0 is a clean
// plate — the object is still fully off the left edge — which is what lets the
// background model be seeded from it without baking a ghost into the scene.

const SIZE = 64;
const LAST = SIZE - 1;
const FRAME_COUNT = 8;
const RADIUS = 9;
const SEED = 90210;
const OBJECT = [0.85, 0.7, 0.34]; // luminance ≈ 0.704
const LUM = [0.299, 0.587, 0.114];
const HOT_PIXEL_CHANCE = 0.01;
const HOT_PIXEL_LIFT = 0.3;

// Tuning constants. These appear as literals inside the kernel sources the
// learner reads, so the references below MUST keep using the same names.
const ALPHA_SMOOTH = 0.25; // task 2 — output smoothing, follows the scene fast
const ALPHA_MODEL = 0.05; // tasks 4-6 — background model, learns slowly
const THRESHOLD = 0.12; // tasks 3, 4, 6 — luminance change that counts as motion
const RAMP_LO = 0.1; // task 5 — soft mask ramp
const RAMP_HI = 0.22;

// The scene below is built at a size rather than at SIZE, because the module
// card wants this same shot at four times the pixels (see cardInputs on task 5)
// and everything about it is measured in pixels: where the object is, how big
// it is, where the poster and the desk edge fall. Each constant is therefore
// written relative to SIZE and multiplied by size / SIZE — which is 1 for the
// lesson, so the frames a learner gets are the frames they always got.

// Where the object is in frame i. It starts fully off the left edge, so frame 0
// shows the empty scene.
function objectCenter(i, size = SIZE) {
  const s = size / SIZE;
  return [(-12 + 9 * i) * s, (30 + Math.round(4 * Math.sin(i * 0.8))) * s];
}

// The static scene, identical in every frame. The two rectangles are half-open
// in scene coordinates — `>= 6 * s` up to `< 23 * s` is x = 6…22 at s = 1 and
// exactly four times that span at s = 4, which `<= 22 * s` would not be.
function scenePixel(x, y, size = SIZE) {
  const s = size / SIZE;
  const nx = x / (size - 1);
  const ny = y / (size - 1);
  let r = 0.1 + 0.18 * nx;
  let g = 0.16 + 0.2 * ny;
  let b = 0.3 + 0.16 * (1 - nx);
  if (x >= 6 * s && x < 23 * s && y >= 8 * s && y < 25 * s) {
    r += 0.14; // the poster on the wall
    g += 0.14;
    b += 0.02;
  }
  if (y >= 46 * s) {
    r *= 0.55; // the desk, in shadow
    g *= 0.55;
    b *= 0.55;
  }
  return [r, g, b];
}

// The sequence, as ImageData — the one image shape every gpu.js backend reads
// on the GPU (engine/utils.plainToImageData). Channels are quantized to 8-bit
// steps, which is what makes .plain an exact host-side view of what the kernel
// sees. Same seed → same eight frames, always.
// The sensor noise stays per-PIXEL rather than scaling with size: a hot pixel is
// a pixel, so the card gets the same 1% of them, finer-grained. Everything with
// a position or a size scales.
function makeFrames(utils, seed = SEED, size = SIZE) {
  const frames = new Array(FRAME_COUNT);
  const radius = RADIUS * (size / SIZE);
  for (let i = 0; i < FRAME_COUNT; i++) {
    const rand = utils.seededRandom(seed + i * 7919);
    const [cx, cy] = objectCenter(i, size);
    const plain = new Array(size);
    for (let y = 0; y < size; y++) {
      const row = new Array(size);
      for (let x = 0; x < size; x++) {
        const dx = x - cx;
        const dy = y - cy;
        const base = dx * dx + dy * dy <= radius * radius ? OBJECT : scenePixel(x, y, size);
        const hot = rand() < HOT_PIXEL_CHANCE ? HOT_PIXEL_LIFT : 0;
        row[x] = quantizePixel([
          base[0] + hot + (rand() - 0.5) * 0.04,
          base[1] + hot + (rand() - 0.5) * 0.04,
          base[2] + hot + (rand() - 0.5) * 0.04,
          1,
        ]);
      }
      plain[y] = row;
    }
    frames[i] = plainToImageData(plain);
  }
  return frames;
}

// ---- host-side references --------------------------------------------------

function grid(fn) {
  const out = new Array(SIZE);
  for (let y = 0; y < SIZE; y++) {
    const row = new Array(SIZE);
    for (let x = 0; x < SIZE; x++) row[x] = fn(y, x);
    out[y] = row;
  }
  return out;
}

function clampIdx(i) {
  return i < 0 ? 0 : i > LAST ? LAST : i;
}

function lumOf(pixel) {
  return LUM[0] * pixel[0] + LUM[1] * pixel[1] + LUM[2] * pixel[2];
}

// A frame's luminance, host-side. `.plain` is the nested [y][x] = [r, g, b, a]
// view every course image carries — tests read that, never the ImageData bytes.
function lumMap(frame) {
  const plain = frame.plain;
  return grid((y, x) => lumOf(plain[y][x]));
}

// 3×3 mean with clamped edges — the box blur from Convolution & Filters.
function refBox3(map) {
  return grid((y, x) => {
    let sum = 0;
    for (let dy = -1; dy <= 1; dy++) {
      for (let dx = -1; dx <= 1; dx++) sum += map[clampIdx(y + dy)][clampIdx(x + dx)];
    }
    return sum / 9;
  });
}

function refTone(v) {
  return Math.min(Math.max((v - 0.35) * 1.8 + 0.5, 0), 1);
}

// Task 1: one frame through luminance → 3×3 denoise → tone curve.
function refPipeline(frame) {
  const blurred = refBox3(lumMap(frame));
  return grid((y, x) => refTone(blurred[y][x]));
}

// Task 2: the running average over the whole sequence, seeded from frame 0.
function refTemporal(frames, alpha) {
  let state = lumMap(frames[0]);
  for (let i = 1; i < frames.length; i++) {
    const now = lumMap(frames[i]);
    const previous = state;
    state = grid((y, x) => (1 - alpha) * previous[y][x] + alpha * now[y][x]);
  }
  return state;
}

// Task 3: |Δluminance| between two frames, and the binary mask it thresholds to.
function refDelta(previous, current) {
  const a = lumMap(previous);
  const b = lumMap(current);
  return grid((y, x) => Math.abs(b[y][x] - a[y][x]));
}

function refThreshold(delta, threshold) {
  return grid((y, x) => (delta[y][x] > threshold ? 1 : 0));
}

// Task 3: 3×3 majority vote — five of nine neighbours have to agree. Isolated
// hot pixels lose; the interior of a real blob does not.
function refMajority(mask) {
  return grid((y, x) => {
    let votes = 0;
    for (let dy = -1; dy <= 1; dy++) {
      for (let dx = -1; dx <= 1; dx++) votes += mask[clampIdx(y + dy)][clampIdx(x + dx)];
    }
    return votes >= 5 ? 1 : 0;
  });
}

// Tasks 4-6: the background model after folding in frames 1…upto.
function refModel(frames, alpha, upto) {
  let model = lumMap(frames[0]);
  for (let i = 1; i <= upto; i++) {
    const now = lumMap(frames[i]);
    const previous = model;
    model = grid((y, x) => (1 - alpha) * previous[y][x] + alpha * now[y][x]);
  }
  return model;
}

// Task 4: segment against the model as it stands, THEN fold the frame in.
// Returns every mask plus the model that produced the last one.
function refSegmentRun(frames, alpha, threshold) {
  let model = lumMap(frames[0]);
  const masks = [];
  const models = [];
  for (let i = 1; i < frames.length; i++) {
    const now = lumMap(frames[i]);
    models.push(model);
    masks.push(grid((y, x) => (Math.abs(now[y][x] - model[y][x]) > threshold ? 1 : 0)));
    const previous = model;
    model = grid((y, x) => (1 - alpha) * previous[y][x] + alpha * now[y][x]);
  }
  return { masks, models, model };
}

// Task 5: the soft mask — a ramp from lo to hi, normalised into 0…1.
function refSoftMask(frame, model, lo, hi) {
  const now = lumMap(frame);
  return grid((y, x) => {
    const d = Math.abs(now[y][x] - model[y][x]);
    return Math.min(Math.max((d - lo) / (hi - lo), 0), 1);
  });
}

// Task 5: foreground over a 5×5-blurred copy of the same frame.
function refCompose(frame, mask, blend) {
  const plain = frame.plain;
  return grid((y, x) => {
    let sr = 0;
    let sg = 0;
    let sb = 0;
    for (let dy = -2; dy <= 2; dy++) {
      for (let dx = -2; dx <= 2; dx++) {
        const q = plain[clampIdx(y + dy)][clampIdx(x + dx)];
        sr += q[0];
        sg += q[1];
        sb += q[2];
      }
    }
    const src = plain[y][x];
    const back = [sr / 25, sg / 25, sb / 25];
    return blend(src, back, mask[y][x]);
  });
}

function overBlurred(src, back, m) {
  return [src[0] * m + back[0] * (1 - m), src[1] * m + back[1] * (1 - m), src[2] * m + back[2] * (1 - m)];
}

// Task 6: the per-frame moving-pixel counts a shippable filter would report,
// first frame included — it is seeded from itself, so its mask is empty.
function refLiveCounts(frames, alpha, threshold) {
  let model = null;
  const counts = [];
  for (let i = 0; i < frames.length; i++) {
    const now = lumMap(frames[i]);
    if (model === null) model = now;
    let count = 0;
    for (let y = 0; y < SIZE; y++) {
      for (let x = 0; x < SIZE; x++) {
        if (Math.abs(now[y][x] - model[y][x]) > threshold) count++;
      }
    }
    counts.push(count);
    const previous = model;
    model = grid((y, x) => (1 - alpha) * previous[y][x] + alpha * now[y][x]);
  }
  return counts;
}

// ---- reading results -------------------------------------------------------

// Mode-safe read of a pipeline result: a Texture on GL, a plain array on CPU.
async function toArr(value) {
  return value && typeof value.toArray === 'function' ? await value.toArray() : value;
}

function sumGrid(map) {
  let total = 0;
  for (let y = 0; y < SIZE; y++) {
    for (let x = 0; x < SIZE; x++) total += map[y][x];
  }
  return total;
}

function logged(ctx, text) {
  return ctx.logs.some(line => line.type === 'log' && line.text && line.text.includes(text));
}

// ---- near-miss diagnosis ---------------------------------------------------
//
// A failing assert that reports two numbers tells a learner nothing about WHICH
// slip produced them. A probe pairs the value one specific known mistake would
// produce with a sentence naming that mistake; diagnose() speaks only when the
// observation matches a probe within the test's own tolerance AND the correct
// answer does not — so a cell where two candidates coincide stays silent, as do
// observations matching probes that disagree with each other. A wrong diagnosis
// is worse than none.
function diagnose(got, expected, eps, probes) {
  const hits = probes
    .filter(p => Math.abs(got - p[0]) <= eps && Math.abs(expected - p[0]) > eps)
    .map(p => p[1]);
  return hits.length && hits.every(m => m === hits[0]) ? hits[0] : null;
}

// The whole-grid form. A single matching cell is weak evidence when a candidate
// is built from the same data the right answer is, so these probes have to
// predict EVERY cell (and disagree with the right answer somewhere) before they
// may speak. `skip` optionally masks out cells the test does not assert.
function diagnoseGrid(got, expected, eps, probes, skip) {
  const hits = probes
    .filter(([candidate]) => {
      let differs = false;
      for (let y = 0; y < SIZE; y++) {
        for (let x = 0; x < SIZE; x++) {
          if (skip && skip[y][x]) continue;
          const row = got[y];
          if (!row || !(Math.abs(row[x] - candidate[y][x]) <= eps)) return false;
          if (Math.abs(expected[y][x] - candidate[y][x]) > eps) differs = true;
        }
      }
      return differs;
    })
    .map(p => p[1]);
  return hits.length && hits.every(m => m === hits[0]) ? hits[0] : null;
}

// A thresholded mask is a step function, and a cell whose difference sits ON
// the step is decided by the last bit of a float — legitimately different
// between the CPU backend's float64 and the GL backend's float32. The sequence
// is built so that essentially nothing lands there, but a test asserts only the
// cells that are not on the fence, and says so rather than pretending.
const FENCE = 5e-3;

function fenceOf(delta, threshold) {
  return grid((y, x) => (Math.abs(delta[y][x] - threshold) < FENCE ? 1 : 0));
}

// A cleaned mask is a function of a 3×3 neighbourhood, so one fence cell puts
// nine output cells beyond the test's reach.
function dilateFence(fence) {
  return grid((y, x) => {
    for (let dy = -1; dy <= 1; dy++) {
      for (let dx = -1; dx <= 1; dx++) {
        if (fence[clampIdx(y + dy)][clampIdx(x + dx)]) return 1;
      }
    }
    return 0;
  });
}

// Compares a whole grid, skipping fence cells, and reports the first cell that
// disagrees. Returns null when everything matched.
function firstMismatch(got, expected, eps, skip) {
  for (let y = 0; y < SIZE; y++) {
    for (let x = 0; x < SIZE; x++) {
      if (skip && skip[y][x]) continue;
      const row = got[y];
      const value = row ? row[x] : undefined;
      if (!(Math.abs(value - expected[y][x]) <= eps)) {
        return { y, x, got: value, want: expected[y][x] };
      }
    }
  }
  return null;
}

// ---- canvas comparison (task 5) -------------------------------------------
//
// getPixels() row order is top-down on one backend and bottom-up on another, so
// a painted image is compared against a reference under BOTH orders and has to
// match one of them everywhere. Same rule for every candidate a probe offers,
// which is what keeps a diagnosis from being an artifact of row order.

function paintedMatches(pixels, ref, tol, flip) {
  for (let row = 0; row < SIZE; row++) {
    const source = ref[flip ? LAST - row : row];
    for (let col = 0; col < SIZE; col++) {
      const i = (row * SIZE + col) * 4;
      const want = source[col];
      for (let c = 0; c < 3; c++) {
        if (Math.abs(pixels[i + c] - Math.min(Math.max(want[c], 0), 1) * 255) > tol) return false;
      }
    }
  }
  return true;
}

function canvasMatches(pixels, ref, tol) {
  return paintedMatches(pixels, ref, tol, false) || paintedMatches(pixels, ref, tol, true);
}

function diagnoseCanvas(pixels, tol, probes) {
  const hits = probes.filter(p => canvasMatches(pixels, p[0], tol)).map(p => p[1]);
  return hits.length && hits.every(m => m === hits[0]) ? hits[0] : null;
}

// ---- task-specific probes --------------------------------------------------

// Task 1: createKernel is a transpile plus a driver compile. Doing it per frame
// is the classic real-time own-goal, and the kernel count gives it away exactly.
function perFrameKernelHint(count) {
  return count > 3 && count % 3 === 0
    ? `${count} kernels were created for a three-stage pipeline — createKernel transpiles your ` +
      `function and compiles a shader, so it belongs ABOVE the frame loop, not inside it. ` +
      `Build three kernels once and call them ${count / 3} times each.`
    : null;
}

// Task 2: the two weights are trivially swappable, and a blend that never
// blends looks fine until you compare it with anything.
function temporalProbes(frames, alpha) {
  return [
    [refTemporal(frames, 1 - alpha),
      `the two weights are swapped — the NEW frame carries alpha and the running average ` +
      `carries 1 − alpha, so most of what you keep is history`],
    [lumMap(frames[frames.length - 1]),
      'that is the last frame\'s luminance on its own — the previous state never reached the result'],
    [lumMap(frames[0]),
      'that is frame 0\'s luminance — the state is being passed through unchanged, so no frame after the first one is doing anything'],
  ];
}

// Task 3: the four ways a frame difference goes wrong, three of which produce a
// perfectly plausible-looking mask.
function motionProbes(previous, current, threshold) {
  const a = lumMap(previous);
  const b = lumMap(current);
  const signed = grid((y, x) => b[y][x] - a[y][x]);
  return [
    [refThreshold(signed, threshold),
      'only pixels that got BRIGHTER are marked — Math.abs is missing, so half of every moving edge is invisible'],
    [grid((y, x) => (Math.abs(b[y][x] - a[y][x]) > threshold ? 0 : 1)),
      'the mask is inverted — the comparison marks the pixels that did NOT change'],
    [grid((y, x) => Math.abs(b[y][x] - a[y][x])),
      'that is the raw difference, not a mask — compare it against the threshold and return 1 or 0'],
    [grid(() => 0),
      'every cell is 0 — nothing anywhere cleared the threshold, so the comparison is never reaching its return 1'],
  ];
}

// Task 4: alpha on the wrong term is one character and completely changes what
// the model is. Both failures are silent — the model still looks like a scene.
function modelProbes(frames, alpha, upto) {
  return [
    [refModel(frames, 1 - alpha, upto),
      `alpha is on the wrong term — the model keeps (1 − alpha) of itself and takes alpha of the ` +
      `new frame. With them swapped the model tracks the frame instead of the scene, so nothing ` +
      `is ever left over to call foreground`],
    [lumMap(frames[0]),
      'the model never changes — it is still frame 0. The update has to fold the new frame in, not return the model it was given'],
    [lumMap(frames[upto]),
      'the model IS the current frame — that is alpha = 1, which learns the foreground as fast as it learns the scene'],
  ];
}

export default {
  uuid: '4d39e404-8261-48a2-9912-176a3ea0b1bf',
  version: 1,
  slug: 'video-filters',
  title: 'Video Filters',
  blurb: 'Sixteen milliseconds a frame, and state that survives between them: temporal filtering, motion masks and a background model.',
  tasks: [
    // ------------------------------------------------------------------ 1
    {
      slug: 'frame-budget',
      title: 'Sixteen Milliseconds',
      intro: `<p>Everything so far in this track processed <em>one</em> picture. Video changes the
        economics completely. At 60 frames per second you get <strong>16.7 milliseconds</strong> to
        do all of it — decode, filter, composite, present — and then the frame is gone whether you
        were finished or not. Miss the budget and you do not get a slower filter, you get a stuttering
        one.</p>
        <p>The same three kernels now run eight times instead of once, which turns two habits that
        were merely wasteful into the whole problem. The first is creating kernels inside the loop:
        <code>createKernel</code> transpiles your JavaScript to shader source and compiles it, so
        doing it per frame pays the compiler sixty times a second. The second is the readback —
        Pipelines &amp; Textures showed the mechanism, and here it is the difference between a filter
        that runs and one that does not.</p>
        <p>The starter below is honest, working, and unshippable. Fix its structure, then read the
        numbers it prints — and hit <strong>⏱ Benchmark</strong> afterwards to see the same argument
        made twice.</p>
        ${ARRAY_LAYOUT}`,
      goal: `<strong>Goal:</strong> hoist the three kernels out of the frame loop, keep the two
        intermediate stages on the GPU, and report the per-frame cost against the 16.7 ms budget.`,
      requirements: [
        'Create the three kernels <strong>once</strong>, above the loop — exactly three for the whole run',
        'Give <code>luminance</code> and <code>denoise</code> <code>pipeline: true</code>; <code>tone</code> stays plain',
        'Process every frame in <code>frames</code>, with no <code>.toArray()</code> inside the loop',
        `Log <code>'processed', out.length, 'frames'</code> and a per-frame line carrying the ms and the fps`,
      ],
      hints: [
        {
          title: 'Hint 1 — what belongs in the loop',
          body: `<p>Everything that depends on <em>which</em> frame you are looking at, and nothing
            else. A kernel does not depend on the frame — it takes one as an argument. Three
            <code>createKernel</code> calls, then a loop that does nothing but call them.</p>`,
        },
        {
          title: 'Hint 2 — where the readbacks are hiding',
          body: `<p>There is no <code>.toArray()</code> in the starter, and the readbacks are still
            there: a non-pipeline kernel's <em>return value</em> is the readback. Add
            <code>pipeline: true</code> to the first two stages and the chain collapses to one
            expression:</p>
<pre><code>out.push(await tone(await denoise(await luminance(frames[i]))));</code></pre>`,
        },
        {
          title: 'Hint 3 — the budget arithmetic',
          body: `<p>One frame at 60 fps is <code>1000 / 60 = 16.7</code> ms. So:</p>
<pre><code>const perFrame = totalMs / frames.length;
const fps = 1000 / perFrame;</code></pre>
<p>and the verdict is just <code>perFrame &lt;= 16.7</code>.</p>`,
        },
      ],
      transfer: `Every real-time GPU API separates "build the pipeline" from "run it", precisely so
        the expensive half happens once: WebGPU's <code>createRenderPipeline</code> versus
        <code>dispatchWorkgroups</code>, Vulkan's pipeline objects, CUDA modules loaded once and
        launched forever. And the per-frame budget is why frame graphs exist at all — a readback
        mid-frame is a full pipeline stall on every one of them.`,
      starterCode: `// Eight frames, three stages each. Works. Would never ship.
const gpu = new GPU({ mode });

const t0 = performance.now();
const out = [];

for (let i = 0; i < frames.length; i++) {
  // TODO: createKernel transpiles your function and compiles a shader.
  // Doing it here pays that bill on every single frame — hoist all three
  // of these above the loop so they are built once and called eight times.
  const luminance = gpu.createKernel(function (frame) {
    const p = frame[this.thread.y][this.thread.x];
    return 0.299 * p[0] + 0.587 * p[1] + 0.114 * p[2];
  }, { output: [64, 64] });

  const denoise = gpu.createKernel(function (map) {
    let sum = 0;
    for (let dy = -1; dy <= 1; dy++) {
      for (let dx = -1; dx <= 1; dx++) {
        let yy = this.thread.y + dy;
        let xx = this.thread.x + dx;
        if (yy < 0) yy = 0;
        if (yy > this.constants.last) yy = this.constants.last;
        if (xx < 0) xx = 0;
        if (xx > this.constants.last) xx = this.constants.last;
        sum += map[yy][xx];
      }
    }
    return sum / 9;
  }, { output: [64, 64], constants: { last: 63 } });

  const tone = gpu.createKernel(function (map) {
    const v = map[this.thread.y][this.thread.x];
    return Math.min(Math.max((v - 0.35) * 1.8 + 0.5, 0), 1);
  }, { output: [64, 64] });

  // TODO: each of these three stages ends in a download and the next one
  // re-uploads. Only the LAST stage should come back to JavaScript — make
  // the first two pipeline kernels.
  const lum = await luminance(frames[i]);
  const clean = await denoise(lum);
  out.push(await tone(clean));
}

const totalMs = performance.now() - t0;

// TODO: 60 fps is one frame every 16.7 ms. Work out the per-frame cost,
// log it with the frame rate it implies, and say whether it fits:
//   console.log('processed', out.length, 'frames');
//   console.log('per frame:', perFrame.toFixed(2), 'ms -', fps.toFixed(0), 'fps');
console.log('total:', totalMs.toFixed(2), 'ms');
`,
      solutionCode: `// Eight frames, three stages each. Built once, run eight times.
const gpu = new GPU({ mode });

// Stage 1 — luminance. Stays on the card.
const luminance = gpu.createKernel(function (frame) {
  const p = frame[this.thread.y][this.thread.x];
  return 0.299 * p[0] + 0.587 * p[1] + 0.114 * p[2];
}, { output: [64, 64], pipeline: true });

// Stage 2 — 3×3 denoise. Also stays on the card.
const denoise = gpu.createKernel(function (map) {
  let sum = 0;
  for (let dy = -1; dy <= 1; dy++) {
    for (let dx = -1; dx <= 1; dx++) {
      let yy = this.thread.y + dy;
      let xx = this.thread.x + dx;
      if (yy < 0) yy = 0;
      if (yy > this.constants.last) yy = this.constants.last;
      if (xx < 0) xx = 0;
      if (xx > this.constants.last) xx = this.constants.last;
      sum += map[yy][xx];
    }
  }
  return sum / 9;
}, { output: [64, 64], pipeline: true, constants: { last: 63 } });

// Stage 3 — tone curve. Final stage, so its return IS the readback you want.
const tone = gpu.createKernel(function (map) {
  const v = map[this.thread.y][this.thread.x];
  return Math.min(Math.max((v - 0.35) * 1.8 + 0.5, 0), 1);
}, { output: [64, 64] });

const t0 = performance.now();
const out = [];
for (let i = 0; i < frames.length; i++) {
  out.push(await tone(await denoise(await luminance(frames[i]))));
}
const totalMs = performance.now() - t0;

const perFrame = totalMs / frames.length;
const fps = 1000 / perFrame;
console.log('processed', out.length, 'frames');
console.log('per frame:', perFrame.toFixed(2), 'ms -', fps.toFixed(0), 'fps');
console.log(perFrame <= 16.7 ? 'fits the 16.7 ms frame budget' : 'over the 16.7 ms frame budget');
`,
      inputs: utils => ({ frames: makeFrames(utils) }),
      inputNotes: {
        frames: 'Eight 64×64 ImageData frames in time order — a bright object crossing a static scene, with sensor noise. Pass one frame at a time into a kernel; inside it frames[i][y][x] is that pixel as [r, g, b, a] from 0 to 1.',
      },
      publicTests: [
        {
          name: 'three kernels for the whole run — not three per frame',
          run: async ctx => {
            ctx.assert(ctx.kernels.length >= 1, 'no kernel was created — call gpu.createKernel()');
            ctx.assert(
              ctx.kernels.length === 3,
              perFrameKernelHint(ctx.kernels.length) ||
                `expected exactly 3 kernels (one per stage), found ${ctx.kernels.length}`
            );
          },
        },
        {
          name: 'stages 1 and 2 keep their results on the GPU',
          run: async ctx => {
            const [lum, denoise, tone] = ctx.kernels;
            ctx.assert(
              lum.kernel && lum.kernel.pipeline === true,
              'the luminance stage should have pipeline: true — its result is not for JavaScript'
            );
            ctx.assert(
              denoise.kernel && denoise.kernel.pipeline === true,
              'the denoise stage should have pipeline: true — its result is not for JavaScript either'
            );
            ctx.assert(
              tone.kernel && !tone.kernel.pipeline,
              'the tone stage should stay a plain kernel — its return IS the one readback you want'
            );
            if (ctx.resolvedMode !== 'cpu') {
              ctx.assert(
                denoise.lastArgs && denoise.lastArgs[0] && typeof denoise.lastArgs[0].toArray === 'function',
                'the denoise stage should be fed the luminance texture directly — no readback in between'
              );
            }
          },
        },
        {
          name: 'the chain still computes the same picture, frame by frame',
          run: async ctx => {
            const [lum, denoise, tone] = ctx.kernels;
            const seq = makeFrames(ctx.utils);
            for (const i of [0, 3, 7]) {
              const out = await tone(await denoise(await lum(seq[i])));
              const ref = refPipeline(seq[i]);
              const miss = firstMismatch(out, ref, 3e-3);
              ctx.assert(
                !miss,
                miss && `frame ${i}, cell [${miss.y}][${miss.x}] — expected ${miss.want.toFixed(4)}, got ${miss.got}`
              );
            }
          },
        },
        {
          name: 'every frame is processed, and the per-frame cost is reported',
          run: async ctx => {
            ctx.assert(
              logged(ctx, 'processed 8 frames'),
              `all eight frames should go through the loop — expected console.log('processed', out.length, 'frames') to print "processed 8 frames"`
            );
            ctx.assert(
              logged(ctx, 'fps'),
              'log the per-frame cost and the frame rate it implies — 1000 / perFrame is the fps'
            );
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            // A different sequence: hardcoding what the public frames produce
            // will not survive this.
            const [lum, denoise, tone] = ctx.kernels;
            const seq = makeFrames(ctx.utils, 31337);
            for (let i = 0; i < seq.length; i++) {
              const out = await tone(await denoise(await lum(seq[i])));
              const ref = refPipeline(seq[i]);
              const miss = firstMismatch(out, ref, 3e-3);
              ctx.assert(
                !miss,
                miss && `frame ${i}, cell [${miss.y}][${miss.x}] — expected ${miss.want.toFixed(4)}, got ${miss.got}`
              );
            }
          },
        },
      ],
    },

    // ------------------------------------------------------------------ 2
    {
      slug: 'temporal-average',
      title: 'Averaging Across Time',
      intro: `<p>A single image can only be denoised by borrowing from its <em>neighbours in space</em>
        — that is what a blur is, and it costs you detail. Video hands you a second axis for free.
        The pixel at (12, 40) is being measured sixty times a second, and the scene is not changing
        that fast; the noise is. Average a pixel with <em>itself</em> across frames and the noise
        falls away while the edges stay exactly where they were.</p>
        <p>The cheap way to do it is a <strong>running average</strong>, one line long and with no
        history to store:</p>
<pre><code>avg = (1 - alpha) * avg + alpha * now</code></pre>
        <p>Each frame nudges the average a little toward itself. With <code>alpha = 0.25</code> a
        change takes a few frames to fully arrive — which is the trade: small <code>alpha</code>
        denoises harder and smears motion into a comet tail, large <code>alpha</code> keeps motion
        crisp and keeps the noise with it.</p>
        <p>Structurally this is the feedback loop from Pipelines &amp; Textures: the kernel reads the
        texture it is about to replace. <code>immutable: true</code> is what makes that legal —
        every call renders to a <em>fresh</em> texture, so last frame's average is safe to read while
        this frame's is being written. Leave it out and gpu.js stops you with the reason; that is the
        library refusing to let you read a half-written buffer.</p>
        ${ARRAY_LAYOUT}`,
      goal: `<strong>Goal:</strong> finish the <code>blend</code> kernel — <code>(1 − alpha)</code>
        of the previous average plus <code>alpha</code> of this frame's luminance — and make the
        feedback loop legal with <code>immutable: true</code>.`,
      requirements: [
        'Add <code>immutable: true</code> to <code>blend</code> (keep <code>pipeline: true</code>)',
        'Compute this frame\'s luminance inside <code>blend</code>: <code>0.299r + 0.587g + 0.114b</code>',
        'Return <code>(1 - alpha) * previous + alpha * now</code>, with <code>alpha</code> from <code>this.constants</code>',
      ],
      hints: [
        {
          title: 'Hint 1 — run it first',
          body: `<p>The starter throws, and the message names both the crime and the sentence: the
            kernel's input and output are the same storage, and <code>immutable = true</code> is the
            fix. gpu.js error messages are unusually honest.</p>`,
        },
        {
          title: 'Hint 2 — which weight goes where',
          body: `<p>The new frame is the small contribution — it is one sample out of many. So
            <code>alpha</code> multiplies <code>now</code>, and <code>1 - alpha</code> multiplies the
            average you already had:</p>
<pre><code>return (1 - this.constants.alpha) * previous[this.thread.y][this.thread.x]
     + this.constants.alpha * now;</code></pre>`,
        },
        {
          title: 'Hint 3 — why the loop starts at 1',
          body: `<p>Frame 0 has no predecessor, so it cannot be blended with anything — it
            <em>is</em> the starting average, which is what <code>seed</code> produces. The blending
            starts at frame 1. Every stateful video filter has this line, and forgetting it is how
            you get a garbage or NaN first frame.</p>`,
        },
      ],
      transfer: `An exponential moving average over frames is the cheapest temporal filter there is,
        and it is everywhere: TAA in game engines accumulates jittered samples into a history buffer
        exactly like this, denoisers for real-time ray tracing blend the current noisy estimate into
        an exponential history, and camera ISPs run one per pixel in hardware. The interesting part
        of all of them is not this line — it is deciding when to <em>throw the history away</em>
        because the scene moved.`,
      starterCode: `// Denoising along the time axis: average each pixel with itself.
const gpu = new GPU({ mode });

// The starting average is simply frame 0's luminance — nothing to blend with yet.
const seed = gpu.createKernel(function (frame) {
  const p = frame[this.thread.y][this.thread.x];
  return 0.299 * p[0] + 0.587 * p[1] + 0.114 * p[2];
}, { output: [64, 64], pipeline: true });

// One frame folded into the running average.
const blend = gpu.createKernel(function (frame, previous) {
  const p = frame[this.thread.y][this.thread.x];
  const now = 0.299 * p[0] + 0.587 * p[1] + 0.114 * p[2];
  // TODO: return (1 - alpha) of the previous average plus alpha of \`now\`.
  return previous[this.thread.y][this.thread.x];
}, {
  output: [64, 64],
  pipeline: true,
  constants: { alpha: 0.25 },
  // TODO: this kernel reads the very texture it is writing. Run it and let
  // the error message tell you the missing setting.
});

let state = await seed(frames[0]);
for (let i = 1; i < frames.length; i++) {
  state = await blend(frames[i], state); // last frame's output, straight back in
}

const smoothed = state.toArray ? await state.toArray() : state;
console.log('smoothed center:', smoothed[32][32].toFixed(4));
`,
      solutionCode: `// Denoising along the time axis: average each pixel with itself.
const gpu = new GPU({ mode });

// The starting average is simply frame 0's luminance — nothing to blend with yet.
const seed = gpu.createKernel(function (frame) {
  const p = frame[this.thread.y][this.thread.x];
  return 0.299 * p[0] + 0.587 * p[1] + 0.114 * p[2];
}, { output: [64, 64], pipeline: true });

// One frame folded into the running average.
const blend = gpu.createKernel(function (frame, previous) {
  const p = frame[this.thread.y][this.thread.x];
  const now = 0.299 * p[0] + 0.587 * p[1] + 0.114 * p[2];
  return (1 - this.constants.alpha) * previous[this.thread.y][this.thread.x]
       + this.constants.alpha * now;
}, {
  output: [64, 64],
  pipeline: true,
  immutable: true, // a fresh texture per call — the feedback loop is now legal
  constants: { alpha: 0.25 },
});

let state = await seed(frames[0]);
for (let i = 1; i < frames.length; i++) {
  state = await blend(frames[i], state); // last frame's output, straight back in
}

const smoothed = state.toArray ? await state.toArray() : state;
console.log('smoothed center:', smoothed[32][32].toFixed(4));
`,
      inputs: utils => ({ frames: makeFrames(utils) }),
      inputNotes: {
        frames: 'Eight 64×64 ImageData frames in time order — a bright object crossing a static scene, with sensor noise. Pass one frame at a time into a kernel; inside it frames[i][y][x] is that pixel as [r, g, b, a] from 0 to 1.',
      },
      publicTests: [
        {
          name: 'the feedback kernel renders to a fresh texture each call',
          run: async ctx => {
            ctx.assert(ctx.kernels.length >= 2, `expected 2 kernels, found ${ctx.kernels.length}`);
            const blend = ctx.kernels.find(k => k.kernel && k.kernel.immutable);
            ctx.assert(
              blend,
              'no immutable kernel found — a kernel that reads its own last output needs ' +
                'immutable: true, or this frame would be reading a buffer it is halfway through writing'
            );
            ctx.assert(blend.kernel.pipeline === true, 'blend should keep pipeline: true as well');
          },
        },
        {
          name: 'the running average matches a reference over all eight frames',
          run: async ctx => {
            const blend = ctx.kernels.find(k => k.kernel && k.kernel.immutable);
            const seed = ctx.kernels.find(k => k.kernel && k.kernel.pipeline && !k.kernel.immutable);
            ctx.assert(seed && blend, 'expected a seed kernel and an immutable blend kernel');
            const seq = makeFrames(ctx.utils);
            let state = await seed(seq[0]);
            for (let i = 1; i < seq.length; i++) state = await blend(seq[i], state);
            const got = await toArr(state);
            const ref = refTemporal(seq, ALPHA_SMOOTH);
            const hint = diagnoseGrid(got, ref, 3e-3, temporalProbes(seq, ALPHA_SMOOTH));
            const miss = firstMismatch(got, ref, 3e-3);
            ctx.assert(
              !miss,
              hint ||
                (miss && `cell [${miss.y}][${miss.x}] — expected ${miss.want.toFixed(4)}, got ${miss.got}`)
            );
          },
        },
        {
          name: 'averaging over time actually removes noise',
          run: async ctx => {
            // A flat scene plus noise: the running average has to sit closer to
            // the truth than any single frame does.
            const blend = ctx.kernels.find(k => k.kernel && k.kernel.immutable);
            const seed = ctx.kernels.find(k => k.kernel && k.kernel.pipeline && !k.kernel.immutable);
            ctx.assert(seed && blend, 'expected a seed kernel and an immutable blend kernel');
            const seq = makeFrames(ctx.utils);
            let state = await seed(seq[0]);
            for (let i = 1; i < seq.length; i++) state = await blend(seq[i], state);
            const got = await toArr(state);
            const ref = refTemporal(seq, ALPHA_SMOOTH);
            // Compare only the region the object never reaches, where the true
            // luminance is constant in time: the desk band along the bottom.
            let smoothed = 0;
            let single = 0;
            const truth = grid((y, x) => lumOf(scenePixel(x, y)));
            const lastFrame = lumMap(seq[seq.length - 1]);
            for (let y = 50; y < 64; y++) {
              for (let x = 0; x < 64; x++) {
                smoothed += Math.abs(ref[y][x] - truth[y][x]);
                single += Math.abs(lastFrame[y][x] - truth[y][x]);
              }
            }
            ctx.assert(
              smoothed < single,
              'the reference average is not quieter than a single frame — this test is broken, not you'
            );
            let yours = 0;
            for (let y = 50; y < 64; y++) {
              for (let x = 0; x < 64; x++) yours += Math.abs(got[y][x] - truth[y][x]);
            }
            ctx.assert(
              yours < single,
              `the averaged result is no closer to the noise-free scene than one raw frame is ` +
                `(${yours.toFixed(2)} vs ${single.toFixed(2)}) — the previous state is not contributing`
            );
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            const blend = ctx.kernels.find(k => k.kernel && k.kernel.immutable);
            const seed = ctx.kernels.find(k => k.kernel && k.kernel.pipeline && !k.kernel.immutable);
            ctx.assert(seed && blend, 'expected a seed kernel and an immutable blend kernel');
            const seq = makeFrames(ctx.utils, 606060);
            let state = await seed(seq[0]);
            for (let i = 1; i < seq.length; i++) state = await blend(seq[i], state);
            const got = await toArr(state);
            const ref = refTemporal(seq, ALPHA_SMOOTH);
            const hint = diagnoseGrid(got, ref, 3e-3, temporalProbes(seq, ALPHA_SMOOTH));
            const miss = firstMismatch(got, ref, 3e-3);
            ctx.assert(
              !miss,
              hint ||
                (miss && `cell [${miss.y}][${miss.x}] — expected ${miss.want.toFixed(4)}, got ${miss.got}`)
            );
          },
        },
        {
          name: 'private test #2',
          run: async ctx => {
            // Two frames only. After one blend the answer is alpha of the way
            // from frame 0 to frame 1 — which pins the weights down exactly.
            const blend = ctx.kernels.find(k => k.kernel && k.kernel.immutable);
            const seed = ctx.kernels.find(k => k.kernel && k.kernel.pipeline && !k.kernel.immutable);
            ctx.assert(seed && blend, 'expected a seed kernel and an immutable blend kernel');
            const seq = makeFrames(ctx.utils, 4242);
            const got = await toArr(await blend(seq[1], await seed(seq[0])));
            const a = lumMap(seq[0]);
            const b = lumMap(seq[1]);
            const ref = grid((y, x) => (1 - ALPHA_SMOOTH) * a[y][x] + ALPHA_SMOOTH * b[y][x]);
            const hint = diagnoseGrid(got, ref, 3e-3, [
              [grid((y, x) => ALPHA_SMOOTH * a[y][x] + (1 - ALPHA_SMOOTH) * b[y][x]),
                'the two weights are swapped — alpha belongs on the new frame, 1 − alpha on the average you already had'],
              [b, 'the previous average contributed nothing — that is just this frame\'s luminance'],
              [a, 'the new frame contributed nothing — that is the seed, unchanged'],
              [grid((y, x) => a[y][x] + b[y][x]),
                'the two terms were added without their weights — they have to sum to 1, or the picture gets brighter every frame'],
            ]);
            const miss = firstMismatch(got, ref, 3e-3);
            ctx.assert(
              !miss,
              hint ||
                (miss && `cell [${miss.y}][${miss.x}] — expected ${miss.want.toFixed(4)}, got ${miss.got}`)
            );
          },
        },
      ],
    },

    // ------------------------------------------------------------------ 3
    {
      slug: 'motion-mask',
      title: 'What Moved?',
      intro: `<p>Subtract one frame from the one before it. Everything that stayed put cancels to
        roughly zero; everything that moved does not. Threshold what is left and you have a
        <strong>motion mask</strong> — one bit per pixel, "something happened here" — and that is
        the first step of essentially every "is anything moving?" system ever shipped, from a
        doorbell camera to a video codec deciding which blocks to re-encode.</p>
        <p>Two things make or break it. The first is the absolute value: a pixel that got
        <em>darker</em> moved exactly as much as one that got brighter, and dropping
        <code>Math.abs</code> silently throws away half of every edge — the mask still looks
        plausible, which is what makes it nasty. The second is noise. A raw thresholded difference
        is speckled with isolated pixels that the sensor invented, so the mask gets a cleanup pass:
        a 3×3 <strong>majority vote</strong>, in the spirit of the morphological open you met in
        Thresholding &amp; Morphology. A lone hot pixel has one vote out of nine and loses. The
        inside of something that really moved has nine and does not.</p>
        <p>And frame 0 has no predecessor. Eight frames give you <strong>seven</strong> differences,
        not eight — the classic off-by-one at the start of a sequence, and the reason so many
        filters flash garbage on their very first frame.</p>
        ${ARRAY_LAYOUT}`,
      goal: `<strong>Goal:</strong> finish the <code>motion</code> kernel — the absolute luminance
        difference between two frames, thresholded to 1 or 0 — and hand the right pair of frames to
        it.`,
      requirements: [
        'In <code>motion</code>, take the luminance of both frames and the <strong>absolute</strong> difference',
        'Return <code>1</code> when that difference exceeds <code>this.constants.threshold</code>, otherwise <code>0</code>',
        'Compare each frame with the one <em>before</em> it — seven differences from eight frames',
      ],
      hints: [
        {
          title: 'Hint 1 — the difference, both ways',
          body: `<p>Motion is a change in either direction:</p>
<pre><code>const change = Math.abs(now - before);
if (change &gt; this.constants.threshold) {
  return 1;
}
return 0;</code></pre>
<p>Drop the <code>Math.abs</code> and the trailing edge of every moving object disappears.</p>`,
        },
        {
          title: 'Hint 2 — which pair of frames',
          body: `<p><code>previous</code> has to be the frame <em>before</em> this one:
            <code>frames[i - 1]</code>. Hand the kernel <code>frames[i]</code> twice and the
            difference is zero everywhere — a perfectly quiet, perfectly useless mask.</p>`,
        },
        {
          title: 'Hint 3 — where the loop starts',
          body: `<p><code>frames[i - 1]</code> only exists from <code>i = 1</code> onwards, so the
            loop starts there. Eight frames, seven differences — the log line prints the count so you
            can see it.</p>`,
        },
      ],
      transfer: `Frame differencing is the oldest trick in video and still the load-bearing one.
        Motion estimation in H.264/AV1 starts from exactly this residual; OpenCV ships
        <code>absdiff</code> plus a threshold as the canonical first example; and every "smart"
        security camera on the market is this kernel plus a blob counter. On any GPU it is a
        one-instruction-per-pixel pass whose real cost is getting the two frames resident at once.`,
      starterCode: `// Two frames in, one bit per pixel out.
const gpu = new GPU({ mode });

const motion = gpu.createKernel(function (current, previous) {
  const a = current[this.thread.y][this.thread.x];
  const b = previous[this.thread.y][this.thread.x];
  const now = 0.299 * a[0] + 0.587 * a[1] + 0.114 * a[2];
  const before = 0.299 * b[0] + 0.587 * b[1] + 0.114 * b[2];
  // TODO: how far did this pixel move, in EITHER direction? Return 1 when
  // that exceeds this.constants.threshold, and 0 when it does not.
  return 0;
}, {
  output: [64, 64],
  pipeline: true,
  constants: { threshold: 0.12 },
});

// The cleanup pass: a 3×3 majority vote. Five of nine neighbours have to
// agree before a pixel stays lit, so isolated sensor noise loses and the
// inside of a real moving blob does not. (Given — you wrote this shape in
// Thresholding & Morphology.)
const cleanup = gpu.createKernel(function (mask) {
  let votes = 0;
  for (let dy = -1; dy <= 1; dy++) {
    for (let dx = -1; dx <= 1; dx++) {
      let yy = this.thread.y + dy;
      let xx = this.thread.x + dx;
      if (yy < 0) yy = 0;
      if (yy > this.constants.last) yy = this.constants.last;
      if (xx < 0) xx = 0;
      if (xx > this.constants.last) xx = this.constants.last;
      votes += mask[yy][xx];
    }
  }
  if (votes >= 5) {
    return 1;
  }
  return 0;
}, { output: [64, 64], constants: { last: 63 } });

const masks = [];
// TODO: frame 0 has no predecessor. Where does this loop really start,
// and which frame belongs in \`previous\`?
for (let i = 0; i < frames.length; i++) {
  const previous = frames[i];
  masks.push(await cleanup(await motion(frames[i], previous)));
}

console.log('motion masks:', masks.length);

const last = masks[masks.length - 1];
let moving = 0;
for (let y = 0; y < 64; y++) {
  for (let x = 0; x < 64; x++) moving += last[y][x];
}
console.log('moving pixels in the last mask:', moving);
`,
      solutionCode: `// Two frames in, one bit per pixel out.
const gpu = new GPU({ mode });

const motion = gpu.createKernel(function (current, previous) {
  const a = current[this.thread.y][this.thread.x];
  const b = previous[this.thread.y][this.thread.x];
  const now = 0.299 * a[0] + 0.587 * a[1] + 0.114 * a[2];
  const before = 0.299 * b[0] + 0.587 * b[1] + 0.114 * b[2];
  const change = Math.abs(now - before); // darker counts as much as brighter
  if (change > this.constants.threshold) {
    return 1;
  }
  return 0;
}, {
  output: [64, 64],
  pipeline: true,
  constants: { threshold: 0.12 },
});

// The cleanup pass: a 3×3 majority vote. Five of nine neighbours have to
// agree before a pixel stays lit, so isolated sensor noise loses and the
// inside of a real moving blob does not. (Given — you wrote this shape in
// Thresholding & Morphology.)
const cleanup = gpu.createKernel(function (mask) {
  let votes = 0;
  for (let dy = -1; dy <= 1; dy++) {
    for (let dx = -1; dx <= 1; dx++) {
      let yy = this.thread.y + dy;
      let xx = this.thread.x + dx;
      if (yy < 0) yy = 0;
      if (yy > this.constants.last) yy = this.constants.last;
      if (xx < 0) xx = 0;
      if (xx > this.constants.last) xx = this.constants.last;
      votes += mask[yy][xx];
    }
  }
  if (votes >= 5) {
    return 1;
  }
  return 0;
}, { output: [64, 64], constants: { last: 63 } });

const masks = [];
// Eight frames, seven differences: frame 0 has nothing to be compared with.
for (let i = 1; i < frames.length; i++) {
  masks.push(await cleanup(await motion(frames[i], frames[i - 1])));
}

console.log('motion masks:', masks.length);

const last = masks[masks.length - 1];
let moving = 0;
for (let y = 0; y < 64; y++) {
  for (let x = 0; x < 64; x++) moving += last[y][x];
}
console.log('moving pixels in the last mask:', moving);
`,
      inputs: utils => ({ frames: makeFrames(utils) }),
      inputNotes: {
        frames: 'Eight 64×64 ImageData frames in time order — a bright object crossing a static scene, with sensor noise. Pass one frame at a time into a kernel; inside it frames[i][y][x] is that pixel as [r, g, b, a] from 0 to 1.',
      },
      publicTests: [
        {
          name: 'eight frames produce seven differences',
          run: async ctx => {
            ctx.assert(ctx.kernels.length >= 2, `expected 2 kernels, found ${ctx.kernels.length}`);
            ctx.assert(
              logged(ctx, 'motion masks: 7'),
              logged(ctx, 'motion masks: 8')
                ? 'eight masks came out of eight frames — frame 0 has no predecessor, so the loop starts at 1 and produces seven differences'
                : 'expected the run to report "motion masks: 7" — one difference for each consecutive pair'
            );
          },
        },
        {
          name: 'the raw mask is binary and marks change in both directions',
          run: async ctx => {
            const motion = ctx.kernels.find(k => k.kernel && k.kernel.pipeline);
            ctx.assert(motion, 'no pipeline kernel found — keep pipeline: true on the motion kernel');
            const seq = makeFrames(ctx.utils);
            const got = await toArr(await motion(seq[4], seq[3]));
            const delta = refDelta(seq[3], seq[4]);
            const ref = refThreshold(delta, THRESHOLD);
            const skip = fenceOf(delta, THRESHOLD);
            const hint = diagnoseGrid(got, ref, 0.25, motionProbes(seq[3], seq[4], THRESHOLD), skip);
            const miss = firstMismatch(got, ref, 0.25, skip);
            ctx.assert(
              !miss,
              hint ||
                (miss && `cell [${miss.y}][${miss.x}] should be ${miss.want}, got ${miss.got} ` +
                  `(that pixel's luminance moved by ${delta[miss.y][miss.x].toFixed(3)}, and the threshold is ${THRESHOLD})`)
            );
          },
        },
        {
          name: 'the cleanup pass survives the noise and keeps the blob',
          run: async ctx => {
            const motion = ctx.kernels.find(k => k.kernel && k.kernel.pipeline);
            const cleanup = ctx.kernels.find(k => k.kernel && !k.kernel.pipeline);
            ctx.assert(motion && cleanup, 'expected a pipeline motion kernel and a plain cleanup kernel');
            const seq = makeFrames(ctx.utils);
            const got = await cleanup(await motion(seq[7], seq[6]));
            const delta = refDelta(seq[6], seq[7]);
            const raw = refThreshold(delta, THRESHOLD);
            const ref = refMajority(raw);
            const skip = dilateFence(fenceOf(delta, THRESHOLD));
            const miss = firstMismatch(got, ref, 0.25, skip);
            ctx.assert(
              !miss,
              miss && `cleaned cell [${miss.y}][${miss.x}] should be ${miss.want}, got ${miss.got}`
            );
            // And the whole point: cleanup should have thrown away specks
            // without throwing away the moving object.
            ctx.assert(
              sumGrid(ref) > 150,
              'the reference mask is empty — this test is broken, not you'
            );
          },
        },
        {
          name: 'the run\'s own last mask is not empty',
          run: async ctx => {
            // The kernel tests above hand the kernel the right pair of frames
            // themselves, so this is the only place the DRIVER's choice of
            // frames shows up.
            const line = ctx.logs
              .map(l => (l.type === 'log' && l.text ? /moving pixels in the last mask:\s*(\d+)/.exec(l.text) : null))
              .find(Boolean);
            ctx.assert(line, 'expected the run to log "moving pixels in the last mask: N"');
            const moving = Number(line[1]);
            ctx.assert(
              moving > 100,
              moving === 0
                ? 'the last mask is completely empty — the kernel is being handed the same frame ' +
                  'twice, and a frame differs from itself nowhere. `previous` has to be frames[i - 1].'
                : `the last mask holds only ${moving} moving pixels; the object crosses several ` +
                  `hundred between two consecutive frames`
            );
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            const motion = ctx.kernels.find(k => k.kernel && k.kernel.pipeline);
            const cleanup = ctx.kernels.find(k => k.kernel && !k.kernel.pipeline);
            ctx.assert(motion && cleanup, 'expected a pipeline motion kernel and a plain cleanup kernel');
            const seq = makeFrames(ctx.utils, 24680);
            for (let i = 1; i < seq.length; i++) {
              const delta = refDelta(seq[i - 1], seq[i]);
              const raw = refThreshold(delta, THRESHOLD);
              const skip = fenceOf(delta, THRESHOLD);
              const got = await toArr(await motion(seq[i], seq[i - 1]));
              const hint = diagnoseGrid(got, raw, 0.25, motionProbes(seq[i - 1], seq[i], THRESHOLD), skip);
              const miss = firstMismatch(got, raw, 0.25, skip);
              ctx.assert(
                !miss,
                hint || (miss && `pair ${i - 1}→${i}, cell [${miss.y}][${miss.x}] should be ${miss.want}, got ${miss.got}`)
              );
              const cleaned = await cleanup(await motion(seq[i], seq[i - 1]));
              const cleanMiss = firstMismatch(cleaned, refMajority(raw), 0.25, dilateFence(skip));
              ctx.assert(
                !cleanMiss,
                cleanMiss && `pair ${i - 1}→${i}, cleaned cell [${cleanMiss.y}][${cleanMiss.x}] should be ${cleanMiss.want}, got ${cleanMiss.got}`
              );
            }
          },
        },
        {
          name: 'private test #2',
          run: async ctx => {
            // A frame against itself must be silent, and a frame against one
            // where the object has moved must not be.
            const motion = ctx.kernels.find(k => k.kernel && k.kernel.pipeline);
            const seq = makeFrames(ctx.utils, 13579);
            const still = sumGrid(await toArr(await motion(seq[5], seq[5])));
            ctx.assert(
              still < 40,
              `${still} pixels registered as moving between a frame and itself — a difference of ` +
                `something with itself is zero, so only the threshold comparison can be producing these`
            );
            const moved = sumGrid(await toArr(await motion(seq[5], seq[4])));
            ctx.assert(
              moved > 200,
              `only ${moved} pixels registered between two consecutive frames — the object crosses ` +
                `hundreds of pixels between them, so something is swallowing the difference`
            );
          },
        },
      ],
    },

    // ------------------------------------------------------------------ 4
    {
      slug: 'background-model',
      title: 'Learning the Empty Room',
      intro: `<p>Frame differencing has a blind spot you can see in its own output: it only ever
        finds the <em>edges</em> of a moving object. The middle of a large uniform blob looks
        identical from one frame to the next, so it reports as still. And an object that stops
        moving vanishes entirely.</p>
        <p>The fix is to stop comparing against the last frame and start comparing against a
        <strong>model of the empty scene</strong> — an estimate of what each pixel looks like when
        nothing is happening there. Keep that model as a running average with a <em>very</em> small
        <code>alpha</code>, the same one-liner as the last task with the dial turned right down:</p>
<pre><code>model = (1 - alpha) * model + alpha * now</code></pre>
        <p>At <code>alpha = 0.05</code> the model needs about twenty frames to accept a change, so
        an object crossing the frame in eight never gets absorbed — but the sun going behind a
        cloud eventually does. That is the entire trade, and it is worth saying out loud:
        <strong>too fast</strong> and a person who stops moving is quietly re-labelled as furniture;
        <strong>too slow</strong> and every genuine change — a chair moved, a light switched on —
        leaves a ghost burning in the mask for a minute. Nobody has a principled way to pick it.
        People measure.</p>
        <p>Then foreground is whatever the current frame disagrees with the model about — the same
        absolute difference and threshold as before, against a different reference. Watch what comes
        out: a solid object, not a pair of crescents.</p>
        ${ARRAY_LAYOUT}`,
      goal: `<strong>Goal:</strong> write the exponential update in <code>learn</code> and the
        subtraction in <code>foreground</code>, then run the model over the sequence.`,
      requirements: [
        'In <code>learn</code>, return <code>(1 - alpha) * model + alpha * now</code> — <code>alpha</code> on the new frame',
        'In <code>foreground</code>, return <code>1</code> when <code>|now - model|</code> exceeds the threshold, else <code>0</code>',
        'Segment each frame against the model as it stands, <em>then</em> fold that frame in',
      ],
      hints: [
        {
          title: 'Hint 1 — which term wears the alpha',
          body: `<p>The model is mostly memory and only slightly news, so the big weight sits on the
            model:</p>
<pre><code>return (1 - this.constants.alpha) * model[this.thread.y][this.thread.x]
     + this.constants.alpha * now;</code></pre>
<p>Put <code>alpha</code> on the wrong term and the model becomes the current frame in about one
            frame flat — after which nothing is ever foreground again.</p>`,
        },
        {
          title: 'Hint 2 — the subtraction',
          body: `<p>Identical in shape to the frame difference from the last task, only the
            reference changed:</p>
<pre><code>if (Math.abs(now - model[this.thread.y][this.thread.x]) &gt; this.constants.threshold) {
  return 1;
}
return 0;</code></pre>`,
        },
        {
          title: 'Hint 3 — order inside the loop',
          body: `<p>Detect first, learn second. If you fold the frame in before you compare against
            it, the model has already moved a little way toward the object you are trying to find —
            you are grading your own homework.</p>`,
        },
      ],
      transfer: `Every serious background subtractor is this line with more machinery on top: OpenCV's
        <code>MOG2</code> keeps a mixture of Gaussians per pixel instead of one mean,
        <code>KNN</code> keeps a sample history, and both still expose a learning rate that behaves
        exactly like this <code>alpha</code>. On a GPU the appeal never changes — one number of state
        per pixel, one multiply-add per frame, perfectly parallel, and no history buffer to carry.`,
      starterCode: `// A model of the empty room, updated a little at a time.
const gpu = new GPU({ mode });

// Seed the model with the first frame you are given. (Frame 0 of this
// sequence is a clean plate — the object is still off the left edge.)
const seedModel = gpu.createKernel(function (frame) {
  const p = frame[this.thread.y][this.thread.x];
  return 0.299 * p[0] + 0.587 * p[1] + 0.114 * p[2];
}, { output: [64, 64], pipeline: true });

// Fold one frame into the model.
const learn = gpu.createKernel(function (frame, model) {
  const p = frame[this.thread.y][this.thread.x];
  const now = 0.299 * p[0] + 0.587 * p[1] + 0.114 * p[2];
  // TODO: keep (1 - alpha) of the model and take alpha of \`now\`.
  return model[this.thread.y][this.thread.x];
}, {
  output: [64, 64],
  pipeline: true,
  immutable: true,
  constants: { alpha: 0.05 },
});

// Anything this frame disagrees with the model about is foreground.
const foreground = gpu.createKernel(function (frame, model) {
  const p = frame[this.thread.y][this.thread.x];
  const now = 0.299 * p[0] + 0.587 * p[1] + 0.114 * p[2];
  // TODO: 1 where |now - model| exceeds this.constants.threshold, else 0.
  return 0;
}, {
  output: [64, 64],
  constants: { threshold: 0.12 },
});

let model = await seedModel(frames[0]);
let mask = null;
for (let i = 1; i < frames.length; i++) {
  mask = await foreground(frames[i], model); // detect against the model as it stands
  model = await learn(frames[i], model);     // then let it learn this frame
}

let count = 0;
for (let y = 0; y < 64; y++) {
  for (let x = 0; x < 64; x++) count += mask[y][x];
}
console.log('foreground pixels in the last frame:', count);
`,
      solutionCode: `// A model of the empty room, updated a little at a time.
const gpu = new GPU({ mode });

// Seed the model with the first frame you are given. (Frame 0 of this
// sequence is a clean plate — the object is still off the left edge.)
const seedModel = gpu.createKernel(function (frame) {
  const p = frame[this.thread.y][this.thread.x];
  return 0.299 * p[0] + 0.587 * p[1] + 0.114 * p[2];
}, { output: [64, 64], pipeline: true });

// Fold one frame into the model.
const learn = gpu.createKernel(function (frame, model) {
  const p = frame[this.thread.y][this.thread.x];
  const now = 0.299 * p[0] + 0.587 * p[1] + 0.114 * p[2];
  return (1 - this.constants.alpha) * model[this.thread.y][this.thread.x]
       + this.constants.alpha * now;
}, {
  output: [64, 64],
  pipeline: true,
  immutable: true,
  constants: { alpha: 0.05 },
});

// Anything this frame disagrees with the model about is foreground.
const foreground = gpu.createKernel(function (frame, model) {
  const p = frame[this.thread.y][this.thread.x];
  const now = 0.299 * p[0] + 0.587 * p[1] + 0.114 * p[2];
  if (Math.abs(now - model[this.thread.y][this.thread.x]) > this.constants.threshold) {
    return 1;
  }
  return 0;
}, {
  output: [64, 64],
  constants: { threshold: 0.12 },
});

let model = await seedModel(frames[0]);
let mask = null;
for (let i = 1; i < frames.length; i++) {
  mask = await foreground(frames[i], model); // detect against the model as it stands
  model = await learn(frames[i], model);     // then let it learn this frame
}

let count = 0;
for (let y = 0; y < 64; y++) {
  for (let x = 0; x < 64; x++) count += mask[y][x];
}
console.log('foreground pixels in the last frame:', count);
`,
      inputs: utils => ({ frames: makeFrames(utils) }),
      inputNotes: {
        frames: 'Eight 64×64 ImageData frames in time order — a bright object crossing a static scene, with sensor noise. Pass one frame at a time into a kernel; inside it frames[i][y][x] is that pixel as [r, g, b, a] from 0 to 1.',
      },
      publicTests: [
        {
          name: 'three kernels: a seed, an immutable update, a plain detector',
          run: async ctx => {
            ctx.assert(ctx.kernels.length >= 3, `expected 3 kernels, found ${ctx.kernels.length}`);
            const learn = ctx.kernels.find(k => k.kernel && k.kernel.immutable);
            const seed = ctx.kernels.find(k => k.kernel && k.kernel.pipeline && !k.kernel.immutable);
            const detect = ctx.kernels.find(k => k.kernel && !k.kernel.pipeline);
            ctx.assert(learn, 'the model update reads its own last output — it needs immutable: true');
            ctx.assert(seed, 'no seed kernel found — the model has to start from something');
            ctx.assert(detect, 'no plain kernel found — the foreground mask is the one result you want in JavaScript');
          },
        },
        {
          name: 'the model learns the scene slowly, not the object quickly',
          run: async ctx => {
            const learn = ctx.kernels.find(k => k.kernel && k.kernel.immutable);
            const seed = ctx.kernels.find(k => k.kernel && k.kernel.pipeline && !k.kernel.immutable);
            const seq = makeFrames(ctx.utils);
            let model = await seed(seq[0]);
            for (let i = 1; i < seq.length; i++) model = await learn(seq[i], model);
            const got = await toArr(model);
            const ref = refModel(seq, ALPHA_MODEL, seq.length - 1);
            const hint = diagnoseGrid(got, ref, 3e-3, modelProbes(seq, ALPHA_MODEL, seq.length - 1));
            const miss = firstMismatch(got, ref, 3e-3);
            ctx.assert(
              !miss,
              hint ||
                (miss && `model cell [${miss.y}][${miss.x}] — expected ${miss.want.toFixed(4)}, got ${miss.got}`)
            );
          },
        },
        {
          name: 'the foreground mask is the whole object, not just its edges',
          run: async ctx => {
            const learn = ctx.kernels.find(k => k.kernel && k.kernel.immutable);
            const seed = ctx.kernels.find(k => k.kernel && k.kernel.pipeline && !k.kernel.immutable);
            const detect = ctx.kernels.find(k => k.kernel && !k.kernel.pipeline);
            const seq = makeFrames(ctx.utils);
            const run = refSegmentRun(seq, ALPHA_MODEL, THRESHOLD);
            let model = await seed(seq[0]);
            let got = null;
            for (let i = 1; i < seq.length; i++) {
              got = await detect(seq[i], model);
              model = await learn(seq[i], model);
            }
            const last = seq.length - 1;
            const now = lumMap(seq[last]);
            const reference = run.models[run.models.length - 1];
            const delta = grid((y, x) => Math.abs(now[y][x] - reference[y][x]));
            const ref = run.masks[run.masks.length - 1];
            const skip = fenceOf(delta, THRESHOLD);
            const hint = diagnoseGrid(got, ref, 0.25, [
              [refThreshold(grid((y, x) => now[y][x] - reference[y][x]), THRESHOLD),
                'only pixels BRIGHTER than the model are marked — Math.abs is missing, so anything darker than the scene never registers'],
              [grid((y, x) => (delta[y][x] > THRESHOLD ? 0 : 1)),
                'the mask is inverted — it is marking everything that agrees with the model'],
              [delta,
                'that is the raw difference from the model, not a mask — compare it against the threshold and return 1 or 0'],
            ], skip);
            const miss = firstMismatch(got, ref, 0.25, skip);
            ctx.assert(
              !miss,
              hint || (miss && `mask cell [${miss.y}][${miss.x}] should be ${miss.want}, got ${miss.got}`)
            );
            ctx.assert(
              logged(ctx, 'foreground pixels'),
              'log how many foreground pixels the last frame produced'
            );
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            const learn = ctx.kernels.find(k => k.kernel && k.kernel.immutable);
            const seed = ctx.kernels.find(k => k.kernel && k.kernel.pipeline && !k.kernel.immutable);
            const detect = ctx.kernels.find(k => k.kernel && !k.kernel.pipeline);
            ctx.assert(seed && learn && detect, 'expected a seed, an immutable update and a plain detector');
            const seq = makeFrames(ctx.utils, 8675309);
            const run = refSegmentRun(seq, ALPHA_MODEL, THRESHOLD);
            let model = await seed(seq[0]);
            for (let i = 1; i < seq.length; i++) {
              const now = lumMap(seq[i]);
              const reference = run.models[i - 1];
              const delta = grid((y, x) => Math.abs(now[y][x] - reference[y][x]));
              const skip = fenceOf(delta, THRESHOLD);
              const got = await detect(seq[i], model);
              const miss = firstMismatch(got, run.masks[i - 1], 0.25, skip);
              ctx.assert(
                !miss,
                miss && `frame ${i}, mask cell [${miss.y}][${miss.x}] should be ${miss.want}, got ${miss.got} — ` +
                  `check that the frame is detected BEFORE it is folded into the model`
              );
              model = await learn(seq[i], model);
              const modelMiss = firstMismatch(await toArr(model), refModel(seq, ALPHA_MODEL, i), 3e-3);
              ctx.assert(
                !modelMiss,
                modelMiss && `after frame ${i}, model cell [${modelMiss.y}][${modelMiss.x}] — ` +
                  `expected ${modelMiss.want.toFixed(4)}, got ${modelMiss.got}`
              );
            }
          },
        },
        {
          name: 'private test #2',
          run: async ctx => {
            // The claim the whole task rests on: against a background model the
            // object comes out solid, where frame differencing finds only its
            // edges. Both are computed from the learner's own kernels.
            const learn = ctx.kernels.find(k => k.kernel && k.kernel.immutable);
            const seed = ctx.kernels.find(k => k.kernel && k.kernel.pipeline && !k.kernel.immutable);
            const detect = ctx.kernels.find(k => k.kernel && !k.kernel.pipeline);
            const seq = makeFrames(ctx.utils, 112358);
            let model = await seed(seq[0]);
            let mask = null;
            for (let i = 1; i < seq.length; i++) {
              mask = await detect(seq[i], model);
              model = await learn(seq[i], model);
            }
            const solid = sumGrid(mask);
            const disc = Math.PI * RADIUS * RADIUS;
            ctx.assert(
              solid > 0.7 * disc,
              `the last frame's foreground covers only ${solid} pixels; the object alone is about ` +
                `${Math.round(disc)}. A background model should find the whole object, not its outline — ` +
                `check that alpha sits on the new frame and not on the model`
            );
          },
        },
      ],
    },

    // ------------------------------------------------------------------ 5
    {
      slug: 'virtual-background',
      title: 'The Payoff: A Virtual Background',
      intro: `<p>Time to cash the whole track in. You have a model of the empty scene, and you have
        the arithmetic to say how much each pixel disagrees with it. Turn that disagreement into a
        <strong>soft mask</strong>, blur a copy of the frame to stand in for a replaced backdrop,
        and composite one over the other. That is the effect everybody has seen on a video call, and
        it is four kernels.</p>
        <p>Two details make it look like a product rather than a demo. The first is that the mask
        must be <strong>normalised</strong> — a number from 0 to 1, nothing else. Composite with
        <code>fg × m + bg × (1 − m)</code> and a mask of 1.4 does not mean "very foreground", it
        means the background is subtracted from the picture; a mask of −0.3 means the background is
        added twice. So the raw difference gets ramped and clamped:</p>
<pre><code>m = clamp((d - lo) / (hi - lo), 0, 1)</code></pre>
        <p>The second is the <strong>feather</strong>: the ramp alone gives a hard, jagged edge, and
        a 3×3 mean over the mask softens it — the same box blur from Convolution &amp; Filters,
        aimed at the mask instead of the picture. It has a second job: an isolated hot pixel that
        made it through arrives as a lone 1 and leaves as a 0.11, which is invisible.</p>
        <p>Nothing in the chain touches JavaScript. The frame goes up, the model lives on the card,
        the mask never comes down, and the graphical pass eats the mask texture and writes pixels.
        Readbacks: zero.</p>
        ${ARRAY_LAYOUT}`,
      goal: `<strong>Goal:</strong> finish the normalised ramp in <code>softMask</code> and the
        composite line in <code>compose</code>, so the foreground stays sharp over a blurred
        background.`,
      requirements: [
        'In <code>softMask</code>, ramp <code>|now − model|</code> from <code>lo</code> to <code>hi</code> and clamp the result into 0…1',
        'In <code>compose</code>, paint <code>source × m + blurred × (1 − m)</code> per channel',
        'Every stage stays on the GPU — the graphical pass is fed the mask <em>texture</em>',
        'Render the result with <code>render(compose.canvas)</code>',
      ],
      hints: [
        {
          title: 'Hint 1 — the ramp, normalised',
          body: `<p>Below <code>lo</code> it is all background, above <code>hi</code> all
            foreground, and in between it slides:</p>
<pre><code>const span = this.constants.hi - this.constants.lo;
return Math.min(Math.max((d - this.constants.lo) / span, 0), 1);</code></pre>
<p>The <code>Math.min</code>/<code>Math.max</code> pair is not decoration — without it the mask
            leaves 0…1 and the composite starts subtracting light.</p>`,
        },
        {
          title: 'Hint 2 — the composite',
          body: `<p><code>m = 1</code> has to give you the source pixel and <code>m = 0</code> the
            blurred one, so the mask multiplies the <em>foreground</em>:</p>
<pre><code>this.color(
  p[0] * m + backR * (1 - m),
  p[1] * m + backG * (1 - m),
  p[2] * m + backB * (1 - m),
  1
);</code></pre>
<p>Swap the two and you get a sharp background with a blurry person in it, which is a look, just
            not this one.</p>`,
        },
        {
          title: 'Hint 3 — where the background comes from',
          body: `<p>The 5×5 loop in <code>compose</code> is already written: it averages the frame's
            own neighbourhood, so the "replaced" backdrop is a blurred copy of the real one. Swap
            that average for a fixed colour, or for a second image, and you have a green screen
            instead.</p>`,
        },
      ],
      transfer: `This is a render graph: named passes, explicit dependencies, every resource resident
        on the device — the architecture behind a Frostbite frame graph, a Metal command buffer full
        of encoder passes, or CUDA Graphs' pre-recorded launch chains. Shipping virtual backgrounds
        replace the luminance model with a segmentation network, but the tail of the pipeline —
        ramp, feather, composite — is still exactly these three lines, because it is the part that
        has to run in under a millisecond.`,
      starterCode: `const gpu = new GPU({ mode });

// Passes 1 and 2 — the background model, from the last task.
const seedModel = gpu.createKernel(function (frame) {
  const p = frame[this.thread.y][this.thread.x];
  return 0.299 * p[0] + 0.587 * p[1] + 0.114 * p[2];
}, { output: [64, 64], pipeline: true });

const learn = gpu.createKernel(function (frame, model) {
  const p = frame[this.thread.y][this.thread.x];
  const now = 0.299 * p[0] + 0.587 * p[1] + 0.114 * p[2];
  return (1 - this.constants.alpha) * model[this.thread.y][this.thread.x]
       + this.constants.alpha * now;
}, { output: [64, 64], pipeline: true, immutable: true, constants: { alpha: 0.05 } });

// Pass 3 — how foreground is this pixel, from 0 to 1?
const softMask = gpu.createKernel(function (frame, model) {
  const p = frame[this.thread.y][this.thread.x];
  const now = 0.299 * p[0] + 0.587 * p[1] + 0.114 * p[2];
  const d = Math.abs(now - model[this.thread.y][this.thread.x]);
  // TODO: ramp d from this.constants.lo to this.constants.hi and CLAMP the
  // result into 0…1. Anything outside that range breaks the composite.
  return d;
}, { output: [64, 64], pipeline: true, constants: { lo: 0.1, hi: 0.22 } });

// Pass 4 — feather the mask. A 3×3 mean softens the cut and demotes any
// surviving speck from 1 to 0.11. (Convolution & Filters' box blur, aimed
// at the mask instead of the picture.)
const feather = gpu.createKernel(function (mask) {
  let sum = 0;
  for (let dy = -1; dy <= 1; dy++) {
    for (let dx = -1; dx <= 1; dx++) {
      let yy = this.thread.y + dy;
      let xx = this.thread.x + dx;
      if (yy < 0) yy = 0;
      if (yy > this.constants.last) yy = this.constants.last;
      if (xx < 0) xx = 0;
      if (xx > this.constants.last) xx = this.constants.last;
      sum += mask[yy][xx];
    }
  }
  return sum / 9;
}, { output: [64, 64], pipeline: true, constants: { last: 63 } });

// Pass 5 — the composite. Texture in, pixels out.
const compose = gpu.createKernel(function (frame, mask) {
  const m = mask[this.thread.y][this.thread.x];
  const p = frame[this.thread.y][this.thread.x];

  // The stand-in backdrop: a 5×5 blur of the frame itself.
  let sr = 0;
  let sg = 0;
  let sb = 0;
  for (let dy = -2; dy <= 2; dy++) {
    for (let dx = -2; dx <= 2; dx++) {
      let yy = this.thread.y + dy;
      let xx = this.thread.x + dx;
      if (yy < 0) yy = 0;
      if (yy > this.constants.last) yy = this.constants.last;
      if (xx < 0) xx = 0;
      if (xx > this.constants.last) xx = this.constants.last;
      const q = frame[yy][xx];
      sr += q[0];
      sg += q[1];
      sb += q[2];
    }
  }
  const backR = sr / 25;
  const backG = sg / 25;
  const backB = sb / 25;

  // TODO: paint the source pixel where m is 1 and the blurred backdrop
  // where m is 0, sliding between them in between.
  this.color(p[0], p[1], p[2], 1);
}, { output: [64, 64], graphical: true, constants: { last: 63 } });

// Build the model from every frame but the last, then filter the last one.
let model = await seedModel(frames[0]);
for (let i = 1; i < frames.length - 1; i++) {
  model = await learn(frames[i], model);
}

const live = frames[frames.length - 1];
await compose(live, await feather(await softMask(live, model)));
render(compose.canvas);
`,
      solutionCode: `const gpu = new GPU({ mode });

// Passes 1 and 2 — the background model, from the last task.
const seedModel = gpu.createKernel(function (frame) {
  const p = frame[this.thread.y][this.thread.x];
  return 0.299 * p[0] + 0.587 * p[1] + 0.114 * p[2];
}, { output: [64, 64], pipeline: true });

const learn = gpu.createKernel(function (frame, model) {
  const p = frame[this.thread.y][this.thread.x];
  const now = 0.299 * p[0] + 0.587 * p[1] + 0.114 * p[2];
  return (1 - this.constants.alpha) * model[this.thread.y][this.thread.x]
       + this.constants.alpha * now;
}, { output: [64, 64], pipeline: true, immutable: true, constants: { alpha: 0.05 } });

// Pass 3 — how foreground is this pixel, from 0 to 1?
const softMask = gpu.createKernel(function (frame, model) {
  const p = frame[this.thread.y][this.thread.x];
  const now = 0.299 * p[0] + 0.587 * p[1] + 0.114 * p[2];
  const d = Math.abs(now - model[this.thread.y][this.thread.x]);
  const span = this.constants.hi - this.constants.lo;
  return Math.min(Math.max((d - this.constants.lo) / span, 0), 1);
}, { output: [64, 64], pipeline: true, constants: { lo: 0.1, hi: 0.22 } });

// Pass 4 — feather the mask. A 3×3 mean softens the cut and demotes any
// surviving speck from 1 to 0.11. (Convolution & Filters' box blur, aimed
// at the mask instead of the picture.)
const feather = gpu.createKernel(function (mask) {
  let sum = 0;
  for (let dy = -1; dy <= 1; dy++) {
    for (let dx = -1; dx <= 1; dx++) {
      let yy = this.thread.y + dy;
      let xx = this.thread.x + dx;
      if (yy < 0) yy = 0;
      if (yy > this.constants.last) yy = this.constants.last;
      if (xx < 0) xx = 0;
      if (xx > this.constants.last) xx = this.constants.last;
      sum += mask[yy][xx];
    }
  }
  return sum / 9;
}, { output: [64, 64], pipeline: true, constants: { last: 63 } });

// Pass 5 — the composite. Texture in, pixels out.
const compose = gpu.createKernel(function (frame, mask) {
  const m = mask[this.thread.y][this.thread.x];
  const p = frame[this.thread.y][this.thread.x];

  // The stand-in backdrop: a 5×5 blur of the frame itself.
  let sr = 0;
  let sg = 0;
  let sb = 0;
  for (let dy = -2; dy <= 2; dy++) {
    for (let dx = -2; dx <= 2; dx++) {
      let yy = this.thread.y + dy;
      let xx = this.thread.x + dx;
      if (yy < 0) yy = 0;
      if (yy > this.constants.last) yy = this.constants.last;
      if (xx < 0) xx = 0;
      if (xx > this.constants.last) xx = this.constants.last;
      const q = frame[yy][xx];
      sr += q[0];
      sg += q[1];
      sb += q[2];
    }
  }
  const backR = sr / 25;
  const backG = sg / 25;
  const backB = sb / 25;

  this.color(
    p[0] * m + backR * (1 - m),
    p[1] * m + backG * (1 - m),
    p[2] * m + backB * (1 - m),
    1
  );
}, { output: [64, 64], graphical: true, constants: { last: 63 } });

// Build the model from every frame but the last, then filter the last one.
let model = await seedModel(frames[0]);
for (let i = 1; i < frames.length - 1; i++) {
  model = await learn(frames[i], model);
}

const live = frames[frames.length - 1];
await compose(live, await feather(await softMask(live, model)));
render(compose.canvas);
`,
      inputs: utils => ({ frames: makeFrames(utils) }),
      // The catalogue card is this composite shown at ~300 CSS px, and 64×64 is
      // the LESSON's frame budget — eight frames through five passes, quick on
      // a laptop — not the card's. Same scene, same eight frames, same seed, at
      // four times the width. The capture script widens the kernels to match
      // (CARD_SCALE), blur radii included, or the backdrop would stop looking
      // blurred.
      cardInputs: utils => ({ frames: makeFrames(utils, SEED, SIZE * 4) }),
      inputNotes: {
        frames: 'Eight 64×64 ImageData frames in time order — a bright object crossing a static scene, with sensor noise. Pass one frame at a time into a kernel; inside it frames[i][y][x] is that pixel as [r, g, b, a] from 0 to 1.',
      },
      publicTests: [
        {
          name: 'five passes, four textures and a graphical finale',
          run: async ctx => {
            ctx.assert(ctx.kernels.length >= 5, `expected 5 kernels, found ${ctx.kernels.length}`);
            const [seed, learn, mask, feather, compose] = ctx.kernels;
            ctx.assert(seed.kernel && seed.kernel.pipeline === true, 'seedModel should keep pipeline: true');
            ctx.assert(learn.kernel && learn.kernel.immutable === true, 'learn reads its own output — keep immutable: true');
            ctx.assert(mask.kernel && mask.kernel.pipeline === true, 'softMask should keep pipeline: true');
            ctx.assert(feather.kernel && feather.kernel.pipeline === true, 'feather should keep pipeline: true');
            ctx.assert(compose.kernel && compose.kernel.graphical, 'the last kernel should be the graphical composite');
            ctx.assert(ctx.canvas, 'no canvas — did you call render(compose.canvas)?');
            ctx.assert(
              ctx.canvas.width === 64 && ctx.canvas.height === 64,
              `expected a 64×64 canvas, got ${ctx.canvas.width}×${ctx.canvas.height}`
            );
            if (ctx.resolvedMode !== 'cpu') {
              ctx.assert(
                compose.lastArgs && compose.lastArgs[1] && typeof compose.lastArgs[1].toArray === 'function',
                'the composite should be fed the feathered mask TEXTURE — nothing in this chain is downloaded'
              );
            }
          },
        },
        {
          name: 'the mask is normalised — never below 0, never above 1',
          run: async ctx => {
            const [seed, learn, mask] = ctx.kernels;
            const seq = makeFrames(ctx.utils);
            let model = await seed(seq[0]);
            for (let i = 1; i < seq.length - 1; i++) model = await learn(seq[i], model);
            const got = await toArr(await mask(seq[seq.length - 1], model));
            let lo = Infinity;
            let hi = -Infinity;
            for (let y = 0; y < 64; y++) {
              for (let x = 0; x < 64; x++) {
                lo = Math.min(lo, got[y][x]);
                hi = Math.max(hi, got[y][x]);
              }
            }
            ctx.assert(
              Number.isFinite(lo) && Number.isFinite(hi),
              'the mask contains values that are not numbers — check the ramp for a division by zero'
            );
            ctx.assert(
              lo >= -1e-3 && hi <= 1 + 1e-3,
              `the mask runs from ${lo.toFixed(3)} to ${hi.toFixed(3)}, and a mask has to live in 0…1 — ` +
                `the Math.min / Math.max clamp is missing, so the composite will subtract light where ` +
                `the mask goes negative and blow out where it goes past 1`
            );
            const ref = refSoftMask(seq[seq.length - 1], refModel(seq, ALPHA_MODEL, seq.length - 2), RAMP_LO, RAMP_HI);
            const miss = firstMismatch(got, ref, 5e-3);
            ctx.assert(
              !miss,
              miss && `mask cell [${miss.y}][${miss.x}] — expected ${miss.want.toFixed(4)}, got ${miss.got}`
            );
          },
        },
        {
          name: 'the composite is the sharp foreground over the blurred backdrop',
          run: async ctx => {
            const [seed, learn, mask, feather, compose] = ctx.kernels;
            const seq = makeFrames(ctx.utils);
            let model = await seed(seq[0]);
            for (let i = 1; i < seq.length - 1; i++) model = await learn(seq[i], model);
            const live = seq[seq.length - 1];
            await compose(live, await feather(await mask(live, model)));
            const pixels = compose.getPixels();
            ctx.assert(pixels.length === 64 * 64 * 4, 'pixel buffer should hold 64×64 RGBA values');

            const soft = refBox3(refSoftMask(live, refModel(seq, ALPHA_MODEL, seq.length - 2), RAMP_LO, RAMP_HI));
            const ref = refCompose(live, soft, overBlurred);
            const tol = 5;
            const hint = diagnoseCanvas(pixels, tol, [
              [refCompose(live, soft, (src, back, m) => overBlurred(back, src, m)),
                'the composite is inside out — a blurred subject over a sharp background. The mask ' +
                  'multiplies the SOURCE pixel and (1 − m) multiplies the backdrop'],
              [refCompose(live, soft, (src, back, m) => [src[0] * m + back[0] * m, src[1] * m + back[1] * m, src[2] * m + back[2] * m]),
                'both terms are weighted by m, so the two layers never sum to a full pixel — the backdrop needs (1 − m)'],
              [refCompose(live, soft, src => src),
                'that is the frame itself — the backdrop and the mask never reached this.color()'],
            ]);
            ctx.assert(
              canvasMatches(pixels, ref, tol),
              hint || 'the painted frame is not the mask-weighted blend of the source pixel and its 5×5 blur'
            );
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            const [seed, learn, mask, feather, compose] = ctx.kernels;
            const seq = makeFrames(ctx.utils, 271828);
            let model = await seed(seq[0]);
            for (let i = 1; i < seq.length - 1; i++) model = await learn(seq[i], model);
            const live = seq[seq.length - 1];
            await compose(live, await feather(await mask(live, model)));
            const pixels = compose.getPixels();
            const soft = refBox3(refSoftMask(live, refModel(seq, ALPHA_MODEL, seq.length - 2), RAMP_LO, RAMP_HI));
            const ref = refCompose(live, soft, overBlurred);
            ctx.assert(
              canvasMatches(pixels, ref, 5),
              'on a different sequence the painted frame is not the mask-weighted blend of the source and its blur'
            );
          },
        },
        {
          name: 'private test #2',
          run: async ctx => {
            // A frame that is nothing but the static scene: the model agrees
            // with it everywhere, the mask is 0, and the whole picture should
            // come back blurred. Anything sharp means the mask is leaking.
            const [seed, learn, mask, feather, compose] = ctx.kernels;
            const seq = makeFrames(ctx.utils, 999983);
            let model = await seed(seq[0]);
            for (let i = 1; i < seq.length - 1; i++) model = await learn(seq[i], model);
            const still = seq[0]; // the clean plate — the object is off-frame
            const got = await toArr(await mask(still, model));
            let lit = 0;
            for (let y = 0; y < 64; y++) {
              for (let x = 0; x < 64; x++) {
                if (got[y][x] > 0.5) lit++;
              }
            }
            ctx.assert(
              lit < 60,
              `${lit} pixels of an empty frame came back as foreground — against a model built from ` +
                `that same scene the mask should be 0 nearly everywhere`
            );
            await compose(still, await feather(await mask(still, model)));
            const pixels = compose.getPixels();
            const soft = refBox3(refSoftMask(still, refModel(seq, ALPHA_MODEL, seq.length - 2), RAMP_LO, RAMP_HI));
            ctx.assert(
              canvasMatches(pixels, refCompose(still, soft, overBlurred), 5),
              'an empty frame should come back essentially all backdrop — blurred everywhere'
            );
          },
        },
      ],
    },

    // ------------------------------------------------------------------ 6
    {
      slug: 'going-live',
      title: 'Going Live, Honestly',
      intro: `<p>Everything you have built runs on a sequence of frames that this module fabricated.
        Here is what changes when the frames come from a camera instead: <strong>nothing, in the
        kernels</strong>. gpu.js accepts an <code>HTMLVideoElement</code> as a kernel argument
        directly and re-uploads whatever is currently on screen each time you call it. The whole
        wiring is this:</p>
<pre><code>const video = document.createElement('video');
video.autoplay = true;
video.playsInline = true;
video.srcObject = await navigator.mediaDevices.getUserMedia({ video: true });
await video.play();

const gpu = new GPU();
const filter = gpu.createKernel(function (feed) {
  const p = feed[this.thread.y][this.thread.x];
  const l = 0.299 * p[0] + 0.587 * p[1] + 0.114 * p[2];
  this.color(l, l, l, 1);
}, { output: [640, 480], graphical: true });

document.body.appendChild(filter.canvas);

async function tick() {
  await filter(video);          // the current video frame, uploaded for you
  requestAnimationFrame(tick);  // ~16.7 ms later, again
}
requestAnimationFrame(tick);</code></pre>
        <p><strong>That code cannot run in this course, and this course is not going to pretend
        otherwise.</strong> Your code here executes inside a Web Worker — that is what lets a runaway
        kernel be killed instead of freezing the page — and a Worker has no
        <code>navigator.mediaDevices</code>, no <code>getUserMedia</code> and no
        <code>HTMLVideoElement</code>. There is no camera to reach and no video element to hand a
        kernel. The eight-frame sequence is the stand-in, and every line you have written works
        unchanged the day you paste it onto a page with the loop above.</p>
        <p>What is left is the shape of the thing: a filter that is <strong>built once</strong> and
        holds its own state, exposing one function you call per frame. Which is where the last
        wrinkle lives — the first frame. There is no history yet, so the model has to be born from
        the frame in your hand. Get that wrong and the whole first frame reads as motion, or the
        state is <code>null</code> and the arithmetic comes back as <code>NaN</code>. Every stateful
        filter has this three-line initialiser, and it is always the last thing anyone tests.</p>
        ${ARRAY_LAYOUT}`,
      goal: `<strong>Goal:</strong> finish <code>onFrame</code> so the filter seeds its model from
        the <em>first</em> frame only, then detects and learns on every frame after it.`,
      requirements: [
        'Seed <code>model</code> from the first frame only — later frames must not reset it',
        'Return the foreground mask for every frame the filter is given, the first one included',
        'Fold each frame into the model <em>after</em> detecting against it',
      ],
      hints: [
        {
          title: 'Hint 1 — "first frame" is a state, not an index',
          body: `<p>A live filter never sees an array — it is handed one frame and asked for one
            answer. So "is this the first frame?" has to be a property of the filter, not of a loop
            counter:</p>
<pre><code>if (model === null) {
  model = await seedModel(image);
}</code></pre>
<p>Which is why <code>model</code> starts as <code>null</code> rather than as a texture.</p>`,
        },
        {
          title: 'Hint 2 — what the first frame should report',
          body: `<p>Seeded from itself, the frame agrees with the model everywhere, so the mask is
            empty and the count is <code>0</code>. That is the correct answer, and it is much better
            than a first frame that lights up completely.</p>`,
        },
        {
          title: 'Hint 3 — the three lines',
          body: `<pre><code>if (model === null) model = await seedModel(image);
const mask = await detect(image, model);
model = await learn(image, model);
return mask;</code></pre>
<p>Detect, then learn. Reversing them lets each frame teach the model about itself before you
            ask the model what is new.</p>`,
        },
      ],
      transfer: `Build-once, call-per-frame is the shape of every real-time pipeline: a WebGPU app
        creates its pipelines and bind groups at startup and only records command buffers inside the
        frame callback; a CUDA video filter allocates its device buffers and loads its modules once
        and launches per frame; MediaPipe and OpenCV's <code>VideoCapture</code> loops are the same
        skeleton. The state that survives between calls — your background model — is the part that
        makes it a video filter rather than eight unrelated image filters.`,
      starterCode: `// The shippable shape: built once, one call per frame, state kept inside.
const gpu = new GPU({ mode });

const seedModel = gpu.createKernel(function (image) {
  const p = image[this.thread.y][this.thread.x];
  return 0.299 * p[0] + 0.587 * p[1] + 0.114 * p[2];
}, { output: [64, 64], pipeline: true });

const learn = gpu.createKernel(function (image, model) {
  const p = image[this.thread.y][this.thread.x];
  const now = 0.299 * p[0] + 0.587 * p[1] + 0.114 * p[2];
  return (1 - this.constants.alpha) * model[this.thread.y][this.thread.x]
       + this.constants.alpha * now;
}, { output: [64, 64], pipeline: true, immutable: true, constants: { alpha: 0.05 } });

const detect = gpu.createKernel(function (image, model) {
  const p = image[this.thread.y][this.thread.x];
  const now = 0.299 * p[0] + 0.587 * p[1] + 0.114 * p[2];
  if (Math.abs(now - model[this.thread.y][this.thread.x]) > this.constants.threshold) {
    return 1;
  }
  return 0;
}, { output: [64, 64], constants: { threshold: 0.12 } });

// The state that survives between frames. null means "nothing seen yet".
let model = null;

// One frame in, one mask out. On a page this is what requestAnimationFrame
// calls, with the <video> element in place of \`image\`.
async function onFrame(image) {
  // TODO: seed the model from the FIRST frame only — this line runs on
  // every frame, so the model is reborn each time and nothing is ever new.
  model = await seedModel(image);
  const mask = await detect(image, model);
  // TODO: let the model learn this frame, after detecting against it.
  return mask;
}

// requestAnimationFrame's stand-in: the sequence, one frame at a time.
for (let i = 0; i < frames.length; i++) {
  const mask = await onFrame(frames[i]);
  let moving = 0;
  for (let y = 0; y < 64; y++) {
    for (let x = 0; x < 64; x++) moving += mask[y][x];
  }
  console.log('frame ' + i + ': ' + moving + ' moving pixels');
}
`,
      solutionCode: `// The shippable shape: built once, one call per frame, state kept inside.
const gpu = new GPU({ mode });

const seedModel = gpu.createKernel(function (image) {
  const p = image[this.thread.y][this.thread.x];
  return 0.299 * p[0] + 0.587 * p[1] + 0.114 * p[2];
}, { output: [64, 64], pipeline: true });

const learn = gpu.createKernel(function (image, model) {
  const p = image[this.thread.y][this.thread.x];
  const now = 0.299 * p[0] + 0.587 * p[1] + 0.114 * p[2];
  return (1 - this.constants.alpha) * model[this.thread.y][this.thread.x]
       + this.constants.alpha * now;
}, { output: [64, 64], pipeline: true, immutable: true, constants: { alpha: 0.05 } });

const detect = gpu.createKernel(function (image, model) {
  const p = image[this.thread.y][this.thread.x];
  const now = 0.299 * p[0] + 0.587 * p[1] + 0.114 * p[2];
  if (Math.abs(now - model[this.thread.y][this.thread.x]) > this.constants.threshold) {
    return 1;
  }
  return 0;
}, { output: [64, 64], constants: { threshold: 0.12 } });

// The state that survives between frames. null means "nothing seen yet".
let model = null;

// One frame in, one mask out. On a page this is what requestAnimationFrame
// calls, with the <video> element in place of \`image\`.
async function onFrame(image) {
  if (model === null) {
    model = await seedModel(image); // the first frame is its own history
  }
  const mask = await detect(image, model);
  model = await learn(image, model); // detect first, learn second
  return mask;
}

// requestAnimationFrame's stand-in: the sequence, one frame at a time.
for (let i = 0; i < frames.length; i++) {
  const mask = await onFrame(frames[i]);
  let moving = 0;
  for (let y = 0; y < 64; y++) {
    for (let x = 0; x < 64; x++) moving += mask[y][x];
  }
  console.log('frame ' + i + ': ' + moving + ' moving pixels');
}
`,
      inputs: utils => ({ frames: makeFrames(utils) }),
      inputNotes: {
        frames: 'Eight 64×64 ImageData frames in time order — a bright object crossing a static scene, with sensor noise. Pass one frame at a time into a kernel; inside it frames[i][y][x] is that pixel as [r, g, b, a] from 0 to 1.',
      },
      publicTests: [
        {
          name: 'the filter answers every frame it is given — eight in, eight out',
          run: async ctx => {
            const lines = ctx.logs
              .filter(line => line.type === 'log' && line.text)
              .map(line => /^frame (\d+): (\d+) moving pixels$/.exec(line.text.trim()))
              .filter(Boolean);
            ctx.assert(
              lines.length === 8,
              lines.length === 7
                ? 'only seven frames were reported — a live filter is handed the first frame too, ' +
                  'and has to answer for it. Seed the model from it and report its (empty) mask.'
                : `expected one "frame N: M moving pixels" line per frame, found ${lines.length}`
            );
            const seen = lines.map(m => Number(m[1]));
            for (let i = 0; i < 8; i++) {
              ctx.assert(seen[i] === i, `frames should be reported in order — line ${i + 1} says frame ${seen[i]}`);
            }
          },
        },
        {
          name: 'the first frame is its own history, and the ones after it are not',
          run: async ctx => {
            const counts = ctx.logs
              .filter(line => line.type === 'log' && line.text)
              .map(line => /^frame (\d+): (\d+) moving pixels$/.exec(line.text.trim()))
              .filter(Boolean)
              .map(m => Number(m[2]));
            ctx.assert(counts.length >= 1, 'no per-frame report found');
            ctx.assert(
              counts[0] === 0,
              counts[0] > 3000
                ? 'the whole first frame reads as motion — the model starts out empty instead of ' +
                  'being seeded from the first frame you receive'
                : `the first frame reports ${counts[0]} moving pixels; seeded from itself it should agree with the model everywhere`
            );
            const later = counts.slice(3);
            ctx.assert(
              later.length > 0 && later.every(c => c > 100),
              later.every(c => c === 0)
                ? 'every frame reports zero motion — the model is being re-seeded on each frame, ' +
                  'so it always equals the frame it is compared against'
                : `frames 3 onward should each find hundreds of foreground pixels, got ${later.join(', ')}`
            );
          },
        },
        {
          name: 'the counts match a reference run of the same filter',
          run: async ctx => {
            const counts = ctx.logs
              .filter(line => line.type === 'log' && line.text)
              .map(line => /^frame (\d+): (\d+) moving pixels$/.exec(line.text.trim()))
              .filter(Boolean)
              .map(m => Number(m[2]));
            const ref = refLiveCounts(makeFrames(ctx.utils), ALPHA_MODEL, THRESHOLD);
            ctx.assert(counts.length === ref.length, `expected ${ref.length} reported frames, got ${counts.length}`);
            for (let i = 0; i < ref.length; i++) {
              // A handful of pixels either side: a difference sitting exactly on
              // the threshold is decided by the last bit of a float, and the two
              // backends do not have to agree about it.
              ctx.assert(
                Math.abs(counts[i] - ref[i]) <= 8,
                `frame ${i} reports ${counts[i]} moving pixels, expected about ${ref[i]} — ` +
                  `check that each frame is detected against the model BEFORE the model learns it`
              );
            }
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            // Drive the learner's own kernels the way onFrame should have, on a
            // different sequence, and check the model and mask at every step.
            const learn = ctx.kernels.find(k => k.kernel && k.kernel.immutable);
            const seed = ctx.kernels.find(k => k.kernel && k.kernel.pipeline && !k.kernel.immutable);
            const detect = ctx.kernels.find(k => k.kernel && !k.kernel.pipeline);
            ctx.assert(seed && learn && detect, 'expected a seed kernel, an immutable update and a plain detector');
            const seq = makeFrames(ctx.utils, 161803);
            const run = refSegmentRun(seq, ALPHA_MODEL, THRESHOLD);
            let model = await seed(seq[0]);
            const first = await detect(seq[0], model);
            ctx.assert(
              sumGrid(first) === 0,
              `a frame compared against a model seeded from itself should produce an empty mask, got ${sumGrid(first)} lit pixels`
            );
            model = await learn(seq[0], model);
            for (let i = 1; i < seq.length; i++) {
              const now = lumMap(seq[i]);
              const reference = run.models[i - 1];
              const delta = grid((y, x) => Math.abs(now[y][x] - reference[y][x]));
              const got = await detect(seq[i], model);
              const miss = firstMismatch(got, run.masks[i - 1], 0.25, fenceOf(delta, THRESHOLD));
              ctx.assert(
                !miss,
                miss && `frame ${i}, mask cell [${miss.y}][${miss.x}] should be ${miss.want}, got ${miss.got}`
              );
              model = await learn(seq[i], model);
            }
          },
        },
        {
          name: 'private test #2',
          run: async ctx => {
            // Handing the filter the SAME frame over and over must settle to
            // silence: the model converges on it and nothing is foreground.
            const learn = ctx.kernels.find(k => k.kernel && k.kernel.immutable);
            const seed = ctx.kernels.find(k => k.kernel && k.kernel.pipeline && !k.kernel.immutable);
            const detect = ctx.kernels.find(k => k.kernel && !k.kernel.pipeline);
            const seq = makeFrames(ctx.utils, 314159);
            const still = seq[4];
            let model = await seed(still);
            for (let i = 0; i < 6; i++) model = await learn(still, model);
            const lit = sumGrid(await detect(still, model));
            ctx.assert(
              lit === 0,
              `${lit} pixels still read as foreground after the model has seen the same frame seven ` +
                `times — an exponential update converges on whatever it is fed`
            );
          },
        },
      ],
    },
  ],
};
