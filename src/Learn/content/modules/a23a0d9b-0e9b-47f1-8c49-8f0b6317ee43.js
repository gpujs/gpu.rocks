// Module: Seam Carving: Content-Aware Resizing —
// uuid a23a0d9b-0e9b-47f1-8c49-8f0b6317ee43 (short id a23a0d9b).
// The file name is the uuid; identity lives in the exported object below,
// never in the path. No legacyId: this module has never had a pre-uuid URL.
//
// Six tasks: the ENERGY map (a Sobel magnitude, borrowed from Convolution &
// Filters, made border-safe and width-agnostic) → the cumulative DP, one
// kernel launch per row → the backtrack, deliberately left on the host →
// removing the seam as a GATHER → thirty-two removals with a frame scrubber
// and a rising cost curve → and the honest ending: what it does to a face,
// and the mask everybody actually ships.
//
// Kernel-authoring rules (contract): no closures inside kernel functions;
// EVERY kernel call is awaited (the course runs gpu.js `async` mode, where a
// kernel hands back a Promise on every backend); the picture narrows as seams
// come out, so every kernel that touches it is dynamicOutput +
// dynamicArguments and takes its width from this.output.x rather than a
// constant; a kernel that consumes its own last output is immutable: true.
//
// BACKEND. The picture is an ImageData (engine/utils.plainToImageData), and
// the paint pass is graphical — both of which WebGPU declines, so tasks 1, 5
// and 6 run their whole chain on WebGL in mode "auto". That is expected and
// correct, not a fallback: gpu.js pulls a kernel down to match the backend of
// the texture it consumes, so the chain stays on one backend by itself.
//
// FLOAT DETERMINISM. Large flat regions of the picture have energy 0, so the
// cumulative map is full of EXACT TIES and the cheapest seam is genuinely not
// unique — float32 and float64 pick different ones, of identical cost. Every
// assertion here is therefore tie-free by construction: tasks 2 and 4 compare
// VALUES (min(a, a) is a whatever the tie-break), task 3 checks that the seam
// it is shown is optimal rather than that it is one particular optimum, and
// tasks 5 and 6 assert the shape of the cost curve and the final width.

import { ARRAY_LAYOUT } from '../layoutNote.js';
import { plainToImageData } from '../../engine/utils.js';

const W = 128; // the picture, in columns…
const H = 72; // …and rows. Wide and short: one DP launch per ROW, so the
//               launch count is H, and the width is what we get to watch shrink.
const SEAMS = 32; // removals in the payoff — the addendum's cap, and enough
//                   to exhaust the cheap corridor and expose the knee.
const PENALTY = 20; // the mask's energy bonus. MEASURED: the dearest seam this
//                     picture has anywhere in the masked run prices at 18.6,
//                     and the ones actually taken top out at 14.3, so 20 on a
//                     single pixel already prices a seam out. The cumulative
//                     map peaks at 416 — that is deep inside the mask, where no
//                     seam goes; with no mask at all it peaks at 17.4 — and
//                     float32 still resolves 5e-5 at 416.
const CORRIDOR = [6, 26]; // the smooth band the first ~18 seams eat
const FACE = { cx: 92, cy: 34, r: 18 };

// The module card shows this task's last frame at ~300 CSS px — 600 device px
// on a phone — so 128 columns arrive as a near-fivefold upscale. CARD is the
// scale the capture asks for
// (scripts/capture-module-renders.mjs rewrites the kernel constants to match,
// and takes cardInputs below instead of inputs).
//
// Two, not four, and the reason is the shape of the bill rather than taste. The
// picture must narrow by the same FRACTION or the card stops being this card,
// so the run is SEAMS * CARD removals of a picture H * CARD rows tall — and the
// cumulative map costs one awaited launch per row, so the launch count grows as
// CARD²: 64 removals × 150 launches at 2, and 128 × 294 at 4. Only the first
// fits the 10 s run watchdog (engine/runner.js) that the frames have to arrive
// within — canvases ride home in the run's RESULT, never in the streamed log
// lines, so a run the watchdog kills hands the capture no picture at all.
const CARD = 2;

// ---- the picture ----------------------------------------------------------
//
// A synthetic scene, built so that seam carving's behaviour is legible rather
// than flattering: a speckled background (every seam crossing it pays), one
// smooth vertical corridor on the left (cheap — the first ~18 seams eat it),
// two straight vertical poles, and a soft-edged face whose interior is
// perfectly flat (cheap — so the seams eat that next, which is exactly the
// failure the last task is about).
//
// MEASURED, so the prose can be honest about it: over 32 removals EVERY row in
// which the face is at least 30 columns wide loses exactly 7 of them (its
// widest row goes 37 → 30), and not the same 7 — 28 distinct face columns are
// cut somewhere, and the face's own centre column is removed outright in 13 of
// its 33 rows while landing in column 64, 66 or 67 in the rest. Both poles come
// through dead straight: not one seam pixel ever lands on either of them. Task
// 6 says exactly that rather than promising a kink this picture does not
// produce.
//
// Takes a scale so the card can be captured at CARD times the resolution
// without the lesson moving: EVERY length here is written as a multiple of the
// lesson's own, so at scale 1 `at(n)` is n and this is the picture it always
// was — verified byte-identical, not assumed. The face, the poles, the corridor
// and the soft edge all grow with the picture, which is the whole point: a
// scene whose features stayed 18 px wide in a 256-column picture would be a
// different scene, not the same one sampled better.
function makeScene(utils, scale = 1) {
  const rand = utils.seededRandom(0x5ea3);
  const clamp01 = v => (v < 0 ? 0 : v > 1 ? 1 : v);
  const q8 = v => Math.round(clamp01(v) * 255) / 255; // 8-bit exact: an
  //     ImageData cannot hold anything else, so quantize before the tests
  //     compute expectations from these numbers.
  const w = W * scale;
  const h = H * scale;
  const at = n => n * scale; // a length in lesson pixels, at this scale
  const face = { cx: at(FACE.cx), cy: at(FACE.cy), r: at(FACE.r) };
  const image = new Array(h);
  for (let y = 0; y < h; y++) {
    const row = new Array(w);
    for (let x = 0; x < w; x++) {
      const t = y / (h - 1);
      let r = 0.4 + 0.3 * t;
      let g = 0.56 + 0.26 * t;
      let b = 0.82 - 0.06 * t;
      // `smooth` runs 0 (full texture) to 1 (perfectly flat). Flat is cheap,
      // and cheap is what a seam eats.
      let smooth = x >= at(CORRIDOR[0]) && x < at(CORRIDOR[1]) ? 1 : 0;
      if (y >= at(30) && y < at(40)) smooth = 1; // a smooth horizontal band, so
      //                                    a seam can wander sideways for free
      const dx = x - face.cx;
      const dy = (y - face.cy) * 1.1;
      const d = Math.sqrt(dx * dx + dy * dy);
      if (d <= face.r) {
        const k = Math.min(1, (face.r - d) / at(6)); // soft edge: cheap to enter
        r += (0.95 - r) * k;
        g += (0.79 - g) * k;
        b += (0.64 - b) * k;
        smooth = Math.max(smooth, k);
      }
      for (const px of [44, 62]) {
        if (x >= at(px) && x < at(px + 2)) {
          r = 0.3;
          g = 0.31;
          b = 0.38;
          smooth = 1;
        }
      }
      if (d <= face.r) {
        const eye = Math.min(
          Math.hypot(x - (face.cx - face.r * 0.38), (y - (face.cy - face.r * 0.22)) * 1.15),
          Math.hypot(x - (face.cx + face.r * 0.38), (y - (face.cy - face.r * 0.22)) * 1.15)
        );
        if (eye <= face.r * 0.15) {
          r = 0.13;
          g = 0.13;
          b = 0.17;
        }
        const my = y - (face.cy + face.r * 0.42);
        if (Math.abs(dx) <= face.r * 0.44 && my >= 0 && my <= face.r * 0.13) {
          r = 0.56;
          g = 0.23;
          b = 0.23;
        }
      }
      const speckle = (rand() - 0.5) * 0.3 * (1 - smooth);
      const n = (rand() - 0.5) * 0.012; // a whisper, so "flat" is never a
      //                                   perfectly repeated byte
      row[x] = [q8(r + speckle + n), q8(g + speckle + n), q8(b + speckle + n), 1];
    }
    image[y] = row;
  }
  return image;
}

// The face, painted over: 1 where nothing may be removed, 0 everywhere else.
// Scaled with the scene it protects — the mask lives in image space, so a mask
// built at one size over a picture built at another protects the wrong pixels.
function makeMask(scale = 1) {
  const w = W * scale;
  const h = H * scale;
  const at = n => n * scale;
  const face = { cx: at(FACE.cx), cy: at(FACE.cy), r: at(FACE.r) };
  const mask = new Array(h);
  for (let y = 0; y < h; y++) {
    const row = new Array(w).fill(0);
    for (let x = 0; x < w; x++) {
      const dx = x - face.cx;
      const dy = (y - face.cy) * 1.1;
      if (Math.sqrt(dx * dx + dy * dy) <= face.r + at(2)) row[x] = 1;
    }
    mask[y] = row;
  }
  return mask;
}

// ---- the reference pipeline, in plain float64 JS --------------------------
//
// Every one of these has a kernel twin in the tasks below; they are the
// expectations the tests compare against, and the fixtures the middle tasks
// are handed as inputs.

const round4 = v => Math.round(v * 1e4) / 1e4;

function grayOf(image) {
  return image.map(row => row.map(p => 0.299 * p[0] + 0.587 * p[1] + 0.114 * p[2]));
}

// Sobel gradient magnitude with the neighbour coordinates CLAMPED, so the
// border has an energy instead of a hole — a seam is allowed to run down the
// edge of the picture, so every column needs a price.
function energyOf(gray) {
  const h = gray.length;
  const w = gray[0].length;
  const out = new Array(h);
  for (let y = 0; y < h; y++) {
    const row = new Array(w);
    for (let x = 0; x < w; x++) {
      const xm = Math.max(x - 1, 0);
      const xp = Math.min(x + 1, w - 1);
      const ym = Math.max(y - 1, 0);
      const yp = Math.min(y + 1, h - 1);
      const gx =
        gray[ym][xp] + 2 * gray[y][xp] + gray[yp][xp] -
        (gray[ym][xm] + 2 * gray[y][xm] + gray[yp][xm]);
      const gy =
        gray[yp][xm] + 2 * gray[yp][x] + gray[yp][xp] -
        (gray[ym][xm] + 2 * gray[ym][x] + gray[ym][xp]);
      row[x] = Math.sqrt(gx * gx + gy * gy);
    }
    out[y] = row;
  }
  return out;
}

function cumulativeOf(energy) {
  const h = energy.length;
  const w = energy[0].length;
  const rows = [Array.from(energy[0])];
  for (let y = 1; y < h; y++) {
    const prev = rows[y - 1];
    const row = new Array(w);
    for (let x = 0; x < w; x++) {
      let best = prev[x];
      if (x > 0) best = Math.min(best, prev[x - 1]);
      if (x + 1 < w) best = Math.min(best, prev[x + 1]);
      row[x] = energy[y][x] + best;
    }
    rows.push(row);
  }
  return rows;
}

function argminOf(row) {
  let best = 0;
  for (let i = 1; i < row.length; i++) if (row[i] < row[best]) best = i;
  return best;
}

function backtrackOf(cost) {
  const h = cost.length;
  const w = cost[0].length;
  const seam = new Array(h);
  let x = argminOf(cost[h - 1]);
  seam[h - 1] = x;
  for (let y = h - 2; y >= 0; y--) {
    const row = cost[y];
    let best = x;
    if (x > 0 && row[x - 1] < row[best]) best = x - 1;
    if (x + 1 < w && row[x + 1] < row[best]) best = x + 1;
    x = best;
    seam[y] = x;
  }
  return seam;
}

// The two walks that look plausible and are not. Both are exact integer
// sequences, so a probe naming one has to reproduce every row of the learner's
// answer before it may speak — and if both matched at once, diagnoseSeam sees
// two different sentences and says nothing, which is the behaviour we want.
//
// They are computed here rather than inside one test because a wrong walk does
// not always fail the test its mistake is named in: the top-down walk on THIS
// picture happens to finish on a cheapest cell, so the test that checks the
// ending passes it and the test that checks the steps is the one that fails.
function downwardSeam(cost) {
  const h = cost.length;
  const w = cost[0].length;
  const out = new Array(h);
  let x = argminOf(cost[0]);
  out[0] = x;
  for (let y = 1; y < h; y++) {
    let b = x;
    if (x > 0 && cost[y][x - 1] < cost[y][b]) b = x - 1;
    if (x + 1 < w && cost[y][x + 1] < cost[y][b]) b = x + 1;
    x = b;
    out[y] = x;
  }
  return out;
}

function noCentreSeam(cost) {
  const h = cost.length;
  const w = cost[0].length;
  const out = new Array(h);
  let x = argminOf(cost[h - 1]);
  out[h - 1] = x;
  for (let y = h - 2; y >= 0; y--) {
    let b = x > 0 ? x - 1 : x + 1;
    if (x + 1 < w && cost[y][x + 1] < cost[y][b]) b = x + 1;
    x = b;
    out[y] = x;
  }
  return out;
}

const WALK_PROBES = cost => [
  [cost.map(argminOf),
    'that is the cheapest cell of each row taken independently — which is not a seam. A seam may only move one column per row, so each step has to come from the window x - 1 … x + 1 around where it already is'],
  [downwardSeam(cost),
    'that seam was walked from the top downwards. Only the BOTTOM row of the cumulative map holds finished prices — row 0 is just raw energy — so the walk has to start at the cheapest cell of the last row and go up'],
  [noCentreSeam(cost),
    'the window is missing its middle: a seam is allowed to go straight up, so the three candidates are x - 1, x AND x + 1'],
];

function carveOf(plane, seam) {
  return plane.map((row, y) => {
    const out = new Array(row.length - 1);
    for (let x = 0; x < out.length; x++) out[x] = x < seam[y] ? row[x] : row[x + 1];
    return out;
  });
}

// The whole loop, host-side: what tasks 5 and 6 are asked to reproduce.
// `mask` optional; when present its pixels carry PENALTY and reflow with the
// picture, which is the point of the last task.
function carveRunUncached(gray, count, mask) {
  const costs = [];
  let g = gray;
  let m = mask ? mask.map(row => Array.from(row)) : null;
  for (let k = 0; k < count; k++) {
    const e = energyOf(g);
    if (m) {
      for (let y = 0; y < e.length; y++) {
        for (let x = 0; x < e[0].length; x++) e[y][x] += PENALTY * m[y][x];
      }
    }
    const rows = cumulativeOf(e);
    const seam = backtrackOf(rows);
    costs.push(rows[rows.length - 1][seam[seam.length - 1]]);
    g = carveOf(g, seam);
    if (m) m = carveOf(m, seam);
  }
  return { costs, gray: g, mask: m };
}

// Tasks 5 and 6 compare against the same unmasked run several times over
// (once to build an input, three more inside tests), and it is ~30 ms of
// float64 arithmetic each time. Memoize on the seam count — and on the picture's
// WIDTH, because the card capture builds the same scene at CARD scale in the
// same page, and a key of the count alone would hand it the lesson's costs.
const runCache = new Map();
function carveRun(gray, count, mask) {
  if (mask) return carveRunUncached(gray, count, mask);
  const key = `${gray[0].length}:${count}`;
  if (!runCache.has(key)) runCache.set(key, carveRunUncached(gray, count));
  return runCache.get(key);
}

// One scene, built once per scale: every fixture below is derived from it, and
// the tests recompute nothing the learner is not also given. A Map rather than
// one slot because the lesson's fixtures and the card capture's larger ones can
// both be asked for in one page, and a single slot would serve the second
// caller the first caller's picture.
const fixtureCache = new Map();
function fixtures(utils, scale = 1) {
  if (fixtureCache.has(scale)) return fixtureCache.get(scale);
  const image = makeScene(utils, scale);
  // rounded to 4 dp so the numbers the learner is SHOWN are the numbers the
  // tests use — and so a fixture printed in the inputs panel is readable
  const gray = grayOf(image).map(row => row.map(round4));
  const energy = energyOf(gray).map(row => row.map(round4));
  const cost = cumulativeOf(energy);
  const built = { image, gray, energy, cost, seam: backtrackOf(cost) };
  fixtureCache.set(scale, built);
  return built;
}

function sceneImage(utils, scale = 1) {
  return plainToImageData(fixtures(utils, scale).image);
}

// A small flat plane, for probing a kernel's identity or its border handling.
function flatPlane(w, h, value) {
  return new Array(h).fill(0).map(() => new Array(w).fill(value));
}

// ---- kernel sources shared by the payoff tasks ----------------------------
//
// Tasks 5 and 6 hand back everything tasks 1–4 built, prewired, so the work
// left is the orchestration. Keeping the sources here means the starter and
// the solution cannot drift apart, and neither can drift from task 1's answer.

const CHANNEL_KERNEL = `// ImageData in, one numeric plane out: 0/1/2 are r/g/b, 3 is the luminance
// the energy map is built from. Four planes that all have to reflow together.
const channel = gpu.createKernel(function (image, c) {
  const p = image[this.thread.y][this.thread.x];
  if (c === 0) return p[0];
  if (c === 1) return p[1];
  if (c === 2) return p[2];
  return 0.299 * p[0] + 0.587 * p[1] + 0.114 * p[2];
}, { output: [128, 72] });`;

const ENERGY_KERNEL = `// Task 1: the Sobel magnitude, border-clamped, width from this.output.x.
const energy = gpu.createKernel(function (gray) {
  const x = this.thread.x;
  const y = this.thread.y;
  const xm = Math.max(x - 1, 0);
  const xp = Math.min(x + 1, this.output.x - 1);
  const ym = Math.max(y - 1, 0);
  const yp = Math.min(y + 1, this.output.y - 1);
  const gx = (gray[ym][xp] + 2 * gray[y][xp] + gray[yp][xp])
           - (gray[ym][xm] + 2 * gray[y][xm] + gray[yp][xm]);
  const gy = (gray[yp][xm] + 2 * gray[yp][x] + gray[yp][xp])
           - (gray[ym][xm] + 2 * gray[ym][x] + gray[ym][xp]);
  return Math.sqrt(gx * gx + gy * gy);
}, { output: [128, 72], dynamicOutput: true, dynamicArguments: true });`;

const STEP_KERNEL = `// Task 2: one row of the cumulative map. One launch per row.
const step = gpu.createKernel(function (eRow, prev) {
  const x = this.thread.x;
  let best = prev[x];
  if (x > 0) best = Math.min(best, prev[x - 1]);
  if (x + 1 < this.output.x) best = Math.min(best, prev[x + 1]);
  return eRow[x] + best;
}, { output: [128], immutable: true, dynamicOutput: true, dynamicArguments: true });`;

const CARVE_KERNEL = `// Task 4: remove the seam by gathering — one column narrower.
const carve = gpu.createKernel(function (plane, seam) {
  const x = this.thread.x;
  const y = this.thread.y;
  if (x < seam[y]) return plane[y][x];
  return plane[y][x + 1];
}, { output: [127, 72], immutable: true, dynamicOutput: true, dynamicArguments: true });`;

const PAINT_KERNEL = `// The canvas never shrinks — the PICTURE does. Everything from column w
// rightwards is painted as empty frame, so the narrowing is visible.
const paint = gpu.createKernel(function (r, g, b, w) {
  const x = this.thread.x;
  // this.color() paints from the bottom up — thread.y 0 is the BOTTOM row of
  // the canvas, on every backend — so read the rows in reverse to put row 0
  // back at the top of the picture where it belongs.
  const y = this.output.y - 1 - this.thread.y;
  if (x < w) {
    this.color(r[y][x], g[y][x], b[y][x], 1);
  } else {
    this.color(0.09, 0.10, 0.12, 1);
  }
}, { output: [128, 72], graphical: true, dynamicArguments: true });`;

const BACKTRACK_FN = `// Task 3: 72 sequential steps, three numbers each. It stays on the host.
function backtrack(cost) {
  const h = cost.length;
  const w = cost[0].length;
  const seam = new Array(h);
  let x = 0;
  for (let i = 1; i < w; i++) if (cost[h - 1][i] < cost[h - 1][x]) x = i;
  seam[h - 1] = x;
  for (let y = h - 2; y >= 0; y--) {
    let best = x;
    if (x > 0 && cost[y][x - 1] < cost[y][best]) best = x - 1;
    if (x + 1 < w && cost[y][x + 1] < cost[y][best]) best = x + 1;
    x = best;
    seam[y] = x;
  }
  return seam;
}`;

// ---- near-miss diagnosis --------------------------------------------------
//
// The house rule: when a failing value is exactly what one specific mistake
// would produce, name that mistake instead of reporting two numbers — and
// stay SILENT when two candidates are indistinguishable, because a confident
// wrong diagnosis is worse than a plain numeric mismatch. diagnose() speaks
// only when the observation matches a probe within the test's own tolerance
// AND the correct answer does not, and only when every probe that matched
// agrees on the sentence.
function diagnose(got, expected, eps, probes) {
  const hits = probes
    .filter(p => Math.abs(got - p[0]) <= eps && Math.abs(expected - p[0]) > eps)
    .map(p => p[1]);
  return hits.length && hits.every(m => m === hits[0]) ? hits[0] : null;
}

// The same idea for a whole grid: a probe must predict EVERY cell before it
// may speak, which is what stops "you forgot the square root" firing on a
// picture whose energies happen to be near 1 (where v and v² agree).
function diagnoseGrid(out, expected, eps, probes) {
  const h = expected.length;
  const w = expected[0].length;
  const hits = probes
    .filter(([value]) => {
      let differs = false;
      for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
          const candidate = value(y, x);
          if (!out[y] || !(Math.abs(out[y][x] - candidate) <= eps)) return false;
          if (Math.abs(expected[y][x] - candidate) > eps) differs = true;
        }
      }
      return differs;
    })
    .map(p => p[1]);
  return hits.length && hits.every(m => m === hits[0]) ? hits[0] : null;
}

// …and for a 1D row.
function diagnoseRow(out, expected, eps, probes) {
  const hits = probes
    .filter(([value]) => {
      let differs = false;
      for (let i = 0; i < expected.length; i++) {
        if (!(Math.abs(out[i] - value(i)) <= eps)) return false;
        if (Math.abs(expected[i] - value(i)) > eps) differs = true;
      }
      return differs;
    })
    .map(p => p[1]);
  return hits.length && hits.every(m => m === hits[0]) ? hits[0] : null;
}

// A seam is a sequence of integers, so its probes compare exactly. A probe
// only ever runs after the seam has already been judged wrong, so "differs
// from the right answer" is not needed here — but the candidate must match on
// every row before it is named.
function diagnoseSeam(got, probes) {
  const hits = probes
    .filter(([value]) => value.length === got.length && value.every((v, i) => v === got[i]))
    .map(p => p[1]);
  return hits.length && hits.every(m => m === hits[0]) ? hits[0] : null;
}

// "Your border is wrong" is a DIAGNOSIS, not a caption, so it has to be earned
// the same way every other probe here does. These two say yes only when the
// numbers actually single the edge out: everything in the middle is right and
// at least one edge cell is not. Without that gate a learner who scaled the
// whole map by 0.9 gets sent to look at a clamp that was never the problem.
function borderOnly(out, expected, eps) {
  const h = expected.length;
  const w = expected[0].length;
  let borderWrong = false;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const bad = !out[y] || !(Math.abs(out[y][x] - expected[y][x]) <= eps);
      if (bad && x > 0 && y > 0 && x < w - 1 && y < h - 1) return false;
      if (bad) borderWrong = true;
    }
  }
  return borderWrong;
}

// …and the 1D version, for a single row of the cumulative map.
function endsOnly(out, expected, eps) {
  let endWrong = false;
  for (let i = 0; i < expected.length; i++) {
    const bad = !(Math.abs(out[i] - expected[i]) <= eps);
    if (bad && i > 0 && i < expected.length - 1) return false;
    if (bad) endWrong = true;
  }
  return endWrong;
}

const NOT_A_NUMBER =
  'that cell came back NaN — the neighbour lookup ran off the edge of the picture. ' +
  'Clamp the coordinates (Math.max(x - 1, 0) and Math.min(x + 1, this.output.x - 1)) ' +
  'instead of reading outside it: a seam is allowed to run right down the border, so ' +
  'every column needs a price.';

function nanHint(out) {
  for (let y = 0; y < out.length; y++) {
    for (let x = 0; x < out[y].length; x++) {
      if (!Number.isFinite(out[y][x])) return NOT_A_NUMBER;
    }
  }
  return null;
}

// ---- locating the learner's kernels ---------------------------------------

// Task 1's energy kernel is the only DYNAMIC one in that task (the prewired
// luminance and paint passes are fixed-size), which is what the requirements
// ask for and what tasks 5 and 6 need. Same discriminator Reductions uses for
// its ladder rung.
function findDynamicKernel(ctx, predicate) {
  return ctx.kernels.find(k => k.kernel && k.kernel.dynamicOutput && !k.kernel.graphical &&
    (!predicate || predicate(k))) || null;
}

// Every number that appeared in a console.log line (Reductions' helper).
function loggedNumbers(logs) {
  const out = [];
  for (const line of logs) {
    if (line.type !== 'log' || !line.text) continue;
    const matches = line.text.match(/-?\d+(?:\.\d+)?/g);
    if (matches) for (const m of matches) out.push(parseFloat(m));
  }
  return out;
}

// The series a plot() call carried. The payload is plain JSON and crosses no
// boundary before the tests see it, so the FULL array is readable here — which
// is how a task whose answer is a plain-JavaScript array gets tested at all.
function plottedSeries(ctx, predicate) {
  for (const line of ctx.logs) {
    if (line.type !== 'plot' || !line.plot) continue;
    for (const series of line.plot.series) {
      if (!predicate || predicate(series)) return series.values;
    }
  }
  return null;
}

function allPlottedSeries(ctx) {
  const out = [];
  for (const line of ctx.logs) {
    if (line.type !== 'plot' || !line.plot) continue;
    for (const series of line.plot.series) out.push(series);
  }
  return out;
}

function canvasFrames(ctx) {
  return ctx.logs.filter(line => line.type === 'canvas').length;
}

const mean = arr => arr.reduce((a, b) => a + b, 0) / arr.length;

// The painted frame is W wide whatever the picture is doing, and everything
// from the picture's width rightwards is the empty-frame colour. So the width
// of that band IS the answer, and reading it column-wise makes the check
// immune to the row order getPixels() happens to hand back.
const FRAME_RGB = [Math.round(0.09 * 255), Math.round(0.1 * 255), Math.round(0.12 * 255)];

function columnIsFrame(pixels, x) {
  for (let y = 0; y < H; y++) {
    const i = (y * W + x) * 4;
    if (
      Math.abs(pixels[i] - FRAME_RGB[0]) > 2 ||
      Math.abs(pixels[i + 1] - FRAME_RGB[1]) > 2 ||
      Math.abs(pixels[i + 2] - FRAME_RGB[2]) > 2
    ) {
      return false;
    }
  }
  return true;
}

export default {
  uuid: 'a23a0d9b-0e9b-47f1-8c49-8f0b6317ee43',
  version: 1,
  slug: 'seam-carving',
  title: 'Seam Carving: Content-Aware Resizing',
  blurb:
    'Shrink a picture by deleting its most boring pixels — an energy map, a wavefront DP one launch per row, and a gather that reflows the image.',
  tasks: [
    // ------------------------------------------------------------------ 1
    {
      slug: 'energy-map',
      title: 'What Can We Afford to Lose?',
      intro: `<p>To make a picture narrower you can squash it, crop it — or delete the pixels
        nobody would miss. Seam carving does the third: it removes a <strong>seam</strong>, a
        connected path of one pixel per row, threaded through the least interesting part of the
        picture, and does it once per column you want to lose. Everything then slides across to
        close the gap, so the interesting parts keep their proportions and the boring parts get
        squeezed out.</p>
        <p>"Interesting" needs a number, and the usual one is the <strong>gradient
        magnitude</strong>: flat regions score near zero, edges score high. That is exactly the
        Sobel pass <em>Convolution &amp; Filters</em> already derives — the two weight grids are in
        the starter and we are not deriving them again. What changes here is two small things, and
        both matter later:</p>
        <p>The energy must be defined <strong>at the border</strong>. A seam is allowed to run
        straight down the edge of the picture, so column 0 needs a price like every other column —
        clamp the neighbour coordinates instead of painting the border black. And the kernel must
        work at <strong>any width</strong>, because after the first seam comes out the picture is
        127 columns wide, then 126… so the size comes from <code>this.output.x</code> and
        <code>this.output.y</code>, never from a constant.</p>
        ${ARRAY_LAYOUT}`,
      goal: `<strong>Goal:</strong> finish <code>energy</code> so that cell <code>[y][x]</code>
        holds <code>Math.sqrt(gx * gx + gy * gy)</code> for the 3×3 neighbourhood of
        <code>gray</code>, with every neighbour coordinate clamped to the picture.`,
      requirements: [
        'Clamp all four neighbour coordinates — <code>Math.max(x - 1, 0)</code> and <code>Math.min(x + 1, this.output.x - 1)</code>, the same on <code>y</code> — so the border has an energy rather than a hole',
        'Compute <code>gx</code> (right column minus left) and <code>gy</code> (bottom row minus top) with the two Sobel grids in the starter',
        'Return the magnitude <code>Math.sqrt(gx * gx + gy * gy)</code>',
        'Keep <code>dynamicOutput: true</code> and <code>dynamicArguments: true</code> — the same kernel runs at every width the picture passes through',
      ],
      hints: [
        {
          title: 'Hint 1 — the clamp',
          body: `<p>Four one-liners, and they are the only thing standing between you and a NaN in
            the first and last row and column:</p>
<pre><code>const xm = Math.max(x - 1, 0);
const xp = Math.min(x + 1, this.output.x - 1);</code></pre>
<p>— and the same pair on <code>y</code>, against <code>this.output.y - 1</code>. A pixel on
            the edge now simply sees itself twice, which is what "replicate the border" means.</p>`,
        },
        {
          title: 'Hint 2 — the two sums',
          body: `<p>Read them straight off the grids in the starter — right column minus left,
            middle row counted double:</p>
<pre><code>const gx = (gray[ym][xp] + 2 * gray[y][xp] + gray[yp][xp])
         - (gray[ym][xm] + 2 * gray[y][xm] + gray[yp][xm]);</code></pre>
<p><code>gy</code> is the same move on rows: bottom row minus top row, middle column
            counted double.</p>`,
        },
        {
          title: 'Hint 3 — why not a constant',
          body: `<p><code>this.output.x</code> is the width of <em>this</em> launch, not of the
            original picture. Hard-code <code>127</code> and the kernel is right exactly once —
            the first time — and then quietly clamps to a column that no longer exists.</p>`,
        },
      ],
      transfer: `Every content-aware tool starts by building a cost field and only then decides
        what to do with it. NVIDIA's NPP and OpenCV's CUDA module both ship Sobel as a primitive;
        a video encoder builds the same gradient field to decide which macroblocks deserve bits;
        and "saliency map first, decision second" is the shape of seam carving, content-aware
        fill and adaptive sampling alike.`,
      starterCode: `// Energy: how much does this pixel's neighbourhood change? Flat = cheap.
const gpu = new GPU({ mode });

// The picture arrives as ImageData. One pass turns it into luminance
// (module "Data In, Data Out" writes this one) — already done for you.
const luminance = gpu.createKernel(function (image) {
  const p = image[this.thread.y][this.thread.x];
  return 0.299 * p[0] + 0.587 * p[1] + 0.114 * p[2];
}, { output: [128, 72] });

// Sobel's two weight grids, exactly as Convolution & Filters derives them:
//
//        Gx              Gy
//    -1   0  +1      -1  -2  -1
//    -2   0  +2       0   0   0
//    -1   0  +1      +1  +2  +1
//
const energy = gpu.createKernel(function (gray) {
  const x = this.thread.x;
  const y = this.thread.y;
  // TODO 1: clamp the four neighbour coordinates to the picture. The width
  //         and height of THIS launch are this.output.x and this.output.y.
  const xm = x;
  const xp = x;
  const ym = y;
  const yp = y;
  // TODO 2: gx = right column - left column, gy = bottom row - top row,
  //         then return Math.sqrt(gx * gx + gy * gy).
  return gray[y][x];
}, {
  output: [128, 72],
  dynamicOutput: true,   // the picture narrows every time a seam comes out
  dynamicArguments: true,
});

// A look at what we built: bright where the picture is busy.
const paint = gpu.createKernel(function (e) {
  // thread.y 0 is the BOTTOM row of a canvas, so read the rows in reverse
  const v = e[this.output.y - 1 - this.thread.y][this.thread.x];
  this.color(v, v, v, 1);
}, { output: [128, 72], graphical: true });

const gray = await luminance(photo);
const map = await energy(gray);
await paint(map);
render(paint.canvas);

// A logged numeric array draws its own sparkline: this is one row of energy,
// left to right — flat corridor, texture, two poles, and the face.
console.log('energy across row 20:', map[20]);
`,
      solutionCode: `// Energy: how much does this pixel's neighbourhood change? Flat = cheap.
const gpu = new GPU({ mode });

const luminance = gpu.createKernel(function (image) {
  const p = image[this.thread.y][this.thread.x];
  return 0.299 * p[0] + 0.587 * p[1] + 0.114 * p[2];
}, { output: [128, 72] });

const energy = gpu.createKernel(function (gray) {
  const x = this.thread.x;
  const y = this.thread.y;
  const xm = Math.max(x - 1, 0);
  const xp = Math.min(x + 1, this.output.x - 1);
  const ym = Math.max(y - 1, 0);
  const yp = Math.min(y + 1, this.output.y - 1);
  const gx = (gray[ym][xp] + 2 * gray[y][xp] + gray[yp][xp])
           - (gray[ym][xm] + 2 * gray[y][xm] + gray[yp][xm]);
  const gy = (gray[yp][xm] + 2 * gray[yp][x] + gray[yp][xp])
           - (gray[ym][xm] + 2 * gray[ym][x] + gray[ym][xp]);
  return Math.sqrt(gx * gx + gy * gy);
}, {
  output: [128, 72],
  dynamicOutput: true,
  dynamicArguments: true,
});

const paint = gpu.createKernel(function (e) {
  // thread.y 0 is the BOTTOM row of a canvas, so read the rows in reverse
  const v = e[this.output.y - 1 - this.thread.y][this.thread.x];
  this.color(v, v, v, 1);
}, { output: [128, 72], graphical: true });

const gray = await luminance(photo);
const map = await energy(gray);
await paint(map);
render(paint.canvas);

console.log('energy across row 20:', map[20]);
`,
      inputs: utils => ({ photo: sceneImage(utils) }),
      publicTests: [
        {
          name: 'a dynamic energy kernel, and a flat field costs nothing',
          run: async ctx => {
            ctx.assert(ctx.kernels.length >= 2, 'no energy kernel was created — call gpu.createKernel()');
            const energy = findDynamicKernel(ctx);
            ctx.assert(
              energy,
              'no kernel with dynamicOutput: true found — the energy kernel needs it (and dynamicArguments) so the same kernel can run at every width the picture shrinks through'
            );
            energy.setOutput([16, 12]);
            const out = await energy(flatPlane(16, 12, 0.42));
            ctx.assert(out && out.length === 12 && out[0].length === 16,
              `expected a 16×12 energy map, got ${out && out.length}×${out && out[0] && out[0].length}`);
            const hint = nanHint(out);
            for (let y = 0; y < 12; y++) {
              for (let x = 0; x < 16; x++) {
                ctx.assertClose(out[y][x], 0, 1e-4, hint ||
                  `cell [${y}][${x}] of a perfectly flat field — nothing changes anywhere, so every energy is 0, borders included`);
              }
            }
          },
        },
        {
          name: 'a vertical edge lights up — and lights up at the border too',
          run: async ctx => {
            const energy = findDynamicKernel(ctx);
            ctx.assert(energy, 'no dynamicOutput energy kernel found');
            // a step from 0 to 1 between columns 0 and 1: the edge sits ON the
            // border, so a kernel that skips the border scores it 0
            const plane = flatPlane(8, 6, 1);
            for (let y = 0; y < 6; y++) plane[y][0] = 0;
            energy.setOutput([8, 6]);
            const out = await energy(plane);
            const expected = energyOf(plane);
            const hint = nanHint(out) || diagnoseGrid(out, expected, 2e-3, [
              [(y, x) => expected[y][x] * expected[y][x],
                'those are squared magnitudes — the energy is the LENGTH of the gradient vector, so Math.sqrt(gx * gx + gy * gy)'],
            ]);
            // only say "the border" when the border is what is wrong
            const edge = !hint && borderOnly(out, expected, 2e-3);
            for (let y = 0; y < 6; y++) {
              for (let x = 0; x < 8; x++) {
                ctx.assertClose(out[y][x], expected[y][x], 2e-3, hint ||
                  (edge
                    ? `cell [${y}][${x}] — every cell in the middle is right and only the border is not, which is the clamp: Math.max(x - 1, 0) and Math.min(x + 1, this.output.x - 1), the same on y. A seam may run right down the edge of the picture, so column 0 needs a price like every other column`
                    : `cell [${y}][${x}]`));
              }
            }
          },
        },
        {
          name: 'the real picture: <code>√(gx² + gy²)</code> everywhere',
          run: async ctx => {
            const energy = findDynamicKernel(ctx);
            ctx.assert(energy, 'no dynamicOutput energy kernel found');
            const gray = fixtures(ctx.utils).gray;
            energy.setOutput([W, H]);
            const out = await energy(gray);
            const expected = energyOf(gray);
            const hint = nanHint(out) || diagnoseGrid(out, expected, 3e-3, [
              [(y, x) => expected[y][x] * expected[y][x],
                'those are squared magnitudes — take Math.sqrt(gx * gx + gy * gy)'],
            ]);
            for (const [y, x] of [[0, 0], [0, 63], [35, 0], [20, 45], [34, 92], [71, 127]]) {
              ctx.assertClose(out[y][x], expected[y][x], 3e-3, hint || `cell [${y}][${x}]`);
            }
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            const energy = findDynamicKernel(ctx);
            ctx.assert(energy, 'no dynamicOutput energy kernel found');
            const gray = fixtures(ctx.utils).gray;
            energy.setOutput([W, H]);
            const out = await energy(gray);
            const expected = energyOf(gray);
            const hint = nanHint(out) || diagnoseGrid(out, expected, 3e-3, [
              [(y, x) => expected[y][x] * expected[y][x], 'those are squared magnitudes — take the square root'],
            ]);
            const edge = !hint && borderOnly(out, expected, 3e-3);
            for (let y = 0; y < H; y++) {
              for (let x = 0; x < W; x++) {
                ctx.assertClose(out[y][x], expected[y][x], 3e-3, hint ||
                  (edge
                    ? `cell [${y}][${x}] — the whole interior is right and only the outermost row and column are not, which is the clamp: Math.max(x - 1, 0) and Math.min(x + 1, this.output.x - 1), and the same pair on y`
                    : `cell [${y}][${x}]`));
              }
            }
          },
        },
        {
          name: 'private test #2',
          run: async ctx => {
            // The same kernel at a NARROWER width: this is the run that a
            // hard-coded 127 gets wrong, and the one every later task depends
            // on, because the picture is a different width on every removal.
            const energy = findDynamicKernel(ctx);
            ctx.assert(energy, 'no dynamicOutput energy kernel found');
            const gray = fixtures(ctx.utils).gray;
            const narrow = gray.map(row => row.slice(0, 96));
            energy.setOutput([96, H]);
            const out = await energy(narrow);
            const expected = energyOf(narrow);
            // A hard-coded clamp is right at 128 and wrong here, and wrong in
            // exactly one place: the last column, the only one whose x + 1 the
            // clamp exists to catch. What it reads instead is off the end of
            // the argument and differs per backend, so the probe cannot predict
            // a VALUE — it recognises the SHAPE of the damage instead, and only
            // speaks when every other column came out right.
            const lastColumnOnly = (() => {
              let lastWrong = false;
              for (let y = 0; y < H; y++) {
                for (let x = 0; x < 96; x++) {
                  const bad = !out[y] || !(Math.abs(out[y][x] - expected[y][x]) <= 3e-3);
                  if (bad && x < 95) return false;
                  if (bad) lastWrong = true;
                }
              }
              return lastWrong;
            })();
            const hint = nanHint(out) || (lastColumnOnly
              ? 'every column but the last one is right — so the clamp is using a width this launch does not have. this.output.x is the width of THIS call (96 here); a hard-coded 127 points one past the end of a picture that has already had seams taken out of it'
              : null);
            for (let y = 0; y < H; y++) {
              for (let x = 0; x < 96; x++) {
                ctx.assertClose(out[y][x], expected[y][x], 3e-3, hint || `cell [${y}][${x}] at width 96`);
              }
            }
          },
        },
      ],
    },

    // ------------------------------------------------------------------ 2
    {
      slug: 'cumulative-cost',
      title: 'The Cheapest Path Down: One Launch per Row',
      intro: `<p>A seam is a path: one pixel per row, and from row to row it may step at most one
        column left or right. The cheapest such path is a two-line dynamic program. Let
        <code>M[y][x]</code> be the price of the cheapest seam that <em>ends</em> at pixel
        <code>(x, y)</code>:</p>
<pre><code>M[0][x] = e[0][x]
M[y][x] = e[y][x] + min( M[y-1][x-1], M[y-1][x], M[y-1][x+1] )</code></pre>
        <p>Look at what that recurrence does and does not say. Cell <code>[y][x]</code> depends on
        row <code>y - 1</code> and on nothing else in its own row — so <strong>every cell of a row
        is independent of every other cell of that row</strong>, while the rows themselves are
        strictly ordered. That is the whole trick: one kernel launch per row, 128 threads wide and
        71 launches deep — row 0 is free, because nothing is above it. Parallel across, sequential
        down. (It is the same wavefront
        <em>Wavefronts: Aligning DNA on the Diagonal</em> finds along an anti-diagonal — there the
        independent set has to be dug out; here the rows hand it to you.)</p>
        <p>One wrinkle worth knowing: each launch's output is the next launch's input, so the
        kernel must hand back a <em>fresh</em> buffer every call rather than recycling one.
        <code>immutable: true</code> is that promise, and every ping-ponging kernel in this course
        carries it.</p>`,
      goal: `<strong>Goal:</strong> write <code>step</code> — one row of the recurrence — and drive
        it down the picture, one awaited launch per row.`,
      requirements: [
        'Cell <code>x</code> takes the smallest of <code>prev[x - 1]</code>, <code>prev[x]</code> and <code>prev[x + 1]</code>, skipping the ones that fall off the ends',
        'Add this row\'s own energy: <code>eRow[x] + best</code>',
        'Drive it in JavaScript: one <code>await step(...)</code> per row, in order — row <em>y</em> needs row <em>y - 1</em>\'s answer, so never fire them together',
        '<code>plot</code> the finished bottom row so the cost across the picture is visible',
      ],
      hints: [
        {
          title: 'Hint 1 — guarding the two ends',
          body: `<p>Column 0 has no <code>x - 1</code> and the last column has no <code>x + 1</code>.
            Start from the cell directly above — which always exists — and only fold in the
            diagonals when they do:</p>
<pre><code>let best = prev[x];
if (x &gt; 0) best = Math.min(best, prev[x - 1]);
if (x + 1 &lt; this.output.x) best = Math.min(best, prev[x + 1]);</code></pre>`,
        },
        {
          title: 'Hint 2 — the driver',
          body: `<p>Row 0 is free: nothing is above it, so its cumulative cost <em>is</em> its
            energy. After that it is one launch per row:</p>
<pre><code>for (let y = 1; y &lt; energy.length; y++) {
  rows.push(await step(energy[y], rows[y - 1]));
}</code></pre>
<p>Await each one before launching the next. Fire them all at once and every row after the
            first reads a promise instead of a row.</p>`,
        },
      ],
      transfer: `"Find the axis along which the cells are independent, then launch once per step
        along the other one" is the whole wavefront family: CUDA's dynamic-programming samples,
        the banded Smith–Waterman kernels in bioinformatics, and WebGPU compute passes separated
        by a barrier all look like this. The launch count is the depth of the dependency chain,
        and that is the number you optimise.`,
      starterCode: `// One launch per row. Across a row, nothing depends on anything.
const gpu = new GPU({ mode });

const step = gpu.createKernel(function (eRow, prev) {
  const x = this.thread.x;
  // TODO 1: the cheapest of the three cells above — prev[x - 1], prev[x],
  //         prev[x + 1] — with the two edge columns guarded.
  let best = prev[x];
  // TODO 2: add this row's own energy, eRow[x], and return it.
  return best;
}, {
  output: [128],
  immutable: true,        // each call's output is the next call's input
  dynamicOutput: true,
  dynamicArguments: true,
});

// Row 0 has nothing above it, so its cumulative cost is its energy. Start
// from a Float32Array: gpu.js locks an argument's type on the first call,
// and that is what every launch hands back.
const rows = [Float32Array.from(energy[0])];

// TODO 3: one awaited launch per remaining row.
// for (let y = 1; y < energy.length; y++) { … }

const bottom = rows[rows.length - 1];
console.log('cheapest seam costs', Math.min(...bottom), '· dearest', Math.max(...bottom));
plot(bottom, { title: 'cumulative cost of the bottom row', xLabel: 'column' });
`,
      solutionCode: `// One launch per row. Across a row, nothing depends on anything.
const gpu = new GPU({ mode });

const step = gpu.createKernel(function (eRow, prev) {
  const x = this.thread.x;
  let best = prev[x];
  if (x > 0) best = Math.min(best, prev[x - 1]);
  if (x + 1 < this.output.x) best = Math.min(best, prev[x + 1]);
  return eRow[x] + best;
}, {
  output: [128],
  immutable: true,
  dynamicOutput: true,
  dynamicArguments: true,
});

const rows = [Float32Array.from(energy[0])];
for (let y = 1; y < energy.length; y++) {
  rows.push(await step(energy[y], rows[y - 1]));
}

const bottom = rows[rows.length - 1];
console.log('cheapest seam costs', Math.min(...bottom), '· dearest', Math.max(...bottom));
plot(bottom, { title: 'cumulative cost of the bottom row', xLabel: 'column' });
`,
      inputs: utils => ({ energy: fixtures(utils).energy }),
      publicTests: [
        {
          name: 'one rung of the recurrence, on a hand-checkable row',
          run: async ctx => {
            ctx.assert(ctx.kernels.length >= 1, 'no kernel was created — call gpu.createKernel()');
            const step = ctx.kernel;
            ctx.assert(step.kernel && step.kernel.dynamicOutput,
              'the step kernel needs dynamicOutput: true (and dynamicArguments) — the picture narrows later and the same kernel has to run at every width');
            const prev = [5, 1, 9, 4, 7];
            const eRow = [10, 20, 30, 40, 50];
            step.setOutput([5]);
            // `prev` goes in as a Float32Array because that is what the
            // driving loop feeds it — gpu.js's WebGL backend refuses a plain
            // array where a typed one was compiled in, however identical the
            // numbers are.
            const out = await step(eRow, Float32Array.from(prev));
            ctx.assert(out && out.length === 5, `expected 5 values, got ${out && out.length}`);
            const expected = [11, 21, 31, 44, 54];
            const hint = diagnoseRow(Array.from(out), expected, 1e-3, [
              [i => expected[i] - eRow[i],
                "the row's own energy never got added — the cell is eRow[x] + the cheapest parent"],
              [i => eRow[i] + prev[i],
                'only the cell directly above was consulted — a seam may also step one column left or right, so the parent is the cheapest of three'],
              [i => eRow[i] + Math.max(...[prev[i - 1], prev[i], prev[i + 1]].filter(v => v !== undefined)),
                'that is the DEAREST of the three parents — the whole point is the cheapest one, so Math.min'],
            ]);
            for (let i = 0; i < 5; i++) {
              ctx.assertClose(out[i], expected[i], 1e-3, hint || `cell ${i}`);
            }
          },
        },
        {
          name: 'the two edge columns have only two parents, not three',
          run: async ctx => {
            const step = ctx.kernel;
            // prev[0] is the dearest cell, so a kernel that reads prev[-1]
            // (undefined → NaN, or whatever is next in the texture) shows up
            const prev = [100, 1, 2, 3, 100];
            const eRow = [0, 0, 0, 0, 0];
            step.setOutput([5]);
            const out = await step(eRow, Float32Array.from(prev));
            const expected = [1, 1, 1, 2, 3];
            // A missing guard is only worth naming when the numbers point at
            // it: the two ends wrong and everything between them right. On the
            // CPU backend an unguarded lookup is a NaN and says so; on a GPU
            // one it is whatever was next in the buffer, which is why the shape
            // of the damage is the tell rather than any particular value.
            const endsAreTheProblem = endsOnly(Array.from(out), expected, 1e-3);
            for (let i = 0; i < 5; i++) {
              ctx.assertClose(out[i], expected[i], 1e-3,
                Number.isFinite(out[i])
                  ? (endsAreTheProblem
                    ? `cell ${i} — every cell in the middle is right and only the two ends are not. Cell 0 has no prev[x - 1] and the last cell has no prev[x + 1]: guard those two lookups with x > 0 and x + 1 < this.output.x rather than reading off the end of the row`
                    : `cell ${i}`)
                  : `cell ${i} came back NaN — the lookup ran off the end of the row. Guard it: only fold in prev[x - 1] when x > 0, and prev[x + 1] when x + 1 < this.output.x`);
            }
          },
        },
        {
          name: 'the whole picture: the plotted bottom row matches the recurrence',
          run: async ctx => {
            const expected = fixtures(ctx.utils).cost[H - 1];
            const got = plottedSeries(ctx, s => s.values.length === W);
            ctx.assert(got,
              'no plotted series of 128 values found — plot(bottom, …) the finished bottom row, which is also how this test reads your answer');
            const hint = diagnoseRow(got, expected, 5e-3, [
              [i => fixtures(ctx.utils).energy[0][i],
                'the plotted row is row 0 of the ENERGY map — the driving loop never ran, so `rows` still holds only the row you seeded it with'],
              [i => fixtures(ctx.utils).energy[H - 1][i],
                'the plotted row is the bottom row of the ENERGY map, not the cumulative one — nothing was accumulated on the way down'],
            ]);
            for (let i = 0; i < W; i++) {
              ctx.assertClose(got[i], expected[i], 5e-3, hint || `bottom-row cost at column ${i}`);
            }
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            // Drive the learner's kernel down a small energy map of our own and
            // compare every cell of every row, not just the last.
            const step = ctx.kernel;
            const rand = ctx.utils.seededRandom(4242);
            const w = 24;
            const h = 15;
            const e = new Array(h).fill(0).map(() => new Array(w).fill(0).map(() => Math.round(rand() * 900) / 100));
            const expected = cumulativeOf(e);
            step.setOutput([w]);
            const rows = [Float32Array.from(e[0])];
            for (let y = 1; y < h; y++) rows.push(await step(e[y], rows[y - 1]));
            for (let y = 1; y < h; y++) {
              const hint = diagnoseRow(Array.from(rows[y]), expected[y], 2e-3, [
                [x => e[y][x] + expected[y - 1][x],
                  'only the cell directly above was consulted — the parent is the cheapest of prev[x - 1], prev[x] and prev[x + 1]'],
              ]);
              for (let x = 0; x < w; x++) {
                ctx.assertClose(rows[y][x], expected[y][x], 2e-3, hint || `cell [${y}][${x}]`);
              }
            }
          },
        },
        {
          name: 'private test #2',
          run: async ctx => {
            // …and at a different width, because task 5 re-runs this same
            // kernel 32 times, one column narrower each time.
            const step = ctx.kernel;
            const energy = fixtures(ctx.utils).energy.map(row => row.slice(0, 90));
            const expected = cumulativeOf(energy);
            step.setOutput([90]);
            const rows = [Float32Array.from(energy[0])];
            for (let y = 1; y < H; y++) rows.push(await step(energy[y], rows[y - 1]));
            for (let x = 0; x < 90; x++) {
              ctx.assertClose(rows[H - 1][x], expected[H - 1][x], 5e-3, `bottom-row cost at column ${x}, width 90`);
            }
          },
        },
      ],
    },

    // ------------------------------------------------------------------ 3
    {
      slug: 'read-the-seam',
      title: 'Reading the Seam Back Out',
      intro: `<p>The cost map now knows the price of every seam: <code>cost[71][x]</code> is what
        the cheapest seam ending at column <code>x</code> costs. What it does not contain is the
        seam. To get that you start at the cheapest cell of the bottom row and walk
        <em>upwards</em>, at each step moving to the cheapest of the (at most) three cells you
        could have come from.</p>
        <p>Here is the part worth saying out loud: <strong>this walk does not go on the GPU.</strong>
        It is 72 steps, each of which reads three numbers and picks one, and each step needs the
        answer to the one before it. A kernel launch costs more than the whole walk does. Knowing
        which part of an algorithm to leave on the host is not a compromise — it is the skill. (The
        same is true of the traceback in <em>Wavefronts: Aligning DNA on the Diagonal</em>: the
        expensive half is parallel, the cheap half is a loop.)</p>
        <p>The picture also has flat regions, and flat regions have ties. There is usually more than
        one cheapest seam; any of them is a correct answer, and the tests below check that the seam
        you produce is <em>optimal</em>, not that it is one particular optimum.</p>`,
      goal: `<strong>Goal:</strong> write <code>backtrack(cost)</code> so it returns one column
        index per row, top to bottom, tracing the cheapest seam — then <code>plot</code> it.`,
      requirements: [
        'Start at the column of the smallest value in the LAST row of <code>cost</code>',
        'Walking up, from column <code>x</code> in row <code>y + 1</code> the seam can only have come from <code>x - 1</code>, <code>x</code> or <code>x + 1</code> in row <code>y</code> — take the cheapest that exists',
        'Return an array of <code>cost.length</code> column indices, one per row',
        '<code>plot(seam, …)</code> — it draws the path, and it is how the tests read your answer',
      ],
      hints: [
        {
          title: 'Hint 1 — where the walk starts',
          body: `<p>The bottom row holds the finished prices, so the cheapest seam is the one that
            ends at its smallest entry:</p>
<pre><code>let x = 0;
for (let i = 1; i &lt; w; i++) if (cost[h - 1][i] &lt; cost[h - 1][x]) x = i;</code></pre>`,
        },
        {
          title: 'Hint 2 — the step upwards',
          body: `<p>Exactly the mirror of the recurrence that built the map — the same window of
            three, the same two guards:</p>
<pre><code>let best = x;
if (x &gt; 0 &amp;&amp; cost[y][x - 1] &lt; cost[y][best]) best = x - 1;
if (x + 1 &lt; w &amp;&amp; cost[y][x + 1] &lt; cost[y][best]) best = x + 1;
x = best;</code></pre>
<p>Only cells within one column of where you already are — that is what makes the result a
            connected seam rather than 72 unrelated minima.</p>`,
        },
        {
          title: 'Hint 3 — direction',
          body: `<p>The loop runs <code>for (let y = h - 2; y &gt;= 0; y--)</code>. Walking the other
            way looks plausible and is wrong: the cumulative map was <em>built</em> downwards, so
            only the bottom row holds finished prices. Row 0's numbers are raw energies.</p>`,
        },
      ],
      transfer: `Every dynamic program ends this way: a parallel fill and a serial traceback. CUDA
        DP kernels return the score matrix and walk it on the host; production Smith–Waterman
        implementations do the same, and even hand back only the score when the alignment is not
        needed. The lesson generalises past DP — if a step is O(n) with a serial dependency and the
        fill was O(n²) in parallel, the launch overhead alone decides where it belongs.`,
      starterCode: `// The cost map knows the price. The seam has to be read back out of it —
// on the host, because 72 dependent steps is not a job for a kernel.

function backtrack(cost) {
  const h = cost.length;
  const w = cost[0].length;
  const seam = new Array(h).fill(0);

  // TODO 1: find the column of the cheapest cell in the LAST row of cost,
  //         and record it as this seam's bottom end.
  let x = 0;
  seam[h - 1] = x;

  // TODO 2: walk upwards, row by row. From column x you could only have come
  //         from x - 1, x or x + 1 in the row above — take the cheapest of
  //         those that exist, and record it in seam[y].

  return seam;
}

const seam = backtrack(cost);

// The plot IS the seam: column against row, wandering down the picture.
plot(seam, { title: 'the cheapest seam: column by row', xLabel: 'row' });
console.log('ends at column', seam[seam.length - 1],
  '· total energy', cost[cost.length - 1][seam[seam.length - 1]]);
`,
      solutionCode: `// The cost map knows the price. The seam has to be read back out of it —
// on the host, because 72 dependent steps is not a job for a kernel.

function backtrack(cost) {
  const h = cost.length;
  const w = cost[0].length;
  const seam = new Array(h).fill(0);

  let x = 0;
  for (let i = 1; i < w; i++) if (cost[h - 1][i] < cost[h - 1][x]) x = i;
  seam[h - 1] = x;

  for (let y = h - 2; y >= 0; y--) {
    let best = x;
    if (x > 0 && cost[y][x - 1] < cost[y][best]) best = x - 1;
    if (x + 1 < w && cost[y][x + 1] < cost[y][best]) best = x + 1;
    x = best;
    seam[y] = x;
  }

  return seam;
}

const seam = backtrack(cost);

plot(seam, { title: 'the cheapest seam: column by row', xLabel: 'row' });
console.log('ends at column', seam[seam.length - 1],
  '· total energy', cost[cost.length - 1][seam[seam.length - 1]]);
`,
      inputs: utils => ({ cost: fixtures(utils).cost }),
      publicTests: [
        {
          name: 'a seam is one column per row, and it is connected',
          run: async ctx => {
            const seam = plottedSeries(ctx, s => s.values.length === H);
            ctx.assert(seam,
              `no plotted series of ${H} values found — call plot(seam, …); it draws the path, and it is how this test reads your answer`);
            const cost = fixtures(ctx.utils).cost;
            for (let y = 0; y < H; y++) {
              ctx.assert(Number.isInteger(seam[y]) && seam[y] >= 0 && seam[y] < W,
                `seam[${y}] is ${seam[y]} — every entry must be a column index between 0 and ${W - 1}`);
            }
            const jumps = [];
            for (let y = 0; y + 1 < H; y++) jumps.push(Math.abs(seam[y + 1] - seam[y]));
            const worst = Math.max(...jumps);
            ctx.assert(worst <= 1,
              diagnoseSeam(seam, WALK_PROBES(cost)) ||
              `the seam jumps ${worst} columns between two rows — it may move at most one, or it is not a connected path`);
          },
        },
        {
          name: 'it ends at the cheapest cell of the bottom row',
          run: async ctx => {
            const seam = plottedSeries(ctx, s => s.values.length === H);
            ctx.assert(seam, `no plotted series of ${H} values found — call plot(seam, …)`);
            const cost = fixtures(ctx.utils).cost;
            const best = Math.min(...cost[H - 1]);
            const got = cost[H - 1][seam[H - 1]];
            ctx.assertClose(got, best, 1e-6,
              diagnoseSeam(seam, WALK_PROBES(cost)) ||
              `the seam ends at column ${seam[H - 1]}, which costs ${got.toFixed(3)}; the cheapest ending costs ${best.toFixed(3)}`);
          },
        },
        {
          name: 'every step upwards took the cheapest available parent',
          run: async ctx => {
            const seam = plottedSeries(ctx, s => s.values.length === H);
            ctx.assert(seam, `no plotted series of ${H} values found — call plot(seam, …)`);
            const cost = fixtures(ctx.utils).cost;
            const walk = diagnoseSeam(seam, WALK_PROBES(cost));
            for (let y = H - 2; y >= 0; y--) {
              const from = seam[y + 1];
              let best = Infinity;
              for (let x = Math.max(0, from - 1); x <= Math.min(W - 1, from + 1); x++) {
                best = Math.min(best, cost[y][x]);
              }
              ctx.assertClose(cost[y][seam[y]], best, 1e-6, walk ||
                `at row ${y} the seam sits at column ${seam[y]} (cost ${cost[y][seam[y]].toFixed(3)}), but the cheapest cell it could have come from costs ${best.toFixed(3)}`);
            }
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            // The complete statement of correctness, in one line: a connected
            // seam whose bottom cell is a global minimum of the last row IS an
            // optimal seam, whichever of the tied optima it happens to be.
            const seam = plottedSeries(ctx, s => s.values.length === H);
            ctx.assert(seam, `no plotted series of ${H} values found — call plot(seam, …)`);
            const cost = fixtures(ctx.utils).cost;
            const energy = fixtures(ctx.utils).energy;
            let total = 0;
            for (let y = 0; y < H; y++) total += energy[y][seam[y]];
            ctx.assertClose(total, Math.min(...cost[H - 1]), 2e-3,
              diagnoseSeam(seam, WALK_PROBES(cost)) ||
              'the energies along the seam do not add up to its cumulative price — the path recorded is not the path the cost map priced');
          },
        },
        {
          name: 'private test #2',
          run: async ctx => {
            // The total is logged, so a learner can see the number the next
            // task's curve is made of.
            const cost = fixtures(ctx.utils).cost;
            const best = Math.min(...cost[H - 1]);
            const nums = loggedNumbers(ctx.logs);
            ctx.assert(nums.some(v => Math.abs(v - best) <= 2e-3),
              `log the seam's total energy — expected to see ≈${best.toFixed(3)} in the console output`);
          },
        },
      ],
    },

    // ------------------------------------------------------------------ 4
    {
      slug: 'carve-one-seam',
      title: 'Take It Out, Let the Picture Close Up',
      intro: `<p>Deleting the seam is a splice on a CPU: for each row, remove one element and let
        the rest shuffle down. On a GPU there is no splice. A thread writes its own output cell and
        nothing else — it cannot push its neighbour along, which is the "no scatter" rule
        <em>Thinking in Parallel</em> makes a whole module of.</p>
        <p>So turn the question round, which is what a gather always is. The output is one column
        narrower than the input. The thread that owns output cell <code>(x, y)</code> asks: which
        input pixel belongs <em>here</em>? Everything left of the seam has not moved. Everything
        from the seam rightwards has slid one column left, so it comes from
        <code>x + 1</code>. One <code>if</code>, no shuffling, and every row does its own thing at
        the same time even though every row's seam is at a different column.</p>
        <p>The output being narrower than the input is the reason for <code>dynamicOutput</code> —
        <code>carve.setOutput([w - 1, 72])</code> before each call, and the same kernel keeps
        working all the way down.</p>`,
      goal: `<strong>Goal:</strong> write <code>carve</code> so its output is <code>plane</code>
        with the seam pixel removed from every row, and everything to its right pulled one column
        left.`,
      requirements: [
        'Read the seam position for THIS row: <code>seam[this.thread.y]</code> — every row removes a different column',
        'Output cell <code>x</code> comes from <code>plane[y][x]</code> when <code>x &lt; seam[y]</code>, and from <code>plane[y][x + 1]</code> otherwise',
        'Keep the output one column narrower than the input',
      ],
      hints: [
        {
          title: 'Hint 1 — which input pixel is mine?',
          body: `<p>Say the seam is at column 3 in this row. Output cells 0, 1, 2 are input cells
            0, 1, 2 — nothing moved. Output cell 3 is input cell <strong>4</strong>: input cell 3
            was the seam and is gone. Output cell 4 is input 5, and so on.</p>`,
        },
        {
          title: 'Hint 2 — the whole kernel',
          body: `<pre><code>const x = this.thread.x;
const y = this.thread.y;
if (x &lt; seam[y]) return plane[y][x];
return plane[y][x + 1];</code></pre>
<p>Note the strict <code>&lt;</code>. With <code>&lt;=</code> the seam pixel survives and its
            right-hand neighbour is deleted instead — the picture still narrows by one, so the
            shapes all check out and the wrong pixel is gone.</p>`,
        },
      ],
      transfer: `Compaction by gather is the standard GPU answer to "remove some elements": a
        thread computes where its data comes from rather than where it goes, because destinations
        collide and sources never do. <em>Stream Compaction</em> builds the general version with a
        prefix sum; here the geometry hands you the offset for free — it is 0 or 1, decided by one
        comparison. CUDA's <code>thrust::remove_if</code> and WebGPU compaction passes are the
        same shape underneath.`,
      starterCode: `// No splice on a GPU. Ask where each output pixel COMES FROM.
const gpu = new GPU({ mode });

const carve = gpu.createKernel(function (plane, seam) {
  const x = this.thread.x;
  const y = this.thread.y;
  // TODO: which input pixel belongs in output cell (x, y)? Everything left
  // of this row's seam has not moved; everything from the seam rightwards
  // came from one column further right.
  return plane[y][x];
}, {
  output: [127, 72],
  immutable: true,
  dynamicOutput: true,
  dynamicArguments: true,
});

const narrower = await carve(plane, seam);

console.log(plane[0].length, 'columns in ·', narrower[0].length, 'columns out');
console.log('row 0 before:', plane[0]);
console.log('row 0 after: ', narrower[0]);
`,
      solutionCode: `// No splice on a GPU. Ask where each output pixel COMES FROM.
const gpu = new GPU({ mode });

const carve = gpu.createKernel(function (plane, seam) {
  const x = this.thread.x;
  const y = this.thread.y;
  if (x < seam[y]) return plane[y][x];
  return plane[y][x + 1];
}, {
  output: [127, 72],
  immutable: true,
  dynamicOutput: true,
  dynamicArguments: true,
});

const narrower = await carve(plane, seam);

console.log(plane[0].length, 'columns in ·', narrower[0].length, 'columns out');
console.log('row 0 before:', plane[0]);
console.log('row 0 after: ', narrower[0]);
`,
      inputs: utils => ({ plane: fixtures(utils).gray, seam: fixtures(utils).seam }),
      publicTests: [
        {
          name: 'a five-column row loses exactly its seam pixel',
          run: async ctx => {
            ctx.assert(ctx.kernels.length >= 1, 'no kernel was created — call gpu.createKernel()');
            const carve = ctx.kernel;
            const plane = [
              [10, 11, 12, 13, 14],
              [20, 21, 22, 23, 24],
              [30, 31, 32, 33, 34],
            ];
            const seam = [2, 0, 4];
            carve.setOutput([4, 3]);
            const out = await carve(plane, seam);
            ctx.assert(out && out.length === 3 && out[0].length === 4,
              `expected a 4×3 result, got ${out && out.length} rows of ${out && out[0] && out[0].length}`);
            const expected = carveOf(plane, seam);
            const offByOne = carveOf(plane, seam.map(s => s + 1));
            const inverted = plane.map((row, y) =>
              row.slice(0, 4).map((v, x) => (x < seam[y] ? row[x + 1] : row[x])));
            const oneColumn = plane.map(row => row.slice(0, 4).map((v, x) => (x < seam[0] ? row[x] : row[x + 1])));
            const hint = diagnoseGrid(out, expected, 1e-3, [
              [(y, x) => offByOne[y][x],
                'the seam pixel itself survived and its right-hand neighbour went instead — the comparison is a strict x < seam[y]'],
              [(y, x) => inverted[y][x],
                'the two branches are the wrong way round: pixels LEFT of the seam have not moved, pixels from the seam rightwards come from x + 1'],
              [(y, x) => oneColumn[y][x],
                'every row removed the same column — the seam is a different column in each row, so index it with this.thread.y'],
            ]);
            for (let y = 0; y < 3; y++) {
              for (let x = 0; x < 4; x++) {
                ctx.assertClose(out[y][x], expected[y][x], 1e-3, hint || `cell [${y}][${x}]`);
              }
            }
          },
        },
        {
          name: 'the real picture: one column narrower, everything else intact',
          run: async ctx => {
            const carve = ctx.kernel;
            const { gray, seam } = fixtures(ctx.utils);
            carve.setOutput([W - 1, H]);
            const out = await carve(gray, seam);
            ctx.assert(out && out.length === H && out[0].length === W - 1,
              `expected a ${W - 1}×${H} result, got ${out && out.length} rows of ${out && out[0] && out[0].length}`);
            const expected = carveOf(gray, seam);
            const oneColumn = gray.map(row => row.slice(0, W - 1).map((v, x) => (x < seam[0] ? row[x] : row[x + 1])));
            const hint = diagnoseGrid(out, expected, 1e-4, [
              [(y, x) => oneColumn[y][x],
                'every row removed the same column — that is a straight cut, not a seam. The column to drop is seam[this.thread.y]'],
            ]);
            for (const y of [0, 1, 20, 35, 36, 70, 71]) {
              for (let x = 0; x < W - 1; x++) {
                ctx.assertClose(out[y][x], expected[y][x], 1e-4, hint || `cell [${y}][${x}]`);
              }
            }
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            const carve = ctx.kernel;
            const { gray, seam } = fixtures(ctx.utils);
            carve.setOutput([W - 1, H]);
            const out = await carve(gray, seam);
            const expected = carveOf(gray, seam);
            for (let y = 0; y < H; y++) {
              for (let x = 0; x < W - 1; x++) {
                ctx.assertClose(out[y][x], expected[y][x], 1e-4, `cell [${y}][${x}]`);
              }
            }
          },
        },
        {
          name: 'private test #2',
          run: async ctx => {
            // Twice in a row, at two different widths — which is what the
            // payoff task does thirty-two times.
            const carve = ctx.kernel;
            const { gray, energy } = fixtures(ctx.utils);
            const seamA = fixtures(ctx.utils).seam;
            carve.setOutput([W - 1, H]);
            const once = await carve(gray, seamA);
            const narrowedEnergy = carveOf(energy, seamA);
            const seamB = backtrackOf(cumulativeOf(narrowedEnergy));
            carve.setOutput([W - 2, H]);
            const twice = await carve(once, seamB);
            const expected = carveOf(carveOf(gray, seamA), seamB);
            ctx.assert(twice && twice[0].length === W - 2,
              `expected ${W - 2} columns after two removals, got ${twice && twice[0] && twice[0].length}`);
            for (let y = 0; y < H; y++) {
              for (let x = 0; x < W - 2; x++) {
                ctx.assertClose(twice[y][x], expected[y][x], 1e-4, `cell [${y}][${x}] after two removals`);
              }
            }
          },
        },
      ],
    },

    // ------------------------------------------------------------------ 5
    {
      slug: 'carve-many',
      title: 'Payoff: Thirty-Two Seams',
      intro: `<p>One seam is a curiosity. Thirty-two is a resize. And the loop has a catch that
        matters: once a seam is gone the picture is a <em>different picture</em>, so the energy map
        and the cost map both have to be built again from the carved luminance — not from the
        original. Energy → cost → seam → carve, and round again.</p>
        <p>Everything you wrote is here already. What is left is the orchestration, and the order
        is the whole thing: each stage awaits the one before it, and inside the cost map the rows
        must go one at a time, because row <em>y</em> reads row <em>y - 1</em>'s answer. Three
        colour planes and the luminance plane all reflow with the same seam — the carve kernel has
        no idea what it is carving, which is why one kernel does all four.</p>
        <p>Two console tricks pay for themselves here. <code>render()</code> once per removal
        collapses into a <strong>frame scrubber</strong> you can drag back and forth — that is
        where you actually see the picture reflow. And plotting each seam's energy against its
        removal number gives you the curve that tells you when to stop: the cheap seams go first,
        and when the curve knees upwards the picture has run out of things it can afford to lose.
        The slider under the console re-runs the whole program, so you can carve less and compare.</p>
        ${ARRAY_LAYOUT}`,
      goal: `<strong>Goal:</strong> fill in <code>carveOnce</code> — energy, cumulative cost,
        backtrack, then carve every plane with that one seam — and return the new planes together
        with the seam's price.`,
      requirements: [
        'Build the energy map from the CURRENT luminance plane (<code>planes[3]</code>) at the current width — <code>energy.setOutput([w, 72])</code>',
        'Build the cumulative rows with one awaited <code>step</code> launch per row, in order',
        '<code>backtrack</code> the rows to a seam, and take its price from the bottom row',
        'Carve <strong>every</strong> plane with that seam at <code>[w - 1, 72]</code>, and return the new planes plus the cost',
      ],
      hints: [
        {
          title: 'Hint 1 — the current width',
          body: `<p>Nothing needs to be tracked by hand: the planes know how wide they are.</p>
<pre><code>const w = planes[0][0].length;
energy.setOutput([w, 72]);
step.setOutput([w]);
carve.setOutput([w - 1, 72]);</code></pre>`,
        },
        {
          title: 'Hint 2 — the cost map, again',
          body: `<p>Exactly the driver from the cumulative-cost task, over the energy map you just
            built:</p>
<pre><code>const rows = [Float32Array.from(e[0])];
for (let y = 1; y &lt; 72; y++) rows.push(await step(e[y], rows[y - 1]));</code></pre>`,
        },
        {
          title: 'Hint 3 — carving four planes with one seam',
          body: `<p>A <code>for</code> loop, not <code>.map()</code> — a callback cannot hold an
            <code>await</code>, and every one of these is a kernel call:</p>
<pre><code>const next = [];
for (let i = 0; i &lt; planes.length; i++) {
  next.push(await carve(planes[i], seam));
}
return { planes: next, cost: rows[71][seam[71]] };</code></pre>`,
        },
      ],
      transfer: `A per-frame chain of dependent passes with a host-side loop around it is what a
        real pipeline looks like on every platform: CUDA streams a sequence of launches, WebGPU
        records passes into a command encoder, Metal encodes one compute pass per stage. And the
        cost of this particular shape is visible in the numbers — 71 cost-map launches per removal,
        well over two thousand for the whole run, each one tiny. When a wavefront gets slow it is
        almost never the arithmetic; it is the launch count.`,
      starterCode: `// Thirty-two removals, and the picture reflows around what is left.
const gpu = new GPU({ mode });

// Drag me: the program is a pure function of its controls, so moving this
// re-runs the whole carve.
const seams = slider('seams to remove', { min: 20, max: 32, value: 32, step: 1 });

${CHANNEL_KERNEL}

${ENERGY_KERNEL}

${STEP_KERNEL}

${CARVE_KERNEL}

${PAINT_KERNEL}

${BACKTRACK_FN}

// One removal: energy → cumulative cost → seam → carve every plane.
async function carveOnce(planes) {
  const w = planes[0][0].length;
  // TODO 1: the energy map of the CURRENT luminance plane, planes[3], at
  //         width w. (energy.setOutput([w, 72]) first.)
  // TODO 2: the cumulative rows — one awaited step launch per row, in order.
  // TODO 3: backtrack to a seam, and read its price off the bottom row.
  // TODO 4: carve every plane in \`planes\` with that seam, at [w - 1, 72].
  return { planes, cost: 0 };
}

// r, g, b and the luminance the energy is built from: four planes that all
// have to reflow together.
let planes = [];
for (let c = 0; c < 4; c++) planes.push(await channel(photo, c));

const costs = [];
await paint(planes[0], planes[1], planes[2], planes[0][0].length);
render(paint.canvas);

for (let k = 0; k < seams; k++) {
  const out = await carveOnce(planes);
  planes = out.planes;
  costs.push(out.cost);
  // one render() per removal — consecutive ones become a frame scrubber
  await paint(planes[0], planes[1], planes[2], planes[0][0].length);
  render(paint.canvas);
}

console.log('carved down to', planes[0][0].length, 'columns');
plot(costs, { title: 'energy of each seam removed', xLabel: 'removal' });
`,
      solutionCode: `// Thirty-two removals, and the picture reflows around what is left.
const gpu = new GPU({ mode });

const seams = slider('seams to remove', { min: 20, max: 32, value: 32, step: 1 });

${CHANNEL_KERNEL}

${ENERGY_KERNEL}

${STEP_KERNEL}

${CARVE_KERNEL}

${PAINT_KERNEL}

${BACKTRACK_FN}

// One removal: energy → cumulative cost → seam → carve every plane.
async function carveOnce(planes) {
  const w = planes[0][0].length;

  energy.setOutput([w, 72]);
  const e = await energy(planes[3]);

  step.setOutput([w]);
  const rows = [Float32Array.from(e[0])];
  for (let y = 1; y < 72; y++) rows.push(await step(e[y], rows[y - 1]));

  const seam = backtrack(rows);
  const cost = rows[71][seam[71]];

  carve.setOutput([w - 1, 72]);
  const next = [];
  for (let i = 0; i < planes.length; i++) next.push(await carve(planes[i], seam));

  return { planes: next, cost };
}

let planes = [];
for (let c = 0; c < 4; c++) planes.push(await channel(photo, c));

const costs = [];
await paint(planes[0], planes[1], planes[2], planes[0][0].length);
render(paint.canvas);

for (let k = 0; k < seams; k++) {
  const out = await carveOnce(planes);
  planes = out.planes;
  costs.push(out.cost);
  await paint(planes[0], planes[1], planes[2], planes[0][0].length);
  render(paint.canvas);
}

console.log('carved down to', planes[0][0].length, 'columns');
plot(costs, { title: 'energy of each seam removed', xLabel: 'removal' });
`,
      inputs: utils => ({ photo: sceneImage(utils) }),
      publicTests: [
        {
          name: 'one rendered frame per removal — the scrubber',
          run: async ctx => {
            const costs = plottedSeries(ctx);
            ctx.assert(costs && costs.length >= 4,
              'no plotted cost curve found — plot(costs, …) after the loop, one value per removal');
            const frames = canvasFrames(ctx);
            ctx.assert(frames >= costs.length,
              `${frames} frame(s) rendered for ${costs.length} removals — call render(paint.canvas) inside the loop; consecutive renders become a scrubber you can drag`);
          },
        },
        {
          name: 'the picture is narrower by one column per seam',
          run: async ctx => {
            const costs = plottedSeries(ctx);
            ctx.assert(costs, 'no plotted cost curve found');
            const finalWidth = W - costs.length;
            const pixels = await ctx.getPixels();
            ctx.assert(pixels && pixels.length === W * H * 4,
              'no painted canvas to read — did you render(paint.canvas)?');
            ctx.assert(columnIsFrame(pixels, finalWidth),
              `column ${finalWidth} of the last frame still holds picture: after ${costs.length} removals the picture should be ${finalWidth} columns wide. Did carveOnce hand back the CARVED planes?`);
            ctx.assert(!columnIsFrame(pixels, finalWidth - 1),
              `column ${finalWidth - 1} is empty frame — the picture is narrower than ${finalWidth} columns, so more was removed than one column per seam`);
            const nums = loggedNumbers(ctx.logs);
            ctx.assert(nums.some(v => v === finalWidth),
              `log the final width — expected to see ${finalWidth} in the console output`);
          },
        },
        {
          name: 'the cheap seams go first — the curve climbs',
          run: async ctx => {
            const costs = plottedSeries(ctx);
            ctx.assert(costs && costs.length >= 20,
              `expected at least 20 seam energies in the plot, got ${costs && costs.length}`);
            const q = Math.max(1, Math.floor(costs.length / 4));
            const first = mean(costs.slice(0, q));
            const last = mean(costs.slice(-q));
            ctx.assert(costs.every(c => c > 0),
              'some seam cost 0 or less — the price of a seam is its bottom-row cumulative cost, which is a sum of energies');
            ctx.assert(last > 2 * first,
              `the first seams cost ≈${first.toFixed(2)} and the last ≈${last.toFixed(2)} — they should climb steeply once the smooth corridor is used up. A flat curve means the energy map was built once, outside the loop: after a seam comes out the picture is a different picture, so energy has to be recomputed from the CARVED luminance plane`);
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            // The whole run, in float64, host-side. Exact seams cannot be
            // compared (flat regions make the optimum genuinely non-unique),
            // but the total price of the run is stable to a few percent, and
            // it is what separates a real carve from a plausible-looking one.
            const costs = plottedSeries(ctx);
            ctx.assert(costs, 'no plotted cost curve found');
            const reference = carveRun(fixtures(ctx.utils).gray, costs.length).costs;
            const got = costs.reduce((a, b) => a + b, 0);
            const want = reference.reduce((a, b) => a + b, 0);
            const stale = reference[0] * costs.length;
            ctx.assert(Math.abs(got - want) <= 0.3 * want,
              Math.abs(got - stale) <= 0.15 * stale
                ? 'every seam cost about what the FIRST one cost — the energy map is being rebuilt from the original picture each time instead of from the carved luminance plane'
                : `the seams removed cost ${got.toFixed(1)} in total; the cheapest 32 seams of this picture cost ${want.toFixed(1)}`);
          },
        },
        {
          name: 'private test #2',
          run: async ctx => {
            // The knee is the point of the plot: it is the signal that says
            // "stop here". Without it there is nothing to read off the curve.
            const costs = plottedSeries(ctx);
            ctx.assert(costs, 'no plotted cost curve found');
            const lo = Math.min(...costs);
            const hi = Math.max(...costs);
            // `lo > 0` is not decoration: without it an all-zero curve (a
            // carveOnce that never filled in its cost) satisfies hi >= 4 * lo
            // and this test passes on nothing at all.
            ctx.assert(lo > 0 && hi >= 4 * lo,
              lo > 0
                ? `the dearest seam (${hi.toFixed(2)}) is only ${(hi / lo).toFixed(1)}× the cheapest (${lo.toFixed(2)}) — this picture has a smooth corridor that runs out, so the curve should knee sharply upwards partway through`
                : 'the cheapest seam in the curve costs 0 — a seam\'s price is its bottom-row cumulative cost, rows[71][seam[71]], and every energy in it is positive');
          },
        },
      ],
    },

    // ------------------------------------------------------------------ 6
    {
      slug: 'protect-the-face',
      title: 'What It Does Badly, and the Fix Everybody Ships',
      intro: `<p>Drag the last task's scrubber slowly and watch the face. The eyes creep together,
        the head goes oval, and by the end it is a different person: every row of it wide enough to
        matter loses exactly seven columns, so its widest row comes back thirty columns instead of
        thirty-seven — and not the same seven in every row (twenty-eight different columns of the
        face are cut somewhere), so its features stop lining up vertically. That skew is the straight-line artefact seam carving is famous for,
        showing up here as a warp. Seam carving has no idea what a face is. It knows that skin is
        smooth and that smooth is cheap, so once the sky is used up the face is the next best
        bargain in the picture.</p>
        <p>The two poles, meanwhile, come through perfectly straight — every seam in this run
        happened to pass entirely to one side of them. That is luck, not a property, and it is the
        uncomfortable part: which structures survive depends on where the cheap material happens to
        be, and you cannot read it off the picture beforehand.</p>
        <p>This is why nothing ships it unattended. Every product that offers content-aware resize
        offers a brush next to it, and the brush paints a <strong>mask</strong>: a region whose
        energy gets a large constant added, so every seam routes around it. Twenty is plenty here —
        the dearest seam this picture has anywhere in the run prices at under nineteen, and the ones
        actually taken top out at fourteen, so a single protected pixel already prices a seam out of
        the neighbourhood.</p>
        <p>One trap comes free with the idea, and it is the reason the mask is carried in
        <code>planes</code> with everything else: the mask lives in <em>image</em> space. Carve the
        picture without carving the mask and the protection slides off the thing it was protecting,
        one column at a time.</p>`,
      goal: `<strong>Goal:</strong> write <code>maskedEnergy</code>, then measure the result — count
        the protected pixels that survived, and plot the protected run's costs against the
        unprotected ones.`,
      requirements: [
        '<code>maskedEnergy</code> returns the Sobel magnitude <em>plus</em> <code>this.constants.penalty * mask[y][x]</code>',
        'Count the 1s left in the carved mask (<code>planes[4]</code>) after the run and <code>console.log</code> it — it should equal what you started with',
        'Plot both curves together: <code>plot({ ... })</code> with <code>unmaskedCosts</code> and your own <code>costs</code>',
      ],
      hints: [
        {
          title: 'Hint 1 — the penalty term',
          body: `<p>The Sobel part is untouched; the mask is one more term on the end:</p>
<pre><code>return Math.sqrt(gx * gx + gy * gy)
     + this.constants.penalty * mask[y][x];</code></pre>
<p>Because the mask is 0 outside the protected region, this costs the rest of the picture
            exactly nothing.</p>`,
        },
        {
          title: 'Hint 2 — counting what survived',
          body: `<p><code>planes[4]</code> is the mask after the same 32 carves as the picture, so
            it is a plain 2D array one column narrower per removal:</p>
<pre><code>let left = 0;
for (let y = 0; y &lt; 72; y++) {
  for (let x = 0; x &lt; planes[4][y].length; x++) left += planes[4][y][x];
}</code></pre>
<p>If that number has dropped, a seam went through the face.</p>`,
        },
        {
          title: 'Hint 3 — two series in one chart',
          body: `<p><code>plot</code> takes an object of named series and draws them on the same
            axes:</p>
<pre><code>plot({ 'no mask': unmaskedCosts, 'face protected': costs },
     { title: 'what protection costs', xLabel: 'removal' });</code></pre>`,
        },
      ],
      transfer: `Weighted energy is how this is done everywhere — Photoshop's content-aware scale
        takes a protect/remove mask, and the same "add a large constant to the cost field" move
        drives graph-cut segmentation, path planning around obstacles, and every route planner that
        has ever been told to avoid motorways. The deeper lesson is the honest one: an algorithm
        that optimises a proxy will happily destroy whatever the proxy does not measure, and the
        fix is always to put the missing knowledge into the objective rather than to hope.`,
      starterCode: `// The same carve, with a brush stroke over the face.
const gpu = new GPU({ mode });

${CHANNEL_KERNEL}

// The energy map, plus one term. Everything else is task 1's kernel.
const maskedEnergy = gpu.createKernel(function (gray, mask) {
  const x = this.thread.x;
  const y = this.thread.y;
  const xm = Math.max(x - 1, 0);
  const xp = Math.min(x + 1, this.output.x - 1);
  const ym = Math.max(y - 1, 0);
  const yp = Math.min(y + 1, this.output.y - 1);
  const gx = (gray[ym][xp] + 2 * gray[y][xp] + gray[yp][xp])
           - (gray[ym][xm] + 2 * gray[y][xm] + gray[yp][xm]);
  const gy = (gray[yp][xm] + 2 * gray[yp][x] + gray[yp][xp])
           - (gray[ym][xm] + 2 * gray[ym][x] + gray[ym][xp]);
  // TODO 1: add this.constants.penalty * mask[y][x] to the magnitude, so a
  //         seam that clips the protected region prices itself out.
  return Math.sqrt(gx * gx + gy * gy);
}, {
  output: [128, 72],
  constants: { penalty: 20 },
  dynamicOutput: true,
  dynamicArguments: true,
});

${STEP_KERNEL}

${CARVE_KERNEL}

${PAINT_KERNEL}

${BACKTRACK_FN}

async function carveOnce(planes) {
  const w = planes[0][0].length;

  maskedEnergy.setOutput([w, 72]);
  const e = await maskedEnergy(planes[3], planes[4]);

  step.setOutput([w]);
  const rows = [Float32Array.from(e[0])];
  for (let y = 1; y < 72; y++) rows.push(await step(e[y], rows[y - 1]));

  const seam = backtrack(rows);
  const cost = rows[71][seam[71]];

  carve.setOutput([w - 1, 72]);
  const next = [];
  for (let i = 0; i < planes.length; i++) next.push(await carve(planes[i], seam));

  return { planes: next, cost };
}

// Five planes now: r, g, b, luminance — and the mask, which lives in image
// space and so has to reflow with everything else.
let planes = [];
for (let c = 0; c < 4; c++) planes.push(await channel(photo, c));
planes.push(faceMask);

let protectedBefore = 0;
for (let y = 0; y < 72; y++) for (let x = 0; x < 128; x++) protectedBefore += faceMask[y][x];

const costs = [];
await paint(planes[0], planes[1], planes[2], planes[0][0].length);
render(paint.canvas);

for (let k = 0; k < 32; k++) {
  const out = await carveOnce(planes);
  planes = out.planes;
  costs.push(out.cost);
  await paint(planes[0], planes[1], planes[2], planes[0][0].length);
  render(paint.canvas);
}

console.log('carved down to', planes[0][0].length, 'columns');
console.log('protected pixels before:', protectedBefore);

// TODO 2: count the 1s left in the carved mask, planes[4], and log it.
// TODO 3: plot unmaskedCosts and costs on the same axes.
`,
      solutionCode: `// The same carve, with a brush stroke over the face.
const gpu = new GPU({ mode });

${CHANNEL_KERNEL}

const maskedEnergy = gpu.createKernel(function (gray, mask) {
  const x = this.thread.x;
  const y = this.thread.y;
  const xm = Math.max(x - 1, 0);
  const xp = Math.min(x + 1, this.output.x - 1);
  const ym = Math.max(y - 1, 0);
  const yp = Math.min(y + 1, this.output.y - 1);
  const gx = (gray[ym][xp] + 2 * gray[y][xp] + gray[yp][xp])
           - (gray[ym][xm] + 2 * gray[y][xm] + gray[yp][xm]);
  const gy = (gray[yp][xm] + 2 * gray[yp][x] + gray[yp][xp])
           - (gray[ym][xm] + 2 * gray[ym][x] + gray[ym][xp]);
  return Math.sqrt(gx * gx + gy * gy)
       + this.constants.penalty * mask[y][x];
}, {
  output: [128, 72],
  constants: { penalty: 20 },
  dynamicOutput: true,
  dynamicArguments: true,
});

${STEP_KERNEL}

${CARVE_KERNEL}

${PAINT_KERNEL}

${BACKTRACK_FN}

async function carveOnce(planes) {
  const w = planes[0][0].length;

  maskedEnergy.setOutput([w, 72]);
  const e = await maskedEnergy(planes[3], planes[4]);

  step.setOutput([w]);
  const rows = [Float32Array.from(e[0])];
  for (let y = 1; y < 72; y++) rows.push(await step(e[y], rows[y - 1]));

  const seam = backtrack(rows);
  const cost = rows[71][seam[71]];

  carve.setOutput([w - 1, 72]);
  const next = [];
  for (let i = 0; i < planes.length; i++) next.push(await carve(planes[i], seam));

  return { planes: next, cost };
}

let planes = [];
for (let c = 0; c < 4; c++) planes.push(await channel(photo, c));
planes.push(faceMask);

let protectedBefore = 0;
for (let y = 0; y < 72; y++) for (let x = 0; x < 128; x++) protectedBefore += faceMask[y][x];

const costs = [];
await paint(planes[0], planes[1], planes[2], planes[0][0].length);
render(paint.canvas);

for (let k = 0; k < 32; k++) {
  const out = await carveOnce(planes);
  planes = out.planes;
  costs.push(out.cost);
  await paint(planes[0], planes[1], planes[2], planes[0][0].length);
  render(paint.canvas);
}

console.log('carved down to', planes[0][0].length, 'columns');
console.log('protected pixels before:', protectedBefore);

let protectedAfter = 0;
for (let y = 0; y < 72; y++) {
  for (let x = 0; x < planes[4][y].length; x++) protectedAfter += planes[4][y][x];
}
console.log('protected pixels after:', protectedAfter);

plot({ 'no mask': unmaskedCosts, 'face protected': costs },
  { title: 'what protection costs', xLabel: 'removal' });
`,
      inputs: utils => ({
        photo: sceneImage(utils),
        faceMask: makeMask(),
        unmaskedCosts: carveRun(fixtures(utils).gray, SEAMS).costs.map(v => Math.round(v * 1e3) / 1e3),
      }),
      // The card is this run's LAST frame — the carved picture and the band of
      // empty frame beside it — shown at ~300 px, and 128x72 is the lesson's
      // size, not the card's. Same scene, same mask, and SEAMS * CARD removals
      // so the picture narrows by the same quarter of its width: the card
      // differs from the lesson's only in how finely it is sampled.
      cardInputs: utils => ({
        photo: sceneImage(utils, CARD),
        faceMask: makeMask(CARD),
        unmaskedCosts: carveRun(fixtures(utils, CARD).gray, SEAMS * CARD).costs.map(
          v => Math.round(v * 1e3) / 1e3
        ),
      }),
      publicTests: [
        {
          name: 'the run still narrows the picture by 32 columns',
          run: async ctx => {
            const pixels = await ctx.getPixels();
            ctx.assert(pixels && pixels.length === W * H * 4,
              'no painted canvas to read — did you render(paint.canvas)?');
            ctx.assert(columnIsFrame(pixels, W - SEAMS),
              `column ${W - SEAMS} of the last frame still holds picture — after ${SEAMS} removals it should be empty frame`);
            ctx.assert(!columnIsFrame(pixels, W - SEAMS - 1),
              `column ${W - SEAMS - 1} is empty frame — more than one column per seam came out`);
          },
        },
        {
          name: 'not one protected pixel was removed',
          run: async ctx => {
            const before = makeMask().reduce((sum, row) => sum + row.reduce((a, b) => a + b, 0), 0);
            const nums = loggedNumbers(ctx.logs);
            // The prewired "before" line already put `before` in the console
            // once, so the surviving count only agrees when it shows up TWICE.
            const seen = nums.filter(v => v === before).length;
            const survivors = nums.filter(v => Number.isInteger(v) && v > before * 0.4 && v < before);
            // TWO different mistakes lose protected pixels — no penalty term at
            // all, and a mask that never got carved with the picture — and this
            // task teaches both, so guessing between them is not allowed. The
            // cost curve tells them apart: a run that costs the same as the
            // unprotected one is not paying any penalty, while a run that
            // matches early and turns dear late IS paying it and has simply let
            // the mask drift out of register with the pixels underneath.
            const unmasked = carveRun(fixtures(ctx.utils).gray, SEAMS).costs;
            const routing = allPlottedSeries(ctx).some(s =>
              s.values.length === SEAMS &&
              s.values.slice(0, 12).every((v, i) => Math.abs(v - unmasked[i]) <= 0.6) &&
              s.values.some((v, i) => v > unmasked[i] + 0.2));
            ctx.assert(
              seen >= 2,
              survivors.length
                ? (routing
                  ? `the mask started with ${before} pixels and ${Math.max(...survivors)} survived — and the penalty IS reaching the energy map, because your seams turn dearer than the unprotected ones late in the run. So it is the mask that has come loose: it lives in image space, so it has to be carved with the SAME seam as the picture, or the protection slides off the thing it was protecting, one column at a time`
                  : `the mask started with ${before} pixels and ${Math.max(...survivors)} survived — seams cut straight through the protected region. Add this.constants.penalty * mask[y][x] to the magnitude, so a seam that clips the mask prices itself out`)
                : `log how many protected pixels are left in the carved mask after the run — with the mask working it should still be ${before}`
            );
          },
        },
        {
          name: 'both curves on one chart, and protection is not free',
          run: async ctx => {
            const series = allPlottedSeries(ctx);
            ctx.assert(series.length >= 2,
              'plot both cost curves on the same axes — plot({ …: unmaskedCosts, …: costs }, …)');
            const unmasked = carveRun(fixtures(ctx.utils).gray, SEAMS).costs;
            const mine = series.find(s => s.values.length === SEAMS &&
              s.values.some((v, i) => Math.abs(v - unmasked[i]) > 0.2));
            ctx.assert(mine,
              'both plotted series look like the unmasked run — the masked carve should take a different, dearer route once the smooth corridor is gone');
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            // Routing round the face costs something, and it costs it at the
            // END — where the face was about to become the cheapest thing left.
            const series = allPlottedSeries(ctx);
            const unmasked = carveRun(fixtures(ctx.utils).gray, SEAMS).costs;
            const mine = series.find(s => s.values.length === SEAMS &&
              s.values.some((v, i) => Math.abs(v - unmasked[i]) > 0.2));
            ctx.assert(mine, 'no masked cost curve found among the plotted series');
            let dearer = 0;
            for (let i = SEAMS - 8; i < SEAMS; i++) if (mine.values[i] > unmasked[i]) dearer++;
            ctx.assert(dearer >= 6,
              `only ${dearer} of the last 8 protected seams cost more than the unprotected ones — with the face off limits the late seams have to take a dearer route`);
          },
        },
        {
          name: 'private test #2',
          run: async ctx => {
            // The mask is only worth anything if it did not also cost the
            // early seams: nothing changes until the seams reach the face.
            const series = allPlottedSeries(ctx);
            const unmasked = carveRun(fixtures(ctx.utils).gray, SEAMS).costs;
            const mine = series.find(s => s.values.length === SEAMS &&
              s.values.some((v, i) => Math.abs(v - unmasked[i]) > 0.2));
            ctx.assert(mine, 'no masked cost curve found among the plotted series');
            for (let i = 0; i < 12; i++) {
              ctx.assertClose(mine.values[i], unmasked[i], 0.6,
                `removal ${i} costs a different amount with the mask on. The first seams run down the smooth corridor, nowhere near the face, so the penalty should not touch them — is the penalty being added everywhere instead of only where mask[y][x] is 1?`);
            }
          },
        },
      ],
    },
  ],
};
