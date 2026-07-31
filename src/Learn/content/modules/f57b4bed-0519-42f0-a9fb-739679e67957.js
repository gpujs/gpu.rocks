// Module: Template Matching — uuid f57b4bed-0519-42f0-a9fb-739679e67957
// (short id f57b4bed). The file name is the uuid; identity lives in the
// exported object below, never in the path. No legacyId: this module is new.
//
// Computer Vision — Template Matching.
//
// Five tasks: sum of squared differences, one thread per candidate position →
// breaking it with a brightness change → normalized cross-correlation, which
// is immune to it → hoisting the template's statistics out of 7,921 threads →
// locating a patch and refusing one that is not there.
//
// The thesis: a naive difference score measures distance in absolute
// brightness, not similarity. Normalisation is what makes a similarity measure
// trustworthy — a lesson that outlives vision.
//
// Kernel-authoring rules (contract): no closures inside kernel functions, only
// numbers / nested number arrays as inputs, this.thread.* for indexing,
// this.constants.* for compile-time values (legal as loop bounds), 2D data
// indexed map[y][x]. Every task passes in CPU mode; the scene is 96×96 and the
// template 8×8 so verification stays quick.

import { ARRAY_LAYOUT } from '../layoutNote.js';

// ---- the scene ------------------------------------------------------------
//
// A 96×96 luminance map — one number per pixel, the kind the grayscale pass in
// Data In, Data Out (or any Sobel/blur pass in Convolution & Filters) produces
// from a photograph. Two things are planted in it:
//
//   • the TEMPLATE, verbatim, at PLANT — so SSD there is exactly 0;
//   • a DECOY: the template's photographic negative, LIFT darker than it.
//
// The decoy is what makes task 2 honest. Add LIFT to every scene value and the
// decoy's mean lands exactly where the template's was, while its shape stays
// the worst possible match — so SSD, which cannot tell brightness from shape,
// picks it over a perfect match. NCC scores it −1 and is never fooled.

const SCENE = 96; // scene is SCENE × SCENE
const PATCH = 8; // template is PATCH × PATCH
const POSITIONS = SCENE - PATCH + 1; // 89 candidate positions per axis
const COUNT = PATCH * PATCH; // 64 pixels per window
const LIFT = 0.28; // brightness added to make brightScene
const THRESHOLD = 0.9; // NCC above this counts as a detection

const PLANT = { x: 58, y: 21 }; // where the template really is
const DECOY = { x: 12, y: 63 }; // where SSD goes wrong once the scene brightens

// Three decimal places: clean in the Task inputs panel, and exact enough that
// a score computed here and a score computed in a kernel agree.
function q3(value) {
  return Math.round(value * 1000) / 1000;
}

function meanOf(grid) {
  let sum = 0;
  for (let j = 0; j < grid.length; j++) {
    for (let i = 0; i < grid[j].length; i++) sum += grid[j][i];
  }
  return sum / (grid.length * grid[0].length);
}

// The template: smooth, asymmetric (so a transposed read is detectable), mean
// ≈ 0.45, spread ≈ 0.044. Deterministic — no random numbers at all.
function makePatch() {
  const patch = new Array(PATCH);
  for (let j = 0; j < PATCH; j++) {
    const row = new Array(PATCH);
    for (let i = 0; i < PATCH; i++) {
      row[i] = q3(0.45 + 0.09 * Math.sin(0.85 * i + 0.45 * j + 0.3) * Math.cos(0.3 * j - 0.2));
    }
    patch[j] = row;
  }
  return patch;
}

// The negative: same structure, flipped and dimmed, its mean exactly LIFT
// below the template's.
function makeDecoy(patch) {
  const mean = meanOf(patch);
  const decoy = new Array(PATCH);
  for (let j = 0; j < PATCH; j++) {
    const row = new Array(PATCH);
    for (let i = 0; i < PATCH; i++) row[i] = q3(mean - LIFT - 0.8 * (patch[j][i] - mean));
    decoy[j] = row;
  }
  return decoy;
}

// The template turned a quarter turn: identical mean, identical spread,
// identical histogram — and nowhere in the scene.
function rotatePatch(patch) {
  const out = new Array(PATCH);
  for (let j = 0; j < PATCH; j++) {
    const row = new Array(PATCH);
    for (let i = 0; i < PATCH; i++) row[i] = patch[PATCH - 1 - i][j];
    out[j] = row;
  }
  return out;
}

// Textured everywhere — no window is flat, so no window has zero variance and
// NCC is defined at all 7,921 positions.
function makeScene(utils, seed = 3702, plant = PLANT, decoy = DECOY) {
  const rand = utils.seededRandom(seed);
  const scene = new Array(SCENE);
  for (let y = 0; y < SCENE; y++) {
    const row = new Array(SCENE);
    for (let x = 0; x < SCENE; x++) {
      const smooth =
        0.48 +
        0.055 * Math.sin(x / 11.3 + 0.6) * Math.cos(y / 9.7 - 0.4) +
        0.035 * Math.sin((x + y) / 17.1);
      row[x] = q3(smooth + 0.11 * (rand() - 0.5));
    }
    scene[y] = row;
  }
  const patch = makePatch();
  const negative = makeDecoy(patch);
  for (let j = 0; j < PATCH; j++) {
    for (let i = 0; i < PATCH; i++) {
      scene[plant.y + j][plant.x + i] = patch[j][i];
      scene[decoy.y + j][decoy.x + i] = negative[j][i];
    }
  }
  return scene;
}

function brighten(scene) {
  return scene.map(row => row.map(value => q3(value + LIFT)));
}

// ---- CPU references -------------------------------------------------------

function ssdAt(scene, patch, x, y) {
  let sum = 0;
  for (let j = 0; j < PATCH; j++) {
    for (let i = 0; i < PATCH; i++) {
      const d = scene[y + j][x + i] - patch[j][i];
      sum += d * d;
    }
  }
  return sum;
}

// Sum of ABSOLUTE differences, and the plain signed sum: the two things a
// missing `* d` turns the score into.
function sadAt(scene, patch, x, y) {
  let sum = 0;
  for (let j = 0; j < PATCH; j++) {
    for (let i = 0; i < PATCH; i++) sum += Math.abs(scene[y + j][x + i] - patch[j][i]);
  }
  return sum;
}

function signedAt(scene, patch, x, y) {
  let sum = 0;
  for (let j = 0; j < PATCH; j++) {
    for (let i = 0; i < PATCH; i++) sum += scene[y + j][x + i] - patch[j][i];
  }
  return sum;
}

function mapOf(score) {
  const out = new Array(POSITIONS);
  for (let y = 0; y < POSITIONS; y++) {
    const row = new Array(POSITIONS);
    for (let x = 0; x < POSITIONS; x++) row[x] = score(x, y);
    out[y] = row;
  }
  return out;
}

function ssdMap(scene, patch) {
  return mapOf((x, y) => ssdAt(scene, patch, x, y));
}

// Everything a template contributes to a score, in the two forms the module
// uses: the raw moments (task 3, computed per thread) and the centred template
// plus its length (task 4, computed once).
function patchStats(patch) {
  let sum = 0;
  let sumSq = 0;
  for (let j = 0; j < PATCH; j++) {
    for (let i = 0; i < PATCH; i++) {
      sum += patch[j][i];
      sumSq += patch[j][i] * patch[j][i];
    }
  }
  const mean = sum / COUNT;
  const centered = new Array(PATCH);
  let normSq = 0;
  for (let j = 0; j < PATCH; j++) {
    const row = new Array(PATCH);
    for (let i = 0; i < PATCH; i++) {
      row[i] = patch[j][i] - mean;
      normSq += row[i] * row[i];
    }
    centered[j] = row;
  }
  return { sum, sumSq, spread: sumSq - (sum * sum) / COUNT, mean, centered, norm: Math.sqrt(normSq) };
}

// The window's raw moments — shared by the reference and by every probe, so a
// probe can never disagree with the reference about what the window holds.
function windowMoments(scene, patch, x, y) {
  let sumW = 0;
  let sumW2 = 0;
  let sumWT = 0;
  for (let j = 0; j < PATCH; j++) {
    for (let i = 0; i < PATCH; i++) {
      const w = scene[y + j][x + i];
      sumW += w;
      sumW2 += w * w;
      sumWT += w * patch[j][i];
    }
  }
  return { sumW, sumW2, sumWT };
}

function nccAt(scene, patch, x, y, stats) {
  const t = stats || patchStats(patch);
  const m = windowMoments(scene, patch, x, y);
  const cov = m.sumWT - (m.sumW * t.sum) / COUNT;
  const varW = m.sumW2 - (m.sumW * m.sumW) / COUNT;
  return cov / Math.sqrt(varW * t.spread);
}

function nccMap(scene, patch) {
  const stats = patchStats(patch);
  return mapOf((x, y) => nccAt(scene, patch, x, y, stats));
}

function bestMin(map) {
  let best = { x: 0, y: 0, score: Infinity };
  for (let y = 0; y < map.length; y++) {
    for (let x = 0; x < map[y].length; x++) {
      if (map[y][x] < best.score) best = { x, y, score: map[y][x] };
    }
  }
  return best;
}

function bestMax(map) {
  let best = { x: 0, y: 0, score: -Infinity };
  for (let y = 0; y < map.length; y++) {
    for (let x = 0; x < map[y].length; x++) {
      if (map[y][x] > best.score) best = { x, y, score: map[y][x] };
    }
  }
  return best;
}

// Every number that appeared in a console.log line.
function loggedNumbers(logs) {
  const out = [];
  for (const line of logs) {
    if (line.type !== 'log' || !line.text) continue;
    const matches = line.text.match(/-?\d+(?:\.\d+)?/g);
    if (matches) for (const m of matches) out.push(parseFloat(m));
  }
  return out;
}

function logged(nums, value, eps = 0.5) {
  return nums.some(n => Math.abs(n - value) <= eps);
}

// ---- near-miss diagnosis --------------------------------------------------
//
// A failing assert that reports only two numbers tells a learner nothing about
// WHICH slip produced them. A probe pairs the value one specific known mistake
// would produce with a sentence naming that mistake; diagnose() speaks only
// when the observed value matches a probe within the test's own tolerance AND
// the correct value does not — so a position where two candidates coincide
// (the exact match, where the sum of squared and the sum of absolute
// differences are both 0) stays silent, as do observations matching probes
// that disagree with each other. A wrong diagnosis is worse than none.
function diagnose(got, expected, eps, probes) {
  const hits = probes
    .filter(p => Math.abs(got - p[0]) <= eps && Math.abs(expected - p[0]) > eps)
    .map(p => p[1]);
  return hits.length && hits.every(m => m === hits[0]) ? hits[0] : null;
}

// One position is weak evidence when a candidate can coincide with the right
// answer by accident: over a window whose differences all happen to share a
// sign, the sum of ABSOLUTE differences and the plain signed sum are the same
// number, so naming either one there would be a coin flip. This form therefore
// demands that a probe predict EVERY sampled position — and differ from the
// correct answer at at least one of them — before it may speak.
function diagnoseCells(cells, got, expected, eps, probes) {
  const hits = probes
    .filter(([value]) => {
      let differs = false;
      for (const [x, y] of cells) {
        const v = value(x, y);
        if (!(Math.abs(got(x, y) - v) <= eps)) return false;
        if (Math.abs(expected(x, y) - v) > eps) differs = true;
      }
      return differs;
    })
    .map(p => p[1]);
  return hits.length && hits.every(m => m === hits[0]) ? hits[0] : null;
}

// Swapping this.thread.x and this.thread.y transposes the whole score map, and
// the score map is square, so the mistake is invisible to any test that only
// checks shape. Positions on the diagonal (y === x) are their own transpose and
// can never show it, which is why the case lists below are off-diagonal.
function transposeCellHint(got, transposed, eps, x, y) {
  return Math.abs(got - transposed) <= eps
    ? 'that is the score for position (' + y + ', ' + x + ') — this.thread.x and ' +
      'this.thread.y are swapped, so the whole map came back transposed. Rows come first: ' +
      'scene[this.thread.y + j][this.thread.x + i]'
    : null;
}

// The one mistake that shows up in the map's SHAPE rather than its values.
// Both directions of the off-by-one land here, and each gets its own sentence
// because the consequences are different: too small silently never tests the
// last positions, too large reads off the end of the scene.
function mapShapeHint(rows) {
  const arithmetic =
    SCENE + ' − ' + PATCH + ' + 1 = ' + POSITIONS + ' positions per axis';
  if (rows === SCENE) {
    return 'the map is ' + SCENE + ' wide, one cell per scene pixel — but a window whose ' +
      'top-left corner sits at column ' + (SCENE - PATCH + 1) + ' would need columns past ' +
      (SCENE - 1) + ', which do not exist. The last legal corner is ' + (SCENE - PATCH) + ': ' +
      arithmetic;
  }
  if (rows === POSITIONS - 1 || rows === SCENE - PATCH) {
    return 'one position short — the corner at ' + (SCENE - PATCH) + ' is legal (its window ' +
      'ends exactly at ' + (SCENE - 1) + '), so it has to be tested too: ' + arithmetic;
  }
  return null;
}

// Two quite different ways a cell comes back as a non-number, and they deserve
// different sentences: nothing at all there means the map is too small, while
// a NaN means the arithmetic inside the thread went wrong.
function notFiniteHint(got) {
  if (got === undefined) {
    return 'there is no score at that position at all — the map is smaller than ' + POSITIONS +
      '×' + POSITIONS + ', so some perfectly legal positions were never tested. ' +
      SCENE + ' − ' + PATCH + ' + 1 = ' + POSITIONS + ' positions per axis.';
  }
  return Number.isFinite(got)
    ? null
    : 'that score is not a number. Either a loop ran one step too far and read past the ' +
      'template, or the denominator came out zero — check that the window sums are being ' +
      'combined the way the identities say.';
}

// Task 1: the sum that forgot to square, in its two flavours. Read across a
// whole list of positions (see diagnoseCells) — at a single position the two
// can produce the same number.
const SSD_CELLS = [[0, 1], [17, 40], [70, 5], [3, 55], [61, 24], [88, 0], [24, 61], [12, 63]];

function ssdProbes(scene, patch) {
  return [
    [(x, y) => sadAt(scene, patch, x, y),
      'that is the sum of ABSOLUTE differences — a perfectly good score (it is called SAD), ' +
      'but not this one. Square each difference instead: const d = …; sum += d * d;'],
    [(x, y) => signedAt(scene, patch, x, y),
      'the differences were added without squaring, so positives cancelled negatives — which is ' +
      'why a badly matching window can score near zero. Square each one: sum += d * d;'],
  ];
}

// Task 3 and 4: the three ways the normalisation goes wrong that still produce
// a plausible-looking number. (A fourth — leaving the numerator un-centred —
// produces scores in the hundreds, and the range test catches that one.)
function nccProbes(scene, patch, x, y, stats) {
  const t = stats || patchStats(patch);
  const m = windowMoments(scene, patch, x, y);
  const cov = m.sumWT - (m.sumW * t.sum) / COUNT;
  const varW = m.sumW2 - (m.sumW * m.sumW) / COUNT;
  const means =
    'no mean was subtracted anywhere — that is Σwt ⁄ √(Σw²·Σt²), which sits just under 1 for ' +
    'every window because brightness dominates it. Subtract the means first: ' +
    'cov = sumWT − sumW * sumT / n, and the same correction goes into both variances.';
  return [
    [m.sumWT / Math.sqrt(m.sumW2 * t.sumSq), means],
    [cov / Math.sqrt(m.sumW2 * t.sumSq),
      'the numerator subtracts the means but the denominator does not. The same correction ' +
      'belongs in both variances: varW = sumW2 − sumW * sumW / n, varT = sumT2 − sumT * sumT / n.'],
    [cov / (varW * t.spread),
      'the denominator is a product of variances, not of standard deviations — which is why the ' +
      'score left the −1…1 range. Take the square root: cov / Math.sqrt(varW * varT).'],
  ];
}

// The whole-map version of the same gotcha, and the one that catches every
// denominator mistake at once: a normalized correlation simply cannot leave
// [−1, 1], so a map that does is definitive without needing to guess which
// slip caused it.
function rangeProblem(map) {
  for (let y = 0; y < map.length; y++) {
    for (let x = 0; x < map[y].length; x++) {
      const v = map[y][x];
      if (!Number.isFinite(v)) {
        return 'the score at position (' + x + ', ' + y + ') is ' + v + ', not a number — the ' +
          'denominator came out zero or negative. Σw² − (Σw)²/n is never negative when it is ' +
          'assembled in that order.';
      }
      if (Math.abs(v) > 1.001) {
        return 'the score at position (' + x + ', ' + y + ') is ' + v.toFixed(3) + ', and a ' +
          'normalized correlation cannot leave −1…1. The classic cause is dividing by the two ' +
          'variances instead of by their square roots — the denominator is ' +
          'Math.sqrt(varW * varT). (Leaving the means in the numerator does this too.)';
      }
    }
  }
  return null;
}

// Tasks 3 and 5: the sign convention is inverted between the two measures, and
// on this scene taking NCC's minimum lands on the planted negative — a wrong
// answer specific enough to name. Only speaks when the right coordinates are
// absent, so a learner who logged both stays undiagnosed.
function reportedHint(nums, plant, decoy) {
  const right = logged(nums, plant.x) && logged(nums, plant.y);
  if (right) return null;
  if (logged(nums, decoy.x) && logged(nums, decoy.y)) {
    return 'you reported (' + decoy.x + ', ' + decoy.y + '), which is where this map is at its ' +
      'LOWEST. NCC is a similarity — +1 is a perfect match and −1 a perfect anti-match — so its ' +
      'best is the maximum. (SSD was the other way round: a distance, best at its minimum.) ' +
      'That position holds the template with its lights and darks swapped, and it scores −1.';
  }
  if (logged(nums, plant.x + PATCH / 2) && logged(nums, plant.y + PATCH / 2)) {
    return 'that is the CENTRE of the matched window. A score map cell (x, y) belongs to the ' +
      'window whose top-left corner is at (x, y) — report the corner, or say that you are ' +
      'reporting the centre.';
  }
  return null;
}

export default {
  uuid: 'f57b4bed-0519-42f0-a9fb-739679e67957',
  version: 1,
  slug: 'template-matching',
  title: 'Template Matching',
  blurb: 'Finding a patch in a picture — and why a raw difference score is fooled by a light switch.',
  tasks: [
    // ------------------------------------------------------------------ 1
    {
      slug: 'ssd-score-map',
      title: 'Score Every Position at Once',
      intro: `<p>Template matching asks the simplest question in object finding: <strong>where in
        this picture is that patch?</strong> Slide the patch over every position it could occupy,
        score how well it fits at each one, and keep the best. That sounds like a loop, and on a
        GPU it is the exact opposite of a loop — every candidate position is scored from data
        alone, with no reference to any other position. One thread per position, all 7,921 of
        them at once.</p>
        <p>The obvious score is the <strong>sum of squared differences</strong>. Line the 8×8
        template up with its top-left corner at (x, y), subtract it from the scene pixel by pixel,
        square each difference so a positive cannot cancel a negative, and add them up. Zero is a
        perfect match; bigger is worse.</p>
<pre><code>d   = scene[y + j][x + i] − patch[j][i]
SSD = sum of d² over the 8×8 window</code></pre>
        <p>One thing to settle before you write a line: <strong>the score map is smaller than the
        scene</strong>. A window whose corner sits at column 89 would need columns 89…96, and this
        scene stops at 95. The last legal corner is 88, so there are 96 − 8 + 1 = <strong>89</strong>
        positions along each axis, and the map is 89×89. <code>scene</code> here is a luminance
        map — one number per pixel, the kind a grayscale pass hands you — but it is indexed like
        any other image.</p>
        ${ARRAY_LAYOUT}`,
      goal: `<strong>Goal:</strong> build the 89×89 SSD score map for <code>patch</code> over
        <code>scene</code>, and log the position of the best match.`,
      requirements: [
        'Set <code>output</code> to the number of candidate positions — <code>96 − 8 + 1</code> per axis, not 96',
        'Sum over the whole template with a double loop bounded by <code>this.constants.size</code>',
        'Square every difference: <code>const d = …; sum += d * d;</code>',
        '<code>console.log</code> the position <code>bestMatch()</code> returns',
      ],
      hints: [
        {
          title: 'Hint 1 — which window is mine?',
          body: `<p>Thread (x, y) owns the window whose <em>top-left corner</em> is at
            <code>scene[y][x]</code>. Its pixels are <code>scene[y + j][x + i]</code> for
            <code>j</code> and <code>i</code> from 0 to 7 — and those same <code>j</code>,
            <code>i</code> index the template as <code>patch[j][i]</code>. No clamping is needed
            anywhere: the output shape already guarantees every read is in bounds.</p>`,
        },
        {
          title: 'Hint 2 — the loop body',
          body: `<pre><code>const d = scene[y + j][x + i] - patch[j][i];
sum += d * d;</code></pre>
<p>— two statements, inside two nested <code>for</code> loops that both run to
            <code>this.constants.size</code>.</p>`,
        },
        {
          title: 'Hint 3 — the whole kernel',
          body: `<pre><code>const x = this.thread.x;
const y = this.thread.y;
let sum = 0;
for (let j = 0; j &lt; this.constants.size; j++) {
  for (let i = 0; i &lt; this.constants.size; i++) {
    const d = scene[y + j][x + i] - patch[j][i];
    sum += d * d;
  }
}
return sum;</code></pre>
<p>— and <code>output: [89, 89]</code>.</p>`,
        },
      ],
      transfer: `This is OpenCV's <code>matchTemplate</code> and NVIDIA NPP's
        <code>nppiSQRDistanceNorm</code>, and it is one of the friendliest workloads a GPU ever
        sees: no communication between threads, no atomics, perfectly regular reads, and
        neighbouring threads reading overlapping windows straight out of cache. A WGSL compute
        shader or a CUDA 2D block does it with the same two nested loops.`,
      starterCode: `// One thread per candidate position. 89 × 89 = 7,921 of them.
const gpu = new GPU({ mode });

const ssd = gpu.createKernel(function (scene, patch) {
  const x = this.thread.x;
  const y = this.thread.y;
  // TODO: sum (scene[y + j][x + i] - patch[j][i])² over the whole
  // this.constants.size × this.constants.size template.
  return 0;
}, {
  // TODO: 88 is wrong. How many top-left corners actually fit?
  output: [88, 88],
  constants: { size: 8 },
});

// Scanning 7,921 scores in JavaScript is not the lesson here — Reductions and
// Top-K Selection do exactly this on the GPU, in parallel, and properly.
function bestMatch(map) {
  let best = Infinity;
  let bx = 0;
  let by = 0;
  for (let y = 0; y < map.length; y++) {
    for (let x = 0; x < map[y].length; x++) {
      if (map[y][x] < best) {
        best = map[y][x];
        bx = x;
        by = y;
      }
    }
  }
  return { x: bx, y: by, score: best };
}

const map = ssd(scene, patch);
const hit = bestMatch(map);
console.log('best match at x =', hit.x, ' y =', hit.y, ' score =', hit.score);
`,
      solutionCode: `// One thread per candidate position. 89 × 89 = 7,921 of them.
const gpu = new GPU({ mode });

const ssd = gpu.createKernel(function (scene, patch) {
  const x = this.thread.x;
  const y = this.thread.y;
  let sum = 0;
  for (let j = 0; j < this.constants.size; j++) {
    for (let i = 0; i < this.constants.size; i++) {
      const d = scene[y + j][x + i] - patch[j][i];
      sum += d * d;
    }
  }
  return sum;
}, {
  output: [89, 89],
  constants: { size: 8 },
});

// Scanning 7,921 scores in JavaScript is not the lesson here — Reductions and
// Top-K Selection do exactly this on the GPU, in parallel, and properly.
function bestMatch(map) {
  let best = Infinity;
  let bx = 0;
  let by = 0;
  for (let y = 0; y < map.length; y++) {
    for (let x = 0; x < map[y].length; x++) {
      if (map[y][x] < best) {
        best = map[y][x];
        bx = x;
        by = y;
      }
    }
  }
  return { x: bx, y: by, score: best };
}

const map = ssd(scene, patch);
const hit = bestMatch(map);
console.log('best match at x =', hit.x, ' y =', hit.y, ' score =', hit.score);
`,
      inputs: utils => ({ scene: makeScene(utils), patch: makePatch() }),
      publicTests: [
        {
          name: 'the score map is <code>89×89</code> — one cell per candidate position',
          run: async ctx => {
            ctx.assert(ctx.kernels.length >= 1, 'no kernel was created — call gpu.createKernel()');
            const out = ctx.kernel(makeScene(ctx.utils), makePatch());
            ctx.assert(
              out && out.length === POSITIONS,
              mapShapeHint(out && out.length) ||
                `expected ${POSITIONS} rows of scores, got ${out && out.length}`
            );
            ctx.assert(
              out[0] && out[0].length === POSITIONS,
              mapShapeHint(out[0] && out[0].length) ||
                `expected ${POSITIONS} scores per row, got ${out[0] && out[0].length}`
            );
          },
        },
        {
          name: 'each cell is the sum of <em>squared</em> differences over its 8×8 window',
          run: async ctx => {
            const scene = makeScene(ctx.utils);
            const patch = makePatch();
            const out = ctx.kernel(scene, patch);
            const ref = ssdMap(scene, patch);
            const unsquared = diagnoseCells(
              SSD_CELLS, (x, y) => out[y][x], (x, y) => ref[y][x], 1.5e-3, ssdProbes(scene, patch)
            );
            for (const [x, y] of SSD_CELLS) {
              const hint =
                notFiniteHint(out[y][x]) ||
                transposeCellHint(out[y][x], ref[x][y], 1.5e-3, x, y) ||
                unsquared;
              ctx.assertClose(out[y][x], ref[y][x], 1.5e-3, hint || `position (${x}, ${y})`);
            }
          },
        },
        {
          name: 'the patch is found — SSD is exactly 0 where it sits, and the position is logged',
          run: async ctx => {
            const scene = makeScene(ctx.utils);
            const patch = makePatch();
            const out = ctx.kernel(scene, patch);
            ctx.assertClose(
              out[PLANT.y][PLANT.x],
              0,
              1e-3,
              transposeCellHint(out[PLANT.y][PLANT.x], out[PLANT.x][PLANT.y], 1e-3, PLANT.x, PLANT.y) ||
                'the template was planted verbatim in the scene, so the window it occupies should ' +
                  'differ from it by nothing at all — that score has to be 0'
            );
            const best = bestMin(out);
            ctx.assert(
              best.x === PLANT.x && best.y === PLANT.y,
              `the lowest score should be at (${PLANT.x}, ${PLANT.y}), but it is at (${best.x}, ${best.y})`
            );
            const nums = loggedNumbers(ctx.logs);
            ctx.assert(
              logged(nums, PLANT.x) && logged(nums, PLANT.y),
              `log the best match — expected x = ${PLANT.x} and y = ${PLANT.y} in the console output`
            );
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            const plant = { x: 31, y: 66 };
            const scene = makeScene(ctx.utils, 8814, plant, { x: 70, y: 9 });
            const patch = makePatch();
            const out = ctx.kernel(scene, patch);
            ctx.assert(out && out.length === POSITIONS, mapShapeHint(out && out.length) ||
              `expected ${POSITIONS} rows of scores`);
            const ref = ssdMap(scene, patch);
            const unsquared = diagnoseCells(
              SSD_CELLS, (x, y) => out[y][x], (x, y) => ref[y][x], 1.5e-3, ssdProbes(scene, patch)
            );
            for (let y = 0; y < POSITIONS; y++) {
              for (let x = 0; x < POSITIONS; x++) {
                if (Math.abs(out[y][x] - ref[y][x]) <= 1.5e-3) continue;
                const hint =
                  notFiniteHint(out[y][x]) ||
                  transposeCellHint(out[y][x], ref[x][y], 1.5e-3, x, y) ||
                  unsquared;
                ctx.assertClose(out[y][x], ref[y][x], 1.5e-3, hint || `position (${x}, ${y})`);
              }
            }
            const best = bestMin(out);
            ctx.assert(
              best.x === plant.x && best.y === plant.y,
              `on a fresh scene the best match should be at (${plant.x}, ${plant.y}), got (${best.x}, ${best.y})`
            );
          },
        },
      ],
    },

    // ------------------------------------------------------------------ 2
    {
      slug: 'brightness-breaks-it',
      title: 'The Score That Lies',
      intro: `<p>Now break it. <code>brightScene</code> is the same scene photographed in brighter
        light: every value 0.28 higher, nothing moved, nothing changed shape. The patch is still
        exactly where it was. Run the same kernel over it — the kernel is not what is wrong here —
        and the best match walks off to a completely different place.</p>
        <p>Here is why, in one line of algebra. Add δ to every scene value and the score at a
        window becomes</p>
<pre><code>SSD(w + δ, t) = SSD(w, t)
              + 2δ · sum(wᵢ − tᵢ)
              + n · δ²</code></pre>
        <p>At the true match the pixels agree, so <code>sum(wᵢ − tᵢ)</code> is zero and there is
        nothing to offset the last term: a <em>perfect</em> match now scores
        <code>64 × 0.28² = 5.02</code>. Meanwhile any window that is <strong>darker</strong> than
        the template has a negative <code>sum(wᵢ − tᵢ)</code>, and the middle term pays it a
        discount. Somewhere in this scene sits a patch that is dark and matches badly; brighten
        the picture and its discount beats a perfect match outright.</p>
        <p>That is the whole lesson of this module, and it is not really about vision. SSD is not
        a measure of similarity — it is a measure of <strong>distance in absolute value</strong>,
        and every camera, every light, every exposure, every gain setting moves absolute values
        around. A score that cannot tell "brighter" from "different" will confidently point at
        the wrong thing.</p>`,
      goal: `<strong>Goal:</strong> score both scenes with the same SSD kernel and show the damage
        — log where each one thinks the patch is, and the two bright-scene scores that explain it.`,
      requirements: [
        'Score <code>scene</code> and <code>brightScene</code> with the same kernel',
        '<code>console.log</code> the best position on each map — they disagree',
        'From the bright map, <code>console.log</code> the score at the true position and the score the winner got — the winner\'s is smaller',
      ],
      hints: [
        {
          title: 'Hint 1 — nothing about the kernel changes',
          body: `<p>Same kernel, called twice. <code>brightScene</code> has exactly the same shape
            as <code>scene</code>, so the second call costs you one line.</p>`,
        },
        {
          title: 'Hint 2 — reading a known cell',
          body: `<p>The map is indexed <code>map[y][x]</code>, so the score the bright map gives
            the true position is <code>brightMap[TRUE_Y][TRUE_X]</code>. Compare it against
            <code>bestMatch(brightMap).score</code>.</p>`,
        },
      ],
      transfer: `Every practitioner meets this wall. It is why OpenCV ships
        <code>TM_CCOEFF_NORMED</code> alongside <code>TM_SQDIFF</code>, why stereo matchers use
        census transforms or rank filters instead of raw differences, and why "we normalised the
        inputs and the model started working" is the most common debugging story in machine
        learning. A raw difference is a distance in whatever units the sensor happened to
        produce.`,
      starterCode: `// Same kernel, two scenes. The kernel is not what is wrong here.
const gpu = new GPU({ mode });

const ssd = gpu.createKernel(function (scene, patch) {
  const x = this.thread.x;
  const y = this.thread.y;
  let sum = 0;
  for (let j = 0; j < this.constants.size; j++) {
    for (let i = 0; i < this.constants.size; i++) {
      const d = scene[y + j][x + i] - patch[j][i];
      sum += d * d;
    }
  }
  return sum;
}, {
  output: [89, 89],
  constants: { size: 8 },
});

function bestMatch(map) {
  let best = Infinity;
  let bx = 0;
  let by = 0;
  for (let y = 0; y < map.length; y++) {
    for (let x = 0; x < map[y].length; x++) {
      if (map[y][x] < best) {
        best = map[y][x];
        bx = x;
        by = y;
      }
    }
  }
  return { x: bx, y: by, score: best };
}

// Where the patch really is — task 1 found it.
const TRUE_X = 58;
const TRUE_Y = 21;

const plainMap = ssd(scene, patch);
const brightMap = ssd(brightScene, patch);

// TODO: log the best position on each map.
// TODO: log brightMap's score at the true position, and the score its winner
//       got. The winner's is smaller — that is the failure, in two numbers.
`,
      solutionCode: `// Same kernel, two scenes. The kernel is not what is wrong here.
const gpu = new GPU({ mode });

const ssd = gpu.createKernel(function (scene, patch) {
  const x = this.thread.x;
  const y = this.thread.y;
  let sum = 0;
  for (let j = 0; j < this.constants.size; j++) {
    for (let i = 0; i < this.constants.size; i++) {
      const d = scene[y + j][x + i] - patch[j][i];
      sum += d * d;
    }
  }
  return sum;
}, {
  output: [89, 89],
  constants: { size: 8 },
});

function bestMatch(map) {
  let best = Infinity;
  let bx = 0;
  let by = 0;
  for (let y = 0; y < map.length; y++) {
    for (let x = 0; x < map[y].length; x++) {
      if (map[y][x] < best) {
        best = map[y][x];
        bx = x;
        by = y;
      }
    }
  }
  return { x: bx, y: by, score: best };
}

// Where the patch really is — task 1 found it.
const TRUE_X = 58;
const TRUE_Y = 21;

const plainMap = ssd(scene, patch);
const brightMap = ssd(brightScene, patch);

const plainBest = bestMatch(plainMap);
const brightBest = bestMatch(brightMap);
console.log('as photographed: best at x =', plainBest.x, ' y =', plainBest.y);
console.log('brighter light:  best at x =', brightBest.x, ' y =', brightBest.y);

console.log('bright score at the true position:', brightMap[TRUE_Y][TRUE_X]);
console.log('bright score the winner got:      ', brightBest.score);
`,
      inputs: utils => {
        const scene = makeScene(utils);
        return { scene, brightScene: brighten(scene), patch: makePatch() };
      },
      publicTests: [
        {
          name: 'on the scene as photographed, SSD still finds the patch',
          run: async ctx => {
            ctx.assert(ctx.kernels.length >= 1, 'no kernel was created — call gpu.createKernel()');
            const scene = makeScene(ctx.utils);
            const patch = makePatch();
            const out = ctx.kernel(scene, patch);
            ctx.assert(
              out && out.length === POSITIONS,
              mapShapeHint(out && out.length) || `expected a ${POSITIONS}×${POSITIONS} score map`
            );
            const best = bestMin(out);
            ctx.assert(
              best.x === PLANT.x && best.y === PLANT.y,
              `the plain scene's best match should still be at (${PLANT.x}, ${PLANT.y}), got (${best.x}, ${best.y})`
            );
            const nums = loggedNumbers(ctx.logs);
            ctx.assert(
              logged(nums, PLANT.x) && logged(nums, PLANT.y),
              `log the plain scene's best position — expected x = ${PLANT.x} and y = ${PLANT.y}`
            );
          },
        },
        {
          name: '0.28 of extra light moves the winner to <code>(12, 63)</code>',
          run: async ctx => {
            const bright = brighten(makeScene(ctx.utils));
            const patch = makePatch();
            const out = ctx.kernel(bright, patch);
            const best = bestMin(out);
            ctx.assert(
              best.x === DECOY.x && best.y === DECOY.y,
              `the brightened scene's lowest SSD should be at (${DECOY.x}, ${DECOY.y}) — the dark ` +
                `patch that gets the discount — but this map's is at (${best.x}, ${best.y}). Is ` +
                'the kernel being given brightScene?'
            );
            const nums = loggedNumbers(ctx.logs);
            ctx.assert(
              logged(nums, DECOY.x) && logged(nums, DECOY.y),
              `log the brightened scene's best position too — expected x = ${DECOY.x} and y = ${DECOY.y}`
            );
          },
        },
        {
          name: 'the two numbers that prove it are logged: <code>5.02</code> beaten by <code>0.40</code>',
          run: async ctx => {
            const bright = brighten(makeScene(ctx.utils));
            const patch = makePatch();
            const trueScore = ssdAt(bright, patch, PLANT.x, PLANT.y);
            const winnerScore = ssdAt(bright, patch, DECOY.x, DECOY.y);
            const nums = loggedNumbers(ctx.logs);
            ctx.assert(
              logged(nums, trueScore, 0.02),
              `log the bright map's score at the true position — expected ≈${trueScore.toFixed(2)} ` +
                '(that is 64 × 0.28², what a perfect match now costs)'
            );
            ctx.assert(
              logged(nums, winnerScore, 0.02),
              `log the score the winner got — expected ≈${winnerScore.toFixed(2)}, comfortably ` +
                `below the ${trueScore.toFixed(2)} a perfect match scores`
            );
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            const plant = { x: 9, y: 40 };
            const decoy = { x: 74, y: 74 };
            const scene = makeScene(ctx.utils, 51217, plant, decoy);
            const bright = brighten(scene);
            const patch = makePatch();
            const plainOut = ctx.kernel(scene, patch);
            const brightOut = ctx.kernel(bright, patch);
            const plainRef = ssdMap(scene, patch);
            const brightRef = ssdMap(bright, patch);
            const unsquared = diagnoseCells(
              SSD_CELLS, (x, y) => plainOut[y][x], (x, y) => plainRef[y][x], 1.5e-3,
              ssdProbes(scene, patch)
            );
            for (let y = 0; y < POSITIONS; y++) {
              for (let x = 0; x < POSITIONS; x++) {
                if (Math.abs(plainOut[y][x] - plainRef[y][x]) > 1.5e-3) {
                  const hint =
                    notFiniteHint(plainOut[y][x]) ||
                    transposeCellHint(plainOut[y][x], plainRef[x][y], 1.5e-3, x, y) ||
                    unsquared;
                  ctx.assertClose(plainOut[y][x], plainRef[y][x], 1.5e-3, hint || `position (${x}, ${y})`);
                }
                if (Math.abs(brightOut[y][x] - brightRef[y][x]) > 1.5e-3) {
                  ctx.assertClose(brightOut[y][x], brightRef[y][x], 1.5e-3,
                    `brightened scene, position (${x}, ${y})`);
                }
              }
            }
            const plainBest = bestMin(plainOut);
            ctx.assert(
              plainBest.x === plant.x && plainBest.y === plant.y,
              `on a fresh plain scene the best match should be at (${plant.x}, ${plant.y}), got (${plainBest.x}, ${plainBest.y})`
            );
            const brightBest = bestMin(brightOut);
            ctx.assert(
              brightBest.x === decoy.x && brightBest.y === decoy.y,
              `on a fresh brightened scene SSD should be fooled into (${decoy.x}, ${decoy.y}), got (${brightBest.x}, ${brightBest.y})`
            );
          },
        },
      ],
    },

    // ------------------------------------------------------------------ 3
    {
      slug: 'normalized-correlation',
      title: 'Normalize It',
      intro: `<p>The fix is to stop comparing brightness and start comparing <em>shape</em>.
        Subtract each window's own mean, subtract the template's mean, and divide by how much
        each of them varies. What survives is <strong>normalized cross-correlation</strong>:</p>
<pre><code>cov  = sum( (wᵢ − w̄) · (tᵢ − t̄) )
varW = sum( (wᵢ − w̄)² )
varT = sum( (tᵢ − t̄)² )

NCC  = cov / sqrt(varW · varT)</code></pre>
        <p>Subtracting the means removes anything <em>added</em> to the light; dividing by the
        spreads removes anything the light was <em>multiplied</em> by. The result is bounded:
        <code>+1</code> is a perfect match, <code>0</code> is no relationship at all, and
        <code>−1</code> is a perfect <em>anti</em>-match — the same shape with its lights and
        darks swapped. That bound is a gift, because a score that leaves −1…1 is proof the
        arithmetic is wrong.</p>
        <p>Written that way it looks like three passes over the window: one to find the means,
        one for the spreads, one for the product. It is not. Every one of those three quantities
        is a sum over the same 64 pixels the thread is already reading, and two schoolbook
        identities turn all three into plain running totals:</p>
<pre><code>cov  = sumWT − sumW · sumT / n
varW = sumW2 − sumW · sumW / n
varT = sumT2 − sumT · sumT / n</code></pre>
        <p>So the thread keeps five accumulators — <code>sumW</code>, <code>sumW2</code>,
        <code>sumT</code>, <code>sumT2</code>, <code>sumWT</code> — fills them in one pass, and
        assembles the score after the loop. Five running totals, one divide, no mean subtracted
        from anything explicitly. (A window with no variation at all would put a zero in that
        denominator; production code adds a tiny epsilon for it. Nothing in this scene is flat,
        so the plain formula is safe here.)</p>`,
      goal: `<strong>Goal:</strong> build the 89×89 NCC map over <code>brightScene</code> and log
        the winning position and its score. The match snaps back to where the patch really is.`,
      requirements: [
        'Accumulate <code>sumW</code>, <code>sumW2</code>, <code>sumT</code>, <code>sumT2</code> and <code>sumWT</code> in one pass over the window',
        'Assemble <code>cov</code>, <code>varW</code> and <code>varT</code> with the identities above',
        'Return <code>cov / Math.sqrt(varW * varT)</code> — a square root of the product, not the product',
        'NCC is a similarity, so <code>bestMatch()</code> has to keep the <strong>largest</strong> score',
      ],
      hints: [
        {
          title: 'Hint 1 — five accumulators, one loop',
          body: `<p>Declare all five before the loops and add to each one inside:</p>
<pre><code>const w = scene[y + j][x + i];
const t = patch[j][i];
sumW += w;
sumW2 += w * w;
sumT += t;
sumT2 += t * t;
sumWT += w * t;</code></pre>`,
        },
        {
          title: 'Hint 2 — assembling the score',
          body: `<pre><code>const n = this.constants.count;
const cov = sumWT - (sumW * sumT) / n;
const varW = sumW2 - (sumW * sumW) / n;
const varT = sumT2 - (sumT * sumT) / n;
return cov / Math.sqrt(varW * varT);</code></pre>
<p>— note that <code>varW</code> and <code>varT</code> here are the sums of squared
            deviations, not the sums divided by <code>n</code>. Dividing both by <code>n</code>
            would cancel out of the ratio anyway, so there is no point paying for it.</p>`,
        },
        {
          title: 'Hint 3 — the other half of the change',
          body: `<p>Task 1's <code>bestMatch</code> kept the smallest score, because SSD was a
            distance. NCC is a similarity: <code>if (map[y][x] &gt; best)</code>, starting from
            <code>-Infinity</code>. Leave it as a minimum and this map will hand you its most
            spectacularly wrong position instead of its right one.</p>`,
        },
      ],
      transfer: `Normalising before you compare is one of the most portable ideas in computing.
        It is <code>TM_CCOEFF_NORMED</code> in OpenCV and <code>nppiCrossCorrValid_NormLevel</code>
        in CUDA's NPP; it is cosine similarity over centred vectors in every retrieval system; it
        is the Pearson correlation in statistics; and it is exactly what a batch-norm or
        layer-norm layer does inside a neural network, for exactly the same reason — so that what
        comes next responds to structure instead of to scale.`,
      starterCode: `// Same 7,921 threads. A score that brightness cannot move.
const gpu = new GPU({ mode });

const ncc = gpu.createKernel(function (scene, patch) {
  const x = this.thread.x;
  const y = this.thread.y;
  let sumW = 0;
  let sumW2 = 0;
  let sumT = 0;
  let sumT2 = 0;
  let sumWT = 0;
  // TODO: one pass over the 8×8 window, filling all five accumulators.

  // TODO: assemble cov, varW and varT with the two identities, then
  // return cov / Math.sqrt(varW * varT).
  return 0;
}, {
  output: [89, 89],
  constants: { size: 8, count: 64 },
});

function bestMatch(map) {
  let best = map[0][0];
  let bx = 0;
  let by = 0;
  for (let y = 0; y < map.length; y++) {
    for (let x = 0; x < map[y].length; x++) {
      // TODO: NCC is a similarity — keep the LARGER score, not the smaller.
      if (map[y][x] < best) {
        best = map[y][x];
        bx = x;
        by = y;
      }
    }
  }
  return { x: bx, y: by, score: best };
}

const map = ncc(brightScene, patch);
const hit = bestMatch(map);
console.log('best match at x =', hit.x, ' y =', hit.y, ' score =', hit.score);
`,
      solutionCode: `// Same 7,921 threads. A score that brightness cannot move.
const gpu = new GPU({ mode });

const ncc = gpu.createKernel(function (scene, patch) {
  const x = this.thread.x;
  const y = this.thread.y;
  let sumW = 0;
  let sumW2 = 0;
  let sumT = 0;
  let sumT2 = 0;
  let sumWT = 0;
  for (let j = 0; j < this.constants.size; j++) {
    for (let i = 0; i < this.constants.size; i++) {
      const w = scene[y + j][x + i];
      const t = patch[j][i];
      sumW += w;
      sumW2 += w * w;
      sumT += t;
      sumT2 += t * t;
      sumWT += w * t;
    }
  }
  const n = this.constants.count;
  const cov = sumWT - (sumW * sumT) / n;
  const varW = sumW2 - (sumW * sumW) / n;
  const varT = sumT2 - (sumT * sumT) / n;
  return cov / Math.sqrt(varW * varT);
}, {
  output: [89, 89],
  constants: { size: 8, count: 64 },
});

function bestMatch(map) {
  let best = map[0][0];
  let bx = 0;
  let by = 0;
  for (let y = 0; y < map.length; y++) {
    for (let x = 0; x < map[y].length; x++) {
      if (map[y][x] > best) {
        best = map[y][x];
        bx = x;
        by = y;
      }
    }
  }
  return { x: bx, y: by, score: best };
}

const map = ncc(brightScene, patch);
const hit = bestMatch(map);
console.log('best match at x =', hit.x, ' y =', hit.y, ' score =', hit.score);
`,
      inputs: utils => {
        const scene = makeScene(utils);
        return { scene, brightScene: brighten(scene), patch: makePatch() };
      },
      publicTests: [
        {
          name: 'an <code>89×89</code> map, and every score inside <code>−1 … 1</code>',
          run: async ctx => {
            ctx.assert(ctx.kernels.length >= 1, 'no kernel was created — call gpu.createKernel()');
            const bright = brighten(makeScene(ctx.utils));
            const out = ctx.kernel(bright, makePatch());
            ctx.assert(
              out && out.length === POSITIONS && out[0] && out[0].length === POSITIONS,
              mapShapeHint(out && out.length) || `expected a ${POSITIONS}×${POSITIONS} score map`
            );
            const problem = rangeProblem(out);
            ctx.assert(!problem, problem || '');
          },
        },
        {
          name: 'brightness cannot move it — the same scores come back from both scenes',
          run: async ctx => {
            const scene = makeScene(ctx.utils);
            const bright = brighten(scene);
            const patch = makePatch();
            const stats = patchStats(patch);
            const plainOut = ctx.kernel(scene, patch);
            const brightOut = ctx.kernel(bright, patch);
            const cases = [[0, 1], [17, 40], [70, 5], [3, 55], [61, 24], [88, 0], [24, 61],
              [PLANT.x, PLANT.y], [DECOY.x, DECOY.y]];
            for (const [x, y] of cases) {
              const expected = nccAt(scene, patch, x, y, stats);
              for (const [label, out, src] of [['scene', plainOut, scene], ['brightScene', brightOut, bright]]) {
                const hint =
                  notFiniteHint(out[y][x]) ||
                  transposeCellHint(out[y][x], nccAt(src, patch, y, x, stats), 3e-3, x, y) ||
                  diagnose(out[y][x], expected, 3e-3, nccProbes(src, patch, x, y, stats));
                ctx.assertClose(out[y][x], expected, 3e-3, hint ||
                  `${label}, position (${x}, ${y}) — the two scenes must give the SAME score here`);
              }
            }
          },
        },
        {
          name: 'the match snaps back to <code>(58, 21)</code> at a score of <code>1.000</code>',
          run: async ctx => {
            const bright = brighten(makeScene(ctx.utils));
            const patch = makePatch();
            const out = ctx.kernel(bright, patch);
            ctx.assertClose(out[PLANT.y][PLANT.x], 1, 3e-3,
              'the window at the planted position IS the template, so its normalized correlation ' +
                'with the template has to be exactly 1');
            const best = bestMax(out);
            ctx.assert(
              best.x === PLANT.x && best.y === PLANT.y,
              `the highest score should be at (${PLANT.x}, ${PLANT.y}), but it is at (${best.x}, ${best.y})`
            );
            ctx.assertClose(out[DECOY.y][DECOY.x], -1, 3e-3,
              `the negative planted at (${DECOY.x}, ${DECOY.y}) — the patch that fooled SSD — is ` +
                'the template with its lights and darks swapped, so it should score −1');
            const nums = loggedNumbers(ctx.logs);
            ctx.assert(
              logged(nums, PLANT.x) && logged(nums, PLANT.y),
              reportedHint(nums, PLANT, DECOY) ||
                `log the winning position — expected x = ${PLANT.x} and y = ${PLANT.y}`
            );
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            const plant = { x: 44, y: 44 };
            const decoy = { x: 5, y: 5 };
            const scene = makeScene(ctx.utils, 20604, plant, decoy);
            const patch = makePatch();
            const stats = patchStats(patch);
            const out = ctx.kernel(brighten(scene), patch);
            const problem = rangeProblem(out);
            ctx.assert(!problem, problem || '');
            const ref = nccMap(scene, patch);
            for (let y = 0; y < POSITIONS; y++) {
              for (let x = 0; x < POSITIONS; x++) {
                if (Math.abs(out[y][x] - ref[y][x]) <= 3e-3) continue;
                const hint =
                  notFiniteHint(out[y][x]) ||
                  transposeCellHint(out[y][x], ref[x][y], 3e-3, x, y) ||
                  diagnose(out[y][x], ref[y][x], 3e-3, nccProbes(scene, patch, x, y, stats));
                ctx.assertClose(out[y][x], ref[y][x], 3e-3, hint || `position (${x}, ${y})`);
              }
            }
            const best = bestMax(out);
            ctx.assert(
              best.x === plant.x && best.y === plant.y,
              `on a fresh scene the highest score should be at (${plant.x}, ${plant.y}), got (${best.x}, ${best.y})`
            );
          },
        },
      ],
    },

    // ------------------------------------------------------------------ 4
    {
      slug: 'precompute-the-template',
      title: 'Hoist What Never Changes',
      intro: `<p>Look again at what those 7,921 threads just did. Two of the five running totals —
        <code>sumT</code> and <code>sumT2</code> — depend only on the template. Every thread
        walked the same 64 template values, arrived at the same two numbers, used them once and
        threw them away. That is 7,920 calculations too many.</p>
        <p>So do it once, in JavaScript, before the kernel runs — and do it in the shape the
        kernel actually wants: the template with its mean already subtracted, plus the length of
        that centred template.</p>
<pre><code>patchMean = (sum of patch) / 64
patchCentered[j][i]
          = patch[j][i] − patchMean
patchNorm = sqrt(sum of patchCentered²)</code></pre>
        <p>That simplifies the numerator too. Once the centred values sum to zero,
        <code>sum((wᵢ − w̄) · cᵢ)</code> equals <code>sum(wᵢ · cᵢ)</code> — the window's own mean
        cancels itself out and never has to be subtracted from anything. The thread drops from
        five accumulators to three and from two square roots to one:</p>
<pre><code>varW = sumW2 − sumW · sumW / n
NCC  = sumWC / ( sqrt(varW) · patchNorm )</code></pre>
        <p>Press <strong>Benchmark</strong> before and after and watch the difference. Hoisting
        loop-invariant work out of a loop is the oldest optimisation there is; what makes it worth
        a task is that a GPU multiplies the saving by the thread count, so the same three lines
        buy far more here than they would in a <code>for</code> loop.</p>
        <p>There is a bigger version of this idea that this module deliberately does <em>not</em>
        build. <code>sumW</code> and <code>sumW2</code> can also be precomputed — for the entire
        scene, once — as <strong>integral images</strong> (summed-area tables): a table where each
        cell holds the sum of everything above and to the left of it, so any rectangle's total
        costs four lookups and three subtractions no matter how large the rectangle is. That is
        genuinely how large-template matching is done at scale. It is also a two-dimensional
        prefix sum, which is a module of its own — Prefix Sums (Scan) builds the one-dimensional
        version — and at 8×8 those four lookups would replace 64 reads this thread is making
        anyway. The win arrives when the template is 64×64, and so does the module.</p>`,
      goal: `<strong>Goal:</strong> compute the template's statistics once in JavaScript, pass
        them in, and get the same NCC map from a kernel that does strictly less work per thread.`,
      requirements: [
        'Compute <code>patchMean</code>, <code>patchCentered</code> and <code>patchNorm</code> in plain JavaScript, outside the kernel',
        'The kernel takes exactly three arguments — <code>(scene, centered, norm)</code>',
        'Keep three accumulators: <code>sumW</code>, <code>sumW2</code> and <code>sumWC</code>',
        'Same answer as before — log the winning position and its score',
      ],
      hints: [
        {
          title: 'Hint 1 — centring the template',
          body: `<p>Two passes over 64 values, in ordinary JavaScript:</p>
<pre><code>let sum = 0;
for (let j = 0; j &lt; 8; j++) {
  for (let i = 0; i &lt; 8; i++) sum += patch[j][i];
}
const patchMean = sum / 64;</code></pre>
<p>then build <code>patchCentered</code> as <code>patch[j][i] - patchMean</code>, accumulating
            the squares into <code>patchNorm</code> as you go — and take the square root at the
            end.</p>`,
        },
        {
          title: 'Hint 2 — the shorter loop body',
          body: `<pre><code>const w = scene[y + j][x + i];
sumW += w;
sumW2 += w * w;
sumWC += w * centered[j][i];</code></pre>
<p>— no <code>sumT</code>, no <code>sumT2</code>, and nothing to subtract from
            <code>sumWC</code>.</p>`,
        },
        {
          title: 'Hint 3 — the return',
          body: `<pre><code>const varW = sumW2 - (sumW * sumW) / this.constants.count;
return sumWC / (Math.sqrt(varW) * norm);</code></pre>
<p>— <code>norm</code> is a plain number argument; gpu.js is perfectly happy passing
            scalars alongside arrays.</p>`,
        },
      ],
      transfer: `Every mature matcher does this. OpenCV precomputes the template's sum and
        sum-of-squares once inside <code>matchTemplate</code>; cuDNN and MIOpen hoist per-filter
        constants out of every convolution launch; CUDA programmers park exactly this kind of
        small, read-only, uniformly-accessed data in <code>__constant__</code> memory, and WGSL
        puts it in a uniform buffer. The rule is the same everywhere: anything that does not vary
        with the thread index does not belong inside the thread.`,
      starterCode: `// The template's statistics are the same at all 7,921 positions.
// Compute them once, here, and hand the kernel the finished numbers.
const gpu = new GPU({ mode });

// TODO: the template's mean; the template with that mean subtracted;
// and the length of the centred template.
const patchMean = 0;
const patchCentered = patch;
const patchNorm = 1;

const ncc = gpu.createKernel(function (scene, centered, norm) {
  const x = this.thread.x;
  const y = this.thread.y;
  let sumW = 0;
  let sumW2 = 0;
  let sumWC = 0;
  // TODO: one pass over the window — the window's sum, its sum of squares,
  // and its dot product with centered[j][i].

  const varW = sumW2 - (sumW * sumW) / this.constants.count;
  return sumWC / (Math.sqrt(varW) * norm);
}, {
  output: [89, 89],
  constants: { size: 8, count: 64 },
});

function bestMatch(map) {
  let best = -Infinity;
  let bx = 0;
  let by = 0;
  for (let y = 0; y < map.length; y++) {
    for (let x = 0; x < map[y].length; x++) {
      if (map[y][x] > best) {
        best = map[y][x];
        bx = x;
        by = y;
      }
    }
  }
  return { x: bx, y: by, score: best };
}

const hit = bestMatch(ncc(brightScene, patchCentered, patchNorm));
console.log('best match at x =', hit.x, ' y =', hit.y, ' score =', hit.score);
`,
      solutionCode: `// The template's statistics are the same at all 7,921 positions.
// Compute them once, here, and hand the kernel the finished numbers.
const gpu = new GPU({ mode });

let patchSum = 0;
for (let j = 0; j < 8; j++) {
  for (let i = 0; i < 8; i++) patchSum += patch[j][i];
}
const patchMean = patchSum / 64;

const patchCentered = [];
let normSq = 0;
for (let j = 0; j < 8; j++) {
  const row = [];
  for (let i = 0; i < 8; i++) {
    const c = patch[j][i] - patchMean;
    row.push(c);
    normSq += c * c;
  }
  patchCentered.push(row);
}
const patchNorm = Math.sqrt(normSq);

const ncc = gpu.createKernel(function (scene, centered, norm) {
  const x = this.thread.x;
  const y = this.thread.y;
  let sumW = 0;
  let sumW2 = 0;
  let sumWC = 0;
  for (let j = 0; j < this.constants.size; j++) {
    for (let i = 0; i < this.constants.size; i++) {
      const w = scene[y + j][x + i];
      sumW += w;
      sumW2 += w * w;
      sumWC += w * centered[j][i];
    }
  }
  const varW = sumW2 - (sumW * sumW) / this.constants.count;
  return sumWC / (Math.sqrt(varW) * norm);
}, {
  output: [89, 89],
  constants: { size: 8, count: 64 },
});

function bestMatch(map) {
  let best = -Infinity;
  let bx = 0;
  let by = 0;
  for (let y = 0; y < map.length; y++) {
    for (let x = 0; x < map[y].length; x++) {
      if (map[y][x] > best) {
        best = map[y][x];
        bx = x;
        by = y;
      }
    }
  }
  return { x: bx, y: by, score: best };
}

const hit = bestMatch(ncc(brightScene, patchCentered, patchNorm));
console.log('best match at x =', hit.x, ' y =', hit.y, ' score =', hit.score);
`,
      inputs: utils => {
        const scene = makeScene(utils);
        return { brightScene: brighten(scene), patch: makePatch() };
      },
      publicTests: [
        {
          name: 'the kernel takes the finished template — three arguments, not two',
          run: async ctx => {
            ctx.assert(ctx.kernels.length >= 1, 'no kernel was created — call gpu.createKernel()');
            const names = (ctx.kernel.kernel && ctx.kernel.kernel.argumentNames) || [];
            ctx.assert(
              names.length === 3,
              `the kernel should take three arguments — (scene, centered, norm) — but it takes ` +
                `${names.length}${names.length ? ` (${names.join(', ')})` : ''}. Anything the ` +
                'template alone decides belongs outside the kernel, computed once.'
            );
          },
        },
        {
          name: 'same scores as before, from three accumulators instead of five',
          run: async ctx => {
            const scene = makeScene(ctx.utils);
            const bright = brighten(scene);
            const patch = makePatch();
            const stats = patchStats(patch);
            const out = ctx.kernel(bright, stats.centered, stats.norm);
            ctx.assert(
              out && out.length === POSITIONS && out[0] && out[0].length === POSITIONS,
              mapShapeHint(out && out.length) || `expected a ${POSITIONS}×${POSITIONS} score map`
            );
            const problem = rangeProblem(out);
            ctx.assert(!problem, problem || '');
            const cases = [[0, 1], [17, 40], [70, 5], [3, 55], [61, 24], [88, 0], [24, 61],
              [PLANT.x, PLANT.y], [DECOY.x, DECOY.y]];
            for (const [x, y] of cases) {
              const expected = nccAt(bright, patch, x, y, stats);
              const hint =
                notFiniteHint(out[y][x]) ||
                transposeCellHint(out[y][x], nccAt(bright, patch, y, x, stats), 3e-3, x, y) ||
                diagnose(out[y][x], expected, 3e-3, nccProbes(bright, patch, x, y, stats));
              ctx.assertClose(out[y][x], expected, 3e-3, hint || `position (${x}, ${y})`);
            }
          },
        },
        {
          name: 'the template statistics are right, and the winner is logged',
          run: async ctx => {
            const stats = patchStats(makePatch());
            const nums = loggedNumbers(ctx.logs);
            ctx.assert(
              logged(nums, PLANT.x) && logged(nums, PLANT.y),
              reportedHint(nums, PLANT, DECOY) ||
                `log the winning position — expected x = ${PLANT.x} and y = ${PLANT.y}`
            );
            ctx.assert(
              logged(nums, 1, 3e-3),
              'log the winning score too — a perfect match scores 1.000. A score that is not 1 ' +
                `usually means patchNorm is wrong: it is Math.sqrt(sum of squares) = ` +
                `${stats.norm.toFixed(4)}, not the sum of squares itself.`
            );
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            const scene = makeScene(ctx.utils, 8814, { x: 31, y: 66 }, { x: 70, y: 9 });
            const bright = brighten(scene);
            const patch = makePatch();
            const stats = patchStats(patch);
            const out = ctx.kernel(bright, stats.centered, stats.norm);
            const ref = nccMap(bright, patch);
            for (let y = 0; y < POSITIONS; y++) {
              for (let x = 0; x < POSITIONS; x++) {
                if (Math.abs(out[y][x] - ref[y][x]) <= 3e-3) continue;
                const hint =
                  notFiniteHint(out[y][x]) ||
                  transposeCellHint(out[y][x], ref[x][y], 3e-3, x, y) ||
                  diagnose(out[y][x], ref[y][x], 3e-3, nccProbes(bright, patch, x, y, stats));
                ctx.assertClose(out[y][x], ref[y][x], 3e-3, hint || `position (${x}, ${y})`);
              }
            }
            const best = bestMax(out);
            ctx.assert(
              best.x === 31 && best.y === 66,
              `on a fresh scene the highest score should be at (31, 66), got (${best.x}, ${best.y})`
            );
          },
        },
      ],
    },

    // ------------------------------------------------------------------ 5
    {
      slug: 'find-the-patch',
      title: 'Payoff: Present or Absent?',
      intro: `<p>Everything so far, pointed at a real question. <code>brightScene</code> hides an
        8×8 patch; <code>patch</code> is that patch. <code>rotatedPatch</code> is the same eight
        by eight values turned a quarter turn — identical mean, identical spread, identical
        histogram, and <strong>nowhere in the scene</strong>. Find the one; refuse the other.</p>
        <p>The kernel is finished (it is task 4's, unchanged). What is left is the part that
        catches people twice: reading an answer off a score map.</p>
        <p><strong>Take the maximum.</strong> NCC is a similarity, so its best is its largest.
        SSD was a distance, so its best was its smallest. The convention is inverted between the
        two measures, and reaching for the wrong one does not give you a slightly worse answer —
        it gives you the map's most emphatically <em>wrong</em> position.</p>
        <p><strong>The coordinates are the window's top-left corner.</strong> Cell (x, y) scored
        the window that starts at (x, y) and runs 8 pixels right and down. That corner is the
        answer. The centre is <code>(x + 4, y + 4)</code> if that is what you want — just be sure
        you know which one you are reporting, because the score map is 89 wide where the scene is
        96, and quietly mixing the two coordinate systems is how a detector ends up drawing boxes
        in the wrong place.</p>
        <p>And then the honest part. A search over 7,921 positions <em>always</em> returns a
        winner — the best score is a best score whether or not anything is there. What turns
        matching into <strong>detection</strong> is a <strong>threshold</strong>: a line below
        which "the best I found" means "nothing". Here the patch that is present scores 1.000 and
        the one that is absent tops out near 0.47, so 0.9 separates them with room to spare. That
        number is not universal — it depends on the noise, on the template, and on how much
        deformation you are willing to forgive — and calibrating it against data whose answers you
        already know is most of the work in building a real detector.</p>`,
      goal: `<strong>Goal:</strong> report where <code>patch</code> is, report that
        <code>rotatedPatch</code> is not there, and let <code>THRESHOLD</code> be what decides.`,
      requirements: [
        'Score both templates against <code>brightScene</code> with the same kernel',
        'Finish <code>bestMatch</code>: scan for the <strong>largest</strong> score and return the window\'s top-left corner',
        '<code>console.log</code> the position found for <code>patch</code>, and the best score each template managed',
        'For each template, <code>console.log</code> whether its best score clears <code>THRESHOLD</code> — one <code>true</code>, one <code>false</code>',
      ],
      hints: [
        {
          title: 'Hint 1 — the scan',
          body: `<p>Start from <code>-Infinity</code> and keep the larger:</p>
<pre><code>let best = -Infinity;
let bx = 0;
let by = 0;
for (let y = 0; y &lt; map.length; y++) {
  for (let x = 0; x &lt; map[y].length; x++) {
    if (map[y][x] &gt; best) {
      best = map[y][x];
      bx = x;
      by = y;
    }
  }
}</code></pre>
<p><code>bx</code> and <code>by</code> are already the corner — no offset to add.</p>`,
        },
        {
          title: 'Hint 2 — the verdict',
          body: `<p><code>prepare()</code> hands the kernel what task 4 built, so each report is
            three lines:</p>
<pre><code>const map = ncc(brightScene, t.centered, t.norm);
const hit = bestMatch(map);
console.log(label, hit.x, hit.y, hit.score,
  hit.score &gt;= THRESHOLD);</code></pre>`,
        },
      ],
      transfer: `Thresholding a similarity map is the last mile of nearly every classical
        detector — Viola-Jones cascades, ORB and SIFT keypoint matching with Lowe's ratio test,
        stereo correspondence rejecting low-confidence disparities — and it survives into modern
        ones as the confidence score on every bounding box a neural network emits. The score tells
        you which position is most like the template; only a threshold tells you whether the
        template is there at all.`,
      starterCode: `// The finished matcher. Two templates: one is in the scene, one is not.
const gpu = new GPU({ mode });

const THRESHOLD = 0.9;

const ncc = gpu.createKernel(function (scene, centered, norm) {
  const x = this.thread.x;
  const y = this.thread.y;
  let sumW = 0;
  let sumW2 = 0;
  let sumWC = 0;
  for (let j = 0; j < this.constants.size; j++) {
    for (let i = 0; i < this.constants.size; i++) {
      const w = scene[y + j][x + i];
      sumW += w;
      sumW2 += w * w;
      sumWC += w * centered[j][i];
    }
  }
  const varW = sumW2 - (sumW * sumW) / this.constants.count;
  return sumWC / (Math.sqrt(varW) * norm);
}, {
  output: [89, 89],
  constants: { size: 8, count: 64 },
});

// Task 4, packaged: any template in, the two numbers the kernel wants out.
function prepare(template) {
  let sum = 0;
  for (let j = 0; j < 8; j++) {
    for (let i = 0; i < 8; i++) sum += template[j][i];
  }
  const mean = sum / 64;
  const centered = [];
  let normSq = 0;
  for (let j = 0; j < 8; j++) {
    const row = [];
    for (let i = 0; i < 8; i++) {
      const c = template[j][i] - mean;
      row.push(c);
      normSq += c * c;
    }
    centered.push(row);
  }
  return { centered: centered, norm: Math.sqrt(normSq) };
}

function bestMatch(map) {
  // TODO: scan the whole map for its LARGEST score, and return the (x, y)
  // it came from — that (x, y) is the window's top-left corner.
  return { x: 0, y: 0, score: map[0][0] };
}

function report(label, template) {
  const t = prepare(template);
  const hit = bestMatch(ncc(brightScene, t.centered, t.norm));
  // TODO: log the label, the position, the score, and whether the score
  // clears THRESHOLD.
}

report('patch:        ', patch);
report('rotatedPatch: ', rotatedPatch);
`,
      solutionCode: `// The finished matcher. Two templates: one is in the scene, one is not.
const gpu = new GPU({ mode });

const THRESHOLD = 0.9;

const ncc = gpu.createKernel(function (scene, centered, norm) {
  const x = this.thread.x;
  const y = this.thread.y;
  let sumW = 0;
  let sumW2 = 0;
  let sumWC = 0;
  for (let j = 0; j < this.constants.size; j++) {
    for (let i = 0; i < this.constants.size; i++) {
      const w = scene[y + j][x + i];
      sumW += w;
      sumW2 += w * w;
      sumWC += w * centered[j][i];
    }
  }
  const varW = sumW2 - (sumW * sumW) / this.constants.count;
  return sumWC / (Math.sqrt(varW) * norm);
}, {
  output: [89, 89],
  constants: { size: 8, count: 64 },
});

// Task 4, packaged: any template in, the two numbers the kernel wants out.
function prepare(template) {
  let sum = 0;
  for (let j = 0; j < 8; j++) {
    for (let i = 0; i < 8; i++) sum += template[j][i];
  }
  const mean = sum / 64;
  const centered = [];
  let normSq = 0;
  for (let j = 0; j < 8; j++) {
    const row = [];
    for (let i = 0; i < 8; i++) {
      const c = template[j][i] - mean;
      row.push(c);
      normSq += c * c;
    }
    centered.push(row);
  }
  return { centered: centered, norm: Math.sqrt(normSq) };
}

function bestMatch(map) {
  let best = -Infinity;
  let bx = 0;
  let by = 0;
  for (let y = 0; y < map.length; y++) {
    for (let x = 0; x < map[y].length; x++) {
      if (map[y][x] > best) {
        best = map[y][x];
        bx = x;
        by = y;
      }
    }
  }
  return { x: bx, y: by, score: best };
}

function report(label, template) {
  const t = prepare(template);
  const hit = bestMatch(ncc(brightScene, t.centered, t.norm));
  console.log(
    label, 'corner at x =', hit.x, ' y =', hit.y,
    ' score =', hit.score, ' present?', hit.score >= THRESHOLD
  );
}

report('patch:        ', patch);
report('rotatedPatch: ', rotatedPatch);
`,
      inputs: utils => {
        const scene = makeScene(utils);
        const patch = makePatch();
        return { brightScene: brighten(scene), patch, rotatedPatch: rotatePatch(patch) };
      },
      publicTests: [
        {
          name: 'both templates get scored — one peaks at <code>1.000</code>, one nowhere near',
          run: async ctx => {
            ctx.assert(ctx.kernels.length >= 1, 'no kernel was created — call gpu.createKernel()');
            const bright = brighten(makeScene(ctx.utils));
            const patch = makePatch();
            const rotated = rotatePatch(patch);
            const here = patchStats(patch);
            const notHere = patchStats(rotated);
            const found = ctx.kernel(bright, here.centered, here.norm);
            ctx.assert(
              found && found.length === POSITIONS && found[0] && found[0].length === POSITIONS,
              mapShapeHint(found && found.length) || `expected a ${POSITIONS}×${POSITIONS} score map`
            );
            const problem = rangeProblem(found);
            ctx.assert(!problem, problem || '');
            const best = bestMax(found);
            ctx.assert(
              best.x === PLANT.x && best.y === PLANT.y,
              `patch should peak at (${PLANT.x}, ${PLANT.y}), got (${best.x}, ${best.y})`
            );
            ctx.assertClose(best.score, 1, 3e-3, 'the peak score for a template that IS present');
            const missing = bestMax(ctx.kernel(bright, notHere.centered, notHere.norm));
            ctx.assert(
              missing.score < THRESHOLD,
              `the rotated template is not in this scene, so its best score should stay below ` +
                `${THRESHOLD} — got ${missing.score.toFixed(3)}`
            );
          },
        },
        {
          name: 'the reported position is the window\'s top-left corner: <code>(58, 21)</code>',
          run: async ctx => {
            const nums = loggedNumbers(ctx.logs);
            ctx.assert(
              logged(nums, PLANT.x) && logged(nums, PLANT.y),
              reportedHint(nums, PLANT, DECOY) ||
                `log where patch was found — expected x = ${PLANT.x} and y = ${PLANT.y}`
            );
          },
        },
        {
          name: 'both best scores are logged, and <code>THRESHOLD</code> gives one verdict each',
          run: async ctx => {
            const bright = brighten(makeScene(ctx.utils));
            const patch = makePatch();
            const notHere = patchStats(rotatePatch(patch));
            const missing = bestMax(nccMap(bright, rotatePatch(patch)));
            const nums = loggedNumbers(ctx.logs);
            ctx.assert(
              logged(nums, 1, 3e-3),
              'log the best score for patch — a template that is present scores 1.000'
            );
            ctx.assert(
              logged(nums, missing.score, 0.01),
              `log the best score for rotatedPatch too — expected ≈${missing.score.toFixed(3)}, ` +
                `which is what "the best position available" looks like when the template is not there`
            );
            const verdicts = ctx.logs.filter(l => l.type === 'log' && l.text);
            ctx.assert(
              verdicts.some(l => /\btrue\b/.test(l.text)),
              'log the threshold verdict for patch — score >= THRESHOLD is true, and logging that ' +
                'comparison puts a true in the console'
            );
            ctx.assert(
              verdicts.some(l => /\bfalse\b/.test(l.text)),
              `log the threshold verdict for rotatedPatch — ${missing.score.toFixed(3)} does not ` +
                `clear ${THRESHOLD}, so that comparison is false. Without a threshold there is no ` +
                'difference between "found it" and "here is the least bad position".'
            );
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            const plant = { x: 9, y: 40 };
            const bright = brighten(makeScene(ctx.utils, 51217, plant, { x: 74, y: 74 }));
            const patch = makePatch();
            const rotated = rotatePatch(patch);
            const here = patchStats(patch);
            const notHere = patchStats(rotated);
            const found = ctx.kernel(bright, here.centered, here.norm);
            const ref = nccMap(bright, patch);
            for (let y = 0; y < POSITIONS; y++) {
              for (let x = 0; x < POSITIONS; x++) {
                if (Math.abs(found[y][x] - ref[y][x]) <= 3e-3) continue;
                ctx.assertClose(found[y][x], ref[y][x], 3e-3, `position (${x}, ${y})`);
              }
            }
            const best = bestMax(found);
            ctx.assert(
              best.x === plant.x && best.y === plant.y,
              `on a fresh scene patch should peak at (${plant.x}, ${plant.y}), got (${best.x}, ${best.y})`
            );
            ctx.assertClose(best.score, 1, 3e-3, 'the peak score on a fresh scene');
            const missing = bestMax(ctx.kernel(bright, notHere.centered, notHere.norm));
            ctx.assert(
              missing.score < THRESHOLD,
              `on a fresh scene the rotated template should still stay under ${THRESHOLD}, got ${missing.score.toFixed(3)}`
            );
          },
        },
      ],
    },
  ],
};
