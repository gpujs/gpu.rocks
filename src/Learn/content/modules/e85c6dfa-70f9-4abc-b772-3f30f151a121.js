// Module: Optical Flow — uuid e85c6dfa-70f9-4abc-b772-3f30f151a121 (short id e85c6dfa).
// The file name is the uuid; identity lives in the exported object below,
// never in the path. No legacyId: this module post-dates the pre-uuid urls.
//
// Optical Flow — how much did each pixel move between two frames, and when are
// you allowed to believe the answer.
//
// Five tasks: brightness constancy and the three derivatives it needs → the
// aperture problem, made provable on a pair whose true motion is known → the
// 2x2 Lucas-Kanade least-squares solve, one tiny system per pixel → the
// smaller eigenvalue as a confidence score, which is why "good features to
// track" exists → painting the flow field as direction-hue / magnitude-
// saturation.
//
// Kernel-authoring rules (contract): no closures inside kernel functions, only
// numbers / nested number arrays / ImageData as inputs, this.thread.* for
// indexing, this.constants.* for compile-time values (legal as loop bounds),
// image convention image[y][x] = [r, g, b, a] with channels 0-1, 3D output
// [w, h, d] indexed [z][y][x]. Every task passes in CPU mode.
//
// SIGN CONVENTION, stated once and obeyed everywhere in this file:
//   * It = second frame - first frame, at the same pixel;
//   * (u, v) is where the content WENT, in pixels per frame — positive u is
//     rightward (increasing x), positive v is downward (increasing y, because
//     row-major images count rows downward);
//   * so frameB(x, y) = frameA(x - u, y - v), and Ix*u + Iy*v + It = 0.
// Flip either half and every vector comes out backwards, which is exactly the
// mistake the probes below are built to name.

import { ARRAY_LAYOUT } from '../layoutNote.js';
import { plainToImageData } from '../../engine/utils.js';

const SIZE = 64;
const LAST = SIZE - 1;
const WINDOW = 5; // Lucas-Kanade window, 5x5
const HALF = (WINDOW - 1) / 2;
const DET_EPS = 1e-6; // below this a 2x2 system is refused, not solved
const MAX_FLOW = 1.5; // flow magnitude that paints fully saturated

function clampIndex(i) {
  return i < 0 ? 0 : i > LAST ? LAST : i;
}

function level8(v) {
  const r = Math.round(v);
  return r < 0 ? 0 : r > 255 ? 255 : r;
}

// ---- the two scenes --------------------------------------------------------
//
// Both are defined for EVERY integer coordinate, negative ones included, so a
// translated frame is the same function sampled at a shifted position rather
// than a shifted copy with a hole in it. Both return whole 8-bit levels, which
// is what makes the frames survive the trip through an ImageData unchanged —
// so a host-side reference computed from these numbers is exactly what the
// kernel reads, on every backend.

// The working scene: three vertical bands, each a different kind of trouble.
//   x < 16    a low-contrast wash        — nothing to track at all
//   x < 34    strong vertical stripes    — one direction only (the aperture)
//   x >= 34   two-dimensional texture    — corners, the only trustworthy part
// Nothing left of column 34 varies with y, so Iy is EXACTLY zero there, which
// is what makes the confidence story in tasks 3 and 4 provable rather than
// approximate.
function sceneLevel(x, y) {
  if (x < 16) return level8(128 + 4 * Math.sin((2 * Math.PI * x) / 15));
  if (x < 34) return level8(128 + 70 * Math.sin((2 * Math.PI * x) / 14));
  return level8(
    128 + 60 * Math.sin((2 * Math.PI * x) / 20) + 60 * Math.sin((2 * Math.PI * y) / 26)
  );
}

// The aperture scene: a sawtooth ramp running along the (1, 1) diagonal, so
// every iso-intensity line is anti-diagonal and the gradient points the same
// way at every single pixel. Intensity climbs exactly 12 levels per diagonal
// step, which makes the whole demonstration exact arithmetic instead of a
// numerical near-miss.
function apertureLevel(x, y) {
  const step = ((((x + y) % 16) + 16) % 16);
  return 40 + 12 * step;
}

// A scene sampled into two frames: frame B is the same scene shifted by
// (dx, dy), i.e. B(x, y) = scene(x - dx, y - dy) — content that moved right
// and down. Returns two grids of intensities in 0-1.
function framePair(scene, dx, dy) {
  const a = new Array(SIZE);
  const b = new Array(SIZE);
  for (let y = 0; y < SIZE; y++) {
    const rowA = new Array(SIZE);
    const rowB = new Array(SIZE);
    for (let x = 0; x < SIZE; x++) {
      rowA[x] = scene(x, y) / 255;
      rowB[x] = scene(x - dx, y - dy) / 255;
    }
    a[y] = rowA;
    b[y] = rowB;
  }
  return [a, b];
}

// An intensity grid as the one image shape every gpu.js backend puts on the
// GPU. The frames are gray on purpose: r, g and b all carry the same number,
// so a kernel reads intensity straight out of channel 0 with no luminance
// detour, and .plain[y][x][0] is the identical value host-side.
function grayImage(grid) {
  const plain = new Array(SIZE);
  for (let y = 0; y < SIZE; y++) {
    const row = new Array(SIZE);
    for (let x = 0; x < SIZE; x++) {
      const v = grid[y][x];
      row[x] = [v, v, v, 1];
    }
    plain[y] = row;
  }
  return plainToImageData(plain);
}

const mainFrames = () => framePair(sceneLevel, 1, 1);
const apertureFrames = () => framePair(apertureLevel, 4, 0);

// ---- CPU references --------------------------------------------------------

// The three derivative planes, [Ix, Iy, It], each SIZE x SIZE.
//
// `blend` picks which frames the SPATIAL derivatives are measured on:
//   'mid' — the average of the two frames, which is where the derivative
//           actually belongs (halfway between them in time) and what the task
//           asks for;
//   'a' / 'b' — one frame only, the near-miss the probes name.
function derivativeRef(a, b, blend = 'mid') {
  const sample = (y, x) =>
    blend === 'a' ? a[y][x] : blend === 'b' ? b[y][x] : (a[y][x] + b[y][x]) / 2;
  const ix = new Array(SIZE);
  const iy = new Array(SIZE);
  const it = new Array(SIZE);
  for (let y = 0; y < SIZE; y++) {
    const rowX = new Array(SIZE);
    const rowY = new Array(SIZE);
    const rowT = new Array(SIZE);
    const up = clampIndex(y - 1);
    const down = clampIndex(y + 1);
    for (let x = 0; x < SIZE; x++) {
      const left = clampIndex(x - 1);
      const right = clampIndex(x + 1);
      rowX[x] = (sample(y, right) - sample(y, left)) / 2;
      rowY[x] = (sample(down, x) - sample(up, x)) / 2;
      rowT[x] = b[y][x] - a[y][x];
    }
    ix[y] = rowX;
    iy[y] = rowY;
    it[y] = rowT;
  }
  return [ix, iy, it];
}

// The five window sums Lucas-Kanade needs, clamp-to-edge like every other
// neighbourhood pass in this course.
function windowSums(d, x, y) {
  let sxx = 0;
  let sxy = 0;
  let syy = 0;
  let sxt = 0;
  let syt = 0;
  for (let wy = 0; wy < WINDOW; wy++) {
    const sy = clampIndex(y + wy - HALF);
    for (let wx = 0; wx < WINDOW; wx++) {
      const sx = clampIndex(x + wx - HALF);
      const ix = d[0][sy][sx];
      const iy = d[1][sy][sx];
      const it = d[2][sy][sx];
      sxx += ix * ix;
      sxy += ix * iy;
      syy += iy * iy;
      sxt += ix * it;
      syt += iy * it;
    }
  }
  return { sxx, sxy, syy, sxt, syt };
}

function detOf(s) {
  return s.sxx * s.syy - s.sxy * s.sxy;
}

// Lucas-Kanade flow, [U, V]. `variant` selects a deliberate mistake for the
// probes: 'nocross' drops the Ix*Iy term, 'swap' exchanges u and v.
function flowRef(d, variant = 'ok') {
  const u = new Array(SIZE);
  const v = new Array(SIZE);
  for (let y = 0; y < SIZE; y++) {
    const rowU = new Array(SIZE);
    const rowV = new Array(SIZE);
    for (let x = 0; x < SIZE; x++) {
      const s = windowSums(d, x, y);
      if (variant === 'nocross') {
        // Ix*Iy never accumulated: the 2x2 system collapses to two
        // independent 1x1 ones.
        rowU[x] = s.sxx === 0 ? 0 : -s.sxt / s.sxx;
        rowV[x] = s.syy === 0 ? 0 : -s.syt / s.syy;
        continue;
      }
      const det = detOf(s);
      if (Math.abs(det) < DET_EPS) {
        rowU[x] = 0;
        rowV[x] = 0;
        continue;
      }
      const solvedU = (s.sxy * s.syt - s.syy * s.sxt) / det;
      const solvedV = (s.sxy * s.sxt - s.sxx * s.syt) / det;
      rowU[x] = variant === 'swap' ? solvedV : solvedU;
      rowV[x] = variant === 'swap' ? solvedU : solvedV;
    }
    u[y] = rowU;
    v[y] = rowV;
  }
  return [u, v];
}

// Per-pixel normal flow — the shortest (u, v) satisfying one pixel's equation.
function normalFlowRef(d) {
  const u = new Array(SIZE);
  const v = new Array(SIZE);
  for (let y = 0; y < SIZE; y++) {
    const rowU = new Array(SIZE);
    const rowV = new Array(SIZE);
    for (let x = 0; x < SIZE; x++) {
      const ix = d[0][y][x];
      const iy = d[1][y][x];
      const it = d[2][y][x];
      const g = ix * ix + iy * iy;
      rowU[x] = (-it * ix) / g;
      rowV[x] = (-it * iy) / g;
    }
    u[y] = rowU;
    v[y] = rowV;
  }
  return [u, v];
}

// Confidence: the SMALLER eigenvalue of the 2x2 window matrix. `variant`
// 'max' is the larger one, 'det' the determinant — the two near-misses.
function confidenceRef(d, variant = 'min') {
  const out = new Array(SIZE);
  for (let y = 0; y < SIZE; y++) {
    const row = new Array(SIZE);
    for (let x = 0; x < SIZE; x++) {
      const s = windowSums(d, x, y);
      const trace = s.sxx + s.syy;
      const det = detOf(s);
      const root = Math.sqrt(Math.max(0, trace * trace - 4 * det));
      row[x] =
        variant === 'det' ? det : variant === 'max' ? (trace + root) / 2 : (trace - root) / 2;
    }
    out[y] = row;
  }
  return out;
}

// The painted colour for one flow vector, exactly as task 5 specifies it:
// direction picks a hue, magnitude picks how far from white it travels.
// Returns three 0-255 bytes.
function flowColour(u, v) {
  const mag = Math.sqrt(u * u + v * v);
  const s = Math.min(1, mag / MAX_FLOW);
  const hh = (Math.atan2(v, u) / (2 * Math.PI) + 0.5) * 6;
  let r = 0;
  let g = 0;
  let b = 0;
  if (hh < 1) {
    r = 1;
    g = hh;
  } else if (hh < 2) {
    r = 2 - hh;
    g = 1;
  } else if (hh < 3) {
    g = 1;
    b = hh - 2;
  } else if (hh < 4) {
    g = 4 - hh;
    b = 1;
  } else if (hh < 5) {
    r = hh - 4;
    b = 1;
  } else {
    r = 1;
    b = 6 - hh;
  }
  return [1 - s + s * r, 1 - s + s * g, 1 - s + s * b].map(c => Math.round(c * 255));
}

// A constant flow field, for orientation-independent colour checks.
function constantFlow(u, v) {
  const plane = value => {
    const grid = new Array(SIZE);
    for (let y = 0; y < SIZE; y++) grid[y] = new Array(SIZE).fill(value);
    return grid;
  };
  return [plane(u), plane(v)];
}

// ---- near-miss diagnosis ---------------------------------------------------
//
// The course's shared discipline: when a failing value is exactly what some
// specific mistake would produce, name that mistake instead of reporting two
// numbers. A probe pairs such a value with its sentence; diagnose() speaks only
// when the observation matches a probe within the test's own tolerance AND the
// correct value does not — so a cell where two candidates coincide (anywhere Ix
// happens to equal Iy, say) stays silent, as do observations matching probes
// that disagree with each other. A wrong diagnosis is worse than none.
function diagnose(got, expected, eps, probes) {
  const hits = probes
    .filter(p => Number.isFinite(p[0]) && Math.abs(got - p[0]) <= eps && Math.abs(expected - p[0]) > eps)
    .map(p => p[1]);
  return hits.length && hits.every(m => m === hits[0]) ? hits[0] : null;
}

// A 0/0 or a division by a hair — both mean the same missing guard, and both
// are visible without knowing the right answer.
function blowUpHint(got, expected, limit = 1e3) {
  if (Number.isFinite(got) && Math.abs(got) < limit) return null;
  return Math.abs(expected) < limit
    ? 'that vector is NaN or astronomically large — the 2x2 system was solved without checking ' +
        'the determinant first. Where the window is flat or all one edge the determinant is zero, ' +
        'and dividing by it produces exactly this. Return 0 when Math.abs(det) is below ' +
        'this.constants.eps.'
    : null;
}

// Task 1: the two spatial planes read off each other.
const AXIS_SWAP =
  'that is the OTHER spatial plane at this pixel — Ix and Iy are swapped. Plane 0 differs along ' +
  'x (left/right neighbours), plane 1 along y (up/down): image[y][x] puts the row first.';

const TIME_BACKWARDS =
  'that is exactly the negative of the answer — the temporal derivative is signed backwards. ' +
  'It is the SECOND frame minus the first, so a pixel that got brighter has It > 0. Flip it and ' +
  'every flow vector in the module points the wrong way.';

const ONE_FRAME_ONLY =
  'that is one frame\'s gradient on its own — all three derivatives have to describe the same ' +
  'instant, the one BETWEEN the two frames, so the spatial ones are measured on the average of ' +
  'frameA and frameB. Using a single frame scatters every flow vector this module goes on to ' +
  'compute — about six times the error, for one extra pair of reads.';

const UNDIVIDED =
  'that is twice the answer — a central difference spans TWO pixels (x - 1 to x + 1), so the ' +
  'difference is divided by 2.';

const GRADIENT_NEGATED =
  'that is exactly the negative of the answer — the two neighbours are subtracted the wrong way ' +
  'round. A central difference is (further along the axis) minus (further back): right - left, ' +
  'and down - up.';

// Task 2 and 3: the sign of the whole flow vector.
const FLOW_NEGATED =
  'every component is exactly negated — this course reports where the content WENT, so a scene ' +
  'sliding right has u > 0. That is the sign that makes Ix*u + Iy*v + It = 0 come out zero; drop ' +
  'the minus in front of It and you get the reverse.';

const UV_SWAPPED =
  'that is the other component of the correct vector — u and v are exchanged, which transposes ' +
  'the whole flow field. u pairs with Ix (plane 0, the x gradient) and v with Iy (plane 1).';

const NO_CROSS_TERM =
  'that is what you get when Sxy — the sum of Ix*Iy — never gets accumulated: the 2x2 system ' +
  'falls apart into two independent divisions. The off-diagonal term is what couples u and v, ' +
  'and it is almost never zero.';

// Task 4: the wrong eigenvalue, or no eigenvalue at all.
const LARGER_EIGENVALUE =
  'that is the LARGER eigenvalue. It is big along the stripe band too — an edge has one strong ' +
  'direction and one dead one — which is exactly the case confidence has to reject. The smaller ' +
  'root is the one that has to be large: (trace - sqrt(trace*trace - 4*det)) / 2.';

const RAW_DETERMINANT =
  'that is the determinant itself, not the smaller eigenvalue. The determinant is the PRODUCT of ' +
  'the two eigenvalues, so a huge one and a tiny one can multiply up to a respectable number.';

export default {
  uuid: 'e85c6dfa-70f9-4abc-b772-3f30f151a121',
  version: 1,
  slug: 'optical-flow',
  title: 'Optical Flow',
  blurb:
    'Per-pixel motion between two frames: the aperture problem, a 2&times;2 least-squares solve per thread, and knowing when not to believe the answer.',
  tasks: [
    // ------------------------------------------------------------------ 1
    {
      slug: 'brightness-constancy',
      title: 'One Equation, Two Unknowns',
      intro: `<p>Optical flow asks a simple-sounding question: for every pixel of frame 1, where did
        it go in frame 2? The only assumption anyone can make is <strong>brightness constancy</strong> —
        a moving point keeps its intensity, it just shows up somewhere else. Write that down and
        expand it to first order and you get one equation per pixel:</p>
<pre><code>Ix·u + Iy·v + It = 0</code></pre>
        <p><code>Ix</code> and <code>Iy</code> are the spatial gradients — the same central differences
        the Sobel pass in Convolution &amp; Filters is built from — and <code>It</code> is how much this
        pixel's intensity changed between the frames. <code>u</code> and <code>v</code> are what you
        want, and there is only one equation for the two of them; every method in this module is a
        different way of buying a second. One subtlety first, though: all three derivatives have to
        describe the <em>same instant</em>, the moment halfway between the frames. <code>It</code>
        naturally sits there, so the spatial gradients are measured on the <strong>average of the two
        frames</strong>. Take them from frame 1 alone and the answers scatter — on these frames the
        typical error goes from about 0.03 pixels to about 0.19.</p>
        <p>The frames arrive as task inputs rather than from a camera, and that is a wall rather than
        a shortcut: your code runs inside a Web Worker, which has no <code>navigator.mediaDevices</code>,
        no <code>getUserMedia</code> and no <code>&lt;video&gt;</code> element. On an ordinary page the
        real thing is short — <code>getUserMedia</code> into a <code>&lt;video&gt;</code>, then pass
        that element straight to a kernel as an argument, which gpu.js accepts as an image source, and
        keep the previous frame around. Everything below that point is identical.</p>
        ${ARRAY_LAYOUT}`,
      goal: `<strong>Goal:</strong> fill a 3-plane derivative field for the pair
        <code>frameA</code>/<code>frameB</code> — plane 0 is <code>Ix</code>, plane 1 is
        <code>Iy</code>, plane 2 is <code>It</code>.`,
      requirements: [
        'Keep <code>output: [64, 64, 3]</code> — the result is indexed <code>d[z][y][x]</code>, one plane per derivative',
        'Plane 0 (<code>this.thread.z === 0</code>): <code>Ix</code> = <code>(right − left) / 2</code>, where <em>left</em> and <em>right</em> are the <strong>average of the two frames</strong> at <code>x - 1</code> and <code>x + 1</code>, both clamped to <code>0…this.constants.last</code>',
        'Plane 1: <code>Iy</code>, the same central difference down the <code>y</code> axis',
        'Plane 2: <code>It</code> = this pixel in the <strong>second</strong> frame minus the same pixel in the first',
        'The frames are gray, so a pixel\'s intensity is just its channel 0',
      ],
      hints: [
        {
          title: 'Hint 1 — reading one intensity',
          body: `<p>An image cell is an <code>[r, g, b, a]</code> array, and these frames are gray,
            so all you need is the first channel:</p>
<pre><code>const a = frameA[y][x];
const intensity = a[0];</code></pre>`,
        },
        {
          title: 'Hint 2 — the horizontal gradient',
          body: `<p>Clamp the two neighbour columns like every other neighbourhood pass in the
            course, average the frames at each of them, then take the central difference:</p>
<pre><code>let left = x - 1;
if (left &lt; 0) left = 0;
let right = x + 1;
if (right &gt; this.constants.last) right = this.constants.last;

const aL = frameA[y][left];
const bL = frameB[y][left];
const aR = frameA[y][right];
const bR = frameB[y][right];
const midLeft = (aL[0] + bL[0]) / 2;
const midRight = (aR[0] + bR[0]) / 2;
return (midRight - midLeft) / 2;</code></pre>`,
        },
        {
          title: 'Hint 3 — the temporal one',
          body: `<p>No neighbours at all — the same pixel, the two frames, second minus first:</p>
<pre><code>const a = frameA[y][x];
const b = frameB[y][x];
return b[0] - a[0];</code></pre>
          <p>That order is the whole sign convention of this module. Reverse it and every flow
            vector you compute from here on points backwards.</p>`,
        },
      ],
      transfer: `Packing several per-pixel quantities into the planes of one output is how every
        real pipeline does it: a CUDA kernel writes an <code>float2</code>/<code>float4</code>
        surface, a WGSL compute shader writes an <code>rgba16float</code> storage texture, Metal
        writes an MTLTexture with the gradients in RG and the time difference in B. One pass, one
        launch, three fields.`,
      starterCode: `// Three derivative planes for one pair of frames: Ix, Iy, It.
const gpu = new GPU({ mode });

const derivatives = gpu.createKernel(function (frameA, frameB) {
  const x = this.thread.x;
  const y = this.thread.y;

  if (this.thread.z === 0) {
    // TODO: Ix — central difference along x, measured on the AVERAGE
    // of frameA and frameB, neighbour columns clamped to 0…this.constants.last
    return 0;
  }

  if (this.thread.z === 1) {
    // TODO: Iy — the same central difference, but along y
    return 0;
  }

  // TODO: It — this pixel in the SECOND frame minus the same pixel in the first
  return 0;
}, {
  output: [64, 64, 3],
  constants: { last: 63 },
});

const d = await derivatives(frameA, frameB);
console.log('Ix at (45, 30):', d[0][30][45]);
console.log('Iy at (45, 30):', d[1][30][45]);
console.log('It at (45, 30):', d[2][30][45]);
`,
      solutionCode: `// Three derivative planes for one pair of frames: Ix, Iy, It.
const gpu = new GPU({ mode });

const derivatives = gpu.createKernel(function (frameA, frameB) {
  const x = this.thread.x;
  const y = this.thread.y;

  if (this.thread.z === 0) {
    let left = x - 1;
    if (left < 0) left = 0;
    let right = x + 1;
    if (right > this.constants.last) right = this.constants.last;
    const aL = frameA[y][left];
    const bL = frameB[y][left];
    const aR = frameA[y][right];
    const bR = frameB[y][right];
    const midLeft = (aL[0] + bL[0]) / 2;
    const midRight = (aR[0] + bR[0]) / 2;
    return (midRight - midLeft) / 2;
  }

  if (this.thread.z === 1) {
    let up = y - 1;
    if (up < 0) up = 0;
    let down = y + 1;
    if (down > this.constants.last) down = this.constants.last;
    const aU = frameA[up][x];
    const bU = frameB[up][x];
    const aD = frameA[down][x];
    const bD = frameB[down][x];
    const midUp = (aU[0] + bU[0]) / 2;
    const midDown = (aD[0] + bD[0]) / 2;
    return (midDown - midUp) / 2;
  }

  const a = frameA[y][x];
  const b = frameB[y][x];
  return b[0] - a[0];
}, {
  output: [64, 64, 3],
  constants: { last: 63 },
});

const d = await derivatives(frameA, frameB);
console.log('Ix at (45, 30):', d[0][30][45]);
console.log('Iy at (45, 30):', d[1][30][45]);
console.log('It at (45, 30):', d[2][30][45]);
`,
      inputs: () => {
        const [a, b] = mainFrames();
        return { frameA: grayImage(a), frameB: grayImage(b) };
      },
      publicTests: [
        {
          name: 'produces three <code>64×64</code> planes',
          run: async ctx => {
            ctx.assert(ctx.kernels.length >= 1, 'no kernel was created — call gpu.createKernel()');
            const [a, b] = mainFrames();
            const out = await ctx.kernel(grayImage(a), grayImage(b));
            ctx.assert(out && out.length === 3, `expected 3 planes, got ${out && out.length}`);
            ctx.assert(out[0] && out[0].length === 64, 'each plane should hold 64 rows');
            ctx.assert(out[0][0] && out[0][0].length === 64, 'each row should hold 64 values');
          },
        },
        {
          name: 'the left half of the scene has no vertical structure — <code>Iy</code> is exactly 0 there',
          run: async ctx => {
            const [a, b] = mainFrames();
            const ref = derivativeRef(a, b);
            const out = await ctx.kernel(grayImage(a), grayImage(b));
            // Columns 0-31 are a wash and a set of vertical stripes: nothing
            // changes down a column, so Iy must vanish and Ix must not. A pair
            // of planes read off each other fails this instantly.
            for (const y of [4, 19, 33, 48, 61]) {
              for (const x of [3, 9, 21, 27, 31]) {
                const hint = Math.abs(out[1][y][x] - ref[0][y][x]) <= 1e-5 ? AXIS_SWAP : null;
                ctx.assertClose(out[1][y][x], 0, 1e-5, hint || `Iy at (${x}, ${y})`);
                ctx.assert(
                  Math.abs(out[0][y][x]) > 1e-4,
                  `Ix at (${x}, ${y}) is ${out[0][y][x]} — the stripes there are a strong ` +
                    `horizontal gradient, so Ix cannot be zero`
                );
              }
            }
          },
        },
        {
          name: 'all three planes match the central differences',
          run: async ctx => {
            const [a, b] = mainFrames();
            const out = await ctx.kernel(grayImage(a), grayImage(b));
            const ref = derivativeRef(a, b);
            const fromA = derivativeRef(a, b, 'a');
            const fromB = derivativeRef(a, b, 'b');
            const eps = 1e-5;
            // Cells chosen so the near-miss candidates stay TELLABLE APART: at
            // a cell where, say, twice the answer happens to equal frame B's
            // lone gradient, diagnose() has to fall silent and the learner gets
            // two bare numbers instead of a sentence.
            const cases = [[50, 12], [43, 6], [52, 44], [60, 50], [38, 40], [24, 20], [8, 30], [0, 0], [63, 63]];
            for (const [x, y] of cases) {
              for (let z = 0; z < 3; z++) {
                const expected = ref[z][y][x];
                const probes = [[-expected, z === 2 ? TIME_BACKWARDS : GRADIENT_NEGATED]];
                if (z < 2) {
                  probes.push([ref[1 - z][y][x], AXIS_SWAP]);
                  probes.push([fromA[z][y][x], ONE_FRAME_ONLY]);
                  probes.push([fromB[z][y][x], ONE_FRAME_ONLY]);
                  probes.push([2 * expected, UNDIVIDED]);
                }
                const hint = diagnose(out[z][y][x], expected, eps, probes);
                ctx.assertClose(
                  out[z][y][x],
                  expected,
                  eps,
                  hint || `plane ${z} at (${x}, ${y})`
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
            const [a, b] = mainFrames();
            const out = await ctx.kernel(grayImage(a), grayImage(b));
            const ref = derivativeRef(a, b);
            const fromA = derivativeRef(a, b, 'a');
            const eps = 1e-5;
            for (let z = 0; z < 3; z++) {
              for (let y = 0; y < 64; y++) {
                for (let x = 0; x < 64; x++) {
                  const expected = ref[z][y][x];
                  const probes = [[-expected, z === 2 ? TIME_BACKWARDS : GRADIENT_NEGATED]];
                  if (z < 2) {
                    probes.push([ref[1 - z][y][x], AXIS_SWAP]);
                    probes.push([fromA[z][y][x], ONE_FRAME_ONLY]);
                    probes.push([2 * expected, UNDIVIDED]);
                  }
                  const hint = diagnose(out[z][y][x], expected, eps, probes);
                  ctx.assertClose(out[z][y][x], expected, eps, hint || `plane ${z} at (${x}, ${y})`);
                }
              }
            }
          },
        },
        {
          name: 'private test #2',
          run: async ctx => {
            // A second scene, with the motion along the diagonal instead of
            // down-and-right: the same kernel, different numbers.
            const [a, b] = apertureFrames();
            const out = await ctx.kernel(grayImage(a), grayImage(b));
            const ref = derivativeRef(a, b);
            const eps = 1e-5;
            for (let z = 0; z < 3; z++) {
              for (let y = 0; y < 64; y += 3) {
                for (let x = 0; x < 64; x += 3) {
                  const expected = ref[z][y][x];
                  const hint = diagnose(out[z][y][x], expected, eps, [
                    [-expected, z === 2 ? TIME_BACKWARDS : GRADIENT_NEGATED],
                  ]);
                  ctx.assertClose(out[z][y][x], expected, eps, hint || `plane ${z} at (${x}, ${y})`);
                }
              }
            }
          },
        },
      ],
    },

    // ------------------------------------------------------------------ 2
    {
      slug: 'aperture-problem',
      title: 'The Aperture Problem',
      intro: `<p>One equation, two unknowns, is not "nearly enough information". It is a
        <strong>line</strong> of answers. <code>Ix·u + Iy·v + It = 0</code> is the equation of a
        straight line in the <code>(u, v)</code> plane, and every point on it fits this pixel's
        evidence equally well. Looking harder at the pixel will not narrow it down; the information
        is not there.</p>
        <p>What <em>is</em> there is the component of the motion <strong>along the gradient</strong>.
        Perpendicular to the gradient the intensity does not change, so sliding that way is
        invisible. Pick the shortest vector on the line and you get the <strong>normal flow</strong>:</p>
<pre><code>u = −It · Ix / (Ix² + Iy²)
v = −It · Iy / (Ix² + Iy²)</code></pre>
        <p><code>derivs</code> here comes from a pair you can check by hand. The scene is a sawtooth
        ramp climbing 12 levels per step along the <code>(1, 1)</code> diagonal — so every edge in it
        runs anti-diagonally — and frame 2 is that scene moved <strong>4 pixels to the right and 0
        down</strong>. You know the answer. The pixel does not, and cannot.</p>`,
      goal: `<strong>Goal:</strong> compute the normal flow for every pixel — plane 0 is
        <code>u</code>, plane 1 is <code>v</code> — and watch it fail to find a motion you know
        exactly.`,
      requirements: [
        'Keep <code>output: [64, 64, 2]</code>: plane 0 is <code>u</code>, plane 1 is <code>v</code>',
        'Read this pixel\'s three derivatives from <code>derivs[0][y][x]</code>, <code>derivs[1][y][x]</code>, <code>derivs[2][y][x]</code>',
        'Return <code>-It * Ix / (Ix*Ix + Iy*Iy)</code> for plane 0 and <code>-It * Iy / (Ix*Ix + Iy*Iy)</code> for plane 1',
        '<strong>Sign convention:</strong> <code>(u, v)</code> is where the content <em>went</em>, in pixels per frame — positive <code>u</code> is rightward, positive <code>v</code> is downward. That is the sign that makes <code>Ix·u + Iy·v + It</code> come out zero',
      ],
      hints: [
        {
          title: 'Hint 1 — one pixel, no neighbours',
          body: `<p>Nothing is gathered here: thread <code>(x, y)</code> reads exactly three numbers
            and returns one. The whole body fits in six lines.</p>`,
        },
        {
          title: 'Hint 2 — name the squared gradient once',
          body: `<pre><code>const ix = derivs[0][y][x];
const iy = derivs[1][y][x];
const it = derivs[2][y][x];
const g = ix * ix + iy * iy;</code></pre>
          <p>then plane 0 returns <code>(-it * ix) / g</code> and plane 1 returns
            <code>(-it * iy) / g</code>. This pair has a gradient at every pixel, so no guard is
            needed yet — task 3 is where that stops being true.</p>`,
        },
      ],
      transfer: `Normal flow is not a toy: it is what a single-pixel constraint can honestly give
        you, and it is the per-thread starting point every dense-flow implementation refines —
        CUDA's <code>NVOF</code> optical-flow engine, OpenCV's <code>calcOpticalFlowFarneback</code>,
        the motion-vector passes inside DLSS and FSR. All of them add an assumption on top; none of
        them can conjure the missing component out of one pixel.`,
      starterCode: `// Normal flow: the shortest vector that satisfies one pixel's equation.
const gpu = new GPU({ mode });

const normalFlow = gpu.createKernel(function (derivs) {
  const x = this.thread.x;
  const y = this.thread.y;
  const ix = derivs[0][y][x];
  const iy = derivs[1][y][x];
  const it = derivs[2][y][x];

  // TODO: plane 0 (this.thread.z === 0) returns u, plane 1 returns v.
  // u = -it * ix / (ix*ix + iy*iy),  v = -it * iy / (ix*ix + iy*iy)
  return 0;
}, {
  output: [64, 64, 2],
});

const flow = await normalFlow(derivs);
console.log('true motion: (4, 0)');
console.log('estimated at (30, 21):', flow[0][21][30], flow[1][21][30]);
`,
      solutionCode: `// Normal flow: the shortest vector that satisfies one pixel's equation.
const gpu = new GPU({ mode });

const normalFlow = gpu.createKernel(function (derivs) {
  const x = this.thread.x;
  const y = this.thread.y;
  const ix = derivs[0][y][x];
  const iy = derivs[1][y][x];
  const it = derivs[2][y][x];
  const g = ix * ix + iy * iy;

  if (this.thread.z === 0) {
    return (-it * ix) / g;
  }
  return (-it * iy) / g;
}, {
  output: [64, 64, 2],
});

const flow = await normalFlow(derivs);
console.log('true motion: (4, 0)');
console.log('estimated at (30, 21):', flow[0][21][30], flow[1][21][30]);
`,
      inputs: () => {
        const [a, b] = apertureFrames();
        return { derivs: derivativeRef(a, b) };
      },
      publicTests: [
        {
          name: 'produces two <code>64×64</code> planes',
          run: async ctx => {
            ctx.assert(ctx.kernels.length >= 1, 'no kernel was created — call gpu.createKernel()');
            const [a, b] = apertureFrames();
            const out = await ctx.kernel(derivativeRef(a, b));
            ctx.assert(out && out.length === 2, `expected 2 planes, got ${out && out.length}`);
            ctx.assert(out[0] && out[0].length === 64, 'each plane should hold 64 rows');
            ctx.assert(out[0][0] && out[0][0].length === 64, 'each row should hold 64 values');
          },
        },
        {
          name: 'the true motion was <code>(4, 0)</code>; the estimate is <code>(2, 2)</code>',
          run: async ctx => {
            const [a, b] = apertureFrames();
            const d = derivativeRef(a, b);
            const out = await ctx.kernel(d);
            // Away from the sawtooth's reset the ramp is exactly linear, so the
            // normal flow is exactly (2, 2): the projection of the true (4, 0)
            // onto the gradient direction. Nothing approximate about it.
            for (const [x, y] of [[20, 15], [30, 21], [40, 35], [50, 44], [25, 50], [12, 45]]) {
              for (let z = 0; z < 2; z++) {
                const hint = diagnose(out[z][y][x], 2, 2e-3, [[-2, FLOW_NEGATED]]);
                ctx.assertClose(
                  out[z][y][x],
                  2,
                  2e-3,
                  hint ||
                    `${z === 0 ? 'u' : 'v'} at (${x}, ${y}) — the scene's edges all run ` +
                      `anti-diagonally, so the recoverable part of a (4, 0) motion is (2, 2)`
                );
              }
            }
            const u = out[0][21][30];
            const v = out[1][21][30];
            ctx.assert(
              Math.abs(u - 4) > 0.5 || Math.abs(v) > 0.5,
              `the estimate came out as the true motion (4, 0), which one pixel cannot know — ` +
                `check the formula rather than celebrating`
            );
          },
        },
        {
          name: 'the estimate satisfies the equation — and so does the true motion',
          run: async ctx => {
            const [a, b] = apertureFrames();
            const d = derivativeRef(a, b);
            const out = await ctx.kernel(d);
            // The point of the whole task: the constraint does not single out an
            // answer. Your (2, 2), the real (4, 0) and the absurd (0, 4) all
            // drive the residual to zero, so the pixel has no way to choose.
            for (const [x, y] of [[20, 15], [30, 21], [40, 35], [25, 50]]) {
              const ix = d[0][y][x];
              const iy = d[1][y][x];
              const it = d[2][y][x];
              const residual = (u, v) => Math.abs(ix * u + iy * v + it);
              ctx.assert(
                residual(out[0][y][x], out[1][y][x]) < 2e-4,
                `at (${x}, ${y}) your (u, v) leaves Ix·u + Iy·v + It = ` +
                  `${residual(out[0][y][x], out[1][y][x]).toFixed(5)}, which should be 0`
              );
              ctx.assert(residual(4, 0) < 1e-9, `internal: (4, 0) should satisfy the equation at (${x}, ${y})`);
              ctx.assert(residual(0, 4) < 1e-9, `internal: (0, 4) should satisfy the equation at (${x}, ${y})`);
            }
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            const [a, b] = apertureFrames();
            const d = derivativeRef(a, b);
            const out = await ctx.kernel(d);
            const [refU, refV] = normalFlowRef(d);
            for (let y = 0; y < 64; y++) {
              for (let x = 0; x < 64; x++) {
                const expected = [refU[y][x], refV[y][x]];
                for (let z = 0; z < 2; z++) {
                  const eps = 2e-3 + 2e-3 * Math.abs(expected[z]);
                  const hint = diagnose(out[z][y][x], expected[z], eps, [
                    [-expected[z], FLOW_NEGATED],
                  ]);
                  ctx.assertClose(
                    out[z][y][x],
                    expected[z],
                    eps,
                    hint || `${z === 0 ? 'u' : 'v'} at (${x}, ${y})`
                  );
                }
              }
            }
          },
        },
        {
          name: 'private test #2',
          run: async ctx => {
            // A scene where Ix and Iy genuinely differ, so pairing u with the
            // wrong gradient stops being invisible.
            const [a, b] = mainFrames();
            const d = derivativeRef(a, b);
            const out = await ctx.kernel(d);
            const [refU, refV] = normalFlowRef(d);
            for (let y = 0; y < 64; y++) {
              for (let x = 0; x < 64; x++) {
                const g = d[0][y][x] * d[0][y][x] + d[1][y][x] * d[1][y][x];
                if (g < 1e-12) continue; // no gradient at all: the formula is 0/0 here
                const expected = [refU[y][x], refV[y][x]];
                for (let z = 0; z < 2; z++) {
                  const eps = 2e-3 + 2e-3 * Math.abs(expected[z]);
                  const hint = diagnose(out[z][y][x], expected[z], eps, [
                    [-expected[z], FLOW_NEGATED],
                    [expected[1 - z], UV_SWAPPED],
                  ]);
                  ctx.assertClose(
                    out[z][y][x],
                    expected[z],
                    eps,
                    hint || `${z === 0 ? 'u' : 'v'} at (${x}, ${y})`
                  );
                }
              }
            }
          },
        },
      ],
    },

    // ------------------------------------------------------------------ 3
    {
      slug: 'lucas-kanade',
      title: 'Lucas–Kanade: Buy a Second Equation',
      intro: `<p>You cannot get more information out of one pixel, so buy it from the neighbours.
        <strong>Lucas–Kanade</strong> assumes that everything inside a small window moves together.
        A 5×5 window gives 25 copies of <code>Ix·u + Iy·v + It = 0</code> sharing one unknown
        <code>(u, v)</code> — 25 equations, 2 unknowns, over-determined instead of
        under-determined. Least squares turns that into a 2×2 system:</p>
<pre><code>| Sxx  Sxy |   | u |        | Sxt |
|          | · |   |  =  −  |     |
| Sxy  Syy |   | v |        | Syt |</code></pre>
        <p>where <code>Sxx = Σ Ix²</code>, <code>Sxy = Σ Ix·Iy</code>, <code>Syy = Σ Iy²</code>,
        <code>Sxt = Σ Ix·It</code> and <code>Syt = Σ Iy·It</code>, all summed over the window. A 2×2
        system has a closed form, so there is no solver and no iteration — just a determinant:</p>
<pre><code>det = Sxx·Syy − Sxy²
u   = (Sxy·Syt − Syy·Sxt) / det
v   = (Sxy·Sxt − Sxx·Syt) / det</code></pre>
        <p>Every pixel solves its own tiny system, reading only its own neighbourhood and writing
        only its own cell. That is the ideal GPU shape: 4,096 independent 2×2 solves with no
        coordination whatsoever — the gather formulation, exactly as in Convolution &amp; Filters,
        with a little linear algebra at the end.</p>
        <p>And a warning the input is built to deliver. <code>det</code> goes to <strong>zero</strong>
        where the window has nothing to say: a flat patch (no gradient at all) or a stretch of
        parallel edges (all gradients pointing the same way — task 2's problem, unchanged by a
        bigger window). Divide by it anyway and you get NaN or vectors hundreds of pixels long.
        Guard it.</p>`,
      goal: `<strong>Goal:</strong> solve the 5×5 Lucas–Kanade system for every pixel — plane 0 is
        <code>u</code>, plane 1 is <code>v</code> — and return <code>0</code> wherever the
        determinant is too small to trust.`,
      requirements: [
        'Accumulate <code>Sxx</code>, <code>Sxy</code>, <code>Syy</code>, <code>Sxt</code>, <code>Syt</code> over the 5×5 window, sample indexes clamped to <code>0…this.constants.last</code>',
        'Compute <code>det = Sxx * Syy - Sxy * Sxy</code>',
        'If <code>Math.abs(det) &lt; this.constants.eps</code>, return <code>0</code> — the window has nothing to say',
        'Otherwise plane 0 returns <code>(Sxy * Syt - Syy * Sxt) / det</code> and plane 1 returns <code>(Sxy * Sxt - Sxx * Syt) / det</code>',
        'Same sign convention as task 2: <code>(u, v)</code> is where the content went, positive <code>u</code> rightward and positive <code>v</code> downward',
      ],
      hints: [
        {
          title: 'Hint 1 — the window loop',
          body: `<p>The same clamped double loop the box blur used, five wide instead of three:</p>
<pre><code>for (let wy = 0; wy &lt; 5; wy++) {
  for (let wx = 0; wx &lt; 5; wx++) {
    let sy = this.thread.y + wy - 2;
    if (sy &lt; 0) sy = 0;
    if (sy &gt; this.constants.last) sy = this.constants.last;
    // …same for sx, then read the three derivatives at [sy][sx]
  }
}</code></pre>`,
        },
        {
          title: 'Hint 2 — five running sums, not two',
          body: `<p>Inside the loop, all five products accumulate together:</p>
<pre><code>const ix = derivs[0][sy][sx];
const iy = derivs[1][sy][sx];
const it = derivs[2][sy][sx];
sxx += ix * ix;
sxy += ix * iy;
syy += iy * iy;
sxt += ix * it;
syt += iy * it;</code></pre>
          <p><code>sxy</code> is the one that gets forgotten. It is the off-diagonal term — the
            thing that couples <code>u</code> and <code>v</code> — and dropping it silently turns
            the 2×2 solve into two unrelated divisions.</p>`,
        },
        {
          title: 'Hint 3 — the guard and the two returns',
          body: `<pre><code>const det = sxx * syy - sxy * sxy;
if (Math.abs(det) &lt; this.constants.eps) {
  return 0;
}
if (this.thread.z === 0) {
  return (sxy * syt - syy * sxt) / det;
}
return (sxy * sxt - sxx * syt) / det;</code></pre>
          <p>Compute the sums once, before the branch on <code>z</code> — both components need all
            five of them.</p>`,
        },
      ],
      transfer: `This is the shape GPUs were built for: a fixed-size neighbourhood read, a handful
        of registers, a closed-form solve, no synchronisation. A CUDA implementation stages the
        derivative tiles in shared memory and keeps the five sums in registers; a WGSL compute
        shader does the same with a workgroup <code>var&lt;workgroup&gt;</code> tile. gpu.js has
        neither, so every thread re-reads its own window — more traffic, identical answer, and the
        algorithm is unchanged.`,
      starterCode: `// One 2×2 least-squares solve per pixel, from a 5×5 window of equations.
const gpu = new GPU({ mode });

const lucasKanade = gpu.createKernel(function (derivs) {
  const x = this.thread.x;
  const y = this.thread.y;

  let sxx = 0;
  let sxy = 0;
  let syy = 0;
  let sxt = 0;
  let syt = 0;

  // TODO: loop over the 5×5 window around (x, y), clamping both sample
  // coordinates to 0…this.constants.last, and accumulate the five sums.

  // TODO: det = sxx * syy - sxy * sxy;
  //       return 0 when Math.abs(det) < this.constants.eps,
  //       otherwise plane 0 → (sxy * syt - syy * sxt) / det
  //                 plane 1 → (sxy * sxt - sxx * syt) / det
  return 0;
}, {
  output: [64, 64, 2],
  constants: { last: 63, eps: 1e-6 },
});

const flow = await lucasKanade(derivs);
console.log('textured region, true motion is (1, 1):', flow[0][30][45], flow[1][30][45]);
console.log('flat region:', flow[0][10][6], flow[1][10][6]);
`,
      solutionCode: `// One 2×2 least-squares solve per pixel, from a 5×5 window of equations.
const gpu = new GPU({ mode });

const lucasKanade = gpu.createKernel(function (derivs) {
  const x = this.thread.x;
  const y = this.thread.y;

  let sxx = 0;
  let sxy = 0;
  let syy = 0;
  let sxt = 0;
  let syt = 0;

  for (let wy = 0; wy < 5; wy++) {
    for (let wx = 0; wx < 5; wx++) {
      let sy = y + wy - 2;
      if (sy < 0) sy = 0;
      if (sy > this.constants.last) sy = this.constants.last;
      let sx = x + wx - 2;
      if (sx < 0) sx = 0;
      if (sx > this.constants.last) sx = this.constants.last;
      const ix = derivs[0][sy][sx];
      const iy = derivs[1][sy][sx];
      const it = derivs[2][sy][sx];
      sxx += ix * ix;
      sxy += ix * iy;
      syy += iy * iy;
      sxt += ix * it;
      syt += iy * it;
    }
  }

  const det = sxx * syy - sxy * sxy;
  if (Math.abs(det) < this.constants.eps) {
    return 0;
  }
  if (this.thread.z === 0) {
    return (sxy * syt - syy * sxt) / det;
  }
  return (sxy * sxt - sxx * syt) / det;
}, {
  output: [64, 64, 2],
  constants: { last: 63, eps: 1e-6 },
});

const flow = await lucasKanade(derivs);
console.log('textured region, true motion is (1, 1):', flow[0][30][45], flow[1][30][45]);
console.log('flat region:', flow[0][10][6], flow[1][10][6]);
`,
      inputs: () => {
        const [a, b] = mainFrames();
        return { derivs: derivativeRef(a, b) };
      },
      publicTests: [
        {
          name: 'the textured band recovers the real motion, <code>(1, 1)</code>',
          run: async ctx => {
            ctx.assert(ctx.kernels.length >= 1, 'no kernel was created — call gpu.createKernel()');
            const [a, b] = mainFrames();
            const d = derivativeRef(a, b);
            const out = await ctx.kernel(d);
            ctx.assert(out && out.length === 2, `expected 2 planes, got ${out && out.length}`);
            ctx.assert(out[0] && out[0].length === 64, 'each plane should hold 64 rows');
            const [refU, refV] = flowRef(d);
            const [noCrossU, noCrossV] = flowRef(d, 'nocross');
            for (const [x, y] of [[42, 10], [46, 20], [50, 30], [54, 40], [58, 50], [44, 55]]) {
              const got = [out[0][y][x], out[1][y][x]];
              const ref = [refU[y][x], refV[y][x]];
              const noCross = [noCrossU[y][x], noCrossV[y][x]];
              for (let z = 0; z < 2; z++) {
                const hint =
                  blowUpHint(got[z], ref[z]) ||
                  diagnose(got[z], ref[z], 0.06, [
                    [-ref[z], FLOW_NEGATED],
                    [noCross[z], NO_CROSS_TERM],
                  ]);
                ctx.assertClose(
                  got[z],
                  1,
                  0.06,
                  hint ||
                    `${z === 0 ? 'u' : 'v'} at (${x}, ${y}) — the whole scene moved (1, 1), and ` +
                      `this window has texture in both directions, so it should say so`
                );
              }
            }
          },
        },
        {
          name: 'the flat and striped bands are refused, not guessed — <code>0</code>, not NaN',
          run: async ctx => {
            const [a, b] = mainFrames();
            const d = derivativeRef(a, b);
            const out = await ctx.kernel(d);
            // Columns 0-31 never vary with y, so Iy is exactly 0, so Sxy and Syy
            // are exactly 0, so the determinant is exactly 0 — on every backend,
            // in every precision. An unguarded solve divides by it.
            for (const y of [5, 21, 37, 58]) {
              for (const x of [3, 9, 14, 22, 28, 31]) {
                for (let z = 0; z < 2; z++) {
                  const hint = blowUpHint(out[z][y][x], 0);
                  ctx.assertClose(
                    out[z][y][x],
                    0,
                    1e-6,
                    hint ||
                      `${z === 0 ? 'u' : 'v'} at (${x}, ${y}) — that band is flat or all one ` +
                        `edge, so the determinant is zero and the answer has to be 0`
                  );
                }
              }
            }
          },
        },
        {
          name: 'every solved pixel matches the closed-form 2×2 solution',
          run: async ctx => {
            const [a, b] = mainFrames();
            const d = derivativeRef(a, b);
            const out = await ctx.kernel(d);
            const [refU, refV] = flowRef(d);
            const [noCrossU, noCrossV] = flowRef(d, 'nocross');
            const cases = [[40, 30], [45, 30], [50, 12], [55, 45], [60, 50], [33, 40], [32, 20], [63, 8]];
            for (const [x, y] of cases) {
              const got = [out[0][y][x], out[1][y][x]];
              const ref = [refU[y][x], refV[y][x]];
              const noCross = [noCrossU[y][x], noCrossV[y][x]];
              for (let z = 0; z < 2; z++) {
                const eps = 3e-3 + 3e-3 * Math.abs(ref[z]);
                const hint =
                  blowUpHint(got[z], ref[z]) ||
                  diagnose(got[z], ref[z], eps, [
                    [-ref[z], FLOW_NEGATED],
                    [ref[1 - z], UV_SWAPPED],
                    [noCross[z], NO_CROSS_TERM],
                  ]);
                ctx.assertClose(got[z], ref[z], eps, hint || `${z === 0 ? 'u' : 'v'} at (${x}, ${y})`);
              }
            }
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            const [a, b] = mainFrames();
            const d = derivativeRef(a, b);
            const out = await ctx.kernel(d);
            const [refU, refV] = flowRef(d);
            const [noCrossU, noCrossV] = flowRef(d, 'nocross');
            for (let y = 0; y < 64; y++) {
              for (let x = 0; x < 64; x++) {
                const got = [out[0][y][x], out[1][y][x]];
                const ref = [refU[y][x], refV[y][x]];
                const noCross = [noCrossU[y][x], noCrossV[y][x]];
                for (let z = 0; z < 2; z++) {
                  const eps = 3e-3 + 3e-3 * Math.abs(ref[z]);
                  const hint =
                    blowUpHint(got[z], ref[z]) ||
                    diagnose(got[z], ref[z], eps, [
                      [-ref[z], FLOW_NEGATED],
                      [ref[1 - z], UV_SWAPPED],
                      [noCross[z], NO_CROSS_TERM],
                    ]);
                  ctx.assertClose(got[z], ref[z], eps, hint || `${z === 0 ? 'u' : 'v'} at (${x}, ${y})`);
                }
              }
            }
          },
        },
        {
          name: 'private test #2',
          run: async ctx => {
            // Task 2's scene, fed to Lucas-Kanade: every gradient in it points
            // the same way, so the 2x2 matrix is singular no matter how big the
            // window gets. A wider aperture is not an answer to the aperture
            // problem.
            const [a, b] = apertureFrames();
            const d = derivativeRef(a, b);
            const out = await ctx.kernel(d);
            for (let y = 3; y <= 60; y++) {
              for (let x = 3; x <= 60; x++) {
                for (let z = 0; z < 2; z++) {
                  const hint = blowUpHint(out[z][y][x], 0);
                  ctx.assertClose(
                    out[z][y][x],
                    0,
                    1e-6,
                    hint ||
                      `${z === 0 ? 'u' : 'v'} at (${x}, ${y}) — every gradient in this scene is ` +
                        `parallel, so the determinant is zero and the window has to decline`
                  );
                }
              }
            }
          },
        },
      ],
    },

    // ------------------------------------------------------------------ 4
    {
      slug: 'good-features',
      title: 'Which Answers to Believe',
      intro: `<p>Task 3 returned a number for every pixel it could solve, and some of those numbers
        are worthless. A useful tracker does not just answer — it says <em>how much</em> it should be
        believed, and the material for that is already sitting in the window matrix
        <code>M = [[Sxx, Sxy], [Sxy, Syy]]</code>.</p>
        <p><code>M</code>'s two eigenvalues measure how much intensity change the window sees in its
        two principal directions. Three cases, and they are the whole story:</p>
<pre><code>        λmax     λmin    verdict
flat    ≈ 0      ≈ 0     nothing to track
edge    large    ≈ 0     one direction
corner  large    large   trustworthy</code></pre>
        <p>The middle row is task 2's aperture problem wearing a matrix. So the
        <strong>smaller</strong> eigenvalue is the confidence score: it is large only when the window
        is pinned down in <em>both</em> directions. That measure has a name — it is the
        Shi–Tomasi score, and thresholding it is literally what "good features to track" means. A
        2×2 symmetric matrix has a closed-form spectrum, so this is one line of arithmetic, not an
        eigensolver:</p>
<pre><code>trace = Sxx + Syy
det   = Sxx·Syy − Sxy²
λmin  = (trace − √(trace² − 4·det)) / 2</code></pre>`,
      goal: `<strong>Goal:</strong> produce a 64×64 confidence map — the smaller eigenvalue of each
        pixel's 5×5 window matrix.`,
      requirements: [
        'Reuse the 5×5 window sums from task 3, but only the three you need: <code>Sxx</code>, <code>Sxy</code>, <code>Syy</code>',
        'Output is plain 2D — <code>output: [64, 64]</code>, one number per pixel',
        'Return the <strong>smaller</strong> root: <code>(trace - Math.sqrt(disc)) / 2</code>, where <code>disc = trace * trace - 4 * det</code>',
        'Clamp the discriminant with <code>Math.max(0, …)</code> before the square root — it is mathematically non-negative, but float32 rounding can nudge it below zero and hand you a NaN',
      ],
      hints: [
        {
          title: 'Hint 1 — three sums, not five',
          body: `<p><code>It</code> plays no part here. Confidence is a property of the
            <em>window</em>, not of the motion — you can decide a pixel is untrackable before you
            look at the second frame at all.</p>`,
        },
        {
          title: 'Hint 2 — the closed form',
          body: `<pre><code>const trace = sxx + syy;
const det = sxx * syy - sxy * sxy;
const disc = Math.max(0, trace * trace - 4 * det);
return (trace - Math.sqrt(disc)) / 2;</code></pre>
          <p>Both roots share everything but a sign; <code>+</code> gives the larger eigenvalue,
            <code>−</code> the smaller. Take the smaller one — an edge scores well on the larger and
            is exactly what you are trying to reject.</p>`,
        },
      ],
      transfer: `Every corner detector is this matrix with a different scalar squeezed out of it:
        Shi–Tomasi takes <code>λmin</code>, Harris takes <code>det − k·trace²</code> to dodge the
        square root, FAST skips the matrix entirely and tests a pixel ring instead. OpenCV's
        <code>goodFeaturesToTrack</code>, the corner pass in ARKit and ARCore, and every
        visual-odometry front end run this per-pixel score and then keep the local maxima — which is
        a reduction, then a compaction, over a map you just computed in one launch.`,
      starterCode: `// Confidence, not just answers: the smaller eigenvalue of the window matrix.
const gpu = new GPU({ mode });

const confidence = gpu.createKernel(function (derivs) {
  const x = this.thread.x;
  const y = this.thread.y;

  let sxx = 0;
  let sxy = 0;
  let syy = 0;

  // TODO: the same clamped 5×5 window as task 3, accumulating the three
  // sums that do not involve It.

  // TODO: trace = sxx + syy;  det = sxx * syy - sxy * sxy;
  //       return the SMALLER root of the 2×2 spectrum.
  return 0;
}, {
  output: [64, 64],
  constants: { last: 63 },
});

const map = await confidence(derivs);
console.log('flat band:', map[20][6]);
console.log('stripe band:', map[20][24]);
console.log('textured band:', map[30][45]);
`,
      solutionCode: `// Confidence, not just answers: the smaller eigenvalue of the window matrix.
const gpu = new GPU({ mode });

const confidence = gpu.createKernel(function (derivs) {
  const x = this.thread.x;
  const y = this.thread.y;

  let sxx = 0;
  let sxy = 0;
  let syy = 0;

  for (let wy = 0; wy < 5; wy++) {
    for (let wx = 0; wx < 5; wx++) {
      let sy = y + wy - 2;
      if (sy < 0) sy = 0;
      if (sy > this.constants.last) sy = this.constants.last;
      let sx = x + wx - 2;
      if (sx < 0) sx = 0;
      if (sx > this.constants.last) sx = this.constants.last;
      const ix = derivs[0][sy][sx];
      const iy = derivs[1][sy][sx];
      sxx += ix * ix;
      sxy += ix * iy;
      syy += iy * iy;
    }
  }

  const trace = sxx + syy;
  const det = sxx * syy - sxy * sxy;
  const disc = Math.max(0, trace * trace - 4 * det);
  return (trace - Math.sqrt(disc)) / 2;
}, {
  output: [64, 64],
  constants: { last: 63 },
});

const map = await confidence(derivs);
console.log('flat band:', map[20][6]);
console.log('stripe band:', map[20][24]);
console.log('textured band:', map[30][45]);
`,
      inputs: () => {
        const [a, b] = mainFrames();
        return { derivs: derivativeRef(a, b) };
      },
      publicTests: [
        {
          name: 'produces a <code>64×64</code> map',
          run: async ctx => {
            ctx.assert(ctx.kernels.length >= 1, 'no kernel was created — call gpu.createKernel()');
            const [a, b] = mainFrames();
            const out = await ctx.kernel(derivativeRef(a, b));
            ctx.assert(out && out.length === 64, `expected 64 rows, got ${out && out.length}`);
            ctx.assert(out[0] && out[0].length === 64, 'each row should hold 64 values');
          },
        },
        {
          name: 'flat scores 0, edges score 0, corners score high',
          run: async ctx => {
            const [a, b] = mainFrames();
            const d = derivativeRef(a, b);
            const out = await ctx.kernel(d);
            const larger = confidenceRef(d, 'max');
            // Nothing left of column 34 varies with y, so Syy and Sxy are
            // exactly 0 there and the smaller eigenvalue is exactly 0 — for the
            // low-contrast wash AND for the high-contrast stripes. The stripes
            // are the interesting half: plenty of gradient, still untrackable.
            for (const [x, y, band] of [[6, 20, 'the low-contrast wash'], [12, 44, 'the low-contrast wash'], [24, 20, 'the stripes'], [29, 51, 'the stripes']]) {
              const hint = diagnose(out[y][x], 0, 1e-5, [[larger[y][x], LARGER_EIGENVALUE]]);
              ctx.assertClose(
                out[y][x],
                0,
                1e-5,
                hint || `${band} at (${x}, ${y}) is pinned down in at most one direction, so the smaller eigenvalue is 0`
              );
            }
            for (const [x, y] of [[45, 30], [55, 12], [40, 44], [58, 20]]) {
              ctx.assert(
                out[y][x] > 4e-4,
                `the textured band at (${x}, ${y}) scored ${out[y][x]} — it has structure in ` +
                  `both directions, so its smaller eigenvalue should be well clear of zero`
              );
            }
          },
        },
        {
          name: 'the map matches the closed-form smaller eigenvalue',
          run: async ctx => {
            const [a, b] = mainFrames();
            const d = derivativeRef(a, b);
            const out = await ctx.kernel(d);
            const ref = confidenceRef(d);
            const larger = confidenceRef(d, 'max');
            const determinant = confidenceRef(d, 'det');
            for (const [x, y] of [[45, 30], [50, 12], [55, 45], [60, 50], [40, 30], [24, 20], [6, 10], [35, 33]]) {
              const eps = 2e-6 + 1e-2 * Math.abs(ref[y][x]);
              const hint = diagnose(out[y][x], ref[y][x], eps, [
                [larger[y][x], LARGER_EIGENVALUE],
                [determinant[y][x], RAW_DETERMINANT],
                [larger[y][x] + ref[y][x], 'that is the trace — the SUM of the two eigenvalues. An edge has a large trace and no confidence at all'],
              ]);
              ctx.assertClose(out[y][x], ref[y][x], eps, hint || `confidence at (${x}, ${y})`);
            }
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            const [a, b] = mainFrames();
            const d = derivativeRef(a, b);
            const out = await ctx.kernel(d);
            const ref = confidenceRef(d);
            const larger = confidenceRef(d, 'max');
            const determinant = confidenceRef(d, 'det');
            for (let y = 0; y < 64; y++) {
              for (let x = 0; x < 64; x++) {
                const eps = 2e-6 + 1e-2 * Math.abs(ref[y][x]);
                const hint = diagnose(out[y][x], ref[y][x], eps, [
                  [larger[y][x], LARGER_EIGENVALUE],
                  [determinant[y][x], RAW_DETERMINANT],
                ]);
                ctx.assertClose(out[y][x], ref[y][x], eps, hint || `confidence at (${x}, ${y})`);
              }
            }
          },
        },
        {
          name: 'private test #2',
          run: async ctx => {
            // Task 2's scene again: uniformly strong gradients, uniformly zero
            // confidence. A big number in the map here would mean the score is
            // measuring contrast rather than trackability.
            const [a, b] = apertureFrames();
            const d = derivativeRef(a, b);
            const out = await ctx.kernel(d);
            const larger = confidenceRef(d, 'max');
            for (let y = 3; y <= 60; y += 1) {
              for (let x = 3; x <= 60; x += 1) {
                const hint = diagnose(out[y][x], 0, 1e-6, [[larger[y][x], LARGER_EIGENVALUE]]);
                ctx.assertClose(out[y][x], 0, 1e-6, hint || `confidence at (${x}, ${y})`);
              }
            }
          },
        },
      ],
    },

    // ------------------------------------------------------------------ 5
    {
      slug: 'flow-colours',
      title: 'Paint the Flow Field',
      intro: `<p>A flow field is two numbers per pixel, and nobody can read that as a table. The
        standard picture encodes it as colour: <strong>direction becomes hue</strong>, walking once
        around the colour wheel as the vector turns once around the circle, and
        <strong>magnitude becomes saturation</strong>, so a still pixel is white and a fast one is
        vivid. Hue and saturation as an angle and a radius is the polar half of HSV — the same wheel
        Colour Spaces works in — and here it costs one <code>Math.atan2</code>.</p>
        <p><code>flow</code> is the field task 3 produced for this scene, handed to you finished:
        <code>flow[0][y][x]</code> is <code>u</code> and <code>flow[1][y][x]</code> is
        <code>v</code>. The hue→RGB conversion is written for you in the starter — it is six
        straight lines around a hexagon, not the lesson. Your part is the encoding.</p>
        <p>Watch what the picture tells you: the whole scene translated by <code>(1, 1)</code>, and
        half of it comes out <strong>white</strong>. That is not a bug. Those are the pixels
        task 4 scored zero, painted honestly as "no idea".</p>`,
      goal: `<strong>Goal:</strong> paint the flow field — hue from the vector's direction,
        saturation from its length.`,
      requirements: [
        'A graphical kernel: <code>graphical: true</code>, <code>output: [64, 64]</code>',
        'Saturation: <code>s = Math.min(1, magnitude / this.constants.maxFlow)</code>, where magnitude is <code>Math.sqrt(u * u + v * v)</code>',
        'Hue: <code>hh = (Math.atan2(v, u) / (2 * Math.PI) + 0.5) * 6</code> — the sextant the direction lands in, from 0 to 6',
        'Paint the hue blended toward white by <code>s</code>: <code>this.color(1 - s + s * r, 1 - s + s * g, 1 - s + s * b, 1)</code>, so a zero-length vector comes out pure white',
      ],
      hints: [
        {
          title: 'Hint 1 — why the + 0.5',
          body: `<p><code>Math.atan2</code> returns an angle in <code>−π…π</code>. Dividing by
            <code>2π</code> maps that to <code>−0.5…0.5</code>, and the <code>+ 0.5</code> slides it
            to <code>0…1</code> without a branch or a modulo. Which direction gets which hue is
            arbitrary; that the mapping is <em>one-to-one</em> is the part that matters.</p>`,
        },
        {
          title: 'Hint 2 — the two lines above the hexagon',
          body: `<pre><code>const mag = Math.sqrt(u * u + v * v);
const s = Math.min(1, mag / this.constants.maxFlow);
const hh = (Math.atan2(v, u) / (2 * Math.PI) + 0.5) * 6;</code></pre>
          <p>and the last line under it is the blend: <code>1 - s + s * r</code> is
            <code>r</code> when <code>s</code> is 1 and white when <code>s</code> is 0.</p>`,
        },
      ],
      transfer: `The Middlebury flow-colour wheel is this exact encoding, and it is what every
        optical-flow paper prints. The pattern generalises: whenever a pass produces a vector per
        pixel — normals, velocity, curvature — direction-to-hue keeps it readable, and it costs one
        extra fragment/compute pass in WebGPU, CUDA or Metal alike. It is also the cheapest
        debugging tool in graphics: a wrong sign in a flow field is invisible in a table and
        blindingly obvious as a picture in the complementary colour.`,
      starterCode: `// Direction becomes hue, magnitude becomes saturation, stillness stays white.
const gpu = new GPU({ mode });

const paintFlow = gpu.createKernel(function (flow) {
  const u = flow[0][this.thread.y][this.thread.x];
  const v = flow[1][this.thread.y][this.thread.x];

  // TODO: s — the flow's length over this.constants.maxFlow, capped at 1
  const s = 0;
  // TODO: hh — (Math.atan2(v, u) / (2 * Math.PI) + 0.5) * 6
  const hh = 0;

  // Hue → RGB, written for you: six straight lines around the colour hexagon.
  let r = 0;
  let g = 0;
  let b = 0;
  if (hh < 1) {
    r = 1;
    g = hh;
  } else if (hh < 2) {
    r = 2 - hh;
    g = 1;
  } else if (hh < 3) {
    g = 1;
    b = hh - 2;
  } else if (hh < 4) {
    g = 4 - hh;
    b = 1;
  } else if (hh < 5) {
    r = hh - 4;
    b = 1;
  } else {
    r = 1;
    b = 6 - hh;
  }

  // TODO: blend each channel toward white by s and paint it.
  this.color(1, 0, 1, 1);
}, {
  output: [64, 64],
  graphical: true,
  constants: { maxFlow: 1.5 },
});

await paintFlow(flow);
render(paintFlow.canvas);
`,
      solutionCode: `// Direction becomes hue, magnitude becomes saturation, stillness stays white.
const gpu = new GPU({ mode });

const paintFlow = gpu.createKernel(function (flow) {
  const u = flow[0][this.thread.y][this.thread.x];
  const v = flow[1][this.thread.y][this.thread.x];

  const mag = Math.sqrt(u * u + v * v);
  const s = Math.min(1, mag / this.constants.maxFlow);
  const hh = (Math.atan2(v, u) / (2 * Math.PI) + 0.5) * 6;

  // Hue → RGB: six straight lines around the colour hexagon.
  let r = 0;
  let g = 0;
  let b = 0;
  if (hh < 1) {
    r = 1;
    g = hh;
  } else if (hh < 2) {
    r = 2 - hh;
    g = 1;
  } else if (hh < 3) {
    g = 1;
    b = hh - 2;
  } else if (hh < 4) {
    g = 4 - hh;
    b = 1;
  } else if (hh < 5) {
    r = hh - 4;
    b = 1;
  } else {
    r = 1;
    b = 6 - hh;
  }

  this.color(1 - s + s * r, 1 - s + s * g, 1 - s + s * b, 1);
}, {
  output: [64, 64],
  graphical: true,
  constants: { maxFlow: 1.5 },
});

await paintFlow(flow);
render(paintFlow.canvas);
`,
      inputs: () => {
        const [a, b] = mainFrames();
        return { flow: flowRef(derivativeRef(a, b)) };
      },
      publicTests: [
        {
          name: 'produces a <code>64×64</code> graphical canvas',
          run: async ctx => {
            ctx.assert(ctx.kernels.length >= 1, 'no kernel was created — call gpu.createKernel()');
            ctx.assert(ctx.canvas, 'no canvas — is the kernel graphical: true, and did you call render()?');
            ctx.assert(
              ctx.canvas.width === 64 && ctx.canvas.height === 64,
              `expected a 64×64 canvas, got ${ctx.canvas.width}×${ctx.canvas.height}`
            );
            const pixels = ctx.getPixels();
            ctx.assert(pixels.length === 64 * 64 * 4, 'pixel buffer should hold 64×64 RGBA values');
          },
        },
        {
          name: 'the untrackable bands come out white',
          run: async ctx => {
            const pixels = ctx.getPixels();
            // Columns 0-31 are exactly zero flow in every row, so this holds
            // whichever way round getPixels() hands back its rows.
            for (const row of [4, 20, 40, 60]) {
              for (const col of [2, 11, 20, 29]) {
                const i = (row * 64 + col) * 4;
                const rgb = `rgb(${pixels[i]}, ${pixels[i + 1]}, ${pixels[i + 2]})`;
                ctx.assert(
                  pixels[i] >= 250 && pixels[i + 1] >= 250 && pixels[i + 2] >= 250,
                  `col ${col} carries zero flow, so it should paint white — got ${rgb}. ` +
                    `A zero vector must land at s = 0, which the 1 - s + s * channel blend gives you.`
                );
              }
            }
          },
        },
        {
          name: 'the textured band paints the hue of a down-and-right motion',
          run: async ctx => {
            const pixels = ctx.getPixels();
            // (1, 1) sits at hh = 3.75 — the blue side of the hexagon, with a
            // quarter of green and no red at all. Column position survives any
            // flip of the row order, so this is orientation-proof.
            for (const row of [3, 17, 31, 48, 62]) {
              for (const col of [44, 45, 46, 55, 56, 57]) {
                const i = (row * 64 + col) * 4;
                const rgb = `rgb(${pixels[i]}, ${pixels[i + 1]}, ${pixels[i + 2]})`;
                ctx.assert(
                  pixels[i + 2] >= 250 && pixels[i] <= 60,
                  `col ${col} moved down and right, which lands in the blue sextant — expected a ` +
                    `saturated blue, got ${rgb}`
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
            // Constant flow fields: one colour over the whole canvas, so the
            // check cannot depend on row order at all. Rightward is cyan,
            // leftward is red, downward is violet — three points far apart on
            // the wheel.
            for (const [u, v, name] of [[1, 0, 'rightward'], [-1, 0, 'leftward'], [0, 1, 'downward']]) {
              await ctx.kernel(constantFlow(u, v));
              const pixels = ctx.getPixels();
              const [er, eg, eb] = flowColour(u, v);
              for (let i = 0; i < pixels.length; i += 397 * 4) {
                const rgb = `rgb(${pixels[i]}, ${pixels[i + 1]}, ${pixels[i + 2]})`;
                ctx.assert(
                  Math.abs(pixels[i] - er) <= 3 &&
                    Math.abs(pixels[i + 1] - eg) <= 3 &&
                    Math.abs(pixels[i + 2] - eb) <= 3,
                  `a constant ${name} flow of (${u}, ${v}) should paint rgb(${er}, ${eg}, ${eb}) ` +
                    `everywhere — got ${rgb} at byte ${i}`
                );
              }
            }
          },
        },
        {
          name: 'private test #2',
          run: async ctx => {
            // Saturation is a length, so it must not care about direction, and a
            // vector past maxFlow must clamp rather than wrap.
            await ctx.kernel(constantFlow(0, 0));
            let pixels = ctx.getPixels();
            ctx.assert(
              pixels[0] === 255 && pixels[1] === 255 && pixels[2] === 255,
              `a zero flow field should be pure white, got rgb(${pixels[0]}, ${pixels[1]}, ${pixels[2]})`
            );
            for (const [u, v] of [[6, 6], [-6, 6], [0, -9]]) {
              await ctx.kernel(constantFlow(u, v));
              pixels = ctx.getPixels();
              const [er, eg, eb] = flowColour(u, v);
              const channels = [pixels[0], pixels[1], pixels[2]];
              ctx.assert(
                Math.min(...channels) <= 3,
                `a flow of (${u}, ${v}) is far past maxFlow, so it should be fully saturated — ` +
                  `one channel has to reach 0, got rgb(${channels.join(', ')})`
              );
              ctx.assert(
                Math.abs(channels[0] - er) <= 3 &&
                  Math.abs(channels[1] - eg) <= 3 &&
                  Math.abs(channels[2] - eb) <= 3,
                `a flow of (${u}, ${v}) should paint rgb(${er}, ${eg}, ${eb}), got ` +
                  `rgb(${channels.join(', ')})`
              );
            }
          },
        },
      ],
    },
  ],
};
