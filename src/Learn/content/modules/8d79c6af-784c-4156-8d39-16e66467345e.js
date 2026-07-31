// Module: Colour Spaces — uuid 8d79c6af-784c-4156-8d39-16e66467345e
// (short id 8d79c6af). The file name is the uuid; identity lives in the
// exported object below, never in the path. No legacyId: this module postdates
// the uuid switch, and it declares no track (track membership lives in
// content/tracks.js).
//
// Colour Spaces — the opening analysis module of the Computer Vision track,
// and the gentlest possible on-ramp to it: every task here is one thread per
// pixel, no neighbours, no communication. Pure map, in the sense Thinking in
// Parallel gives that word.
//
// Five tasks: relative luminance, and the two cheaper greys it disagrees with
// → RGB to HSV, where hue is an angle and the max/min logic is genuinely
// branchy → what a wrapping channel does to ordinary arithmetic → selecting by
// colour, which is the module's thesis in one exercise → the payoff, an
// image's dominant hue.
//
// Kernel-authoring rules (contract): no closures inside kernel functions,
// images arrive as ImageData and are read image[this.thread.y][this.thread.x]
// = [r, g, b, a] with channels 0–1, loop bounds come from this.constants, and
// every task passes in CPU mode. Images stay at 64×64 so verification is quick.
//
// NUMERIC CONTRACT. Hue is computed from 8-bit-exact pixels whose chroma is
// never a whisker above zero: every swatch, ball and leaf in this file has
// either chroma 0 exactly (a real grey, hue undefined) or chroma ≥ 0.09, so
// (g − b) / C can never be a ratio of two rounding errors and the GL backend's
// float32 and JavaScript's float64 agree on which 60° wedge a pixel is in.
// Slide those colours towards grey and that guarantee goes with them.

import { ARRAY_LAYOUT } from '../layoutNote.js';
import { plainToImageData, quantizePixel } from '../../engine/utils.js';

const SIZE = 64; // every image in this module

// Task 1. Two sets of weights, and they are not interchangeable: L601 weights
// the channels AS STORED (gamma-encoded), REL weights linear light.
const L601 = [0.299, 0.587, 0.114];
const REL = [0.2126, 0.7152, 0.0722];
const ANCHOR_Y = 1; // the pixel task 1 prints three ways
const ANCHOR_X = 35;

const NO_HUE = -1; // the sentinel: not an angle, so it cannot be mistaken for one

// Task 3.
const PAIRS = 64;

// Task 4: the wedge around red, which straddles the 360°/0° seam.
const TARGET_HUE = 0;
const HUE_TOL = 15;
const MASK_SAT = 0.35;

// Task 5.
const BINS = 12;
const BIN_WIDTH = 30;
const SAT_FLOOR = 0.15;

// ---- reference colour maths ------------------------------------------------

// The sRGB transfer function, inverted: stored channel → linear light.
function toLinear(c) {
  return c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
}

function averageOf(p) {
  return (p[0] + p[1] + p[2]) / 3;
}

function lumaOf(p) {
  return L601[0] * p[0] + L601[1] * p[1] + L601[2] * p[2];
}

function relativeOf(p) {
  return REL[0] * toLinear(p[0]) + REL[1] * toLinear(p[1]) + REL[2] * toLinear(p[2]);
}

function valueOf(p) {
  return Math.max(p[0], Math.max(p[1], p[2]));
}

function minOf(p) {
  return Math.min(p[0], Math.min(p[1], p[2]));
}

function chromaOf(p) {
  return valueOf(p) - minOf(p);
}

function saturationOf(p) {
  const v = valueOf(p);
  return v === 0 ? 0 : chromaOf(p) / v;
}

// Hue in degrees, or NO_HUE for a pixel with no chroma at all. Which channel
// is the max picks the 60° wedge; the other two say where inside it.
function hueOf(p) {
  const c = chromaOf(p);
  if (c === 0) return NO_HUE;
  const v = valueOf(p);
  if (v === p[0]) {
    const h = 60 * ((p[1] - p[2]) / c);
    return h < 0 ? h + 360 : h;
  }
  if (v === p[1]) return 60 * ((p[2] - p[0]) / c + 2);
  return 60 * ((p[0] - p[1]) / c + 4);
}

// The red wedge before it is brought back onto the wheel — negative whenever
// blue beats green, and the single most common HSV bug there is.
function hueUnwrapped(p) {
  const c = chromaOf(p);
  if (c === 0) return NO_HUE;
  return valueOf(p) === p[0] ? 60 * ((p[1] - p[2]) / c) : hueOf(p);
}

// b − a folded onto the short way round: −180 … 180.
function shortestArc(a, b) {
  let d = b - a;
  if (d > 180) d -= 360;
  if (d < -180) d += 360;
  return d;
}

// The midpoint of two angles, taken the short way and folded back into 0 … 360.
function circularMean(a, b) {
  let m = a + shortestArc(a, b) / 2;
  if (m < 0) m += 360;
  if (m >= 360) m -= 360;
  return m;
}

// How far h is from target on the wheel, 0 … 180.
function hueDistance(h, target) {
  return Math.abs(shortestArc(target, h));
}

// ---- deterministic images --------------------------------------------------
//
// Every image these build is an ImageData (engine/utils.plainToImageData): the
// one image shape gpu.js runs on the GPU for graphical AND numeric kernels
// alike. A kernel built with an ImageData must never then be handed a nested
// array — gpu.js binds an argument's container type on the first call — so
// every image a test passes in is one of these too.

function constantImage(size, pixel) {
  const row = new Array(size).fill(quantizePixel(pixel));
  return plainToImageData(new Array(size).fill(row));
}

// Task 2 and 5's chart: 8 × 8 patches of 8 × 8 pixels.
//   row 0  the six pure hues, plus orange and violet
//   row 1  the same eight at half VALUE      — hue and saturation unmoved
//   row 2  the same eight at half SATURATION — hue and value unmoved
//   row 3  greys, black and white            — no hue at all
//   rows 4–7  skins, foliage, skies, warms: colours from actual pictures
const SWATCHES = [
  [1, 0, 0], [1, 0.5, 0], [1, 1, 0], [0, 1, 0], [0, 1, 1], [0, 0, 1], [0.5, 0, 1], [1, 0, 1],
  [0.5, 0, 0], [0.5, 0.25, 0], [0.5, 0.5, 0], [0, 0.5, 0], [0, 0.5, 0.5], [0, 0, 0.5], [0.25, 0, 0.5], [0.5, 0, 0.5],
  [1, 0.5, 0.5], [1, 0.75, 0.5], [1, 1, 0.5], [0.5, 1, 0.5], [0.5, 1, 1], [0.5, 0.5, 1], [0.75, 0.5, 1], [1, 0.5, 1],
  [0, 0, 0], [0.14, 0.14, 0.14], [0.29, 0.29, 0.29], [0.43, 0.43, 0.43],
  [0.57, 0.57, 0.57], [0.71, 0.71, 0.71], [0.86, 0.86, 0.86], [1, 1, 1],
  [0.94, 0.78, 0.66], [0.85, 0.65, 0.5], [0.72, 0.52, 0.38], [0.58, 0.4, 0.28],
  [0.45, 0.3, 0.2], [0.34, 0.22, 0.14], [0.24, 0.15, 0.1], [0.16, 0.1, 0.06],
  [0.16, 0.34, 0.12], [0.22, 0.45, 0.16], [0.3, 0.56, 0.2], [0.4, 0.66, 0.26],
  [0.5, 0.74, 0.34], [0.6, 0.82, 0.44], [0.7, 0.88, 0.56], [0.8, 0.93, 0.68],
  [0.06, 0.18, 0.4], [0.1, 0.28, 0.56], [0.16, 0.4, 0.7], [0.26, 0.52, 0.8],
  [0.4, 0.66, 0.88], [0.55, 0.78, 0.93], [0.12, 0.5, 0.5], [0.2, 0.66, 0.62],
  [0.96, 0.86, 0.5], [0.92, 0.7, 0.24], [0.86, 0.5, 0.16], [0.76, 0.28, 0.18],
  [0.6, 0.14, 0.24], [0.44, 0.12, 0.36], [0.3, 0.12, 0.44], [0.18, 0.12, 0.34],
];

const PATCH = SIZE / 8; // 8 px

function swatchImage() {
  const plain = new Array(SIZE);
  for (let y = 0; y < SIZE; y++) {
    const row = new Array(SIZE);
    for (let x = 0; x < SIZE; x++) {
      row[x] = quantizePixel(SWATCHES[Math.floor(y / PATCH) * 8 + Math.floor(x / PATCH)]);
    }
    plain[y] = row;
  }
  return plainToImageData(plain);
}

// The centre pixel of patch (row, col) — what a test samples.
function patchCentre(row, col) {
  return [row * PATCH + PATCH / 2, col * PATCH + PATCH / 2];
}

// Task 4's frame: a red ball lit from above against a teal wall, with a grey
// step card and a black patch along the bottom.
//
// The ball has a warm side and a cool side — 4° and 356° — because a real
// object does, and because a wedge around red therefore has to straddle the
// seam rather than sit tidily above zero.
const BALL_WARM = [0.86, 0.16, 0.11]; // hue ≈ 4°
const BALL_COOL = [0.86, 0.11, 0.16]; // hue ≈ 356°
const WALL = [0.13, 0.42, 0.5]; // hue ≈ 193°

function chromaScene(size) {
  const plain = new Array(size);
  const cx = (size - 1) / 2;
  const cy = size * 0.47;
  const radius = size * 0.3;
  for (let y = 0; y < size; y++) {
    const row = new Array(size);
    const t = y / (size - 1);
    for (let x = 0; x < size; x++) {
      if (y >= size - 6 && x < 16) {
        const g = 0.2 + 0.05 * x; // grey step card
        row[x] = quantizePixel([g, g, g, 1]);
      } else if (y >= size - 6 && x < 24) {
        row[x] = quantizePixel([0, 0, 0, 1]); // black patch
      } else {
        const dx = x - cx;
        const dy = y - cy;
        const inBall = dx * dx + dy * dy <= radius * radius;
        const base = inBall ? (x < cx ? BALL_WARM : BALL_COOL) : WALL;
        const k = inBall ? 1 - 1.1 * t : 1 - 0.5 * t;
        row[x] = quantizePixel([base[0] * k, base[1] * k, base[2] * k, 1]);
      }
    }
    plain[y] = row;
  }
  return plainToImageData(plain);
}

// Task 5's picture: sky over foliage, one trunk, and a scree of near-grey
// stones whose faint tints are rounding noise rather than colour.
function forestImage(utils, size, seed = 3301) {
  const rand = utils.seededRandom(seed);
  const plain = new Array(size);
  for (let y = 0; y < size; y++) {
    const row = new Array(size);
    for (let x = 0; x < size; x++) {
      let pixel;
      if (y < 16) {
        const k = 0.85 + 0.3 * (y / 16) + 0.06 * rand();
        pixel = [0.3 * k, 0.55 * k, 0.85 * k];
      } else if (y >= size - 6 && x < 16) {
        const g = 0.35 + 0.3 * rand();
        pixel = [g + 0.008 * (rand() - 0.5), g, g + 0.008 * (rand() - 0.5)];
      } else if (y >= 40 && x >= 28 && x < 36) {
        const k = 0.8 + 0.4 * rand();
        pixel = [0.42 * k, 0.28 * k, 0.18 * k];
      } else {
        const g = 0.5 * (0.55 + 0.75 * rand());
        pixel = [g * (0.42 + 0.06 * rand()), g, g * (0.3 + 0.06 * rand())];
      }
      row[x] = quantizePixel([pixel[0], pixel[1], pixel[2], 1]);
    }
    plain[y] = row;
  }
  return plainToImageData(plain);
}

// ---- deterministic arrays --------------------------------------------------

// 64 pairs of hue readings. The FIRST EIGHT straddle the seam on purpose —
// that is what the exercise is about — and the rest are random pairs that never
// come within 10° of being exactly opposite, because two antipodal angles have
// no midpoint to agree on and a test must not ask for one.
function makeHuePairs(utils, seed = 8149) {
  const rand = utils.seededRandom(seed);
  const planted = [
    [350, 10], [10, 350], [358.5, 4.5], [340, 20],
    [355, 35], [300, 50], [12, 348], [0, 30],
  ];
  const hueA = new Array(PAIRS);
  const hueB = new Array(PAIRS);
  for (let i = 0; i < PAIRS; i++) {
    if (i < planted.length) {
      hueA[i] = planted[i][0];
      hueB[i] = planted[i][1];
      continue;
    }
    let a;
    let b;
    do {
      a = Math.round(rand() * 3599) / 10;
      b = Math.round(rand() * 3599) / 10;
    } while (Math.abs(shortestArc(a, b)) > 170);
    hueA[i] = a;
    hueB[i] = b;
  }
  return { hueA, hueB };
}

// ---- reference answers over whole images -----------------------------------

// Task 4's answer, and the four near-misses worth naming.
function hsvMask(p) {
  const h = hueOf(p);
  return hueDistance(h, TARGET_HUE) <= HUE_TOL && saturationOf(p) >= MASK_SAT ? 1 : 0;
}

function rgbMask(p) {
  return p[0] > 0.5 && p[0] - p[1] > 0.3 && p[0] - p[2] > 0.3 ? 1 : 0;
}

function maskNoFold(p) {
  const h = hueOf(p);
  return Math.abs(h - TARGET_HUE) <= HUE_TOL && saturationOf(p) >= MASK_SAT ? 1 : 0;
}

function maskNoSatFloor(p) {
  return hueDistance(hueOf(p), TARGET_HUE) <= HUE_TOL ? 1 : 0;
}

function maskAndWedge(p) {
  const h = hueOf(p);
  return h > 360 - HUE_TOL && h < HUE_TOL ? 1 : 0; // never true: && across the seam
}

// Task 5.
function binOf(p) {
  if (saturationOf(p) < SAT_FLOOR) return -1;
  return Math.min(BINS - 1, Math.floor(hueOf(p) / BIN_WIDTH));
}

function binNoFloor(p) {
  const h = hueOf(p);
  return h < 0 ? -1 : Math.min(BINS - 1, Math.floor(h / BIN_WIDTH));
}

function histogramOf(plain, bin) {
  const counts = new Array(BINS).fill(0);
  for (let y = 0; y < plain.length; y++) {
    for (let x = 0; x < plain[y].length; x++) {
      const b = bin(plain[y][x]);
      if (b >= 0 && b < BINS) counts[b]++;
    }
  }
  return counts;
}

function fullestBin(counts) {
  let best = 0;
  for (let b = 1; b < counts.length; b++) if (counts[b] > counts[best]) best = b;
  return best;
}

function sumOf(values) {
  let total = 0;
  for (let i = 0; i < values.length; i++) total += values[i];
  return total;
}

// ---- near-miss diagnosis ---------------------------------------------------
//
// A failing assert that reports only two numbers tells a learner nothing about
// WHICH slip produced them. A probe pairs the value one specific known mistake
// would produce with a sentence naming that mistake; diagnose() speaks only
// when the observed value matches a probe within the test's own tolerance AND
// the correct value does not — so a pixel where two candidates coincide (a
// swatch whose channel average happens to equal its luminance) stays silent, as
// do observations matching probes that disagree with each other. A wrong
// diagnosis is worse than none.
function diagnose(got, expected, eps, probes) {
  const hits = probes
    .filter(p => Math.abs(got - p[0]) <= eps && Math.abs(expected - p[0]) > eps)
    .map(p => p[1]);
  return hits.length && hits.every(m => m === hits[0]) ? hits[0] : null;
}

// The whole-array form. One matching element means nothing when the candidate
// is a function of the index — two different wrap bugs agree about every pair
// that does not straddle the seam — so a probe must predict EVERY element and
// disagree with the right answer somewhere before it may speak.
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

// The whole-grid form, for masks and bin maps: same rule, two dimensions.
function diagnoseGrid(out, plain, expected, eps, probes) {
  const hits = probes
    .filter(([value]) => {
      let differs = false;
      for (let y = 0; y < plain.length; y++) {
        for (let x = 0; x < plain[y].length; x++) {
          const c = value(plain[y][x]);
          if (!(out[y] && Math.abs(out[y][x] - c) <= eps)) return false;
          if (Math.abs(expected(plain[y][x]) - c) > eps) differs = true;
        }
      }
      return differs;
    })
    .map(p => p[1]);
  return hits.length && hits.every(m => m === hits[0]) ? hits[0] : null;
}

// Swapping this.thread.x and this.thread.y reads the transpose of the image —
// invisible to a histogram (a transposed image has the same one) and to any
// test that only checks a range, so the per-cell tests are where it gets
// caught. Cells on the diagonal are their own transpose and can never show it,
// which is why every case list below is off-diagonal.
function transposeCellHint(got, transposed, eps, y, x, name) {
  return Math.abs(got - transposed) <= eps
    ? `that is the value for cell [${x}][${y}] — this.thread.x and this.thread.y are ` +
      `swapped. Rows come first: ${name}[this.thread.y][this.thread.x]`
    : null;
}

// Task 1: the six near-misses that are nearly, but not, relative luminance.
function luminanceProbes(p) {
  return [
    [averageOf(p),
      'that is the plain channel average — the eye is nowhere near equally sensitive to the three primaries, so an unweighted mean is not a brightness at all'],
    [lumaOf(p),
      'that is Rec. 601 luma, 0.299 R + 0.587 G + 0.114 B applied to the stored channels — this task wants relative luminance, which linearises first and then weights 0.2126 / 0.7152 / 0.0722'],
    [REL[0] * p[0] + REL[1] * p[1] + REL[2] * p[2],
      'the weights are the right ones but the channels are still gamma-encoded — the sRGB → linear step is missing'],
    [L601[0] * toLinear(p[0]) + L601[1] * toLinear(p[1]) + L601[2] * toLinear(p[2]),
      'the channels were linearised and then weighted 0.299 / 0.587 / 0.114 — those are the luma weights, for gamma-encoded channels. Linear light wants 0.2126 / 0.7152 / 0.0722'],
    [REL[0] * toLinear(toLinear(p[0])) + REL[1] * toLinear(toLinear(p[1])) + REL[2] * toLinear(toLinear(p[2])),
      'the sRGB → linear step was applied twice — once to the channel and once again to the result'],
    [REL[2] * toLinear(p[0]) + REL[1] * toLinear(p[1]) + REL[0] * toLinear(p[2]),
      'the weights are in the wrong order — 0.2126 belongs on red and 0.0722 on blue'],
    [REL[0] * Math.pow(p[0], 2.2) + REL[1] * Math.pow(p[1], 2.2) + REL[2] * Math.pow(p[2], 2.2),
      'that is the gamma-2.2 approximation, not the sRGB transfer function — sRGB has a straight-line toe below 0.04045 and an exponent of 2.4 above it'],
  ];
}

// Task 2: the ways a hue angle comes out slightly wrong. Only offered for
// pixels that HAVE a hue; the two zero cases get their own sentences below.
function hueProbes(p) {
  const c = chromaOf(p);
  if (c === 0) return [];
  const probes = [
    [hueOf(p) / 360,
      'that is the hue as a fraction of a full turn — this task wants degrees, 0 … 360'],
    [c, 'that is the chroma, max − min — hue is an angle, not a magnitude'],
  ];
  const raw = hueUnwrapped(p);
  if (raw < 0) {
    probes.push([raw,
      'the red wedge came out negative — (g − b) / C drops below zero whenever blue beats green, so add 360 to bring the angle back onto the wheel']);
  }
  const v = valueOf(p);
  if (v === p[1]) {
    probes.push([60 * ((p[2] - p[0]) / c + 4),
      'the green and blue wedges are swapped — green is centred on 120°, which is + 2 sixths, and blue on 240°, which is + 4']);
  } else if (v === p[2]) {
    probes.push([60 * ((p[0] - p[1]) / c + 2),
      'the green and blue wedges are swapped — green is centred on 120°, which is + 2 sixths, and blue on 240°, which is + 4']);
  }
  return probes;
}

// Task 2: what a pixel with no chroma at all came back as.
function noHueHint(got) {
  if (typeof got !== 'number' || Number.isNaN(got)) {
    return 'a grey pixel has chroma 0, so (g − b) / C is 0 / 0 — NaN. Test for zero chroma BEFORE you divide, and return -1';
  }
  if (Math.abs(got) <= 1e-6) {
    return 'a grey pixel came back as 0, and 0 is a real hue: red. Grey has no hue at all, so it needs a value that is not an angle — this task uses -1';
  }
  return null;
}

function saturationProbes(p) {
  const probes = [
    [chromaOf(p),
      'that is the chroma, max − min — saturation is the chroma relative to the value, C / V'],
    [valueOf(p), 'that is the value, max(r, g, b) — saturation is (max − min) / max'],
  ];
  if (minOf(p) > 0) {
    probes.push([chromaOf(p) / minOf(p),
      'the chroma was divided by the MINIMUM channel — saturation divides by the maximum, which is V']);
  }
  return probes;
}

function blackSaturationHint(got) {
  return typeof got !== 'number' || Number.isNaN(got)
    ? 'a black pixel has V = 0, so C / V is 0 / 0 — NaN. Guard the division: when the value is 0 the saturation is 0'
    : null;
}

// Task 3: the four ways of averaging two angles badly.
function midpointProbes(a, b) {
  return [
    [i => (a[i] + b[i]) / 2,
      'every midpoint is the plain arithmetic mean — which is exactly the bug: hue 350 and hue 10 are 20° apart, and their mean says 180°, the opposite colour. Fold the difference onto the short way round first'],
    [i => a[i] + shortestArc(a[i], b[i]) / 2,
      'the short way round was taken, but the answer was never folded back into 0 … 360 — a midpoint of 360 is 0, and a midpoint of -5 is 355'],
    [i => b[i],
      'the whole difference was added instead of half of it, so every answer is simply the second hue'],
    [i => a[i], 'nothing was added — every answer is the first hue'],
  ];
}

// Task 4.
function maskProbes() {
  return [
    [maskAndWedge,
      'nothing matched at all. A wedge around red runs from 345° up over the seam to 15°, and h > 345 && h < 15 is false for every angle there is — no number is both. Measure the distance round the wheel instead'],
    [maskNoFold,
      'only the warm half of the ball matched. The cool half of it sits just below 360°, so h − target comes out at about 356 rather than about −4, and 356 is nowhere near within 15 — fold the difference into −180 … 180 before you take its size'],
    [maskNoSatFloor,
      'the grey card and the black patch are in the mask. Neither has a hue, so both report -1, and -1 is one degree from red if you take it seriously — the saturation floor is what keeps meaningless angles out'],
    [rgbMask,
      "that is the RGB mask's own answer — the shadowed part of the ball is missing, which is the whole point of the exercise"],
  ];
}

// Task 5. Every candidate here applies the saturation floor, because a probe
// that ignores it would never match code that honours it.
function binProbes() {
  return [
    [p => (saturationOf(p) < SAT_FLOOR ? -1 : Math.min(BINS - 1, Math.round(hueOf(p) / BIN_WIDTH))),
      'the bin index was rounded, not floored — every hue more than half way through its bin got pushed into the next one'],
  ];
}

// The missing floor, caught structurally rather than by value. A near-grey
// pixel's hue is the direction of an 8-bit rounding error: a real angle
// arithmetically, noise in fact, and float32 and float64 do not even agree on
// which bin it lands in — so counting how many of them were binned at all says
// far more than any candidate value could.
function floorHint(out, plain) {
  let binned = 0;
  for (let y = 0; y < plain.length; y++) {
    for (let x = 0; x < plain[y].length; x++) {
      if (saturationOf(plain[y][x]) < SAT_FLOOR && out[y] && out[y][x] >= 0) binned++;
    }
  }
  return binned
    ? `${binned} pixels below the saturation floor still came back with a bin number. Those are ` +
      `the stones: their hue is the direction of a rounding error rather than a colour, which is ` +
      `exactly what the floor is there to refuse`
    : null;
}

function totalHint(counts, n, noHue) {
  const total = Math.round(sumOf(Array.from(counts)));
  if (total === n - noHue) return null;
  if (total === n) {
    return `every one of the ${n} pixels was counted, but ${noHue} of them have no hue to count — those are meant to fall out of the histogram`;
  }
  if (total > n) {
    return `the counts add up to ${total}, more than the ${n} pixels in the picture — some pixel was counted by more than one bin`;
  }
  return `the counts add up to ${total}, and they should come to ${n - noHue}: ${n} pixels less the ${noHue} with no usable hue`;
}

// ---- run inspection --------------------------------------------------------

function loggedNumbers(logs) {
  const out = [];
  for (const line of logs) {
    if (line.type !== 'log' || !line.text) continue;
    const matches = line.text.match(/-?\d+(?:\.\d+)?/g);
    if (matches) for (const m of matches) out.push(parseFloat(m));
  }
  return out;
}

function loggedNear(logs, value, eps) {
  return loggedNumbers(logs).some(v => Math.abs(v - value) <= eps);
}

// Kernels that produce a 2D grid of numbers — as opposed to the graphical one,
// or a 1D histogram. `arity`, when given, keeps only those taking that many
// arguments.
function numericGrids(ctx, arity) {
  return ctx.kernels.filter(k => {
    const built = k.kernel;
    if (!built || built.graphical || !built.output || built.output.length !== 2) return false;
    return arity === undefined || (built.argumentNames || []).length === arity;
  });
}

function kernelOfRank(ctx, rank) {
  return ctx.kernels.find(k => k.kernel && k.kernel.output && k.kernel.output.length === rank) || null;
}

function graphicalKernel(ctx) {
  return ctx.kernels.find(k => k.kernel && k.kernel.graphical) || null;
}

// Identity by ARITY, for tasks where one kernel takes two maps and the others
// take one image. Robust whether or not the learner's answer is right, which
// order-based identification is not.
function kernelWithArgs(ctx, n) {
  return ctx.kernels.find(k => {
    const built = k.kernel;
    return built && !built.graphical && (built.argumentNames || []).length === n;
  }) || null;
}

// Several tasks build two or three grid kernels that look identical from the
// outside — same rank, same arity — so identify them by BEHAVIOUR rather than
// by creation order, which a learner is free to change. A flat pure-green image
// separates every kernel in this module: hue answers 120, saturation answers 1,
// a red mask answers 0.
async function probeOnGreen(k) {
  try {
    const out = await k(constantImage(SIZE, [0, 1, 0, 1]));
    return out && out[0] && typeof out[0][0] === 'number' ? out[0][0] : null;
  } catch (e) {
    return null;
  }
}

async function findGrid(ctx, matches) {
  for (const k of numericGrids(ctx, 1)) {
    const sample = await probeOnGreen(k);
    if (sample !== null && matches(sample)) return k;
  }
  return null;
}

// The provided hue kernel of tasks 4 and 5 — 120° on pure green.
async function hueKernel(ctx) {
  return findGrid(ctx, v => Math.abs(v - 120) <= 1);
}

// Task 2's pair. When an answer is wrong enough that the probe cannot place it,
// fall back to creation order — the order the starter declares them in — so the
// learner still gets a real failure message about their own kernel rather than
// "no kernel found".
async function findHueAndSaturation(ctx) {
  const grids = numericGrids(ctx, 1);
  const hue = await findGrid(ctx, v => Math.abs(v - 120) <= 1);
  const saturation = await findGrid(ctx, v => Math.abs(v - 1) <= 1e-3);
  const rest = grids.filter(k => k !== hue && k !== saturation);
  return {
    hue: hue || rest.shift() || null,
    saturation: saturation || rest.shift() || null,
  };
}

export default {
  uuid: '8d79c6af-784c-4156-8d39-16e66467345e',
  version: 1,
  slug: 'colour-spaces',
  title: 'Colour Spaces',
  blurb: 'Leaving RGB: perceptual luminance, the hue wheel, and why a channel that wraps breaks ordinary arithmetic.',
  tasks: [
    // ------------------------------------------------------------------ 1
    {
      slug: 'relative-luminance',
      title: 'Three Greys, One Pixel',
      intro: `<p>Nearly every vision algorithm's first move is to leave RGB, because RGB tangles
        together the two things you usually want to reason about separately: what colour something
        is, and how bright it is. Brightness is the easier half, and the one most often got wrong.</p>
        <p>There are three answers to "how bright is this pixel", and they do not agree. The channel
        average <code>(r + g + b) / 3</code> is what everyone writes first, and it is simply false —
        the eye is roughly five times more sensitive to green than to blue, so a pure green and a
        pure blue that "average" the same are nowhere near equally bright. Weighting the channels
        fixes that: <code>0.299r + 0.587g + 0.114b</code> is Rec. 601 <strong>luma</strong>, the
        recipe Data In, Data Out had you write.</p>
        <p>But luma weights the numbers <em>as stored</em>, and sRGB channels are
        <strong>gamma-encoded</strong>: 0.5 in a PNG is not half the light of 1.0, it is about 21% of
        it. Relative luminance undoes that encoding first and then weights the actual light. It is
        the number a photometer would agree with, and the one every contrast-ratio rule is built
        on.</p>
<pre><code>t = (c + 0.055) / 1.055

c &lt;= 0.04045  linear = c / 12.92
otherwise     linear = Math.pow(t, 2.4)

Y = 0.2126*R + 0.7152*G + 0.0722*B</code></pre>
        ${ARRAY_LAYOUT}`,
      goal: `<strong>Goal:</strong> compute a 64 × 64 relative-luminance map of <code>photo</code>,
        and log that map's value for the one pixel the starter already prints two other ways.`,
      requirements: [
        'Keep the kernel numeric — <code>output: [64, 64]</code>, one thread per pixel',
        'Linearise <em>each</em> of r, g and b with the sRGB transfer function above, before any weighting',
        'Weight the linear channels <code>0.2126 R + 0.7152 G + 0.0722 B</code>',
        '<code>console.log</code> the relative luminance of <code>photo[1][35]</code> beside the two greys already printed',
      ],
      hints: [
        {
          title: 'Hint 1 — one channel at a time',
          body: `<p>The transfer function is a two-case branch, and it is the same branch three
            times over. Pull the channels into <code>let</code> variables so you can rewrite them
            in place:</p>
<pre><code>let r = pixel[0];
if (r &lt;= 0.04045) {
  r = r / 12.92;
} else {
  r = Math.pow((r + 0.055) / 1.055, 2.4);
}</code></pre>`,
        },
        {
          title: 'Hint 2 — then the weights',
          body: `<p>Once <code>r</code>, <code>g</code> and <code>b</code> hold linear light, the
            last line is just the weighted sum:</p>
<pre><code>return 0.2126 * r + 0.7152 * g + 0.0722 * b;</code></pre>
          <p>Note that these are <em>not</em> the 0.299 / 0.587 / 0.114 of the earlier module. Those
            weights belong to gamma-encoded channels; these belong to linear ones. Mixing the two
            pairs up is the classic version of this bug.</p>`,
        },
        {
          title: 'Hint 3 — reading the answer',
          body: `<p><code>map[1][35]</code> is the pixel the starter prints. Expect all three
            numbers to differ, and the linearised one to be much the smallest: most of what looks
            like brightness in an sRGB file is the encoding, not the light.</p>`,
        },
      ],
      transfer: `Every graphics API knows about this and will do it for you if you ask: an
        <code>rgba8unorm-srgb</code> texture in WebGPU, <code>GL_SRGB8_ALPHA8</code> in OpenGL and
        <code>MTLPixelFormatRGBA8Unorm_sRGB</code> in Metal all linearise on read and re-encode on
        write, in fixed-function hardware, for free. Blending or filtering in gamma space because
        you forgot to ask is one of the oldest bugs in rendering — it is why badly-resized images
        get darker, and why naive alpha compositing leaves dark fringes.`,
      starterCode: `// One thread per pixel: a pure map, no neighbours involved.
const gpu = new GPU({ mode });

const relativeLuminance = gpu.createKernel(function (photo) {
  const pixel = photo[this.thread.y][this.thread.x];
  // TODO: undo the sRGB gamma encoding on each channel FIRST,
  // then weight the linear channels 0.2126 / 0.7152 / 0.0722.
  return 0.2126 * pixel[0] + 0.7152 * pixel[1] + 0.0722 * pixel[2];
}, { output: [64, 64] });

const map = await relativeLuminance(photo);

// The same pixel, three ways. Two of them are done for you, on the host —
// photo.at(x, y) is the host-side view of photo[y][x].
const p = photo.at(35, 1);
console.log('channel average:   ', (p[0] + p[1] + p[2]) / 3);
console.log('Rec. 601 luma:     ', 0.299 * p[0] + 0.587 * p[1] + 0.114 * p[2]);
// TODO: log the relative luminance of that same pixel, out of your map.
`,
      solutionCode: `// One thread per pixel: a pure map, no neighbours involved.
const gpu = new GPU({ mode });

const relativeLuminance = gpu.createKernel(function (photo) {
  const pixel = photo[this.thread.y][this.thread.x];

  // sRGB -> linear light, once per channel.
  let r = pixel[0];
  let g = pixel[1];
  let b = pixel[2];
  if (r <= 0.04045) { r = r / 12.92; } else { r = Math.pow((r + 0.055) / 1.055, 2.4); }
  if (g <= 0.04045) { g = g / 12.92; } else { g = Math.pow((g + 0.055) / 1.055, 2.4); }
  if (b <= 0.04045) { b = b / 12.92; } else { b = Math.pow((b + 0.055) / 1.055, 2.4); }

  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}, { output: [64, 64] });

const map = await relativeLuminance(photo);

// The same pixel, three ways. Two of them are done for you, on the host —
// photo.at(x, y) is the host-side view of photo[y][x].
const p = photo.at(35, 1);
console.log('channel average:   ', (p[0] + p[1] + p[2]) / 3);
console.log('Rec. 601 luma:     ', 0.299 * p[0] + 0.587 * p[1] + 0.114 * p[2]);
console.log('relative luminance:', map[1][35]);
`,
      inputs: utils => ({ photo: utils.makeTestImage(SIZE) }),
      publicTests: [
        {
          name: 'produces a 64 × 64 luminance map',
          run: async ctx => {
            ctx.assert(ctx.kernels.length >= 1, 'no kernel was created — call gpu.createKernel()');
            const out = await ctx.kernel(ctx.utils.makeTestImage(SIZE));
            ctx.assert(out && out.length === SIZE, `expected ${SIZE} rows, got ${out && out.length}`);
            ctx.assert(out[0] && out[0].length === SIZE, `each row should hold ${SIZE} values`);
          },
        },
        {
          name: 'each cell is <code>0.2126R + 0.7152G + 0.0722B</code> on the <em>linearised</em> channels',
          run: async ctx => {
            const image = ctx.utils.makeTestImage(SIZE);
            const plain = image.plain;
            const out = await ctx.kernel(image);
            // Cells chosen for spread: at each of these the six near-misses are
            // at least 0.05 apart, so a probe can speak instead of staying quiet.
            for (const [y, x] of [[10, 62], [2, 57], [17, 62], [5, 51]]) {
              const p = plain[y][x];
              const expected = relativeOf(p);
              const hint =
                transposeCellHint(out[y][x], relativeOf(plain[x][y]), 2e-3, y, x, 'photo') ||
                diagnose(out[y][x], expected, 2e-3, luminanceProbes(p));
              ctx.assertClose(out[y][x], expected, 2e-3, hint || `cell [${y}][${x}]`);
            }
          },
        },
        {
          name: 'all three greys for pixel <code>[1][35]</code> are logged',
          run: async ctx => {
            const p = ctx.utils.makeTestImage(SIZE).plain[ANCHOR_Y][ANCHOR_X];
            ctx.assert(loggedNear(ctx.logs, averageOf(p), 5e-4),
              `the channel average of that pixel (${averageOf(p).toFixed(4)}) is no longer in the console output`);
            ctx.assert(loggedNear(ctx.logs, lumaOf(p), 5e-4),
              `the Rec. 601 luma of that pixel (${lumaOf(p).toFixed(4)}) is no longer in the console output`);
            ctx.assert(loggedNear(ctx.logs, relativeOf(p), 3e-3),
              `log map[1][35] too — expected the relative luminance ≈${relativeOf(p).toFixed(4)} in the console, ` +
              `next to the ${averageOf(p).toFixed(4)} and ${lumaOf(p).toFixed(4)} already there`);
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            const image = ctx.utils.makeTestImage(SIZE);
            const plain = image.plain;
            const out = await ctx.kernel(image);
            for (let y = 0; y < SIZE; y++) {
              for (let x = 0; x < SIZE; x++) {
                const p = plain[y][x];
                const expected = relativeOf(p);
                const hint =
                  transposeCellHint(out[y][x], relativeOf(plain[x][y]), 2e-3, y, x, 'photo') ||
                  diagnose(out[y][x], expected, 2e-3, luminanceProbes(p));
                ctx.assertClose(out[y][x], expected, 2e-3, hint || `cell [${y}][${x}]`);
              }
            }
          },
        },
        {
          name: 'private test #2',
          run: async ctx => {
            // Flat colours, including the two ends and one below the transfer
            // function's knee, where the straight-line branch is the only
            // correct answer.
            for (const pixel of [[0, 0, 0, 1], [1, 1, 1, 1], [0.03, 0.03, 0.03, 1], [0.5, 0.2, 0.9, 1]]) {
              const image = constantImage(SIZE, pixel);
              const p = image.at(0, 0);
              const expected = relativeOf(p);
              const out = await ctx.kernel(image);
              const hint = diagnose(out[0][0], expected, 2e-3, luminanceProbes(p));
              ctx.assertClose(out[0][0], expected, 2e-3, hint ||
                `a flat [${p.slice(0, 3).map(v => v.toFixed(2)).join(', ')}] image`);
              ctx.assertClose(out[SIZE - 1][SIZE - 1], expected, 2e-3, hint || 'the far corner of a flat image');
            }
          },
        },
      ],
    },

    // ------------------------------------------------------------------ 2
    {
      slug: 'rgb-to-hsv',
      title: 'Hue, Saturation, Value',
      intro: `<p>RGB says how much of each light to mix. It does not say what colour something
        <em>is</em>. <strong>HSV</strong> does, by splitting the question three ways: hue (which
        colour), saturation (how far from grey), value (how bright). Two of the three take one line
        each. The third is where the interesting code lives.</p>
        <p>Start from the largest and smallest channel. <code>V = max</code>. The gap between them is
        the <strong>chroma</strong>, <code>C = max − min</code> — how far this pixel is from grey —
        and saturation is that gap as a fraction of the value, <code>C / V</code>.</p>
        <p>Hue is an <em>angle</em>: red at 0°, green at 120°, blue at 240°, round to red again at
        360°. Which channel is the max picks a 60° wedge of that wheel, and the other two channels
        say where you sit inside it. When the chroma is zero there is no wedge at all: a grey pixel
        has no hue. Not "hue 0" — 0 is red. No hue, and it needs a value that is not an angle, which
        here is <code>-1</code>.</p>
<pre><code>V = max(r, g, b)
C = V - min(r, g, b)
S = C / V

V is r   H = 60 * ((g - b) / C)
         + 360 when that comes out negative
V is g   H = 60 * ((b - r) / C + 2)
V is b   H = 60 * ((r - g) / C + 4)</code></pre>
        ${ARRAY_LAYOUT}`,
      goal: `<strong>Goal:</strong> finish the two kernels — <code>hue</code>, in degrees, with
        <code>-1</code> where there is no hue, and <code>saturation</code>. The graphical kernel
        below is already written and will paint whatever hue channel you produce.`,
      requirements: [
        '<code>hue</code>: return <code>-1</code> when the chroma is <code>0</code>, and otherwise the angle in degrees',
        'The red wedge is the one that can come out negative — add <code>360</code> to bring it back onto the wheel',
        '<code>saturation</code>: return <code>(max − min) / max</code>, and <code>0</code> when <code>max</code> is <code>0</code> rather than dividing by it',
        'Leave <code>paintHue</code> alone — it renders your hue channel at full strength so you can see it',
      ],
      hints: [
        {
          title: 'Hint 1 — the three quantities first',
          body: `<p>Every branch below is written in terms of the same three numbers, so name them
            once at the top of the kernel:</p>
<pre><code>const v = Math.max(pixel[0], Math.max(pixel[1], pixel[2]));
const m = Math.min(pixel[0], Math.min(pixel[1], pixel[2]));
const c = v - m;</code></pre>`,
        },
        {
          title: 'Hint 2 — which wedge am I in?',
          body: `<p><code>v</code> is one of the three channels exactly — <code>Math.max</code>
            returns one of its arguments, it does not compute a new number — so you can compare
            against it directly:</p>
<pre><code>if (c === 0) {
  return -1;
}
if (v === pixel[0]) {
  const h = 60 * ((pixel[1] - pixel[2]) / c);
  if (h &lt; 0) { return h + 360; }
  return h;
}
if (v === pixel[1]) {
  return 60 * ((pixel[2] - pixel[0]) / c + 2);
}
return 60 * ((pixel[0] - pixel[1]) / c + 4);</code></pre>`,
        },
        {
          title: 'Hint 3 — the two zeros',
          body: `<p>Both kernels have a divide-by-nothing case and they are not the same case.
            Hue divides by the <em>chroma</em>, which is zero for any grey. Saturation divides by
            the <em>value</em>, which is zero only for black. Test before you divide in both, or
            those pixels come back <code>NaN</code> and quietly poison everything downstream.</p>`,
        },
      ],
      transfer: `This is <code>cvtColor(src, dst, COLOR_BGR2HSV)</code>, and on a GPU it is exactly
        what you just wrote: per-pixel, no communication, embarrassingly parallel. Watch the shape
        of it, though — the wedge is chosen by a branch, and threads in the same warp (CUDA) or
        subgroup (WebGPU/Metal) execute in lockstep, so a tile containing several wedges pays for
        every branch it contains rather than just its own. Branch <em>divergence</em> is the cost
        model here, and it is why production colour-conversion shaders are often written
        branch-free with <code>step()</code> and <code>mix()</code> instead.`,
      starterCode: `// One thread per pixel. Three quantities, two kernels, one branchy angle.
const gpu = new GPU({ mode });

const hue = gpu.createKernel(function (photo) {
  const pixel = photo[this.thread.y][this.thread.x];
  const v = Math.max(pixel[0], Math.max(pixel[1], pixel[2]));
  const m = Math.min(pixel[0], Math.min(pixel[1], pixel[2]));
  const c = v - m;
  // TODO: -1 when there is no chroma at all; otherwise the angle in degrees.
  // Which channel equals v picks the wedge.
  return 0;
}, { output: [64, 64] });

const saturation = gpu.createKernel(function (photo) {
  const pixel = photo[this.thread.y][this.thread.x];
  const v = Math.max(pixel[0], Math.max(pixel[1], pixel[2]));
  const m = Math.min(pixel[0], Math.min(pixel[1], pixel[2]));
  // TODO: (v - m) / v, but guard the black pixel where v is 0.
  return v - m;
}, { output: [64, 64] });

// Already written: paints your hue channel at full saturation and value, so
// the wheel is all you see. Pixels with no hue come out flat grey.
const paintHue = gpu.createKernel(function (h) {
  const angle = h[this.thread.y][this.thread.x];
  let r = 0.24;
  let g = 0.24;
  let b = 0.28;
  if (angle >= 0 && angle < 60) { r = 1; g = angle / 60; b = 0; }
  else if (angle >= 60 && angle < 120) { r = (120 - angle) / 60; g = 1; b = 0; }
  else if (angle >= 120 && angle < 180) { r = 0; g = 1; b = (angle - 120) / 60; }
  else if (angle >= 180 && angle < 240) { r = 0; g = (240 - angle) / 60; b = 1; }
  else if (angle >= 240 && angle < 300) { r = (angle - 240) / 60; g = 0; b = 1; }
  else if (angle >= 300) { r = 1; g = 0; b = (360 - angle) / 60; }
  this.color(r, g, b, 1);
}, { output: [64, 64], graphical: true });

const hues = await hue(chart);
const sats = await saturation(chart);
console.log('top-left swatch is pure red:  hue', hues[0][0], ' saturation', sats[0][0]);
console.log('the grey row has no hue:      hue', hues[28][4], ' saturation', sats[28][4]);

await paintHue(hues);
render(paintHue.canvas);
`,
      solutionCode: `// One thread per pixel. Three quantities, two kernels, one branchy angle.
const gpu = new GPU({ mode });

const hue = gpu.createKernel(function (photo) {
  const pixel = photo[this.thread.y][this.thread.x];
  const v = Math.max(pixel[0], Math.max(pixel[1], pixel[2]));
  const m = Math.min(pixel[0], Math.min(pixel[1], pixel[2]));
  const c = v - m;
  if (c === 0) {
    return -1;
  }
  if (v === pixel[0]) {
    const h = 60 * ((pixel[1] - pixel[2]) / c);
    if (h < 0) {
      return h + 360;
    }
    return h;
  }
  if (v === pixel[1]) {
    return 60 * ((pixel[2] - pixel[0]) / c + 2);
  }
  return 60 * ((pixel[0] - pixel[1]) / c + 4);
}, { output: [64, 64] });

const saturation = gpu.createKernel(function (photo) {
  const pixel = photo[this.thread.y][this.thread.x];
  const v = Math.max(pixel[0], Math.max(pixel[1], pixel[2]));
  const m = Math.min(pixel[0], Math.min(pixel[1], pixel[2]));
  if (v === 0) {
    return 0;
  }
  return (v - m) / v;
}, { output: [64, 64] });

// Already written: paints your hue channel at full saturation and value, so
// the wheel is all you see. Pixels with no hue come out flat grey.
const paintHue = gpu.createKernel(function (h) {
  const angle = h[this.thread.y][this.thread.x];
  let r = 0.24;
  let g = 0.24;
  let b = 0.28;
  if (angle >= 0 && angle < 60) { r = 1; g = angle / 60; b = 0; }
  else if (angle >= 60 && angle < 120) { r = (120 - angle) / 60; g = 1; b = 0; }
  else if (angle >= 120 && angle < 180) { r = 0; g = 1; b = (angle - 120) / 60; }
  else if (angle >= 180 && angle < 240) { r = 0; g = (240 - angle) / 60; b = 1; }
  else if (angle >= 240 && angle < 300) { r = (angle - 240) / 60; g = 0; b = 1; }
  else if (angle >= 300) { r = 1; g = 0; b = (360 - angle) / 60; }
  this.color(r, g, b, 1);
}, { output: [64, 64], graphical: true });

const hues = await hue(chart);
const sats = await saturation(chart);
console.log('top-left swatch is pure red:  hue', hues[0][0], ' saturation', sats[0][0]);
console.log('the grey row has no hue:      hue', hues[28][4], ' saturation', sats[28][4]);

await paintHue(hues);
render(paintHue.canvas);
`,
      inputs: () => ({ chart: swatchImage() }),
      publicTests: [
        {
          name: 'a hue map, a saturation map and a painted canvas',
          run: async ctx => {
            ctx.assert(ctx.kernels.length >= 2, `expected at least 2 kernels, found ${ctx.kernels.length}`);
            const { hue, saturation } = await findHueAndSaturation(ctx);
            ctx.assert(hue && saturation, 'expected two numeric 64 × 64 kernels — one hue map, one saturation map');
            const out = await hue(swatchImage());
            ctx.assert(out && out.length === SIZE && out[0].length === SIZE,
              `the hue map should be ${SIZE} × ${SIZE}`);
            ctx.assert(graphicalKernel(ctx), 'no graphical kernel found — paintHue should still be there');
            ctx.assert(ctx.canvas && ctx.canvas.width === SIZE && ctx.canvas.height === SIZE,
              'no 64 × 64 canvas — did you keep the render(paintHue.canvas) call?');
          },
        },
        {
          name: 'the eight pure swatches land on their wheel positions',
          run: async ctx => {
            const { hue } = await findHueAndSaturation(ctx);
            ctx.assert(hue, 'no hue kernel found');
            const image = swatchImage();
            const plain = image.plain;
            const out = await hue(image);
            for (let col = 0; col < 8; col++) {
              const [y, x] = patchCentre(0, col);
              const p = plain[y][x];
              const hint = diagnose(out[y][x], hueOf(p), 0.05, hueProbes(p));
              ctx.assertClose(out[y][x], hueOf(p), 0.05, hint ||
                `the swatch at [${y}][${x}] is rgb(${p.slice(0, 3).map(v => v.toFixed(2)).join(', ')})`);
            }
          },
        },
        {
          name: 'hue survives a change of brightness, saturation reports it',
          run: async ctx => {
            // Row 1 is row 0 at half value: the hue must not move at all, and
            // that invariance is the entire reason for this colour space.
            const { hue, saturation } = await findHueAndSaturation(ctx);
            ctx.assert(hue && saturation, 'expected a hue kernel and a saturation kernel');
            const image = swatchImage();
            const plain = image.plain;
            const hues = await hue(image);
            const sats = await saturation(image);
            for (let col = 0; col < 8; col++) {
              const [y0, x0] = patchCentre(0, col);
              const [y1, x1] = patchCentre(1, col);
              const p1 = plain[y1][x1];
              const hHint = diagnose(hues[y1][x1], hueOf(p1), 0.3, hueProbes(p1));
              ctx.assertClose(hues[y1][x1], hues[y0][x0], 0.3, hHint ||
                `the swatch at [${y1}][${x1}] is the one at [${y0}][${x0}] at half brightness — same hue`);
              const sHint = diagnose(sats[y1][x1], saturationOf(p1), 2e-3, saturationProbes(p1));
              ctx.assertClose(sats[y1][x1], saturationOf(p1), 2e-3, sHint ||
                `saturation at [${y1}][${x1}]`);
            }
          },
        },
        {
          name: 'grey has no hue, and black has no saturation',
          run: async ctx => {
            const { hue, saturation } = await findHueAndSaturation(ctx);
            ctx.assert(hue && saturation, 'expected a hue kernel and a saturation kernel');
            const image = swatchImage();
            const hues = await hue(image);
            const sats = await saturation(image);
            for (let col = 0; col < 8; col++) {
              const [y, x] = patchCentre(3, col);
              ctx.assertClose(hues[y][x], NO_HUE, 1e-3, noHueHint(hues[y][x]) ||
                `the grey swatch at [${y}][${x}] has no hue — expected the -1 sentinel`);
              ctx.assertClose(sats[y][x], 0, 1e-3, blackSaturationHint(sats[y][x]) ||
                `a grey swatch has zero saturation, at [${y}][${x}]`);
            }
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            const { hue, saturation } = await findHueAndSaturation(ctx);
            ctx.assert(hue && saturation, 'expected a hue kernel and a saturation kernel');
            const image = swatchImage();
            const plain = image.plain;
            const hues = await hue(image);
            const sats = await saturation(image);
            for (let y = 0; y < SIZE; y++) {
              for (let x = 0; x < SIZE; x++) {
                const p = plain[y][x];
                const expectedH = hueOf(p);
                const hHint = expectedH === NO_HUE
                  ? noHueHint(hues[y][x])
                  : transposeCellHint(hues[y][x], hueOf(plain[x][y]), 0.05, y, x, 'photo') ||
                    diagnose(hues[y][x], expectedH, 0.05, hueProbes(p));
                ctx.assertClose(hues[y][x], expectedH, 0.05, hHint || `hue at [${y}][${x}]`);
                const sHint = blackSaturationHint(sats[y][x]) ||
                  diagnose(sats[y][x], saturationOf(p), 2e-3, saturationProbes(p));
                ctx.assertClose(sats[y][x], saturationOf(p), 2e-3, sHint || `saturation at [${y}][${x}]`);
              }
            }
          },
        },
        {
          name: 'private test #2',
          run: async ctx => {
            // Flat colours, one per interesting case: the negative red wedge,
            // the green and blue wedges, pure black, and a mid grey.
            const { hue, saturation } = await findHueAndSaturation(ctx);
            ctx.assert(hue && saturation, 'expected a hue kernel and a saturation kernel');
            for (const pixel of [
              [1, 0.2, 0.6, 1], [0.2, 0.7, 0.35, 1], [0.15, 0.3, 0.8, 1],
              [0, 0, 0, 1], [0.4, 0.4, 0.4, 1], [1, 1, 1, 1],
            ]) {
              const image = constantImage(SIZE, pixel);
              const p = image.at(0, 0);
              const gotH = (await hue(image))[2][3];
              const gotS = (await saturation(image))[2][3];
              const expectedH = hueOf(p);
              const hHint = expectedH === NO_HUE
                ? noHueHint(gotH)
                : diagnose(gotH, expectedH, 0.05, hueProbes(p));
              ctx.assertClose(gotH, expectedH, 0.05, hHint ||
                `hue of a flat rgb(${p.slice(0, 3).map(v => v.toFixed(2)).join(', ')}) image`);
              const sHint = blackSaturationHint(gotS) ||
                diagnose(gotS, saturationOf(p), 2e-3, saturationProbes(p));
              ctx.assertClose(gotS, saturationOf(p), 2e-3, sHint ||
                `saturation of a flat rgb(${p.slice(0, 3).map(v => v.toFixed(2)).join(', ')}) image`);
            }
          },
        },
      ],
    },

    // ------------------------------------------------------------------ 3
    {
      slug: 'hue-wraps',
      title: 'The Midpoint of 350° and 10°',
      intro: `<p>Hue is an angle, and angles wrap. That one fact quietly breaks arithmetic you have
        been doing safely your entire career.</p>
        <p>Take two readings of the same pixel, from two frames: 350° and 10°. Both are red. They are
        20° apart on the wheel. Average them the obvious way and you get 180° — cyan. Not a
        slightly-off red: the <em>opposite colour</em>, out of two inputs that were nearly identical.
        Every mean, every interpolation, every blur that touches a hue channel has this hole in
        it.</p>
        <p>The repair is to stop pretending the number line has no seam. Take <code>b − a</code>,
        fold it onto the short way round by adding or subtracting 360 until it lands in
        −180 … 180, walk half of it from <code>a</code>, and fold the answer back into 0 … 360.
        Four lines, no cleverness — just refusing to subtract two angles as if they were
        distances.</p>`,
      goal: `<strong>Goal:</strong> return the midpoint of <code>hueA[i]</code> and
        <code>hueB[i]</code> the short way round, as an angle in <code>0 … 360</code>.`,
      requirements: [
        'One thread per pair — <code>output: [64]</code>, indexed with <code>this.thread.x</code>',
        'Fold <code>b − a</code> into <code>−180 … 180</code> <em>before</em> you halve it',
        'Fold the answer back: every output must land in <code>0 … 360</code>, none negative and none <code>360</code> or more',
        'The first pair, <code>350</code> and <code>10</code>, must come out at <code>0</code> — not <code>180</code>',
      ],
      hints: [
        {
          title: 'Hint 1 — the short way round',
          body: `<p>The difference between two angles is only ever at most 180°. If the plain
            subtraction gives you more than that, you went the long way round:</p>
<pre><code>let d = b[this.thread.x] - a[this.thread.x];
if (d &gt; 180) { d = d - 360; }
if (d &lt; -180) { d = d + 360; }</code></pre>`,
        },
        {
          title: 'Hint 2 — half a step, then home',
          body: `<p>Walk half of that difference from <code>a</code>, then bring the result back
            onto the wheel. One check each way is enough, because <code>a</code> is already in
            range and you moved it by at most 90°:</p>
<pre><code>let m = a[this.thread.x] + d / 2;
if (m &lt; 0) { m = m + 360; }
if (m &gt;= 360) { m = m - 360; }
return m;</code></pre>`,
        },
        {
          title: 'Hint 3 — check the first four',
          body: `<p>The first eight pairs of the input straddle the seam deliberately.
            <code>(350, 10)</code> and <code>(10, 350)</code> must both give <code>0</code>,
            <code>(358.5, 4.5)</code> gives <code>1.5</code>, and <code>(340, 20)</code> gives
            <code>0</code> as well. If those four come out near 180, the fold is missing.</p>`,
        },
      ],
      transfer: `Wrapping quantities are everywhere in vision and nowhere in your standard library:
        gradient orientation in HOG and SIFT, optical-flow direction, the phase channel of an FFT,
        compass bearings, the time of day. The standard fix for averaging <em>many</em> of them is
        prettier than this one and just as parallel — turn each angle into a unit vector, sum the
        vectors (which is a plain reduction), and take <code>atan2</code> of the total. That is what
        circular statistics libraries do, what CUDA and WGSL kernels do, and it removes the branches
        entirely, which on a GPU is worth having.`,
      starterCode: `// 64 pairs of angles, one thread each. Angles are not numbers on a line.
const gpu = new GPU({ mode });

const midpoint = gpu.createKernel(function (a, b) {
  // TODO: the midpoint of two angles, taken the SHORT way round,
  // folded back into 0 ... 360.
  return (a[this.thread.x] + b[this.thread.x]) / 2;
}, { output: [64] });

const mid = await midpoint(hueA, hueB);

for (let i = 0; i < 4; i++) {
  console.log(hueA[i] + '° and ' + hueB[i] + '° → ' + mid[i] + '°');
}
`,
      solutionCode: `// 64 pairs of angles, one thread each. Angles are not numbers on a line.
const gpu = new GPU({ mode });

const midpoint = gpu.createKernel(function (a, b) {
  // the short way round: the difference between two angles is never above 180
  let d = b[this.thread.x] - a[this.thread.x];
  if (d > 180) { d = d - 360; }
  if (d < -180) { d = d + 360; }

  // half a step from a, then back onto the wheel
  let m = a[this.thread.x] + d / 2;
  if (m < 0) { m = m + 360; }
  if (m >= 360) { m = m - 360; }
  return m;
}, { output: [64] });

const mid = await midpoint(hueA, hueB);

for (let i = 0; i < 4; i++) {
  console.log(hueA[i] + '° and ' + hueB[i] + '° → ' + mid[i] + '°');
}
`,
      inputs: utils => makeHuePairs(utils),
      publicTests: [
        {
          name: 'the pair <code>(350, 10)</code> averages to <code>0</code>, not <code>180</code>',
          run: async ctx => {
            ctx.assert(ctx.kernels.length >= 1, 'no kernel was created — call gpu.createKernel()');
            const { hueA, hueB } = makeHuePairs(ctx.utils);
            const out = await ctx.kernel(hueA, hueB);
            ctx.assert(out && out.length === PAIRS, `expected ${PAIRS} midpoints, got ${out && out.length}`);
            const hint = diagnose(out[0], 0, 1e-2, [
              [180, 'the plain mean of 350 and 10 is 180 — cyan, from two reds. That is the bug this task is about: fold the difference onto the short way round before you halve it'],
              [360, 'the short way round was taken, but 360 was never folded back to 0'],
              [hueB[0], 'that is simply the second hue — the whole difference was added instead of half of it'],
              [hueA[0], 'that is simply the first hue — the difference was folded but never added'],
            ]);
            ctx.assertClose(out[0], 0, 1e-2, hint || 'the midpoint of 350° and 10°');
          },
        },
        {
          name: 'every one of the 64 midpoints is the circular mean',
          run: async ctx => {
            const { hueA, hueB } = makeHuePairs(ctx.utils);
            const out = await ctx.kernel(hueA, hueB);
            const hint = diagnoseAll(PAIRS, i => out[i], i => circularMean(hueA[i], hueB[i]), 1e-2,
              midpointProbes(hueA, hueB));
            for (let i = 0; i < PAIRS; i++) {
              ctx.assertClose(out[i], circularMean(hueA[i], hueB[i]), 1e-2, hint ||
                `pair ${i}: ${hueA[i]}° and ${hueB[i]}°`);
            }
          },
        },
        {
          name: 'every answer lands in <code>0 … 360</code>',
          run: async ctx => {
            const { hueA, hueB } = makeHuePairs(ctx.utils);
            const out = await ctx.kernel(hueA, hueB);
            for (let i = 0; i < PAIRS; i++) {
              ctx.assert(Number.isFinite(out[i]),
                `pair ${i} came back as ${out[i]} — an angle has to be a number`);
              ctx.assert(out[i] >= 0 && out[i] < 360,
                `pair ${i} came back as ${out[i]}°, which is off the wheel — after halving the ` +
                `difference the result still has to be folded back into 0 … 360`);
            }
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            // A fresh set of pairs the learner has never seen, plus the six
            // exact cases that a fold either gets right or gets very wrong.
            const { hueA, hueB } = makeHuePairs(ctx.utils, 20261);
            const out = await ctx.kernel(hueA, hueB);
            const hint = diagnoseAll(PAIRS, i => out[i], i => circularMean(hueA[i], hueB[i]), 1e-2,
              midpointProbes(hueA, hueB));
            for (let i = 0; i < PAIRS; i++) {
              ctx.assertClose(out[i], circularMean(hueA[i], hueB[i]), 1e-2, hint ||
                `pair ${i}: ${hueA[i]}° and ${hueB[i]}°`);
            }
          },
        },
        {
          name: 'private test #2',
          run: async ctx => {
            const a = new Array(PAIRS);
            const b = new Array(PAIRS);
            const cases = [
              [359, 1, 0], [1, 359, 0], [0, 0, 0], [90, 90, 90],
              [270, 30, 330], [30, 270, 330], [200, 160, 180], [45, 135, 90],
            ];
            for (let i = 0; i < PAIRS; i++) {
              const [x, y] = cases[i % cases.length];
              a[i] = x;
              b[i] = y;
            }
            const out = await ctx.kernel(a, b);
            for (let i = 0; i < cases.length; i++) {
              const [x, y, expected] = cases[i];
              const hint = diagnose(out[i], expected, 1e-2, [
                [(x + y) / 2, 'the plain arithmetic mean — the difference was never folded onto the short way round'],
                [expected + 360, 'the answer overshot 360 and was never folded back'],
                [expected - 360, 'the answer went below 0 and was never folded back'],
              ]);
              ctx.assertClose(out[i], expected, 1e-2, hint || `${x}° and ${y}°`);
            }
          },
        },
      ],
    },

    // ------------------------------------------------------------------ 4
    {
      slug: 'colour-mask',
      title: 'Select by Colour',
      intro: `<p>Here is the whole module's argument, in one exercise. You want every pixel of the red
        ball in <code>frame</code> — all of it, top to bottom, shadow included.</p>
        <p>In RGB that is a threshold on numbers that move when the light moves. "Red" comes out as
        something like <code>r &gt; 0.5 and r − g &gt; 0.3</code>, and it works beautifully on the lit
        top of the ball. Turn the light down and every one of those numbers falls with it, until the
        test stops being true — for a pixel that is exactly as red as it ever was. That kernel is
        written for you below, so you can watch it happen.</p>
        <p>In HSV, brightness lives in V and nowhere else. Multiply a pixel's r, g and b by the same
        factor and H and S do not move at all. So the test becomes "hue near red, saturated enough",
        and the shadow costs you nothing. One wrinkle, and it is task 3's wrinkle: red sits at 0°, so
        a 15° wedge around it runs from 345° up over the seam to 15°. <code>h &gt; 345 &amp;&amp; h
        &lt; 15</code> is true for no angle whatsoever. Measure the distance <em>round the wheel</em>
        instead.</p>
        ${ARRAY_LAYOUT}`,
      goal: `<strong>Goal:</strong> finish <code>hsvMask</code> — <code>1</code> for pixels within
        <code>this.constants.tol</code> degrees of <code>this.constants.target</code> on the wheel
        and at least <code>this.constants.minSat</code> saturated, <code>0</code> for everything
        else — and log how many pixels each mask found.`,
      requirements: [
        'Fold <code>h − target</code> into <code>−180 … 180</code> before comparing — the wedge straddles 0°',
        'Require <code>s &gt;= this.constants.minSat</code>: a pixel with no hue reports <code>-1</code>, which is one degree from red, and the saturation floor is what keeps it out',
        'Return exactly <code>1</code> or <code>0</code>, nothing in between',
        '<code>console.log</code> both mask counts — the HSV one should find the whole ball, the RGB one only its lit half',
      ],
      hints: [
        {
          title: 'Hint 1 — distance round the wheel',
          body: `<p>Exactly the fold from task 3, then drop the sign:</p>
<pre><code>let d = h - this.constants.target;
if (d &gt; 180) { d = d - 360; }
if (d &lt; -180) { d = d + 360; }
if (d &lt; 0) { d = -d; }</code></pre>
          <p>Now <code>d</code> is a distance in degrees, 0 … 180, and it does not care where the
            seam is.</p>`,
        },
        {
          title: 'Hint 2 — the test itself',
          body: `<p>Two conditions, and both matter:</p>
<pre><code>if (d &lt;= this.constants.tol &amp;&amp; s &gt;= this.constants.minSat) {
  return 1;
}
return 0;</code></pre>`,
        },
        {
          title: 'Hint 3 — reading the counts',
          body: `<p>Total each mask with a plain nested loop in JavaScript after the kernels have
            run. The HSV count should be comfortably the larger — and the gap between them is the
            part of the ball that RGB gave up on because a lamp was dimmer there.</p>`,
        },
      ],
      transfer: `This is chroma keying, and the reason a green screen is <em>green</em>: it is the
        channel a sensor samples most finely, and it is nowhere near skin. Real compositors go
        further into spaces built for exactly this — YCbCr, or CIE L*a*b* — where the two
        chromaticity axes are perpendicular to lightness by construction, so a key becomes a
        distance in a plane rather than a wedge with a seam in it. On the GPU it stays what you just
        wrote: one thread per pixel, no communication, and the mask is a texture the next pass
        reads.`,
      starterCode: `// The same intent — "that's red" — expressed in two colour spaces.
const gpu = new GPU({ mode });

const hue = gpu.createKernel(function (photo) {
  const pixel = photo[this.thread.y][this.thread.x];
  const v = Math.max(pixel[0], Math.max(pixel[1], pixel[2]));
  const m = Math.min(pixel[0], Math.min(pixel[1], pixel[2]));
  const c = v - m;
  if (c === 0) {
    return -1;
  }
  if (v === pixel[0]) {
    const h = 60 * ((pixel[1] - pixel[2]) / c);
    if (h < 0) {
      return h + 360;
    }
    return h;
  }
  if (v === pixel[1]) {
    return 60 * ((pixel[2] - pixel[0]) / c + 2);
  }
  return 60 * ((pixel[0] - pixel[1]) / c + 4);
}, { output: [64, 64] });

// "Red" in RGB: bright, and much redder than it is green or blue.
const rgbMask = gpu.createKernel(function (photo) {
  const pixel = photo[this.thread.y][this.thread.x];
  if (pixel[0] > 0.5 && pixel[0] - pixel[1] > 0.3 && pixel[0] - pixel[2] > 0.3) {
    return 1;
  }
  return 0;
}, { output: [64, 64] });

const hsvMask = gpu.createKernel(function (photo, hueMap) {
  const pixel = photo[this.thread.y][this.thread.x];
  const v = Math.max(pixel[0], Math.max(pixel[1], pixel[2]));
  const m = Math.min(pixel[0], Math.min(pixel[1], pixel[2]));
  let s = 0;
  if (v > 0) { s = (v - m) / v; }
  const h = hueMap[this.thread.y][this.thread.x];

  // TODO: 1 when h is within this.constants.tol degrees of
  // this.constants.target ON THE WHEEL, and s clears this.constants.minSat.
  // The red wedge runs from 345 up over the seam to 15 ... doesn't it?
  if (h > 345 && h < 15) {
    return 1;
  }
  return 0;
}, {
  output: [64, 64],
  constants: { target: 0, tol: 15, minSat: 0.35 },
});

const hues = await hue(frame);
const inRgb = await rgbMask(frame);
const inHsv = await hsvMask(frame, hues);

// TODO: total both masks and log the two counts.
`,
      solutionCode: `// The same intent — "that's red" — expressed in two colour spaces.
const gpu = new GPU({ mode });

const hue = gpu.createKernel(function (photo) {
  const pixel = photo[this.thread.y][this.thread.x];
  const v = Math.max(pixel[0], Math.max(pixel[1], pixel[2]));
  const m = Math.min(pixel[0], Math.min(pixel[1], pixel[2]));
  const c = v - m;
  if (c === 0) {
    return -1;
  }
  if (v === pixel[0]) {
    const h = 60 * ((pixel[1] - pixel[2]) / c);
    if (h < 0) {
      return h + 360;
    }
    return h;
  }
  if (v === pixel[1]) {
    return 60 * ((pixel[2] - pixel[0]) / c + 2);
  }
  return 60 * ((pixel[0] - pixel[1]) / c + 4);
}, { output: [64, 64] });

// "Red" in RGB: bright, and much redder than it is green or blue.
const rgbMask = gpu.createKernel(function (photo) {
  const pixel = photo[this.thread.y][this.thread.x];
  if (pixel[0] > 0.5 && pixel[0] - pixel[1] > 0.3 && pixel[0] - pixel[2] > 0.3) {
    return 1;
  }
  return 0;
}, { output: [64, 64] });

const hsvMask = gpu.createKernel(function (photo, hueMap) {
  const pixel = photo[this.thread.y][this.thread.x];
  const v = Math.max(pixel[0], Math.max(pixel[1], pixel[2]));
  const m = Math.min(pixel[0], Math.min(pixel[1], pixel[2]));
  let s = 0;
  if (v > 0) { s = (v - m) / v; }
  const h = hueMap[this.thread.y][this.thread.x];

  // distance round the wheel, so the seam at 360/0 is not a wall
  let d = h - this.constants.target;
  if (d > 180) { d = d - 360; }
  if (d < -180) { d = d + 360; }
  if (d < 0) { d = -d; }

  if (d <= this.constants.tol && s >= this.constants.minSat) {
    return 1;
  }
  return 0;
}, {
  output: [64, 64],
  constants: { target: 0, tol: 15, minSat: 0.35 },
});

const hues = await hue(frame);
const inRgb = await rgbMask(frame);
const inHsv = await hsvMask(frame, hues);

let rgbCount = 0;
let hsvCount = 0;
for (let y = 0; y < 64; y++) {
  for (let x = 0; x < 64; x++) {
    rgbCount += inRgb[y][x];
    hsvCount += inHsv[y][x];
  }
}
console.log('RGB threshold found:', rgbCount, 'pixels');
console.log('HSV threshold found:', hsvCount, 'pixels');
`,
      inputs: () => ({ frame: chromaScene(SIZE) }),
      publicTests: [
        {
          name: 'the HSV mask is <code>1</code>/<code>0</code> and covers the ball, shadow included',
          run: async ctx => {
            const mask = kernelWithArgs(ctx, 2);
            ctx.assert(mask, 'no kernel taking two arguments found — hsvMask takes the frame and the hue map');
            const image = chromaScene(SIZE);
            const plain = image.plain;
            const hue = await hueKernel(ctx);
            ctx.assert(hue, 'no hue-map kernel found — the one provided answers 120 for pure green, and hsvMask needs its output');
            const out = await mask(image, await hue(image));
            for (let y = 0; y < SIZE; y++) {
              for (let x = 0; x < SIZE; x++) {
                ctx.assert(out[y][x] === 0 || out[y][x] === 1,
                  `cell [${y}][${x}] is ${out[y][x]} — a mask holds exactly 1 or 0`);
              }
            }
            const hint = diagnoseGrid(out, plain, hsvMask, 0.5, maskProbes());
            for (const [y, x] of [[14, 31], [14, 33], [46, 31], [46, 33], [30, 5], [60, 4], [60, 18]]) {
              ctx.assertClose(out[y][x], hsvMask(plain[y][x]), 0.5, hint ||
                `cell [${y}][${x}] is rgb(${plain[y][x].slice(0, 3).map(v => v.toFixed(3)).join(', ')}), ` +
                `hue ${hueOf(plain[y][x]).toFixed(1)}°, saturation ${saturationOf(plain[y][x]).toFixed(2)}`);
            }
          },
        },
        {
          name: 'the mask survives the shadow the RGB threshold loses',
          run: async ctx => {
            const mask = kernelWithArgs(ctx, 2);
            const hue = await hueKernel(ctx);
            ctx.assert(mask && hue, 'expected a hue-map kernel and a two-argument mask kernel');
            const image = chromaScene(SIZE);
            const plain = image.plain;
            const out = await mask(image, await hue(image));
            let got = 0;
            let expected = 0;
            let inRgb = 0;
            for (let y = 0; y < SIZE; y++) {
              for (let x = 0; x < SIZE; x++) {
                got += out[y][x];
                expected += hsvMask(plain[y][x]);
                inRgb += rgbMask(plain[y][x]);
              }
            }
            const hint = diagnoseGrid(out, plain, hsvMask, 0.5, maskProbes());
            ctx.assertClose(got, expected, 0.5, hint ||
              `the HSV mask should hold ${expected} pixels (the RGB one manages ${inRgb})`);
          },
        },
        {
          name: 'both counts are logged',
          run: async ctx => {
            const plain = chromaScene(SIZE).plain;
            let hsv = 0;
            let rgb = 0;
            for (let y = 0; y < SIZE; y++) {
              for (let x = 0; x < SIZE; x++) {
                hsv += hsvMask(plain[y][x]);
                rgb += rgbMask(plain[y][x]);
              }
            }
            ctx.assert(loggedNear(ctx.logs, rgb, 0.5),
              `log the RGB mask's count too — expected ${rgb} in the console output`);
            ctx.assert(loggedNear(ctx.logs, hsv, 0.5),
              `log the HSV mask's count — expected ${hsv} in the console output`);
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            const mask = kernelWithArgs(ctx, 2);
            const hue = await hueKernel(ctx);
            ctx.assert(mask && hue, 'expected a hue-map kernel and a two-argument mask kernel');
            const image = chromaScene(SIZE);
            const plain = image.plain;
            const out = await mask(image, await hue(image));
            const hint = diagnoseGrid(out, plain, hsvMask, 0.5, maskProbes());
            for (let y = 0; y < SIZE; y++) {
              for (let x = 0; x < SIZE; x++) {
                ctx.assertClose(out[y][x], hsvMask(plain[y][x]), 0.5, hint || `cell [${y}][${x}]`);
              }
            }
          },
        },
        {
          name: 'private test #2',
          run: async ctx => {
            // The thesis, as an assertion: dim the WHOLE frame by half and the
            // mask must not change by a single pixel.
            const mask = kernelWithArgs(ctx, 2);
            const hue = await hueKernel(ctx);
            ctx.assert(mask && hue, 'expected a hue-map kernel and a two-argument mask kernel');
            const bright = chromaScene(SIZE);
            const dimmed = plainToImageData(
              bright.plain.map(row => row.map(p => quantizePixel([p[0] / 2, p[1] / 2, p[2] / 2, 1])))
            );
            const a = await mask(bright, await hue(bright));
            const b = await mask(dimmed, await hue(dimmed));
            let differences = 0;
            for (let y = 0; y < SIZE; y++) {
              for (let x = 0; x < SIZE; x++) if (a[y][x] !== b[y][x]) differences++;
            }
            ctx.assert(differences === 0,
              `${differences} pixels changed when the whole frame was dimmed by half. Nothing about ` +
              `the picture's colour changed, only its brightness — if the mask moved, some part of ` +
              `it is still reading a channel that brightness lives in`);
            ctx.assertClose(sumOf(b.map(row => sumOf(Array.from(row)))), sumOf(bright.plain.map(
              row => sumOf(row.map(hsvMask))
            )), 0.5, 'the dimmed frame should give the same mask as the bright one');
          },
        },
      ],
    },

    // ------------------------------------------------------------------ 5
    {
      slug: 'dominant-hue',
      title: 'Payoff: What Colour Is This Picture?',
      intro: `<p>The payoff, and a question a person can answer in a glance: what colour is this
        picture, mostly? Cut the wheel into 12 bins of 30°, count how many pixels fall in each, and
        read off the fullest one.</p>
        <p>The counting is a histogram, and the one-thread-per-bin shape it has to take when you have
        no atomics is exactly what Histograms &amp; Binning derives — so that kernel comes ready
        made below, along with the two you wrote in task 2. What is left for you is the part that is
        about colour: turning each pixel into a bin number, and refusing to answer for the pixels
        that have no colour to report.</p>
        <p>That refusal is the difference between an answer and a rumour. The stones along the bottom
        of this picture are grey to within a rounding error, and the direction of a rounding error is
        still a perfectly valid-looking angle. Bin them and they smear a plausible-looking 96 pixels
        of nonsense across the whole wheel. Drop anything below a saturation floor and the histogram
        only counts pixels that actually have a hue — which is why its counts come to fewer than
        4,096, on purpose.</p>
        ${ARRAY_LAYOUT}`,
      goal: `<strong>Goal:</strong> finish <code>hueBin</code> — the bin index for each pixel, or
        <code>-1</code> for a pixel with no usable hue — then find the fullest bin in JavaScript and
        log it.`,
      requirements: [
        'Return <code>-1</code> when the saturation is below <code>this.constants.floor</code>',
        'Otherwise <code>Math.floor(h / this.constants.width)</code>, clamped to <code>this.constants.bins - 1</code>',
        'Find the fullest bin in plain JavaScript and <code>console.log</code> its index',
        'Check the counts: they should total <code>4000</code>, not 4,096 — the 96 stones are excluded on purpose',
      ],
      hints: [
        {
          title: 'Hint 1 — the floor first',
          body: `<p>The saturation test comes before anything else, because a pixel that fails it
            has no angle worth binning:</p>
<pre><code>if (sat[this.thread.y][this.thread.x] &lt; this.constants.floor) {
  return -1;
}</code></pre>`,
        },
        {
          title: 'Hint 2 — the bin',
          body: `<p>30° per bin, so the index is the hue divided by the width and floored. The
            clamp is the same one Histograms &amp; Binning needed: a hue of exactly 360 would
            otherwise land in bin 12, which no thread owns.</p>
<pre><code>const h = hue[this.thread.y][this.thread.x];
return Math.min(this.constants.bins - 1, Math.floor(h / this.constants.width));</code></pre>`,
        },
        {
          title: 'Hint 3 — reading the answer',
          body: `<p>The fullest bin is a plain loop over 12 numbers — not worth a kernel. Bin
            <em>b</em> covers <code>b * 30</code> to <code>(b + 1) * 30</code> degrees, so printing
            that range alongside the index tells you what colour the picture actually is.</p>`,
        },
      ],
      transfer: `Hue histograms are the backbone of colour-based tracking: the CAMShift tracker that
        ships with OpenCV builds one over a target region and then back-projects it into each new
        frame, precisely because hue survives the target walking through a shadow. The two-pass
        shape — derive a per-pixel quantity into a map, then bin the map — is the same one every
        GPU histogram uses, on every platform, and for the same reason: binning has to read the data
        many times, so you want it reading something cheap.`,
      starterCode: `// Map first (a bin per pixel), bin second (a thread per bin).
const gpu = new GPU({ mode });

// The two kernels from task 2, unchanged.
const hue = gpu.createKernel(function (photo) {
  const pixel = photo[this.thread.y][this.thread.x];
  const v = Math.max(pixel[0], Math.max(pixel[1], pixel[2]));
  const m = Math.min(pixel[0], Math.min(pixel[1], pixel[2]));
  const c = v - m;
  if (c === 0) {
    return -1;
  }
  if (v === pixel[0]) {
    const h = 60 * ((pixel[1] - pixel[2]) / c);
    if (h < 0) {
      return h + 360;
    }
    return h;
  }
  if (v === pixel[1]) {
    return 60 * ((pixel[2] - pixel[0]) / c + 2);
  }
  return 60 * ((pixel[0] - pixel[1]) / c + 4);
}, { output: [64, 64] });

const saturation = gpu.createKernel(function (photo) {
  const pixel = photo[this.thread.y][this.thread.x];
  const v = Math.max(pixel[0], Math.max(pixel[1], pixel[2]));
  const m = Math.min(pixel[0], Math.min(pixel[1], pixel[2]));
  if (v === 0) {
    return 0;
  }
  return (v - m) / v;
}, { output: [64, 64] });

const hueBin = gpu.createKernel(function (hue, sat) {
  // TODO: -1 when this pixel's saturation is below this.constants.floor,
  // otherwise its bin: the hue divided by this.constants.width, floored,
  // and clamped to this.constants.bins - 1.
  return 0;
}, {
  output: [64, 64],
  constants: { bins: 12, width: 30, floor: 0.15 },
});

// One thread per bin, each scanning the whole map — the shape a GPU histogram
// has to take when nobody can increment anybody else's counter.
const histogram = gpu.createKernel(function (bins) {
  let count = 0;
  for (let y = 0; y < this.constants.size; y++) {
    for (let x = 0; x < this.constants.size; x++) {
      if (bins[y][x] === this.thread.x) {
        count++;
      }
    }
  }
  return count;
}, { output: [12], constants: { size: 64 } });

const counts = await histogram(await hueBin(await hue(photo), await saturation(photo)));
console.log('counts:', counts);

// TODO: find the fullest bin and log it. Bin b covers b * 30 ... (b + 1) * 30 degrees.
`,
      solutionCode: `// Map first (a bin per pixel), bin second (a thread per bin).
const gpu = new GPU({ mode });

// The two kernels from task 2, unchanged.
const hue = gpu.createKernel(function (photo) {
  const pixel = photo[this.thread.y][this.thread.x];
  const v = Math.max(pixel[0], Math.max(pixel[1], pixel[2]));
  const m = Math.min(pixel[0], Math.min(pixel[1], pixel[2]));
  const c = v - m;
  if (c === 0) {
    return -1;
  }
  if (v === pixel[0]) {
    const h = 60 * ((pixel[1] - pixel[2]) / c);
    if (h < 0) {
      return h + 360;
    }
    return h;
  }
  if (v === pixel[1]) {
    return 60 * ((pixel[2] - pixel[0]) / c + 2);
  }
  return 60 * ((pixel[0] - pixel[1]) / c + 4);
}, { output: [64, 64] });

const saturation = gpu.createKernel(function (photo) {
  const pixel = photo[this.thread.y][this.thread.x];
  const v = Math.max(pixel[0], Math.max(pixel[1], pixel[2]));
  const m = Math.min(pixel[0], Math.min(pixel[1], pixel[2]));
  if (v === 0) {
    return 0;
  }
  return (v - m) / v;
}, { output: [64, 64] });

const hueBin = gpu.createKernel(function (hue, sat) {
  if (sat[this.thread.y][this.thread.x] < this.constants.floor) {
    return -1;
  }
  const h = hue[this.thread.y][this.thread.x];
  return Math.min(this.constants.bins - 1, Math.floor(h / this.constants.width));
}, {
  output: [64, 64],
  constants: { bins: 12, width: 30, floor: 0.15 },
});

// One thread per bin, each scanning the whole map — the shape a GPU histogram
// has to take when nobody can increment anybody else's counter.
const histogram = gpu.createKernel(function (bins) {
  let count = 0;
  for (let y = 0; y < this.constants.size; y++) {
    for (let x = 0; x < this.constants.size; x++) {
      if (bins[y][x] === this.thread.x) {
        count++;
      }
    }
  }
  return count;
}, { output: [12], constants: { size: 64 } });

const counts = await histogram(await hueBin(await hue(photo), await saturation(photo)));
console.log('counts:', counts);

let total = 0;
let fullest = 0;
for (let b = 0; b < counts.length; b++) {
  total += counts[b];
  if (counts[b] > counts[fullest]) {
    fullest = b;
  }
}
console.log('pixels with a hue:', total, 'of', 64 * 64);
console.log('fullest bin:', fullest, '=', fullest * 30, '...', (fullest + 1) * 30, 'degrees');
`,
      inputs: utils => ({ photo: forestImage(utils, SIZE) }),
      publicTests: [
        {
          name: 'the bin map holds whole bin numbers, and <code>-1</code> for the stones',
          run: async ctx => {
            const bin = kernelWithArgs(ctx, 2);
            ctx.assert(bin, 'no kernel taking two arguments found — hueBin takes the hue map and the saturation map');
            const image = forestImage(ctx.utils, SIZE);
            const plain = image.plain;
            const { hue, saturation } = await findHueAndSaturation(ctx);
            ctx.assert(hue && saturation, 'the provided hue and saturation kernels should still be there');
            const out = await bin(await hue(image), await saturation(image));
            ctx.assert(out && out.length === SIZE && out[0].length === SIZE,
              `the bin map should be ${SIZE} × ${SIZE}`);
            for (let y = 0; y < SIZE; y++) {
              for (let x = 0; x < SIZE; x++) {
                ctx.assert(Number.isFinite(out[y][x]) && Math.abs(out[y][x] - Math.round(out[y][x])) < 1e-3,
                  `cell [${y}][${x}] is ${out[y][x]} — a bin index is a whole number`);
                ctx.assert(out[y][x] >= -1 && out[y][x] < BINS,
                  `cell [${y}][${x}] is bin ${out[y][x]}, and there are only ${BINS} bins (plus -1 for "no hue")`);
              }
            }
            const hint = floorHint(out, plain) || diagnoseGrid(out, plain, binOf, 0.5, binProbes());
            for (const [y, x] of [[4, 40], [30, 12], [50, 31], [61, 3], [20, 55]]) {
              ctx.assertClose(out[y][x], binOf(plain[y][x]), 0.5, hint ||
                `cell [${y}][${x}] has hue ${hueOf(plain[y][x]).toFixed(1)}° ` +
                `and saturation ${saturationOf(plain[y][x]).toFixed(3)}`);
            }
          },
        },
        {
          name: 'the histogram matches, and totals <code>4000</code> rather than 4,096',
          run: async ctx => {
            const bin = kernelWithArgs(ctx, 2);
            const line = kernelOfRank(ctx, 1);
            ctx.assert(bin && line, 'expected a two-argument bin kernel and a 12-thread histogram kernel');
            const image = forestImage(ctx.utils, SIZE);
            const plain = image.plain;
            const { hue, saturation } = await findHueAndSaturation(ctx);
            ctx.assert(hue && saturation, 'the provided hue and saturation kernels should still be there');
            const bins = await bin(await hue(image), await saturation(image));
            const counts = await line(bins);
            const expected = histogramOf(plain, binOf);
            const noHue = SIZE * SIZE - sumOf(expected);
            const hint = floorHint(bins, plain) || diagnoseGrid(bins, plain, binOf, 0.5, binProbes()) ||
              totalHint(counts, SIZE * SIZE, noHue);
            for (let b = 0; b < BINS; b++) {
              ctx.assertClose(counts[b], expected[b], 0.5, hint ||
                `bin ${b} covers ${b * BIN_WIDTH}° … ${(b + 1) * BIN_WIDTH}°`);
            }
          },
        },
        {
          name: 'the fullest bin is found and logged',
          run: async ctx => {
            const plain = forestImage(ctx.utils, SIZE).plain;
            const expected = histogramOf(plain, binOf);
            const best = fullestBin(expected);
            ctx.assert(loggedNear(ctx.logs, best, 1e-6),
              `log the index of the fullest bin — expected ${best} ` +
              `(${best * BIN_WIDTH}° … ${(best + 1) * BIN_WIDTH}°, which is the foliage) in the console output`);
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            const bin = kernelWithArgs(ctx, 2);
            ctx.assert(bin, 'expected a two-argument bin kernel');
            const image = forestImage(ctx.utils, SIZE, 771);
            const plain = image.plain;
            const { hue, saturation } = await findHueAndSaturation(ctx);
            ctx.assert(hue && saturation, 'the provided hue and saturation kernels should still be there');
            const out = await bin(await hue(image), await saturation(image));
            const hint = floorHint(out, plain) || diagnoseGrid(out, plain, binOf, 0.5, binProbes());
            for (let y = 0; y < SIZE; y++) {
              for (let x = 0; x < SIZE; x++) {
                ctx.assertClose(out[y][x], binOf(plain[y][x]), 0.5, hint || `cell [${y}][${x}]`);
              }
            }
          },
        },
        {
          name: 'private test #2',
          run: async ctx => {
            // Flat images, one per edge: a colour that is squarely inside a
            // bin, one sitting exactly on a bin edge, a grey that must be
            // refused, and black.
            const bin = kernelWithArgs(ctx, 2);
            const line = kernelOfRank(ctx, 1);
            ctx.assert(bin && line, 'expected a bin kernel and a histogram kernel');
            const { hue, saturation } = await findHueAndSaturation(ctx);
            ctx.assert(hue && saturation, 'the provided hue and saturation kernels should still be there');
            for (const [pixel, expected] of [
              [[1, 0, 0, 1], 0], [[0, 1, 0, 1], 4], [[0, 0, 1, 1], 8],
              [[0, 1, 1, 1], 6], [[0.5, 0.5, 0.5, 1], -1], [[0, 0, 0, 1], -1],
            ]) {
              const image = constantImage(SIZE, pixel);
              const out = await bin(await hue(image), await saturation(image));
              const hint = diagnose(out[3][7], expected, 0.5, [
                [expected + 1, 'the bin index is one too high — a hue exactly on a bin edge belongs to the bin ABOVE it, so this wants Math.floor, not Math.round'],
                [binNoFloor(image.at(0, 0)), 'a pixel with no usable hue was binned anyway — check the saturation floor before the angle'],
              ]);
              ctx.assertClose(out[3][7], expected, 0.5, hint ||
                `a flat rgb(${pixel.slice(0, 3).join(', ')}) image should be bin ${expected}`);
            }
            // …and one that is entirely grey has an empty histogram.
            const grey = constantImage(SIZE, [0.6, 0.6, 0.6, 1]);
            const counts = await line(await bin(await hue(grey), await saturation(grey)));
            ctx.assertClose(sumOf(Array.from(counts)), 0, 0.5,
              'a picture with no colour in it at all should produce an empty histogram — every ' +
              'pixel is below the saturation floor, so no bin gets anything');
          },
        },
      ],
    },
  ],
};
