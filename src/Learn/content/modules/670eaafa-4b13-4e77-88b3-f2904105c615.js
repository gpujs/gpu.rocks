// Module: Thresholding & Morphology — uuid 670eaafa-4b13-4e77-88b3-f2904105c615 (short id 670eaafa).
// The file name is the uuid; identity lives in the exported object below,
// never in the path. No legacyId: this module postdates uuids.
//
// Computer Vision — Thresholding & Morphology.
//
// Six tasks: one global threshold and the two corners it ruins → Otsu reading
// that number off the histogram → a threshold per neighbourhood, which is the
// box-blur sweep plus one comparison → erosion and dilation as neighbourhood
// min and max (the same sweep again, different reduction) → opening and
// closing, where the order is the whole answer → a payoff that cleans a
// speckled mask and counts what survives.
//
// Kernel-authoring rules (contract): no closures inside kernel functions, only
// numbers / nested number arrays / ImageData as inputs, this.thread.* for
// indexing, this.constants.* for compile-time values (legal as loop bounds),
// image convention image[y][x] = [r, g, b, a] with channels 0–1. Every task
// passes in CPU mode.
//
// BORDER CONVENTION. Every neighbourhood sweep in this module CLAMPS: a sample
// that falls off the frame reuses the nearest in-bounds cell, exactly as
// Convolution & Filters' box blur does. Treating out-of-bounds as background is
// equally defensible and gives a DIFFERENT answer at the frame — an eroded
// shape lying flush against the edge survives under one rule and is eaten under
// the other. The masks below deliberately put shapes against the frame so the
// tests can tell the two apart, and task 4 says so in prose.
//
// NUMERIC MARGINS. Thresholding is a comparison, so every value the tasks
// threshold is kept clear of its decision boundary by a wide margin (≥ 0.007 in
// luminance, ~10⁴× the float32 noise of these kernels). That is why the scenes
// below step the lighting in bands and quantize the luminances onto a 0.02 grid
// with the threshold sitting 0.01 off it: a mask that flips a pixel between the
// CPU and GL backends would make every full-grid test flaky.

import { ARRAY_LAYOUT } from '../layoutNote.js';
import { plainToImageData, quantizePixel } from '../../engine/utils.js';

const SIZE = 128;
const LAST = SIZE - 1;

const LUM = [0.299, 0.587, 0.114];

function luminanceOf(pixel) {
  return LUM[0] * pixel[0] + LUM[1] * pixel[1] + LUM[2] * pixel[2];
}

// ---- scene 1: bright marks on unevenly lit ground -------------------------
//
// Background luminance falls in steps of STEP_L every BAND pixels along the
// diagonal, from TOP_L at the lit corner; marks sit MARK_L above the ground
// they lie on. Every luminance therefore lands on a 0.02 grid, and GLOBAL_T is
// 0.01 away from all of them.

const BAND = 16; // pixels of (x + y) per lighting step
const TOP_L = 0.7; // luminance of the brightest background band
const STEP_L = 0.04; // luminance lost per step
const MARK_L = 0.14; // how much brighter a mark is than the ground beneath it
const GLOBAL_T = 0.49; // the hand-picked global threshold of task 1
const WINDOW = 9; // adaptive window edge
const RADIUS = 4; // (WINDOW - 1) / 2
const AREA = WINDOW * WINDOW;
const BIAS = 0.03; // how much brighter than its neighbourhood a pixel must be

// A warm colour cast, so that "the red channel" and "the luminance" are two
// different pictures: a pixel is (L / CAST_W) * CAST, whose weighted luminance
// is exactly L while its red channel is L / CAST_W. Thresholding red instead of
// luminance therefore moves the boundary by a couple of lighting bands — enough
// cells for a probe to be sure of what it saw.
const CAST = [1, 0.9, 0.55];
const CAST_W = LUM[0] * CAST[0] + LUM[1] * CAST[1] + LUM[2] * CAST[2];

// 64 marks, 5×5, one per 16×16 cell, jittered inside its cell.
function markCells(utils, flip) {
  const rand = utils.seededRandom(flip ? 9152 : 5117);
  const cells = [];
  for (let gy = 0; gy < 8; gy++) {
    for (let gx = 0; gx < 8; gx++) {
      const ox = 4 + Math.floor(rand() * 4);
      const oy = 4 + Math.floor(rand() * 4);
      cells.push({ x: gx * 16 + ox, y: gy * 16 + oy, w: 5, h: 5 });
    }
  }
  return cells;
}

// The scene's ideal luminance, [y][x] plain numbers. `flip` runs the lighting
// down the other diagonal and moves the marks — the private tests' second scene.
function litLuminance(utils, flip) {
  const gray = new Array(SIZE);
  for (let y = 0; y < SIZE; y++) {
    const row = new Array(SIZE);
    for (let x = 0; x < SIZE; x++) {
      const along = flip ? x + (LAST - y) : x + y;
      row[x] = TOP_L - STEP_L * Math.floor(along / BAND);
    }
    gray[y] = row;
  }
  for (const mark of markCells(utils, flip)) {
    for (let y = mark.y; y < mark.y + mark.h; y++) {
      for (let x = mark.x; x < mark.x + mark.w; x++) gray[y][x] += MARK_L;
    }
  }
  return gray;
}

// The same scene as an ImageData — the one image shape every gpu.js backend
// reads on the GPU. Channels are quantized to 8 bits, so a luminance computed
// host-side from .plain is exactly what the kernel sees.
function litImage(utils, flip) {
  const gray = litLuminance(utils, flip);
  const plain = new Array(SIZE);
  for (let y = 0; y < SIZE; y++) {
    const row = new Array(SIZE);
    for (let x = 0; x < SIZE; x++) {
      const scale = gray[y][x] / CAST_W;
      row[x] = quantizePixel([scale * CAST[0], scale * CAST[1], scale * CAST[2], 1]);
    }
    plain[y] = row;
  }
  return plainToImageData(plain);
}

// ---- scene 2: an evenly lit scene, for the histogram ----------------------
//
// Dark ground, bright rectangles, and enough noise on both that the two humps
// of the tone histogram overlap in the middle. An empty valley would make every
// threshold in it score identically and Otsu's answer arbitrary.

const BINS = 256;

function evenLuminance(utils, seed, ground, marks, amp) {
  const rand = utils.seededRandom(seed);
  const rects = shapeRects(rand);
  const gray = new Array(SIZE);
  for (let y = 0; y < SIZE; y++) gray[y] = new Array(SIZE).fill(ground);
  for (const r of rects) {
    for (let y = r.y; y < r.y + r.h; y++) {
      for (let x = r.x; x < r.x + r.w; x++) gray[y][x] = marks;
    }
  }
  for (let y = 0; y < SIZE; y++) {
    for (let x = 0; x < SIZE; x++) {
      // three uniforms added: bounded, roughly bell-shaped, fully deterministic
      const noise = (rand() + rand() + rand() - 1.5) * amp;
      gray[y][x] = Math.min(1, Math.max(0, gray[y][x] + noise));
    }
  }
  return gray;
}

// The 256-bin tone histogram of a luminance map — counts, summing to 16,384.
// This is the histogram Histograms & Binning ends on; it is handed over here
// rather than rebuilt.
function toneHistogram(gray) {
  const bins = new Array(BINS).fill(0);
  for (const row of gray) {
    for (const value of row) bins[Math.min(BINS - 1, Math.floor(value * BINS))] += 1;
  }
  return bins;
}

// ---- binary masks ---------------------------------------------------------
//
// A 4×4 grid of solid rectangles, each parked well inside its own 32-pixel
// cell, so neighbours are at least 8 pixels apart: a 3×3 dilation can never
// merge two of them, and an erosion followed by a dilation restores each one
// exactly. Every dimension is ≥ 10, comfortably wider than the 3×3 element.

function shapeRects(rand) {
  const rects = [];
  for (let gy = 0; gy < 4; gy++) {
    for (let gx = 0; gx < 4; gx++) {
      const w = 10 + Math.floor(rand() * 9);
      const h = 10 + Math.floor(rand() * 9);
      rects.push({
        x: gx * 32 + 4 + Math.floor(rand() * (25 - w)),
        y: gy * 32 + 4 + Math.floor(rand() * (25 - h)),
        w,
        h,
      });
    }
  }
  return rects;
}

function blankMask() {
  const mask = new Array(SIZE);
  for (let y = 0; y < SIZE; y++) mask[y] = new Array(SIZE).fill(0);
  return mask;
}

function paintRect(mask, x, y, w, h, value) {
  for (let j = y; j < y + h; j++) {
    for (let i = x; i < x + w; i++) mask[j][i] = value;
  }
}

// Is the rectangle, grown by `pad` on every side, entirely background?
function areaIsClear(mask, x, y, w, h, pad) {
  for (let j = y - pad; j < y + h + pad; j++) {
    for (let i = x - pad; i < x + w + pad; i++) {
      if (j < 0 || i < 0 || j >= SIZE || i >= SIZE) continue;
      if (mask[j][i] !== 0) return false;
    }
  }
  return true;
}

// The mask tasks 4 and 5 work on: sixteen rectangles (four of them run to the
// frame, so clamping and background-padding disagree), one pinhole punched deep
// inside each big rectangle, 3×3 clumps that survive one opening but not two,
// and 1×1 / 2×2 specks that survive neither.
function noisyMask(utils, seed) {
  const rand = utils.seededRandom(seed);
  const rects = shapeRects(rand);
  rects[4] = { ...rects[4], w: rects[4].w + rects[4].x, x: 0 }; // flush left
  rects[2] = { ...rects[2], h: rects[2].h + rects[2].y, y: 0 }; // flush top
  rects[11] = { ...rects[11], w: SIZE - rects[11].x }; // flush right
  rects[13] = { ...rects[13], h: SIZE - rects[13].y }; // flush bottom

  const mask = blankMask();
  for (const r of rects) paintRect(mask, r.x, r.y, r.w, r.h, 1);

  // Pinholes, kept 5 pixels clear of their rectangle's edge so that even a
  // two-pass opening puts the shape back exactly as it found it.
  for (const r of rects) {
    if (r.w < 12 || r.h < 12) continue;
    const hx = r.x + Math.round(0.4 * (r.w - 1));
    const hy = r.y + Math.round(0.45 * (r.h - 1));
    if (hx < r.x + 5 || hx > r.x + r.w - 6 || hy < r.y + 5 || hy > r.y + r.h - 6) continue;
    let solid = true;
    for (let j = hy - 5; j <= hy + 5; j++) {
      for (let i = hx - 5; i <= hx + 5; i++) if (mask[j][i] !== 1) solid = false;
    }
    if (solid) mask[hy][hx] = 0;
  }

  for (let gy = 0; gy < 16; gy++) {
    for (let gx = 0; gx < 16; gx++) {
      const roll = rand();
      const x = Math.min(121, gx * 8 + 2 + Math.floor(rand() * 4));
      const y = Math.min(121, gy * 8 + 2 + Math.floor(rand() * 4));
      if (roll < 0.16) {
        if (areaIsClear(mask, x, y, 3, 3, 3)) paintRect(mask, x, y, 3, 3, 1);
      } else if (roll < 0.42) {
        const s = roll < 0.3 ? 2 : 1;
        if (areaIsClear(mask, x, y, s, s, 3)) paintRect(mask, x, y, s, s, 1);
      }
    }
  }
  return mask;
}

// The mask task 6 counts: the same sixteen rectangles, none touching the frame,
// no pinholes, and a heavier confetti of 1×1 / 2×2 specks. One opening leaves
// exactly the rectangles.
function sceneMask(utils, seed) {
  const rand = utils.seededRandom(seed);
  const rects = shapeRects(rand);
  const mask = blankMask();
  for (const r of rects) paintRect(mask, r.x, r.y, r.w, r.h, 1);
  for (let gy = 0; gy < 16; gy++) {
    for (let gx = 0; gx < 16; gx++) {
      const roll = rand();
      const x = Math.min(121, gx * 8 + 2 + Math.floor(rand() * 4));
      const y = Math.min(121, gy * 8 + 2 + Math.floor(rand() * 4));
      if (roll < 0.45) {
        const s = roll < 0.18 ? 2 : 1;
        if (areaIsClear(mask, x, y, s, s, 3)) paintRect(mask, x, y, s, s, 1);
      }
    }
  }
  return mask;
}

// ---- CPU references -------------------------------------------------------

function globalMaskRef(image, channel) {
  const plain = image.plain;
  const out = new Array(SIZE);
  for (let y = 0; y < SIZE; y++) {
    const row = new Array(SIZE);
    for (let x = 0; x < SIZE; x++) {
      const p = plain[y][x];
      const value = channel === undefined ? luminanceOf(p) : p[channel];
      row[x] = value > GLOBAL_T ? 1 : 0;
    }
    out[y] = row;
  }
  return out;
}

function invertMask(mask) {
  return mask.map(row => row.map(v => 1 - v));
}

// Otsu's between-class variance for every candidate threshold, plus the class
// mean gap each one produced (a probe needs it to describe the un-squared
// variant). Class 0 is bins 0…t INCLUSIVE.
function otsuRef(bins) {
  const score = new Array(BINS).fill(0);
  const gap = new Array(BINS).fill(0);
  for (let t = 0; t < BINS; t++) {
    let w0 = 0;
    let s0 = 0;
    let w1 = 0;
    let s1 = 0;
    for (let i = 0; i < BINS; i++) {
      if (i <= t) {
        w0 += bins[i];
        s0 += i * bins[i];
      } else {
        w1 += bins[i];
        s1 += i * bins[i];
      }
    }
    if (w0 === 0 || w1 === 0) continue;
    const total = w0 + w1;
    const d = s0 / w0 - s1 / w1;
    gap[t] = d;
    score[t] = (w0 / total) * (w1 / total) * d * d;
  }
  return { score, gap };
}

function adaptiveMaskRef(gray) {
  const out = new Array(SIZE);
  for (let y = 0; y < SIZE; y++) {
    const row = new Array(SIZE);
    for (let x = 0; x < SIZE; x++) {
      let sum = 0;
      for (let dy = 0; dy < WINDOW; dy++) {
        for (let dx = 0; dx < WINDOW; dx++) {
          const sy = Math.min(LAST, Math.max(0, y + dy - RADIUS));
          const sx = Math.min(LAST, Math.max(0, x + dx - RADIUS));
          sum += gray[sy][sx];
        }
      }
      row[x] = gray[y][x] > sum / AREA + BIAS ? 1 : 0;
    }
    out[y] = row;
  }
  return out;
}

// The 3×3 sweep, once per reduction operator and once per border rule.
// pick: 'min' (erode) or 'max' (dilate); border: 'clamp' (this module's rule)
// or 'zero' (out-of-bounds counts as background).
function sweepRef(mask, pick, border) {
  const out = new Array(SIZE);
  for (let y = 0; y < SIZE; y++) {
    const row = new Array(SIZE);
    for (let x = 0; x < SIZE; x++) {
      let acc = pick === 'min' ? 1 : 0;
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          let sy = y + dy;
          let sx = x + dx;
          let value;
          if (border === 'zero' && (sy < 0 || sx < 0 || sy > LAST || sx > LAST)) {
            value = 0;
          } else {
            sy = Math.min(LAST, Math.max(0, sy));
            sx = Math.min(LAST, Math.max(0, sx));
            value = mask[sy][sx];
          }
          acc = pick === 'min' ? Math.min(acc, value) : Math.max(acc, value);
        }
      }
      row[x] = acc;
    }
    out[y] = row;
  }
  return out;
}

const erodeRef = mask => sweepRef(mask, 'min', 'clamp');
const dilateRef = mask => sweepRef(mask, 'max', 'clamp');
const openRef = mask => dilateRef(erodeRef(mask));
const closeRef = mask => erodeRef(dilateRef(mask));
const openTwiceRef = mask => dilateRef(dilateRef(erodeRef(erodeRef(mask))));

// 1 wherever `before` is foreground and `after` is not.
function removedRef(before, after) {
  const out = new Array(SIZE);
  for (let y = 0; y < SIZE; y++) {
    const row = new Array(SIZE);
    for (let x = 0; x < SIZE; x++) row[x] = before[y][x] > after[y][x] ? 1 : 0;
    out[y] = row;
  }
  return out;
}

// 1 at every foreground pixel whose neighbour above and neighbour to the left
// are both background — one per axis-aligned rectangle.
function cornerRef(mask) {
  const out = new Array(SIZE);
  for (let y = 0; y < SIZE; y++) {
    const row = new Array(SIZE);
    for (let x = 0; x < SIZE; x++) {
      const up = Math.max(0, y - 1);
      const left = Math.max(0, x - 1);
      row[x] = mask[y][x] > 0.5 && mask[up][x] < 0.5 && mask[y][left] < 0.5 ? 1 : 0;
    }
    out[y] = row;
  }
  return out;
}

function countOn(grid) {
  let n = 0;
  for (let y = 0; y < grid.length; y++) {
    for (let x = 0; x < grid[y].length; x++) n += grid[y][x] > 0.5 ? 1 : 0;
  }
  return n;
}

// ---- near-miss diagnosis --------------------------------------------------
//
// The course's rule: when a failing value is exactly what one specific mistake
// would produce, name that mistake instead of reporting two numbers. A probe
// pairs such a candidate with its sentence, and it may only speak when the
// observation matches it AND the correct answer does not. Two probes that
// disagree cancel each other. A confident wrong diagnosis is worse than none.

function diagnose(got, expected, eps, probes) {
  const hits = probes
    .filter(p => Math.abs(got - p[0]) <= eps && Math.abs(expected - p[0]) > eps)
    .map(p => p[1]);
  return hits.length && hits.every(m => m === hits[0]) ? hits[0] : null;
}

// Otsu's scores range over ten orders of magnitude between the empty cuts and
// the peak, and one of its likely mistakes (raw counts instead of fractions)
// scales the answer by the pixel count squared. A fixed absolute tolerance
// cannot recognise a candidate 10⁸ times larger than the right answer, so this
// form compares RELATIVELY: within a thousandth of the candidate's own size.
function diagnoseRel(got, expected, rel, probes) {
  const near = (a, b) => Math.abs(a - b) <= rel * Math.max(1, Math.abs(b));
  const hits = probes.filter(p => near(got, p[0]) && !near(expected, p[0])).map(p => p[1]);
  return hits.length && hits.every(m => m === hits[0]) ? hits[0] : null;
}

// A single cell of a 0/1 mask carries almost no information — half the wrong
// answers agree with the right one at any given pixel. So a mask probe has to
// predict EVERY cell of the grid, and disagree with the correct mask somewhere,
// before it may speak.
function gridMatches(got, candidate) {
  if (!got || got.length !== SIZE) return false;
  for (let y = 0; y < SIZE; y++) {
    const row = got[y];
    if (!row || row.length !== SIZE) return false;
    for (let x = 0; x < SIZE; x++) {
      if (!(Math.abs(row[x] - candidate[y][x]) <= 0.25)) return false;
    }
  }
  return true;
}

function gridDiffers(a, b) {
  for (let y = 0; y < SIZE; y++) {
    for (let x = 0; x < SIZE; x++) {
      if (Math.abs(a[y][x] - b[y][x]) > 0.25) return true;
    }
  }
  return false;
}

function diagnoseMask(got, expected, probes) {
  const hits = probes
    .filter(([candidate]) => gridDiffers(expected, candidate) && gridMatches(got, candidate))
    .map(p => p[1]);
  return hits.length && hits.every(m => m === hits[0]) ? hits[0] : null;
}

// First disagreeing cell, for the fallback message. null when the grid matches.
function firstMismatch(got, expected) {
  if (!got || got.length !== SIZE) return 'shape';
  for (let y = 0; y < SIZE; y++) {
    const row = got[y];
    if (!row || row.length !== SIZE) return 'shape';
    for (let x = 0; x < SIZE; x++) {
      if (!(Math.abs(row[x] - expected[y][x]) <= 0.25)) return [y, x, row[x]];
    }
  }
  return null;
}

// The one assertion every mask task ends with: compare a whole grid, name the
// mistake if a probe recognises it, otherwise point at the first bad cell.
function assertMask(ctx, got, expected, probes, what) {
  const bad = firstMismatch(got, expected);
  if (!bad) return;
  const hint = diagnoseMask(got, expected, probes || []);
  if (bad === 'shape') {
    ctx.assert(false, hint || `${what}: expected a ${SIZE}×${SIZE} grid of rows`);
    return;
  }
  const [y, x, value] = bad;
  ctx.assert(
    false,
    hint || `${what}: cell [${y}][${x}] should be ${expected[y][x]}, got ${value}`
  );
}

// Every value in a mask has to be exactly 1 or 0. A cell holding the quantity
// that was supposed to be COMPARED is the giveaway mistake here.
function assertBinary(ctx, got, what, describeValue) {
  for (let y = 0; y < SIZE; y += 7) {
    for (let x = 0; x < SIZE; x += 5) {
      const v = got[y][x];
      if (Math.abs(v) <= 1e-4 || Math.abs(v - 1) <= 1e-4) continue;
      ctx.assert(
        false,
        (describeValue && describeValue(v, y, x)) ||
          `${what}: cell [${y}][${x}] holds ${v} — a mask is exactly 1 or 0, so return one of them`
      );
    }
  }
}

// Task 5 and 6 report through the console, so their tests read it back. The
// label is fixed by the starter, and the number has to be the whole rest of it.
function loggedCount(ctx, label) {
  const re = new RegExp(`^${label.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}\\s*(-?[0-9]+)\\s*$`);
  for (const line of ctx.logs) {
    if (line.type !== 'log' || !line.text) continue;
    const match = re.exec(String(line.text).trim());
    if (match) return Number(match[1]);
  }
  return null;
}

// ---- task-specific probes -------------------------------------------------

function globalProbes(image, expected) {
  return [
    [globalMaskRef(image, 0), 'that mask thresholds the RED channel, not the luminance — a warm image makes those two different pictures. Weight the channels 0.299 R + 0.587 G + 0.114 B first'],
    [invertMask(expected), 'that is exactly the mask inverted — the comparison is the wrong way round. Foreground is BRIGHTER than the threshold'],
  ];
}

function adaptiveProbes(gray, image, expected) {
  const shifted = new Array(SIZE);
  const noBias = new Array(SIZE);
  const belowBias = new Array(SIZE);
  for (let y = 0; y < SIZE; y++) {
    shifted[y] = new Array(SIZE);
    noBias[y] = new Array(SIZE);
    belowBias[y] = new Array(SIZE);
    for (let x = 0; x < SIZE; x++) {
      let sum = 0;
      let corner = 0;
      for (let dy = 0; dy < WINDOW; dy++) {
        for (let dx = 0; dx < WINDOW; dx++) {
          const cy = Math.min(LAST, Math.max(0, y + dy - RADIUS));
          const cx = Math.min(LAST, Math.max(0, x + dx - RADIUS));
          sum += gray[cy][cx];
          // the same window with the centring subtraction forgotten
          corner += gray[Math.min(LAST, y + dy)][Math.min(LAST, x + dx)];
        }
      }
      const mean = sum / AREA;
      shifted[y][x] = gray[y][x] > corner / AREA + BIAS ? 1 : 0;
      noBias[y][x] = gray[y][x] > mean ? 1 : 0;
      belowBias[y][x] = gray[y][x] > mean - BIAS ? 1 : 0;
    }
  }
  return [
    [globalMaskRef(image), 'that is task 1\'s global mask — every pixel is still being compared against one fixed number instead of against its own neighbourhood'],
    [shifted, 'the window is not centred on this thread — sample this.thread.y + dy - this.constants.radius, so that dy = 0 reaches BACK by the radius'],
    [noBias, 'the bias is missing: with a bare <code>gray > mean</code> comparison flat ground splits down the middle on nothing but rounding. Compare against mean + this.constants.c'],
    [belowBias, 'the bias is being subtracted, not added — mean - c lets ground through; a pixel has to be c BRIGHTER than its surroundings'],
    [expected.map(row => row.map(v => 1 - v)), 'that is exactly the mask inverted — the comparison is the wrong way round'],
  ];
}

function erodeProbes(mask) {
  return [
    [dilateRef(mask), 'erosion and dilation are swapped: this kernel GREW the mask where it should have shrunk it. Erosion keeps the smallest sample in the window — Math.min, starting from 1'],
    [sweepRef(mask, 'min', 'zero'), 'that erosion ate the frame as well: an out-of-bounds sample was treated as background, so every shape lying against the edge lost its border row. This module CLAMPS — an off-frame sample reuses the nearest in-bounds cell'],
    [mask, 'that is the mask unchanged — the nine reads never reached the return value'],
  ];
}

function dilateProbes(mask) {
  return [
    [erodeRef(mask), 'erosion and dilation are swapped: this kernel SHRANK the mask where it should have grown it. Dilation keeps the largest sample in the window — Math.max, starting from 0'],
    [mask, 'that is the mask unchanged — the nine reads never reached the return value'],
  ];
}

function cornerProbes(mask) {
  const wrongSide = new Array(SIZE);
  const noSelf = new Array(SIZE);
  for (let y = 0; y < SIZE; y++) {
    wrongSide[y] = new Array(SIZE);
    noSelf[y] = new Array(SIZE);
    for (let x = 0; x < SIZE; x++) {
      const down = Math.min(LAST, y + 1);
      const right = Math.min(LAST, x + 1);
      const up = Math.max(0, y - 1);
      const left = Math.max(0, x - 1);
      wrongSide[y][x] = mask[y][x] > 0.5 && mask[down][x] < 0.5 && mask[y][right] < 0.5 ? 1 : 0;
      noSelf[y][x] = mask[up][x] < 0.5 && mask[y][left] < 0.5 ? 1 : 0;
    }
  }
  return [
    [wrongSide, 'those are BOTTOM-RIGHT corners — the neighbours below and to the right. Both give one hit per rectangle, but the pixel they land on is the far corner; this task asks for the top-left one: mask[y - 1][x] and mask[y][x - 1]'],
    [noSelf, 'background pixels are being counted too — a corner has to be foreground itself before its neighbours matter'],
  ];
}

// ---- shared code fragments ------------------------------------------------

// The two morphology kernels, complete, for the tasks that build on them.
const GIVEN_SWEEPS = `// Given: the two sweeps from the previous task, unchanged.
const erode = gpu.createKernel(function (mask) {
  let lo = 1;
  for (let dy = 0; dy < 3; dy++) {
    for (let dx = 0; dx < 3; dx++) {
      let sy = this.thread.y + dy - 1;
      let sx = this.thread.x + dx - 1;
      if (sy < 0) sy = 0;
      if (sy > this.constants.last) sy = this.constants.last;
      if (sx < 0) sx = 0;
      if (sx > this.constants.last) sx = this.constants.last;
      lo = Math.min(lo, mask[sy][sx]);
    }
  }
  return lo;
}, { output: [128, 128], constants: { last: 127 } });

const dilate = gpu.createKernel(function (mask) {
  let hi = 0;
  for (let dy = 0; dy < 3; dy++) {
    for (let dx = 0; dx < 3; dx++) {
      let sy = this.thread.y + dy - 1;
      let sx = this.thread.x + dx - 1;
      if (sy < 0) sy = 0;
      if (sy > this.constants.last) sy = this.constants.last;
      if (sx < 0) sx = 0;
      if (sx > this.constants.last) sx = this.constants.last;
      hi = Math.max(hi, mask[sy][sx]);
    }
  }
  return hi;
}, { output: [128, 128], constants: { last: 127 } });`;

const COUNT_HELPER = `// Plain JavaScript: how many cells of a mask are foreground.
function count(grid) {
  let n = 0;
  for (let y = 0; y < 128; y++) {
    for (let x = 0; x < 128; x++) n += grid[y][x];
  }
  return n;
}`;

const ASCII_DUMP = `// A look at the result: every 4th pixel, '#' where the mask says foreground.
for (let y = 0; y < 128; y += 4) {
  let line = '';
  for (let x = 0; x < 128; x += 4) line += mask[y][x] > 0.5 ? '#' : '.';
  console.log(line);
}`;

export default {
  uuid: '670eaafa-4b13-4e77-88b3-f2904105c615',
  version: 1,
  slug: 'thresholding-and-morphology',
  title: 'Thresholding & Morphology',
  blurb: 'Turning grey pixels into a clean binary mask: global and adaptive thresholds, then erosion and dilation as a neighbourhood min and max.',
  tasks: [
    // ------------------------------------------------------------------ 1
    {
      slug: 'global-threshold',
      title: 'One Number for the Whole Image',
      intro: `<p>Everything downstream — counting, measuring, tracking — wants a
        <strong>binary mask</strong>: one bit per pixel, foreground or background. The cheapest
        way to make one is a <strong>threshold</strong>. Pick a number; call every pixel
        brighter than it foreground.</p>
        <p>Per pixel, no neighbours, no order, nothing shared: the friendliest shape a kernel
        can have, and exactly the pure map "Thinking in Parallel" calls the easy case. One
        thread, one pixel, one comparison.</p>
        <p>It is also where real images bite back. <code>photo</code> is lit unevenly — bright
        at the top-left corner, fading away to the bottom-right — with small bright marks
        scattered over the whole frame. Run the starter and read the ASCII dump it prints: one
        corner comes back solid, the opposite corner comes back empty, and only a diagonal band
        across the middle finds the marks at all.</p>
        ${ARRAY_LAYOUT}`,
      goal: `<strong>Goal:</strong> return a 128×128 mask — <code>1</code> where this pixel's
        <em>luminance</em> is above <code>this.constants.t</code>, <code>0</code> everywhere
        else.`,
      requirements: [
        'Read this thread\'s pixel: <code>photo[this.thread.y][this.thread.x]</code>',
        'Threshold the <strong>luminance</strong> <code>0.299r + 0.587g + 0.114b</code>, not a single channel',
        'Return exactly <code>1</code> or <code>0</code> — brighter than <code>this.constants.t</code> is foreground',
      ],
      hints: [
        {
          title: 'Hint 1 — luminance first, comparison second',
          body: `<p>Two steps, both of which you have written before: reduce the pixel to one
            number, then compare that number. The image is warm-toned, so red and luminance are
            genuinely different pictures — thresholding <code>pixel[0]</code> gives a mask that
            is wrong by a couple of lighting bands.</p>`,
        },
        {
          title: 'Hint 2 — returning a bit',
          body: `<p>A kernel returns a number, so the "bit" is the number <code>1</code> or the
            number <code>0</code>:</p>
<pre><code>if (lum &gt; this.constants.t) return 1;
return 0;</code></pre>`,
        },
        {
          title: 'Hint 3 — the whole body',
          body: `<pre><code>const p = photo[this.thread.y][this.thread.x];
const lum = 0.299 * p[0] + 0.587 * p[1] + 0.114 * p[2];
if (lum &gt; this.constants.t) return 1;
return 0;</code></pre>`,
        },
      ],
      transfer: `A threshold is one <code>step()</code> in GLSL/WGSL, one predicated store in
        CUDA, and a single fused op in every imaging library — it is so cheap that camera ISPs
        do it in silicon. Which is exactly why the interesting question is never how to compare,
        but what to compare against.`,
      starterCode: `// One thread, one pixel, one comparison. No neighbours needed.
const gpu = new GPU({ mode });

const threshold = gpu.createKernel(function (photo) {
  // TODO: reduce this thread's pixel to its luminance
  // (0.299 R + 0.587 G + 0.114 B), then return 1 when that is above
  // this.constants.t and 0 when it is not.
  return 0;
}, {
  output: [128, 128],
  constants: { t: 0.49 },
});

const mask = await threshold(photo);

${ASCII_DUMP}

// The same story in numbers: two opposite corners of the frame.
let lit = 0;
let dark = 0;
for (let y = 0; y < 32; y++) {
  for (let x = 0; x < 32; x++) {
    lit += mask[y][x];
    dark += mask[y + 96][x + 96];
  }
}
console.log('top-left 32x32 foreground:', lit, 'of 1024');
console.log('bottom-right 32x32 foreground:', dark, 'of 1024');
`,
      solutionCode: `// One thread, one pixel, one comparison. No neighbours needed.
const gpu = new GPU({ mode });

const threshold = gpu.createKernel(function (photo) {
  const p = photo[this.thread.y][this.thread.x];
  const lum = 0.299 * p[0] + 0.587 * p[1] + 0.114 * p[2];
  if (lum > this.constants.t) return 1;
  return 0;
}, {
  output: [128, 128],
  constants: { t: 0.49 },
});

const mask = await threshold(photo);

${ASCII_DUMP}

// The same story in numbers: two opposite corners of the frame.
let lit = 0;
let dark = 0;
for (let y = 0; y < 32; y++) {
  for (let x = 0; x < 32; x++) {
    lit += mask[y][x];
    dark += mask[y + 96][x + 96];
  }
}
console.log('top-left 32x32 foreground:', lit, 'of 1024');
console.log('bottom-right 32x32 foreground:', dark, 'of 1024');
`,
      inputs: utils => ({ photo: litImage(utils, false) }),
      publicTests: [
        {
          name: 'a <code>128×128</code> grid holding nothing but <code>1</code> and <code>0</code>',
          run: async ctx => {
            ctx.assert(ctx.kernels.length >= 1, 'no kernel was created — call gpu.createKernel()');
            const image = litImage(ctx.utils, false);
            const out = await ctx.kernel(image);
            ctx.assert(out && out.length === SIZE, `expected ${SIZE} rows, got ${out && out.length}`);
            ctx.assert(out[0] && out[0].length === SIZE, `each row should hold ${SIZE} values`);
            assertBinary(ctx, out, 'the mask', (v, y, x) => {
              const p = image.plain[y][x];
              return Math.abs(v - luminanceOf(p)) <= 2e-3
                ? `cell [${y}][${x}] holds ${v.toFixed(3)}, which is the pixel's luminance — the ` +
                  'comparison never happened. A mask holds the ANSWER to the comparison: 1 or 0'
                : null;
            });
          },
        },
        {
          name: 'foreground is exactly <code>luminance &gt; 0.49</code>',
          run: async ctx => {
            const image = litImage(ctx.utils, false);
            const out = await ctx.kernel(image);
            const ref = globalMaskRef(image);
            assertMask(ctx, out, ref, globalProbes(image, ref), 'the mask');
          },
        },
        {
          name: 'one number, two answers: the lit corner saturates and the dark corner goes black',
          run: async ctx => {
            // Not a check of your arithmetic — a check that you can see the
            // failure. The same ink is present in both corners.
            const image = litImage(ctx.utils, false);
            const out = await ctx.kernel(image);
            let lit = 0;
            let dark = 0;
            for (let y = 0; y < 32; y++) {
              for (let x = 0; x < 32; x++) {
                lit += out[y][x] > 0.5 ? 1 : 0;
                dark += out[y + 96][x + 96] > 0.5 ? 1 : 0;
              }
            }
            ctx.assert(
              lit === 1024,
              `the top-left 32×32 corner should come back entirely foreground (1024 of 1024) — ` +
                `got ${lit}. Under this lamp the bare ground there is already brighter than 0.49`
            );
            ctx.assert(
              dark === 0,
              `the bottom-right 32×32 corner should come back entirely background (0 of 1024) — ` +
                `got ${dark}. The marks there are real, and this threshold cannot see any of them`
            );
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            // A second scene, lit down the other diagonal: nothing about the
            // first one's geometry can be hard-coded.
            const image = litImage(ctx.utils, true);
            const out = await ctx.kernel(image);
            const ref = globalMaskRef(image);
            assertMask(ctx, out, ref, globalProbes(image, ref), 'the mask');
          },
        },
      ],
    },

    // ------------------------------------------------------------------ 2
    {
      slug: 'otsu-threshold',
      title: 'Let the Histogram Pick the Number',
      intro: `<p>Picking <code>0.49</code> by hand was a cheat. <strong>Otsu's method</strong>
        reads the number off the data instead: try every cut, keep the one whose two sides are
        furthest apart.</p>
        <p>"Furthest apart" has a precise meaning — the <strong>between-class variance</strong>.
        Cut the tone histogram at bin <code>t</code>; let <code>p0</code> and <code>p1</code> be
        the fraction of pixels on each side and <code>mu0</code>, <code>mu1</code> their mean bin
        numbers. Then</p>
<pre><code>score(t) = p0 * p1 * (mu0 - mu1) * (mu0 - mu1)</code></pre>
        <p>and the winner is the <code>t</code> that maximises it. <code>tones</code> is the
        256-bin tone histogram of an <em>evenly</em> lit scene — the same one-thread-per-bin
        build Histograms &amp; Binning finishes on, handed over here rather than counted again.</p>
        <p>The parallel shape is the good part: 256 candidate thresholds, 256 threads, each
        sweeping all 256 bins for itself. 65,536 reads that happen at once, and then a single
        tiny argmax over the answers. (Task 3 is the reminder that Otsu picks the best possible
        single number, and that on a badly lit frame the best possible single number is still
        not good enough.)</p>`,
      goal: `<strong>Goal:</strong> one thread per candidate threshold — return the between-class
        variance of the cut at <code>t = this.thread.x</code>, with class 0 being bins
        <code>0…t</code> <strong>inclusive</strong>.`,
      requirements: [
        'Output <code>[256]</code>: one thread per candidate threshold, <code>t = this.thread.x</code>',
        'Sweep all <code>this.constants.bins</code> bins, accumulating each class\'s count and its count-weighted bin sum',
        'Class 0 is bins <code>0…t</code> <strong>inclusive</strong> — a bin equal to <code>t</code> belongs below the cut',
        'Return <code>p0 * p1 * (mu0 - mu1)²</code>, or <code>0</code> when either class is empty',
      ],
      hints: [
        {
          title: 'Hint 1 — one thread, one candidate',
          body: `<p>Thread <code>t</code> owns exactly one question: <em>what if I cut here?</em>
            It reads the whole histogram to answer it, which is fine — 256 reads is nothing, and
            all 256 threads are doing it at the same time.</p>`,
        },
        {
          title: 'Hint 2 — four running totals',
          body: `<p>One pass, four accumulators: the count and the bin-weighted sum on each side.</p>
<pre><code>for (let i = 0; i &lt; this.constants.bins; i++) {
  if (i &lt;= t) {
    w0 += tones[i];
    s0 += i * tones[i];
  } else {
    w1 += tones[i];
    s1 += i * tones[i];
  }
}</code></pre>
<p>The class means are then <code>s0 / w0</code> and <code>s1 / w1</code>.</p>`,
        },
        {
          title: 'Hint 3 — the empty class',
          body: `<p>At <code>t = 0</code> class 0 may hold no pixels at all, and
            <code>s0 / w0</code> is then <code>0 / 0</code> — a NaN that poisons the whole
            comparison. Guard it: a cut with an empty side separates nothing, so its score is
            <code>0</code>.</p>
<pre><code>if (w0 === 0 || w1 === 0) return 0;</code></pre>`,
        },
      ],
      transfer: `This is the classic "try every candidate in parallel, reduce afterwards" shape:
        one CUDA thread per hypothesis, one WGSL invocation per bin, one Metal thread per
        candidate. OpenCV's <code>THRESH_OTSU</code> runs the same arithmetic serially over 256
        bins because on a CPU that is already free — on a GPU it is free <em>and</em> it fuses
        into whatever pass produced the histogram.`,
      starterCode: `// 256 candidate thresholds, 256 threads, one histogram sweep each.
const gpu = new GPU({ mode });

const between = gpu.createKernel(function (tones) {
  const t = this.thread.x;
  let w0 = 0;
  let s0 = 0;
  let w1 = 0;
  let s1 = 0;
  // TODO: sweep all this.constants.bins bins. Bins 0…t (inclusive) go into
  // w0/s0, the rest into w1/s1. Then return p0 * p1 * (mu0 - mu1)²,
  // and 0 if either class turned out to be empty.
  return 0;
}, {
  output: [256],
  constants: { bins: 256 },
});

const scores = await between(tones);

// The argmax is one tiny reduction — plain JavaScript is the right tool here.
let best = 0;
for (let t = 1; t < 256; t++) {
  if (scores[t] > scores[best]) best = t;
}
console.log('Otsu threshold: bin', best);
console.log('as a grey level:', (best / 255).toFixed(3));
`,
      solutionCode: `// 256 candidate thresholds, 256 threads, one histogram sweep each.
const gpu = new GPU({ mode });

const between = gpu.createKernel(function (tones) {
  const t = this.thread.x;
  let w0 = 0;
  let s0 = 0;
  let w1 = 0;
  let s1 = 0;
  for (let i = 0; i < this.constants.bins; i++) {
    if (i <= t) {
      w0 += tones[i];
      s0 += i * tones[i];
    } else {
      w1 += tones[i];
      s1 += i * tones[i];
    }
  }
  if (w0 === 0 || w1 === 0) return 0;
  const total = w0 + w1;
  const mu0 = s0 / w0;
  const mu1 = s1 / w1;
  return (w0 / total) * (w1 / total) * (mu0 - mu1) * (mu0 - mu1);
}, {
  output: [256],
  constants: { bins: 256 },
});

const scores = await between(tones);

// The argmax is one tiny reduction — plain JavaScript is the right tool here.
let best = 0;
for (let t = 1; t < 256; t++) {
  if (scores[t] > scores[best]) best = t;
}
console.log('Otsu threshold: bin', best);
console.log('as a grey level:', (best / 255).toFixed(3));
`,
      inputs: utils => ({
        tones: toneHistogram(evenLuminance(utils, 3301, 0.3, 0.72, 0.2)),
      }),
      publicTests: [
        {
          name: 'one score per candidate threshold — 256 finite numbers',
          run: async ctx => {
            ctx.assert(ctx.kernels.length >= 1, 'no kernel was created — call gpu.createKernel()');
            const bins = toneHistogram(evenLuminance(ctx.utils, 3301, 0.3, 0.72, 0.2));
            const out = await ctx.kernel(bins);
            ctx.assert(out && out.length === BINS, `expected ${BINS} scores, got ${out && out.length}`);
            for (let t = 0; t < BINS; t++) {
              ctx.assert(
                Number.isFinite(out[t]),
                `score ${t} came back ${out[t]} — at t = ${t} one class is empty, so its mean is ` +
                  '0 / 0. Guard it: <code>if (w0 === 0 || w1 === 0) return 0;</code>'
              );
            }
          },
        },
        {
          name: 'score(t) is <code>p0 · p1 · (mu0 − mu1)²</code> for every cut',
          run: async ctx => {
            const bins = toneHistogram(evenLuminance(ctx.utils, 3301, 0.3, 0.72, 0.2));
            const out = await ctx.kernel(bins);
            const { score, gap } = otsuRef(bins);
            const total = bins.reduce((a, b) => a + b, 0);
            for (let t = 0; t < BINS; t++) {
              const eps = 1e-2 + Math.abs(score[t]) * 3e-4;
              const probes = [
                [t === 0 ? 0 : score[t - 1], 'every score is the score of the cut one bin lower — the sweep stops one bin early. Class 0 is bins 0…t INCLUSIVE, so the test is <code>i &lt;= t</code>, not <code>i &lt; t</code>'],
                [score[t] * total * total, 'the class weights are raw counts, not fractions — p0 and p1 have to be divided by the total pixel count, or every score is scaled by the pixel count squared'],
                [gap[t] === 0 ? 0 : score[t] / Math.abs(gap[t]), 'the gap between the class means is not squared — the score is p0 · p1 · (mu0 − mu1) · (mu0 − mu1)'],
              ];
              const hint = diagnoseRel(out[t], score[t], 1e-3, probes);
              ctx.assertClose(out[t], score[t], eps, hint || `score at t = ${t}`);
            }
          },
        },
        {
          name: 'the peak lands on a threshold that really does split the two tone humps',
          run: async ctx => {
            const bins = toneHistogram(evenLuminance(ctx.utils, 3301, 0.3, 0.72, 0.2));
            const out = await ctx.kernel(bins);
            const { score } = otsuRef(bins);
            const peak = Math.max(...score);
            let best = 0;
            for (let t = 1; t < BINS; t++) if (out[t] > out[best]) best = t;
            ctx.assert(
              score[best] >= peak * (1 - 1e-3),
              `the largest score sits at bin ${best}, which scores ${score[best].toFixed(1)} ` +
                `against the best possible ${peak.toFixed(1)} — the winning cut for this ` +
                'histogram is in the valley between the two humps, near bin 128'
            );
          },
        },
        {
          name: 'the winning bin is logged',
          run: async ctx => {
            const bins = toneHistogram(evenLuminance(ctx.utils, 3301, 0.3, 0.72, 0.2));
            const { score } = otsuRef(bins);
            const peak = Math.max(...score);
            const logged = ctx.logs
              .filter(line => line.type === 'log' && line.text)
              .flatMap(line => (String(line.text).match(/\d+/g) || []).map(Number));
            ctx.assert(
              logged.some(n => n >= 0 && n < BINS && score[n] >= peak * (1 - 1e-3)),
              'no winning bin was logged — walk the 256 scores in JavaScript, keep the index of ' +
                'the largest, and console.log it. That index IS the threshold'
            );
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            // A different scene: darker ground, dimmer marks, tighter noise.
            const bins = toneHistogram(evenLuminance(ctx.utils, 8821, 0.22, 0.6, 0.16));
            const out = await ctx.kernel(bins);
            const { score, gap } = otsuRef(bins);
            const total = bins.reduce((a, b) => a + b, 0);
            for (let t = 0; t < BINS; t++) {
              const eps = 1e-2 + Math.abs(score[t]) * 3e-4;
              const hint = diagnoseRel(out[t], score[t], 1e-3, [
                [t === 0 ? 0 : score[t - 1], 'every score is the score of the cut one bin lower — class 0 is bins 0…t INCLUSIVE'],
                [score[t] * total * total, 'the class weights are raw counts, not fractions of the total'],
                [gap[t] === 0 ? 0 : score[t] / Math.abs(gap[t]), 'the gap between the class means is not squared'],
              ]);
              ctx.assertClose(out[t], score[t], eps, hint || `score at t = ${t}`);
            }
            const peak = Math.max(...score);
            let best = 0;
            for (let t = 1; t < BINS; t++) if (out[t] > out[best]) best = t;
            ctx.assert(
              score[best] >= peak * (1 - 1e-3),
              `on a different histogram the peak landed at bin ${best}, worth ` +
                `${score[best].toFixed(1)} against a possible ${peak.toFixed(1)}`
            );
          },
        },
      ],
    },

    // ------------------------------------------------------------------ 3
    {
      slug: 'adaptive-threshold',
      title: 'A Threshold Per Neighbourhood',
      intro: `<p>Task 1's failure was not bad luck, and no cleverer <em>single</em> number fixes
        it: Otsu would pick the best one that exists and the lit corner would still saturate.
        The premise is what is wrong. One number cannot describe an image whose brightness
        changes across the frame.</p>
        <p>So stop asking for one. <strong>Adaptive thresholding</strong> compares every pixel
        against the mean of <em>its own</em> neighbourhood — a 9×9 box average, which is the
        clamped sweep the box blur in Convolution &amp; Filters already makes — plus a small
        bias <code>c</code>. A pixel is foreground when it is at least <code>c</code> brighter
        than its surroundings. That is a statement about local contrast, and it says nothing
        whatever about the lamp.</p>
        <p>The window size is the one real choice. It has to be comfortably bigger than the
        things you are hunting, or the mean drowns in them and a mark declares itself average;
        and comfortably smaller than the lighting changes, or it stops tracking them and you are
        back to task 1. Here the marks are 5 pixels across and the light drifts over tens of
        pixels, so 9×9 sits nicely in between.</p>
        <p><code>gray</code> is the same scene's luminance, one number per pixel. A luminance
        pass produces it in a real pipeline — module 1.2's finale is exactly that pass — and it
        is handed over here so the sweep is the only thing you write.</p>`,
      goal: `<strong>Goal:</strong> return <code>1</code> where <code>gray[y][x]</code> exceeds
        the mean of its clamped 9×9 neighbourhood by more than <code>this.constants.c</code>,
        and <code>0</code> everywhere else.`,
      requirements: [
        'Sum the <code>this.constants.win</code> × <code>this.constants.win</code> neighbourhood, both coordinates clamped to <code>0…this.constants.last</code>',
        'Centre the window: sample <code>this.thread.y + dy - this.constants.radius</code>, likewise for x',
        'Divide by <code>this.constants.area</code> to get the mean',
        'Return <code>1</code> when this pixel is above <code>mean + this.constants.c</code>, otherwise <code>0</code>',
      ],
      hints: [
        {
          title: 'Hint 1 — it is a box blur that ends in a question',
          body: `<p>The loop is the one from the 3×3 box blur, widened to 9×9 and reading a
            single number per cell instead of three channels. The only new line is the last
            one: instead of painting the mean, compare against it.</p>`,
        },
        {
          title: 'Hint 2 — the clamped sample',
          body: `<pre><code>let sy = this.thread.y + dy - this.constants.radius;
if (sy &lt; 0) sy = 0;
if (sy &gt; this.constants.last) sy = this.constants.last;</code></pre>
<p>— the same four lines for <code>sx</code>, then <code>sum += gray[sy][sx];</code>.</p>`,
        },
        {
          title: 'Hint 3 — the finish',
          body: `<pre><code>const mean = sum / this.constants.area;
if (gray[this.thread.y][this.thread.x] &gt; mean + this.constants.c) return 1;
return 0;</code></pre>
<p>The bias goes on the <em>mean</em>, raising the bar. Subtract it instead and flat
            ground starts reporting itself as foreground.</p>`,
        },
      ],
      transfer: `Every vision toolkit ships this: OpenCV's <code>adaptiveThreshold</code>,
        Sauvola and Niblack binarisation in document scanning, and the local-contrast test at
        the front of most feature detectors. On a GPU the box average is separable and can be
        done in two passes, or in one with a summed-area table — the same trick that makes
        real-time adaptive thresholding cheap on a phone.`,
      starterCode: `// A threshold per pixel: the box-blur sweep, ending in a comparison.
const gpu = new GPU({ mode });

const adaptive = gpu.createKernel(function (gray) {
  let sum = 0;
  // TODO: sum the 9×9 neighbourhood centred on this thread, clamping both
  // coordinates to 0…this.constants.last. Then return 1 when this pixel is
  // more than this.constants.c above the mean, and 0 when it is not.
  return 0;
}, {
  output: [128, 128],
  constants: { last: 127, win: 9, radius: 4, area: 81, c: 0.03 },
});

const mask = await adaptive(gray);

${ASCII_DUMP}

let lit = 0;
let dark = 0;
for (let y = 0; y < 32; y++) {
  for (let x = 0; x < 32; x++) {
    lit += mask[y][x];
    dark += mask[y + 96][x + 96];
  }
}
console.log('top-left 32x32 foreground:', lit, 'of 1024');
console.log('bottom-right 32x32 foreground:', dark, 'of 1024');
`,
      solutionCode: `// A threshold per pixel: the box-blur sweep, ending in a comparison.
const gpu = new GPU({ mode });

const adaptive = gpu.createKernel(function (gray) {
  let sum = 0;
  for (let dy = 0; dy < this.constants.win; dy++) {
    for (let dx = 0; dx < this.constants.win; dx++) {
      let sy = this.thread.y + dy - this.constants.radius;
      let sx = this.thread.x + dx - this.constants.radius;
      if (sy < 0) sy = 0;
      if (sy > this.constants.last) sy = this.constants.last;
      if (sx < 0) sx = 0;
      if (sx > this.constants.last) sx = this.constants.last;
      sum += gray[sy][sx];
    }
  }
  const mean = sum / this.constants.area;
  if (gray[this.thread.y][this.thread.x] > mean + this.constants.c) return 1;
  return 0;
}, {
  output: [128, 128],
  constants: { last: 127, win: 9, radius: 4, area: 81, c: 0.03 },
});

const mask = await adaptive(gray);

${ASCII_DUMP}

let lit = 0;
let dark = 0;
for (let y = 0; y < 32; y++) {
  for (let x = 0; x < 32; x++) {
    lit += mask[y][x];
    dark += mask[y + 96][x + 96];
  }
}
console.log('top-left 32x32 foreground:', lit, 'of 1024');
console.log('bottom-right 32x32 foreground:', dark, 'of 1024');
`,
      inputs: utils => ({ gray: litLuminance(utils, false) }),
      publicTests: [
        {
          name: 'a <code>128×128</code> grid holding nothing but <code>1</code> and <code>0</code>',
          run: async ctx => {
            ctx.assert(ctx.kernels.length >= 1, 'no kernel was created — call gpu.createKernel()');
            const gray = litLuminance(ctx.utils, false);
            const out = await ctx.kernel(gray);
            ctx.assert(out && out.length === SIZE, `expected ${SIZE} rows, got ${out && out.length}`);
            ctx.assert(out[0] && out[0].length === SIZE, `each row should hold ${SIZE} values`);
            assertBinary(ctx, out, 'the mask', (v, y, x) => {
              let sum = 0;
              for (let dy = 0; dy < WINDOW; dy++) {
                for (let dx = 0; dx < WINDOW; dx++) {
                  sum += gray[Math.min(LAST, Math.max(0, y + dy - RADIUS))][
                    Math.min(LAST, Math.max(0, x + dx - RADIUS))
                  ];
                }
              }
              if (Math.abs(v - sum / AREA) <= 2e-3) {
                return `cell [${y}][${x}] holds ${v.toFixed(3)}, which is the neighbourhood MEAN ` +
                  '— the mean is what you compare against, not what you return. Return 1 or 0';
              }
              return Math.abs(v - sum) <= 1e-2
                ? `cell [${y}][${x}] holds ${v.toFixed(2)}, the raw window SUM — divide by ` +
                  'this.constants.area first, then compare'
                : null;
            });
          },
        },
        {
          name: 'foreground is exactly <code>gray &gt; mean(9×9) + c</code>',
          run: async ctx => {
            const gray = litLuminance(ctx.utils, false);
            const image = litImage(ctx.utils, false);
            const out = await ctx.kernel(gray);
            const ref = adaptiveMaskRef(gray);
            assertMask(ctx, out, ref, adaptiveProbes(gray, image, ref), 'the mask');
          },
        },
        {
          name: 'both corners work now — the same marks, found under both lamps',
          run: async ctx => {
            // Task 1 found 1024 of 1024 in the lit corner and 0 of 1024 in the
            // dark one. Every 32×32 corner here holds four 5×5 marks.
            const gray = litLuminance(ctx.utils, false);
            const out = await ctx.kernel(gray);
            let lit = 0;
            let dark = 0;
            for (let y = 0; y < 32; y++) {
              for (let x = 0; x < 32; x++) {
                lit += out[y][x] > 0.5 ? 1 : 0;
                dark += out[y + 96][x + 96] > 0.5 ? 1 : 0;
              }
            }
            ctx.assert(
              lit === 100,
              `the top-left 32×32 corner holds four 5×5 marks and nothing else — expected 100 ` +
                `foreground pixels, got ${lit}. A global threshold gave all 1024 here`
            );
            ctx.assert(
              dark === 100,
              `the bottom-right 32×32 corner holds four 5×5 marks too — expected 100 foreground ` +
                `pixels, got ${dark}. A global threshold gave 0 here, and the marks were always there`
            );
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            const gray = litLuminance(ctx.utils, true);
            const image = litImage(ctx.utils, true);
            const out = await ctx.kernel(gray);
            const ref = adaptiveMaskRef(gray);
            assertMask(ctx, out, ref, adaptiveProbes(gray, image, ref), 'the mask');
          },
        },
        {
          name: 'private test #2',
          run: async ctx => {
            // A flat map: nothing is brighter than its own neighbourhood, so a
            // correct kernel returns an entirely empty mask. Anything that
            // compares against the mean without the bias fails here.
            const flat = new Array(SIZE);
            for (let y = 0; y < SIZE; y++) flat[y] = new Array(SIZE).fill(0.5);
            const out = await ctx.kernel(flat);
            let on = 0;
            for (let y = 0; y < SIZE; y++) {
              for (let x = 0; x < SIZE; x++) on += out[y][x] > 0.5 ? 1 : 0;
            }
            ctx.assert(
              on === 0,
              `a perfectly flat map has no local contrast anywhere, so the mask should be empty ` +
                `— got ${on} foreground pixels. Every pixel here equals its own neighbourhood ` +
                'mean, and equal is not "brighter by more than c"'
            );
          },
        },
      ],
    },

    // ------------------------------------------------------------------ 4
    {
      slug: 'erode-and-dilate',
      title: 'Erode and Dilate: the Sweep, With Min and Max',
      intro: `<p>A fresh mask is never clean. Stray single pixels where a highlight caught the
        sensor; single missing pixels where a shape had a dark fleck. <strong>Morphology</strong>
        is the repair kit, and it is built out of two operations you have, in a real sense,
        already written.</p>
        <p>In Convolution &amp; Filters the 3×3 window read nine samples, multiplied them by
        nine weights and added them up. Keep the window, keep the nine reads, keep the clamped
        edges — and replace the weighted sum with a <strong>minimum</strong>. That is
        <strong>erosion</strong>: a pixel survives only if <em>every</em> one of its neighbours
        is foreground, so shapes lose a one-pixel rind and lone specks vanish. Replace it with a
        <strong>maximum</strong> and you have <strong>dilation</strong>: a pixel lights up if
        <em>any</em> neighbour does, so shapes gain a rind and small holes close over.</p>
        <p>Same access pattern, different reduction operator. That is worth saying out loud,
        because it generalises: a neighbourhood sweep is a <em>shape</em>, and what you do with
        the nine values you gathered is a separate decision. Sum them and you have a filter;
        take their extreme and you have morphology.</p>
        <p>The window has a name — the <strong>structuring element</strong> — and a 3×3 square is
        the plainest one there is. <strong>Edges:</strong> this module <em>clamps</em>, so a
        sample that falls off the frame reuses the nearest in-bounds cell, exactly as the box
        blur did. Treating out-of-bounds as background is just as defensible, and it is a
        different answer: a shape lying flush against the frame erodes away along that edge
        instead of surviving it. Four of the shapes in <code>mask</code> run to the frame, so the
        tests can tell which rule you picked.</p>`,
      goal: `<strong>Goal:</strong> two kernels over the same clamped 3×3 sweep — an
        <strong>eroder</strong> that returns the smallest sample in the window, then a
        <strong>dilator</strong> that returns the largest.`,
      requirements: [
        'Create the eroder <em>first</em> and the dilator <em>second</em> — the tests read them in that order',
        'Sweep the 3×3 neighbourhood with both coordinates clamped to <code>0…this.constants.last</code>',
        'Erosion keeps the minimum (start at <code>1</code>, <code>Math.min</code>); dilation keeps the maximum (start at <code>0</code>, <code>Math.max</code>)',
        'Nothing else changes — a min or max of 1s and 0s is still exactly 1 or 0',
      ],
      hints: [
        {
          title: 'Hint 1 — the same nine reads',
          body: `<p>Copy the box blur's double loop verbatim, clamps and all. Replace the three
            channel sums with one accumulator, and replace <code>+=</code> with
            <code>Math.min</code> or <code>Math.max</code>.</p>`,
        },
        {
          title: 'Hint 2 — the accumulator',
          body: `<p>Start the minimum at the largest value a mask can hold and the maximum at the
            smallest, so the first sample always wins:</p>
<pre><code>let lo = 1;
// … inside the loops …
lo = Math.min(lo, mask[sy][sx]);</code></pre>
<p>and the mirror image — <code>let hi = 0;</code> with <code>Math.max</code> — for
            dilation.</p>`,
        },
        {
          title: 'Hint 3 — which way round?',
          body: `<p>Say it as a sentence. Erosion: "I stay foreground only if <em>all</em> of my
            neighbours are" — that is an AND over the window, and the AND of 1s and 0s is their
            minimum. Dilation: "I become foreground if <em>any</em> neighbour is" — an OR, which
            is their maximum. If your shapes are growing when you asked them to shrink, these
            two are the wrong way round.</p>`,
        },
      ],
      transfer: `Morphology is a first-class citizen everywhere: NVIDIA's NPP has
        <code>nppiErode</code>/<code>nppiDilate</code>, Metal Performance Shaders has
        <code>MPSImageAreaMin</code> and <code>MPSImageAreaMax</code>, and every WGSL post-process
        chain grows one eventually. The optimisation is the same as for a box blur — a rectangular
        structuring element is separable, so an <em>n</em>×<em>n</em> erosion is a horizontal
        pass followed by a vertical one.`,
      starterCode: `// One sweep, two reduction operators.
const gpu = new GPU({ mode });

const erode = gpu.createKernel(function (mask) {
  let lo = 1;
  // TODO: sweep the 3×3 neighbourhood with both coordinates clamped to
  // 0…this.constants.last, and keep the SMALLEST sample you saw.
  return lo;
}, {
  output: [128, 128],
  constants: { last: 127 },
});

const dilate = gpu.createKernel(function (mask) {
  let hi = 0;
  // TODO: the same sweep, keeping the LARGEST sample.
  return hi;
}, {
  output: [128, 128],
  constants: { last: 127 },
});

${COUNT_HELPER}

console.log('mask      :', count(mask));
console.log('eroded    :', count(await erode(mask)));
console.log('dilated   :', count(await dilate(mask)));
`,
      solutionCode: `// One sweep, two reduction operators.
const gpu = new GPU({ mode });

const erode = gpu.createKernel(function (mask) {
  let lo = 1;
  for (let dy = 0; dy < 3; dy++) {
    for (let dx = 0; dx < 3; dx++) {
      let sy = this.thread.y + dy - 1;
      let sx = this.thread.x + dx - 1;
      if (sy < 0) sy = 0;
      if (sy > this.constants.last) sy = this.constants.last;
      if (sx < 0) sx = 0;
      if (sx > this.constants.last) sx = this.constants.last;
      lo = Math.min(lo, mask[sy][sx]);
    }
  }
  return lo;
}, {
  output: [128, 128],
  constants: { last: 127 },
});

const dilate = gpu.createKernel(function (mask) {
  let hi = 0;
  for (let dy = 0; dy < 3; dy++) {
    for (let dx = 0; dx < 3; dx++) {
      let sy = this.thread.y + dy - 1;
      let sx = this.thread.x + dx - 1;
      if (sy < 0) sy = 0;
      if (sy > this.constants.last) sy = this.constants.last;
      if (sx < 0) sx = 0;
      if (sx > this.constants.last) sx = this.constants.last;
      hi = Math.max(hi, mask[sy][sx]);
    }
  }
  return hi;
}, {
  output: [128, 128],
  constants: { last: 127 },
});

${COUNT_HELPER}

console.log('mask      :', count(mask));
console.log('eroded    :', count(await erode(mask)));
console.log('dilated   :', count(await dilate(mask)));
`,
      inputs: utils => ({ mask: noisyMask(utils, 7301) }),
      publicTests: [
        {
          name: 'two kernels — an eroder, then a dilator',
          run: async ctx => {
            ctx.assert(
              ctx.kernels.length >= 2,
              `expected 2 kernels (erode first, dilate second), found ${ctx.kernels.length}`
            );
            const mask = noisyMask(ctx.utils, 7301);
            for (const [i, name] of [[0, 'eroder'], [1, 'dilator']]) {
              const out = await ctx.kernels[i](mask);
              ctx.assert(
                out && out.length === SIZE && out[0] && out[0].length === SIZE,
                `the ${name} should return a ${SIZE}×${SIZE} grid`
              );
              assertBinary(ctx, out, `the ${name}'s output`);
            }
          },
        },
        {
          name: 'erosion is the clamped 3×3 <strong>minimum</strong>',
          run: async ctx => {
            const mask = noisyMask(ctx.utils, 7301);
            const out = await ctx.kernels[0](mask);
            const ref = erodeRef(mask);
            assertMask(ctx, out, ref, erodeProbes(mask), 'the eroded mask');
          },
        },
        {
          name: 'dilation is the clamped 3×3 <strong>maximum</strong>',
          run: async ctx => {
            const mask = noisyMask(ctx.utils, 7301);
            const out = await ctx.kernels[1](mask);
            const ref = dilateRef(mask);
            assertMask(ctx, out, ref, dilateProbes(mask), 'the dilated mask');
          },
        },
        {
          name: 'one shrinks and one grows: every speck dies, every pinhole closes',
          run: async ctx => {
            const mask = noisyMask(ctx.utils, 7301);
            const eroded = await ctx.kernels[0](mask);
            const dilated = await ctx.kernels[1](mask);
            const before = countOn(mask);
            const after = countOn(eroded);
            const grown = countOn(dilated);
            ctx.assert(
              after < before,
              `erosion should leave fewer foreground pixels than it started with (${before}), ` +
                `got ${after} — are the two kernels the wrong way round?`
            );
            ctx.assert(
              grown > before,
              `dilation should leave more foreground pixels than it started with (${before}), ` +
                `got ${grown} — are the two kernels the wrong way round?`
            );
            // Every isolated 1×1 and 2×2 speck is thinner than the element.
            const specks = removedRef(mask, openRef(mask));
            for (let y = 0; y < SIZE; y++) {
              for (let x = 0; x < SIZE; x++) {
                if (specks[y][x] !== 1) continue;
                ctx.assert(
                  eroded[y][x] < 0.5,
                  `the speck at [${y}][${x}] is only a pixel or two across and should not ` +
                    'survive an erosion — a cell survives only when all nine of its samples are 1'
                );
              }
            }
            // Every pinhole is a single background cell inside a solid shape.
            const holes = removedRef(closeRef(mask), mask);
            for (let y = 0; y < SIZE; y++) {
              for (let x = 0; x < SIZE; x++) {
                if (holes[y][x] !== 1) continue;
                ctx.assert(
                  dilated[y][x] > 0.5,
                  `the pinhole at [${y}][${x}] is surrounded by foreground and should be ` +
                    'swallowed by a dilation — a cell lights up when any of its samples is 1'
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
            const mask = noisyMask(ctx.utils, 2255);
            const eroded = await ctx.kernels[0](mask);
            const dilated = await ctx.kernels[1](mask);
            assertMask(ctx, eroded, erodeRef(mask), erodeProbes(mask), 'the eroded mask');
            assertMask(ctx, dilated, dilateRef(mask), dilateProbes(mask), 'the dilated mask');
          },
        },
        {
          name: 'private test #2',
          run: async ctx => {
            // A mask that is entirely foreground. Under clamping both sweeps
            // are the identity; under background padding the erosion would eat
            // the whole frame.
            const solid = new Array(SIZE);
            for (let y = 0; y < SIZE; y++) solid[y] = new Array(SIZE).fill(1);
            const eroded = await ctx.kernels[0](solid);
            assertMask(ctx, eroded, solid, [
              [sweepRef(solid, 'min', 'zero'), 'a completely full mask came back with its frame eroded away — out-of-bounds samples were treated as background. This module clamps: an off-frame sample reuses the nearest in-bounds cell, so a full mask erodes to itself'],
            ], 'eroding a completely full mask');
            const dilated = await ctx.kernels[1](solid);
            assertMask(ctx, dilated, solid, [], 'dilating a completely full mask');
          },
        },
      ],
    },

    // ------------------------------------------------------------------ 5
    {
      slug: 'opening-and-closing',
      title: 'Opening and Closing: Order Is the Answer',
      intro: `<p>Erosion on its own is a blunt instrument: it kills the specks and takes a rind
        off everything else. Dilation on its own is the same mistake in reverse. Run them back to
        back and the size change cancels while the repair survives — and which repair you get
        depends entirely on which one goes first.</p>
        <p><strong>Opening</strong> is erode <em>then</em> dilate. The erosion wipes anything
        thinner than the structuring element, the dilation grows the survivors back to size:
        small bright specks are gone for good and everything else ends up roughly where it
        started. <strong>Closing</strong> is dilate <em>then</em> erode: the dilation swallows
        small dark holes, the erosion pulls the outlines back in, so pinholes fill and the specks
        stay exactly where they were.</p>
        <p>They are not inverses and they are not interchangeable. Opening removes; closing
        fills. Ask for one and write the other and you get precisely the opposite of what you
        wanted — which is the single most reliable way to lose an afternoon to morphology.</p>
        <p>Both kernels are given below, so this task is about the plumbing: chain them, and
        chain them twice. Two erosions followed by two dilations is an opening with a radius-2
        element — it clears out the 3×3 clumps that a single pass is too gentle to touch.</p>`,
      goal: `<strong>Goal:</strong> build an opening, a closing and a two-pass opening from the
        given kernels, and report what each one changed with the exact labels the starter uses.`,
      requirements: [
        'Opening is <code>await dilate(await erode(mask))</code>; closing is <code>await erode(await dilate(mask))</code>',
        'The two-pass opening runs both erosions before either dilation',
        'Write the <code>removed</code> kernel: <code>1</code> where <code>before</code> is foreground and <code>after</code> is not',
        'Log the three counts with the labels already in the starter',
      ],
      hints: [
        {
          title: 'Hint 1 — chaining kernels',
          body: `<p>A kernel's result is an ordinary 2D array, so it goes straight back into
            another kernel: <code>await dilate(await erode(mask))</code> is the whole opening. Every pass is
            a separate launch, which is exactly how a real pipeline does it (and Pipelines &amp;
            Textures shows how to keep the intermediate on the GPU).</p>`,
        },
        {
          title: 'Hint 2 — which is which',
          body: `<p>Read the name outwards. An <em>opening</em> opens gaps up: it must start by
            shrinking, so erosion goes first. A <em>closing</em> closes gaps: it starts by
            growing. If your "opening" is filling holes instead of clearing specks, you have
            written a closing.</p>`,
        },
        {
          title: 'Hint 3 — the difference kernel',
          body: `<pre><code>const removed = gpu.createKernel(function (before, after) {
  if (before[this.thread.y][this.thread.x] &gt; after[this.thread.y][this.thread.x]) return 1;
  return 0;
}, { output: [128, 128] });</code></pre>
<p>Feed it <code>(mask, opened)</code> to see what the opening threw away, and
            <code>(closed, mask)</code> to see what the closing filled in.</p>`,
        },
      ],
      transfer: `Opening and closing are the standard pre-processing pair in OpenCV
        (<code>MORPH_OPEN</code>, <code>MORPH_CLOSE</code>) and in every medical- and
        satellite-imaging toolchain. On a GPU each is a fixed chain of launches with no readback
        in between — the ping-pong between two buffers that WebGPU and CUDA pipelines are built
        around.`,
      starterCode: `// Two orders, two completely different repairs.
const gpu = new GPU({ mode });

${GIVEN_SWEEPS}

const removed = gpu.createKernel(function (before, after) {
  // TODO: 1 where before is foreground and after is not; 0 otherwise.
  return 0;
}, { output: [128, 128] });

${COUNT_HELPER}

// TODO: opening is erode then dilate; closing is dilate then erode;
// the two-pass opening erodes twice before dilating twice.
const opened = mask;
const closed = mask;
const openedTwice = mask;

console.log('opening removed:', count(await removed(mask, opened)));
console.log('closing added:', count(await removed(closed, mask)));
console.log('two passes removed:', count(await removed(mask, openedTwice)));
`,
      solutionCode: `// Two orders, two completely different repairs.
const gpu = new GPU({ mode });

${GIVEN_SWEEPS}

const removed = gpu.createKernel(function (before, after) {
  if (before[this.thread.y][this.thread.x] > after[this.thread.y][this.thread.x]) return 1;
  return 0;
}, { output: [128, 128] });

${COUNT_HELPER}

const opened = await dilate(await erode(mask));
const closed = await erode(await dilate(mask));
const openedTwice = await dilate(await dilate(await erode(await erode(mask))));

console.log('opening removed:', count(await removed(mask, opened)));
console.log('closing added:', count(await removed(closed, mask)));
console.log('two passes removed:', count(await removed(mask, openedTwice)));
`,
      inputs: utils => ({ mask: noisyMask(utils, 7301) }),
      publicTests: [
        {
          name: 'the <code>removed</code> kernel marks what a pass threw away',
          run: async ctx => {
            ctx.assert(ctx.kernels.length >= 3, `expected 3 kernels, found ${ctx.kernels.length}`);
            const mask = noisyMask(ctx.utils, 7301);
            const opened = openRef(mask);
            const out = await ctx.kernel(mask, opened);
            ctx.assert(
              out && out.length === SIZE && out[0] && out[0].length === SIZE,
              `expected a ${SIZE}×${SIZE} grid`
            );
            const ref = removedRef(mask, opened);
            assertMask(ctx, out, ref, [
              [removedRef(opened, mask), 'the two arguments are the wrong way round — this marks what the second grid has and the first does not'],
              [mask, 'that is just the first grid — the comparison against `after` never happened'],
            ], 'removed(before, after)');
          },
        },
        {
          name: 'opening threw away the specks — and only the specks',
          run: async ctx => {
            const mask = noisyMask(ctx.utils, 7301);
            const expected = countOn(removedRef(mask, openRef(mask)));
            const got = loggedCount(ctx, 'opening removed:');
            ctx.assert(
              got !== null,
              'no "opening removed:" line in the console — keep the starter\'s console.log lines exactly as they are'
            );
            // Removing nothing has two candidates — an untouched mask and a
            // closing, which puts every speck back — and no way to tell them
            // apart from one number. One message that covers both is honest;
            // two that disagree would cancel each other and say nothing.
            const hint = diagnose(got, expected, 0.5, [
              [0, 'nothing was removed at all. Either `opened` is still the mask itself, or it is a CLOSING — dilate-then-erode grows the specks and then puts them straight back. An opening erodes FIRST'],
              [countOn(removedRef(mask, erodeRef(mask))), 'that is what a bare erosion removed — the dilation that grows the survivors back has not run'],
            ]);
            ctx.assertClose(
              got,
              expected,
              0.5,
              hint || `opening should have removed ${expected} foreground pixels`
            );
          },
        },
        {
          name: 'closing filled the pinholes — and only the pinholes',
          run: async ctx => {
            const mask = noisyMask(ctx.utils, 7301);
            const expected = countOn(removedRef(closeRef(mask), mask));
            const got = loggedCount(ctx, 'closing added:');
            ctx.assert(
              got !== null,
              'no "closing added:" line in the console — keep the starter\'s console.log lines exactly as they are'
            );
            const hint = diagnose(got, expected, 0.5, [
              [countOn(removedRef(openRef(mask), mask)), 'an opening adds nothing at all, so that 0 is the count for erode-then-dilate. Closing dilates FIRST'],
              [countOn(removedRef(dilateRef(mask), mask)), 'that is what a bare dilation added — every shape is still a pixel fatter, because the erosion that pulls the outlines back has not run'],
            ]);
            ctx.assertClose(
              got,
              expected,
              0.5,
              hint || `closing should have filled ${expected} background pixels`
            );
          },
        },
        {
          name: 'a second pass reaches the clumps a single opening is too gentle for',
          run: async ctx => {
            const mask = noisyMask(ctx.utils, 7301);
            const expected = countOn(removedRef(mask, openTwiceRef(mask)));
            const once = countOn(removedRef(mask, openRef(mask)));
            const got = loggedCount(ctx, 'two passes removed:');
            ctx.assert(
              got !== null,
              'no "two passes removed:" line in the console — keep the starter\'s console.log lines exactly as they are'
            );
            // An opening is idempotent, so running the pair twice in sequence
            // removes exactly what running it once did: the single-pass number
            // is the signature of every interleaved arrangement too.
            const hint = diagnose(got, expected, 0.5, [
              [once, `that is the single-pass number again (${once}) — both erosions have to run before either dilation. Interleaving them just performs the same opening twice, and an opening does nothing the second time`],
            ]);
            ctx.assertClose(
              got,
              expected,
              0.5,
              hint || `a two-pass opening should have removed ${expected} foreground pixels`
            );
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            // The difference kernel against a hand-built pair, so it cannot
            // have been fitted to the task's own mask.
            const before = blankMask();
            const after = blankMask();
            paintRect(before, 10, 10, 20, 20, 1);
            paintRect(after, 12, 12, 16, 16, 1);
            paintRect(after, 60, 60, 4, 4, 1); // only in `after`: never "removed"
            const out = await ctx.kernel(before, after);
            const ref = removedRef(before, after);
            assertMask(ctx, out, ref, [
              [removedRef(after, before), 'the two arguments are the wrong way round'],
            ], 'removed(before, after)');
          },
        },
        {
          name: 'private test #2',
          run: async ctx => {
            // Opening and closing are different operations. If the two reported
            // numbers agree, the same chain was almost certainly run twice.
            const mask = noisyMask(ctx.utils, 7301);
            const removedBy = loggedCount(ctx, 'opening removed:');
            const addedBy = loggedCount(ctx, 'closing added:');
            ctx.assert(
              removedBy !== null && addedBy !== null,
              'both the "opening removed:" and "closing added:" lines have to reach the console'
            );
            ctx.assert(
              removedBy !== addedBy,
              'the opening and the closing reported the same number — they are different ' +
                'operations on this mask, so one of the two chains is in the wrong order'
            );
            ctx.assertClose(removedBy, countOn(removedRef(mask, openRef(mask))), 0.5, 'opening removed');
            ctx.assertClose(addedBy, countOn(removedRef(closeRef(mask), mask)), 0.5, 'closing added');
          },
        },
      ],
    },

    // ------------------------------------------------------------------ 6
    {
      slug: 'clean-and-count',
      title: 'Payoff: Clean the Mask, Count What Is Left',
      intro: `<p>The whole module in one run. <code>noisy</code> is a mask straight off a
        threshold: sixteen solid rectangles and a confetti of stray one- and two-pixel specks.
        Open it once to clear the confetti, then count what survived.</p>
        <p>Counting connected blobs sounds like it needs a real labelling algorithm — and in
        general it does. But every shape here is an axis-aligned rectangle, and a rectangle has
        exactly one <strong>top-left corner</strong>: a foreground pixel whose neighbour above
        and whose neighbour to the left are both background. So count corners and you have
        counted shapes, with a per-pixel predicate and a sum — the same map-then-reduce shape
        Reductions is built on.</p>
        <p>Be straight about the caveat. A U-shaped blob has two top-left corners and this would
        count it twice. The trick is exact for <em>this</em> mask, not for all masks; real
        connected-component labelling is a different and much heavier algorithm.</p>
        <p>One more border note. This mask has a clear frame — nothing touches the edge — so a
        clamped read of a missing neighbour lands on background either way and the count comes
        out exact. Had a shape run to the edge, the clamped read would have returned the shape
        itself and that corner would have gone uncounted: one more place where the border rule is
        a decision, not a detail.</p>`,
      goal: `<strong>Goal:</strong> open <code>noisy</code> once, write a <code>corners</code>
        kernel that marks each rectangle's top-left pixel, and log the blob count before and
        after the cleanup.`,
      requirements: [
        'Clean the mask with one opening: erode, then dilate',
        '<code>corners</code> returns <code>1</code> only for a foreground pixel whose neighbour above <em>and</em> neighbour to the left are background',
        'Clamp both neighbour indexes — a negative index reads outside the mask',
        'Sum the corner grid in JavaScript and log both counts with the starter\'s labels',
      ],
      hints: [
        {
          title: 'Hint 1 — three conditions',
          body: `<p>A cell is a corner when all three hold: it is foreground, the cell above is
            not, and the cell to its left is not. Any one of them failing means 0 — which reads
            nicely as three early returns.</p>`,
        },
        {
          title: 'Hint 2 — clamping just the two you need',
          body: `<p>Only the low side can go out of bounds here, so two clamps are enough:</p>
<pre><code>let up = this.thread.y - 1;
if (up &lt; 0) up = 0;
let left = this.thread.x - 1;
if (left &lt; 0) left = 0;</code></pre>`,
        },
        {
          title: 'Hint 3 — the whole body',
          body: `<pre><code>const y = this.thread.y;
const x = this.thread.x;
if (mask[y][x] &lt; 0.5) return 0;
let up = y - 1;
if (up &lt; 0) up = 0;
let left = x - 1;
if (left &lt; 0) left = 0;
if (mask[up][x] &gt; 0.5) return 0;
if (mask[y][left] &gt; 0.5) return 0;
return 1;</code></pre>`,
        },
      ],
      transfer: `Cleaning a mask and then reducing it to a handful of numbers is what a vision
        pipeline actually does — the mask is never the product. The corner predicate is a
        <em>stencil</em> in CUDA/ROCm terms and the sum is a standard reduction, so on any
        platform this is one filter pass feeding one reduction: precisely the two primitives
        this course keeps coming back to.`,
      starterCode: `// Threshold, clean, count. The whole module in one run.
const gpu = new GPU({ mode });

${GIVEN_SWEEPS}

const corners = gpu.createKernel(function (mask) {
  // TODO: return 1 only when this pixel is foreground AND the pixels above it
  // and to its left are both background. Clamp both neighbour indexes.
  return 0;
}, {
  output: [128, 128],
  constants: { last: 127 },
});

${COUNT_HELPER}

// TODO: one opening — erode first, then dilate.
const clean = noisy;

console.log('blobs before cleaning:', count(await corners(noisy)));
console.log('blobs after cleaning:', count(await corners(clean)));
`,
      solutionCode: `// Threshold, clean, count. The whole module in one run.
const gpu = new GPU({ mode });

${GIVEN_SWEEPS}

const corners = gpu.createKernel(function (mask) {
  const y = this.thread.y;
  const x = this.thread.x;
  if (mask[y][x] < 0.5) return 0;
  let up = y - 1;
  if (up < 0) up = 0;
  let left = x - 1;
  if (left < 0) left = 0;
  if (mask[up][x] > 0.5) return 0;
  if (mask[y][left] > 0.5) return 0;
  return 1;
}, {
  output: [128, 128],
  constants: { last: 127 },
});

${COUNT_HELPER}

const clean = await dilate(await erode(noisy));

console.log('blobs before cleaning:', count(await corners(noisy)));
console.log('blobs after cleaning:', count(await corners(clean)));
`,
      inputs: utils => ({ noisy: sceneMask(utils, 4409) }),
      publicTests: [
        {
          name: 'the corner kernel marks one pixel per rectangle',
          run: async ctx => {
            ctx.assert(ctx.kernels.length >= 3, `expected 3 kernels, found ${ctx.kernels.length}`);
            const noisy = sceneMask(ctx.utils, 4409);
            const clean = openRef(noisy);
            const out = await ctx.kernel(clean);
            ctx.assert(
              out && out.length === SIZE && out[0] && out[0].length === SIZE,
              `expected a ${SIZE}×${SIZE} grid`
            );
            assertBinary(ctx, out, 'the corner grid');
            const ref = cornerRef(clean);
            assertMask(ctx, out, ref, cornerProbes(clean), 'the corner grid');
            ctx.assertClose(countOn(out), 16, 0.5, 'corners found in the cleaned mask');
          },
        },
        {
          name: 'the same kernel counts the dirty mask honestly too',
          run: async ctx => {
            const noisy = sceneMask(ctx.utils, 4409);
            const out = await ctx.kernel(noisy);
            const ref = cornerRef(noisy);
            assertMask(ctx, out, ref, cornerProbes(noisy), 'the corner grid');
            ctx.assertClose(
              countOn(out),
              countOn(ref),
              0.5,
              `every speck is its own little rectangle, so the dirty mask really does hold ` +
                `${countOn(ref)} blobs`
            );
          },
        },
        {
          name: 'the cleanup takes 85 blobs down to 16',
          run: async ctx => {
            const noisy = sceneMask(ctx.utils, 4409);
            const dirty = countOn(cornerRef(noisy));
            const clean = countOn(cornerRef(openRef(noisy)));
            const before = loggedCount(ctx, 'blobs before cleaning:');
            const after = loggedCount(ctx, 'blobs after cleaning:');
            ctx.assert(
              before !== null && after !== null,
              'both counts have to reach the console — keep the starter\'s console.log lines as they are'
            );
            ctx.assertClose(before, dirty, 0.5, `blobs before cleaning (expected ${dirty})`);
            // A closing leaves this mask's blob count exactly where it found
            // it, which is also what doing nothing looks like. The two share
            // one message rather than cancelling each other out.
            const untouched =
              'the cleaned count is the dirty one over again — either `clean` is still the ' +
              'mask itself, or it went through a CLOSING, and dilate-then-erode puts every ' +
              'speck back. An opening erodes FIRST';
            const hint = diagnose(after, clean, 0.5, [
              [dirty, untouched],
              [countOn(cornerRef(closeRef(noisy))), untouched],
              [countOn(cornerRef(erodeRef(noisy))), 'that is the count after a bare erosion — the specks are gone but every shape is still a pixel thinner than it should be, because the dilation has not run'],
            ]);
            ctx.assertClose(after, clean, 0.5, hint || `blobs after cleaning (expected ${clean})`);
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            // Four rectangles at known positions: the count cannot be a
            // constant fitted to the task's own mask.
            const built = blankMask();
            paintRect(built, 5, 5, 12, 9, 1);
            paintRect(built, 40, 20, 8, 30, 1);
            paintRect(built, 70, 70, 25, 25, 1);
            paintRect(built, 100, 10, 6, 6, 1);
            const out = await ctx.kernel(built);
            const ref = cornerRef(built);
            assertMask(ctx, out, ref, cornerProbes(built), 'the corner grid');
            ctx.assertClose(countOn(out), 4, 0.5, 'four rectangles, four corners');
          },
        },
        {
          name: 'private test #2',
          run: async ctx => {
            // A mask with nothing in it has no corners, and a completely full
            // mask has exactly one — the frame's own top-left pixel is the only
            // cell whose clamped neighbours could ever be background, and under
            // clamping they are not, so a full mask has none at all.
            const empty = blankMask();
            const outEmpty = await ctx.kernel(empty);
            assertMask(ctx, outEmpty, cornerRef(empty), [], 'the corner grid of an empty mask');
            const solid = new Array(SIZE);
            for (let y = 0; y < SIZE; y++) solid[y] = new Array(SIZE).fill(1);
            const outSolid = await ctx.kernel(solid);
            assertMask(ctx, outSolid, cornerRef(solid), [], 'the corner grid of a completely full mask');
          },
        },
      ],
    },
  ],
};
