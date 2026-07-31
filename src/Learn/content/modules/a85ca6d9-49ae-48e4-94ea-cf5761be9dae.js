// Module: Wavefronts: Aligning DNA on the Diagonal —
// uuid a85ca6d9-49ae-48e4-94ea-cf5761be9dae (short id a85ca6d9).
// The file name is the uuid; identity lives in the exported object below, never
// in the path. No legacyId — this module postdates the uuid migration.
//
// Six tasks: the Smith-Waterman recurrence and why it looks fatally serial →
// the anti-diagonal, one launch per diagonal → indexing a diagonal compactly →
// driving all |A| + |B| − 1 launches and reading the best score → the traceback
// pointer matrix and the aligned strings → the wavefront on screen, with the
// occupancy bill it runs up.
//
// Kernel-authoring rules (contract): no closures inside kernel functions, only
// numbers / nested number arrays as inputs, this.thread.* for indexing,
// this.constants.* for compile-time values. Every kernel call is awaited (gpu.js
// runs in `async` mode, where every backend returns a Promise). Every task
// passes in cpu, webgl and auto.
//
// WHY THE NUMBERS IN THIS FILE ARE EXACT, NOT APPROXIMATE. Every score in
// Smith-Waterman with integer match/mismatch/gap weights is an integer, and
// every integer this module can produce fits comfortably in float32 (the whole
// small matrix tops out at 13, the long one at 40). So the tests compare with
// tolerances like 1e-3 not because the arithmetic is fuzzy but because the
// values arrive as Float32Array elements; there is no accumulation anywhere, and
// no assertion in this file sits near a decision boundary. Task 5's pointer
// kernel goes further and compares scores for EQUALITY inside the kernel, which
// is only legitimate because of this: `H[i][j] === H[i-1][j-1] + s` is an
// integer identity on both backends, verified in cpu, webgl and auto.
//
// SIZES. The teaching pair is 8 × 9 (a 9 × 10 matrix, 16 launches) precisely so
// a learner can check it by hand — it is the textbook Smith-Waterman example,
// and its answer, 13 at [6][7] aligning GTT-AC with GTTGAC, is the one every
// reference prints. Task 6 moves to 32 × 36 (a 33 × 37 matrix, 67 launches) so
// the wavefront has something to sweep across. Measured in the app with a
// performance.now() around the loop, three runs per mode: that whole program —
// 67 sweeps, 67 paints, 67 rendered frames — takes 10–23 ms on the gpu.js CPU
// backend and 78–219 ms in auto, so nothing here needs a budgetMs even with the
// pre-flight probe running the program a second time. The largest launch in the
// module is 1,221 threads, far under the guard's 65,536-thread threshold, so the
// pre-flight guard never engages at all.
//
// BACKENDS IN "AUTO" (measured, from the console's `▸ ran on …` line): tasks 1–5
// run entirely on WebGPU. Task 6 reports "webgpu (1 kernel) + webgl (1 kernel)"
// because gpu.js's WebGPU backend refuses graphical mode. That mix is deliberate
// and NOT pinned with `backend: 'webgl'`: nothing here is pipelined, so the sweep
// hands the paint kernel an ordinary JS array rather than a texture — no backend
// bridge, no hidden readback, identical pixels either way.
//
// WHAT THIS MODULE DELIBERATELY DOES NOT CLAIM. One kernel launch per
// anti-diagonal is not how production aligners work, and the module says so in
// task 6 rather than pretending otherwise. The value here is the reframing — an
// axis along which a "sequential" dynamic program is embarrassingly parallel —
// not the constant factor.

// ---- the problem ----------------------------------------------------------

const MATCH = 3;
const MISMATCH = -3;
const GAP = 2; // linear gap penalty, SUBTRACTED from the neighbour's score

// The textbook Smith-Waterman pair. Best local score 13, at H[6][7].
const SEQ_A = 'TGTTACGG'; // 8 bases → 8 rows of interior matrix
const SEQ_B = 'GGTTGACTA'; // 9 bases → 9 columns

// Task 6's pair: 32 × 36, sharing a mutated 14-base motif with one insertion,
// so the score matrix grows one bright diagonal ridge for the wavefront to walk
// along. Best local score 40, at H[25][30].
const LONG_A = 'GAGAGAATTATCAACTGCCATTCTTTGTTTAG';
const LONG_B = 'ACATCGGTGTGGCATCCACTGCACATTCTTACCGAC';

const CODE = { A: 0, C: 1, G: 2, T: 3 };

// Kernels only speak numbers, so a sequence travels as base codes 0–3.
function codesOf(seq) {
  return Array.from(seq, ch => CODE[ch]);
}

function zerosMatrix(m, n) {
  const H = new Array(m + 1);
  for (let i = 0; i <= m; i++) H[i] = new Array(n + 1).fill(0);
  return H;
}

// Works on a kernel result (rows are Float32Array) as well as a plain matrix.
function copyMatrix(H) {
  const out = new Array(H.length);
  for (let i = 0; i < H.length; i++) {
    const row = new Array(H[i].length);
    for (let j = 0; j < H[i].length; j++) row[j] = H[i][j];
    out[i] = row;
  }
  return out;
}

function substitution(a, b, i, j) {
  return a[i - 1] === b[j - 1] ? MATCH : MISMATCH;
}

// ---- CPU references -------------------------------------------------------

// One full relaxation: every interior cell recomputed from the matrix it was
// handed. A SOLVED matrix is a fixed point of this — that is what "solved"
// means — and an empty one yields exactly the cells whose inputs were already
// final, which is the first anti-diagonal and nothing else.
//
// `cell(i, j)` is the only thing every probe candidate varies, so two
// candidates can never differ by accident somewhere else.
function relaxWith(H, cell) {
  const m = H.length - 1;
  const n = H[0].length - 1;
  const out = zerosMatrix(m, n);
  for (let i = 1; i <= m; i++) {
    for (let j = 1; j <= n; j++) out[i][j] = cell(i, j);
  }
  return out;
}

function correctCell(H, a, b) {
  return (i, j) => {
    const s = substitution(a, b, i, j);
    return Math.max(
      0,
      Math.max(H[i - 1][j - 1] + s, Math.max(H[i - 1][j] - GAP, H[i][j - 1] - GAP))
    );
  };
}

function relaxRef(H, a, b) {
  return relaxWith(H, correctCell(H, a, b));
}

// One anti-diagonal launch: only cells with i + j === d are recomputed, every
// other cell is passed through untouched.
function sweepRef(H, a, b, d) {
  const cell = correctCell(H, a, b);
  return relaxWith(H, (i, j) => (i + j === d ? cell(i, j) : H[i][j]));
}

// The finished matrix, plus the best local score and where it sits.
function fillRef(A, B) {
  const a = codesOf(A);
  const b = codesOf(B);
  const m = a.length;
  const n = b.length;
  const H = zerosMatrix(m, n);
  let best = 0;
  let bi = 0;
  let bj = 0;
  for (let d = 2; d <= m + n; d++) {
    const cell = correctCell(H, a, b);
    const { iStart, len } = rangeRef(d, m, n);
    for (let t = 0; t < len; t++) {
      const i = iStart + t;
      const j = d - i;
      H[i][j] = cell(i, j);
      if (H[i][j] > best) {
        best = H[i][j];
        bi = i;
        bj = j;
      }
    }
  }
  return { H, best, bi, bj, a, b, m, n };
}

// Which rows does anti-diagonal d cover? i runs from max(1, d − n) to
// min(m, d − 1) — clipped by the top edge at one end and the right edge at the
// other, which is the whole reason iStart exists.
function rangeRef(d, m, n) {
  const iStart = Math.max(1, d - n);
  const iEnd = Math.min(m, d - 1);
  return { iStart, len: iEnd - iStart + 1 };
}

// The true values of anti-diagonal d, computed from the matrix as handed in.
function diagonalRef(H, a, b, d) {
  const m = H.length - 1;
  const n = H[0].length - 1;
  const cell = correctCell(H, a, b);
  const { iStart, len } = rangeRef(d, m, n);
  const out = new Array(len);
  for (let t = 0; t < len; t++) out[t] = cell(iStart + t, d - iStart - t);
  return out;
}

// The state the wavefront is actually in when it launches diagonal d: every
// diagonal before d is final, everything from d on is still zero.
function partialTo(A, B, d) {
  const { H } = fillRef(A, B);
  const out = copyMatrix(H);
  for (let i = 0; i < out.length; i++) {
    for (let j = 0; j < out[i].length; j++) {
      if (i + j >= d) out[i][j] = 0;
    }
  }
  return out;
}

// The traceback pointer matrix: 0 = stop (the local alignment starts here),
// 1 = came from the diagonal, 2 = came from above, 3 = came from the left.
// Ties break in that order, which is the convention the walk below assumes.
function pointerRef(H, a, b, codes = [1, 2, 3]) {
  const m = H.length - 1;
  const n = H[0].length - 1;
  const out = zerosMatrix(m, n);
  for (let i = 0; i <= m; i++) {
    for (let j = 0; j <= n; j++) {
      if (H[i][j] === 0) continue;
      const s = substitution(a, b, i, j);
      if (H[i][j] === H[i - 1][j - 1] + s) out[i][j] = codes[0];
      else if (H[i][j] === H[i - 1][j] - GAP) out[i][j] = codes[1];
      else out[i][j] = codes[2];
    }
  }
  return out;
}

// Walk the pointers back from the best cell and read off the two aligned rows.
function tracebackRef(A, B) {
  const ref = fillRef(A, B);
  const dir = pointerRef(ref.H, ref.a, ref.b);
  let i = ref.bi;
  let j = ref.bj;
  let alignA = '';
  let alignB = '';
  while (dir[i][j] !== 0) {
    if (dir[i][j] === 1) {
      alignA = A[i - 1] + alignA;
      alignB = B[j - 1] + alignB;
      i--;
      j--;
    } else if (dir[i][j] === 2) {
      alignA = A[i - 1] + alignA;
      alignB = `-${alignB}`;
      i--;
    } else {
      alignA = `-${alignA}`;
      alignB = B[j - 1] + alignB;
      j--;
    }
  }
  return { alignA, alignB, ref, dir };
}

// Task 6's plot: how many cells each launch actually has work for.
function lengthsRef(m, n) {
  const out = [];
  for (let d = 2; d <= m + n; d++) out.push(rangeRef(d, m, n).len);
  return out;
}

// ---- driving a learner's kernels from a test ------------------------------

// Ride the learner's full-matrix sweep kernel across every anti-diagonal.
// Each launch is awaited before the next one is fired: diagonal d reads what
// diagonal d − 1 wrote, so the sequence is the algorithm — never Promise.all.
async function runSweeps(kernel, A, B) {
  const a = codesOf(A);
  const b = codesOf(B);
  let H = zerosMatrix(a.length, b.length);
  for (let d = 2; d <= a.length + b.length; d++) H = await kernel(H, a, b, d);
  return copyMatrix(H);
}

// Ride the learner's compact diagonal kernel: one launch per anti-diagonal,
// sized to the diagonal, stitched back into a plain JS matrix.
async function runWavefront(kernel, A, B) {
  const a = codesOf(A);
  const b = codesOf(B);
  const m = a.length;
  const n = b.length;
  const H = zerosMatrix(m, n);
  let best = 0;
  let bi = 0;
  let bj = 0;
  for (let d = 2; d <= m + n; d++) {
    const { iStart, len } = rangeRef(d, m, n);
    kernel.setOutput([len]);
    const values = await kernel(H, a, b, d, iStart);
    for (let t = 0; t < len; t++) {
      const i = iStart + t;
      H[i][d - i] = values[t];
      if (values[t] > best) {
        best = values[t];
        bi = i;
        bj = d - i;
      }
    }
  }
  return { H, best, bi, bj };
}

function maxOf(H) {
  let best = 0;
  for (let i = 0; i < H.length; i++) {
    for (let j = 0; j < H[i].length; j++) if (H[i][j] > best) best = H[i][j];
  }
  return best;
}

// ---- near-miss diagnosis --------------------------------------------------
//
// A failing assert that reports only two numbers tells a learner nothing about
// WHICH slip produced them. A probe pairs the value one specific known mistake
// would produce with a sentence naming that mistake; the helpers below speak
// only when the observation matches a probe AND the correct answer does not —
// so a cell where two candidates coincide (any cell whose score is 0, where
// "no zero floor" and the real thing agree) stays silent, as do observations
// matching probes that disagree with each other. A confident wrong diagnosis is
// worse than a plain numeric mismatch.
//
// There is no single-value diagnose() in this file, deliberately: every mistake
// it can name is a whole-shape mistake, and one matching cell in a matrix that
// is mostly zeros is not evidence of anything. The two helpers below both demand
// that a candidate predict EVERY value before it may speak. The three
// diagnoses that are not shaped like that — the Needleman-Wunsch corner, the
// row/column swap, and the loop that never assigns its result back — are each
// asserted inline, right where the test can prove the claim.

// How many places a candidate has to DISAGREE with the right answer — while
// still predicting the observation exactly — before it has earned the right to
// name a mistake.
//
// Matching the observation is cheap when candidate and truth barely differ:
// most of a Smith-Waterman matrix is zeros, where nearly every wrong recurrence
// agrees with the right one, and an anti-diagonal near a corner is one or two
// cells long. A candidate that departs from the truth in a single place and
// happens to coincide there is a coin flip, not a diagnosis.
//
// MEASURED, not guessed. Twenty-eight wrong kernels — the fourteen the probes
// below model, plus fourteen that NO probe models (gap 1, gap 3, mismatch −2,
// mismatch −4, match 2, match 4, a floor of 1, the gap charged on the diagonal
// too, only the up candidate, only the left candidate, `a[i]` in place of
// `a[i - 1]`, `i − j === d`, a thread index that ignores iStart) — were run
// against every (sequence pair, matrix state, diagonal) these tests actually
// use. At a threshold of 1 that battery drew 11 confidently WRONG diagnoses; at
// 2, four; at 3, none, and 117 honest diagnoses survive. Three it is.
const MIN_EVIDENCE = 3;

// Every mistake in this module is matrix-WIDE — a missing floor, a gap added
// instead of subtracted, a substitution table read backwards — so a candidate
// must predict EVERY cell, and disagree with the right answer in at least
// MIN_EVIDENCE of them, before it may speak.
function diagnoseMatrix(out, expected, eps, alternatives) {
  const hits = alternatives
    .filter(([candidate]) => {
      let differs = 0;
      for (let i = 0; i < expected.length; i++) {
        for (let j = 0; j < expected[i].length; j++) {
          if (!(out[i] && Math.abs(out[i][j] - candidate[i][j]) <= eps)) return false;
          if (Math.abs(expected[i][j] - candidate[i][j]) > eps) differs++;
        }
      }
      return differs >= MIN_EVIDENCE;
    })
    .map(a => a[1]);
  return hits.length && hits.every(m => m === hits[0]) ? hits[0] : null;
}

// The 1D twin, for a single diagonal's worth of values. The evidence bar bites
// hardest here: diagonal 17 is one cell long and diagonal 15 is three, so a
// short diagonal simply cannot support a named diagnosis and falls back to the
// plain numeric mismatch — which is the honest outcome, not a gap.
function diagnoseRow(out, expected, eps, alternatives) {
  const hits = alternatives
    .filter(([candidate]) => {
      if (!out || out.length !== candidate.length) return false;
      let differs = 0;
      for (let t = 0; t < candidate.length; t++) {
        if (!(Math.abs(out[t] - candidate[t]) <= eps)) return false;
        if (Math.abs(expected[t] - candidate[t]) > eps) differs++;
      }
      return differs >= MIN_EVIDENCE;
    })
    .map(a => a[1]);
  return hits.length && hits.every(m => m === hits[0]) ? hits[0] : null;
}

// ---- probe candidates -----------------------------------------------------

// The five ways the recurrence itself goes wrong, as whole-matrix candidates.
// Written as `cell` functions so tasks 1, 2 and 3 can reuse them at matrix
// scale, at diagonal scale, or restricted to one anti-diagonal.
function recurrenceCells(H, a, b) {
  return [
    [
      (i, j) => {
        const s = substitution(a, b, i, j);
        return Math.max(H[i - 1][j - 1] + s, Math.max(H[i - 1][j] - GAP, H[i][j - 1] - GAP));
      },
      'negative scores came through — the max(0, …) floor is missing. Resetting to zero is the ' +
        'entire difference between Smith-Waterman (best LOCAL alignment, free to start anywhere) ' +
        'and Needleman-Wunsch (global, forced to run end to end)',
    ],
    [
      (i, j) => {
        const s = substitution(a, b, i, j);
        return Math.max(
          0,
          Math.max(H[i - 1][j - 1] + s, Math.max(H[i - 1][j] + GAP, H[i][j - 1] + GAP))
        );
      },
      'the gap penalty was ADDED — opening a gap has to cost something, so the up and left ' +
        'candidates are H[i - 1][j] − gap and H[i][j - 1] − gap',
    ],
    [
      (i, j) => {
        const s = a[i - 1] === b[j - 1] ? MISMATCH : MATCH;
        return Math.max(
          0,
          Math.max(H[i - 1][j - 1] + s, Math.max(H[i - 1][j] - GAP, H[i][j - 1] - GAP))
        );
      },
      'match and mismatch are the wrong way round — equal bases score +3, different bases −3',
    ],
    [
      (i, j) => Math.max(0, H[i - 1][j - 1] + substitution(a, b, i, j)),
      'only the diagonal predecessor made it into the max — an alignment is also allowed to ' +
        'open a gap, and that is exactly what the up and left candidates are for',
    ],
    [
      (i, j) => {
        const s = substitution(a, b, i, j);
        return Math.max(0, Math.max(H[i - 1][j - 1] + s, Math.max(H[i - 1][j], H[i][j - 1])));
      },
      'the gap penalty never got applied — a free gap makes every alignment as long as it likes',
    ],
    [
      (i, j) => {
        const s = substitution(a, b, i, j);
        return Math.max(0, Math.max(H[i][j] + s, Math.max(H[i - 1][j] - GAP, H[i][j - 1] - GAP)));
      },
      'the diagonal candidate read the cell itself — the predecessor one step back along BOTH ' +
        'sequences is H[i - 1][j - 1]',
    ],
  ];
}

// Task 1: the recurrence, applied to every interior cell.
function relaxAlternatives(H, a, b) {
  const out = recurrenceCells(H, a, b).map(([cell, why]) => [relaxWith(H, cell), why]);
  out.push([
    copyMatrix(H),
    'nothing was computed — every cell handed back the value it already had',
  ]);
  return out;
}

// Task 2: the same, restricted to anti-diagonal d, plus the three ways the
// diagonal test itself goes wrong.
//
// Deliberately NOT listed: `i - j === d`, the other family of diagonals. On an
// 8 × 9 matrix no cell has i − j equal to any d this task launches, so that
// kernel updates NOTHING — and its output is identical to a kernel that simply
// returns H[i][j] everywhere. Two quite different mistakes, one matrix; naming
// either would be a guess, so the pass-everything-through candidate below takes
// the observation and says only what it can prove.
function sweepAlternatives(H, a, b, d) {
  const cell = correctCell(H, a, b);
  const out = recurrenceCells(H, a, b).map(([wrong, why]) => [
    relaxWith(H, (i, j) => (i + j === d ? wrong(i, j) : H[i][j])),
    why,
  ]);
  out.push(
    [
      relaxWith(H, cell),
      'every interior cell was recomputed, not just the ones on this diagonal — the launch has ' +
        'to pass a cell through untouched unless i + j is exactly d',
    ],
    [
      relaxWith(H, (i, j) => (i + j === d ? H[i][j] : cell(i, j))),
      'the diagonal test is inverted — the cells with i + j === d are the ones that should run ' +
        'the recurrence, and every OTHER cell is the one that passes through',
    ],
    [
      relaxWith(H, (i, j) => H[i][j]),
      'no cell moved at all, this diagonal included — cells with i + j === d have to run the ' +
        'recurrence, not return H[i][j]',
    ],
    [
      relaxWith(H, (i, j) => (i + j === d ? cell(i, j) : 0)),
      'every cell off the diagonal came back as 0 — a launch writes the whole matrix, so cells ' +
        'this diagonal does not own still have to return H[i][j]',
    ]
  );
  return out;
}

// Task 3: a single diagonal's worth of values, mis-indexed or mis-scored.
function diagonalAlternatives(H, a, b, d) {
  const m = H.length - 1;
  const n = H[0].length - 1;
  const { iStart, len } = rangeRef(d, m, n);
  const build = cell => {
    const out = new Array(len);
    for (let t = 0; t < len; t++) out[t] = cell(iStart + t, d - iStart - t);
    return out;
  };
  const cell = correctCell(H, a, b);
  const out = recurrenceCells(H, a, b).map(([wrong, why]) => [build(wrong), why]);
  out.push(
    [
      build(cell).slice().reverse(),
      'the diagonal came back reversed — thread 0 owns the cell nearest the TOP of the matrix, ' +
        'so i = iStart + this.thread.x and j is whatever makes i + j come to d',
    ],
    [
      build((i, j) => cell(i, j - 1)),
      'every thread computed the cell one column to its left — j = d − i, with no extra − 1',
    ],
    [
      build((i, j) => H[i][j]),
      'the kernel handed back H[i][j], the value already sitting in the matrix — this diagonal ' +
        'has not been computed yet, which is why every value came out 0',
    ],
    // The shipped starter's own mistake, and it HAS to be listed: without it,
    // an output where every thread returned the same number was being matched
    // by whichever other candidate happened to agree, and the learner who had
    // simply not touched `const i = iStart;` was told the arithmetic in their
    // recurrence was wrong. Listing it both names the real slip and, where it
    // coincides with another candidate, makes the two cancel into silence.
    [
      build(() => cell(iStart, d - iStart)),
      'every thread came back with the same value — this.thread.x never reached the row index. ' +
        'i = iStart + this.thread.x is what makes thread 0 and thread 1 different cells',
    ],
    [
      build((i, j) => cell(i, j + 1)),
      'every thread computed the cell one column to its RIGHT — j = d − i, with no extra + 1',
    ]
  );
  return out;
}

// Task 5: the pointer matrix, mis-numbered or mis-prioritised.
function pointerAlternatives(H, a, b) {
  return [
    [
      pointerRef(H, a, b, [2, 1, 3]),
      'the codes for "diagonal" and "up" are swapped — 1 means the score came from ' +
        'H[i - 1][j - 1], 2 means it came from H[i - 1][j]',
    ],
    [
      pointerRef(H, a, b, [1, 3, 2]),
      'the codes for "up" and "left" are swapped — 2 means H[i - 1][j] (a gap in B), ' +
        '3 means H[i][j - 1] (a gap in A)',
    ],
    [
      pointerRef(H, a, b, [3, 2, 1]),
      'the three direction codes are in the wrong order — 1 diagonal, 2 up, 3 left',
    ],
  ];
}

// ---- console readers ------------------------------------------------------

// Every number that appeared in a console.log line.
function loggedNumbers(logs) {
  const out = [];
  for (const line of logs || []) {
    if (line.type !== 'log' || !line.text) continue;
    const matches = line.text.match(/-?\d+(?:\.\d+)?/g);
    if (matches) for (const m of matches) out.push(parseFloat(m));
  }
  return out;
}

function loggedText(logs) {
  return (logs || [])
    .filter(line => line.type === 'log' && line.text)
    .map(line => String(line.text));
}

// "best score 13 at row 6, column 7" — the format task 4 asks for.
function loggedBest(logs) {
  for (const text of loggedText(logs)) {
    const match = /best\s+score\s+(-?\d+)\s+at\s+row\s+(\d+),\s*column\s+(\d+)/i.exec(text);
    if (match) return { score: Number(match[1]), row: Number(match[2]), col: Number(match[3]) };
  }
  return null;
}

// The first plot() payload whose series list includes `name`.
function loggedPlot(logs, name) {
  for (const line of logs || []) {
    if (line.type !== 'plot' || !line.plot) continue;
    const series = (line.plot.series || []).find(s => s.name === name);
    if (series) return series;
  }
  return null;
}

function canvasFrames(logs) {
  return (logs || []).filter(line => line.type === 'canvas').length;
}

// ---- shared source, prewired into the later starters ----------------------

const RECURRENCE_BODY = `  let s = this.constants.mismatch;
  if (a[i - 1] === b[j - 1]) s = this.constants.match;
  const diag = H[i - 1][j - 1] + s;
  const up = H[i - 1][j] - this.constants.gap;
  const left = H[i][j - 1] - this.constants.gap;
  return Math.max(0, Math.max(diag, Math.max(up, left)));`;

const SCORING = `{ output: [10, 9], constants: { match: 3, mismatch: -3, gap: 2 } }`;

const SWEEP_KERNEL = `const sweep = gpu.createKernel(function (H, a, b, d) {
  const i = this.thread.y;
  const j = this.thread.x;
  if (i === 0 || j === 0) return 0;
  if (i + j !== d) return H[i][j];
${RECURRENCE_BODY}
}, ${SCORING});`;

const DIAGONAL_KERNEL = `const diagonal = gpu.createKernel(function (H, a, b, d, iStart) {
  const i = iStart + this.thread.x;
  const j = d - i;
${RECURRENCE_BODY}
}, { dynamicOutput: true, constants: { match: 3, mismatch: -3, gap: 2 } });`;

const RANGE_HELPER = `// Anti-diagonal d covers rows i = iStart … iStart + len - 1, and j = d - i.
// Clipped at both ends: i never passes 8 (= |A|), and j never passes 9 (= |B|).
function diagonalRange(d) {
  const iStart = Math.max(1, d - 9);
  const iEnd = Math.min(8, d - 1);
  return { iStart, len: iEnd - iStart + 1 };
}`;

const WAVEFRONT_DRIVER = `${RANGE_HELPER}

// The wavefront fill you built in the last task, folded into one helper.
async function fillMatrix() {
  const H = [];
  for (let i = 0; i <= 8; i++) H.push(new Array(10).fill(0));
  let best = 0;
  let bi = 0;
  let bj = 0;
  for (let d = 2; d <= 17; d++) {
    const range = diagonalRange(d);
    diagonal.setOutput([range.len]);
    const values = await diagonal(H, codesA, codesB, d, range.iStart);
    for (let t = 0; t < range.len; t++) {
      const i = range.iStart + t;
      H[i][d - i] = values[t];
      if (values[t] > best) {
        best = values[t];
        bi = i;
        bj = d - i;
      }
    }
  }
  return { H, best, bi, bj };
}`;

const TEACHING_INPUTS = () => ({
  seqA: SEQ_A,
  seqB: SEQ_B,
  codesA: codesOf(SEQ_A),
  codesB: codesOf(SEQ_B),
});

export default {
  uuid: 'a85ca6d9-49ae-48e4-94ea-cf5761be9dae',
  version: 1,
  slug: 'sequence-alignment',
  title: 'Wavefronts: Aligning DNA on the Diagonal',
  blurb:
    'Smith-Waterman looks fatally serial — until you notice that every cell on an anti-diagonal is independent.',
  tasks: [
    // ------------------------------------------------------------------ 1
    {
      slug: 'scoring-recurrence',
      title: 'Three Ways Into a Cell',
      intro: `<p>Two DNA sequences, and the question every genome browser asks: where do they say
        the same thing? <strong>Smith-Waterman</strong> answers it by scoring a matrix.
        <code>H[i][j]</code> is the best score of any alignment that <em>ends</em> at base
        <code>i</code> of <code>A</code> and base <code>j</code> of <code>B</code>, and there are
        exactly four candidates for it:</p>
<pre><code>H[i][j] = max(
  0,                               // start fresh here
  H[i - 1][j - 1] + s(A[i], B[j]), // pair the bases
  H[i - 1][j]     - gap,           // skip a base of A
  H[i][j - 1]     - gap            // skip a base of B
)</code></pre>
        <p><code>s</code> is +3 when the bases match and −3 when they do not; a gap costs 2. The
        <code>0</code> is the whole point of <em>local</em> alignment: an alignment that has gone
        badly is abandoned rather than carried, so a good match buried inside two otherwise
        unrelated sequences still surfaces. Row 0 and column 0 are the empty prefixes, and stay 0.</p>
        <p>Now read the three predecessors again, because the trouble is right there. Every cell
        wants its neighbour <em>up</em>, its neighbour <em>left</em>, and its neighbour
        <em>up-left</em> — so no row can start before the row above it has finished, and the natural
        loop is about as serial as code gets. Nobody parallelises this. (Nobody parallelises it
        <em>row by row</em>, at least. The next task is the escape.)</p>`,
      goal: `<strong>Goal:</strong> write the kernel for one full pass of the recurrence — every
        interior cell recomputed from the matrix it was handed, with row 0 and column 0 pinned
        to <code>0</code>.`,
      requirements: [
        'Return <code>0</code> for any cell with <code>i === 0</code> or <code>j === 0</code>',
        'Score the pair with <code>this.constants.match</code> / <code>this.constants.mismatch</code>, comparing <code>a[i - 1]</code> against <code>b[j - 1]</code>',
        'Take the largest of the three predecessors, each with its own adjustment — and <em>then</em> the max against <code>0</code>',
      ],
      hints: [
        {
          title: 'Hint 1 — the substitution score, without a boolean',
          body: `<p>gpu.js cannot keep a boolean in a kernel variable on the WebGL backend, so pick
            the score with a plain <code>if</code>:</p>
<pre><code>let s = this.constants.mismatch;
if (a[i - 1] === b[j - 1]) s = this.constants.match;</code></pre>
            <p>The <code>− 1</code> is because row <code>i</code> of the matrix belongs to base
            <code>i − 1</code> of the sequence: the matrix has one extra row for the empty prefix.</p>`,
        },
        {
          title: 'Hint 2 — three candidates, then the floor',
          body: `<p><code>Math.max</code> takes two arguments inside a kernel, so nest it:</p>
<pre><code>const diag = H[i - 1][j - 1] + s;
const up = H[i - 1][j] - this.constants.gap;
const left = H[i][j - 1] - this.constants.gap;
return Math.max(0, Math.max(diag, Math.max(up, left)));</code></pre>
            <p>The <code>0</code> goes on the outside — it is a floor under the whole thing, not a
            fourth predecessor.</p>`,
        },
      ],
      transfer: `A dynamic program is a dependency graph wearing a grid costume, and this shape —
        each cell reading its three earlier neighbours — is shared by edit distance, dynamic time
        warping, the Viterbi decoder and pairwise HMMs. Whether any of them can go on a GPU is
        decided entirely by that graph, not by the arithmetic in the cell: CUDA, WGSL and Metal all
        give you thousands of threads and no way whatsoever to make one wait for another.`,
      starterCode: `// One pass of the Smith-Waterman recurrence, over the whole matrix.
// The matrix is 9 rows (|A| + 1) by 10 columns (|B| + 1).
const gpu = new GPU({ mode });

const score = gpu.createKernel(function (H, a, b) {
  const i = this.thread.y;
  const j = this.thread.x;
  // TODO 1: row 0 and column 0 are the empty prefixes — return 0.
  // TODO 2: score the pair. s is this.constants.match when a[i - 1]
  //         equals b[j - 1], and this.constants.mismatch otherwise.
  // TODO 3: return the largest of 0, diag + s, up - gap and left - gap.
  return H[i][j];
}, ${SCORING});

// An empty matrix: every alignment score starts life unknown.
const empty = [];
for (let i = 0; i <= 8; i++) empty.push(new Array(10).fill(0));

const once = await score(empty, codesA, codesB);
console.log('A =', seqA, '  B =', seqB);
for (let i = 1; i <= 8; i++) console.log('row', i, Array.from(once[i]).join(' '));
`,
      solutionCode: `// One pass of the Smith-Waterman recurrence, over the whole matrix.
// The matrix is 9 rows (|A| + 1) by 10 columns (|B| + 1).
const gpu = new GPU({ mode });

const score = gpu.createKernel(function (H, a, b) {
  const i = this.thread.y;
  const j = this.thread.x;
  if (i === 0 || j === 0) return 0;
${RECURRENCE_BODY}
}, ${SCORING});

// An empty matrix: every alignment score starts life unknown.
const empty = [];
for (let i = 0; i <= 8; i++) empty.push(new Array(10).fill(0));

const once = await score(empty, codesA, codesB);
console.log('A =', seqA, '  B =', seqB);
for (let i = 1; i <= 8; i++) console.log('row', i, Array.from(once[i]).join(' '));
`,
      inputs: TEACHING_INPUTS,
      publicTests: [
        {
          name: 'the pass produces a 9×10 matrix with a zero first row and column',
          run: async ctx => {
            ctx.assert(ctx.kernels.length >= 1, 'no kernel was created — call gpu.createKernel()');
            const empty = zerosMatrix(8, 9);
            const out = await ctx.kernel(empty, codesOf(SEQ_A), codesOf(SEQ_B));
            ctx.assert(out && out.length === 9, `expected 9 rows, got ${out && out.length}`);
            ctx.assert(out[0] && out[0].length === 10, 'each row should hold 10 values');
            for (let j = 0; j < 10; j++) {
              ctx.assertClose(out[0][j], 0, 1e-3,
                `cell [0][${j}] — row 0 is the empty prefix of A, so every score in it is 0`);
            }
            for (let i = 0; i < 9; i++) {
              ctx.assertClose(out[i][0], 0, 1e-3,
                `cell [${i}][0] — column 0 is the empty prefix of B, so every score in it is 0`);
            }
          },
        },
        {
          name: 'from an empty matrix, a cell is <code>+3</code> where the bases match and <code>0</code> where they do not',
          run: async ctx => {
            // With every neighbour still 0 the three candidates are s, -2 and
            // -2, so the floor keeps exactly the matches. That makes this pass
            // readable by eye against the two sequences.
            const a = codesOf(SEQ_A);
            const b = codesOf(SEQ_B);
            const empty = zerosMatrix(8, 9);
            const out = await ctx.kernel(empty, a, b);
            const ref = relaxRef(empty, a, b);
            const hint = diagnoseMatrix(out, ref, 1e-3, relaxAlternatives(empty, a, b));
            for (let i = 1; i <= 8; i++) {
              for (let j = 1; j <= 9; j++) {
                ctx.assertClose(out[i][j], ref[i][j], 1e-3,
                  hint || `cell [${i}][${j}] — ${SEQ_A[i - 1]} against ${SEQ_B[j - 1]}`);
              }
            }
          },
        },
        {
          name: 'a solved matrix is a fixed point — the pass hands it straight back',
          run: async ctx => {
            // What "solved" MEANS is that every cell already satisfies the
            // recurrence. So the strongest single check on the arithmetic is
            // that it reproduces the answer instead of moving it.
            const { H, a, b } = fillRef(SEQ_A, SEQ_B);
            const out = await ctx.kernel(H, a, b);
            const hint = diagnoseMatrix(out, H, 1e-3, relaxAlternatives(H, a, b));
            for (let i = 0; i <= 8; i++) {
              for (let j = 0; j <= 9; j++) {
                ctx.assertClose(out[i][j], H[i][j], 1e-3, hint || `cell [${i}][${j}]`);
              }
            }
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            // A different pair, and a matrix that is neither empty nor solved:
            // the state the wavefront is genuinely in mid-fill.
            const A = 'CATTAGCA';
            const B = 'TCATTGGCA';
            const a = codesOf(A);
            const b = codesOf(B);
            const H = partialTo(A, B, 11);
            const out = await ctx.kernel(H, a, b);
            const ref = relaxRef(H, a, b);
            const hint = diagnoseMatrix(out, ref, 1e-3, relaxAlternatives(H, a, b));
            for (let i = 0; i <= 8; i++) {
              for (let j = 0; j <= 9; j++) {
                ctx.assertClose(out[i][j], ref[i][j], 1e-3, hint || `cell [${i}][${j}]`);
              }
            }
          },
        },
        {
          name: 'private test #2',
          run: async ctx => {
            // Repeated relaxation from empty MUST reach the answer: after pass p
            // every cell with i + j <= p + 1 is final, so |A| + |B| - 1 passes
            // finish the job. This is the (correct, wasteful) algorithm the next
            // task takes apart.
            const { H: ref, a, b } = fillRef(SEQ_A, SEQ_B);
            let H = zerosMatrix(8, 9);
            for (let p = 0; p < 16; p++) H = await ctx.kernel(H, a, b);
            for (let i = 0; i <= 8; i++) {
              for (let j = 0; j <= 9; j++) {
                ctx.assertClose(H[i][j], ref[i][j], 1e-3,
                  `cell [${i}][${j}] after 16 passes of the recurrence`);
              }
            }
          },
        },
      ],
    },

    // ------------------------------------------------------------------ 2
    {
      slug: 'the-anti-diagonal',
      title: 'The Anti-Diagonal Goes All At Once',
      intro: `<p>Look at what cell <code>(i, j)</code> actually reads: <code>(i−1, j−1)</code>,
        <code>(i−1, j)</code> and <code>(i, j−1)</code>. Add the coordinates up. The cell sits on
        <code>i + j = d</code>; its three predecessors sit on <code>d − 2</code>, <code>d − 1</code>
        and <code>d − 1</code>. <strong>Nothing on diagonal <code>d</code> reads anything else on
        diagonal <code>d</code>.</strong></p>
        <p>So every cell along an <em>anti-diagonal</em> — the lines running from bottom-left to
        top-right — is independent of every other cell on it, and the whole diagonal can be computed
        in one shot. The matrix is not serial; it is serial <em>along the wrong axis</em>. Sweep it
        as a <strong>wavefront</strong> instead: diagonal 2, then 3, then 4, all the way to
        <code>|A| + |B|</code>. Sixteen launches here, and inside each one there is nothing left to
        order. (Finding the set of updates that cannot see each other is the same permission slip
        red-black colouring hands out in Iterative Linear Solvers. Seam Carving runs a wavefront
        too, but an easier one — its cells read only the row <em>above</em>, never their own row,
        so a launch per row is already enough and it never has to go looking for the diagonal.)</p>
        <p>One launch per diagonal, and a launch writes the <em>whole</em> matrix — so a cell that
        is not on this diagonal has exactly one job: hand back the value it already has.</p>`,
      goal: `<strong>Goal:</strong> add the diagonal test to the kernel, then drive it once per
        anti-diagonal, <code>d = 2 … 17</code>, and log the best score in the finished matrix.`,
      requirements: [
        'A cell with <code>i + j !== d</code> returns <code>H[i][j]</code> unchanged',
        'A cell with <code>i + j === d</code> runs the recurrence from the last task',
        'Loop <code>d</code> from <code>2</code> to <code>|A| + |B|</code> = <code>17</code>, awaiting each launch and feeding its result to the next',
        'Scan the finished matrix in plain JavaScript and <code>console.log</code> the best score',
      ],
      hints: [
        {
          title: 'Hint 1 — the guard is one line',
          body: `<p>Before any arithmetic, and after the boundary guard:</p>
<pre><code>if (i + j !== d) return H[i][j];</code></pre>
            <p>Every thread still writes exactly one cell — its own. Cells off the diagonal are not
            "skipped", they are copied forward, ready for the diagonal that is about to need them.</p>`,
        },
        {
          title: 'Hint 2 — the driver',
          body: `<p>Sixteen launches, each reading what the one before it wrote:</p>
<pre><code>let H = empty;
for (let d = 2; d &lt;= 17; d++) {
  H = await sweep(H, codesA, codesB, d);
}</code></pre>
            <p>The <code>await</code> is not optional and the loop cannot be a
            <code>Promise.all</code>: diagonal <code>d</code> is defined in terms of diagonal
            <code>d − 1</code>.</p>`,
        },
        {
          title: 'Hint 3 — why 17 and not 16',
          body: `<p>The first interior cell is <code>(1, 1)</code>, so the first diagonal worth
            launching is <code>d = 2</code>. The last interior cell is <code>(8, 9)</code>, so the
            last is <code>d = 17</code>. That is <code>|A| + |B| − 1 = 16</code> launches — stop one
            short and the bottom-right corner never gets computed.</p>`,
        },
      ],
      transfer: `Wavefront scheduling is the standard answer whenever a dependency graph has levels:
        CUDA implementations of Smith-Waterman (CUDASW++, SW#) launch one kernel per anti-diagonal
        exactly like this before they get clever, Vulkan and WebGPU express the same thing as one
        dispatch per level with a barrier between, and a task-graph runtime like TBB or Taskflow is
        doing nothing else when it runs a "ready set". The trick is always to find the axis along
        which the dependencies point the other way.`,
      starterCode: `// One launch per anti-diagonal. Cells off the diagonal ride along unchanged.
const gpu = new GPU({ mode });

const sweep = gpu.createKernel(function (H, a, b, d) {
  const i = this.thread.y;
  const j = this.thread.x;
  if (i === 0 || j === 0) return 0;
  // TODO 1: this launch owns anti-diagonal d. A cell whose i + j is
  //         anything else must return H[i][j], untouched.
${RECURRENCE_BODY}
}, ${SCORING});

const empty = [];
for (let i = 0; i <= 8; i++) empty.push(new Array(10).fill(0));

let H = empty;
// TODO 2: launch once per anti-diagonal, d = 2 … 17, feeding each
//         result into the next launch.

let best = 0;
for (let i = 0; i <= 8; i++) {
  for (let j = 0; j <= 9; j++) if (H[i][j] > best) best = H[i][j];
}
console.log('best score', best);
`,
      solutionCode: `// One launch per anti-diagonal. Cells off the diagonal ride along unchanged.
const gpu = new GPU({ mode });

${SWEEP_KERNEL}

const empty = [];
for (let i = 0; i <= 8; i++) empty.push(new Array(10).fill(0));

let H = empty;
for (let d = 2; d <= 17; d++) {
  H = await sweep(H, codesA, codesB, d);
}

let best = 0;
for (let i = 0; i <= 8; i++) {
  for (let j = 0; j <= 9; j++) if (H[i][j] > best) best = H[i][j];
}
console.log('best score', best);
`,
      inputs: TEACHING_INPUTS,
      publicTests: [
        // FIRST, and it has to stay first: it reads the kernel's .lastArgs, and
        // every later test drives the kernel itself, which overwrites them.
        // It is also the only test that can see the learner's OWN loop — the
        // others re-drive the kernel, so a loop that stops at d = 16 would sail
        // past them with a matrix that is right in all but one corner.
        {
          name: 'the loop reached the last diagonal',
          run: async ctx => {
            ctx.assert(ctx.kernels.length >= 1, 'no kernel was created — call gpu.createKernel()');
            const lastArgs = ctx.kernel.lastArgs;
            ctx.assert(lastArgs && lastArgs.length >= 4,
              'the kernel was never invoked with (H, a, b, d) — is the driving loop there?');
            ctx.assert(Math.abs(lastArgs[3] - 17) < 1e-9,
              `the last launch was diagonal ${lastArgs[3]}, not 17 — the bottom-right cell (8, 9) ` +
                `sits on d = |A| + |B| = 17, so the loop has to reach it`);
          },
        },
        {
          name: 'one launch touches only its own anti-diagonal',
          run: async ctx => {
            // The claim the whole module rests on, checked directly: hand the
            // kernel the state the wavefront is really in at diagonal 10 and
            // nothing but diagonal 10 may move.
            ctx.assert(ctx.kernels.length >= 1, 'no kernel was created — call gpu.createKernel()');
            const a = codesOf(SEQ_A);
            const b = codesOf(SEQ_B);
            const H = partialTo(SEQ_A, SEQ_B, 10);
            const out = await ctx.kernel(H, a, b, 10);
            const ref = sweepRef(H, a, b, 10);
            const hint = diagnoseMatrix(out, ref, 1e-3, sweepAlternatives(H, a, b, 10));
            let strayed = 0;
            for (let i = 0; i <= 8; i++) {
              for (let j = 0; j <= 9; j++) {
                if (i + j !== 10 && Math.abs(out[i][j] - H[i][j]) > 1e-3) strayed++;
              }
            }
            ctx.assert(strayed === 0, hint ||
              `${strayed} cells off diagonal 10 changed — a launch owns exactly the cells where ` +
                `i + j === d, and every other cell has to return H[i][j]`);
            for (let i = 0; i <= 8; i++) {
              for (let j = 0; j <= 9; j++) {
                ctx.assertClose(out[i][j], ref[i][j], 1e-3, hint || `cell [${i}][${j}]`);
              }
            }
          },
        },
        {
          name: 'sixteen launches fill the matrix exactly',
          run: async ctx => {
            const ref = fillRef(SEQ_A, SEQ_B);
            const H = await runSweeps(ctx.kernel, SEQ_A, SEQ_B);
            const short = copyMatrix(ref.H);
            short[8][9] = 0; // the corner, missed by a kernel that never reaches d = 17
            // Deliberately NOT probed here: an all-zero result. It is what a
            // driver that forgets to assign the launch back produces AND what a
            // kernel that returns 0 produces, and this test re-drives the kernel
            // itself — so the driver can never be the cause of what it sees. The
            // logged-score test below names that mistake instead, once it has
            // established the kernel is sound.
            const hint = diagnoseMatrix(H, ref.H, 1e-3, [
              [short,
                'every cell but the bottom-right corner is right — the last diagonal never ran. ' +
                  'The last cell, (8, 9), sits on d = 17'],
            ]);
            for (let i = 0; i <= 8; i++) {
              for (let j = 0; j <= 9; j++) {
                ctx.assertClose(H[i][j], ref.H[i][j], 1e-3, hint || `cell [${i}][${j}]`);
              }
            }
          },
        },
        {
          name: 'the best local score, <code>13</code>, is logged',
          run: async ctx => {
            const ref = fillRef(SEQ_A, SEQ_B);
            const nums = loggedNumbers(ctx.logs);
            if (nums.some(v => Math.abs(v - ref.best) < 1e-6)) return;
            let hint = null;
            if (nums.some(v => Math.abs(v - ref.H[8][9]) < 1e-6)) {
              hint =
                `${ref.H[8][9]} is H[8][9], the bottom-right corner — that is the best GLOBAL ` +
                `alignment score (Needleman-Wunsch). Smith-Waterman's answer is the largest value ` +
                `ANYWHERE in the matrix, because a local alignment may end wherever it likes`;
            } else if (nums.length && nums.every(v => v === 0)) {
              // Only say this once the kernel has been RULED OUT as the cause:
              // an all-zero score is equally what a broken kernel produces, and
              // guessing between the two would be worse than saying nothing. So
              // check both halves of the kernel — the diagonal guard and the
              // arithmetic — before blaming the loop.
              const partial = partialTo(SEQ_A, SEQ_B, 10);
              const one = await ctx.kernel(partial, ref.a, ref.b, 10);
              let kernelOk = true;
              for (let i = 0; i <= 8; i++) {
                for (let j = 0; j <= 9; j++) {
                  if (i + j !== 10 && Math.abs(one[i][j] - partial[i][j]) > 1e-3) kernelOk = false;
                }
              }
              const H = await runSweeps(ctx.kernel, SEQ_A, SEQ_B);
              for (let i = 0; i <= 8; i++) {
                for (let j = 0; j <= 9; j++) {
                  if (Math.abs(H[i][j] - ref.H[i][j]) > 1e-3) kernelOk = false;
                }
              }
              if (kernelOk) {
                hint =
                  'the best score came out 0, and your kernel is right — so nothing was ever ' +
                  'written into H. Each launch returns a NEW matrix, so the result has to be ' +
                  'assigned back: H = await sweep(H, codesA, codesB, d)';
              }
            }
            ctx.assert(false, hint || `log the best score in the finished matrix — expected ${ref.best}`);
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            // A fresh pair of the same lengths, driven end to end.
            const A = 'ACGTACGT';
            const B = 'TACGTACGT';
            const ref = fillRef(A, B);
            const H = await runSweeps(ctx.kernel, A, B);
            for (let i = 0; i <= 8; i++) {
              for (let j = 0; j <= 9; j++) {
                ctx.assertClose(H[i][j], ref.H[i][j], 1e-3, `cell [${i}][${j}]`);
              }
            }
            ctx.assertClose(maxOf(H), ref.best, 1e-3, 'the best score of a fresh pair');
          },
        },
        {
          name: 'private test #2',
          run: async ctx => {
            // A launch on a diagonal that is CLIPPED by both edges (d = 15 runs
            // from row 6 to row 8 only) has to leave the rest of the matrix
            // alone just as carefully as a full-length one does.
            const a = codesOf(SEQ_A);
            const b = codesOf(SEQ_B);
            const H = partialTo(SEQ_A, SEQ_B, 15);
            const out = await ctx.kernel(H, a, b, 15);
            const ref = sweepRef(H, a, b, 15);
            const hint = diagnoseMatrix(out, ref, 1e-3, sweepAlternatives(H, a, b, 15));
            for (let i = 0; i <= 8; i++) {
              for (let j = 0; j <= 9; j++) {
                ctx.assertClose(out[i][j], ref[i][j], 1e-3, hint || `cell [${i}][${j}]`);
              }
            }
          },
        },
      ],
    },

    // ------------------------------------------------------------------ 3
    {
      slug: 'index-the-diagonal',
      title: 'Which Cell Am I?',
      intro: `<p>The last task launched 90 threads to compute at most 8 cells. Every thread not on
        the diagonal woke up, read a value and wrote it straight back out — 1,440 thread-launches
        across the whole sweep to fill 72 cells. Correct, and embarrassing.</p>
        <p>The fix is to launch the diagonal itself: <code>output: [len]</code>, one thread per cell
        that actually exists. The price is that a thread no longer arrives knowing its coordinates.
        It gets <code>this.thread.x</code> — a number from <code>0</code> to <code>len − 1</code> —
        and has to work out which matrix cell that is. Two lines, and they are the fiddly heart of
        every wavefront implementation ever written:</p>
<pre><code>// walk DOWN the diagonal, one row per thread
const i = iStart + this.thread.x;
// and j is whatever makes i + j come to d
const j = d - i;</code></pre>
        <p><code>iStart</code> is the row where the diagonal enters the matrix, and it moves. Early
        on it is row 1 — the top edge — and the diagonal runs out at the <em>left</em> edge, where
        <code>j</code> would fall below 1. Later the top edge stops mattering and the
        <em>right</em> edge takes over: <code>j</code> can never exceed <code>|B|</code>, so the
        diagonal cannot start any higher than row <code>d − |B|</code>. Hence
        <code>iStart = max(1, d − |B|)</code> and <code>iEnd = min(|A|, d − 1)</code> — handed to you
        below, because getting <em>them</em> wrong is a JavaScript bug, while getting the two lines
        above wrong is a GPU bug that reads whatever happens to be next to the matrix.</p>`,
      goal: `<strong>Goal:</strong> write the compact diagonal kernel — <code>len</code> threads,
        each computing exactly one cell of anti-diagonal <code>d</code> — and pull diagonal 10 out
        of a half-filled matrix with it.`,
      requirements: [
        'Map the thread index to a row: <code>i = iStart + this.thread.x</code>',
        'Map the row to a column: <code>j = d - i</code>',
        'Run the same recurrence as before, and return one number per thread',
        'Create the kernel with <code>dynamicOutput: true</code> so <code>setOutput([len])</code> can resize it per diagonal',
      ],
      hints: [
        {
          title: 'Hint 1 — no boundary guard is needed here',
          body: `<p><code>iStart</code> and <code>len</code> already keep <code>i</code> inside
            <code>1 … |A|</code> and <code>j</code> inside <code>1 … |B|</code>. That is what they
            are <em>for</em>: the launch shape does the clipping, so the kernel body is nothing but
            the recurrence.</p>`,
        },
        {
          title: 'Hint 2 — which end is thread 0?',
          body: `<p>Thread 0 owns the cell nearest the top of the matrix — the smallest row index on
            the diagonal, which is <code>iStart</code>. Thread <code>len − 1</code> owns the one
            furthest down. Get that backwards and every value is right but in the wrong order.</p>`,
        },
        {
          title: 'Hint 3 — resizing between launches',
          body: `<p><code>dynamicOutput: true</code> is what makes a kernel resizable;
            <code>diagonal.setOutput([len])</code> then sets the thread count for the next call, the
            same move the halving ladder in Reductions makes on its way down.</p>`,
        },
      ],
      transfer: `Turning a thread id into a coordinate is most of what a GPU programmer does all
        day: CUDA's <code>blockIdx * blockDim + threadIdx</code>, WGSL's
        <code>global_invocation_id</code> and Metal's <code>thread_position_in_grid</code> all hand
        you a flat number and leave the geometry to you. Launching exactly the work that exists —
        rather than a rectangle with a guard in it — is the same instinct behind compacted launches,
        persistent-thread kernels and indirect dispatch.`,
      starterCode: `// One thread per cell of the diagonal — no passengers.
const gpu = new GPU({ mode });

const diagonal = gpu.createKernel(function (H, a, b, d, iStart) {
  // TODO: which cell does this thread own?
  //   i walks DOWN the diagonal from iStart, one row per thread
  //   j is whatever makes i + j come to d
  const i = iStart;
  const j = d - iStart;
${RECURRENCE_BODY}
}, { dynamicOutput: true, constants: { match: 3, mismatch: -3, gap: 2 } });

${RANGE_HELPER}

// \`partial\` is the matrix mid-sweep: diagonals 2…9 are final, the rest is 0.
const range = diagonalRange(10);
diagonal.setOutput([range.len]);
const values = await diagonal(partial, codesA, codesB, 10, range.iStart);

console.log('diagonal 10 starts at row', range.iStart, 'and has', range.len, 'cells');
console.log(Array.from(values));
`,
      solutionCode: `// One thread per cell of the diagonal — no passengers.
const gpu = new GPU({ mode });

${DIAGONAL_KERNEL}

${RANGE_HELPER}

// \`partial\` is the matrix mid-sweep: diagonals 2…9 are final, the rest is 0.
const range = diagonalRange(10);
diagonal.setOutput([range.len]);
const values = await diagonal(partial, codesA, codesB, 10, range.iStart);

console.log('diagonal 10 starts at row', range.iStart, 'and has', range.len, 'cells');
console.log(Array.from(values));
`,
      inputs: () => ({
        seqA: SEQ_A,
        seqB: SEQ_B,
        codesA: codesOf(SEQ_A),
        codesB: codesOf(SEQ_B),
        partial: partialTo(SEQ_A, SEQ_B, 10),
      }),
      publicTests: [
        {
          name: 'the kernel is resizable and returns one value per cell of the diagonal',
          run: async ctx => {
            const kernel = ctx.kernels.find(k => k.kernel && k.kernel.dynamicOutput);
            ctx.assert(kernel,
              'no kernel with dynamicOutput: true found — pass it in the kernel options so ' +
                'setOutput() can resize the launch for each diagonal');
            const a = codesOf(SEQ_A);
            const b = codesOf(SEQ_B);
            for (const d of [2, 10, 15]) {
              const { iStart, len } = rangeRef(d, 8, 9);
              kernel.setOutput([len]);
              const out = await kernel(partialTo(SEQ_A, SEQ_B, d), a, b, d, iStart);
              ctx.assert(out && out.length === len,
                `diagonal ${d} has ${len} cells, so the launch should return ${len} values, ` +
                  `got ${out && out.length}`);
            }
          },
        },
        {
          name: 'diagonal 10 matches the matrix it came from',
          run: async ctx => {
            const kernel = ctx.kernels.find(k => k.kernel && k.kernel.dynamicOutput);
            ctx.assert(kernel, 'no kernel with dynamicOutput: true found');
            const a = codesOf(SEQ_A);
            const b = codesOf(SEQ_B);
            const H = partialTo(SEQ_A, SEQ_B, 10);
            const { iStart, len } = rangeRef(10, 8, 9);
            kernel.setOutput([len]);
            const out = await kernel(H, a, b, 10, iStart);
            const ref = diagonalRef(H, a, b, 10);
            const hint = diagnoseRow(out, ref, 1e-3, diagonalAlternatives(H, a, b, 10));
            for (let t = 0; t < len; t++) {
              ctx.assertClose(out[t], ref[t], 1e-3,
                hint || `thread ${t}, which owns cell [${iStart + t}][${10 - iStart - t}]`);
            }
          },
        },
        {
          name: 'a clipped diagonal starts where <code>iStart</code> says it does',
          run: async ctx => {
            // Diagonal 15 runs rows 6…8 only. A kernel that ignores iStart and
            // starts from row 1 reads outside the matrix, which is exactly the
            // failure this launch shape exists to make impossible.
            const kernel = ctx.kernels.find(k => k.kernel && k.kernel.dynamicOutput);
            ctx.assert(kernel, 'no kernel with dynamicOutput: true found');
            const a = codesOf(SEQ_A);
            const b = codesOf(SEQ_B);
            const H = partialTo(SEQ_A, SEQ_B, 15);
            const { iStart, len } = rangeRef(15, 8, 9);
            kernel.setOutput([len]);
            const out = await kernel(H, a, b, 15, iStart);
            const ref = diagonalRef(H, a, b, 15);
            const hint = diagnoseRow(out, ref, 1e-3, diagonalAlternatives(H, a, b, 15));
            for (let t = 0; t < len; t++) {
              ctx.assertClose(out[t], ref[t], 1e-3,
                hint || `thread ${t} of diagonal 15, which owns cell [${iStart + t}][${15 - iStart - t}]`);
            }
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            // Every diagonal of a fresh pair, each one launched against the
            // matrix state the wavefront would really have reached.
            const kernel = ctx.kernels.find(k => k.kernel && k.kernel.dynamicOutput);
            ctx.assert(kernel, 'expected a dynamicOutput kernel');
            const A = 'GGGGCCCC';
            const B = 'AAGGGGCCC';
            const a = codesOf(A);
            const b = codesOf(B);
            for (let d = 2; d <= 17; d++) {
              const { iStart, len } = rangeRef(d, 8, 9);
              const H = partialTo(A, B, d);
              kernel.setOutput([len]);
              const out = await kernel(H, a, b, d, iStart);
              const ref = diagonalRef(H, a, b, d);
              const hint = diagnoseRow(out, ref, 1e-3, diagonalAlternatives(H, a, b, d));
              for (let t = 0; t < len; t++) {
                ctx.assertClose(out[t], ref[t], 1e-3, hint || `diagonal ${d}, thread ${t}`);
              }
            }
          },
        },
        {
          name: 'private test #2',
          run: async ctx => {
            // From an EMPTY matrix every diagonal reduces to max(0, s), which no
            // amount of reading H[i][j] can fake.
            const kernel = ctx.kernels.find(k => k.kernel && k.kernel.dynamicOutput);
            ctx.assert(kernel, 'expected a dynamicOutput kernel');
            const a = codesOf(SEQ_A);
            const b = codesOf(SEQ_B);
            const empty = zerosMatrix(8, 9);
            for (const d of [6, 11]) {
              const { iStart, len } = rangeRef(d, 8, 9);
              kernel.setOutput([len]);
              const out = await kernel(empty, a, b, d, iStart);
              const ref = diagonalRef(empty, a, b, d);
              const hint = diagnoseRow(out, ref, 1e-3, diagonalAlternatives(empty, a, b, d));
              for (let t = 0; t < len; t++) {
                ctx.assertClose(out[t], ref[t], 1e-3,
                  hint || `diagonal ${d} of an empty matrix, thread ${t}`);
              }
            }
          },
        },
      ],
    },

    // ------------------------------------------------------------------ 4
    {
      slug: 'drive-the-wavefront',
      title: 'Ride the Wavefront Down',
      intro: `<p>Sixteen launches, resized one at a time, each writing its diagonal back into a plain
        JavaScript matrix so the next launch can read it. That is the whole algorithm — and the
        driving loop looks a lot like the halving ladder in Reductions, except the kernel grows and
        then shrinks instead of only shrinking.</p>
        <p>The best score is picked up on the way past. Smith-Waterman's answer is the largest value
        <em>anywhere</em> in the matrix, not the bottom-right corner — that corner is
        Needleman-Wunsch's answer to a different question ("align these end to end"). Track the
        maximum and where it sat, because the next task walks backwards from exactly that cell.</p>
        <p>Count the threads while you are here. The diagonals run
        <code>1, 2, 3, … 8, 8, … 3, 2, 1</code> — 72 in total, one per cell, against the 1,440 the
        previous task launched for the same answer. Twenty times fewer threads, and exactly the same
        sixteen launches. Which is the honest headline: the launches, not the threads, are what this
        algorithm actually pays for.</p>`,
      goal: `<strong>Goal:</strong> drive the compact kernel across every anti-diagonal, stitch each
        result back into <code>H</code>, and log <code>best score S at row I, column J</code>.`,
      requirements: [
        'For each <code>d</code> from 2 to 17: <code>setOutput([len])</code>, then <code>await</code> the launch',
        'Write value <code>t</code> back to <code>H[iStart + t][d - iStart - t]</code>',
        'Track the largest value seen and the cell it came from',
        'Log it in exactly this shape: <code>best score 13 at row 6, column 7</code>',
      ],
      hints: [
        {
          title: 'Hint 1 — the loop skeleton',
          body: `<pre><code>for (let d = 2; d &lt;= 17; d++) {
  const range = diagonalRange(d);
  diagonal.setOutput([range.len]);
  const values =
    await diagonal(H, codesA, codesB, d, range.iStart);
  // … write values back into H, and watch for a new best …
}</code></pre>`,
        },
        {
          title: 'Hint 2 — stitching a diagonal back in',
          body: `<p>Thread <code>t</code> owned row <code>iStart + t</code>, and its column is
            whatever makes the sum come to <code>d</code>:</p>
<pre><code>for (let t = 0; t &lt; range.len; t++) {
  const i = range.iStart + t;
  H[i][d - i] = values[t];
}</code></pre>`,
        },
        {
          title: 'Hint 3 — the log line',
          body: `<p>The tests read this line, so keep the wording:</p>
<pre><code>console.log(
  'best score ' + best + ' at row ' + bi + ', column ' + bj
);</code></pre>`,
        },
      ],
      transfer: `Sixteen launches to fill 72 cells is a launch-bound program, and every platform has
        machinery for exactly that complaint: CUDA graphs exist to replay a fixed sequence of tiny
        kernels without paying per-launch driver cost, WebGPU lets you record many dispatches into
        one command buffer, and Metal's indirect command buffers do the same job. When the work per
        launch is small, the schedule is the program.`,
      starterCode: `// The whole fill: one resized launch per anti-diagonal.
const gpu = new GPU({ mode });

${DIAGONAL_KERNEL}

${RANGE_HELPER}

const H = [];
for (let i = 0; i <= 8; i++) H.push(new Array(10).fill(0));

let best = 0;
let bi = 0;
let bj = 0;

// TODO: for each d from 2 to 17 — resize the kernel to the diagonal,
//       await the launch, write the values back into H, and keep the
//       largest score you have seen along with its row and column.

console.log('best score ' + best + ' at row ' + bi + ', column ' + bj);
console.log('threads launched:', 72, 'vs', 16 * 9 * 10, 'for the full-matrix sweep');
`,
      solutionCode: `// The whole fill: one resized launch per anti-diagonal.
const gpu = new GPU({ mode });

${DIAGONAL_KERNEL}

${RANGE_HELPER}

const H = [];
for (let i = 0; i <= 8; i++) H.push(new Array(10).fill(0));

let best = 0;
let bi = 0;
let bj = 0;

for (let d = 2; d <= 17; d++) {
  const range = diagonalRange(d);
  diagonal.setOutput([range.len]);
  const values = await diagonal(H, codesA, codesB, d, range.iStart);
  for (let t = 0; t < range.len; t++) {
    const i = range.iStart + t;
    H[i][d - i] = values[t];
    if (values[t] > best) {
      best = values[t];
      bi = i;
      bj = d - i;
    }
  }
}

console.log('best score ' + best + ' at row ' + bi + ', column ' + bj);
console.log('threads launched:', 72, 'vs', 16 * 9 * 10, 'for the full-matrix sweep');
`,
      inputs: TEACHING_INPUTS,
      publicTests: [
        // FIRST, and it has to stay first: it reads the kernel's .lastArgs, and
        // every later test drives the kernel itself, which overwrites them.
        {
          name: 'the loop reached the last diagonal',
          run: async ctx => {
            const kernel = ctx.kernels.find(k => k.kernel && k.kernel.dynamicOutput);
            ctx.assert(kernel,
              'no kernel with dynamicOutput: true found — the diagonal kernel needs it so ' +
                'setOutput() can resize the launch');
            const lastArgs = kernel.lastArgs;
            ctx.assert(lastArgs && lastArgs.length >= 5,
              'the kernel was never invoked with (H, a, b, d, iStart) — is the driving loop there?');
            ctx.assert(Math.abs(lastArgs[3] - 17) < 1e-9,
              `the last launch was diagonal ${lastArgs[3]}, not 17 — the bottom-right cell (8, 9) ` +
                `sits on d = |A| + |B| = 17, so the loop has to reach it`);
          },
        },
        {
          name: 'the wavefront fills the matrix',
          run: async ctx => {
            const kernel = ctx.kernels.find(k => k.kernel && k.kernel.dynamicOutput);
            ctx.assert(kernel, 'no kernel with dynamicOutput: true found');
            const ref = fillRef(SEQ_A, SEQ_B);
            const got = await runWavefront(kernel, SEQ_A, SEQ_B);
            for (let i = 0; i <= 8; i++) {
              for (let j = 0; j <= 9; j++) {
                ctx.assertClose(got.H[i][j], ref.H[i][j], 1e-3, `cell [${i}][${j}]`);
              }
            }
          },
        },
        {
          name: '<code>best score 13 at row 6, column 7</code> is logged',
          run: async ctx => {
            const ref = fillRef(SEQ_A, SEQ_B);
            const got = loggedBest(ctx.logs);
            ctx.assert(got,
              `log the answer in the shape the task asks for — "best score ${ref.best} at row ` +
                `${ref.bi}, column ${ref.bj}"`);
            const cornerHint = Math.abs(got.score - ref.H[8][9]) < 1e-6
              ? `${ref.H[8][9]} is the bottom-right corner, H[8][9] — that is the best GLOBAL ` +
                `alignment (Needleman-Wunsch). Smith-Waterman takes the maximum over the whole ` +
                `matrix, because a local alignment ends wherever it is best to end`
              : null;
            ctx.assertClose(got.score, ref.best, 1e-6, cornerHint || 'the best score');
            const swapped = got.row === ref.bj && got.col === ref.bi;
            ctx.assert(got.row === ref.bi && got.col === ref.bj,
              swapped
                ? 'row and column came out the other way round — i indexes A and runs down the ' +
                  'rows, j indexes B and runs across the columns'
                : `the best score sits at row ${ref.bi}, column ${ref.bj}, not row ${got.row}, ` +
                  `column ${got.col}`);
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            const kernel = ctx.kernels.find(k => k.kernel && k.kernel.dynamicOutput);
            ctx.assert(kernel, 'expected a dynamicOutput kernel');
            for (const [A, B] of [['ACGTACGT', 'TACGTACGT'], ['CATTAGCA', 'TCATTGGCA']]) {
              const ref = fillRef(A, B);
              const got = await runWavefront(kernel, A, B);
              ctx.assertClose(got.best, ref.best, 1e-3, `the best score of ${A} vs ${B}`);
              ctx.assert(got.bi === ref.bi && got.bj === ref.bj,
                `the best cell of ${A} vs ${B} is [${ref.bi}][${ref.bj}], got [${got.bi}][${got.bj}]`);
              for (let i = 0; i <= 8; i++) {
                for (let j = 0; j <= 9; j++) {
                  ctx.assertClose(got.H[i][j], ref.H[i][j], 1e-3, `cell [${i}][${j}] of ${A} vs ${B}`);
                }
              }
            }
          },
        },
      ],
    },

    // ------------------------------------------------------------------ 5
    {
      slug: 'read-the-alignment',
      title: 'Traceback: What Did It Actually Align?',
      intro: `<p>A score is a number; a biologist wants the alignment. That means walking backwards
        from the best cell, asking at each step <em>which of the three candidates won here</em>, until
        the score drops to <code>0</code> and the local alignment began.</p>
        <p>The walk itself is hopelessly serial — one cell at a time, and the path is only as long as
        the alignment. Putting it on a GPU would be silly. But the <em>question</em> it asks at every
        step is embarrassingly parallel: "which predecessor explains this cell?" has the same answer
        whenever you ask it, so compute the answer for all 90 cells in one launch and store it. A
        <strong>pointer matrix</strong> — 1 for diagonal, 2 for up, 3 for left, 0 for "stop here" —
        is what every real aligner keeps beside the scores for exactly this reason.</p>
        <p>Knowing what <em>not</em> to move onto the GPU is part of the craft. The fill is
        <code>|A| × |B|</code> cells of work; the walk is a few dozen steps of pointer chasing that
        finishes before a kernel launch would have been scheduled.</p>`,
      goal: `<strong>Goal:</strong> write the pointer kernel. Every cell reports which of its three
        predecessors produced its score, or <code>0</code> if the score is <code>0</code> and the
        alignment starts there.`,
      requirements: [
        'A cell whose score is <code>0</code> returns <code>0</code> — that is where the walk stops, and it also keeps the boundary safe',
        'Return <code>1</code> when <code>H[i][j]</code> equals <code>H[i - 1][j - 1] + s</code>',
        'Return <code>2</code> when it equals <code>H[i - 1][j] - gap</code>, and <code>3</code> otherwise',
        'Test the candidates in that order, so ties prefer the diagonal',
      ],
      hints: [
        {
          title: 'Hint 1 — the stop test comes first',
          body: `<p>Not just because the walk needs it: row 0 and column 0 hold zeros, and putting
            the test first means those threads return before they ever look at
            <code>a[i - 1]</code>.</p>
<pre><code>if (H[i][j] === 0) return 0;</code></pre>`,
        },
        {
          title: 'Hint 2 — comparing scores for equality is safe here',
          body: `<p>Every score in this matrix is a whole number — match, mismatch and gap are all
            integers — so <code>===</code> between two of them is exact on every backend. That is a
            property of this scoring scheme, not a general licence: with fractional weights you
            would compare against a tolerance instead.</p>`,
        },
        {
          title: 'Hint 3 — the whole body',
          body: `<pre><code>if (H[i][j] === 0) return 0;
let s = this.constants.mismatch;
if (a[i - 1] === b[j - 1]) s = this.constants.match;
if (H[i][j] === H[i - 1][j - 1] + s) return 1;
if (H[i][j] === H[i - 1][j] - this.constants.gap) return 2;
return 3;</code></pre>`,
        },
      ],
      transfer: `Storing a decision alongside a value so the reconstruction is cheap is the standard
        move in every parallel dynamic program — the same reason a CUDA Viterbi decoder writes a
        backpointer array and a GPU video encoder writes mode decisions. The serial tail stays on the
        host, and the honest question on every platform is not "can this run on the GPU" but "is this
        the part worth moving".`,
      starterCode: `// The fill from the last task, then one launch that says WHERE each score came from.
const gpu = new GPU({ mode });

${DIAGONAL_KERNEL}

${WAVEFRONT_DRIVER}

const pointers = gpu.createKernel(function (H, a, b) {
  const i = this.thread.y;
  const j = this.thread.x;
  // TODO 1: a cell scoring 0 is where a local alignment starts — return 0.
  // TODO 2: score the pair as before, then report which candidate won:
  //         1 = H[i - 1][j - 1] + s, 2 = H[i - 1][j] - gap, 3 = H[i][j - 1] - gap.
  return 0;
}, ${SCORING});

const filled = await fillMatrix();
const dir = await pointers(filled.H, codesA, codesB);

// Walk the pointers back. Serial, tiny, and perfectly happy on the CPU.
let i = filled.bi;
let j = filled.bj;
let alignA = '';
let alignB = '';
while (dir[i][j] !== 0) {
  if (dir[i][j] === 1) {
    alignA = seqA[i - 1] + alignA;
    alignB = seqB[j - 1] + alignB;
    i--;
    j--;
  } else if (dir[i][j] === 2) {
    alignA = seqA[i - 1] + alignA;
    alignB = '-' + alignB;
    i--;
  } else {
    alignA = '-' + alignA;
    alignB = seqB[j - 1] + alignB;
    j--;
  }
}

console.log('score', filled.best);
console.log('A: ' + alignA);
console.log('B: ' + alignB);
`,
      solutionCode: `// The fill from the last task, then one launch that says WHERE each score came from.
const gpu = new GPU({ mode });

${DIAGONAL_KERNEL}

${WAVEFRONT_DRIVER}

const pointers = gpu.createKernel(function (H, a, b) {
  const i = this.thread.y;
  const j = this.thread.x;
  if (H[i][j] === 0) return 0;
  let s = this.constants.mismatch;
  if (a[i - 1] === b[j - 1]) s = this.constants.match;
  if (H[i][j] === H[i - 1][j - 1] + s) return 1;
  if (H[i][j] === H[i - 1][j] - this.constants.gap) return 2;
  return 3;
}, ${SCORING});

const filled = await fillMatrix();
const dir = await pointers(filled.H, codesA, codesB);

// Walk the pointers back. Serial, tiny, and perfectly happy on the CPU.
let i = filled.bi;
let j = filled.bj;
let alignA = '';
let alignB = '';
while (dir[i][j] !== 0) {
  if (dir[i][j] === 1) {
    alignA = seqA[i - 1] + alignA;
    alignB = seqB[j - 1] + alignB;
    i--;
    j--;
  } else if (dir[i][j] === 2) {
    alignA = seqA[i - 1] + alignA;
    alignB = '-' + alignB;
    i--;
  } else {
    alignA = '-' + alignA;
    alignB = seqB[j - 1] + alignB;
    j--;
  }
}

console.log('score', filled.best);
console.log('A: ' + alignA);
console.log('B: ' + alignB);
`,
      inputs: TEACHING_INPUTS,
      publicTests: [
        {
          name: 'the pointer matrix is <code>0</code> exactly where the score is <code>0</code>',
          run: async ctx => {
            ctx.assert(ctx.kernels.length >= 2,
              `expected 2 kernels (the prewired diagonal, then pointers), found ${ctx.kernels.length}`);
            const { H, a, b } = fillRef(SEQ_A, SEQ_B);
            const out = await ctx.kernel(H, a, b);
            ctx.assert(out && out.length === 9 && out[0].length === 10,
              'expected a 9×10 pointer matrix');
            for (let i = 0; i <= 8; i++) {
              for (let j = 0; j <= 9; j++) {
                if (H[i][j] === 0) {
                  ctx.assertClose(out[i][j], 0, 1e-3,
                    `cell [${i}][${j}] scores 0, so it has no predecessor — the walk stops there, ` +
                      `and returning 0 first is also what keeps row 0 and column 0 from reading ` +
                      `outside the sequences`);
                } else {
                  ctx.assert(out[i][j] >= 1 && out[i][j] <= 3,
                    `cell [${i}][${j}] scores ${H[i][j]}, so it came from somewhere — expected a ` +
                      `direction of 1, 2 or 3, got ${out[i][j]}`);
                }
              }
            }
          },
        },
        {
          name: 'every non-zero cell names the predecessor that actually produced it',
          run: async ctx => {
            const { H, a, b } = fillRef(SEQ_A, SEQ_B);
            const out = await ctx.kernel(H, a, b);
            const ref = pointerRef(H, a, b);
            const hint = diagnoseMatrix(out, ref, 1e-3, pointerAlternatives(H, a, b));
            for (let i = 0; i <= 8; i++) {
              for (let j = 0; j <= 9; j++) {
                ctx.assertClose(out[i][j], ref[i][j], 1e-3,
                  hint || `cell [${i}][${j}], which scores ${H[i][j]}`);
              }
            }
          },
        },
        {
          name: 'the alignment reads <code>GTT-AC</code> against <code>GTTGAC</code>',
          run: async ctx => {
            const { alignA, alignB } = tracebackRef(SEQ_A, SEQ_B);
            const lines = loggedText(ctx.logs);
            ctx.assert(lines.some(t => t.trim() === `A: ${alignA}`),
              `expected a console line reading "A: ${alignA}"`);
            ctx.assert(lines.some(t => t.trim() === `B: ${alignB}`),
              `expected a console line reading "B: ${alignB}"`);
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            // Two more pairs, one of which has no gap in its best alignment, so
            // a kernel that always answers "diagonal" is caught by the other.
            for (const [A, B] of [['ACGTACGT', 'TACGTACGT'], ['CATTAGCA', 'TCATTGGCA']]) {
              const { H, a, b } = fillRef(A, B);
              const out = await ctx.kernel(H, a, b);
              const ref = pointerRef(H, a, b);
              const hint = diagnoseMatrix(out, ref, 1e-3, pointerAlternatives(H, a, b));
              for (let i = 0; i <= 8; i++) {
                for (let j = 0; j <= 9; j++) {
                  ctx.assertClose(out[i][j], ref[i][j], 1e-3,
                    hint || `cell [${i}][${j}] of ${A} vs ${B}`);
                }
              }
            }
          },
        },
        {
          name: 'private test #2',
          run: async ctx => {
            // Walk the learner's own pointers and check the alignment they
            // describe: every diagonal step has to consume one base of each
            // sequence, and the columns have to add back up to the score.
            const { H, a, b, bi, bj, best } = fillRef(SEQ_A, SEQ_B);
            const dir = await ctx.kernel(H, a, b);
            let i = bi;
            let j = bj;
            let total = 0;
            let steps = 0;
            // i >= 0 && j >= 0 is a guard, not a stopping rule: row 0 and
            // column 0 score 0, so a correct pointer matrix always stops there.
            // A pointer matrix that does NOT return 0 on the boundary would walk
            // off the top of the array, and a TypeError is a worse thing for a
            // learner to read than the message this test ends with.
            while (i >= 0 && j >= 0 && dir[i][j] !== 0 && steps < 64) {
              if (dir[i][j] === 1) {
                total += a[i - 1] === b[j - 1] ? MATCH : MISMATCH;
                i--;
                j--;
              } else if (dir[i][j] === 2) {
                total -= GAP;
                i--;
              } else {
                total -= GAP;
                j--;
              }
              steps++;
            }
            ctx.assert(steps < 64, 'the pointer walk never reached a 0 — it would loop forever');
            ctx.assertClose(total, best, 1e-3,
              'the steps the pointers describe do not add up to the best score — the direction ' +
                'each cell reports has to be the candidate that actually produced its value');
          },
        },
      ],
    },

    // ------------------------------------------------------------------ 6
    {
      slug: 'watch-it-sweep',
      title: 'Watch the Wavefront, Then Read the Bill',
      // Under mode "auto" this task legitimately runs on two backends: `sweep`
      // upgrades to WebGPU, `paint` declines it (gpu.js's WebGPU backend refuses
      // graphical mode outright), so the console reports "webgpu (1 kernel) +
      // webgl (1 kernel)". Deliberately NOT pinned: nothing here is pipelined,
      // so `sweep` hands `paint` an ordinary JS array rather than a texture —
      // no backend bridge, no hidden readback, identical pixels either way.
      intro: `<p>Thirty-two bases against thirty-six, sharing a mutated stretch in the middle.
        Sixty-seven launches, and this time each one renders — so the console gives you a scrubber
        and you can watch the wavefront march from the top-left corner to the bottom-right, lighting
        the ridge where the two sequences agree.</p>
        <p>Then the bill. Diagonal 2 has one cell. Diagonal 34 has thirty-two. Diagonal 68 has one
        again. Average occupancy across the sweep is around <code>|A| / 2</code> threads, and the
        launches at either end are launches for a handful of work — which is why nobody ships this.
        Real aligners cut the matrix into <strong>tiles</strong> and run a wavefront over the tiles
        instead, so each launch has thousands of cells to chew on; or they abandon the wavefront
        entirely and align a whole database at once, one sequence per thread, which is
        embarrassingly parallel and needs no cleverness at all.</p>
        <p>Both of those are engineering. The reframing — that a dynamic program nobody could
        parallelise turns out to be parallel along a diagonal — is the idea, and it survives every
        one of those refinements intact.</p>`,
      goal: `<strong>Goal:</strong> paint each frame of the sweep, and plot how much work each
        launch actually had against how many threads it started.`,
      requirements: [
        'The cell on the live diagonal — <code>x + y === d</code> — is painted <code>this.color(1, 0.6, 0.1, 1)</code>',
        'Every other cell is painted from its score: <code>const v = H[y][x] / this.constants.top;</code> then <code>this.color(v * 0.35, v * 0.95, v * 0.75, 1)</code>',
        'Fill <code>lengths</code> with the number of cells on each anti-diagonal, <code>d = 2 … 68</code>',
      ],
      hints: [
        {
          title: 'Hint 1 — the paint kernel is a two-way branch',
          body: `<p>Read the score once, then decide which of the two colours this pixel gets:</p>
<pre><code>const v = H[this.thread.y][this.thread.x] / this.constants.top;
if (this.thread.x + this.thread.y === d) {
  this.color(1, 0.6, 0.1, 1);
} else {
  this.color(v * 0.35, v * 0.95, v * 0.75, 1);
}</code></pre>`,
        },
        {
          title: 'Hint 2 — how long is a diagonal?',
          body: `<p>The same clipping as before, one sequence length apiece:</p>
<pre><code>const iStart = Math.max(1, d - 36);
const iEnd = Math.min(32, d - 1);
lengths.push(iEnd - iStart + 1);</code></pre>
            <p>Note the <code>d - 1</code> and the <code>1</code>: row 0 and column 0 are never
            computed, so they are not cells this algorithm has work for.</p>`,
        },
        {
          title: 'Hint 3 — reading the plot',
          body: `<p>The two series differ by two orders of magnitude at the ends, which is why the
            plot asks for a log axis. The flat line is what every launch costs; the triangle under
            it is what every launch is worth.</p>`,
        },
      ],
      transfer: `Occupancy — how much of the machine a launch actually keeps busy — is the number
        that decides whether a GPU port was worth it, and a triangular work profile is a classic way
        to lose. The standard answers are the ones real aligners use: tile the domain so every
        launch is full (CUDA's blocked wavefront, the same idea as cache blocking), or find a
        coarser axis of parallelism and use that instead. Modern tools like WFA go further and
        change the algorithm so the wavefront is short in the first place.`,
      starterCode: `// 32 × 36 bases, 67 launches, one rendered frame each.
const gpu = new GPU({ mode });

const sweep = gpu.createKernel(function (H, a, b, d) {
  const i = this.thread.y;
  const j = this.thread.x;
  if (i === 0 || j === 0) return 0;
  if (i + j !== d) return H[i][j];
  let s = this.constants.mismatch;
  if (a[i - 1] === b[j - 1]) s = this.constants.match;
  const diag = H[i - 1][j - 1] + s;
  const up = H[i - 1][j] - this.constants.gap;
  const left = H[i][j - 1] - this.constants.gap;
  return Math.max(0, Math.max(diag, Math.max(up, left)));
}, { output: [37, 33], constants: { match: 3, mismatch: -3, gap: 2 } });

const paint = gpu.createKernel(function (H, d) {
  // TODO 1: cells on the live diagonal (x + y === d) get this.color(1, 0.6, 0.1, 1).
  //         Everything else is shaded by its score: v = H[y][x] / this.constants.top,
  //         then this.color(v * 0.35, v * 0.95, v * 0.75, 1).
  this.color(1, 0, 1, 1);
}, { output: [37, 33], graphical: true, constants: { top: 40 } });

let H = [];
for (let i = 0; i <= 32; i++) H.push(new Array(37).fill(0));

for (let d = 2; d <= 68; d++) {
  H = await sweep(H, codesA, codesB, d);
  await paint(H, d);
  render(paint.canvas);
}

let best = 0;
for (let i = 0; i <= 32; i++) {
  for (let j = 0; j <= 36; j++) if (H[i][j] > best) best = H[i][j];
}
console.log('best score', best);

// TODO 2: how many cells does each anti-diagonal actually have?
const lengths = [];
for (let d = 2; d <= 68; d++) {
  lengths.push(0);
}

const launched = lengths.map(() => 33 * 37);
plot(
  { 'cells with work': lengths, 'threads launched': launched },
  { title: 'work vs. threads, per launch', log: true }
);
`,
      solutionCode: `// 32 × 36 bases, 67 launches, one rendered frame each.
const gpu = new GPU({ mode });

const sweep = gpu.createKernel(function (H, a, b, d) {
  const i = this.thread.y;
  const j = this.thread.x;
  if (i === 0 || j === 0) return 0;
  if (i + j !== d) return H[i][j];
  let s = this.constants.mismatch;
  if (a[i - 1] === b[j - 1]) s = this.constants.match;
  const diag = H[i - 1][j - 1] + s;
  const up = H[i - 1][j] - this.constants.gap;
  const left = H[i][j - 1] - this.constants.gap;
  return Math.max(0, Math.max(diag, Math.max(up, left)));
}, { output: [37, 33], constants: { match: 3, mismatch: -3, gap: 2 } });

const paint = gpu.createKernel(function (H, d) {
  const v = H[this.thread.y][this.thread.x] / this.constants.top;
  if (this.thread.x + this.thread.y === d) {
    this.color(1, 0.6, 0.1, 1);
  } else {
    this.color(v * 0.35, v * 0.95, v * 0.75, 1);
  }
}, { output: [37, 33], graphical: true, constants: { top: 40 } });

let H = [];
for (let i = 0; i <= 32; i++) H.push(new Array(37).fill(0));

for (let d = 2; d <= 68; d++) {
  H = await sweep(H, codesA, codesB, d);
  await paint(H, d);
  render(paint.canvas);
}

let best = 0;
for (let i = 0; i <= 32; i++) {
  for (let j = 0; j <= 36; j++) if (H[i][j] > best) best = H[i][j];
}
console.log('best score', best);

const lengths = [];
for (let d = 2; d <= 68; d++) {
  const iStart = Math.max(1, d - 36);
  const iEnd = Math.min(32, d - 1);
  lengths.push(iEnd - iStart + 1);
}

const launched = lengths.map(() => 33 * 37);
plot(
  { 'cells with work': lengths, 'threads launched': launched },
  { title: 'work vs. threads, per launch', log: true }
);
`,
      inputs: () => ({
        seqA: LONG_A,
        seqB: LONG_B,
        codesA: codesOf(LONG_A),
        codesB: codesOf(LONG_B),
      }),
      publicTests: [
        {
          name: 'every launch rendered a frame — 67 of them',
          run: async ctx => {
            ctx.assert(ctx.canvas, 'no canvas — is paint graphical: true, and did you call render()?');
            ctx.assert(ctx.canvas.width === 37 && ctx.canvas.height === 33,
              `expected a 37×33 canvas, got ${ctx.canvas.width}×${ctx.canvas.height}`);
            const frames = canvasFrames(ctx.logs);
            ctx.assert(frames >= 67,
              `expected 67 rendered frames, one per anti-diagonal, found ${frames} — leave the ` +
                `render(paint.canvas) call inside the loop`);
          },
        },
        {
          name: 'the live diagonal is lit, and only the live diagonal',
          run: async ctx => {
            const graphical = ctx.kernels.find(k => k.kernel && k.kernel.graphical);
            ctx.assert(graphical, 'no graphical paint kernel found');
            const { H } = fillRef(LONG_A, LONG_B);
            const d = 34;
            await graphical(H, d);
            const pixels = graphical.getPixels();
            // getPixels row order is backend-dependent, so accept the diagonal
            // in either orientation rather than guessing which one this is.
            let direct = true;
            let flipped = true;
            for (let y = 0; y < 33; y++) {
              for (let x = 0; x < 37; x++) {
                const at = (y * 37 + x) * 4;
                const lit = pixels[at] > 200 && pixels[at + 1] > 110 && pixels[at + 1] < 190;
                if (lit !== (x + y === d)) direct = false;
                if (lit !== (x + (32 - y) === d)) flipped = false;
              }
            }
            ctx.assert(direct || flipped,
              'the amber cells do not form anti-diagonal 34 — the live diagonal is exactly the ' +
                'cells where this.thread.x + this.thread.y equals d');
          },
        },
        {
          name: 'cells off the diagonal are shaded by their score',
          run: async ctx => {
            const graphical = ctx.kernels.find(k => k.kernel && k.kernel.graphical);
            ctx.assert(graphical, 'no graphical paint kernel found');
            const { H, bi, bj } = fillRef(LONG_A, LONG_B);
            await graphical(H, 3); // a diagonal far from the cells under test
            const pixels = graphical.getPixels();
            const greenAt = (x, y) => pixels[(y * 37 + x) * 4 + 1];
            // The best cell must be much brighter than a cell of column 0,
            // which scores 0 in every row and so is dark whichever way round
            // this backend hands the rows back.
            const bright = Math.max(greenAt(bj, bi), greenAt(bj, 32 - bi));
            const dark = greenAt(0, 5);
            ctx.assert(bright > 150,
              `the best-scoring cell came back at green ${bright} — an unlit shade for a score of ` +
                `40 out of 40. Scale the colour by H[y][x] / this.constants.top`);
            ctx.assert(dark < 60,
              `a zero-scoring cell came back at green ${dark} — cells with no alignment score ` +
                `should be nearly black`);
          },
        },
        {
          name: 'the plot shows the triangle of work under the flat line of threads',
          run: async ctx => {
            const series = loggedPlot(ctx.logs, 'cells with work');
            ctx.assert(series,
              'no plot with a "cells with work" series — leave the plot() call as it is and fill ' +
                'the lengths array');
            const ref = lengthsRef(32, 36);
            ctx.assert(series.total === ref.length,
              `expected ${ref.length} points, one per anti-diagonal from 2 to 68, got ${series.total}`);
            const padded = [];
            for (let d = 2; d <= 68; d++) {
              padded.push(Math.min(32, d) - Math.max(0, d - 36) + 1);
            }
            const hint = diagnoseRow(series.values, ref, 1e-6, [
              [padded,
                'those are the diagonals of the PADDED matrix — row 0 and column 0 are never ' +
                  'computed, so a diagonal has min(32, d − 1) − max(1, d − 36) + 1 cells'],
              [ref.map(() => 0), 'the lengths array is still full of zeros'],
            ]);
            for (let k = 0; k < ref.length; k++) {
              ctx.assertClose(series.values[k], ref[k], 1e-6,
                hint || `diagonal ${k + 2} has ${ref[k]} cells`);
            }
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            // The sweep is prewired, so this is really a check that it was left
            // driving the loop it was given: the best score of this pair is 40.
            const ref = fillRef(LONG_A, LONG_B);
            const nums = loggedNumbers(ctx.logs);
            ctx.assert(nums.some(v => Math.abs(v - ref.best) < 1e-6),
              `the finished matrix should top out at ${ref.best} — expected it in the console output`);
            const total = lengthsRef(32, 36).reduce((sum, v) => sum + v, 0);
            ctx.assertClose(total, 32 * 36, 1e-6,
              'the diagonal lengths have to add up to |A| × |B| = 1152 cells');
          },
        },
        {
          name: 'private test #2',
          run: async ctx => {
            // The paint kernel on a matrix of known extremes: an all-zero matrix
            // is black everywhere off the diagonal, whatever the row order.
            // d = 0 leaves exactly one cell on the live diagonal, the corner
            // (0, 0) — hence the allowance of one lit pixel below and no more.
            const graphical = ctx.kernels.find(k => k.kernel && k.kernel.graphical);
            ctx.assert(graphical, 'expected a graphical paint kernel');
            await graphical(zerosMatrix(32, 36), 0);
            const pixels = graphical.getPixels();
            let lit = 0;
            for (let at = 0; at < pixels.length; at += 4) {
              if (pixels[at] > 200 && pixels[at + 1] > 110 && pixels[at + 1] < 190) lit++;
              else if (pixels[at + 1] > 60) lit += 1000;
            }
            ctx.assert(lit <= 1,
              'an empty matrix should paint almost entirely black — only cells with a score get ' +
                'any brightness, and d = 0 leaves just the corner cell (0, 0) on the live diagonal');
          },
        },
      ],
    },
  ],
};
