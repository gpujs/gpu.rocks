// Module: Prefix Sums (Scan) — uuid 351cfa41-ceee-4120-97e2-338870fa3aed
// (short id 351cfa41). The file name is the uuid; identity lives in the
// exported object below, never in the path. No legacyId: this module is new,
// so it predates no URL.
//
// Six tasks: the serial loop that cannot be a kernel → the honest O(n²)
// per-thread gather → the Hillis-Steele doubling ladder driven from JS →
// inclusive vs exclusive and why exclusive is the one worth having →
// Blelloch's work-efficient upsweep/downsweep → the payoff, offsets placing
// variable-sized runs of output.
//
// Kernel-authoring rules (contract): no closures inside kernel functions,
// only numbers / nested number arrays as inputs, this.thread.* for indexing,
// this.constants.* for compile-time loop bounds, statically bounded loops.
// Every task passes in CPU mode as well as GPU mode.
//
// NUMERIC NOTE. Every `values` array holds multiples of 0.125 in 0 … 2, and
// every count is a small integer. Both are exactly representable in float32,
// and so is every partial sum this module builds (the largest is 2,048, and
// 2048 · 8 is far below 2^24) — so a scan computed on the GPU agrees with the
// float64 references here to the bit, and the tests can be tight instead of
// tolerant of arithmetic that never actually goes wrong.

const N = 1024; // the array the three ladder tasks scan
const ITEMS = 32; // sessions in the sign-up sheet (tasks 4 and 6)
const SLOTS = 128; // COUNTS sums to exactly this

// Sign-ups per session: 32 small integers summing to exactly 128. Six sessions
// drew nobody, which is the case a scan has to get right — a zero-length run
// still needs a start offset, and it must own no slots at all.
const COUNTS = [
  6, 0, 8, 5, 5, 0, 4, 6,
  3, 4, 0, 8, 5, 2, 5, 3,
  6, 0, 4, 5, 7, 2, 0, 5,
  4, 3, 6, 6, 0, 8, 2, 6,
];

// Eight days of rainfall in mm, 1 dp — few enough that the whole array and its
// whole running total each print in one console line.
function makeRainfall(utils, seed = 1108) {
  const rand = utils.seededRandom(seed);
  const data = new Array(8);
  for (let i = 0; i < 8; i++) data[i] = Math.round(rand() * 180) / 10;
  return data;
}

// n deterministic values in 0 … 2, in steps of 0.125 (see the numeric note).
function makeValues(utils, n, seed = 3517) {
  const rand = utils.seededRandom(seed);
  const data = new Array(n);
  for (let i = 0; i < n; i++) data[i] = Math.round(rand() * 16) / 8;
  return data;
}

function sumOf(arr) {
  let total = 0;
  for (let i = 0; i < arr.length; i++) total += arr[i];
  return total;
}

function inclusiveScan(arr) {
  const out = new Array(arr.length);
  let running = 0;
  for (let i = 0; i < arr.length; i++) {
    running += arr[i];
    out[i] = running;
  }
  return out;
}

function exclusiveScan(arr) {
  const out = new Array(arr.length);
  let running = 0;
  for (let i = 0; i < arr.length; i++) {
    out[i] = running;
    running += arr[i];
  }
  return out;
}

// Sum of arr[i … n−1] — the scan run the other way, which is what a learner
// gets from `if (j >= this.thread.x)` in task 2.
function suffixScan(arr) {
  const out = new Array(arr.length);
  let running = 0;
  for (let i = arr.length - 1; i >= 0; i--) {
    running += arr[i];
    out[i] = running;
  }
  return out;
}

// The flat list task 6 builds: counts[i] copies of i, in order.
function expandCounts(counts) {
  const out = [];
  for (let i = 0; i < counts.length; i++) {
    for (let k = 0; k < counts[i]; k++) out.push(i);
  }
  return out;
}

// ---- Hillis-Steele references ---------------------------------------------
//
// One pass, parameterised so every near-miss variant below is the SAME code
// with one thing changed: `combine(own, partner)` is the fold, and `from` is
// the first index that has a partner — `stride` for the correct guard,
// `stride + 1` for a `>` where `>=` belongs.

function scanPass(v, stride, combine = (own, partner) => own + partner, from = stride) {
  const out = Array.from(v);
  for (let i = from; i < v.length; i++) out[i] = combine(v[i], v[i - stride]);
  return out;
}

function partnerOnlyPass(v, stride) {
  return scanPass(v, stride, (own, partner) => partner);
}

function strictGuardPass(v, stride) {
  return scanPass(v, stride, undefined, stride + 1);
}

// The same pass reaching RIGHT instead of left — a suffix ladder.
function rightPass(v, stride) {
  const out = Array.from(v);
  for (let i = 0; i + stride < v.length; i++) out[i] = v[i] + v[i + stride];
  return out;
}

function doublingStrides(n) {
  const strides = [];
  for (let stride = 1; stride < n; stride *= 2) strides.push(stride);
  return strides;
}

function ladderRef(values, strides, pass = scanPass) {
  let v = Array.from(values);
  for (const stride of strides) v = pass(v, stride);
  return v;
}

// ---- Blelloch references ---------------------------------------------------

// Upsweep: at stride s only the TOP cell of each 2s-wide block works.
function upsweepPass(v, stride) {
  const out = Array.from(v);
  const block = 2 * stride;
  for (let i = 0; i < v.length; i++) {
    if ((i + 1) % block === 0) out[i] = v[i] + v[i - stride];
  }
  return out;
}

// Downsweep: the block top keeps its own value plus its left partner's old
// subtotal; the left partner takes over the block top's old value.
function downsweepPass(v, stride) {
  const out = Array.from(v);
  const block = 2 * stride;
  for (let i = 0; i < v.length; i++) {
    if ((i + 1) % block === 0) out[i] = v[i] + v[i - stride];
    else if ((i + 1 + stride) % block === 0) out[i] = v[i + stride];
  }
  return out;
}

// The upsweep with the block width hardcoded at 2 instead of 2·stride: right
// at stride 1, wrong at every stride after it.
function hardcodedBlockPass(v, stride) {
  const out = Array.from(v);
  for (let i = 0; i < v.length; i++) {
    if ((i + 1) % 2 === 0 && i >= stride) out[i] = v[i] + v[i - stride];
  }
  return out;
}

// The downsweep with its two cases the wrong way round.
function swappedDownsweepPass(v, stride) {
  const out = Array.from(v);
  const block = 2 * stride;
  for (let i = 0; i < v.length; i++) {
    if ((i + 1) % block === 0) out[i] = v[i - stride];
    else if ((i + 1 + stride) % block === 0) out[i] = v[i] + v[i + stride];
  }
  return out;
}

const SWAPPED_SWEEP =
  'the two cases are the wrong way round — the block TOP is the one that adds ' +
  '(data[i] + data[i - stride]); its left partner simply takes over the top\'s old value';

function clearedLast(v) {
  const out = Array.from(v);
  out[out.length - 1] = 0;
  return out;
}

function upsweptRef(values, n) {
  return ladderRef(values, doublingStrides(n), upsweepPass);
}

// ---- drivers over the LEARNER's kernels ------------------------------------
//
// gpu.js locks an argument's type on a kernel's first invocation, so every
// array a test feeds back in is a Float32Array — the type every pass returns.

// Each pass is awaited before the next one launches: pass d reads what pass
// d − 1 returned, so the ladder is a sequential chain, never a Promise.all.
async function runLadder(step, values, n) {
  let v = Float32Array.from(values);
  for (let stride = 1; stride < n; stride *= 2) v = await step(v, stride);
  return v;
}

async function runSweeps(up, down, values, n) {
  let v = Float32Array.from(values);
  for (let stride = 1; stride < n; stride *= 2) v = await up(v, stride);
  v = Float32Array.from(v);
  v[n - 1] = 0;
  for (let stride = n / 2; stride >= 1; stride /= 2) v = await down(v, stride);
  return v;
}

// ---- kernel finders --------------------------------------------------------

function arityOf(k) {
  return ((k && k.kernel && k.kernel.argumentNames) || []).length;
}

function kernelsWithArity(ctx, arity) {
  return ctx.kernels.filter(k => arityOf(k) === arity);
}

function matches(got, expected, eps) {
  if (!got || got.length !== expected.length) return false;
  for (let i = 0; i < expected.length; i++) {
    if (!(Math.abs(got[i] - expected[i]) <= eps)) return false;
  }
  return true;
}

// Ramp 1 … n, used to tell the two sweep kernels apart by BEHAVIOUR rather
// than by creation order: after one pass at stride 1 the upsweep leaves the
// even cells alone and the downsweep does not, so the two can never be
// confused for each other.
function sweepProbe(n) {
  const probe = new Array(n);
  for (let i = 0; i < n; i++) probe[i] = i + 1;
  return probe;
}

async function findSweepKernels(ctx, n) {
  const probe = sweepProbe(n);
  const upRef = upsweepPass(probe, 1);
  const downRef = downsweepPass(probe, 1);
  let up = null;
  let down = null;
  for (const k of kernelsWithArity(ctx, 2)) {
    let out;
    try {
      out = await k(Float32Array.from(probe), 1);
    } catch (e) {
      continue;
    }
    if (!up && matches(out, upRef, 1e-3)) up = k;
    else if (!down && matches(out, downRef, 1e-3)) down = k;
  }
  return { up, down };
}

// Task 6's kernel is the one that answers 128 slots — identified by what it
// produces, so an extra kernel left lying around cannot be mistaken for it.
async function findOwnerKernel(ctx, offsets, counts) {
  for (let i = ctx.kernels.length - 1; i >= 0; i--) {
    const k = ctx.kernels[i];
    let out;
    try {
      out = await k(Float32Array.from(offsets), counts);
    } catch (e) {
      continue;
    }
    if (out && out.length === SLOTS) return k;
  }
  return null;
}

// ---- near-miss diagnosis ---------------------------------------------------
//
// A failing assert that reports only two numbers tells a learner nothing about
// WHICH slip produced them. A probe pairs the value one specific known mistake
// would produce with a sentence naming that mistake; diagnose() speaks only
// when the observed value matches a probe within the test's own tolerance AND
// the correct value does not — so a cell where two candidates coincide (cell 0,
// which every scan variant leaves alone) stays silent, as do observations
// matching probes that disagree with each other. A wrong diagnosis is worse
// than none.
function diagnose(got, expected, eps, probes) {
  const hits = probes
    .filter(p => Math.abs(got - p[0]) <= eps && Math.abs(expected - p[0]) > eps)
    .map(p => p[1]);
  return hits.length && hits.every(m => m === hits[0]) ? hits[0] : null;
}

// Whole-array form. Every scan variant agrees with the right answer on cell 0
// and usually on the first few cells, so one matching cell is worthless
// evidence: a probe here must predict EVERY cell (and disagree with the right
// answer somewhere) before it may speak. Probe values are read by index; a
// missing cell makes the comparison NaN, which fails.
//
// `probes` entries are [array, message] — always a prebuilt array, never a
// function that recomputes a reference per cell.
function diagnoseAll(count, got, expected, eps, probes) {
  const hits = probes
    .filter(([value]) => {
      let differs = false;
      for (let i = 0; i < count; i++) {
        if (!(Math.abs(got[i] - value[i]) <= eps)) return false;
        if (Math.abs(expected[i] - value[i]) > eps) differs = true;
      }
      return differs;
    })
    .map(p => p[1]);
  return hits.length && hits.every(m => m === hits[0]) ? hits[0] : null;
}

// A correct scan of finite input is finite everywhere, so a NaN can only be a
// read from an index that does not exist. (On WebGL the same mistake clamps
// the texel lookup instead of producing a NaN, so this speaks on the cpu
// backend and stays quiet on the gpu one — where the plain numeric mismatch
// is all anyone can honestly say.)
function nonFiniteHint(out, n, message) {
  for (let i = 0; i < n; i++) {
    if (!Number.isFinite(out[i])) return `cell ${i} is ${out[i]} — ${message}`;
  }
  return null;
}

const STRIDE_GUARD =
  'a thread read a negative index. The threads below `stride` have no partner ' +
  'this pass: guard the read with if (this.thread.x >= stride) and pass the ' +
  'value straight through otherwise';

const SHIFT_GUARD =
  'a thread read past the end of the array. Cell 0 has nothing before it, so it ' +
  'returns 0; every other cell reads inclusive[this.thread.x - 1]';

// ---- console helpers -------------------------------------------------------

// Every number that appeared in a console.log line, in the order it appeared.
function loggedNumbers(logs) {
  const out = [];
  for (const line of logs) {
    if (line.type !== 'log' || !line.text) continue;
    const found = line.text.match(/-?\d+(?:\.\d+)?/g);
    if (found) for (const m of found) out.push(parseFloat(m));
  }
  return out;
}

function loggedContains(nums, value, eps) {
  return nums.some(v => Math.abs(v - value) <= eps);
}

// `expected` as a consecutive run somewhere in the logged numbers — how a
// whole array printed with console.log is checked without dictating its label.
function loggedRun(nums, expected, eps) {
  for (let start = 0; start + expected.length <= nums.length; start++) {
    let ok = true;
    for (let k = 0; k < expected.length; k++) {
      if (Math.abs(nums[start + k] - expected[k]) > eps) {
        ok = false;
        break;
      }
    }
    if (ok) return true;
  }
  return false;
}

// diagnose(), for a value that should be somewhere in the console rather than
// in a particular cell. Silent unless exactly one mistake's value is present.
function loggedHint(nums, expected, eps, probes) {
  const hits = probes
    .filter(p => Math.abs(expected - p[0]) > eps && loggedContains(nums, p[0], eps))
    .map(p => p[1]);
  return hits.length && hits.every(m => m === hits[0]) ? hits[0] : null;
}

// ---- task-specific probe sets ----------------------------------------------

// Task 1: the arrays a learner logs instead of the running total.
function serialProbes(rain) {
  return [
    [rain, 'that is the daily rainfall again — nothing accumulated. running[i] has to lean on running[i - 1]'],
    [exclusiveScan(rain), 'every cell is short by its own day — an INCLUSIVE running total adds rainfall[i] on top of running[i - 1]'],
    [suffixScan(rain), 'that total runs backwards from the last day — a running total starts at day 0 and works forwards'],
  ];
}

// Task 2: the four ways the per-thread prefix walk goes wrong.
function naiveProbes(data) {
  const everything = new Array(data.length).fill(sumOf(data));
  return [
    [data, 'each cell is just its own element — the loop never accumulated anything'],
    [everything, 'every thread summed the WHOLE array — the mask is what stops thread i at element i: if (j <= this.thread.x)'],
    [exclusiveScan(data), 'each cell stops one element early — an inclusive scan includes your own element, so the mask is j <= this.thread.x, not j < this.thread.x'],
    [suffixScan(data), 'you summed from your own cell to the END — a prefix sum runs from element 0 up to and including you'],
  ];
}

// Task 3: what a broken PASS (wrong kernel) produces, whoever drives it.
function ladderKernelProbes(data, strides) {
  return [
    [Array.from(data), 'the pass changed nothing — every cell returned its own value'],
    [ladderRef(data, strides, partnerOnlyPass),
      'your cell kept only the partner\'s value — a scan pass ADDS: data[x] + data[x - stride]'],
    [ladderRef(data, strides, rightPass),
      'you added the partner on your RIGHT — a prefix sum reaches left: data[this.thread.x - stride]'],
    [ladderRef(data, strides, strictGuardPass),
      'the cell exactly `stride` from the start was skipped — its partner is element 0, so the guard is >=, not >'],
  ];
}

// Task 3: the driver, read off the stride of the LAST pass it ran. A ladder
// over n cells must finish at stride n/2; anything else names its own mistake.
function lastStrideProbes(n) {
  const passes = doublingStrides(n).length;
  return [
    [1, 'only one pass ran — the ladder needs a loop, and the stride has to double on every turn of it'],
    [passes, `the stride is growing by 1, not doubling — the last pass used ${passes}, so the loop is adding instead of doubling. Pass d reaches 2^d cells to the left`],
    [n / 4, `one pass short — ${n} cells need log2(${n}) = ${passes} passes, so the loop runs while stride < ${n}`],
  ];
}

// Task 4: the shifts that are one cell out.
function exclusiveProbes(inclusive) {
  const n = inclusive.length;
  const shiftedLeft = new Array(n);
  const onlyCellZero = new Array(n);
  for (let i = 0; i < n; i++) {
    shiftedLeft[i] = i + 1 < n ? inclusive[i + 1] : NaN;
    onlyCellZero[i] = i === 0 ? 0 : inclusive[i];
  }
  return [
    [inclusive, 'that is the inclusive scan — an exclusive scan slides everything one cell to the right and drops a 0 into cell 0'],
    [shiftedLeft, 'the shift went the wrong way — cell i wants the cell BEFORE it, inclusive[this.thread.x - 1]'],
    [onlyCellZero, 'cell 0 is right but nothing else moved — every cell shifts, not just the first'],
  ];
}

// Task 5: a wrong upsweep pass, seen after one call at stride 1.
function upsweepProbes(v) {
  const n = v.length;
  const passthrough = Array.from(v);
  const everyCell = new Array(n);
  const partnerOnly = new Array(n);
  for (let i = 0; i < n; i++) {
    everyCell[i] = i >= 1 ? v[i] + v[i - 1] : v[i];
    partnerOnly[i] = (i + 1) % 2 === 0 ? v[i - 1] : v[i];
  }
  return [
    [passthrough, 'that pass changed nothing — the top cell of each 2·stride block has to absorb its left partner'],
    [everyCell, 'EVERY cell added its partner — that is the Hillis-Steele pass. The upsweep is a tree: at stride s only the top cell of each 2s-wide block works'],
    [partnerOnly, 'the block top kept only its partner\'s subtotal — the upsweep ADDS: data[i] + data[i - stride]'],
  ];
}

// Task 5: the same, for the downsweep — each half of its job done without the
// other half.
function downsweepProbes(v) {
  const n = v.length;
  const passthrough = Array.from(v);
  const topOnly = new Array(n);
  const partnerOnly = new Array(n);
  for (let i = 0; i < n; i++) {
    topOnly[i] = (i + 1) % 2 === 0 ? v[i] + v[i - 1] : v[i];
    partnerOnly[i] = (i + 2) % 2 === 0 ? v[i + 1] : v[i];
  }
  return [
    [passthrough, 'that pass changed nothing — a downsweep hands the block top\'s value down to its left partner and adds the partner\'s old subtotal back into the top'],
    [topOnly, 'only the block top was updated — its left partner also has to take over the top\'s OLD value: data[this.thread.x + stride]'],
    [partnerOnly, 'only the left partner was updated — the block top also has to add its partner\'s old subtotal: data[this.thread.x] + data[this.thread.x - stride]'],
    [swappedDownsweepPass(v, 1), SWAPPED_SWEEP],
  ];
}

// Task 6: the ways a slot ends up owned by the wrong session.
function ownerProbes(counts, offsets, slots) {
  const allZero = new Array(slots).fill(0);
  const ownIndex = new Array(slots);
  const strictLower = new Array(slots);
  for (let slot = 0; slot < slots; slot++) {
    ownIndex[slot] = slot;
    let found = 0;
    for (let i = 0; i < counts.length; i++) {
      // `<` where `<=` belongs: the first seat of every run loses its owner.
      if (offsets[i] < slot && slot < offsets[i] + counts[i]) found = i;
    }
    strictLower[slot] = found;
  }
  return [
    [allZero, 'every slot came back 0 — the search never updated its answer'],
    [ownIndex, 'every slot returned its own index — it should return the SESSION that owns it'],
    [strictLower, 'the slot that STARTS a block found no owner — the lower test is offsets[i] <= slot, with the equals sign: a session owns the very first seat of its own run'],
  ];
}

export default {
  uuid: '351cfa41-ceee-4120-97e2-338870fa3aed',
  version: 1,
  slug: 'prefix-sum',
  title: 'Prefix Sums (Scan)',
  blurb: 'Running totals in parallel — the doubling ladder, exclusive scans, and the offsets every variable-sized output depends on.',
  tasks: [
    // ------------------------------------------------------------------ 1
    {
      slug: 'serial-scan',
      title: 'The Sum So Far',
      intro: `<p>A <strong>prefix sum</strong> — a <em>scan</em> — is a running total. Give it
        <code>[3, 1, 4, 1]</code> and it answers <code>[3, 4, 8, 9]</code>: cell <code>i</code>
        holds everything from the start up to and including element <code>i</code>. A reduction
        collapses an array to a single number; a scan keeps <em>every</em> partial answer along
        the way, which turns out to be far more useful.</p>
        <p>In JavaScript it is two lines, and the shape of those two lines is the whole
        problem:</p>
<pre><code>out[0] = x[0];
out[i] = out[i - 1] + x[i];</code></pre>
        <p>Look at what cell <code>i</code> needs: not its neighbour's <em>input</em>, but its
        neighbour's <strong>answer</strong>. Every thread on a GPU starts at the same instant, so
        when thread 7 reaches for <code>out[6]</code> nobody has computed it yet — and nobody will,
        because thread 6 is waiting on thread 5. That is a serial dependency chain as long as the
        array, and it cannot be a kernel. Write it here in plain JavaScript first; the rest of
        this module is five ways around it.</p>`,
      goal: `<strong>Goal:</strong> fill <code>running</code> so that <code>running[i]</code> is
        the total rainfall of days <code>0 … i</code>, then log the array and the season total.`,
      requirements: [
        'No kernel yet — plain JavaScript, so the dependency is impossible to miss',
        '<code>running[0]</code> is just <code>rainfall[0]</code>; every later cell adds that day to the cell before it',
        '<code>console.log</code> the whole <code>running</code> array, and the season total',
      ],
      hints: [
        {
          title: 'Hint 1 — seed the chain',
          body: `<p>Cell 0 has nothing before it, so it is the only cell that does not read
            <code>running[i - 1]</code>. Set it first, then loop from <code>i = 1</code>.</p>`,
        },
        {
          title: 'Hint 2 — the loop',
          body: `<pre><code>running[0] = rainfall[0];
for (let i = 1; i &lt; rainfall.length; i++) {
  running[i] = running[i - 1] + rainfall[i];
}</code></pre>
<p>The season total is the last cell — an inclusive scan ends with the
            reduction already done.</p>`,
        },
      ],
      transfer: `Every serious GPU platform ships a scan primitive precisely because you cannot
        write one by accident: CUDA has <code>thrust::inclusive_scan</code> and CUB's
        <code>DeviceScan</code>, ROCm has rocPRIM's <code>inclusive_scan</code>, Metal Shading
        Language has <code>simd_prefix_inclusive_sum</code>, and WGSL's subgroup extension has
        <code>subgroupInclusiveAdd</code>. All of them exist to break the chain you are about to
        feel.`,
      starterCode: `// No kernel here. Plain JavaScript, so the dependency is unmissable.
const running = new Array(rainfall.length);

// TODO: running[i] should be the total of rainfall[0 ... i].
// Right now every cell is just that day's rain — nothing accumulates.
for (let i = 0; i < rainfall.length; i++) {
  running[i] = rainfall[i];
}

console.log('daily  :', rainfall);
console.log('running:', running);
console.log('season total:', running[running.length - 1]);
`,
      solutionCode: `// No kernel here. Plain JavaScript, so the dependency is unmissable.
const running = new Array(rainfall.length);

// Cell 0 seeds the chain; every later cell leans on the one before it.
running[0] = rainfall[0];
for (let i = 1; i < rainfall.length; i++) {
  running[i] = running[i - 1] + rainfall[i];
}

console.log('daily  :', rainfall);
console.log('running:', running);
console.log('season total:', running[running.length - 1]);
`,
      inputs: utils => ({ rainfall: makeRainfall(utils) }),
      publicTests: [
        {
          name: 'the eight running totals reach the console',
          run: async ctx => {
            const rain = makeRainfall(ctx.utils);
            const expected = inclusiveScan(rain);
            const nums = loggedNumbers(ctx.logs);
            // Whole-array probes: an eight-value run is specific enough that a
            // match names the mistake outright.
            const wrong = serialProbes(rain).find(([value]) => loggedRun(nums, value, 1e-9));
            ctx.assert(
              loggedRun(nums, expected, 1e-9),
              (wrong && wrong[1]) ||
                `log the running total — expected the eight values ${expected.map(v => Number(v.toFixed(2))).join(', ')} in the console output`
            );
          },
        },
        {
          name: 'the season total is the last running total',
          run: async ctx => {
            const rain = makeRainfall(ctx.utils);
            const total = sumOf(rain);
            const nums = loggedNumbers(ctx.logs);
            ctx.assert(
              loggedContains(nums, total, 1e-9),
              `log the season total — expected ${Number(total.toFixed(2))} in the console output. ` +
                'It is the last cell of the running total: an inclusive scan finishes with the whole sum in it.'
            );
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            const rain = makeRainfall(ctx.utils);
            const expected = inclusiveScan(rain);
            const nums = loggedNumbers(ctx.logs);
            const wrong = serialProbes(rain).find(([value]) => loggedRun(nums, value, 1e-9));
            ctx.assert(
              loggedRun(nums, expected, 1e-9),
              (wrong && wrong[1]) || 'the eight running totals should appear in the console, in order'
            );
            // Every step exactly one day bigger than the last — a shifted or
            // reversed run cannot satisfy this and the run check at once.
            for (let i = 1; i < 8; i++) {
              ctx.assertClose(
                expected[i] - expected[i - 1], rain[i], 1e-9,
                `step ${i} of the running total should be day ${i}'s rainfall`
              );
            }
            ctx.assert(loggedContains(nums, sumOf(rain), 1e-9), 'the season total should appear too');
          },
        },
      ],
    },

    // ------------------------------------------------------------------ 2
    {
      slug: 'naive-gather',
      title: 'Everyone Sums Their Own Prefix',
      intro: `<p>The way out of a dependency chain is to refuse to wait. Thread <code>i</code>
        does not ask thread <code>i − 1</code> for its answer — it computes its own from scratch,
        summing <code>values[0 … i]</code> itself. No thread needs anything but the original
        input, so all 1,024 of them run at once. Correct, embarrassingly parallel, and a
        <strong>gather</strong>: reads from anywhere, a write only to its own cell.</p>
        <p>And wasteful. Thread 1,023 does 1,024 additions, thread 512 does 513, and the whole
        thing costs about <strong>n²/2 ≈ 524,000 additions</strong> where the serial loop needed
        1,023. That is the price of refusing to wait, and it is worth paying once: this is the
        honest baseline every cleverer scan has to beat, and the one you can put a stopwatch
        on.</p>
        <p>One wrinkle. You cannot write <code>for (let j = 0; j &lt;= this.thread.x; j++)</code> —
        in gpu.js's WebGL backend a loop bound must be known when the shader is compiled, and
        <code>this.thread.x</code> is not. So loop over the whole array and <em>mask</em>: every
        thread walks all 1,024 elements and only counts the ones at or before its own index.</p>`,
      goal: `<strong>Goal:</strong> return the inclusive prefix sum of <code>values</code> — one
        thread per cell, each summing its own prefix — and log the grand total.`,
      requirements: [
        'Loop <code>for (let j = 0; j &lt; this.constants.n; j++)</code> — a compile-time bound',
        'Add <code>data[j]</code> only while <code>j &lt;= this.thread.x</code>',
        'The last cell already holds the grand total — <code>console.log</code> it',
      ],
      hints: [
        {
          title: 'Hint 1 — which elements are mine?',
          body: `<p>Thread 7 wants elements 0 through 7, its own included. Thread 0 wants only
            element 0. So the test inside the loop is <code>j &lt;= this.thread.x</code> — with
            the equals sign, because the scan is <em>inclusive</em>.</p>`,
        },
        {
          title: 'Hint 2 — the loop body',
          body: `<pre><code>let sum = 0;
for (let j = 0; j &lt; this.constants.n; j++) {
  if (j &lt;= this.thread.x) {
    sum += data[j];
  }
}
return sum;</code></pre>`,
        },
      ],
      transfer: `The brute-force scan is not only a straw man — it is what you actually want at
        the very bottom of the hierarchy, where a handful of values already sit in registers and
        a smarter algorithm's bookkeeping costs more than the redundant adds. Above that size it
        loses badly, which is why CUB, rocPRIM and Thrust all switch strategy by scale instead of
        shipping one scan.`,
      starterCode: `// 1024 threads, each summing its own prefix. Nobody waits for anybody.
const gpu = new GPU({ mode });

const prefix = gpu.createKernel(function (data) {
  // TODO: accumulate data[j] for every j at or before this thread's index.
  // The loop bound has to be a compile-time constant, so walk the whole
  // array and mask with an if.
  return data[this.thread.x];
}, {
  output: [1024],
  constants: { n: 1024 },
});

const scan = await prefix(values);
console.log('scan[0]:', scan[0], ' scan[1]:', scan[1], ' scan[2]:', scan[2]);
// TODO: the last cell is already the grand total — log it.
console.log('grand total:', 0);
`,
      solutionCode: `// 1024 threads, each summing its own prefix. Nobody waits for anybody.
const gpu = new GPU({ mode });

const prefix = gpu.createKernel(function (data) {
  let sum = 0;
  for (let j = 0; j < this.constants.n; j++) {
    if (j <= this.thread.x) {
      sum += data[j];
    }
  }
  return sum;
}, {
  output: [1024],
  constants: { n: 1024 },
});

const scan = await prefix(values);
console.log('scan[0]:', scan[0], ' scan[1]:', scan[1], ' scan[2]:', scan[2]);
console.log('grand total:', scan[1023]);
`,
      inputs: utils => ({ values: makeValues(utils, N, 3517) }),
      publicTests: [
        {
          name: 'one output cell per input value',
          run: async ctx => {
            ctx.assert(ctx.kernels.length >= 1, 'no kernel was created — call gpu.createKernel()');
            const out = await ctx.kernel(makeValues(ctx.utils, N, 3517));
            ctx.assert(out && out.length === N, `expected ${N} output values, got ${out && out.length}`);
          },
        },
        {
          name: 'cell <code>i</code> totals <code>values[0 … i]</code>',
          run: async ctx => {
            const data = new Array(N);
            for (let i = 0; i < N; i++) data[i] = ((i % 7) + 1) / 4;
            const out = await ctx.kernel(data);
            const expected = inclusiveScan(data);
            const hint = nonFiniteHint(out, N, 'a read landed outside the array — every index in this kernel is 0 … n - 1') ||
              diagnoseAll(N, out, expected, 1e-3, naiveProbes(data));
            ctx.assertClose(out[0], data[0], 1e-3, hint || 'cell 0 has nothing before it, so it is just values[0]');
            for (let i = 0; i < N; i++) {
              ctx.assertClose(out[i], expected[i], 1e-3, hint || `cell ${i}`);
            }
          },
        },
        {
          name: 'the grand total is logged',
          run: async ctx => {
            const data = makeValues(ctx.utils, N, 3517);
            const total = sumOf(data);
            const nums = loggedNumbers(ctx.logs);
            const hint = loggedHint(nums, total, 1e-3, [
              [total - data[N - 1], 'that total stops one element early — the LAST cell of an inclusive scan is the grand total'],
            ]);
            ctx.assert(
              loggedContains(nums, total, 1e-3),
              hint || `log the grand total — expected ${total} in the console output`
            );
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            const data = makeValues(ctx.utils, N, 8642);
            const out = await ctx.kernel(data);
            ctx.assert(out && out.length === N, `expected ${N} output values`);
            const expected = inclusiveScan(data);
            const hint = nonFiniteHint(out, N, 'a read landed outside the array') ||
              diagnoseAll(N, out, expected, 1e-3, naiveProbes(data));
            for (let i = 0; i < N; i++) {
              ctx.assertClose(out[i], expected[i], 1e-3, hint || `cell ${i}`);
            }
          },
        },
      ],
    },

    // ------------------------------------------------------------------ 3
    {
      slug: 'hillis-steele',
      title: 'The Doubling Ladder',
      intro: `<p>Half a million additions for a thousand-element scan is a lot. Here is the trick
        that gets it down to ten thousand: run <strong>log₂(n) passes</strong>, and on pass
        <code>d</code> have every cell add the value <code>2^d</code> places to its left. Stride 1,
        then 2, then 4, 8, … After pass <code>d</code> every cell holds the sum of the
        <code>2^(d+1)</code> elements ending at it, so ten passes over 1,024 cells leave each one
        holding its whole prefix. This is the <strong>Hillis-Steele</strong> scan — the same
        stride ladder the Reductions module climbs to collapse an array, run the other way:
        doubling instead of halving, and keeping every partial answer instead of only the
        last.</p>
        <p>One kernel, called ten times from a plain JavaScript loop with the stride as an
        <em>argument</em>. That multi-pass gather formulation is the point: gpu.js gives you no
        atomics and no shared memory, so a pass boundary is the only synchronisation there is —
        and it is the same shape as the barrier-separated steps a CUDA or WebGPU scan uses.</p>
        <p>It also hands you something for free. An in-place scan has a famous race: cell 7 reads
        cell 6 while cell 6 is busy overwriting itself, and back comes somebody's half-finished
        answer. Real GPU code prevents that with a barrier or a second buffer
        (<em>ping-pong</em> buffering). Here a kernel cannot write into the array it is reading —
        each pass <strong>returns a new array</strong> and the next pass consumes it, so the race
        is simply unavailable. As long as you really do feed each pass the previous pass's
        result.</p>`,
      goal: `<strong>Goal:</strong> write the one-pass kernel, drive ten passes from JavaScript
        with the stride doubling each time, and log <code>scan[511]</code> and the grand total.`,
      requirements: [
        'The kernel takes <code>(data, stride)</code> and returns <code>data[x] + data[x − stride]</code>',
        'Threads below <code>stride</code> have no partner — they pass their own value through',
        'Drive the passes from JS: <code>stride</code> = 1, 2, 4, … while <code>stride &lt; 1024</code>',
        'Each pass reads the array the <em>previous</em> pass returned',
      ],
      hints: [
        {
          title: 'Hint 1 — one pass',
          body: `<p>Every cell wants the value <code>stride</code> places to its left — but cells
            <code>0 … stride − 1</code> have no such cell. They keep what they already have:</p>
<pre><code>if (this.thread.x &gt;= stride) {
  return data[this.thread.x] + data[this.thread.x - stride];
}
return data[this.thread.x];</code></pre>`,
        },
        {
          title: 'Hint 2 — the driver',
          body: `<p>Ten passes, and the stride <em>doubles</em> — <code>1, 2, 4, 8, …</code>, not
            <code>1, 2, 3</code>. The reassignment is what makes pass <code>d</code> read what
            pass <code>d − 1</code> returned:</p>
<pre><code>for (let stride = 1; stride &lt; N; stride *= 2) {
  v = await scanStep(v, stride);
}</code></pre>`,
        },
        {
          title: 'Hint 3 — why Float32Array',
          body: `<p>gpu.js locks an argument's type on a kernel's first call, and every pass hands
            back a <code>Float32Array</code>. Start the ladder from one —
            <code>Float32Array.from(values)</code> — so pass 1 sees the same type as passes
            2 … 10.</p>`,
        },
      ],
      transfer: `This exact ladder is burned into GPU silicon at warp scale. Metal's
        <code>simd_prefix_inclusive_sum</code>, WGSL's <code>subgroupInclusiveAdd</code> and the
        CUDA idiom built from <code>__shfl_up_sync</code> are all Hillis-Steele over 32 or 64
        lanes, with a lane-id comparison playing the part of your <code>if (x &gt;= stride)</code>
        guard. What you are writing by hand across kernel launches, the hardware does in five
        instructions inside a warp.`,
      starterCode: `// One kernel, log2(1024) = 10 calls. The stride doubles every pass.
const gpu = new GPU({ mode });
const N = 1024;

const scanStep = gpu.createKernel(function (data, stride) {
  // TODO: add the value \`stride\` places to your left — if it exists.
  // Threads below \`stride\` have no partner and keep their own value.
  return data[this.thread.x];
}, { output: [N] });

// gpu.js locks an argument's type on the first call, so start from a
// Float32Array — the same type every pass hands back.
let v = Float32Array.from(values);

// TODO: ten passes, stride 1, 2, 4, ... 512. Each pass must read the
// array the PREVIOUS pass returned.
v = await scanStep(v, 1);

console.log('scan[511]:', v[511]);
console.log('grand total:', v[N - 1]);
`,
      solutionCode: `// One kernel, log2(1024) = 10 calls. The stride doubles every pass.
const gpu = new GPU({ mode });
const N = 1024;

const scanStep = gpu.createKernel(function (data, stride) {
  if (this.thread.x >= stride) {
    return data[this.thread.x] + data[this.thread.x - stride];
  }
  return data[this.thread.x];
}, { output: [N] });

// gpu.js locks an argument's type on the first call, so start from a
// Float32Array — the same type every pass hands back.
let v = Float32Array.from(values);

for (let stride = 1; stride < N; stride *= 2) {
  v = await scanStep(v, stride);
}

console.log('scan[511]:', v[511]);
console.log('grand total:', v[N - 1]);
`,
      inputs: utils => ({ values: makeValues(utils, N, 7311) }),
      publicTests: [
        {
          // FIRST, because it reads the arguments of the LAST call the learner's
          // own driver made — and every test below re-invokes the kernel.
          name: 'your driver climbs ten doubling passes, each on the last one\'s output',
          run: async ctx => {
            const step = ctx.kernels.find(k => arityOf(k) === 2);
            ctx.assert(step, 'no kernel taking (data, stride) was found — the stride is an argument, so one kernel can serve every pass');
            const last = step.lastArgs;
            ctx.assert(
              Array.isArray(last) && last.length === 2,
              'the kernel was never invoked as scanStep(v, stride) — the driver passes the stride in as the second argument'
            );
            const strideHint = diagnose(last[1], N / 2, 0, lastStrideProbes(N));
            ctx.assertClose(last[1], N / 2, 0, strideHint ||
              `the last pass should use stride ${N / 2}, the widest reach an array of ${N} needs`);
            // …and it must have been handed the previous pass's output, not
            // the original array all over again. A kernel that returns its own
            // cell unchanged would leave the same fingerprint, and that is the
            // kernel's fault rather than the driver's — so check the kernel
            // does something before blaming the loop.
            const data = makeValues(ctx.utils, N, 7311);
            const fed = last[0];
            ctx.assert(fed && fed.length === N, 'the first argument of each pass should be the whole array');
            const probe = sweepProbe(N);
            const kernelChangesSomething = !matches(await step(Float32Array.from(probe), 1), probe, 1e-6);
            if (kernelChangesSomething) {
              ctx.assert(
                !matches(fed, data, 1e-6),
                'the last pass was handed the ORIGINAL array again — pass d has to read what pass d-1 returned, which is exactly what v = await scanStep(v, stride) does'
              );
            }
          },
        },
        {
          name: 'one pass at stride 4 folds in the partner four cells left',
          run: async ctx => {
            const step = ctx.kernels.find(k => arityOf(k) === 2);
            ctx.assert(step, 'no kernel taking (data, stride) was found — the stride is an argument, so one kernel can serve every pass');
            const data = makeValues(ctx.utils, N, 7311);
            const out = await step(Float32Array.from(data), 4);
            ctx.assert(out && out.length === N, `expected ${N} values back from one pass, got ${out && out.length}`);
            const expected = scanPass(data, 4);
            const hint = nonFiniteHint(out, N, STRIDE_GUARD) ||
              diagnoseAll(N, out, expected, 1e-3, ladderKernelProbes(data, [4]));
            for (let i = 0; i < N; i++) {
              ctx.assertClose(out[i], expected[i], 1e-3, hint || `cell ${i} after one pass at stride 4`);
            }
          },
        },
        {
          name: 'ten doubling passes make the full inclusive scan',
          run: async ctx => {
            const step = ctx.kernels.find(k => arityOf(k) === 2);
            ctx.assert(step, 'no kernel taking (data, stride) was found');
            const data = new Array(N);
            for (let i = 0; i < N; i++) data[i] = ((i % 5) + 1) / 8;
            // The TEST drives the ladder here, so this checks the kernel on its
            // own — a wrong driver is caught by the logged values instead.
            const out = await runLadder(step, data, N);
            const expected = inclusiveScan(data);
            const hint = nonFiniteHint(out, N, STRIDE_GUARD) ||
              diagnoseAll(N, out, expected, 1e-3, ladderKernelProbes(data, doublingStrides(N)));
            for (let i = 0; i < N; i++) {
              ctx.assertClose(out[i], expected[i], 1e-3, hint || `cell ${i} after the ladder`);
            }
          },
        },
        {
          name: 'your ladder logged <code>scan[511]</code> and the grand total',
          run: async ctx => {
            const expected = inclusiveScan(makeValues(ctx.utils, N, 7311));
            const nums = loggedNumbers(ctx.logs);
            ctx.assert(
              loggedContains(nums, expected[N - 1], 1e-3),
              `log the grand total — expected ${expected[N - 1]} in the console output`
            );
            ctx.assert(
              loggedContains(nums, expected[511], 1e-3),
              `log scan[511] — expected ${expected[511]} in the console output`
            );
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            const step = ctx.kernels.find(k => arityOf(k) === 2);
            ctx.assert(step, 'expected a kernel taking (data, stride)');
            const data = makeValues(ctx.utils, N, 2244);
            const out = await runLadder(step, data, N);
            const expected = inclusiveScan(data);
            const hint = nonFiniteHint(out, N, STRIDE_GUARD) ||
              diagnoseAll(N, out, expected, 1e-3, ladderKernelProbes(data, doublingStrides(N)));
            for (let i = 0; i < N; i++) {
              ctx.assertClose(out[i], expected[i], 1e-3, hint || `cell ${i}`);
            }
            // One pass at a big stride, on its own: the guard has to hold for
            // the whole untouched left half, not just for cell 0.
            const one = await step(Float32Array.from(data), 256);
            const guard = nonFiniteHint(one, N, STRIDE_GUARD);
            for (let i = 0; i < 256; i++) {
              ctx.assertClose(one[i], data[i], 1e-3,
                guard || `cell ${i} has no partner at stride 256 and must pass through unchanged`);
            }
          },
        },
      ],
    },

    // ------------------------------------------------------------------ 4
    {
      slug: 'exclusive-scan',
      title: 'Inclusive, Exclusive, and Why It Matters',
      intro: `<p>Scans come in two flavours. The <strong>inclusive</strong> scan you just built
        answers <em>"everything up to and including me"</em>. The <strong>exclusive</strong> scan
        answers <em>"everything strictly before me"</em>: cell 0 is <code>0</code>, and every
        other cell is the inclusive scan shifted one place right.</p>
        <p>Exclusive is the one everything downstream actually wants, because it answers a
        different question — <strong>where does my run of output start?</strong> Here
        <code>counts</code> is a sign-up sheet: <code>counts[i]</code> people booked session
        <code>i</code>, and you are laying all of them out in one flat seating list. Session
        <code>i</code>'s block begins at <code>exclusive[i]</code>. The inclusive scan would tell
        you where that block <em>ends</em>, which is exactly one seat too late.</p>
        <p>Converting is a one-line gather: cell <code>i</code> reads <code>inclusive[i − 1]</code>,
        and cell 0 returns <code>0</code> because it has nothing before it. One wrinkle worth
        knowing — an exclusive scan <em>throws the grand total away</em>. Its last cell holds
        everything except the last element, so keep the total separately:
        <code>exclusive[n − 1] + counts[n − 1]</code>.</p>`,
      goal: `<strong>Goal:</strong> turn the prewired inclusive scan into the exclusive scan — the
        starting offset of every session — and log the total number of seats.`,
      requirements: [
        'One kernel, taking the inclusive scan as its single argument',
        'Cell 0 returns <code>0</code>; cell <code>i</code> returns <code>inclusive[i − 1]</code>',
        'Log the grand total, which the exclusive scan on its own no longer knows',
      ],
      hints: [
        {
          title: 'Hint 1 — a shift is a gather',
          body: `<p>"Move everything one cell right" is a scatter, and kernels cannot scatter. Ask
            the inverted question instead — <em>whose value lands in MY cell?</em> — and it is a
            one-line read from <code>this.thread.x - 1</code>.</p>`,
        },
        {
          title: 'Hint 2 — the edge',
          body: `<p>Thread 0 must not read <code>inclusive[-1]</code>:</p>
<pre><code>if (this.thread.x === 0) {
  return 0;
}
return inclusive[this.thread.x - 1];</code></pre>`,
        },
        {
          title: 'Hint 3 — the total that got away',
          body: `<p><code>offsets[31]</code> is where the LAST session starts, so the seat count is
            <code>offsets[31] + counts[31]</code>. (The inclusive scan's last cell had it all
            along — that is the reduction hiding inside every scan.)</p>`,
        },
      ],
      transfer: `Exclusive is the library default for exactly this reason:
        <code>cub::DeviceScan::ExclusiveSum</code>, <code>thrust::exclusive_scan</code>, WGSL's
        <code>subgroupExclusiveAdd</code> and Metal's <code>simd_prefix_exclusive_sum</code> all
        answer "where does my output begin?". And they all share the same wrinkle — CUB hands the
        aggregate back through a separate output, because the exclusive scan itself cannot carry
        it.`,
      starterCode: `// counts[i] people booked session i. Where does each session's block start?
const gpu = new GPU({ mode });
const N = 32;

// Last task's ladder, prewired: inclusive[i] = counts[0] + ... + counts[i].
const scanStep = gpu.createKernel(function (data, stride) {
  if (this.thread.x >= stride) {
    return data[this.thread.x] + data[this.thread.x - stride];
  }
  return data[this.thread.x];
}, { output: [N] });

let v = Float32Array.from(counts);
for (let stride = 1; stride < N; stride *= 2) {
  v = await scanStep(v, stride);
}
const inclusive = v;

const toExclusive = gpu.createKernel(function (inclusive) {
  // TODO: cell i should hold the total of everything BEFORE session i.
  // Cell 0 has nothing before it.
  return inclusive[this.thread.x];
}, { output: [N] });

const offsets = await toExclusive(inclusive);

console.log('counts [0..3]:', counts[0], counts[1], counts[2], counts[3]);
console.log('offsets[0..3]:', offsets[0], offsets[1], offsets[2], offsets[3]);
// TODO: the exclusive scan dropped the grand total. Log it.
console.log('total seats:', 0);
`,
      solutionCode: `// counts[i] people booked session i. Where does each session's block start?
const gpu = new GPU({ mode });
const N = 32;

// Last task's ladder, prewired: inclusive[i] = counts[0] + ... + counts[i].
const scanStep = gpu.createKernel(function (data, stride) {
  if (this.thread.x >= stride) {
    return data[this.thread.x] + data[this.thread.x - stride];
  }
  return data[this.thread.x];
}, { output: [N] });

let v = Float32Array.from(counts);
for (let stride = 1; stride < N; stride *= 2) {
  v = await scanStep(v, stride);
}
const inclusive = v;

const toExclusive = gpu.createKernel(function (inclusive) {
  if (this.thread.x === 0) {
    return 0;
  }
  return inclusive[this.thread.x - 1];
}, { output: [N] });

const offsets = await toExclusive(inclusive);

console.log('counts [0..3]:', counts[0], counts[1], counts[2], counts[3]);
console.log('offsets[0..3]:', offsets[0], offsets[1], offsets[2], offsets[3]);
console.log('total seats:', offsets[N - 1] + counts[N - 1]);
`,
      inputs: () => ({ counts: COUNTS.slice() }),
      publicTests: [
        {
          name: 'a shift kernel: cell 0 is <code>0</code>, cell <code>i</code> is <code>inclusive[i − 1]</code>',
          run: async ctx => {
            const shift = ctx.kernels.find(k => arityOf(k) === 1);
            ctx.assert(shift, 'no kernel taking the inclusive scan as its one argument was found — the conversion is a shift, so it needs nothing else');
            const counts = COUNTS.slice();
            const inclusive = inclusiveScan(counts);
            const out = await shift(Float32Array.from(inclusive));
            ctx.assert(out && out.length === ITEMS, `expected ${ITEMS} offsets, got ${out && out.length}`);
            const expected = exclusiveScan(counts);
            const hint = nonFiniteHint(out, ITEMS, SHIFT_GUARD) ||
              diagnoseAll(ITEMS, out, expected, 1e-3, exclusiveProbes(inclusive));
            ctx.assertClose(out[0], 0, 1e-6, hint || 'cell 0 has nothing before it, so its offset is exactly 0');
            for (let i = 0; i < ITEMS; i++) {
              ctx.assertClose(out[i], expected[i], 1e-3, hint || `offset ${i}`);
            }
          },
        },
        {
          name: 'an empty session still gets an offset — and takes no seats',
          run: async ctx => {
            const shift = ctx.kernels.find(k => arityOf(k) === 1);
            ctx.assert(shift, 'expected a one-argument shift kernel');
            const counts = COUNTS.slice();
            const out = await shift(Float32Array.from(inclusiveScan(counts)));
            for (let i = 1; i < ITEMS; i++) {
              if (counts[i - 1] !== 0) continue;
              // A zero-length run starts exactly where the previous one did.
              ctx.assertClose(
                out[i], out[i - 1], 1e-6,
                `session ${i - 1} has no sign-ups, so session ${i} must start exactly where session ${i - 1} does`
              );
            }
          },
        },
        {
          name: 'the total seat count is logged',
          run: async ctx => {
            const counts = COUNTS.slice();
            const total = sumOf(counts);
            const exclusive = exclusiveScan(counts);
            const nums = loggedNumbers(ctx.logs);
            const hint = loggedHint(nums, total, 1e-6, [
              [exclusive[ITEMS - 1], 'that is where the LAST session starts, not the seat count — add its own counts[31] on top'],
            ]);
            ctx.assert(
              loggedContains(nums, total, 1e-6),
              hint || `log the total seat count — expected ${total} in the console output`
            );
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            const shift = ctx.kernels.find(k => arityOf(k) === 1);
            ctx.assert(shift, 'expected a one-argument shift kernel');
            // A different sign-up sheet, with a zero run at the very front,
            // where an off-by-one shift is easiest to get away with.
            const counts = new Array(ITEMS);
            for (let i = 0; i < ITEMS; i++) counts[i] = (i * 5) % 7 === 0 ? 0 : (i % 6) + 1;
            const inclusive = inclusiveScan(counts);
            const expected = exclusiveScan(counts);
            const out = await shift(Float32Array.from(inclusive));
            const hint = nonFiniteHint(out, ITEMS, SHIFT_GUARD) ||
              diagnoseAll(ITEMS, out, expected, 1e-3, exclusiveProbes(inclusive));
            for (let i = 0; i < ITEMS; i++) {
              ctx.assertClose(out[i], expected[i], 1e-3, hint || `offset ${i}`);
            }
            ctx.assertClose(
              out[ITEMS - 1] + counts[ITEMS - 1], sumOf(counts), 1e-3,
              'the last offset plus the last count has to recover the grand total'
            );
          },
        },
      ],
    },

    // ------------------------------------------------------------------ 5
    {
      slug: 'blelloch',
      title: 'Work-Efficient: Upsweep, Downsweep',
      intro: `<p>Hillis-Steele is fast but greedy: ten passes over 1,024 cells is about
        <strong>n·log₂n ≈ 10,000 additions</strong> where the serial loop needed 1,023. The
        <strong>Blelloch</strong> scan gets that down to roughly <strong>2n</strong>, in two
        halves of one balanced tree.</p>
        <p><em>Upsweep</em> is a plain tree reduction — the one the Reductions module builds —
        done in place: at stride 1 every odd cell absorbs its left neighbour, at stride 2 every
        fourth cell absorbs the subtotal two places left, and so on. After log₂n passes the last
        cell holds the grand total and each
        "block top" holds its own block's subtotal — a whole tree of partial sums, stored in the
        array it came from. <em>Downsweep</em> then walks that tree back down: put <code>0</code>
        in the last cell, and at each level a node hands its value down to its left partner while
        keeping its own value plus that partner's old subtotal. What falls out is the
        <strong>exclusive</strong> scan.</p>
        <p>Be honest about the payoff. Blelloch does a fraction of the arithmetic — 2n against
        n·log₂n — but needs <em>twice</em> the kernel launches (21 here against 10), and near the
        root of the tree almost every thread is idle. At
        n = 1,024 the simpler ladder usually wins on the clock; work-efficiency only starts paying
        once the array is big enough that arithmetic, not launch overhead, is the bill. Press
        <strong>Benchmark</strong> and watch the better algorithm lose.</p>`,
      goal: `<strong>Goal:</strong> write the two sweep kernels. The prewired driver runs upsweep
        up the tree, clears the last cell, and runs downsweep back down — producing the exclusive
        scan of <code>values</code>.`,
      requirements: [
        'Upsweep: only the top cell of each <code>2·stride</code> block changes, to <code>data[i] + data[i − stride]</code>',
        'Downsweep: the block top becomes <code>data[i] + data[i − stride]</code>, and its left partner takes over the block top\'s old value',
        'Every other cell in both kernels passes its value straight through',
        'Log <code>exclusive[512]</code> and the grand total',
      ],
      hints: [
        {
          title: 'Hint 1 — which cells are active?',
          body: `<p>At stride <code>s</code> the blocks are <code>2s</code> wide, so their tops sit
            at indexes <code>2s − 1, 4s − 1, 6s − 1, …</code> — exactly the cells where
            <code>(i + 1) % (2 * stride) === 0</code>. The top's left partner is
            <code>stride</code> places earlier, so the partner's own test is
            <code>(i + 1 + stride) % (2 * stride) === 0</code>.</p>`,
        },
        {
          title: 'Hint 2 — the upsweep body',
          body: `<pre><code>const i = this.thread.x;
if ((i + 1) % (2 * stride) === 0) {
  return data[i] + data[i - stride];
}
return data[i];</code></pre>
<p>At stride 1 that is cells 1, 3, 5, …; at stride 2 it is cells 3, 7, 11, … —
            half as many workers each pass, which is where the n·log n turns into 2n.</p>`,
        },
        {
          title: 'Hint 3 — the downsweep body',
          body: `<p>Two active cases, and everybody else passes through:</p>
<pre><code>const i = this.thread.x;
const block = 2 * stride;
if ((i + 1) % block === 0) {
  return data[i] + data[i - stride];
}
if ((i + 1 + stride) % block === 0) {
  return data[i + stride];
}
return data[i];</code></pre>
<p>The second case is the left partner taking over the block top's old
            value — which is why both swaps have to happen in the same pass, reading the same
            snapshot.</p>`,
        },
      ],
      transfer: `Blelloch's two sweeps are the textbook work-efficient scan, and they are what
        every GPU course draws on the board. Production libraries have moved past them: CUB's
        <code>DeviceScan</code> uses a single-pass <em>decoupled look-back</em>, where each block
        scans locally and then waits on its predecessors' aggregates, because on modern hardware
        the bill is memory traffic rather than additions — and two full sweeps means reading the
        array twice. Knowing why the elegant answer lost is the real lesson.`,
      starterCode: `// Two sweeps of a balanced tree. ~2n additions instead of n·log2(n).
const gpu = new GPU({ mode });
const N = 1024;

// UPSWEEP — build the reduction tree in place.
const upsweep = gpu.createKernel(function (data, stride) {
  const i = this.thread.x;
  // TODO: only the TOP cell of each 2*stride block works this pass — it
  // absorbs the subtotal \`stride\` places to its left. Everyone else
  // passes their value straight through.
  return data[i];
}, { output: [N] });

// DOWNSWEEP — walk the tree back down.
const downsweep = gpu.createKernel(function (data, stride) {
  const i = this.thread.x;
  // TODO: two kinds of active cell this pass, everyone else passes through:
  //   * the top of each 2*stride block keeps its own value PLUS its left
  //     partner's old subtotal;
  //   * that left partner takes over the block top's old value.
  return data[i];
}, { output: [N] });

// An exclusive scan starts from 0 at the root — prewired.
const clearLast = gpu.createKernel(function (data) {
  if (this.thread.x === this.constants.n - 1) {
    return 0;
  }
  return data[this.thread.x];
}, { output: [N], constants: { n: N } });

let v = Float32Array.from(values);

for (let stride = 1; stride < N; stride *= 2) {
  v = await upsweep(v, stride);
}

const total = v[N - 1]; // the upsweep already reduced the whole array
v = await clearLast(v);

for (let stride = N / 2; stride >= 1; stride /= 2) {
  v = await downsweep(v, stride);
}

console.log('exclusive[512]:', v[512]);
console.log('grand total:', total);
`,
      solutionCode: `// Two sweeps of a balanced tree. ~2n additions instead of n·log2(n).
const gpu = new GPU({ mode });
const N = 1024;

// UPSWEEP — build the reduction tree in place.
const upsweep = gpu.createKernel(function (data, stride) {
  const i = this.thread.x;
  if ((i + 1) % (2 * stride) === 0) {
    return data[i] + data[i - stride];
  }
  return data[i];
}, { output: [N] });

// DOWNSWEEP — walk the tree back down.
const downsweep = gpu.createKernel(function (data, stride) {
  const i = this.thread.x;
  const block = 2 * stride;
  if ((i + 1) % block === 0) {
    return data[i] + data[i - stride];
  }
  if ((i + 1 + stride) % block === 0) {
    return data[i + stride];
  }
  return data[i];
}, { output: [N] });

// An exclusive scan starts from 0 at the root — prewired.
const clearLast = gpu.createKernel(function (data) {
  if (this.thread.x === this.constants.n - 1) {
    return 0;
  }
  return data[this.thread.x];
}, { output: [N], constants: { n: N } });

let v = Float32Array.from(values);

for (let stride = 1; stride < N; stride *= 2) {
  v = await upsweep(v, stride);
}

const total = v[N - 1]; // the upsweep already reduced the whole array
v = await clearLast(v);

for (let stride = N / 2; stride >= 1; stride /= 2) {
  v = await downsweep(v, stride);
}

console.log('exclusive[512]:', v[512]);
console.log('grand total:', total);
`,
      inputs: utils => ({ values: makeValues(utils, N, 5109) }),
      publicTests: [
        {
          name: 'two sweep kernels — one climbs the tree, one walks it back down',
          run: async ctx => {
            const candidates = kernelsWithArity(ctx, 2);
            ctx.assert(
              candidates.length >= 2,
              `expected an upsweep and a downsweep kernel taking (data, stride), found ${candidates.length}`
            );
            const { up, down } = await findSweepKernels(ctx, N);
            const probe = sweepProbe(N);
            if (!up) {
              // A for loop, not candidates.map(): the probe call is awaited,
              // and `await` is illegal inside a non-async arrow.
              const seen = [];
              for (const k of candidates) {
                try {
                  const out = await k(Float32Array.from(probe), 1);
                  if (out) seen.push(out);
                } catch (e) {
                  // not a (data, stride) kernel — nothing to learn from it
                }
              }
              const upRef = upsweepPass(probe, 1);
              const hint = seen
                .map(out => diagnoseAll(N, out, upRef, 1e-3, upsweepProbes(probe)))
                .find(Boolean);
              ctx.assert(false, hint ||
                'no upsweep pass found — at stride s only the cells where (i + 1) is a multiple of 2s should change, each to data[i] + data[i - s]');
            }
            if (!down) {
              const seen = [];
              for (const k of candidates) {
                if (k === up) continue;
                try {
                  const out = await k(Float32Array.from(probe), 1);
                  if (out) seen.push(out);
                } catch (e) {
                  // not a (data, stride) kernel — nothing to learn from it
                }
              }
              const downRef = downsweepPass(probe, 1);
              const hint = seen
                .map(out => diagnoseAll(N, out, downRef, 1e-3, downsweepProbes(probe)))
                .find(Boolean);
              ctx.assert(false, hint ||
                'no downsweep pass found — a block top becomes data[i] + data[i - s] while its left partner takes over the top\'s old value, data[i + s]');
            }
          },
        },
        {
          name: 'one pass of each sweep, at stride 4',
          run: async ctx => {
            const { up, down } = await findSweepKernels(ctx, N);
            ctx.assert(up && down, 'expected an upsweep and a downsweep kernel');
            const probe = sweepProbe(N);
            const upOut = await up(Float32Array.from(probe), 4);
            const upRef = upsweepPass(probe, 4);
            const upHint =
              nonFiniteHint(upOut, N, 'a block-top read landed before the start of the array — at stride s the block tops sit at 2s − 1, 4s − 1, …, so the test is (i + 1) % (2 * stride) === 0') ||
              diagnoseAll(N, upOut, upRef, 1e-3, [
                [hardcodedBlockPass(probe, 4), 'the block width is stuck at 2 — at stride s the blocks are 2s wide, so the active cells are where (i + 1) % (2 * stride) === 0'],
                [Array.from(probe), 'the pass changed nothing at stride 4 — it works at stride 1 only if the block width is hardcoded'],
              ]);
            for (let i = 0; i < N; i++) {
              ctx.assertClose(upOut[i], upRef[i], 1e-3, upHint || `upsweep cell ${i} at stride 4`);
            }
            const downOut = await down(Float32Array.from(probe), 4);
            const downRef = downsweepPass(probe, 4);
            const downHint =
              nonFiniteHint(downOut, N, 'a read landed outside the array — a block top reads data[i - stride] and its left partner reads data[i + stride], and both indexes exist for every active cell') ||
              diagnoseAll(N, downOut, downRef, 1e-3, [
                [swappedDownsweepPass(probe, 4), SWAPPED_SWEEP],
                [Array.from(probe), 'the pass changed nothing at stride 4 — it works at stride 1 only if the block width is hardcoded'],
              ]);
            for (let i = 0; i < N; i++) {
              ctx.assertClose(downOut[i], downRef[i], 1e-3, downHint || `downsweep cell ${i} at stride 4`);
            }
          },
        },
        {
          name: 'up, clear, down — the exclusive scan of a fresh array',
          run: async ctx => {
            const { up, down } = await findSweepKernels(ctx, N);
            ctx.assert(up && down, 'expected an upsweep and a downsweep kernel');
            const data = new Array(N);
            for (let i = 0; i < N; i++) data[i] = ((i % 9) + 1) / 8;
            const out = await runSweeps(up, down, data, N);
            const expected = exclusiveScan(data);
            const cleared = clearedLast(upsweptRef(data, N));
            const hint = nonFiniteHint(out, N, STRIDE_GUARD) ||
              diagnoseAll(N, out, expected, 1e-3, [
                [downsweepPass(cleared, N / 2),
                  'only one downsweep pass ran — the sweep needs one pass per level, all the way from stride 512 back down to 1'],
                [cleared,
                  'the downsweep never happened — after the upsweep the array is still a tree of subtotals, not a scan'],
              ]);
            for (let i = 0; i < N; i++) {
              ctx.assertClose(out[i], expected[i], 1e-3, hint || `exclusive cell ${i}`);
            }
          },
        },
        {
          name: 'your driver logged <code>exclusive[512]</code> and the grand total',
          run: async ctx => {
            const data = makeValues(ctx.utils, N, 5109);
            const exclusive = exclusiveScan(data);
            const inclusive = inclusiveScan(data);
            const total = sumOf(data);
            const nums = loggedNumbers(ctx.logs);
            const midHint = loggedHint(nums, exclusive[512], 1e-3, [
              [inclusive[512], 'that is the INCLUSIVE scan at 512 — Blelloch produces the exclusive one, where cell i holds everything BEFORE i'],
            ]);
            ctx.assert(
              loggedContains(nums, exclusive[512], 1e-3),
              midHint || `log exclusive[512] — expected ${exclusive[512]} in the console output`
            );
            const totalHint = loggedHint(nums, total, 1e-3, [
              [exclusive[N - 1], 'that is the last exclusive offset, which leaves out values[1023] — the grand total is what the upsweep left in the root before it was cleared'],
            ]);
            ctx.assert(
              loggedContains(nums, total, 1e-3),
              totalHint || `log the grand total — expected ${total} in the console output`
            );
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            const { up, down } = await findSweepKernels(ctx, N);
            ctx.assert(up && down, 'expected an upsweep and a downsweep kernel');
            const data = makeValues(ctx.utils, N, 6060);
            const out = await runSweeps(up, down, data, N);
            const expected = exclusiveScan(data);
            const hint = nonFiniteHint(out, N, STRIDE_GUARD);
            for (let i = 0; i < N; i++) {
              ctx.assertClose(out[i], expected[i], 1e-3, hint || `exclusive cell ${i}`);
            }
            // The upsweep on its own must leave a reduction tree: the root
            // holding the whole sum, every block top its own block's subtotal.
            let swept = Float32Array.from(data);
            for (let stride = 1; stride < N; stride *= 2) swept = await up(swept, stride);
            ctx.assertClose(swept[N - 1], sumOf(data), 1e-3,
              'after the upsweep the last cell should hold the grand total');
            ctx.assertClose(swept[255], sumOf(data.slice(0, 256)), 1e-3,
              'after the upsweep cell 255 should hold the subtotal of cells 0…255');
          },
        },
      ],
    },

    // ------------------------------------------------------------------ 6
    {
      slug: 'offsets-place-the-data',
      title: 'Payoff: Offsets Place the Data',
      intro: `<p>Now the reason scan is the primitive everything else is built on. Each of the 32
        sessions produces a <em>variable</em> number of output rows — <code>counts[i]</code> of
        them — and they all have to land in one flat 128-slot list, in order, with no gaps. The
        exclusive scan of <code>counts</code> is exactly the array of starting offsets, and it is
        prewired for you here out of tasks 3 and 4.</p>
        <p>On a CPU you would loop the sessions and <em>write</em> each block — a scatter, which
        kernels cannot do. So invert it, the way a gather always inverts a scatter: one thread per
        output <strong>slot</strong>, each asking <em>"which session owns me?"</em>. Slot
        <code>s</code> belongs to session <code>i</code> when
        <code>offsets[i] &lt;= s &lt; offsets[i] + counts[i]</code> — the session whose block has
        already started and has not yet run out. Six sessions here booked nobody; their blocks are
        empty, contain no slot at all, and drop out of the search on their own. An offset exists
        whether or not anything lands on it, which is exactly why the scan has to produce one for
        every session.</p>
        <p>Count, scan, place. That is stream compaction, run-length decoding, sparse-matrix
        assembly, and every "each thread emits a different number of results" problem on a GPU —
        all of them a scan wearing a hat.</p>`,
      goal: `<strong>Goal:</strong> fill 128 slots, each one returning the index of the session
        that owns it.`,
      requirements: [
        'One thread per output slot — <code>output: [128]</code>',
        'Search all 32 sessions with a compile-time loop bound (<code>this.constants.items</code>)',
        'Slot <code>s</code> belongs to session <code>i</code> when <code>offsets[i] &lt;= s</code> <em>and</em> <code>s &lt; offsets[i] + counts[i]</code>',
      ],
      hints: [
        {
          title: 'Hint 1 — invert the question',
          body: `<p>You cannot push a session's rows into the list. Ask the other question —
            <em>whose row lands in MY slot?</em> — and every slot searches the 32 sessions for the
            one whose block contains it.</p>`,
        },
        {
          title: 'Hint 2 — mind the first seat',
          body: `<p>Session <code>i</code> owns slot <code>offsets[i]</code> itself, so the lower
            test needs the equals sign: <code>offsets[i] &lt;= slot</code>, not <code>&lt;</code>.
            Get that wrong and every block's opening seat comes back ownerless.</p>`,
        },
        {
          title: 'Hint 3 — the loop',
          body: `<pre><code>const slot = this.thread.x;
let found = 0;
for (let i = 0; i &lt; this.constants.items; i++) {
  if (offsets[i] &lt;= slot &amp;&amp; slot &lt; offsets[i] + counts[i]) {
    found = i;
  }
}
return found;</code></pre>`,
        },
      ],
      transfer: `Count, scan, place is the standard three-kernel recipe for variable-sized output
        on every platform. <code>thrust::copy_if</code> and
        <code>cub::DeviceSelect::Flagged</code> are a scan of a 0/1 flag array with a gather
        bolted on; a WebGPU or Metal particle system whose sources each emit a different number of
        fragments uses the same scan to decide where each one writes; GPU sparse-matrix builders
        scan row lengths to get row pointers. Without a scan, none of it is parallel.`,
      starterCode: `// 32 sessions, 128 seats, one flat list. Which session owns each seat?
const gpu = new GPU({ mode });
const ITEMS = 32;
const SLOTS = 128;

// Tasks 3 and 4, prewired: counts -> inclusive scan -> starting offsets.
const scanStep = gpu.createKernel(function (data, stride) {
  if (this.thread.x >= stride) {
    return data[this.thread.x] + data[this.thread.x - stride];
  }
  return data[this.thread.x];
}, { output: [ITEMS] });

const toExclusive = gpu.createKernel(function (inclusive) {
  if (this.thread.x === 0) {
    return 0;
  }
  return inclusive[this.thread.x - 1];
}, { output: [ITEMS] });

let v = Float32Array.from(counts);
for (let stride = 1; stride < ITEMS; stride *= 2) {
  v = await scanStep(v, stride);
}
const offsets = await toExclusive(v);

// Your kernel: one thread per SLOT.
const ownerOf = gpu.createKernel(function (offsets, counts) {
  // TODO: search the sessions for the one whose block contains this slot.
  // Session i owns slot s while offsets[i] <= s < offsets[i] + counts[i].
  return 0;
}, { output: [SLOTS], constants: { items: ITEMS } });

const owners = await ownerOf(offsets, counts);
console.log('slots 0-9 belong to sessions:',
  owners[0], owners[1], owners[2], owners[3], owners[4],
  owners[5], owners[6], owners[7], owners[8], owners[9]);
console.log('the last slot belongs to session:', owners[SLOTS - 1]);
`,
      solutionCode: `// 32 sessions, 128 seats, one flat list. Which session owns each seat?
const gpu = new GPU({ mode });
const ITEMS = 32;
const SLOTS = 128;

// Tasks 3 and 4, prewired: counts -> inclusive scan -> starting offsets.
const scanStep = gpu.createKernel(function (data, stride) {
  if (this.thread.x >= stride) {
    return data[this.thread.x] + data[this.thread.x - stride];
  }
  return data[this.thread.x];
}, { output: [ITEMS] });

const toExclusive = gpu.createKernel(function (inclusive) {
  if (this.thread.x === 0) {
    return 0;
  }
  return inclusive[this.thread.x - 1];
}, { output: [ITEMS] });

let v = Float32Array.from(counts);
for (let stride = 1; stride < ITEMS; stride *= 2) {
  v = await scanStep(v, stride);
}
const offsets = await toExclusive(v);

// Your kernel: one thread per SLOT.
const ownerOf = gpu.createKernel(function (offsets, counts) {
  const slot = this.thread.x;
  let found = 0;
  for (let i = 0; i < this.constants.items; i++) {
    if (offsets[i] <= slot && slot < offsets[i] + counts[i]) {
      found = i;
    }
  }
  return found;
}, { output: [SLOTS], constants: { items: ITEMS } });

const owners = await ownerOf(offsets, counts);
console.log('slots 0-9 belong to sessions:',
  owners[0], owners[1], owners[2], owners[3], owners[4],
  owners[5], owners[6], owners[7], owners[8], owners[9]);
console.log('the last slot belongs to session:', owners[SLOTS - 1]);
`,
      inputs: () => ({ counts: COUNTS.slice() }),
      publicTests: [
        {
          name: '128 slots, each naming the session that owns it',
          run: async ctx => {
            ctx.assert(ctx.kernels.length >= 1, 'no kernel was created — call gpu.createKernel()');
            const counts = COUNTS.slice();
            const offsets = exclusiveScan(counts);
            const owner = await findOwnerKernel(ctx, offsets, counts);
            ctx.assert(owner, `no kernel producing ${SLOTS} slots was found — there is one thread per output slot, so output: [${SLOTS}]`);
            const out = await owner(Float32Array.from(offsets), counts);
            const expected = expandCounts(counts);
            const hint = diagnoseAll(SLOTS, out, expected, 1e-6, ownerProbes(counts, offsets, SLOTS));
            for (let slot = 0; slot < SLOTS; slot++) {
              ctx.assertClose(out[slot], expected[slot], 1e-6, hint || `slot ${slot}`);
            }
          },
        },
        {
          name: 'every session gets exactly <code>counts[i]</code> consecutive slots',
          run: async ctx => {
            const counts = COUNTS.slice();
            const offsets = exclusiveScan(counts);
            const owner = await findOwnerKernel(ctx, offsets, counts);
            ctx.assert(owner, `no kernel producing ${SLOTS} slots was found`);
            const out = await owner(Float32Array.from(offsets), counts);
            const tally = new Array(counts.length).fill(0);
            for (let slot = 0; slot < SLOTS; slot++) {
              const named = out[slot];
              ctx.assert(
                Number.isFinite(named) && named >= 0 && named < counts.length,
                `slot ${slot} named session ${named}, which does not exist — there are only ${counts.length}`
              );
              tally[Math.round(named)]++;
            }
            for (let i = 0; i < counts.length; i++) {
              ctx.assertClose(
                tally[i], counts[i], 1e-6,
                counts[i] === 0
                  ? `session ${i} has no sign-ups but owns ${tally[i]} slot(s) — an empty run owns nothing, which is what the upper half of the test is for`
                  : `session ${i} should own exactly ${counts[i]} slots, not ${tally[i]}`
              );
            }
            for (let slot = 1; slot < SLOTS; slot++) {
              ctx.assert(
                out[slot] >= out[slot - 1],
                `slot ${slot} belongs to session ${out[slot]} but slot ${slot - 1} belongs to ${out[slot - 1]} — the blocks have to stay in session order`
              );
            }
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            // A different sheet: an empty session at each end, an empty one in
            // the middle, and two that take a large block on their own.
            const counts = new Array(ITEMS).fill(2);
            counts[0] = 0;
            counts[10] = 0;
            counts[ITEMS - 1] = 0;
            counts[9] = 32;
            counts[20] = 42;
            const offsets = exclusiveScan(counts);
            const owner = await findOwnerKernel(ctx, offsets, counts);
            ctx.assert(owner, `no kernel producing ${SLOTS} slots was found`);
            const out = await owner(Float32Array.from(offsets), counts);
            const expected = expandCounts(counts);
            ctx.assert(expected.length === SLOTS, 'the private sheet should still fill exactly 128 slots');
            const hint = diagnoseAll(SLOTS, out, expected, 1e-6, ownerProbes(counts, offsets, SLOTS));
            for (let slot = 0; slot < SLOTS; slot++) {
              ctx.assertClose(out[slot], expected[slot], 1e-6, hint || `slot ${slot}`);
            }
          },
        },
      ],
    },
  ],
};
