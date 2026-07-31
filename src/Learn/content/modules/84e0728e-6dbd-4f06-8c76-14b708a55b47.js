// Module: Bitonic Sort — uuid 84e0728e-6dbd-4f06-8c76-14b708a55b47 (short id 84e0728e).
// The file name is the uuid; identity lives in the exported object below,
// never in the path. No legacyId: this module postdates the uuid scheme.
//
// Five tasks: the compare-exchange as a gather → partner arithmetic (and what
// gpu.js really does with `^`) → the direction bit that makes a sequence
// bitonic → driving the whole fixed schedule from a JS loop → the payoff, a
// real array sorted and checked against Array.prototype.sort.
//
// Kernel-authoring rules (contract): no closures inside kernel functions,
// every value arrives as an argument or this.thread.*, statically bounded
// loops, Math.* per the whitelist. Every task passes in CPU mode as well as
// GPU mode, and n stays ≤ 256 so CPU-mode verification is fast.
//
// TWO gpu.js 2.20 FACTS THIS MODULE IS BUILT AROUND, both verified against
// ~/Documents/gpu.js and against a real WebGL2 run rather than assumed:
//
//  1. `^` WORKS — but it is not an instruction. Both the WebGL and the WebGL2
//     fragment shaders define `int bitwiseXOR(int, int)` as a loop over up to
//     BIT_COUNT = 32 bits (src/backend/web-gl{,2}/fragment-shader.js); the
//     native GLSL ES 3.00 operator is never emitted. So `i ^ stride` is
//     correct and portable here, and costs a 32-iteration loop per call.
//  2. A BOOLEAN CANNOT BE STORED IN A VARIABLE. `const low = i % 2 === 0;`
//     compiles fine in cpu mode and fails to compile in gpu mode:
//       ERROR: '=' : cannot convert from 'bool' to 'lowp float'
//     Conditions are legal inside `if (...)`, never on the right of `=`. That
//     is why every flag in this module is a 0/1 NUMBER, which turns out to be
//     the better teaching shape anyway: the algorithm really is about two
//     bits of the thread index, and numbers let you compare them directly.
//
// Both of those push the same way, towards the arithmetic formulation:
//   strideBit = Math.floor(i / stride) % 2   // 0 → low member of my pair
//   dirBit    = Math.floor(i / stage)  % 2   // 0 → my block sorts ascending
//   partner   = strideBit === 0 ? i + stride : i - stride
//   keep the SMALLER value exactly when strideBit === dirBit.

// n deterministic values in 0…100, 2 dp. Distinct enough that the near-miss
// probes below can tell "kept the minimum" from "kept the partner's value".
function makeValues(utils, n, seed = 8407) {
  const rand = utils.seededRandom(seed);
  const data = new Array(n);
  for (let i = 0; i < n; i++) data[i] = Math.round(rand() * 10000) / 100;
  return data;
}

// The pad value task 5 uses. Larger than anything in makeValues(), so the
// sentinels sort to the end and the real values keep their order. +Infinity is
// the textbook choice; a finite sentinel is the safe one, because a padded
// array has to survive a round trip through a float texture.
const PAD = 1e6;

function ascending(a, b) {
  return a - b;
}

function sortedCopy(arr) {
  return Array.from(arr).sort(ascending);
}

function isPowerOfTwo(n) {
  return Number.isInteger(n) && n > 0 && (n & (n - 1)) === 0;
}

// Total passes in the network for n inputs: log2(n) stages, and stage s
// contributes log2(s) passes → 1 + 2 + … + log2(n).
function passCount(n) {
  const stages = Math.log2(n);
  return (stages * (stages + 1)) / 2;
}

// The (stage, stride) schedule, flattened to a flat number list in the order
// the two loops produce it. Task 4 has the learner build and PRINT this before
// any data is touched, which makes the schedule itself testable — and makes
// the two ways of misreading the nesting distinguishable at a glance, rather
// than only through whatever the sorted array happens to look like.
function flatSchedule(n, { maxStage = n, stridesAscending = false } = {}) {
  const out = [];
  for (let stage = 2; stage <= maxStage; stage *= 2) {
    const strides = [];
    for (let stride = stage / 2; stride >= 1; stride /= 2) strides.push(stride);
    if (stridesAscending) strides.reverse();
    for (const stride of strides) out.push(stage, stride);
  }
  return out;
}

// Does `nums` contain `pattern` as a contiguous run? Contiguous, so a schedule
// printed with anything interleaved between the pairs does not count as one.
function containsRun(nums, pattern) {
  if (!pattern.length || nums.length < pattern.length) return false;
  for (let i = 0; i + pattern.length <= nums.length; i++) {
    let hit = true;
    for (let j = 0; j < pattern.length; j++) {
      if (nums[i + j] !== pattern[j]) {
        hit = false;
        break;
      }
    }
    if (hit) return true;
  }
  return false;
}

// ---- the reference network ------------------------------------------------
//
// One JS implementation of the whole thing, with switches for the three ways
// the schedule goes wrong. Every expectation and every near-miss signature in
// this module comes out of this one function, so a probe can never disagree
// with the answer it is diagnosing.
//
// Float32Array throughout: the kernels return float32, and comparing a float64
// reference against a float32 result at 1e-3 would otherwise be luck rather
// than arithmetic.
function networkRef(values, size, options = {}) {
  const { maxStage = size, stridesAscending = false, flipDirection = false } = options;
  let v = Float32Array.from(values);
  for (let stage = 2; stage <= maxStage; stage *= 2) {
    const strides = [];
    for (let stride = stage / 2; stride >= 1; stride /= 2) strides.push(stride);
    if (stridesAscending) strides.reverse();
    for (const stride of strides) {
      v = Float32Array.from(passRef(v, size, stage, stride, flipDirection));
    }
  }
  return Array.from(v);
}

// One pass — the kernel of tasks 3-5, written in plain JS.
function passRef(values, size, stage, stride, flipDirection = false) {
  const out = new Float32Array(size);
  for (let i = 0; i < size; i++) {
    const strideBit = Math.floor(i / stride) % 2;
    const dirBit = Math.floor(i / stage) % 2;
    const partner = strideBit === 0 ? i + stride : i - stride;
    const me = values[i];
    const other = values[partner];
    const keepMin = flipDirection ? strideBit !== dirBit : strideBit === dirBit;
    out[i] = keepMin ? Math.min(me, other) : Math.max(me, other);
  }
  return Array.from(out);
}

// Task 2's answer: the partner index for a power-of-two stride.
function partnerRef(i, stride) {
  return Math.floor(i / stride) % 2 === 0 ? i + stride : i - stride;
}

// Drive a learner's pass kernel through the whole schedule. The kernel's own
// output length decides n, so a learner who padded to 256 instead of 128 is
// driven at 256. Float32Array from the first call: gpu.js locks an argument's
// type on the kernel's first invocation, and every pass returns float32 — the
// same wrinkle the reduction ladder has.
function runNetwork(pass, values, size) {
  let v = Float32Array.from(values);
  for (let stage = 2; stage <= size; stage *= 2) {
    for (let stride = stage / 2; stride >= 1; stride /= 2) {
      v = pass(v, stage, stride);
    }
  }
  return Array.from(v);
}

// The output width a kernel was created with, or null.
function outputWidth(k) {
  const output = k && k.kernel && k.kernel.output;
  return output && typeof output[0] === 'number' ? output[0] : null;
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

function logged(logs, value, eps) {
  return loggedNumbers(logs).some(v => Math.abs(v - value) <= eps);
}

// ---- near-miss diagnosis --------------------------------------------------
//
// A failing assert that reports only two numbers tells a learner nothing about
// WHICH slip produced them. A probe pairs the value one specific known mistake
// would produce with a sentence naming that mistake, and it may speak only when
// the observation matches it AND the correct answer does not — so candidates
// that coincide stay silent, as do observations matching probes that disagree
// with each other. A wrong diagnosis is worse than none.
//
// Sorting probes are index-shaped, and here one matching cell is worth nothing:
// in any pair that arrived already in order, half a dozen different mistakes
// produce the same value. So a probe must predict EVERY cell of the array (and
// disagree with the right answer somewhere) before it is allowed to speak.
// Probe values are functions of the index; a missing cell makes the comparison
// NaN, which fails, which keeps the probe quiet.
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

// Task 1: the compare-exchange over adjacent pairs. The mistake this task
// exists to catch is the swap — both threads doing the same thing, so the pair
// ends up holding one value twice (or simply trading places untouched).
function pairProbes(arr) {
  const lo = i => (i % 2 === 0 ? i : i - 1);
  const min = i => Math.min(arr[lo(i)], arr[lo(i) + 1]);
  const max = i => Math.max(arr[lo(i)], arr[lo(i) + 1]);
  return [
    [min,
      'every cell came back holding the smaller of its pair — that is both threads performing ' +
      'the same swap, and the larger value is simply gone. Only the low index keeps the minimum; ' +
      'the high index keeps the maximum'],
    [max,
      'every cell came back holding the larger of its pair — both threads made the same choice, ' +
      'so the smaller value was lost. The low index keeps the minimum, the high index the maximum'],
    [i => (i % 2 === 0 ? max(i) : min(i)),
      'the two roles are the wrong way round — the LOW index of a pair keeps the smaller value'],
    [i => arr[i % 2 === 0 ? i + 1 : i - 1],
      "each cell returned its partner's value untouched — that is a swap, not a compare-exchange. " +
      'A thread has to choose which of the two values it keeps'],
    [i => arr[i],
      'the values came back unchanged — the comparison never happened'],
  ];
}

// Task 2: partner = flip one bit. Stepping always the same way is the slip.
function partnerProbes(stride) {
  return [
    [i => i + stride,
      'every thread stepped FORWARD by the stride, so half of them point outside their own ' +
      'block — and the last ones point past the end of the array. The step goes forward when ' +
      'your bit is 0 and back when it is 1'],
    [i => i - stride,
      'every thread stepped BACK by the stride, so half of them point outside their own block — ' +
      'and the first ones go negative. The step goes forward when your bit is 0 and back when ' +
      'it is 1'],
    [i => i,
      'the partner is the thread itself — no bit was flipped'],
    [i => (i % stride === 0 ? i + stride : i - stride),
      'that tests i % stride, which is your position INSIDE the block. The bit you want is the ' +
      'one the stride names: Math.floor(i / stride) % 2'],
    [i => (i % 2 === 0 ? i + 1 : i - 1),
      'the partner is always the immediate neighbour — the flip has to use the stride it was ' +
      'given, not 1'],
  ];
}

// Task 3: the direction bit. Two failures dominate — taking it from the stride
// (or leaving it out), which makes every pair sort the same way, and inverting
// it. The first two probes share a sentence on purpose: `Math.floor(i / stride)
// % 2` IS strideBit, so "direction from the stride" and "no direction at all"
// produce byte-identical output and cannot be told apart. One message that is
// true of both beats two that cancel each other into silence.
function passProbes(values, size, stage, stride) {
  const sameWay =
    'every pair sorted the same way, so nothing bitonic was built. The direction bit comes from ' +
    'the STAGE, not the stride: Math.floor(i / stage) % 2. (Taking it from the stride gives ' +
    'exactly this output, because then the two bits always agree.)';
  const allAsc = passRef(values, size, size * 2, stride); // dirBit 0 everywhere
  const allDesc = passRef(values, size, size * 2, stride, true);
  return [
    [i => allAsc[i], sameWay],
    [i => allDesc[i], sameWay.replace('every pair sorted the same way', 'every pair sorted descending')],
    [i => passRef(values, size, stage, stride, true)[i],
      'the direction bit is inverted: a block whose bit is 0 sorts ASCENDING, so the low member ' +
      'of a pair keeps the minimum when the two bits agree'],
    [i => values[i],
      'the array came back unchanged — no thread exchanged anything'],
  ];
}

export default {
  uuid: '84e0728e-6dbd-4f06-8c76-14b708a55b47',
  version: 1,
  slug: 'bitonic-sort',
  title: 'Bitonic Sort',
  blurb:
    'More comparisons than quicksort, and far faster on a GPU — because the whole comparison ' +
    'schedule is fixed before the data arrives.',
  tasks: [
    // ------------------------------------------------------------------ 1
    {
      slug: 'compare-exchange',
      title: 'The Compare-Exchange, as a Gather',
      intro: `<p>Quicksort is the wrong algorithm here, and not by a little. How deep it recurses
        depends on the data; its partition step writes elements to positions it only discovers as
        it goes; and neighbouring threads would take different branches on every comparison. Three
        separate ways to be slow. Sorting on a GPU is not a port of a CPU sort — it is a different
        algorithm, and this module builds the one GPUs actually use.</p>
        <p>It is made of a single move repeated: the <strong>compare-exchange</strong>. Take two
        positions, put the smaller value in one and the larger in the other. On a CPU you write
        that as a swap. You cannot here — a thread writes exactly one cell, its own. So both
        threads of a pair compute their own answer by reading <em>both</em> values: the one at the
        low index keeps the minimum, the one at the high index keeps the maximum. Same outcome, no
        thread ever touching another thread's cell.</p>
        <p>Sixteen values, eight pairs: 0 with 1, 2 with 3, and so on. Which of the two you are is
        just <code>this.thread.x % 2</code>.</p>`,
      goal: `<strong>Goal:</strong> make each thread find its partner in the adjacent pair, read
        both values, and return the one it should end up holding — minimum at the even index,
        maximum at the odd one.`,
      requirements: [
        'Work out your partner from <code>this.thread.x</code> alone — even indices pair upward, odd indices pair downward',
        'Read <em>both</em> <code>data[i]</code> and <code>data[partner]</code>',
        'Return <code>Math.min</code> at the even index and <code>Math.max</code> at the odd one — never the partner\'s value unconditionally',
      ],
      hints: [
        {
          title: 'Hint 1 — which half of the pair am I?',
          body: `<p><code>this.thread.x % 2</code> is 0 for the low member of a pair and 1 for the
            high one. Keep it in a variable — but as a <em>number</em>, not a comparison: gpu.js
            cannot store a boolean in a variable (it compiles in cpu mode and fails to compile in
            gpu mode), so write <code>const side = i % 2;</code> and test <code>side === 0</code>
            where you need it.</p>`,
        },
        {
          title: 'Hint 2 — the partner',
          body: `<p>Start from the downward step and correct it for the low member:</p>
<pre><code>let partner = i - 1;
if (side === 0) partner = i + 1;</code></pre>`,
        },
        {
          title: 'Hint 3 — the ending',
          body: `<p>Both values in hand, the choice is one line each way:</p>
<pre><code>const me = data[i];
const other = data[partner];
if (side === 0) return Math.min(me, other);
return Math.max(me, other);</code></pre>`,
        },
      ],
      transfer: `Compare-exchange is the primitive every sorting network is built from, and it is
        gather-shaped everywhere for the same reason: CUDA's <code>__shfl_xor_sync</code> hands a
        thread its partner's value so the thread can decide its own result, WGSL and Metal do the
        same through subgroup shuffles or threadgroup memory plus a barrier. Nobody writes to
        anybody else's slot.`,
      starterCode: `// Eight pairs, sixteen threads. Each thread returns ITS OWN value.
const gpu = new GPU({ mode });

const exchange = gpu.createKernel(function (data) {
  const i = this.thread.x;
  const side = i % 2; // 0 = low member of my pair, 1 = high

  // TODO: work out this thread's partner, read both values, and return
  // the one this thread should hold: min at the low index, max at the high.
  return data[i];
}, { output: [16] });

const result = exchange(data);
console.log(result);
`,
      solutionCode: `// Eight pairs, sixteen threads. Each thread returns ITS OWN value.
const gpu = new GPU({ mode });

const exchange = gpu.createKernel(function (data) {
  const i = this.thread.x;
  const side = i % 2; // 0 = low member of my pair, 1 = high

  let partner = i - 1;
  if (side === 0) partner = i + 1;

  const me = data[i];
  const other = data[partner];
  if (side === 0) return Math.min(me, other);
  return Math.max(me, other);
}, { output: [16] });

const result = exchange(data);
console.log(result);
`,
      inputs: utils => ({ data: makeValues(utils, 16, 1601) }),
      publicTests: [
        {
          name: 'kernel returns 16 values — one per thread',
          run: async ctx => {
            ctx.assert(ctx.kernels.length >= 1, 'no kernel was created — call gpu.createKernel()');
            const out = ctx.kernel(makeValues(ctx.utils, 16, 1601));
            ctx.assert(out && out.length === 16, `expected 16 output values, got ${out && out.length}`);
          },
        },
        {
          name: 'each pair ends up sorted: <code>min</code> at the even index, <code>max</code> at the odd',
          run: async ctx => {
            const arr = [8, 3, 1, 9, 6, 2, 7, 4, 15, 11, 13, 10, 5, 12, 0, 14];
            const out = ctx.kernel(arr);
            const expected = i =>
              i % 2 === 0 ? Math.min(arr[i], arr[i + 1]) : Math.max(arr[i - 1], arr[i]);
            const hint = diagnoseAll(16, i => out[i], expected, 1e-3, pairProbes(arr));
            for (let i = 0; i < 16; i++) {
              ctx.assertClose(out[i], expected(i), 1e-3, hint || `cell ${i}`);
            }
          },
        },
        {
          name: 'nothing is lost — the 16 values are the same 16, rearranged',
          run: async ctx => {
            const arr = makeValues(ctx.utils, 16, 4242);
            const out = ctx.kernel(arr);
            const before = sortedCopy(arr);
            const after = sortedCopy(out);
            for (let i = 0; i < 16; i++) {
              ctx.assertClose(
                after[i], before[i], 1e-3,
                'a compare-exchange rearranges values, it never invents or drops one — this ' +
                'result is not a permutation of the input, so some pair kept the same value twice'
              );
            }
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            const arr = makeValues(ctx.utils, 16, 77771);
            const out = ctx.kernel(arr);
            ctx.assert(out && out.length === 16, 'expected 16 output values');
            const expected = i =>
              i % 2 === 0 ? Math.min(arr[i], arr[i + 1]) : Math.max(arr[i - 1], arr[i]);
            const hint = diagnoseAll(16, i => out[i], expected, 1e-3, pairProbes(arr));
            for (let i = 0; i < 16; i++) {
              ctx.assertClose(out[i], expected(i), 1e-3, hint || `cell ${i}`);
            }
          },
        },
        {
          name: 'private test #2',
          run: async ctx => {
            // Already-sorted input: every cell must come back untouched. A
            // kernel that returns its partner's value fails here and nowhere
            // near it looks like a comparison mistake.
            const arr = new Array(16);
            for (let i = 0; i < 16; i++) arr[i] = i * 1.5;
            const out = ctx.kernel(arr);
            for (let i = 0; i < 16; i++) {
              ctx.assertClose(
                out[i], arr[i], 1e-3,
                'every pair of this input was already in order, so the pass should change ' +
                `nothing — cell ${i} moved anyway`
              );
            }
          },
        },
      ],
    },

    // ------------------------------------------------------------------ 2
    {
      slug: 'partner-index',
      title: 'Who Is My Partner?',
      intro: `<p>Adjacent pairs are only the first pass. The network also compares at distance 2,
        4, 8, 16 — always a power of two, always the same pattern regardless of what the values
        are. The classic way to write it is one character long:</p>
<pre><code>partner = i ^ stride;</code></pre>
        <p>XOR with a power of two flips exactly one bit of the index — bit
        <code>log₂(stride)</code>. If your bit is 0 you move forward across the gap; if it is 1 you
        move back. Do it twice and you are home, which is why the pairing is always mutual: no
        thread is anybody's partner twice, and nobody is left over.</p>
        <p><code>^</code> does work in gpu.js — but it is worth knowing what you are buying. Both
        WebGL backends compile it to a helper function that walks up to 32 bits of both operands in
        a loop; the native GLSL integer operator is never emitted. One character of JavaScript, a
        32-iteration loop in the shader. On CUDA or WebGPU, XOR is a single instruction. Here it is
        not, so this module spells the flip out in arithmetic instead — two operations, and it
        hands you something XOR hides: the <em>value</em> of the bit you are flipping.</p>
<pre><code>const bit = Math.floor(i / stride) % 2;   // 0 or 1 — my bit at log2(stride)
partner = bit === 0 ? i + stride : i - stride;</code></pre>
        <p>Hold on to that <code>bit</code>. The next task needs it, and needs a second one just
        like it.</p>`,
      goal: `<strong>Goal:</strong> return the <em>partner index</em> for each of 16 threads at a
        given power-of-two <code>stride</code> — no data involved, pure index arithmetic.`,
      requirements: [
        'The kernel takes <code>stride</code> as an argument and returns an index, not a value',
        'Work for <em>any</em> power-of-two stride — 1, 2, 4 and 8 all have to come out right',
        'Pairing must be mutual: the partner of your partner is you',
      ],
      hints: [
        {
          title: 'Hint 1 — which bit?',
          body: `<p>At <code>stride = 4</code> the indices 0…7 split into 0–3 (bit clear, step
            forward) and 4–7 (bit set, step back). <code>Math.floor(i / 4) % 2</code> is exactly
            that split: 0, 0, 0, 0, 1, 1, 1, 1.</p>`,
        },
        {
          title: 'Hint 2 — the whole kernel',
          body: `<pre><code>const i = this.thread.x;
const bit = Math.floor(i / stride) % 2;
if (bit === 0) return i + stride;
return i - stride;</code></pre>
<p><code>return i ^ stride;</code> gives the same answers, at the cost of that
            32-iteration loop — and it will not help you with the next task.</p>`,
        },
      ],
      transfer: `The XOR partner is the canonical spelling of a sorting network everywhere:
        CUDA's <code>__shfl_xor_sync(mask, value, laneMask)</code> takes the lane XOR mask
        directly, and WGSL's <code>subgroupShuffleXor</code> is named after it. Both are one
        instruction on the hardware — worth remembering that the arithmetic form you write here is
        a gpu.js accommodation, not a universal truth.`,
      starterCode: `// Pure index arithmetic: no data, no comparison. Just "who do I pair with?"
const gpu = new GPU({ mode });

const partner = gpu.createKernel(function (stride) {
  const i = this.thread.x;

  // TODO: flip the bit that \`stride\` names.
  // Which bit is it? Math.floor(i / stride) % 2 tells you its value —
  // 0 means step forward by stride, 1 means step back.
  return i;
}, { output: [16] });

console.log('stride 1:', partner(1));
console.log('stride 4:', partner(4));
`,
      solutionCode: `// Pure index arithmetic: no data, no comparison. Just "who do I pair with?"
const gpu = new GPU({ mode });

const partner = gpu.createKernel(function (stride) {
  const i = this.thread.x;
  const bit = Math.floor(i / stride) % 2;
  if (bit === 0) return i + stride;
  return i - stride;
}, { output: [16] });

console.log('stride 1:', partner(1));
console.log('stride 4:', partner(4));
`,
      publicTests: [
        {
          name: 'stride 1 pairs neighbours: <code>1, 0, 3, 2, 5, 4, …</code>',
          run: async ctx => {
            ctx.assert(ctx.kernels.length >= 1, 'no kernel was created — call gpu.createKernel()');
            const out = ctx.kernel(1);
            ctx.assert(out && out.length === 16, `expected 16 output values, got ${out && out.length}`);
            const hint = diagnoseAll(16, i => out[i], i => partnerRef(i, 1), 1e-3, partnerProbes(1));
            for (let i = 0; i < 16; i++) {
              ctx.assertClose(out[i], partnerRef(i, 1), 1e-3, hint || `thread ${i}`);
            }
          },
        },
        {
          name: 'strides 2, 4 and 8 flip the right bit',
          run: async ctx => {
            for (const stride of [2, 4, 8]) {
              const out = ctx.kernel(stride);
              const hint = diagnoseAll(
                16, i => out[i], i => partnerRef(i, stride), 1e-3, partnerProbes(stride)
              );
              for (let i = 0; i < 16; i++) {
                ctx.assertClose(out[i], partnerRef(i, stride), 1e-3, hint || `stride ${stride}, thread ${i}`);
              }
            }
          },
        },
        {
          name: 'the pairing is mutual and stays inside the array',
          run: async ctx => {
            for (const stride of [1, 2, 4, 8]) {
              const out = ctx.kernel(stride);
              for (let i = 0; i < 16; i++) {
                const p = Math.round(out[i]);
                ctx.assert(
                  p >= 0 && p < 16,
                  `at stride ${stride}, thread ${i} points at index ${out[i]}, which is outside ` +
                  'the array — a bit flip never leaves the block, but a plain step in one ' +
                  'direction does'
                );
                ctx.assert(
                  p !== i,
                  `at stride ${stride}, thread ${i} chose itself as its partner — flipping a bit ` +
                  'always changes the index'
                );
                ctx.assertClose(
                  out[p], i, 1e-3,
                  `at stride ${stride} thread ${i} points at ${p}, but ${p} points at ${out[p]} ` +
                  '— the pairing has to be mutual, or some thread is nobody\'s partner'
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
            // Stride 16 pushes every partner outside a 16-wide output: the
            // arithmetic is still well defined, and a thread that has quietly
            // hard-coded a smaller stride is caught here.
            for (const stride of [1, 2, 4, 8, 16]) {
              const out = ctx.kernel(stride);
              const hint = diagnoseAll(
                16, i => out[i], i => partnerRef(i, stride), 1e-3, partnerProbes(stride)
              );
              for (let i = 0; i < 16; i++) {
                ctx.assertClose(out[i], partnerRef(i, stride), 1e-3, hint || `stride ${stride}, thread ${i}`);
              }
            }
          },
        },
      ],
    },

    // ------------------------------------------------------------------ 3
    {
      slug: 'direction-bit',
      title: 'Which Way Does My Pair Sort?',
      intro: `<p>Every pass so far sorted every pair the same way. A bitonic network does not, and
        that is the whole trick. A sequence that rises and then falls is called <strong>bitonic</strong>,
        and a bitonic sequence is the one thing this network can merge into sorted order in log n
        passes. So the early passes exist to <em>build</em> bitonic runs: neighbouring blocks are
        deliberately sorted in opposite directions, so that gluing two of them together gives up
        then down.</p>
        <p>Which way your block goes is another bit of your index — the one named by
        <code>stage</code>, the size of the block currently being merged:</p>
<pre><code>const dirBit = Math.floor(i / stage) % 2;   // 0 → my block sorts ascending</code></pre>
        <p>So a thread now holds two bits. <code>strideBit</code> says whether it is the low or the
        high member of its pair; <code>dirBit</code> says which way its block is sorting. And the
        rule is as small as it could be: <strong>keep the smaller value exactly when the two bits
        agree.</strong> Low member of an ascending block, or high member of a descending one —
        either way, minimum.</p>
        <p>One detail makes that legal: <code>stride</code> is always smaller than
        <code>stage</code>, so flipping the stride bit never disturbs the direction bit. Both
        members of a pair read the same <code>dirBit</code> and agree about which way they are
        sorting — without exchanging a word.</p>`,
      goal: `<strong>Goal:</strong> write one full bitonic pass over 8 values — the kernel takes
        <code>(data, stage, stride)</code> and returns each thread's new value.`,
      requirements: [
        'Compute both bits: <code>strideBit</code> from <code>stride</code>, <code>dirBit</code> from <code>stage</code>',
        'The partner still comes from <code>strideBit</code>, exactly as in the last task',
        'Return <code>Math.min</code> when the two bits agree and <code>Math.max</code> when they differ',
      ],
      hints: [
        {
          title: 'Hint 1 — reading the rule off the table',
          body: `<p>Four cases, and they collapse to one comparison:</p>
<pre><code>low  + ascending  → min      (0, 0) agree
high + ascending  → max      (1, 0) differ
low  + descending → max      (0, 1) differ
high + descending → min      (1, 1) agree</code></pre>`,
        },
        {
          title: 'Hint 2 — the whole body',
          body: `<pre><code>const i = this.thread.x;
const strideBit = Math.floor(i / stride) % 2;
const dirBit = Math.floor(i / stage) % 2;

let partner = i - stride;
if (strideBit === 0) partner = i + stride;

const me = data[i];
const other = data[partner];
if (strideBit === dirBit) return Math.min(me, other);
return Math.max(me, other);</code></pre>`,
        },
      ],
      transfer: `Every bitonic implementation on every platform carries this pair of bit tests —
        CUDA samples write <code>(i &amp; k) == 0</code>, WGSL compute shaders write the same thing
        with <code>&amp;</code>, and the arithmetic spelling here says exactly the same. What none
        of them need is communication: the direction is a property of your index, so a thread can
        work it out alone, which is what makes the whole network barrier-free within a pass.`,
      starterCode: `// One pass of the network: (data, stage, stride) in, one value per thread out.
const gpu = new GPU({ mode });

const pass = gpu.createKernel(function (data, stage, stride) {
  const i = this.thread.x;
  const strideBit = Math.floor(i / stride) % 2;

  // TODO: work out dirBit from \`stage\` the same way strideBit comes
  // from \`stride\`, find your partner, and keep the SMALLER value when
  // the two bits agree.
  let partner = i - stride;
  if (strideBit === 0) partner = i + stride;

  const me = data[i];
  const other = data[partner];
  if (strideBit === 0) return Math.min(me, other);
  return Math.max(me, other);
}, { output: [8] });

// stage 2, stride 1: pairs (0,1) and (4,5) sort up, (2,3) and (6,7) sort down.
console.log(pass(data, 2, 1));
`,
      solutionCode: `// One pass of the network: (data, stage, stride) in, one value per thread out.
const gpu = new GPU({ mode });

const pass = gpu.createKernel(function (data, stage, stride) {
  const i = this.thread.x;
  const strideBit = Math.floor(i / stride) % 2;
  const dirBit = Math.floor(i / stage) % 2;

  let partner = i - stride;
  if (strideBit === 0) partner = i + stride;

  const me = data[i];
  const other = data[partner];
  if (strideBit === dirBit) return Math.min(me, other);
  return Math.max(me, other);
}, { output: [8] });

// stage 2, stride 1: pairs (0,1) and (4,5) sort up, (2,3) and (6,7) sort down.
console.log(pass(data, 2, 1));
`,
      inputs: utils => ({ data: makeValues(utils, 8, 3103) }),
      publicTests: [
        {
          name: 'stage 2, stride 1 — alternating pairs sort up, down, up, down',
          run: async ctx => {
            ctx.assert(ctx.kernels.length >= 1, 'no kernel was created — call gpu.createKernel()');
            const arr = [5, 2, 9, 1, 7, 3, 8, 4];
            const out = ctx.kernel(arr, 2, 1);
            ctx.assert(out && out.length === 8, `expected 8 output values, got ${out && out.length}`);
            const expected = passRef(arr, 8, 2, 1);
            const hint = diagnoseAll(
              8, i => out[i], i => expected[i], 1e-3, passProbes(arr, 8, 2, 1)
            );
            for (let i = 0; i < 8; i++) {
              ctx.assertClose(out[i], expected[i], 1e-3, hint || `cell ${i}`);
            }
          },
        },
        {
          name: 'the wider passes too — <code>(4, 2)</code>, <code>(4, 1)</code>, <code>(8, 4)</code>',
          run: async ctx => {
            const arr = [12, 5, 3, 14, 9, 1, 7, 11];
            for (const [stage, stride] of [[4, 2], [4, 1], [8, 4], [8, 2], [8, 1]]) {
              const out = ctx.kernel(arr, stage, stride);
              const expected = passRef(arr, 8, stage, stride);
              const hint = diagnoseAll(
                8, i => out[i], i => expected[i], 1e-3, passProbes(arr, 8, stage, stride)
              );
              for (let i = 0; i < 8; i++) {
                ctx.assertClose(out[i], expected[i], 1e-3, hint || `stage ${stage}, stride ${stride}, cell ${i}`);
              }
            }
          },
        },
        {
          name: 'a pass rearranges values — it never invents or drops one',
          run: async ctx => {
            const arr = makeValues(ctx.utils, 8, 5051);
            const out = ctx.kernel(arr, 4, 2);
            const before = sortedCopy(arr);
            const after = sortedCopy(out);
            for (let i = 0; i < 8; i++) {
              ctx.assertClose(
                after[i], before[i], 1e-3,
                'this result is not a permutation of the input — some pair kept the same value ' +
                'twice, which means both of its threads made the same choice'
              );
            }
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            const arr = makeValues(ctx.utils, 8, 8642);
            for (const [stage, stride] of [[2, 1], [4, 2], [4, 1], [8, 4], [8, 2], [8, 1]]) {
              const out = ctx.kernel(arr, stage, stride);
              const expected = passRef(arr, 8, stage, stride);
              const hint = diagnoseAll(
                8, i => out[i], i => expected[i], 1e-3, passProbes(arr, 8, stage, stride)
              );
              for (let i = 0; i < 8; i++) {
                ctx.assertClose(out[i], expected[i], 1e-3, hint || `stage ${stage}, stride ${stride}, cell ${i}`);
              }
            }
          },
        },
        {
          name: 'private test #2',
          run: async ctx => {
            // The full three-stage network on 8 values, driven pass by pass.
            // A kernel that is right only at stride 1 dies here.
            //
            // Plain arrays in and out of every call: gpu.js locks an argument's
            // type on the kernel's first invocation, and this task's own run
            // hands it a plain array. Feeding the returned Float32Array back in
            // would be a different type and would throw in gpu mode — which is
            // why tasks 4 and 5 start from a Float32Array instead.
            const arr = makeValues(ctx.utils, 8, 1357);
            let v = arr.slice();
            for (let stage = 2; stage <= 8; stage *= 2) {
              for (let stride = stage / 2; stride >= 1; stride /= 2) {
                v = Array.from(ctx.kernel(v, stage, stride));
              }
            }
            const got = v;
            const expected = sortedCopy(Float32Array.from(arr));
            for (let i = 0; i < 8; i++) {
              ctx.assertClose(
                got[i], expected[i], 1e-3,
                `running all six passes should leave the 8 values sorted — position ${i} does not match`
              );
            }
          },
        },
      ],
    },

    // ------------------------------------------------------------------ 4
    {
      slug: 'the-network',
      title: 'Drive the Whole Network',
      intro: `<p>One pass is one kernel launch. The network is two plain JavaScript loops around
        it — stages doubling outward, and within each stage strides halving down to 1. Here is the
        thing worth noticing, though: those loops never look at the data. So don't interleave them
        with it. Build the entire <strong>schedule</strong> first, as a list of
        <code>[stage, stride]</code> pairs, and print it:</p>
<pre><code>const schedule = [];
for (let stage = 2; stage &lt;= n; stage *= 2) {
  for (let stride = stage / 2; stride &gt;= 1; stride /= 2) {
    schedule.push([stage, stride]);
  }
}</code></pre>
        <p>All 36 pairs for n = 256, complete, before a single value has been read. Then run them.
        That is the property this whole module is about: every thread derives its partner and its
        direction from its own index, so nothing waits on a comparison, no warp diverges, and
        nobody has to be told anything. A quicksort cannot do this — you do not know its second
        partition until you have done the first.</p>
        <p>The bill comes due in comparisons. Bitonic sort does O(n log²n) of them where quicksort
        does O(n log n): 36 passes × 128 pairs = 4,608 compare-exchanges for 256 values, against
        roughly 2,000. More than twice the work — in 36 sequential steps, with everything inside a
        step happening at once. That trade, a predictable structure bought with extra work, is the
        most transferable idea in this course.</p>`,
      goal: `<strong>Goal:</strong> build and print the whole 36-pass schedule before touching the
        data, run it to sort 256 values, and log the smallest and largest of the result.`,
      requirements: [
        '<code>schedule</code> holds the <code>[stage, stride]</code> pairs: stages doubling from 2 up to and <em>including</em> <code>n</code>, strides halving from <code>stage / 2</code> down to 1',
        'Build it without reading <code>data</code> — the schedule is complete before the first kernel call',
        '<code>console.log</code> the pass count and the schedule itself (already wired up)',
        '<code>console.log</code> the smallest and largest values of the sorted result',
      ],
      hints: [
        {
          title: 'Hint 1 — the two loops',
          body: `<p>Outer loop doubles, inner loop halves, and both bounds are inclusive at the
            far end: <code>stage &lt;= n</code>, <code>stride &gt;= 1</code>. The body is one
            line — <code>schedule.push([stage, stride]);</code></p>`,
        },
        {
          title: 'Hint 2 — the last stage is the one that sorts',
          body: `<p>Stopping at <code>stage &lt; n</code> costs exactly one merge, and that merge
            is the one that turns a bitonic sequence into a sorted array. The result looks
            plausible — it rises, then falls — and it is wrong.</p>`,
        },
        {
          title: 'Hint 3 — the first few pairs',
          body: `<p>A correct schedule starts</p>
<pre><code>[[2,1], [4,2], [4,1], [8,4], [8,2], [8,1], [16,8], …]</code></pre>
<p>— stride 2 <em>before</em> stride 1 inside stage 4, not after.</p>`,
        },
      ],
      transfer: `A host-side loop issuing one kernel launch per pass is exactly how bitonic sort
        ships in practice: CUDA samples launch <code>bitonicSortShared</code> once per (stage,
        stride), WebGPU records one dispatch per pass into a command encoder, and Metal encodes one
        compute pass each. The launches are the synchronisation — everything within a pass is
        independent, and the boundary between passes is the only barrier anyone needs.`,
      starterCode: `// The schedule first, the data second. One kernel launch per pass.
const gpu = new GPU({ mode });
const n = 256;

const pass = gpu.createKernel(function (data, stage, stride) {
  const i = this.thread.x;
  const strideBit = Math.floor(i / stride) % 2;
  const dirBit = Math.floor(i / stage) % 2;

  let partner = i - stride;
  if (strideBit === 0) partner = i + stride;

  const me = data[i];
  const other = data[partner];
  if (strideBit === dirBit) return Math.min(me, other);
  return Math.max(me, other);
}, { output: [n] });

const schedule = [];
// TODO: fill \`schedule\` with every [stage, stride] pair — stages doubling
// 2 → n, and within each stage strides halving stage / 2 → 1.
// Notice that nothing in here can look at \`data\`. That is the point.

console.log('passes:', schedule.length);
console.log('schedule:', JSON.stringify(schedule));

// Float32Array from the start: gpu.js locks an argument's type on the first
// call, and every pass hands back a Float32Array.
let values = Float32Array.from(data);
for (let i = 0; i < schedule.length; i++) {
  values = pass(values, schedule[i][0], schedule[i][1]);
}

console.log('smallest:', values[0]);
console.log('largest:', values[n - 1]);
`,
      solutionCode: `// The schedule first, the data second. One kernel launch per pass.
const gpu = new GPU({ mode });
const n = 256;

const pass = gpu.createKernel(function (data, stage, stride) {
  const i = this.thread.x;
  const strideBit = Math.floor(i / stride) % 2;
  const dirBit = Math.floor(i / stage) % 2;

  let partner = i - stride;
  if (strideBit === 0) partner = i + stride;

  const me = data[i];
  const other = data[partner];
  if (strideBit === dirBit) return Math.min(me, other);
  return Math.max(me, other);
}, { output: [n] });

const schedule = [];
for (let stage = 2; stage <= n; stage *= 2) {
  for (let stride = stage / 2; stride >= 1; stride /= 2) {
    schedule.push([stage, stride]);
  }
}

console.log('passes:', schedule.length);
console.log('schedule:', JSON.stringify(schedule));

// Float32Array from the start: gpu.js locks an argument's type on the first
// call, and every pass hands back a Float32Array.
let values = Float32Array.from(data);
for (let i = 0; i < schedule.length; i++) {
  values = pass(values, schedule[i][0], schedule[i][1]);
}

console.log('smallest:', values[0]);
console.log('largest:', values[n - 1]);
`,
      inputs: utils => ({ data: makeValues(utils, 256, 2560) }),
      publicTests: [
        {
          name: 'the pass kernel drives a full 256-value sort',
          run: async ctx => {
            ctx.assert(ctx.kernels.length >= 1, 'no kernel was created — call gpu.createKernel()');
            ctx.assert(
              outputWidth(ctx.kernel) === 256,
              `the kernel's output should be [256], got ${JSON.stringify(outputWidth(ctx.kernel))}`
            );
            const arr = makeValues(ctx.utils, 256, 191);
            const got = runNetwork(ctx.kernel, arr, 256);
            const expected = sortedCopy(Float32Array.from(arr));
            for (let i = 0; i < 256; i++) {
              ctx.assertClose(got[i], expected[i], 1e-3, `sorted position ${i}`);
            }
          },
        },
        {
          name: 'the whole schedule is logged — <code>36</code> pairs, in order, before any data is read',
          run: async ctx => {
            const nums = loggedNumbers(ctx.logs);
            const right = flatSchedule(256);
            // Both wrong nestings are checked against the numbers the learner
            // actually printed, and only once the correct schedule is known to
            // be absent. They cannot both match: the short schedule is a prefix
            // of the right one, so it only appears when the stages stop early,
            // and the reversed-stride schedule diverges at the third pair.
            let hint = null;
            if (!containsRun(nums, right)) {
              if (containsRun(nums, flatSchedule(256, { stridesAscending: true }))) {
                hint =
                  'the schedule is all there, but the strides run the wrong way inside each ' +
                  'stage — as printed, stage 4 asks for stride 1 before stride 2. Within a stage ' +
                  'they HALVE, from stage / 2 down to 1';
              } else if (containsRun(nums, flatSchedule(256, { maxStage: 128 }))) {
                hint =
                  'the schedule stops after stage 128 — the outer loop has to run while ' +
                  'stage <= n, not stage < n. That last merge is the one that turns a bitonic ' +
                  'sequence into a sorted array';
              }
            }
            ctx.assert(
              containsRun(nums, right),
              hint ||
              `log the whole schedule — expected its ${passCount(256)} (stage, stride) pairs in ` +
              'order, starting [[2,1],[4,2],[4,1],[8,4],…] and ending […,[256,2],[256,1]]'
            );
          },
        },
        {
          name: 'the sorted result is logged, smallest and largest',
          run: async ctx => {
            const data = makeValues(ctx.utils, 256, 2560);
            const sorted = sortedCopy(Float32Array.from(data));
            const smallest = sorted[0];
            const largest = sorted[255];
            ctx.assert(
              logged(ctx.logs, smallest, 1e-2),
              `log the smallest value of the sorted result — expected ≈${smallest.toFixed(2)} ` +
              'in the console output'
            );
            // Each way of misreading the schedule leaves one specific value in
            // the last slot — a 2-decimal number out of 256, which nothing else
            // in this run would log by accident. Both signatures are checked
            // against what the learner actually logged; the diagnosis speaks
            // only when the right answer is absent and exactly one signature is
            // present, so a run matching both stays silent.
            const signatures = [
              [networkRef(data, 256, { maxStage: 128 })[255],
                'that is the last element of a network that stopped one stage early — with ' +
                'stage < n the array comes back bitonic (rising, then falling) rather than ' +
                'sorted, so its last element is small. The bound is stage <= n'],
              [networkRef(data, 256, { stridesAscending: true })[255],
                'that is the last element you get with the strides running the wrong way — ' +
                'within a stage they must HALVE, from stage / 2 down to 1, not grow'],
            ].filter(([value]) => Math.abs(value - largest) > 1e-2 && logged(ctx.logs, value, 1e-2));
            const hint = signatures.length === 1 ? signatures[0][1] : null;
            ctx.assert(
              logged(ctx.logs, largest, 1e-2),
              hint || `log the largest value of the sorted result — expected ≈${largest.toFixed(2)} ` +
              'in the console output'
            );
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            // Duplicates and negatives: nothing about the network cares, and a
            // comparison written as a subtraction would.
            const arr = new Array(256);
            for (let i = 0; i < 256; i++) arr[i] = ((i * 37) % 17) - 8;
            const got = runNetwork(ctx.kernel, arr, 256);
            const expected = sortedCopy(Float32Array.from(arr));
            for (let i = 0; i < 256; i++) {
              ctx.assertClose(got[i], expected[i], 1e-3, `sorted position ${i}`);
            }
          },
        },
        {
          name: 'private test #2',
          run: async ctx => {
            // Already sorted, and exactly reversed — the two inputs a network
            // with an inverted direction bit gets away with least.
            const up = new Array(256);
            for (let i = 0; i < 256; i++) up[i] = i * 0.25;
            const down = up.slice().reverse();
            for (const arr of [up, down]) {
              const got = runNetwork(ctx.kernel, arr, 256);
              for (let i = 0; i < 256; i++) {
                ctx.assertClose(
                  got[i], up[i], 1e-3,
                  `sorted position ${i} — the result must come out ascending whichever order ` +
                  'the input arrived in'
                );
              }
            }
          },
        },
      ],
    },

    // ------------------------------------------------------------------ 5
    {
      slug: 'sort-it',
      title: 'Payoff: Sort a Real Array',
      intro: `<p>One constraint has been quietly true all along: <strong>n must be a power of
        two</strong>. Every thread's partner is its index with one bit flipped, and that only
        lands inside the array when the array fills the whole index space. Give the network 100
        values and threads near the top reach for elements that do not exist — no error, no
        warning, just a result that is quietly wrong in a way that is very hard to see.</p>
        <p>The fix is the one every real implementation uses: <strong>pad up to the next power of
        two</strong> with a sentinel that is guaranteed to sort to the end, run the network on the
        padded array, then slice the padding off. 100 values become 128, sorted, and the last 28
        slots come back full of sentinel. Padding costs a little wasted work and buys you an
        algorithm with no special cases at all.</p>
        <p>+Infinity is the textbook sentinel. This task uses a large finite one instead, because
        a padded array has to survive a round trip through a float texture, and a finite value
        always does. Anything comfortably above your data's maximum works.</p>`,
      goal: `<strong>Goal:</strong> sort all 100 of <code>values</code> by padding up to a power of
        two, running the network, and dropping the padding — then log the smallest and largest of
        the <em>real</em> values.`,
      requirements: [
        'Choose <code>size</code> = the next power of two at or above <code>values.length</code>, and create the kernel with <code>output: [size]</code>',
        'Pad with <code>PAD</code> up to <code>size</code>, run the full stage/stride schedule, then take the first <code>values.length</code> results',
        '<code>console.log</code> the smallest and the largest of the sorted <em>real</em> values — not of the padded array',
      ],
      hints: [
        {
          title: 'Hint 1 — the next power of two',
          body: `<p>Double until you clear the length:</p>
<pre><code>let size = 1;
while (size &lt; values.length) size *= 2;</code></pre>
<p>For 100 values that lands on 128.</p>`,
        },
        {
          title: 'Hint 2 — padding and un-padding',
          body: `<p>Pad before, slice after:</p>
<pre><code>const padded = values.slice();
while (padded.length &lt; size) padded.push(PAD);
// … run the network …
const sorted = Array.from(result).slice(0, values.length);</code></pre>
<p>The sentinels all sort to the end, so the real values keep the front of the
            array in exactly the right order.</p>`,
        },
        {
          title: 'Hint 3 — reading the answer',
          body: `<p><code>result[size - 1]</code> is a sentinel, not your largest value. The
            largest real value is <code>sorted[values.length - 1]</code>, after the slice.</p>`,
        },
      ],
      transfer: `Power-of-two padding is what every production sorter does with this network:
        CUDA's bitonic samples require it outright and pad in the host code, and library sorts
        (CUB, rocPRIM, WebGPU's community sort implementations) hide the same padding inside a
        friendlier signature. The general-case handling never lives in the kernel — it lives in the
        few lines around it, exactly where you just put it.`,
      starterCode: `// 100 values. The network needs a power of two — so give it one.
const gpu = new GPU({ mode });

// TODO: the next power of two at or above values.length (100 → 128).
const size = values.length;

const pass = gpu.createKernel(function (data, stage, stride) {
  const i = this.thread.x;
  const strideBit = Math.floor(i / stride) % 2;
  const dirBit = Math.floor(i / stage) % 2;

  let partner = i - stride;
  if (strideBit === 0) partner = i + stride;

  const me = data[i];
  const other = data[partner];
  if (strideBit === dirBit) return Math.min(me, other);
  return Math.max(me, other);
}, { output: [size] });

// TODO: pad \`values\` up to \`size\` with PAD before sorting.
const padded = values.slice();

let result = Float32Array.from(padded);
for (let stage = 2; stage <= size; stage *= 2) {
  for (let stride = stage / 2; stride >= 1; stride /= 2) {
    result = pass(result, stage, stride);
  }
}

// TODO: drop the padding before reading the answer.
const sorted = Array.from(result);

console.log('smallest:', sorted[0]);
console.log('largest:', sorted[sorted.length - 1]);

const reference = values.slice().sort((a, b) => a - b);
console.log('matches Array.prototype.sort:',
  sorted.length === reference.length &&
  sorted.every((v, i) => Math.abs(v - reference[i]) < 1e-3));
`,
      solutionCode: `// 100 values. The network needs a power of two — so give it one.
const gpu = new GPU({ mode });

let size = 1;
while (size < values.length) size *= 2;

const pass = gpu.createKernel(function (data, stage, stride) {
  const i = this.thread.x;
  const strideBit = Math.floor(i / stride) % 2;
  const dirBit = Math.floor(i / stage) % 2;

  let partner = i - stride;
  if (strideBit === 0) partner = i + stride;

  const me = data[i];
  const other = data[partner];
  if (strideBit === dirBit) return Math.min(me, other);
  return Math.max(me, other);
}, { output: [size] });

// The sentinels sort to the end, so the real values keep the front.
const padded = values.slice();
while (padded.length < size) padded.push(PAD);

let result = Float32Array.from(padded);
for (let stage = 2; stage <= size; stage *= 2) {
  for (let stride = stage / 2; stride >= 1; stride /= 2) {
    result = pass(result, stage, stride);
  }
}

const sorted = Array.from(result).slice(0, values.length);

console.log('smallest:', sorted[0]);
console.log('largest:', sorted[sorted.length - 1]);

const reference = values.slice().sort((a, b) => a - b);
console.log('matches Array.prototype.sort:',
  sorted.length === reference.length &&
  sorted.every((v, i) => Math.abs(v - reference[i]) < 1e-3));
`,
      inputs: utils => ({ values: makeValues(utils, 100, 10007), PAD }),
      publicTests: [
        {
          name: 'the kernel is sized to a power of two at or above 100',
          run: async ctx => {
            ctx.assert(ctx.kernels.length >= 1, 'no kernel was created — call gpu.createKernel()');
            const size = outputWidth(ctx.kernel);
            ctx.assert(size != null, 'the kernel needs a 1D output — output: [size]');
            ctx.assert(
              size >= 100,
              `the output is ${size} wide, which cannot hold 100 values`
            );
            ctx.assert(
              isPowerOfTwo(size),
              `the output is ${size} wide, and ${size} is not a power of two — every partner is ` +
              'an index with one bit flipped, so it only lands inside the array when the array ' +
              'fills the whole index space. Pad up to 128 and sort that'
            );
          },
        },
        {
          name: 'the network sorts a padded array correctly',
          run: async ctx => {
            const size = outputWidth(ctx.kernel);
            ctx.assert(isPowerOfTwo(size) && size >= 100, 'expected a power-of-two output ≥ 100');
            const arr = makeValues(ctx.utils, 100, 606);
            const padded = arr.slice();
            while (padded.length < size) padded.push(PAD);
            const got = runNetwork(ctx.kernel, padded, size);
            const expected = sortedCopy(Float32Array.from(arr));
            for (let i = 0; i < 100; i++) {
              ctx.assertClose(got[i], expected[i], 1e-3, `sorted position ${i}`);
            }
            for (let i = 100; i < size; i++) {
              ctx.assertClose(
                got[i], PAD, 1,
                `position ${i} should hold a sentinel — the padding sorts to the end`
              );
            }
          },
        },
        {
          name: 'the smallest and largest <em>real</em> values are logged',
          run: async ctx => {
            const values = makeValues(ctx.utils, 100, 10007);
            const sorted = sortedCopy(Float32Array.from(values));
            const smallest = sorted[0];
            const largest = sorted[99];
            const nums = loggedNumbers(ctx.logs);
            ctx.assert(
              nums.some(v => Math.abs(v - smallest) <= 1e-2),
              `log the smallest sorted value — expected ≈${smallest.toFixed(2)} in the console output`
            );
            const hint = logged(ctx.logs, PAD, 1) && !logged(ctx.logs, largest, 1e-2)
              ? `${PAD} is the padding value, not your largest number — the sentinels sort to the ` +
                'end of the padded array, so slice them off before reading the result'
              : null;
            ctx.assert(
              nums.some(v => Math.abs(v - largest) <= 1e-2),
              hint || `log the largest sorted value — expected ≈${largest.toFixed(2)} in the ` +
              'console output'
            );
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            // A different awkward length through the same kernel: 70 real
            // values in whatever padded width the learner chose.
            const size = outputWidth(ctx.kernel);
            ctx.assert(isPowerOfTwo(size) && size >= 100, 'expected a power-of-two output ≥ 100');
            const arr = makeValues(ctx.utils, 70, 4004);
            const padded = arr.slice();
            while (padded.length < size) padded.push(PAD);
            const got = runNetwork(ctx.kernel, padded, size);
            const expected = sortedCopy(Float32Array.from(arr));
            for (let i = 0; i < 70; i++) {
              ctx.assertClose(got[i], expected[i], 1e-3, `sorted position ${i}`);
            }
          },
        },
        {
          name: 'private test #2',
          run: async ctx => {
            // Ties everywhere: only 5 distinct values across 100 slots.
            const size = outputWidth(ctx.kernel);
            const arr = new Array(100);
            for (let i = 0; i < 100; i++) arr[i] = ((i * 31) % 5) * 2.5;
            const padded = arr.slice();
            while (padded.length < size) padded.push(PAD);
            const got = runNetwork(ctx.kernel, padded, size);
            const expected = sortedCopy(Float32Array.from(arr));
            for (let i = 0; i < 100; i++) {
              ctx.assertClose(
                got[i], expected[i], 1e-3,
                `sorted position ${i} — duplicates sort like anything else, and there are only ` +
                '5 distinct values here'
              );
            }
          },
        },
      ],
    },
  ],
};
