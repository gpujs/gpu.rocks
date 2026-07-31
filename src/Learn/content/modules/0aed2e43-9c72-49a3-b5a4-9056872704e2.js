// Module: Stream Compaction — uuid 0aed2e43-9c72-49a3-b5a4-9056872704e2
// (short id 0aed2e43). The file name is the uuid; identity lives in the
// exported object below, never in the path. No legacyId: this module postdates
// uuids, and it belongs to no track (it lands in "Others", unordered).
//
// Five tasks: why filter has no kernel, and the flag pass that is the only
// part of it a thread can do alone → the exclusive scan that hands every
// survivor its destination → the compaction itself, written as a gather →
// the same gather in log n with a binary search → the payoff: the whole
// pipeline plus the compacted length.
//
// Kernel-authoring rules (contract): no closures inside kernel functions,
// loop bounds come from this.constants (compile-time known), a thread writes
// only its own cell (return), and every task passes in CPU mode as well as
// GPU mode. Arrays stay at 64 elements so verification is instant.

// The predicate this whole module filters on: keep a sample at or above 50.
// One number, used by the kernels (as this.constants.threshold), by every
// reference below, and by the prose — so "the predicate" is never ambiguous.
const N = 64;
const THRESHOLD = 50;

// 64 deterministic integer readings, 0 … 99. Integers on purpose: a compacted
// array gets logged and compared element by element, and integers survive the
// Float32 round trip exactly, so nothing in this module ever hinges on a
// tolerance the learner cannot see.
function makeSamples(utils, n, seed) {
  const rand = utils.seededRandom(seed);
  const data = new Array(n);
  for (let i = 0; i < n; i++) data[i] = Math.floor(rand() * 100);
  return data;
}

function flagsOf(samples) {
  return samples.map(v => (v >= THRESHOLD ? 1 : 0));
}

// Exclusive prefix sum: scan[i] = how many flags before i are 1. THE
// destination array — scan[i] is where survivor i lands.
function scanOf(flags) {
  const out = new Array(flags.length);
  let seen = 0;
  for (let i = 0; i < flags.length; i++) {
    out[i] = seen;
    seen += flags[i];
  }
  return out;
}

// Inclusive prefix sum: running[i] = how many flags up to AND INCLUDING i are
// 1. The near-miss of the array above, and the key the binary search wants.
function runningOf(flags) {
  const out = new Array(flags.length);
  let seen = 0;
  for (let i = 0; i < flags.length; i++) {
    seen += flags[i];
    out[i] = seen;
  }
  return out;
}

function survivorsOf(samples) {
  return samples.filter(v => v >= THRESHOLD);
}

// The count, computed the way the module teaches it: last offset + last flag.
function countOf(flags) {
  return scanOf(flags)[flags.length - 1] + flags[flags.length - 1];
}

// What a correct compaction holds. Only the first countOf(flags) cells are
// meaningful — the tail is whatever each thread's search failed to find, which
// differs between the linear and the binary formulation, so no test ever looks
// at it. The zero fill here is a placeholder, never an expectation.
function compactRef(samples) {
  const kept = survivorsOf(samples);
  const out = new Array(samples.length).fill(0);
  for (let j = 0; j < kept.length; j++) out[j] = kept[j];
  return out;
}

// Everything a survivor lands one slot too far: what comparing against the
// INCLUSIVE count (offsets[i] + flags[i]) produces instead of the exclusive
// one. Cell 0 never gets written, which is the "first survivor falls off the
// front" half of the same mistake.
function shiftedRef(samples) {
  const kept = survivorsOf(samples);
  const out = new Array(samples.length).fill(0);
  for (let j = 1; j <= kept.length && j < samples.length; j++) out[j] = kept[j - 1];
  return out;
}

// Reference lower-bound gather, for task 4's probes: for output cell j, take
// the smallest index whose `key` exceeds j (or reaches it, when `strict` is
// false) — the same window search the kernel runs, in plain JavaScript.
function searchRef(samples, key, strict) {
  const n = samples.length;
  const out = new Array(n);
  for (let j = 0; j < n; j++) {
    let lo = 0;
    let hi = n;
    while (lo < hi) {
      const mid = Math.floor((lo + hi) / 2);
      if (strict ? key[mid] > j : key[mid] >= j) hi = mid;
      else lo = mid + 1;
    }
    out[j] = samples[Math.min(lo, n - 1)];
  }
  return out;
}

// ---- log reading ----------------------------------------------------------

// Every console.log line, reduced to the numbers it contains. Task 5 asks for
// two logs — a count and a list — and telling them apart is exactly "how many
// numbers are on this line".
function numberLines(logs) {
  const out = [];
  for (const line of logs) {
    if (line.type !== 'log' || !line.text) continue;
    const matches = line.text.match(/-?\d+(?:\.\d+)?/g);
    out.push(matches ? matches.map(Number) : []);
  }
  return out;
}

function sameNumbers(a, b) {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) {
    if (Math.abs(a[i] - b[i]) > 1e-6) return false;
  }
  return true;
}

// ---- near-miss diagnosis --------------------------------------------------
//
// A failing assert that reports only two numbers tells a learner nothing about
// WHICH slip produced them. A probe pairs the value one specific known mistake
// would produce with a sentence naming that mistake; diagnose() speaks only
// when the observed value matches a probe within the test's own tolerance AND
// the correct value does not — so a case where two candidates coincide (a flags
// array whose last entry is 0, where "offsets[63]" and "offsets[63] + flags[63]"
// are the same number) stays silent, as do observations matching probes that
// disagree with each other. A wrong diagnosis is worse than none.
function diagnose(got, expected, eps, probes) {
  const hits = probes
    .filter(p => Math.abs(got - p[0]) <= eps && Math.abs(expected - p[0]) > eps)
    .map(p => p[1]);
  return hits.length && hits.every(m => m === hits[0]) ? hits[0] : null;
}

// The vector form. Most mistakes in this module are whole-array shapes — a mask
// inverted, a scan shifted, a search that never moves — and one matching cell
// is worthless evidence (a shifted compaction agrees with the right answer
// everywhere the same value repeats). So a probe must predict EVERY cell the
// test looks at, and differ from the right answer somewhere, before it may
// speak. Probe values are functions of the index; a missing cell makes the
// comparison NaN, which fails.
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

// Task 1: the three ways a 0/1 mask comes out wrong. The ">" probe can only
// fire when the array under test actually contains a sample equal to the
// threshold — otherwise it predicts the right answer everywhere and
// diagnoseAll's "differs somewhere" rule discards it.
function flagProbes(arr) {
  return [
    [i => (arr[i] > THRESHOLD ? 1 : 0),
      `a sample sitting exactly ON the threshold came back 0 — the predicate is "at or above" (>=), not "above" (>)`],
    [i => (arr[i] >= THRESHOLD ? 0 : 1),
      'the mask is inverted — 1 marks a sample that SURVIVES, 0 one that is dropped'],
    [i => arr[i],
      'the kernel returned the sample itself, not a flag — a mask holds only 1 and 0'],
  ];
}

// Task 2: the scan that counts one too many, and the scan that never stops.
function scanProbes(flags) {
  const inclusive = runningOf(flags);
  const total = countOf(flags);
  return [
    [i => inclusive[i],
      'that is the INCLUSIVE scan — it counts this thread\'s own flag too. A destination is how many survived STRICTLY BEFORE you, so an inclusive scan sends every survivor one slot too far and drops the first one off the front'],
    [() => total,
      'every cell holds the grand total — the count has to stop at this thread\'s index instead of running to the end of the array'],
  ];
}

// Task 3: no compaction at all, a search missing half its condition, and the
// exclusive/inclusive slip seen from the gather side.
function compactProbes(samples) {
  const kept = survivorsOf(samples);
  const last = kept.length ? kept[kept.length - 1] : 0;
  const shifted = shiftedRef(samples);
  return [
    [i => samples[i],
      'that is the input unchanged — cell j has to go looking for the element whose destination is j'],
    [() => last,
      'every cell came back with the same value, the last survivor — the search matched on flags[i] alone; it also has to check that offsets[i] is this thread\'s own index'],
    [i => shifted[i],
      'every cell holds the PREVIOUS survivor and cell 0 holds nothing — you compared against the inclusive count (offsets[i] + flags[i]); a destination is the exclusive count, offsets[i] on its own'],
  ];
}

// Task 4: a window that never advances, the wrong key, and the wrong comparison.
function searchProbes(samples, flags) {
  const offsets = scanOf(flags);
  const running = runningOf(flags);
  const first = samples[0];
  const byOffsets = searchRef(samples, offsets, true);
  const byReaches = searchRef(samples, running, false);
  return [
    [() => first,
      'every cell came back with samples[0] — the window never moved; the branch that rules out the lower half has to push lo past mid'],
    [i => byOffsets[i],
      'every cell holds the element just AFTER the right one — the key is the RUNNING count, offsets[mid] + flags[mid], not the exclusive scan on its own'],
    [i => byReaches[i],
      'every cell holds the PREVIOUS survivor — the test is strictly greater: you want the first index whose running count EXCEEDS this thread\'s index, not the first one that reaches it'],
  ];
}

// Task 5: the two off-by-ones in the survivor count.
function countProbes(flags) {
  const n = flags.length;
  const offsets = scanOf(flags);
  return [
    [offsets[n - 1],
      `that is offsets[${n - 1}] on its own — an exclusive scan at the last index counts everyone BEFORE it, so the last element's own flag is still missing: count = offsets[${n - 1}] + flags[${n - 1}]`],
    [n, `that is the length of the input (${n}), not the number of survivors`],
  ];
}

// Task 5: the count arrives through console.log, so the diagnosis reads the
// lines that carry exactly one number — which is what a logged count looks
// like, and what a logged list of survivors never does.
function countHint(ctx, expected, probes) {
  const singles = numberLines(ctx.logs)
    .filter(nums => nums.length === 1)
    .map(nums => nums[0]);
  const hits = [];
  for (const value of singles) {
    const message = diagnose(value, expected, 0.5, probes);
    if (message) hits.push(message);
  }
  return hits.length && hits.every(m => m === hits[0]) ? hits[0] : null;
}

export default {
  uuid: '0aed2e43-9c72-49a3-b5a4-9056872704e2',
  version: 1,
  slug: 'stream-compaction',
  title: 'Stream Compaction',
  blurb: 'Filtering on a GPU: flag what survives, scan to find out where it lands, then <code>gather</code> it into a packed array.',
  tasks: [
    // ------------------------------------------------------------------ 1
    {
      slug: 'flag-the-survivors',
      title: 'Filter Has No Kernel',
      intro: `<p>On a CPU, filtering is four words: <code>data.filter(v =&gt; v &gt;= 50)</code>.
        Underneath it is a loop with a <strong>moving write cursor</strong> — every element that
        passes gets pushed at wherever the last one left off:</p>
<pre><code>const kept = [];
for (const v of data) {
  if (v &gt;= 50) kept.push(v);   // ← push() knows where the cursor is
}</code></pre>
        <p>That cursor is the problem. Thread 7 can tell you instantly whether
        <code>data[7]</code> survives. It cannot tell you <em>where it goes</em>, because that
        depends on how many of elements 0…6 survived — six other threads' business, none of which
        thread 7 is allowed to ask about. An output position that depends on other threads is not
        something a single kernel can compute, which is why there is no <code>filter</code>
        kernel and never will be.</p>
        <p>So compaction gets built out of pieces, and the first piece is the one part that
        <em>is</em> perfectly independent: the <strong>flag pass</strong>. Turn the predicate into
        a mask of 1s and 0s — a plain map, one thread per element, nobody talking to anybody. Run
        it and look at what you get: the ones and zeros line up under the input, holes and all.
        Flags say <em>who</em> survives. They move nothing.</p>`,
      goal: `<strong>Goal:</strong> return <code>1</code> when this thread's sample is at or above
        <code>this.constants.threshold</code>, and <code>0</code> when it is not.`,
      requirements: [
        'One thread per sample — <code>output: [64]</code>, no loops',
        'Return exactly <code>1</code> or <code>0</code>, never the sample value',
        'The predicate is <em>at or above</em>: a sample of exactly <code>50</code> survives',
      ],
      hints: [
        {
          title: 'Hint 1 — a predicate is just a map',
          body: `<p>Read your own element and compare it — nothing else. The comparison
            <code>samples[this.thread.x] &gt;= this.constants.threshold</code> is the whole
            decision; all that is left is turning it into a number.</p>`,
        },
        {
          title: 'Hint 2 — the one-liner',
          body: `<pre><code>return samples[this.thread.x] &gt;= this.constants.threshold ? 1 : 0;</code></pre>`,
        },
      ],
      transfer: `Every compaction library on every platform starts here, and most of them let you
        hand the mask in yourself: CUB's <code>DeviceSelect::Flagged</code> takes a flags array
        beside the data, Thrust's <code>copy_if</code> takes the predicate and builds the same
        mask internally, and a WebGPU pipeline writes it to a storage buffer with one dispatch.
        The predicate pass is the cheap, embarrassingly parallel part — everything after it is
        the interesting problem.`,
      starterCode: `// The flag pass: a map from "does this survive?" to 1 or 0.
const gpu = new GPU({ mode });

const flag = gpu.createKernel(function (samples) {
  // TODO: return 1 when this thread's sample is at or above
  // this.constants.threshold, and 0 when it is not.
  return samples[this.thread.x];
}, {
  output: [64],
  constants: { threshold: 50 },
});

const flags = flag(samples);
console.log('samples:', samples.slice(0, 12).join(', '));
console.log('flags:  ', Array.from(flags).slice(0, 12).join(', '));
console.log('same 64 slots, holes and all — a flag says WHO survives, not WHERE it goes.');
`,
      solutionCode: `// The flag pass: a map from "does this survive?" to 1 or 0.
const gpu = new GPU({ mode });

const flag = gpu.createKernel(function (samples) {
  return samples[this.thread.x] >= this.constants.threshold ? 1 : 0;
}, {
  output: [64],
  constants: { threshold: 50 },
});

const flags = flag(samples);
console.log('samples:', samples.slice(0, 12).join(', '));
console.log('flags:  ', Array.from(flags).slice(0, 12).join(', '));
console.log('same 64 slots, holes and all — a flag says WHO survives, not WHERE it goes.');
`,
      inputs: utils => ({ samples: makeSamples(utils, N, 1130) }),
      publicTests: [
        {
          name: 'the mask holds only <code>1</code> and <code>0</code> — 64 of them',
          run: async ctx => {
            ctx.assert(ctx.kernels.length >= 1, 'no kernel was created — call gpu.createKernel()');
            const arr = makeSamples(ctx.utils, N, 1130);
            const out = ctx.kernel(arr);
            ctx.assert(out && out.length === N, `expected ${N} output values, got ${out && out.length}`);
            const hint = diagnoseAll(N, i => out[i], i => flagsOf(arr)[i], 1e-3, flagProbes(arr));
            for (let i = 0; i < N; i++) {
              ctx.assert(
                Math.abs(out[i] - 1) <= 1e-3 || Math.abs(out[i]) <= 1e-3,
                hint || `element ${i} is ${out[i]} — a mask holds only 1 (survives) or 0 (dropped)`
              );
            }
          },
        },
        {
          name: 'a sample of exactly <code>50</code> survives — the predicate is <code>&gt;=</code>',
          run: async ctx => {
            // 20 … 83, so index 30 sits exactly on the threshold.
            const arr = new Array(N);
            for (let i = 0; i < N; i++) arr[i] = i + 20;
            const out = ctx.kernel(arr);
            const want = flagsOf(arr);
            const hint = diagnoseAll(N, i => out[i], i => want[i], 1e-3, flagProbes(arr));
            for (let i = 0; i < N; i++) {
              ctx.assertClose(out[i], want[i], 1e-3, hint || `element ${i} (sample ${arr[i]})`);
            }
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            const arr = makeSamples(ctx.utils, N, 40961);
            const out = ctx.kernel(arr);
            ctx.assert(out && out.length === N, `expected ${N} output values`);
            const want = flagsOf(arr);
            const hint = diagnoseAll(N, i => out[i], i => want[i], 1e-3, flagProbes(arr));
            for (let i = 0; i < N; i++) {
              ctx.assertClose(out[i], want[i], 1e-3, hint || `element ${i} (sample ${arr[i]})`);
            }
          },
        },
        {
          name: 'private test #2',
          run: async ctx => {
            // Both extremes: nothing survives, then everything does.
            const none = new Array(N).fill(THRESHOLD - 1);
            const all = new Array(N).fill(THRESHOLD);
            const low = ctx.kernel(none);
            for (let i = 0; i < N; i++) {
              ctx.assertClose(low[i], 0, 1e-3, `sample ${THRESHOLD - 1} is below the threshold — element ${i}`);
            }
            const high = ctx.kernel(all);
            for (let i = 0; i < N; i++) {
              ctx.assertClose(high[i], 1, 1e-3, `sample ${THRESHOLD} is ON the threshold, so it survives — element ${i}`);
            }
          },
        },
      ],
    },

    // ------------------------------------------------------------------ 2
    {
      slug: 'where-do-i-land',
      title: 'Where Do I Land?',
      intro: `<p>A survivor's output slot has a very short definition: <strong>how many survivors
        are in front of me</strong>. Element 9 with four survivors before it lands at index 4.
        That number, for every element at once, is the <strong>exclusive prefix sum</strong> — a
        <em>scan</em> — of the flags: cell <code>i</code> holds the total of flags
        <code>0 … i−1</code>, <em>not counting its own</em>.</p>
        <p>Scan is a whole subject of its own, and the Prefix Sums module derives the log-time
        ladder version properly. Sixty-four elements do not need it: every thread can simply walk
        the flags and count. That is <code>n</code> reads per thread — blunt, but perfectly
        parallel, and right now what matters is what the number <em>means</em>, not how fast you
        can get it.</p>
        <p>Exclusive, not inclusive. Count yourself and every survivor lands one slot too far,
        with the first one shoved off the front of the array.</p>`,
      goal: `<strong>Goal:</strong> for every index, return how many of the flags
        <em>strictly before</em> it are <code>1</code>.`,
      requirements: [
        'Loop the full fixed range — <code>for (let i = 0; i &lt; this.constants.n; i++)</code>',
        'Add <code>flags[i]</code> only when <code>i</code> is strictly less than <code>this.thread.x</code>',
        'Cell <code>0</code> is always <code>0</code>, whatever <code>flags[0]</code> says',
      ],
      hints: [
        {
          title: 'Hint 1 — "strictly before"',
          body: `<p>The loop already visits every flag; it just needs to ignore the ones that are
            not in front of this thread. Guard the accumulation with
            <code>if (i &lt; this.thread.x)</code> — note <code>&lt;</code>, not
            <code>&lt;=</code>.</p>`,
        },
        {
          title: 'Hint 2 — the loop body',
          body: `<pre><code>if (i &lt; this.thread.x) {
  seen += flags[i];
}</code></pre>`,
        },
        {
          title: 'Hint 3 — check it by hand',
          body: `<p>For flags <code>[1, 0, 1, 1]</code> the destinations are
            <code>[0, 1, 1, 2]</code>: the survivor at index 0 goes to slot 0, the one at index 2
            goes to slot 1, the one at index 3 goes to slot 2. Index 1 gets a number too — it just
            never uses it, because it does not survive.</p>`,
        },
      ],
      transfer: `Exclusive scan is one of the two or three primitives every GPU library is built
        on: <code>thrust::exclusive_scan</code>, CUB's <code>DeviceScan::ExclusiveSum</code>,
        rocPRIM's equivalent, and <code>subgroupExclusiveAdd</code> burned into WGSL and hardware.
        Blelloch's 1990 formulation gets it in O(n) work and O(log n) depth — the loop here is the
        honest, slow version of the same answer.`,
      starterCode: `// The scan: every thread counts the survivors in front of it.
const gpu = new GPU({ mode });

const destination = gpu.createKernel(function (flags) {
  let seen = 0;
  for (let i = 0; i < this.constants.n; i++) {
    // TODO: count only the flags STRICTLY BEFORE this thread's index.
    seen += flags[i];
  }
  return seen;
}, {
  output: [64],
  constants: { n: 64 },
});

const offsets = destination(flags);
console.log('flags:   ', Array.from(flags).slice(0, 12).join(', '));
console.log('offsets: ', Array.from(offsets).slice(0, 12).join(', '));
`,
      solutionCode: `// The scan: every thread counts the survivors in front of it.
const gpu = new GPU({ mode });

const destination = gpu.createKernel(function (flags) {
  let seen = 0;
  for (let i = 0; i < this.constants.n; i++) {
    if (i < this.thread.x) {
      seen += flags[i];
    }
  }
  return seen;
}, {
  output: [64],
  constants: { n: 64 },
});

const offsets = destination(flags);
console.log('flags:   ', Array.from(flags).slice(0, 12).join(', '));
console.log('offsets: ', Array.from(offsets).slice(0, 12).join(', '));
`,
      inputs: utils => ({ flags: flagsOf(makeSamples(utils, N, 1121)) }),
      publicTests: [
        {
          name: 'cell <code>0</code> is <code>0</code> even when <code>flags[0]</code> is <code>1</code>',
          run: async ctx => {
            ctx.assert(ctx.kernels.length >= 1, 'no kernel was created — call gpu.createKernel()');
            const arr = new Array(N);
            for (let i = 0; i < N; i++) arr[i] = i % 3 === 0 ? 1 : 0; // arr[0] === 1
            const out = ctx.kernel(arr);
            ctx.assert(out && out.length === N, `expected ${N} output values, got ${out && out.length}`);
            const want = scanOf(arr);
            const hint = diagnoseAll(N, i => out[i], i => want[i], 1e-3, scanProbes(arr));
            ctx.assertClose(out[0], 0, 1e-3, hint ||
              'nothing comes before index 0, so its destination is 0 no matter what flags[0] holds');
          },
        },
        {
          name: 'every cell counts the survivors <em>strictly before</em> it',
          run: async ctx => {
            const arr = new Array(N);
            for (let i = 0; i < N; i++) arr[i] = i % 3 === 0 ? 1 : 0;
            const out = ctx.kernel(arr);
            const want = scanOf(arr);
            const hint = diagnoseAll(N, i => out[i], i => want[i], 1e-3, scanProbes(arr));
            for (let i = 0; i < N; i++) {
              ctx.assertClose(out[i], want[i], 1e-3, hint || `cell ${i}`);
            }
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            const arr = flagsOf(makeSamples(ctx.utils, N, 8088));
            const out = ctx.kernel(arr);
            const want = scanOf(arr);
            const hint = diagnoseAll(N, i => out[i], i => want[i], 1e-3, scanProbes(arr));
            for (let i = 0; i < N; i++) {
              ctx.assertClose(out[i], want[i], 1e-3, hint || `cell ${i}`);
            }
          },
        },
        {
          name: 'private test #2',
          run: async ctx => {
            // All ones: the scan is the index itself. All zeros: flat 0.
            const ones = ctx.kernel(new Array(N).fill(1));
            for (let i = 0; i < N; i++) {
              ctx.assertClose(ones[i], i, 1e-3, `with every flag set, cell ${i} counts ${i} survivors before it`);
            }
            const zeros = ctx.kernel(new Array(N).fill(0));
            for (let i = 0; i < N; i++) {
              ctx.assertClose(zeros[i], 0, 1e-3, `with no flags set, cell ${i} counts nothing`);
            }
          },
        },
      ],
    },

    // ------------------------------------------------------------------ 3
    {
      slug: 'gather-not-scatter',
      title: 'Turn the Scatter Around',
      intro: `<p>You now have the two arrays a compaction needs: <code>flags</code>, and
        <code>offsets</code> — the exclusive scan that tells every survivor its slot. The obvious
        next line is the one you cannot write:</p>
<pre><code>if (flags[i] === 1) out[offsets[i]] = samples[i];   // ✗ no scatter here</code></pre>
        <p>Thinking in Parallel spends a whole task on why: a thread writes one cell, its own, by
        returning a value. So turn the question inside out, exactly as it does. Not <em>"where
        does my value go?"</em> but <em>"whose value lands in <strong>my</strong> cell?"</em> —
        and output cell <code>j</code> can answer that itself. It goes looking for the index that
        is (a) a survivor and (b) carrying destination <code>j</code>. One thread, one pass over
        the flags, one value pulled home.</p>
        <p>The output array is still 64 long, because that is what a kernel launch gives you. Only
        the first few cells will hold survivors; the rest hold whatever each thread's search failed
        to find. That is fine, and normal — you just have to know where the real data stops, which
        is the last task of this module.</p>`,
      goal: `<strong>Goal:</strong> fill each output cell by searching for the element whose
        destination is this cell's index.`,
      requirements: [
        'Loop over all <code>this.constants.n</code> elements — the only write is the <code>return</code>',
        'Take <code>samples[i]</code> when <code>flags[i]</code> is <code>1</code> <em>and</em> <code>offsets[i]</code> is <code>this.thread.x</code>',
        'The survivors come out packed, in input order, starting at cell <code>0</code>',
      ],
      hints: [
        {
          title: 'Hint 1 — ask the other question',
          body: `<p>Thread <code>j</code> is not trying to place anything. It is trying to
            <em>find</em> something: the one index whose destination happens to be <code>j</code>.
            Keep a local <code>value</code>, overwrite it when the search hits, return it.</p>`,
        },
        {
          title: 'Hint 2 — both halves of the condition',
          body: `<p>Non-survivors have an <code>offsets</code> entry too — it just does not belong
            to them. So matching the offset alone is not enough; the flag has to be checked as
            well:</p>
<pre><code>if (flags[i] === 1 &amp;&amp; offsets[i] === this.thread.x) {
  value = samples[i];
}</code></pre>`,
        },
        {
          title: 'Hint 3 — the whole body',
          body: `<pre><code>let value = 0;
for (let i = 0; i &lt; this.constants.n; i++) {
  if (flags[i] === 1 &amp;&amp; offsets[i] === this.thread.x) {
    value = samples[i];
  }
}
return value;</code></pre>`,
        },
      ],
      transfer: `"Turn the scatter into a gather" is the phrase GPU folklore compresses this into,
        and it is exactly how the libraries do it: <code>thrust::copy_if</code> and CUB's
        <code>DeviceSelect</code> both run a flag pass, a scan, and then a data movement driven by
        the scan. Compute APIs <em>can</em> scatter — CUDA and WebGPU threads may store anywhere —
        but two threads aiming at one address is a race, and the fix (atomics) serialises them.
        Scan-then-gather costs no atomics at all.`,
      starterCode: `// No scatter. Cell j goes looking for the element destined for j.
const gpu = new GPU({ mode });

const compact = gpu.createKernel(function (samples, flags, offsets) {
  // TODO: search the whole array for the element whose destination is
  // this thread's index — a survivor (flags[i] === 1) whose offsets[i]
  // equals this.thread.x — and return its sample.
  return samples[this.thread.x];
}, {
  output: [64],
  constants: { n: 64 },
});

const packed = compact(samples, flags, offsets);
console.log('samples:', samples.slice(0, 12).join(', '));
console.log('packed: ', Array.from(packed).slice(0, 12).join(', '));
`,
      solutionCode: `// No scatter. Cell j goes looking for the element destined for j.
const gpu = new GPU({ mode });

const compact = gpu.createKernel(function (samples, flags, offsets) {
  let value = 0;
  for (let i = 0; i < this.constants.n; i++) {
    if (flags[i] === 1 && offsets[i] === this.thread.x) {
      value = samples[i];
    }
  }
  return value;
}, {
  output: [64],
  constants: { n: 64 },
});

const packed = compact(samples, flags, offsets);
console.log('samples:', samples.slice(0, 12).join(', '));
console.log('packed: ', Array.from(packed).slice(0, 12).join(', '));
`,
      inputs: utils => {
        const samples = makeSamples(utils, N, 1042);
        const flags = flagsOf(samples);
        return { samples, flags, offsets: scanOf(flags) };
      },
      publicTests: [
        {
          name: 'the survivors come out packed, in input order',
          run: async ctx => {
            ctx.assert(ctx.kernels.length >= 1, 'no kernel was created — call gpu.createKernel()');
            const samples = makeSamples(ctx.utils, N, 1042);
            const flags = flagsOf(samples);
            const out = ctx.kernel(samples, flags, scanOf(flags));
            ctx.assert(out && out.length === N, `expected ${N} output values, got ${out && out.length}`);
            const want = compactRef(samples);
            const count = countOf(flags);
            const hint = diagnoseAll(count, i => out[i], i => want[i], 1e-3, compactProbes(samples));
            for (let j = 0; j < count; j++) {
              ctx.assertClose(out[j], want[j], 1e-3, hint || `slot ${j} of ${count}`);
            }
          },
        },
        {
          name: 'no holes: every packed cell is a sample that passed the predicate',
          run: async ctx => {
            const samples = makeSamples(ctx.utils, N, 6001);
            const flags = flagsOf(samples);
            const out = ctx.kernel(samples, flags, scanOf(flags));
            const want = compactRef(samples);
            const count = countOf(flags);
            const hint = diagnoseAll(count, i => out[i], i => want[i], 1e-3, compactProbes(samples));
            for (let j = 0; j < count; j++) {
              ctx.assert(
                out[j] >= THRESHOLD - 1e-3,
                hint || `slot ${j} holds ${out[j]}, which is below the threshold — a hole crept into the packed array`
              );
              ctx.assertClose(out[j], want[j], 1e-3, hint || `slot ${j} of ${count}`);
            }
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            const samples = makeSamples(ctx.utils, N, 31337);
            const flags = flagsOf(samples);
            const out = ctx.kernel(samples, flags, scanOf(flags));
            const want = compactRef(samples);
            const count = countOf(flags);
            const hint = diagnoseAll(count, i => out[i], i => want[i], 1e-3, compactProbes(samples));
            for (let j = 0; j < count; j++) {
              ctx.assertClose(out[j], want[j], 1e-3, hint || `slot ${j} of ${count}`);
            }
          },
        },
        {
          name: 'private test #2',
          run: async ctx => {
            // The three degenerate shapes: everything survives, only the last
            // one does, nothing does.
            const all = new Array(N);
            for (let i = 0; i < N; i++) all[i] = THRESHOLD + (i % 40);
            const allFlags = flagsOf(all);
            const allOut = ctx.kernel(all, allFlags, scanOf(allFlags));
            for (let j = 0; j < N; j++) {
              ctx.assertClose(allOut[j], all[j], 1e-3,
                `with every sample surviving the compaction is the identity — slot ${j}`);
            }

            const one = new Array(N).fill(0);
            one[N - 1] = 77;
            const oneFlags = flagsOf(one);
            const oneOut = ctx.kernel(one, oneFlags, scanOf(oneFlags));
            ctx.assertClose(oneOut[0], 77, 1e-3,
              'the only survivor sits at the END of the input, so it lands in slot 0');

            const none = new Array(N).fill(0);
            const noneFlags = flagsOf(none);
            const noneOut = ctx.kernel(none, noneFlags, scanOf(noneFlags));
            ctx.assert(noneOut && noneOut.length === N,
              'with nothing surviving the kernel must still return its 64 cells');
          },
        },
      ],
    },

    // ------------------------------------------------------------------ 4
    {
      slug: 'binary-search-gather',
      title: 'Find It in log n',
      intro: `<p>The search you just wrote reads all 64 flags per thread. Across 64 threads that
        is 4,096 reads to move 30-odd values — worse than the CPU's single pass. It works, and it
        is the right shape, but it throws away the one thing that makes the array searchable:
        <code>offsets</code> <strong>never decreases</strong>.</p>
        <p>Add the flag back to it and you get the <strong>running count</strong> —
        <code>offsets[i] + flags[i]</code>, how many survived up to and including <code>i</code>.
        It is non-decreasing too, and it steps up by exactly one at each survivor. So the element
        for output cell <code>j</code> is the <em>first index whose running count exceeds
        <code>j</code></em>, and a sorted array is something you can binary-search: seven
        halvings settle 64 elements instead of 64 reads.</p>
        <p>This is a <em>lower bound</em> search — keep a window <code>[lo, hi)</code>, look at
        its midpoint, and throw away the half that cannot contain the answer. When the window is
        empty, <code>lo</code> is the index you wanted.</p>`,
      goal: `<strong>Goal:</strong> replace the linear search with a binary search over the
        running count, and return the sample it lands on.`,
      requirements: [
        'Keep a window <code>lo</code> … <code>hi</code>, starting at <code>0</code> and <code>this.constants.n</code>',
        'Halve it <code>this.constants.steps</code> times, testing <code>offsets[mid] + flags[mid]</code> against <code>this.thread.x</code>',
        'Return <code>samples[lo]</code> — clamped to the last index, because <code>lo</code> can finish at <code>n</code>',
      ],
      hints: [
        {
          title: 'Hint 1 — what you are searching for',
          body: `<p>For output cell <code>j</code> you want the smallest index whose running count
            is <strong>greater than</strong> <code>j</code>. Greater than, not equal to: the
            running count reaches <code>j + 1</code> exactly at the survivor destined for slot
            <code>j</code>.</p>`,
        },
        {
          title: 'Hint 2 — one halving',
          body: `<p>If the midpoint's running count already exceeds <code>this.thread.x</code>,
            the answer is at <code>mid</code> or to its left, so <code>hi = mid</code>. Otherwise
            the answer is strictly to the right, so <code>lo = mid + 1</code>. Nothing else
            changes.</p>`,
        },
        {
          title: 'Hint 3 — the whole body',
          body: `<pre><code>let lo = 0;
let hi = this.constants.n;
for (let s = 0; s &lt; this.constants.steps; s++) {
  if (lo &lt; hi) {
    const mid = Math.floor((lo + hi) / 2);
    if (offsets[mid] + flags[mid] &gt; this.thread.x) {
      hi = mid;
    } else {
      lo = mid + 1;
    }
  }
}
return samples[Math.min(lo, this.constants.n - 1)];</code></pre>
<p>The <code>if (lo &lt; hi)</code> guard matters: the window can empty before the seventh
            halving, and a midpoint of an empty window is an out-of-bounds read.</p>`,
        },
      ],
      transfer: `Device-side binary search is a first-class primitive —
        <code>thrust::lower_bound</code>, CUB's <code>DeviceSelect</code> internals, and the
        merge-path partitioning that load-balances GPU merges and sparse-matrix products all lean
        on it. Production compactors often go one step further and build a <em>scatter-address
        table</em> instead: one pass writes each output slot's source index, a second gathers
        through it, trading a buffer for the search entirely. Same inversion, one more array.`,
      starterCode: `// Same gather, log n reads: binary-search the running count.
const gpu = new GPU({ mode });

const compact = gpu.createKernel(function (samples, flags, offsets) {
  let lo = 0;
  let hi = this.constants.n;
  for (let s = 0; s < this.constants.steps; s++) {
    if (lo < hi) {
      const mid = Math.floor((lo + hi) / 2);
      // TODO: compare the running count at mid — offsets[mid] + flags[mid] —
      // against this.thread.x, and throw away the half that cannot hold the
      // answer. One of the two branches has to move lo past mid.
      hi = mid;
    }
  }
  return samples[Math.min(lo, this.constants.n - 1)];
}, {
  output: [64],
  constants: { n: 64, steps: 7 },
});

const packed = compact(samples, flags, offsets);
console.log('samples:', samples.slice(0, 12).join(', '));
console.log('packed: ', Array.from(packed).slice(0, 12).join(', '));
`,
      solutionCode: `// Same gather, log n reads: binary-search the running count.
const gpu = new GPU({ mode });

const compact = gpu.createKernel(function (samples, flags, offsets) {
  let lo = 0;
  let hi = this.constants.n;
  for (let s = 0; s < this.constants.steps; s++) {
    if (lo < hi) {
      const mid = Math.floor((lo + hi) / 2);
      if (offsets[mid] + flags[mid] > this.thread.x) {
        hi = mid;
      } else {
        lo = mid + 1;
      }
    }
  }
  return samples[Math.min(lo, this.constants.n - 1)];
}, {
  output: [64],
  constants: { n: 64, steps: 7 },
});

const packed = compact(samples, flags, offsets);
console.log('samples:', samples.slice(0, 12).join(', '));
console.log('packed: ', Array.from(packed).slice(0, 12).join(', '));
`,
      inputs: utils => {
        const samples = makeSamples(utils, N, 1181);
        const flags = flagsOf(samples);
        return { samples, flags, offsets: scanOf(flags) };
      },
      publicTests: [
        {
          name: 'the binary search packs the survivors, in input order',
          run: async ctx => {
            ctx.assert(ctx.kernels.length >= 1, 'no kernel was created — call gpu.createKernel()');
            const samples = makeSamples(ctx.utils, N, 1181);
            const flags = flagsOf(samples);
            const out = ctx.kernel(samples, flags, scanOf(flags));
            ctx.assert(out && out.length === N, `expected ${N} output values, got ${out && out.length}`);
            const want = compactRef(samples);
            const count = countOf(flags);
            const hint = diagnoseAll(count, i => out[i], i => want[i], 1e-3, searchProbes(samples, flags));
            for (let j = 0; j < count; j++) {
              ctx.assertClose(out[j], want[j], 1e-3, hint || `slot ${j} of ${count}`);
            }
          },
        },
        {
          name: 'the ends of the window: first element, last element, everything',
          run: async ctx => {
            // Only samples[0] survives — the search must stop at index 0.
            const firstOnly = new Array(N).fill(0);
            firstOnly[0] = 91;
            let flags = flagsOf(firstOnly);
            let out = ctx.kernel(firstOnly, flags, scanOf(flags));
            ctx.assertClose(out[0], 91, 1e-3,
              'the only survivor is element 0, so slot 0 holds it — the window has to be able to stop at index 0');

            // Only samples[63] survives — the search must reach the far end.
            const lastOnly = new Array(N).fill(0);
            lastOnly[N - 1] = 88;
            flags = flagsOf(lastOnly);
            out = ctx.kernel(lastOnly, flags, scanOf(flags));
            ctx.assertClose(out[0], 88, 1e-3,
              'the only survivor is the LAST element, so slot 0 holds it — the window has to be able to reach index 63');

            // Everything survives: the compaction is the identity.
            const all = new Array(N);
            for (let i = 0; i < N; i++) all[i] = THRESHOLD + (i % 40);
            flags = flagsOf(all);
            out = ctx.kernel(all, flags, scanOf(flags));
            const hint = diagnoseAll(N, i => out[i], i => all[i], 1e-3, searchProbes(all, flags));
            for (let j = 0; j < N; j++) {
              ctx.assertClose(out[j], all[j], 1e-3, hint || `with every sample surviving, slot ${j} is element ${j}`);
            }
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            const samples = makeSamples(ctx.utils, N, 9091);
            const flags = flagsOf(samples);
            const out = ctx.kernel(samples, flags, scanOf(flags));
            const want = compactRef(samples);
            const count = countOf(flags);
            const hint = diagnoseAll(count, i => out[i], i => want[i], 1e-3, searchProbes(samples, flags));
            for (let j = 0; j < count; j++) {
              ctx.assertClose(out[j], want[j], 1e-3, hint || `slot ${j} of ${count}`);
            }
          },
        },
        {
          name: 'private test #2',
          run: async ctx => {
            // Runs of survivors and runs of rejects, so the window has to land
            // in the middle of a plateau as well as on a step.
            const blocks = new Array(N);
            for (let i = 0; i < N; i++) {
              blocks[i] = Math.floor(i / 8) % 2 === 0 ? 10 + (i % 8) : 60 + (i % 8);
            }
            const flags = flagsOf(blocks);
            const out = ctx.kernel(blocks, flags, scanOf(flags));
            const want = compactRef(blocks);
            const count = countOf(flags);
            const hint = diagnoseAll(count, i => out[i], i => want[i], 1e-3, searchProbes(blocks, flags));
            for (let j = 0; j < count; j++) {
              ctx.assertClose(out[j], want[j], 1e-3, hint || `slot ${j} of ${count}`);
            }
          },
        },
      ],
    },

    // ------------------------------------------------------------------ 5
    {
      slug: 'compacted-length',
      title: 'Payoff: How Many Survived',
      intro: `<p>Three kernels, wired together: flags, then the scan, then the gather. All of it
        is below, finished — the pipeline is yours already. What is missing is the number that
        makes the result usable.</p>
        <p>The output is 64 cells long because the launch was 64 threads wide. Only the first
        <strong>count</strong> of them hold survivors; the rest hold whatever the search failed to
        find, and reading them as data is the classic way to ship a bug. So where does
        <code>count</code> come from? The end of the scan — but not from
        <code>offsets[63]</code> alone. An exclusive scan at the last index counts everyone
        <em>before</em> the last element, so the last element's own flag is still outstanding:</p>
<pre><code>const count = offsets[n - 1] + flags[n - 1];</code></pre>
        <p>Drop the <code>+ flags[n - 1]</code> and the pipeline quietly loses its final element,
        but only when that element happens to survive — which is exactly the kind of bug that
        passes every test you wrote by hand. And note what this number costs: it has to come back
        to JavaScript before anything can use it. That single readback is why compaction is the
        awkward step in an otherwise fully on-device pipeline.</p>`,
      goal: `<strong>Goal:</strong> compute the survivor count from the end of the scan, trim the
        packed output to it, and log both.`,
      requirements: [
        '<code>count = offsets[63] + flags[63]</code> — the last offset <em>plus</em> the last flag',
        'Trim the 64-cell output down to those <code>count</code> values',
        '<code>console.log</code> the count on its own line, and the kept values as a list',
      ],
      hints: [
        {
          title: 'Hint 1 — the count lives at the end of the scan',
          body: `<p><code>offsets[63]</code> is "how many survived among elements 0…62". Element
            63 is not in that total — its flag is. Add them.</p>`,
        },
        {
          title: 'Hint 2 — trimming',
          body: `<p>The kernel returns a <code>Float32Array</code>; turn it into a plain array and
            cut it at the count:</p>
<pre><code>const kept = Array.from(packed).slice(0, count);</code></pre>`,
        },
        {
          title: 'Hint 3 — check yourself',
          body: `<p><code>kept.length</code> should equal <code>count</code>, and every value in
            it should be at least 50. If the last one is missing, you dropped the
            <code>+ flags[63]</code>.</p>`,
        },
      ],
      transfer: `Every real compaction API hands the length back separately, and for the same
        reason: <code>thrust::copy_if</code> returns an end iterator, CUB's
        <code>DeviceSelect::Flagged</code> writes <code>d_num_selected_out</code> to device
        memory, and Vulkan/WebGPU pipelines that want to avoid the readback entirely feed that
        counter straight into an <em>indirect</em> dispatch or draw — the GPU deciding its own
        launch size from a number the CPU never sees.`,
      starterCode: `// The finished pipeline: flags → scan → gather. Only the count is missing.
const gpu = new GPU({ mode });

const flag = gpu.createKernel(function (samples) {
  return samples[this.thread.x] >= this.constants.threshold ? 1 : 0;
}, { output: [64], constants: { threshold: 50 } });

const destination = gpu.createKernel(function (flags) {
  let seen = 0;
  for (let i = 0; i < this.constants.n; i++) {
    if (i < this.thread.x) {
      seen += flags[i];
    }
  }
  return seen;
}, { output: [64], constants: { n: 64 } });

// The linear search from task 3 — task 4's binary search drops straight in.
const compact = gpu.createKernel(function (samples, flags, offsets) {
  let value = 0;
  for (let i = 0; i < this.constants.n; i++) {
    if (flags[i] === 1 && offsets[i] === this.thread.x) {
      value = samples[i];
    }
  }
  return value;
}, { output: [64], constants: { n: 64 } });

const flags = flag(samples);
const offsets = destination(flags);
const packed = compact(samples, flags, offsets);

// TODO: the scan stops one short — offsets[63] counts everyone BEFORE
// element 63, so element 63's own flag is still missing from the total.
const count = offsets[63];

// TODO: keep only the cells that actually hold survivors.
const kept = Array.from(packed);

console.log('survivors:', count);
console.log('kept:', kept.join(', '));
`,
      solutionCode: `// The finished pipeline: flags → scan → gather. Only the count is missing.
const gpu = new GPU({ mode });

const flag = gpu.createKernel(function (samples) {
  return samples[this.thread.x] >= this.constants.threshold ? 1 : 0;
}, { output: [64], constants: { threshold: 50 } });

const destination = gpu.createKernel(function (flags) {
  let seen = 0;
  for (let i = 0; i < this.constants.n; i++) {
    if (i < this.thread.x) {
      seen += flags[i];
    }
  }
  return seen;
}, { output: [64], constants: { n: 64 } });

// The linear search from task 3 — task 4's binary search drops straight in.
const compact = gpu.createKernel(function (samples, flags, offsets) {
  let value = 0;
  for (let i = 0; i < this.constants.n; i++) {
    if (flags[i] === 1 && offsets[i] === this.thread.x) {
      value = samples[i];
    }
  }
  return value;
}, { output: [64], constants: { n: 64 } });

const flags = flag(samples);
const offsets = destination(flags);
const packed = compact(samples, flags, offsets);

// The last offset counts everyone before element 63; its own flag finishes it.
const count = offsets[63] + flags[63];

const kept = Array.from(packed).slice(0, count);

console.log('survivors:', count);
console.log('kept:', kept.join(', '));
`,
      inputs: utils => ({ samples: makeSamples(utils, N, 1130) }),
      publicTests: [
        {
          name: 'the three kernels still compact a fresh array',
          run: async ctx => {
            ctx.assert(
              ctx.kernels.length >= 3,
              `expected 3 kernels — flags, scan, gather — found ${ctx.kernels.length}`
            );
            const [flag, destination, compact] = ctx.kernels;
            const samples = makeSamples(ctx.utils, N, 1170);
            const flags = flag(samples);
            const offsets = destination(flags);
            const out = compact(samples, flags, offsets);
            ctx.assert(out && out.length === N, `expected ${N} output cells, got ${out && out.length}`);
            const want = compactRef(samples);
            const count = countOf(flagsOf(samples));
            for (let j = 0; j < count; j++) {
              ctx.assertClose(out[j], want[j], 1e-3, `slot ${j} of ${count}`);
            }
          },
        },
        {
          name: 'the survivor count is logged',
          run: async ctx => {
            const samples = makeSamples(ctx.utils, N, 1130);
            const flags = flagsOf(samples);
            const count = countOf(flags);
            // Only short lines count as evidence: the list of kept values is
            // full of numbers, and one of them agreeing with the count by
            // accident must not pass this test.
            const seen = numberLines(ctx.logs).some(
              nums => nums.length <= 3 && nums.some(v => Math.abs(v - count) <= 0.5)
            );
            const hint = countHint(ctx, count, countProbes(flags));
            ctx.assert(seen, hint || `log the survivor count — expected ${count} in the console output`);
          },
        },
        {
          name: 'the kept values are logged, trimmed to the count',
          run: async ctx => {
            const samples = makeSamples(ctx.utils, N, 1130);
            const want = survivorsOf(samples);
            const lines = numberLines(ctx.logs);
            const exact = lines.some(nums => sameNumbers(nums, want));
            // A line that STARTS with the survivors but runs on is the untrimmed
            // output: the leftover tail of the gather, logged as if it were data.
            const untrimmed = lines.find(
              nums => nums.length > want.length && sameNumbers(nums.slice(0, want.length), want)
            );
            ctx.assert(exact, untrimmed
              ? `that list has ${untrimmed.length} values but only ${want.length} samples survived — ` +
                'everything past the survivor count is whatever the gather failed to find, not data. ' +
                'Trim the output with .slice(0, count) before you use it.'
              : `log the kept values as a list — expected the ${want.length} samples ≥ ${THRESHOLD}, in input order`);
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            const [flag, destination, compact] = ctx.kernels;
            ctx.assert(flag && destination && compact, 'expected three kernels — flags, scan, gather');
            const samples = makeSamples(ctx.utils, N, 1200);
            const flags = flag(samples);
            const offsets = destination(flags);
            const out = compact(samples, flags, offsets);
            const want = compactRef(samples);
            const refFlags = flagsOf(samples);
            const count = countOf(refFlags);

            // The count, recomputed the way the task teaches it, from the
            // learner's own scan — offsets[63] alone would be one short here.
            ctx.assertClose(offsets[N - 1] + flags[N - 1], count, 0.5,
              'offsets[63] + flags[63] should be the number of survivors');
            for (let j = 0; j < count; j++) {
              ctx.assertClose(out[j], want[j], 1e-3, `slot ${j} of ${count}`);
            }
          },
        },
        {
          name: 'private test #2',
          run: async ctx => {
            // The case the missing + flags[63] hides behind: a fresh array whose
            // LAST element survives. Drop the term and the final value vanishes.
            const [flag, destination, compact] = ctx.kernels;
            ctx.assert(flag && destination && compact, 'expected three kernels — flags, scan, gather');
            const samples = makeSamples(ctx.utils, N, 1014);
            const flags = flag(samples);
            const offsets = destination(flags);
            const out = compact(samples, flags, offsets);
            const want = survivorsOf(samples);
            ctx.assert(samples[N - 1] >= THRESHOLD, 'test fixture: the last sample should survive');
            ctx.assertClose(offsets[N - 1] + flags[N - 1], want.length, 0.5, 'the survivor count');
            ctx.assertClose(out[want.length - 1], want[want.length - 1], 1e-3,
              'the LAST survivor should land in the last occupied slot');
          },
        },
      ],
    },
  ],
};
