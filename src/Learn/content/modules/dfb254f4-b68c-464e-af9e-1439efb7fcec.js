// Module: Histograms & Binning — uuid dfb254f4-b68c-464e-af9e-1439efb7fcec
// (short id dfb254f4). The file name is the uuid; identity lives in the
// exported object below, never in the path. No legacyId: this module postdates
// the uuid switch, and it belongs to no track (it lands in "Others").
//
// Histograms & Binning — the canonical "I need atomics and don't have them"
// problem.
//
// Five tasks: one bin counted by one thread → one thread per bin, the whole
// histogram → binning continuous values, where the real bugs live → partial
// histograms per chunk, merged in a second pass → the payoff, an image's tone
// histogram from a luminance map.
//
// Kernel-authoring rules (contract): no closures inside kernel functions, loop
// bounds come from this.constants (compile-time known), a thread writes only
// its own output cell, and every task passes in CPU mode. Sizes stay ≤ 16,384
// values and ≤ 32 bins so verification is quick.
//
// NUMERIC CONTRACT. Every bin edge in this module is an exact binary fraction:
// task 3 divides a span of 64 into 16 bins of 4, task 5 divides 1 into 32.
// (v − lo) / span * bins therefore evaluates identically in the GL backend's
// float32 and in JavaScript's float64, so a sample can never land in different
// bins on different backends. Change those numbers and that guarantee goes
// with them.

import { ARRAY_LAYOUT } from '../layoutNote.js';
import { plainToImageData, quantizePixel } from '../../engine/utils.js';

const CODE_BINS = 16; // categories in tasks 1, 2 and 4
const TARGET = 5; // the one bin task 1 counts
const LO = -32; // task 3: the bottom of the binned range
const SPAN = 64; // task 3: its width, so bins are 4 wide
const BINS = 16; // task 3: how many bins that span is cut into
const IMG_SIZE = 64; // task 5: the image is 64×64
const IMG_BINS = 32; // task 5: tone bins over luminance 0 … 1
const LUM = [0.299, 0.587, 0.114];

// ---- deterministic inputs --------------------------------------------------

// n category codes in 0 … categories−1, humped and skewed rather than uniform.
// That is not decoration: on a FLAT histogram several of the near-miss probes
// below would predict exactly the right answer and would have to stay silent.
function makeCodes(utils, n, categories, seed = 5309) {
  const rand = utils.seededRandom(seed);
  const data = new Array(n);
  for (let i = 0; i < n; i++) {
    const t = Math.pow((rand() + rand()) / 2, 1.3);
    data[i] = Math.min(categories - 1, Math.floor(t * categories));
  }
  return data;
}

// n sensor samples in −32 … 32 (2 dp), with the edge cases PLANTED: four
// samples sit exactly on the maximum (the phantom-bin case the clamp exists
// for) and ten sit exactly on a bin edge, the minimum included (the half-open
// case). Without them the two headline bugs of task 3 would be invisible.
function makeSamples(utils, n, seed = 1601) {
  const rand = utils.seededRandom(seed);
  const data = new Array(n);
  for (let i = 0; i < n; i++) {
    const t = Math.pow((rand() + rand()) / 2, 1.25);
    data[i] = Math.round((t * SPAN + LO) * 100) / 100;
  }
  const planted = [32, 32, 32, 32, -32, -28, -20, -12, -4, 0, 4, 12, 20, 28];
  planted.forEach((v, k) => {
    data[(k * 271 + 7) % n] = v;
  });
  return data;
}

// Constant-colour image, as an ImageData both backends read on the GPU. A
// kernel built with an ImageData must never be re-invoked with a nested array
// (on WebGL2 that quietly reads the wrong pixels), so every image a test
// passes in is one. Channels are quantized to what 8 bits can hold, which is
// what makes an expectation read off .plain exact.
function constantImage(size, pixel) {
  const row = new Array(size).fill(quantizePixel(pixel));
  return plainToImageData(new Array(size).fill(row));
}

// Two-tone image: the top half one colour, the bottom half another, so the
// whole histogram is two known spikes.
function twoToneImage(size, top, bottom) {
  const a = new Array(size).fill(quantizePixel(top));
  const b = new Array(size).fill(quantizePixel(bottom));
  const plain = new Array(size);
  for (let y = 0; y < size; y++) plain[y] = y < size / 2 ? a : b;
  return plainToImageData(plain);
}

// ---- reference histograms --------------------------------------------------

function sumOf(counts) {
  let total = 0;
  for (let i = 0; i < counts.length; i++) total += counts[i];
  return total;
}

function countOf(values, wanted) {
  let n = 0;
  for (let i = 0; i < values.length; i++) if (values[i] === wanted) n++;
  return n;
}

// counts[b] = how many values map to bin b under `index`. A value whose index
// falls outside 0 … bins−1 is DROPPED — exactly what a one-thread-per-bin
// kernel does with it, and the reason the total is the bug detector it is.
function tally(values, bins, index) {
  const counts = new Array(bins).fill(0);
  for (let i = 0; i < values.length; i++) {
    const b = index(values[i]);
    if (b >= 0 && b < bins) counts[b]++;
  }
  return counts;
}

function luminanceOf(pixel) {
  return LUM[0] * pixel[0] + LUM[1] * pixel[1] + LUM[2] * pixel[2];
}

// The correct task-3 mapping, and the three near-misses it is easy to write
// instead. Kept as named functions because both the reference answer and the
// probes are built from them.
function binOf(v) {
  return Math.min(BINS - 1, Math.floor((v - LO) / SPAN * BINS));
}

function binUnclamped(v) {
  return Math.floor((v - LO) / SPAN * BINS);
}

function binRounded(v) {
  return Math.min(BINS - 1, Math.round((v - LO) / SPAN * BINS));
}

function binUnshifted(v) {
  return Math.min(BINS - 1, Math.floor(v / SPAN * BINS));
}

// A bin whose BOTH edges are inclusive counts every value that sits exactly on
// an edge twice — so this reference cannot be written with tally().
function tallyClosedEdges(values, bins) {
  const counts = new Array(bins).fill(0);
  const width = SPAN / bins;
  for (let i = 0; i < values.length; i++) {
    const v = values[i];
    for (let b = 0; b < bins; b++) {
      const lo = LO + b * width;
      if (v >= lo && v <= lo + width) counts[b]++;
    }
  }
  return counts;
}

// Task 4: the histogram of one contiguous chunk, and of the strided slice a
// learner reaches for instead (Reductions taught the strided walk, and it is
// the wrong walk here — chunks are contiguous tiles).
function chunkTally(codes, chunk, chunkSize, bins) {
  const counts = new Array(bins).fill(0);
  for (let i = 0; i < chunkSize; i++) counts[codes[chunk * chunkSize + i]]++;
  return counts;
}

function stridedTally(codes, chunk, chunks, chunkSize, bins) {
  const counts = new Array(bins).fill(0);
  for (let i = 0; i < chunkSize; i++) counts[codes[i * chunks + chunk]]++;
  return counts;
}

function scaled(counts, factor) {
  return counts.map(c => c * factor);
}

// Task 5: the tone histogram of an image, from its host-side pixels.
function toneHistogram(plain, bins) {
  const counts = new Array(bins).fill(0);
  for (let y = 0; y < plain.length; y++) {
    for (let x = 0; x < plain[y].length; x++) {
      counts[Math.min(bins - 1, Math.floor(luminanceOf(plain[y][x]) * bins))]++;
    }
  }
  return counts;
}

// ---- near-miss diagnosis ---------------------------------------------------
//
// A failing assert that reports only two numbers tells a learner nothing about
// WHICH slip produced them. A probe pairs the value one specific known mistake
// would produce with a sentence naming that mistake; diagnose() speaks only
// when the observed value matches a probe within the test's own tolerance AND
// the correct value does not — so an observation where two candidates coincide
// stays silent, as do observations matching probes that disagree with each
// other. A wrong diagnosis is worse than none.
function diagnose(got, expected, eps, probes) {
  const hits = probes
    .filter(p => Math.abs(got - p[0]) <= eps && Math.abs(expected - p[0]) > eps)
    .map(p => p[1]);
  return hits.length && hits.every(m => m === hits[0]) ? hits[0] : null;
}

// The vector form, and the one this module leans on. A whole histogram is the
// observation, so a probe must predict EVERY bin (and disagree with the right
// answer somewhere) before it may speak: one matching bin means nothing, since
// two different binning bugs routinely agree about one bin. Counts are whole
// numbers, so half a count is all the tolerance anyone needs.
function diagnoseVector(got, expected, probes) {
  const hits = probes
    .filter(([candidate]) => {
      let differs = false;
      for (let b = 0; b < expected.length; b++) {
        if (!(Math.abs(got[b] - candidate[b]) <= 0.5)) return false;
        if (Math.abs(expected[b] - candidate[b]) > 0.5) differs = true;
      }
      return differs;
    })
    .map(p => p[1]);
  return hits.length && hits.every(m => m === hits[0]) ? hits[0] : null;
}

// The module's smoke alarm, said out loud. Every count is one input, so the
// counts MUST sum to the number of inputs; when they don't, the size and sign
// of the gap already names the family of bug.
function totalHint(counts, n, bins) {
  const total = Math.round(sumOf(Array.from(counts)));
  if (total === n) return null;
  if (total === n * bins) {
    return `the counts add up to ${total}, which is ${n} × ${bins} — every thread counted every ` +
      `input, so the test that keeps only this thread's own values is missing`;
  }
  if (total > n) {
    return `the counts add up to ${total}, more than the ${n} inputs — some inputs were counted ` +
      `by two bins at once, which is what an edge test that is inclusive at BOTH ends does`;
  }
  return `the counts add up to ${total}, fewer than the ${n} inputs — ${n - total} of them landed ` +
    `in no bin at all. A histogram has to account for every input exactly once, so a total that ` +
    `misses is the signature of a binning bug, usually an index that fell off one end`;
}

// Task 1: counting one bin, and the three ways of not quite doing it.
function targetProbes(codes, target) {
  return [
    [codes.length, 'every code was counted — the comparison never filtered anything out'],
    [countOf(codes, 0),
      'that is the count for code 0 — with output: [1] this.thread.x is always 0, so compare against this.constants.target instead'],
    [target * countOf(codes, target),
      'the matching values were added up instead of counted — a histogram adds 1 per match, not the value itself'],
  ];
}

// Task 2: the four histograms a learner gets from a slightly wrong comparison.
function codeHistogramProbes(codes, bins) {
  const ref = tally(codes, bins, v => v);
  const everything = new Array(bins).fill(codes.length);
  const bin0 = new Array(bins).fill(ref[0]);
  const shifted = ref.map((_, b) => (b + 1 < bins ? ref[b + 1] : 0));
  const summed = ref.map((c, b) => b * c);
  return [
    [everything, 'every thread counted every code — the if that keeps only this thread\'s own codes is missing'],
    [bin0, 'every bin came back with bin 0\'s count — this.thread.x has to appear in the comparison'],
    [shifted, 'the bins are shifted by one — thread x owns code x, not code x + 1'],
    [summed, 'the matching values were added up instead of counted — a histogram adds 1 per match, not the value itself'],
  ];
}

// Task 3: the four ways of mapping a value to a bin index slightly wrong.
function binningProbes(samples) {
  return [
    [tally(samples, BINS, binUnclamped),
      'the samples equal to the maximum, 32, map to bin 16 — a bin no thread owns — so they fell out of the histogram entirely; clamp the index with Math.min(this.constants.bins - 1, …)'],
    [tally(samples, BINS, binRounded),
      'the index was rounded, not floored — Math.round pushes every sample that is more than half way through its bin into the next one'],
    [tallyClosedEdges(samples, BINS),
      'samples sitting exactly on a bin edge were counted twice — a bin is half-open, lo ≤ v < hi, and only the last bin closes at the top'],
    [tally(samples, BINS, binUnshifted),
      'the − lo shift is missing — without it every negative sample maps to a negative bin index and drops out of the histogram'],
  ];
}

// Task 4: which HALF of the pipeline broke. Reading the grid transposed
// produces garbage that is undefined on the GL backend and merely wrong on the
// CPU one, so there is no value to probe for — but if every cell of the
// partial grid is right and the merged counts are not, the merge is broken by
// definition, and saying so is a fact rather than a guess.
function mergeStageHint(partial, merged, codes, bins, chunks, chunkSize) {
  for (let c = 0; c < chunks; c++) {
    const row = partial[c];
    const ref = chunkTally(codes, c, chunkSize, bins);
    for (let b = 0; b < bins; b++) {
      if (!(Math.abs(row[b] - ref[b]) <= 0.5)) return null; // pass one is the problem
    }
  }
  const shape = `output [bins, chunks] is indexed partial[chunk][bin] — ${chunks} rows of ${bins}, not ${bins} rows of ${chunks}`;
  return Number.isFinite(merged[0])
    ? `every cell of the partial grid is correct, so the merge is what broke. ${shape}, and every one of this.constants.chunks rows has to be added`
    : `every cell of the partial grid is correct, so the merge is what broke: it read past the end of a row. ${shape}`;
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

// Tasks 4 and 5 both build a 2D kernel and a 1D one. Identity by output RANK,
// not by creation order, so a learner who declares them the other way round
// still gets meaningful failures.
function findByRank(ctx) {
  let grid = null;
  let line = null;
  for (const k of ctx.kernels) {
    const output = k.kernel && k.kernel.output;
    if (!output || !output.length) continue;
    if (output.length === 2 && !grid) grid = k;
    else if (output.length === 1 && !line) line = k;
  }
  return { grid, line };
}

// Task 5: swapping this.thread.x and this.thread.y reads the transpose of the
// image. Invisible to the histogram (a transposed image has the SAME
// histogram), so the luminance map is the only place it can be caught. Cells
// on the diagonal are their own transpose and can never show it, which is why
// the case list below is entirely off-diagonal.
function transposeCellHint(got, transposed, eps, y, x) {
  return Math.abs(got - transposed) <= eps
    ? `that is the value for cell [${x}][${y}] — this.thread.x and this.thread.y are swapped. ` +
      'Rows come first: photo[this.thread.y][this.thread.x]'
    : null;
}

function luminanceProbes(pixel) {
  return [
    [(pixel[0] + pixel[1] + pixel[2]) / 3,
      'that is the plain channel average — luminance weights the channels 0.299 R + 0.587 G + 0.114 B'],
    [LUM[2] * pixel[0] + LUM[1] * pixel[1] + LUM[0] * pixel[2],
      'the weights are in the wrong order — 0.299 belongs on red and 0.114 on blue'],
    [pixel[0], 'only the red channel came through — luminance mixes all three'],
  ];
}

export default {
  uuid: 'dfb254f4-b68c-464e-af9e-1439efb7fcec',
  version: 1,
  slug: 'histograms-and-binning',
  title: 'Histograms & Binning',
  blurb: 'Counting values into bins with no atomics — the scatter that has to become a gather.',
  tasks: [
    // ------------------------------------------------------------------ 1
    {
      slug: 'lost-increment',
      title: 'The Increment That Vanishes',
      intro: `<p>On a CPU a histogram is three lines. Make an array of zeros, walk the data, add one
        to the bin each value belongs to. It is the friendliest loop in programming.</p>
<pre><code>const bins = new Array(16).fill(0);
for (let i = 0; i &lt; data.length; i++) bins[data[i]]++;</code></pre>
        <p>Now run that loop on 4,096 threads at once. <code>bins[v]++</code> is not one operation,
        it is three — <strong>read</strong> bin <em>v</em>, <strong>add</strong> one,
        <strong>write</strong> bin <em>v</em> back. Two threads whose values land in the same bin
        both read 7, both compute 8, both write 8. Two increments went in; one came out. Nothing
        crashed and nothing warned — a count is just quietly too low, and differently too low every
        time you run it.</p>
        <p>This is not a gpu.js quirk. It is precisely why CUDA ships <code>atomicAdd</code>: the
        read-modify-write has to become indivisible, and making it indivisible means the colliding
        threads take turns. gpu.js hands you no atomics and no scatter at all — a thread writes one
        cell, its own — which forces the formulation that actually transfers: <strong>invert the
        loop</strong>. Stop asking "which bin does my value go to?" and start asking "which values
        belong to <em>my</em> bin?". Start with one bin.</p>`,
      goal: `<strong>Goal:</strong> make the single thread count how many of the 4,096
        <code>codes</code> equal <code>this.constants.target</code>.`,
      requirements: [
        'Keep <code>output: [1]</code> — one thread, one bin, one count',
        'Loop <code>for (let i = 0; i &lt; this.constants.n; i++)</code> over every code',
        'Add <strong>1</strong> for each code equal to <code>this.constants.target</code> — the value itself is not what a histogram counts',
        'Return the count; no shared array is touched anywhere',
      ],
      hints: [
        {
          title: 'Hint 1 — an accumulator, not an array',
          body: `<p>The count lives in a local <code>let count = 0;</code> that only this thread can
            see. That is the whole reason there is nothing to race over: private variables cannot
            collide.</p>`,
        },
        {
          title: 'Hint 2 — the loop body',
          body: `<pre><code>if (codes[i] === this.constants.target) count++;</code></pre>`,
        },
      ],
      transfer: `Every compute API gives you the scatter this one withholds — and then charges for
        it. CUDA and HIP have <code>atomicAdd</code>, WGSL has <code>atomicAdd</code> on an
        <code>atomic&lt;u32&gt;</code> in a storage buffer, Metal has
        <code>atomic_fetch_add_explicit</code>. They are correct and they are not free: colliding
        threads serialize, and a histogram with one hot bin can reduce a whole warp to single file.
        The gather you are about to write is what the fast implementations fall back to when
        contention gets bad enough — which is why it is worth knowing even where atomics exist.`,
      starterCode: `// One bin, one thread. Nothing is shared, so nothing can race.
const gpu = new GPU({ mode });

const countBin = gpu.createKernel(function (codes) {
  // TODO: loop over all this.constants.n codes and count how many of
  // them equal this.constants.target. Add 1 per match — never the value.
  return 0;
}, {
  output: [1],
  constants: { n: 4096, target: 5 },
});

console.log('codes equal to the target:', (await countBin(codes))[0]);
`,
      solutionCode: `// One bin, one thread. Nothing is shared, so nothing can race.
const gpu = new GPU({ mode });

const countBin = gpu.createKernel(function (codes) {
  let count = 0;
  for (let i = 0; i < this.constants.n; i++) {
    if (codes[i] === this.constants.target) count++;
  }
  return count;
}, {
  output: [1],
  constants: { n: 4096, target: 5 },
});

console.log('codes equal to the target:', (await countBin(codes))[0]);
`,
      inputs: utils => ({ codes: makeCodes(utils, 4096, CODE_BINS, 5309) }),
      publicTests: [
        {
          name: 'one output cell holds one count',
          run: async ctx => {
            ctx.assert(ctx.kernels.length >= 1, 'no kernel was created — call gpu.createKernel()');
            const arr = new Array(4096);
            for (let i = 0; i < 4096; i++) arr[i] = i % CODE_BINS; // 256 of every code
            const out = await ctx.kernel(arr);
            ctx.assert(out && out.length === 1, `expected 1 output value, got ${out && out.length}`);
            const hint = diagnose(out[0], 256, 0.5, [
              [4096, 'every code was counted — only add 1 when codes[i] equals this.constants.target'],
              [TARGET * 256,
                'the matching values were added up instead of counted — a histogram adds 1 per match, not the value itself'],
            ]);
            ctx.assertClose(out[0], 256, 0.5, hint ||
              `this array holds 256 of every code, so the count for code ${TARGET} should be 256`);
          },
        },
        {
          name: `counts the codes equal to <code>${TARGET}</code> in a lopsided array`,
          run: async ctx => {
            const codes = makeCodes(ctx.utils, 4096, CODE_BINS, 5309);
            const out = await ctx.kernel(codes);
            const expected = countOf(codes, TARGET);
            const hint = diagnose(out[0], expected, 0.5, targetProbes(codes, TARGET));
            ctx.assertClose(out[0], expected, 0.5, hint || `the number of codes equal to ${TARGET}`);
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            const codes = makeCodes(ctx.utils, 4096, CODE_BINS, 4093);
            const out = await ctx.kernel(codes);
            ctx.assert(out && out.length === 1, 'expected 1 output value');
            const expected = countOf(codes, TARGET);
            const hint = diagnose(out[0], expected, 0.5, targetProbes(codes, TARGET));
            ctx.assertClose(out[0], expected, 0.5, hint || `the number of codes equal to ${TARGET}`);
          },
        },
      ],
    },

    // ------------------------------------------------------------------ 2
    {
      slug: 'one-thread-per-bin',
      title: 'One Thread Per Bin',
      intro: `<p>Run the last task sixteen times over, once per bin, and you have the whole
        histogram. <code>output: [16]</code> launches sixteen threads; thread <em>x</em> owns bin
        <em>x</em>, scans the entire array, and counts the codes that belong to it. Nobody writes
        into anybody else's cell, so there is nothing left to race over. The scatter became a
        gather — the same move <em>Thinking in Parallel</em> makes, wearing its most useful
        disguise.</p>
        <p>Say the price out loud, because it is real: every one of the 16 threads reads all 4,096
        codes, so this histogram costs <strong>n × bins</strong> reads where the CPU's cost
        <strong>n</strong>. You bought correctness with redundant work. On a GPU that is very often
        the right trade — the redundant reads run in parallel and hit cache, while the
        serialization an atomic costs does not parallelize at all — but it stops being the right
        trade as the bin count grows, and task 4 fixes the other end of it.</p>
        <p>One check catches almost every histogram bug ever written, so build the habit now:
        <strong>the counts must sum to the number of inputs.</strong> Every input belongs to exactly
        one bin, so 4,096 codes must produce counts totalling 4,096. Anything else means values are
        being dropped or double-counted, and the size of the gap usually tells you which.</p>`,
      goal: `<strong>Goal:</strong> produce all 16 counts in one kernel launch, then total them in
        plain JavaScript and log the total.`,
      requirements: [
        '<code>output: [16]</code> — one thread per bin, no loop over the bins',
        'Each thread scans all <code>this.constants.n</code> codes and counts only the ones equal to <code>this.thread.x</code>',
        'Sum the 16 returned counts in ordinary JavaScript and <code>console.log</code> the total (it should come to <code>4096</code>)',
      ],
      hints: [
        {
          title: 'Hint 1 — which bin am I?',
          body: `<p><code>this.thread.x</code> is both this thread's output cell <em>and</em> the
            code it is counting. That coincidence is the entire kernel: thread 5 counts the 5s.</p>`,
        },
        {
          title: 'Hint 2 — the loop body',
          body: `<pre><code>if (codes[i] === this.thread.x) count++;</code></pre>`,
        },
        {
          title: 'Hint 3 — the total',
          body: `<p>A plain loop after the kernel call:</p>
<pre><code>let total = 0;
for (let b = 0; b &lt; counts.length; b++) {
  total += counts[b];
}</code></pre>
<p>If that is not 4096, stop and find out why before you trust a single bar.</p>`,
        },
      ],
      transfer: `"One thread per output bucket, each scanning the input" is the shape shaders used
        for histograms for years before compute shaders and atomics existed, and it is still what
        libraries fall back to when the bin count is small and contention would be brutal. The
        general lesson outlives the example: when a parallel algorithm wants to write where it
        cannot, re-derive it so each output owner reads what it needs. CUDA, WGSL and Metal all
        reward that reformulation even where they would have let you scatter.`,
      starterCode: `// 16 threads, 16 bins. Thread x counts the codes equal to x.
const gpu = new GPU({ mode });

const histogram = gpu.createKernel(function (codes) {
  let count = 0;
  for (let i = 0; i < this.constants.n; i++) {
    // TODO: only count this code when it belongs to THIS thread's bin.
    count++;
  }
  return count;
}, {
  output: [16],
  constants: { n: 4096 },
});

const counts = await histogram(codes);
console.log('counts:', counts);

// TODO: total the 16 counts in plain JavaScript and log the total.
// A correct histogram of 4096 codes sums to 4096 — anything else is a bug.
`,
      solutionCode: `// 16 threads, 16 bins. Thread x counts the codes equal to x.
const gpu = new GPU({ mode });

const histogram = gpu.createKernel(function (codes) {
  let count = 0;
  for (let i = 0; i < this.constants.n; i++) {
    if (codes[i] === this.thread.x) count++;
  }
  return count;
}, {
  output: [16],
  constants: { n: 4096 },
});

const counts = await histogram(codes);
console.log('counts:', counts);

let total = 0;
for (let b = 0; b < counts.length; b++) total += counts[b];
console.log('total:', total);
`,
      inputs: utils => ({ codes: makeCodes(utils, 4096, CODE_BINS, 7717) }),
      publicTests: [
        {
          name: 'sixteen counts that sum to <code>4096</code>',
          run: async ctx => {
            ctx.assert(ctx.kernels.length >= 1, 'no kernel was created — call gpu.createKernel()');
            const codes = makeCodes(ctx.utils, 4096, CODE_BINS, 7717);
            const out = await ctx.kernel(codes);
            ctx.assert(out && out.length === CODE_BINS,
              `expected ${CODE_BINS} counts, one per bin, got ${out && out.length}`);
            const expected = tally(codes, CODE_BINS, v => v);
            const hint = diagnoseVector(out, expected, codeHistogramProbes(codes, CODE_BINS)) ||
              totalHint(out, 4096, CODE_BINS);
            ctx.assert(!hint, hint);
          },
        },
        {
          name: 'bin <em>x</em> holds the number of codes equal to <em>x</em>',
          run: async ctx => {
            const codes = makeCodes(ctx.utils, 4096, CODE_BINS, 7717);
            const out = await ctx.kernel(codes);
            const expected = tally(codes, CODE_BINS, v => v);
            const hint = diagnoseVector(out, expected, codeHistogramProbes(codes, CODE_BINS));
            for (let b = 0; b < CODE_BINS; b++) {
              ctx.assertClose(out[b], expected[b], 0.5, hint || `bin ${b}`);
            }
          },
        },
        {
          name: 'the total <code>4096</code> is computed and logged',
          run: async ctx => {
            const nums = loggedNumbers(ctx.logs);
            ctx.assert(
              nums.some(v => Math.abs(v - 4096) <= 0.5),
              'log the total of the 16 counts with console.log — expected to see 4096 in the console output'
            );
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            const codes = makeCodes(ctx.utils, 4096, CODE_BINS, 2029);
            const out = await ctx.kernel(codes);
            ctx.assert(out && out.length === CODE_BINS, `expected ${CODE_BINS} counts`);
            const expected = tally(codes, CODE_BINS, v => v);
            const probes = codeHistogramProbes(codes, CODE_BINS);
            const hint = diagnoseVector(out, expected, probes) || totalHint(out, 4096, CODE_BINS);
            for (let b = 0; b < CODE_BINS; b++) {
              ctx.assertClose(out[b], expected[b], 0.5, hint || `bin ${b}`);
            }
            ctx.assertClose(sumOf(Array.from(out)), 4096, 0.5, hint || 'the counts must sum to 4096');
          },
        },
      ],
    },

    // ------------------------------------------------------------------ 3
    {
      slug: 'binning-values',
      title: 'Where Does 7.35 Go?',
      intro: `<p>Real measurements are not tidy little category codes. <code>samples</code> holds
        4,096 sensor readings spread over −32 … 32, and sixteen bins across that span makes each bin
        4 units wide. Turning a reading into a bin index is one division and one floor:</p>
<pre><code>// lo = -32, span = 64, bins = 16
const bin = Math.floor((v - lo) / span * bins);</code></pre>
        <p>Two details decide whether the histogram is right, and both of them are where real bugs
        live. First, a bin is <strong>half-open</strong>: bin 14 is <code>[24, 28)</code>, so 24
        belongs to it and 28 belongs to bin 15. <code>Math.floor</code> gets that for free — which is
        exactly why the index is floored and not rounded.</p>
        <p>Second, the <strong>top edge</strong>. A reading exactly equal to the maximum maps to
        <code>(32 − −32) / 64 × 16 = 16</code> — bin 16, one past the last thread, owned by nobody.
        Four samples here sit exactly on it, and without a clamp all four silently stop existing:
        the counts come to 4,092 instead of 4,096. Clamp the index with
        <code>Math.min(bins − 1, …)</code> and they land in the last real bin, which is what closes
        that bin at the top.</p>
        <p>Every sample here is inside the range, so the clamp only ever has to catch the maximum.
        When data really <em>can</em> fall outside the range, clamping quietly piles the outliers
        into the end bins and the total will not say a word about it — so that becomes a decision to
        make on purpose: clamp them in, or drop them out. (And when the range comes from the data
        rather than from you, a min and a max reduction is where it comes from.)</p>`,
      goal: `<strong>Goal:</strong> histogram the 4,096 <code>samples</code> into 16 bins with a
        clamped index, so the counts total 4,096 — and log that total.`,
      requirements: [
        'Map each sample with <code>(v − this.constants.lo) / this.constants.span * this.constants.bins</code>, floored',
        'Clamp the index to <code>this.constants.bins - 1</code> so the maximum lands in the last bin instead of falling out',
        'Count a sample only when its bin equals <code>this.thread.x</code>',
        '<code>console.log</code> the total of the 16 counts — it must be <code>4096</code>',
      ],
      hints: [
        {
          title: 'Hint 1 — run it first',
          body: `<p>The starter already computes an unclamped index and already totals the counts.
            Run it: the total comes out 4,092. Four readings went into a bin that does not exist.
            That gap is the whole task.</p>`,
        },
        {
          title: 'Hint 2 — the clamp',
          body: `<pre><code>const bin = Math.min(this.constants.bins - 1, Math.floor(raw));</code></pre>
<p>— and nothing else changes.</p>`,
        },
        {
          title: 'Hint 3 — why floor and not round',
          body: `<p><code>Math.round</code> looks harmless and moves every reading that is more than
            half way through its bin into the next one — a histogram shifted by half a bin, with the
            right total. The total will not catch that one; only knowing the rule will.</p>`,
        },
      ],
      transfer: `Quantizing a continuous value into an integer index is everywhere in GPU work:
        picking a mip level, hashing a particle into a spatial grid cell, indexing a lookup table,
        choosing a colour ramp entry. Every platform ships the clamp as a primitive —
        <code>clamp()</code> in GLSL, WGSL and MSL, <code>__saturatef</code> and clamped texture
        address modes in CUDA — because the same off-by-one at the top edge has bitten everybody.
        NVIDIA's own histogram samples clamp for exactly this reason.`,
      starterCode: `// 16 bins over -32 ... 32, so every bin is 4 units wide.
const gpu = new GPU({ mode });

const histogram = gpu.createKernel(function (samples) {
  let count = 0;
  for (let i = 0; i < this.constants.n; i++) {
    const raw = (samples[i] - this.constants.lo)
      / this.constants.span * this.constants.bins;
    // TODO: floor alone sends a sample equal to the maximum to bin 16,
    // which no thread owns. Clamp the index to this.constants.bins - 1.
    const bin = Math.floor(raw);
    if (bin === this.thread.x) count++;
  }
  return count;
}, {
  output: [16],
  constants: { n: 4096, bins: 16, lo: -32, span: 64 },
});

const counts = await histogram(samples);
console.log('counts:', counts);

let total = 0;
for (let b = 0; b < counts.length; b++) total += counts[b];
console.log('total:', total);   // must be 4096, and right now it is not
`,
      solutionCode: `// 16 bins over -32 ... 32, so every bin is 4 units wide.
const gpu = new GPU({ mode });

const histogram = gpu.createKernel(function (samples) {
  let count = 0;
  for (let i = 0; i < this.constants.n; i++) {
    const raw = (samples[i] - this.constants.lo)
      / this.constants.span * this.constants.bins;
    const bin = Math.min(this.constants.bins - 1, Math.floor(raw));
    if (bin === this.thread.x) count++;
  }
  return count;
}, {
  output: [16],
  constants: { n: 4096, bins: 16, lo: -32, span: 64 },
});

const counts = await histogram(samples);
console.log('counts:', counts);

let total = 0;
for (let b = 0; b < counts.length; b++) total += counts[b];
console.log('total:', total);   // 4096 — every sample landed somewhere
`,
      inputs: utils => ({ samples: makeSamples(utils, 4096, 1601) }),
      publicTests: [
        {
          name: 'the counts still total <code>4096</code> — nothing fell off the top',
          run: async ctx => {
            ctx.assert(ctx.kernels.length >= 1, 'no kernel was created — call gpu.createKernel()');
            const samples = makeSamples(ctx.utils, 4096, 1601);
            const out = await ctx.kernel(samples);
            ctx.assert(out && out.length === BINS,
              `expected ${BINS} counts, one per bin, got ${out && out.length}`);
            const expected = tally(samples, BINS, binOf);
            const hint = diagnoseVector(out, expected, binningProbes(samples)) ||
              totalHint(out, 4096, BINS);
            ctx.assert(!hint, hint);
          },
        },
        {
          name: 'every bin holds the samples in <code>[lo + 4b, lo + 4b + 4)</code>',
          run: async ctx => {
            const samples = makeSamples(ctx.utils, 4096, 1601);
            const out = await ctx.kernel(samples);
            const expected = tally(samples, BINS, binOf);
            const hint = diagnoseVector(out, expected, binningProbes(samples));
            for (let b = 0; b < BINS; b++) {
              ctx.assertClose(out[b], expected[b], 0.5, hint ||
                `bin ${b} covers [${LO + 4 * b}, ${LO + 4 * b + 4})`);
            }
          },
        },
        {
          name: 'the total <code>4096</code> is computed and logged',
          run: async ctx => {
            const nums = loggedNumbers(ctx.logs);
            ctx.assert(
              nums.some(v => Math.abs(v - 4096) <= 0.5),
              'log the total of the 16 counts with console.log — expected to see 4096 in the console output'
            );
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            const samples = makeSamples(ctx.utils, 4096, 3607);
            const out = await ctx.kernel(samples);
            ctx.assert(out && out.length === BINS, `expected ${BINS} counts`);
            const expected = tally(samples, BINS, binOf);
            const probes = binningProbes(samples);
            const hint = diagnoseVector(out, expected, probes) || totalHint(out, 4096, BINS);
            for (let b = 0; b < BINS; b++) {
              ctx.assertClose(out[b], expected[b], 0.5, hint || `bin ${b}`);
            }
            ctx.assertClose(sumOf(Array.from(out)), 4096, 0.5, hint || 'the counts must sum to 4096');
          },
        },
        {
          name: 'private test #2',
          run: async ctx => {
            // A hand-built array whose every element sits exactly on a bin
            // edge: 256 copies of each of the 16 edges, the top one included.
            // Half-open bins put edge lo + 4b in bin b and the maximum, after
            // clamping, in bin 15 — so bin 15 holds twice what the others do.
            const edges = new Array(4096);
            for (let i = 0; i < 4096; i++) edges[i] = LO + (i % (BINS + 1)) * (SPAN / BINS);
            const out = await ctx.kernel(edges);
            const expected = tally(edges, BINS, binOf);
            const hint = diagnoseVector(out, expected, binningProbes(edges)) ||
              totalHint(out, 4096, BINS);
            for (let b = 0; b < BINS; b++) {
              ctx.assertClose(out[b], expected[b], 0.5, hint ||
                `bin ${b}, on an array made only of bin edges`);
            }
          },
        },
      ],
    },

    // ------------------------------------------------------------------ 4
    {
      slug: 'partial-histograms',
      title: 'Partial Histograms, Then Merge',
      intro: `<p>Sixteen bins is sixteen threads. A GPU with thousands of cores just sat out that
        entire kernel — and each of those sixteen threads had to walk all 16,384 codes by itself.
        Few bins over lots of data is exactly where one-thread-per-bin runs out of parallelism.</p>
        <p>So cut the data into chunks and give every <em>(bin, chunk)</em> pair its own thread.
        Thirty-two chunks of 512 codes turns 16 threads into 16 × 32 = 512, each scanning 512 codes
        instead of 16,384. What comes back is a grid of <strong>partial histograms</strong>: one row
        per chunk, one column per bin. A second pass then adds up each bin's column.</p>
        <p>Mind the shape. <code>output: [bins, chunks]</code> is given width-first but indexed
        row-first, so the grid you get back is <code>partial[chunk][bin]</code> — swap those two and
        you read off the end of a row. Pass two sums a column of 32 numbers, which one loop handles
        comfortably; at 4,096 chunks you would ride the halving ladder from <em>Reductions</em> down
        instead, because that is the same reduction wearing a different hat.</p>
        <p>Keep watching the total — but do not over-trust it here. If every chunk reads chunk 0's
        codes, the counts are entirely wrong and still sum to 16,384. The total catches lost and
        duplicated inputs; it cannot catch inputs you counted the wrong number of times each.</p>`,
      goal: `<strong>Goal:</strong> build the 16 × 32 grid of partial histograms in one kernel, then
        merge it into 16 final counts in a second.`,
      requirements: [
        '<code>partials</code>: <code>output: [16, 32]</code>, thread <code>(x = bin, y = chunk)</code> counts chunk <em>y</em>\'s codes that equal <em>x</em>',
        'Chunk <em>y</em> is contiguous: it starts at <code>this.thread.y * this.constants.chunkSize</code>',
        '<code>merge</code>: <code>output: [16]</code>, thread <em>x</em> sums <code>partial[c][this.thread.x]</code> over all <code>this.constants.chunks</code> chunks',
        'The merged counts sum to <code>16384</code>',
      ],
      hints: [
        {
          title: 'Hint 1 — where does my chunk start?',
          body: `<p>Chunk <em>y</em> owns the 512 codes from <code>y * 512</code> to
            <code>y * 512 + 511</code>, so its <em>i</em>-th code is at
            <code>this.thread.y * this.constants.chunkSize + i</code>. The starter is missing that
            offset, which is why every chunk currently reports chunk 0's histogram.</p>`,
        },
        {
          title: 'Hint 2 — the partials kernel',
          body: `<pre><code>const code = codes[this.thread.y * this.constants.chunkSize + i];
if (code === this.thread.x) count++;</code></pre>`,
        },
        {
          title: 'Hint 3 — the merge',
          body: `<p>One thread per bin, walking down that bin's column of the grid:</p>
<pre><code>let total = 0;
for (let c = 0; c &lt; this.constants.chunks; c++) {
  total += partial[c][this.thread.x];
}
return total;</code></pre>`,
        },
      ],
      transfer: `This is what a production GPU histogram actually does, and the reason is the same
        one: parallelism. A CUDA kernel gives each <em>block</em> a private histogram in shared
        memory, so its <code>atomicAdd</code>s stay on-chip and only conflict within the block, then
        spends one global <code>atomicAdd</code> per bin to merge. WGSL does it with a
        <code>var&lt;workgroup&gt;</code> array of atomics and a single merge at the end; CUB and
        rocPRIM's <code>DeviceHistogram</code> are this structure, tuned. Private partials plus a
        merge pass is the pattern — gpu.js just makes you write the merge as an honest reduction
        instead of hiding it behind an atomic.`,
      starterCode: `// Pass 1: one thread per (bin, chunk). Pass 2: merge each bin's column.
const gpu = new GPU({ mode });

const partials = gpu.createKernel(function (codes) {
  let count = 0;
  for (let i = 0; i < this.constants.chunkSize; i++) {
    // TODO: every chunk is reading chunk 0 right now. Chunk this.thread.y
    // starts at this.thread.y * this.constants.chunkSize.
    if (codes[i] === this.thread.x) count++;
  }
  return count;
}, {
  output: [16, 32],
  constants: { chunkSize: 512 },
});

const merge = gpu.createKernel(function (partial) {
  // TODO: add up all this.constants.chunks partial counts for THIS
  // thread's bin. The grid is indexed partial[chunk][bin].
  return partial[0][this.thread.x];
}, {
  output: [16],
  constants: { chunks: 32 },
});

const grid = await partials(codes);
const counts = await merge(grid);
console.log('counts:', counts);
`,
      solutionCode: `// Pass 1: one thread per (bin, chunk). Pass 2: merge each bin's column.
const gpu = new GPU({ mode });

const partials = gpu.createKernel(function (codes) {
  let count = 0;
  for (let i = 0; i < this.constants.chunkSize; i++) {
    const code = codes[this.thread.y * this.constants.chunkSize + i];
    if (code === this.thread.x) count++;
  }
  return count;
}, {
  output: [16, 32],
  constants: { chunkSize: 512 },
});

const merge = gpu.createKernel(function (partial) {
  let total = 0;
  for (let c = 0; c < this.constants.chunks; c++) {
    total += partial[c][this.thread.x];
  }
  return total;
}, {
  output: [16],
  constants: { chunks: 32 },
});

const grid = await partials(codes);
const counts = await merge(grid);
console.log('counts:', counts);
`,
      inputs: utils => ({ codes: makeCodes(utils, 16384, CODE_BINS, 8191) }),
      publicTests: [
        {
          name: 'two kernels: a 16 × 32 grid of partials, and a 16-bin merge',
          run: async ctx => {
            const { grid, line } = findByRank(ctx);
            ctx.assert(grid, 'no 2D kernel found — the partials kernel needs output: [16, 32]');
            ctx.assert(line, 'no 1D kernel found — the merge kernel needs output: [16]');
            const codes = makeCodes(ctx.utils, 16384, CODE_BINS, 8191);
            const out = await grid(codes);
            ctx.assert(out && out.length === 32,
              `expected 32 rows, one per chunk, got ${out && out.length} — output [bins, chunks] comes back row-first`);
            ctx.assert(out[0] && out[0].length === CODE_BINS,
              `expected ${CODE_BINS} counts per row, got ${out[0] && out[0].length}`);
          },
        },
        {
          name: 'row <em>c</em> is the histogram of chunk <em>c</em> alone',
          run: async ctx => {
            const { grid } = findByRank(ctx);
            ctx.assert(grid, 'no 2D partials kernel found');
            const codes = makeCodes(ctx.utils, 16384, CODE_BINS, 8191);
            const out = await grid(codes);
            for (const c of [0, 1, 17, 31]) {
              const expected = chunkTally(codes, c, 512, CODE_BINS);
              const hint = diagnoseVector(out[c], expected, [
                [chunkTally(codes, 0, 512, CODE_BINS),
                  'every chunk counted chunk 0\'s codes — chunk y starts at this.thread.y * this.constants.chunkSize'],
                [stridedTally(codes, c, 32, 512, CODE_BINS),
                  'that is the histogram of a strided slice — this task cuts the data into contiguous chunks, so chunk y is codes[y * chunkSize] … codes[y * chunkSize + chunkSize - 1]'],
                [new Array(CODE_BINS).fill(512),
                  'every bin in the row counted the whole chunk — the if that keeps only this thread\'s own codes is missing'],
              ]);
              for (let b = 0; b < CODE_BINS; b++) {
                ctx.assertClose(out[c][b], expected[b], 0.5, hint || `chunk ${c}, bin ${b}`);
              }
              ctx.assertClose(sumOf(Array.from(out[c])), 512, 0.5, hint ||
                `chunk ${c} holds 512 codes, so its partial histogram must sum to 512`);
            }
          },
        },
        {
          name: 'the merged counts match the whole-array histogram and sum to <code>16384</code>',
          run: async ctx => {
            const { grid, line } = findByRank(ctx);
            ctx.assert(grid && line, 'expected a 2D partials kernel and a 1D merge kernel');
            const codes = makeCodes(ctx.utils, 16384, CODE_BINS, 8191);
            const partial = await grid(codes);
            const out = await line(partial);
            ctx.assert(out && out.length === CODE_BINS,
              `expected ${CODE_BINS} merged counts, got ${out && out.length}`);
            const expected = tally(codes, CODE_BINS, v => v);
            const chunk0 = chunkTally(codes, 0, 512, CODE_BINS);
            const hint = diagnoseVector(out, expected, [
              [scaled(chunk0, 32),
                'every chunk counted chunk 0\'s codes, 32 times over — the total still comes to 16384, which is exactly why the total alone is not proof'],
              [chunk0, 'only chunk 0\'s partial came back — the merge has to add all this.constants.chunks of them'],
            ]) || mergeStageHint(partial, out, codes, CODE_BINS, 32, 512) ||
              totalHint(out, 16384, CODE_BINS);
            for (let b = 0; b < CODE_BINS; b++) {
              ctx.assertClose(out[b], expected[b], 0.5, hint || `bin ${b}`);
            }
            ctx.assertClose(sumOf(Array.from(out)), 16384, 0.5, hint || 'the counts must sum to 16384');
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            const { grid, line } = findByRank(ctx);
            ctx.assert(grid && line, 'expected a 2D partials kernel and a 1D merge kernel');
            const codes = makeCodes(ctx.utils, 16384, CODE_BINS, 6203);
            const partial = await grid(codes);
            const out = await line(partial);
            const expected = tally(codes, CODE_BINS, v => v);
            const chunk0 = chunkTally(codes, 0, 512, CODE_BINS);
            const hint = diagnoseVector(out, expected, [
              [scaled(chunk0, 32),
                'every chunk counted chunk 0\'s codes — chunk y starts at this.thread.y * this.constants.chunkSize'],
              [chunk0, 'only chunk 0\'s partial came back — the merge has to add all 32 of them'],
            ]) || mergeStageHint(partial, out, codes, CODE_BINS, 32, 512) ||
              totalHint(out, 16384, CODE_BINS);
            for (let c = 0; c < 32; c++) {
              ctx.assertClose(sumOf(Array.from(partial[c])), 512, 0.5, hint ||
                `chunk ${c}'s partial histogram must sum to its 512 codes`);
            }
            for (let b = 0; b < CODE_BINS; b++) {
              ctx.assertClose(out[b], expected[b], 0.5, hint || `bin ${b}`);
            }
          },
        },
      ],
    },

    // ------------------------------------------------------------------ 5
    {
      slug: 'image-histogram',
      title: 'Payoff: An Image\'s Tone Histogram',
      intro: `<p>The payoff, and the histogram everybody has actually seen: an image's
        <strong>tone histogram</strong> — how many pixels are dark, how many mid, how many bright.
        Every photo editor draws one, because it tells you a shot is underexposed before your eyes
        do.</p>
        <p>Two kernels, and the reason for two is worth a sentence. Luminance is a per-pixel
        calculation and there are 4,096 pixels — but there are 32 bins, so a single histogram kernel
        would recompute every pixel's luminance <em>32 times over</em>, once per bin thread. Compute
        it once into a 64 × 64 map, then histogram the map. Map first, bin second; the map pass is
        4,096 luminance evaluations instead of 131,072.</p>
        <p>Luminance runs 0 … 1, so 32 bins over that range is a bin every 0.03125 — the same
        clamped <code>floor</code> as task 3, with <code>lo = 0</code> and <code>span = 1</code>
        doing nothing visible. And the same smoke alarm: 4,096 pixels in, 4,096 counted out.</p>
        ${ARRAY_LAYOUT}`,
      goal: `<strong>Goal:</strong> compute a 64 × 64 luminance map of <code>photo</code>, histogram
        it into 32 tone bins, and log the total.`,
      requirements: [
        '<code>luminance</code>: <code>output: [64, 64]</code>, each cell <code>0.299r + 0.587g + 0.114b</code> of that pixel',
        '<code>histogram</code>: <code>output: [32]</code>, each thread scans the whole map',
        'Bin with the clamped index from task 3: <code>Math.min(bins - 1, Math.floor(l * bins))</code>',
        '<code>console.log</code> the total of the 32 counts — it must be <code>4096</code>',
      ],
      hints: [
        {
          title: 'Hint 1 — the map pass',
          body: `<p>Straight out of any grayscale kernel — read this thread's pixel and return a
            number instead of painting it:</p>
<pre><code>const pixel = photo[this.thread.y][this.thread.x];
return 0.299 * pixel[0] + 0.587 * pixel[1] + 0.114 * pixel[2];</code></pre>`,
        },
        {
          title: 'Hint 2 — scanning a 2D map from a 1D kernel',
          body: `<p>The histogram kernel has 32 threads and a 64 × 64 map, so each thread runs two
            nested loops over the map. Both bounds are constants, which is what the WebGL backend
            needs:</p>
<pre><code>for (let y = 0; y &lt; this.constants.size; y++) {
  for (let x = 0; x &lt; this.constants.size; x++) {
    const bin = Math.min(
      this.constants.bins - 1,
      Math.floor(map[y][x] * this.constants.bins)
    );
    if (bin === this.thread.x) count++;
  }
}</code></pre>`,
        },
        {
          title: 'Hint 3 — read the shape of the answer',
          body: `<p>Once it runs, look at the counts: the first bins and the last bins are empty.
            This image never gets truly black or truly white — which is precisely the thing a tone
            histogram exists to tell you.</p>`,
        },
      ],
      transfer: `Tone histograms are load-bearing infrastructure, not a readout: auto-exposure,
        auto-contrast and histogram equalization all start here, and phone ISPs compute one in fixed
        function hardware on every frame. The two-pass shape generalizes past images — derive the
        quantity once into a buffer, then bin the buffer — and it is the same reason CUDA and WebGPU
        pipelines materialize an intermediate rather than recomputing inside an inner loop. Turning
        these counts into a cumulative curve (the next step of equalization) is a prefix sum, which
        is the one parallel primitive this module does not need.`,
      starterCode: `// Map first (one luminance per pixel), bin second (one thread per bin).
const gpu = new GPU({ mode });

const luminance = gpu.createKernel(function (photo) {
  const pixel = photo[this.thread.y][this.thread.x];
  // TODO: return perceptual luminance — 0.299 R + 0.587 G + 0.114 B
  return pixel[0];
}, { output: [64, 64] });

const histogram = gpu.createKernel(function (map) {
  let count = 0;
  for (let y = 0; y < this.constants.size; y++) {
    for (let x = 0; x < this.constants.size; x++) {
      // TODO: bin map[y][x] into 0 ... bins - 1 with a clamped floor,
      // and count it only when that bin is this thread's own.
      count++;
    }
  }
  return count;
}, {
  output: [32],
  constants: { size: 64, bins: 32 },
});

const map = await luminance(photo);
const counts = await histogram(map);
console.log('counts:', counts);

// TODO: total the counts and log the total. 4096 pixels in, 4096 counted.
`,
      solutionCode: `// Map first (one luminance per pixel), bin second (one thread per bin).
const gpu = new GPU({ mode });

const luminance = gpu.createKernel(function (photo) {
  const pixel = photo[this.thread.y][this.thread.x];
  return 0.299 * pixel[0] + 0.587 * pixel[1] + 0.114 * pixel[2];
}, { output: [64, 64] });

const histogram = gpu.createKernel(function (map) {
  let count = 0;
  for (let y = 0; y < this.constants.size; y++) {
    for (let x = 0; x < this.constants.size; x++) {
      const bin = Math.min(
        this.constants.bins - 1,
        Math.floor(map[y][x] * this.constants.bins)
      );
      if (bin === this.thread.x) count++;
    }
  }
  return count;
}, {
  output: [32],
  constants: { size: 64, bins: 32 },
});

const map = await luminance(photo);
const counts = await histogram(map);
console.log('counts:', counts);

let total = 0;
let fullest = 0;
for (let b = 0; b < counts.length; b++) {
  total += counts[b];
  if (counts[b] > counts[fullest]) fullest = b;
}
console.log('total:', total);
console.log('fullest bin:', fullest, 'of', counts.length);
`,
      inputs: utils => ({ photo: utils.makeTestImage(IMG_SIZE) }),
      publicTests: [
        {
          name: 'two kernels: a 64 × 64 map and a 32-bin histogram',
          run: async ctx => {
            const { grid, line } = findByRank(ctx);
            ctx.assert(grid, 'no 2D kernel found — the luminance map needs output: [64, 64]');
            ctx.assert(line, 'no 1D kernel found — the histogram needs output: [32]');
            const map = await grid(ctx.utils.makeTestImage(IMG_SIZE));
            ctx.assert(map && map.length === IMG_SIZE, `expected ${IMG_SIZE} rows, got ${map && map.length}`);
            ctx.assert(map[0] && map[0].length === IMG_SIZE, `each row should hold ${IMG_SIZE} values`);
            const counts = await line(map);
            ctx.assert(counts && counts.length === IMG_BINS,
              `expected ${IMG_BINS} counts, got ${counts && counts.length}`);
          },
        },
        {
          name: 'the map holds <code>0.299r + 0.587g + 0.114b</code> per pixel',
          run: async ctx => {
            const { grid } = findByRank(ctx);
            ctx.assert(grid, 'no 2D luminance kernel found');
            const image = ctx.utils.makeTestImage(IMG_SIZE);
            const map = await grid(image);
            const plain = image.plain;
            for (const [y, x] of [[0, 3], [7, 41], [33, 12], [62, 5]]) {
              const expected = luminanceOf(plain[y][x]);
              const hint = transposeCellHint(map[y][x], luminanceOf(plain[x][y]), 2e-3, y, x) ||
                diagnose(map[y][x], expected, 2e-3, luminanceProbes(plain[y][x]));
              ctx.assertClose(map[y][x], expected, 2e-3, hint || `cell [${y}][${x}]`);
            }
          },
        },
        {
          name: 'the tone histogram matches, and totals <code>4096</code>',
          run: async ctx => {
            const { grid, line } = findByRank(ctx);
            ctx.assert(grid && line, 'expected a luminance kernel and a histogram kernel');
            const image = ctx.utils.makeTestImage(IMG_SIZE);
            const counts = await line(await grid(image));
            const expected = toneHistogram(image.plain, IMG_BINS);
            const hint = diagnoseVector(counts, expected, [
              [new Array(IMG_BINS).fill(IMG_SIZE * IMG_SIZE),
                'every thread counted every pixel — the if that keeps only this thread\'s own bin is missing'],
              [expected.map((_, b) => (b + 1 < IMG_BINS ? expected[b + 1] : 0)),
                'the bins are shifted by one — thread x owns bin x'],
            ]) || totalHint(counts, IMG_SIZE * IMG_SIZE, IMG_BINS);
            for (let b = 0; b < IMG_BINS; b++) {
              ctx.assertClose(counts[b], expected[b], 0.5, hint ||
                `bin ${b} covers luminance [${(b / IMG_BINS).toFixed(4)}, ${((b + 1) / IMG_BINS).toFixed(4)})`);
            }
            ctx.assertClose(sumOf(Array.from(counts)), IMG_SIZE * IMG_SIZE, 0.5,
              hint || 'the counts must sum to the 4096 pixels');
          },
        },
        {
          name: 'the total <code>4096</code> is computed and logged',
          run: async ctx => {
            const nums = loggedNumbers(ctx.logs);
            ctx.assert(
              nums.some(v => Math.abs(v - IMG_SIZE * IMG_SIZE) <= 0.5),
              'log the total of the 32 counts with console.log — expected to see 4096 in the console output'
            );
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            // A single flat colour: every one of the 4096 pixels must land in
            // the same bin, and no other bin may hold anything at all.
            const { grid, line } = findByRank(ctx);
            ctx.assert(grid && line, 'expected a luminance kernel and a histogram kernel');
            const image = constantImage(IMG_SIZE, [0.2, 0.4, 0.6, 1]);
            const counts = await line(await grid(image));
            const expected = toneHistogram(image.plain, IMG_BINS);
            const hint = totalHint(counts, IMG_SIZE * IMG_SIZE, IMG_BINS);
            for (let b = 0; b < IMG_BINS; b++) {
              ctx.assertClose(counts[b], expected[b], 0.5, hint ||
                `bin ${b} on a flat-colour image — every pixel shares one luminance, so exactly one bin holds 4096`);
            }
          },
        },
        {
          name: 'private test #2',
          run: async ctx => {
            // Two flat halves: two spikes of 2048, everything else empty.
            const { grid, line } = findByRank(ctx);
            ctx.assert(grid && line, 'expected a luminance kernel and a histogram kernel');
            const image = twoToneImage(IMG_SIZE, [0.1, 0.15, 0.2, 1], [0.7, 0.8, 0.6, 1]);
            const counts = await line(await grid(image));
            const expected = toneHistogram(image.plain, IMG_BINS);
            const hint = totalHint(counts, IMG_SIZE * IMG_SIZE, IMG_BINS);
            for (let b = 0; b < IMG_BINS; b++) {
              ctx.assertClose(counts[b], expected[b], 0.5, hint ||
                `bin ${b} on a two-tone image — exactly two bins should hold 2048 each`);
            }
          },
        },
      ],
    },
  ],
};
