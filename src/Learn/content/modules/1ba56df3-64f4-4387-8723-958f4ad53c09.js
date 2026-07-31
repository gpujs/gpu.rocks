// Module: Top-K Selection — uuid 1ba56df3-64f4-4387-8723-958f4ad53c09
// (short id 1ba56df3). The file name is the uuid; identity lives in the
// exported object below, never in the path. No legacyId: this module postdates
// the uuid switch, and it belongs to no track — it lands in "Others", which is
// unordered, so nothing here assumes a neighbouring module has been done.
//
// Top-K Selection.
//
// Five tasks: rank by counting (and the tie that breaks it) → invert the
// scatter to gather a packed, sorted result → the same two moves on a 2D grid,
// carrying indices instead of values → the linear alternative, a bisection on
// the value axis → and the honest comparison of the two formulations.
//
// The module is really about one character: whether a comparison is > or >=.
// Task 1 needs both (an earlier equal element outranks you, a later one does
// not); task 4's bracket needs >= or it settles one element too low. Every
// gotcha probe in here is downstream of that.
//
// Kernel-authoring rules (contract): no closures inside kernel functions, loop
// bounds come from this.constants (compile-time known), only numbers / nested
// numeric arrays as arguments, and every task passes in CPU mode. Sizes stay
// at 4,096 for the O(n²) tasks and 65,536 for the linear ones so verification
// is fast — except in task 5, which has to show the two formulations trading
// places and therefore runs the quadratic pass at 131,072 as well. That one
// declares a budgetMs and skips its big ranking pass on the CPU backend; the
// reasoning is written out beside the declaration.

// ---- deterministic inputs --------------------------------------------------

// Relevance scores, whole numbers, deliberately lumpy: cubing a uniform draw
// piles most documents near zero and spreads the good ones out, so the array
// is full of REPEATS — which is the whole point of tasks 1–3. At seed 2026 the
// 10th and 11th scores are both 993, so no threshold on earth separates them
// and only the index tie-break can.
function makeScores(utils, n, seed = 2026) {
  const rand = utils.seededRandom(seed);
  const data = new Array(n);
  for (let i = 0; i < n; i++) data[i] = Math.round(Math.pow(rand(), 3) * 1000);
  return data;
}

// Finer-grained scores for the threshold tasks: an exponential tail spreads
// the top of the list over thousands of points, so the k-th and (k+1)-th
// scores differ and an exact cutoff exists. Still whole numbers — which is
// what makes "stop when the bracket is narrower than 1" a legal stopping rule.
function makeFineScores(utils, n, seed) {
  const rand = utils.seededRandom(seed);
  const data = new Array(n);
  for (let i = 0; i < n; i++) data[i] = Math.round(-8000 * Math.log(1 - rand()));
  return data;
}

// A size×size brightness field: three soft glows plus sensor noise, quantized
// to whole numbers so the flat dark regions are full of ties.
function makeBrightness(utils, size, seed = 1701) {
  const rand = utils.seededRandom(seed);
  const grid = new Array(size);
  for (let y = 0; y < size; y++) {
    const row = new Array(size);
    const ny = y / size;
    for (let x = 0; x < size; x++) {
      const nx = x / size;
      const glow =
        Math.exp(-((nx - 0.22) * (nx - 0.22) + (ny - 0.68) * (ny - 0.68)) * 30) +
        0.82 * Math.exp(-((nx - 0.74) * (nx - 0.74) + (ny - 0.31) * (ny - 0.31)) * 55) +
        0.55 * Math.exp(-((nx - 0.55) * (nx - 0.55) + (ny - 0.86) * (ny - 0.86)) * 90);
      row[x] = Math.round(600 * glow + 40 * rand());
    }
    grid[y] = row;
  }
  return grid;
}

// ---- host-side references --------------------------------------------------

// THE rank: how many elements outrank this one, where an element earlier in
// the array wins a tie and a later one loses it. Because that is a total
// order, the result is a permutation of 0…n−1 — exactly one element per slot.
// Computed by sorting rather than by the kernel's O(n²) count: the tests call
// it often, and both definitions agree by construction.
function ranksOf(values) {
  const order = new Array(values.length);
  for (let i = 0; i < values.length; i++) order[i] = i;
  order.sort((a, b) => values[b] - values[a] || a - b);
  const ranks = new Array(values.length);
  for (let r = 0; r < order.length; r++) ranks[order[r]] = r;
  return ranks;
}

// The k largest values, largest first.
function topValues(values, k) {
  const ranks = ranksOf(values);
  const out = new Array(k).fill(0);
  for (let i = 0; i < values.length; i++) if (ranks[i] < k) out[ranks[i]] = values[i];
  return out;
}

// How many values are strictly above t — the quantity the whole threshold half
// of this module is built on. Named for the host side, so a test reads clearly
// beside the learner's kernel of the same job.
function howManyAbove(values, t) {
  let n = 0;
  for (let i = 0; i < values.length; i++) if (values[i] > t) n++;
  return n;
}

// Total of a kernel's partial counts, the way the tasks do it in JavaScript.
function sumOf(partials) {
  let sum = 0;
  for (let i = 0; i < partials.length; i++) sum += partials[i];
  return sum;
}

function flatten2D(grid) {
  const out = [];
  for (let y = 0; y < grid.length; y++) {
    for (let x = 0; x < grid[y].length; x++) out.push(grid[y][x]);
  }
  return out;
}

// The flat index of the cell at the mirrored position: y * size + x becomes
// x * size + y. What a picker that assembles its answer column-first returns.
function swapFlat(index, size) {
  return (index % size) * size + Math.floor(index / size);
}

// The flat (row-major) indices of the k brightest cells, brightest first.
function topIndices(grid, k) {
  const flat = flatten2D(grid);
  const ranks = ranksOf(flat);
  const out = new Array(k).fill(-1);
  for (let i = 0; i < flat.length; i++) if (ranks[i] < k) out[ranks[i]] = i;
  return out;
}

// The strided partial counts the task-4 kernel is supposed to produce.
function partialCounts(values, t, threads, chunk) {
  const out = new Array(threads).fill(0);
  for (let x = 0; x < threads; x++) {
    let n = 0;
    for (let i = 0; i < chunk; i++) if (values[i * threads + x] > t) n++;
    out[x] = n;
  }
  return out;
}

// The bracket a bisection starts from: below lo everything passes, above hi
// nothing does.
function bracketOf(values) {
  let lo = values[0];
  let hi = values[0];
  for (let i = 1; i < values.length; i++) {
    if (values[i] < lo) lo = values[i];
    if (values[i] > hi) hi = values[i];
  }
  return [lo - 1, hi];
}

// ---- the wrong answers, by name --------------------------------------------
//
// Each of these is a complete alternative result: what the whole output array
// looks like when one specific slip is made. diagnoseArray() below only speaks
// when the observation matches one of them EVERYWHERE, which is what makes a
// named diagnosis safe to print.

// `>` for every j — no tie-break at all. Tied scores land on the same rank, so
// some ranks are shared and the ones in between are never used.
function strictRanksOf(values) {
  const sorted = [...values].sort((a, b) => b - a);
  const firstAt = new Map();
  for (let i = 0; i < sorted.length; i++) {
    if (!firstAt.has(sorted[i])) firstAt.set(sorted[i], i);
  }
  return values.map(v => firstAt.get(v));
}

// `>=` for every j — your own score is in the count, so the largest comes back
// as 1 instead of 0 and every tie group shares the group's LAST slot.
function inclusiveRanksOf(values) {
  const sorted = [...values].sort((a, b) => b - a);
  const lastAt = new Map();
  for (let i = 0; i < sorted.length; i++) lastAt.set(sorted[i], i + 1);
  return values.map(v => lastAt.get(v));
}

// The comparison inverted: counting the scores you beat instead of the scores
// that beat you. Rank 0 becomes the SMALLEST value — the bottom k, not the top.
function ascendingRanksOf(values) {
  return ranksOf(values.map(v => -v));
}

// The tie-break aimed the wrong way: a LATER equal element takes precedence.
// Indistinguishable from the right answer on data with no repeats, which is
// why the tasks that probe for it use data that has plenty.
function lateTieRanksOf(values) {
  const order = new Array(values.length);
  for (let i = 0; i < values.length; i++) order[i] = i;
  order.sort((a, b) => values[b] - values[a] || b - a);
  const ranks = new Array(values.length);
  for (let r = 0; r < order.length; r++) ranks[order[r]] = r;
  return ranks;
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

function matchesEvery(got, want, eps) {
  if (!got || got.length < want.length) return false;
  for (let i = 0; i < want.length; i++) {
    if (!(Math.abs(got[i] - want[i]) <= eps)) return false;
  }
  return true;
}

// The array-shaped form. One agreeing element proves nothing when 4,096 of
// them agree by accident, so a probe here has to predict EVERY element of the
// observed output and disagree with the right answer somewhere before it may
// speak.
function diagnoseArray(got, expected, eps, probes) {
  const hits = probes
    .filter(([want]) => matchesEvery(got, want, eps) && !matchesEvery(expected, want, eps))
    .map(p => p[1]);
  return hits.length && hits.every(m => m === hits[0]) ? hits[0] : null;
}

function firstMismatch(got, want, eps) {
  if (!got) return 0;
  for (let i = 0; i < want.length; i++) {
    if (!(Math.abs(got[i] - want[i]) <= eps)) return i;
  }
  return -1;
}

// The four ways a rank-by-counting kernel comes out wrong. Built only when a
// test has already found a mismatch — each one costs an n log n sort.
function rankAlternatives(values) {
  return [
    [strictRanksOf(values),
      'every tie collapsed onto one rank — equal scores each counted the same elements ahead of ' +
      'them, so some ranks are claimed twice and the ones in between are claimed by nobody. An ' +
      'element EARLIER in the array has to win the tie: use >= when j < this.thread.x'],
    [inclusiveRanksOf(values),
      'every element counted itself — a >= against every j puts your own score in the total, so ' +
      'the largest comes back as 1 instead of 0. Only equal scores at a LOWER index should count'],
    [ascendingRanksOf(values),
      'that is the rank from the bottom: you counted the scores you beat instead of the scores ' +
      'that beat you. Rank 0 has to mean "nothing outranks me"'],
    [lateTieRanksOf(values),
      'the tie-break points the wrong way — a LATER equal score is taking precedence. The test is ' +
      '>= for j < this.thread.x and > for everyone else, so the earlier element wins'],
  ];
}

// ---- log reading -----------------------------------------------------------

// The first console.log line containing `needle`, or null.
function lineWith(logs, needle) {
  for (const line of logs) {
    if (line.type === 'log' && line.text && line.text.indexOf(needle) !== -1) return line.text;
  }
  return null;
}

// The number that follows `label` on that line — 'cutoff: 68327' → 68327.
// Reading a labelled number rather than every number on the line keeps a
// timing or an array length from being mistaken for the answer.
function numberAfter(text, label) {
  if (!text) return null;
  const match = new RegExp(`${label}\\s*(-?\\d+(?:\\.\\d+)?)`).exec(text);
  return match ? parseFloat(match[1]) : null;
}

// Did this line report a duration at all? '… 12.4 ms' → 12.4.
function millisecondsOn(text) {
  if (!text) return null;
  const match = /(-?\d+(?:\.\d+)?)\s*ms/.exec(text);
  return match ? parseFloat(match[1]) : null;
}

// ---- kernel lookup ---------------------------------------------------------

// The kernel with this exact output shape. Identity by shape, not by creation
// order: a learner who rearranges the starter still gets useful failures.
function kernelWithOutput(ctx, dims) {
  return (
    ctx.kernels.find(k => {
      const out = k.kernel && k.kernel.output;
      return (
        out && out.length === dims.length && dims.every((d, i) => out[i] === d)
      );
    }) || null
  );
}

// What a cutoff that lets the wrong number of scores through is trying to say.
function cutoffHint(values, t, k) {
  if (t === null) return null;
  const above = howManyAbove(values, t);
  if (above === k) return null;
  if (above === k + 1) {
    return `${above} scores clear that cutoff, not ${k} — the bracket settled one element too ` +
      `low. The invariant is "at least k scores are above lo", so the guess is kept when ` +
      `count >= k, not when count > k`;
  }
  if (above === k - 1) {
    return `only ${above} scores clear that cutoff — it settled one element too high. Report lo ` +
      `(the side of the bracket that still lets k through), not hi, and count with > rather ` +
      `than >=`;
  }
  if (above > values.length / 2) {
    return `${above.toLocaleString('en-US')} scores clear that cutoff — the bracket never moved. ` +
      `Check that the loop actually narrows it: one of lo or hi has to become mid every pass`;
  }
  return `${above.toLocaleString('en-US')} scores clear that cutoff, but exactly ${k} should`;
}

export default {
  uuid: '1ba56df3-64f4-4387-8723-958f4ad53c09',
  version: 1,
  slug: 'top-k-selection',
  title: 'Top-K Selection',
  blurb: 'The ten largest of a million values: rank by counting, gather the winners, or bisect for a cutoff — and when each one wins.',
  tasks: [
    // ------------------------------------------------------------------ 1
    {
      slug: 'rank-by-counting',
      title: 'Rank by Counting',
      intro: `<p>"Give me the ten largest of these four thousand scores." On a CPU you keep a heap
        of ten and walk the data once — and that plan does not port, because the heap's contents
        after element <em>i</em> depend on every element before it. Serial by construction.</p>
        <p>So ask a question every element can answer <em>alone</em>: <strong>how many scores beat
        me?</strong> That count is the element's <strong>rank</strong>, rank 0 means nothing beats
        it, and anything with a rank below <code>k</code> is in the top <code>k</code>. No sorting,
        no shared state, one thread per element — each of them reading the whole array, which makes
        this O(n²) work and gloriously parallel.</p>
        <p>Ties are where it bites. Two equal scores each counting the other come back with the
        <em>same</em> rank: two elements claim one slot, and the slot after it is claimed by
        nobody. The fix is a <strong>tie-break on the index</strong> — an element earlier in the
        array outranks you when the scores are equal, a later one does not. That turns the ranks
        into a permutation of 0…4095, exactly one element per slot. These scores repeat constantly,
        so you will feel it immediately.</p>`,
      goal: `<strong>Goal:</strong> return, for each element, the number of scores that outrank it —
        strictly larger anywhere, or <em>equal at a lower index</em>.`,
      requirements: [
        'One thread per score: <code>output: [4096]</code>, loop bound <code>this.constants.n</code>',
        'A strictly larger score always counts',
        'An equal score counts only when its index is below <code>this.thread.x</code>',
        'The largest score must come back with rank <code>0</code>',
      ],
      hints: [
        {
          title: 'Hint 1 — one loop, two comparisons',
          body: `<p>Split on the index, not on the value. For <code>j &lt; this.thread.x</code> an
            equal score wins, so that side tests <code>&gt;=</code>; for every other <code>j</code>
            an equal score loses, so that side tests <code>&gt;</code>.</p>`,
        },
        {
          title: 'Hint 2 — the loop body',
          body: `<pre><code>const other = scores[j];
if (j &lt; this.thread.x) {
  if (other &gt;= mine) ahead++;
} else if (other &gt; mine) {
  ahead++;
}</code></pre>`,
        },
        {
          title: 'Hint 3 — checking yourself',
          body: `<p>Every rank from 0 to 4095 should appear exactly <em>once</em>. If two elements
            share a rank, then somewhere a <code>&gt;</code> is doing a <code>&gt;=</code>'s job (or
            the other way round).</p>`,
        },
      ],
      transfer: `Counting ranks is how a GPU sorts small things — it is the first sort in every CUDA
        and WebGPU tutorial, and the reason CUB's <code>DeviceRadixSort</code> and bitonic networks
        exist is that O(n²) stops being free somewhere above a few thousand elements. The index
        tie-break is what makes such a sort <em>stable</em>, the same guarantee
        <code>thrust::stable_sort</code> and <code>std::stable_sort</code> sell.`,
      starterCode: `// One thread per score. Each one asks: how many scores beat mine?
const gpu = new GPU({ mode });

const rankScores = gpu.createKernel(function (scores) {
  const mine = scores[this.thread.x];
  let ahead = 0;
  for (let j = 0; j < this.constants.n; j++) {
    // TODO: this counts every element. Count scores[j] only when it
    // outranks mine — strictly larger, or equal with j below
    // this.thread.x.
    ahead++;
  }
  return ahead;
}, {
  output: [4096],
  constants: { n: 4096 },
});

const ranks = rankScores(scores);
console.log('rank of element 0:', ranks[0], '(its score is', scores[0] + ')');
`,
      solutionCode: `// One thread per score. Each one asks: how many scores beat mine?
const gpu = new GPU({ mode });

const rankScores = gpu.createKernel(function (scores) {
  const mine = scores[this.thread.x];
  let ahead = 0;
  for (let j = 0; j < this.constants.n; j++) {
    const other = scores[j];
    if (j < this.thread.x) {
      // an earlier element wins a tie
      if (other >= mine) ahead++;
    } else if (other > mine) {
      ahead++;
    }
  }
  return ahead;
}, {
  output: [4096],
  constants: { n: 4096 },
});

const ranks = rankScores(scores);
console.log('rank of element 0:', ranks[0], '(its score is', scores[0] + ')');
`,
      inputs: utils => ({ scores: makeScores(utils, 4096) }),
      publicTests: [
        {
          name: 'every rank 0…4095 is used exactly once',
          run: async ctx => {
            ctx.assert(ctx.kernels.length >= 1, 'no kernel was created — call gpu.createKernel()');
            const values = makeScores(ctx.utils, 4096);
            const out = ctx.kernel(values);
            ctx.assert(out && out.length === 4096, `expected 4096 ranks, got ${out && out.length}`);
            const used = new Array(4096).fill(0);
            let strays = 0;
            for (let i = 0; i < 4096; i++) {
              const r = Math.round(out[i]);
              if (r >= 0 && r < 4096) used[r]++;
              else strays++;
            }
            let shared = 0;
            let empty = 0;
            for (let r = 0; r < 4096; r++) {
              if (used[r] > 1) shared++;
              else if (used[r] === 0) empty++;
            }
            const broken = strays || shared || empty;
            const hint = broken
              ? diagnoseArray(out, ranksOf(values), 0.5, rankAlternatives(values))
              : null;
            ctx.assert(
              !broken,
              hint ||
                `the ranks are not a permutation of 0…4095: ${shared} claimed by more than one ` +
                  `element, ${empty} claimed by none` +
                  (strays ? `, ${strays} outside the range` : '') +
                  ' — these scores repeat, so the index tie-break is what keeps every rank unique'
            );
          },
        },
        {
          name: 'rank 0 is the largest score, and every other rank matches',
          run: async ctx => {
            // 250 distinct values over 4096 slots: every score repeats about
            // sixteen times, so the tie-break decides almost every rank.
            const values = new Array(4096);
            for (let i = 0; i < 4096; i++) values[i] = (i * 37) % 250;
            const out = ctx.kernel(values);
            const want = ranksOf(values);
            const bad = firstMismatch(out, want, 0.5);
            const hint = bad < 0 ? null : diagnoseArray(out, want, 0.5, rankAlternatives(values));
            let top = -1;
            for (let i = 0; i < 4096; i++) if (Math.round(out[i]) === 0) top = i;
            ctx.assert(
              top >= 0 && values[top] === 249,
              hint ||
                (top < 0
                  ? 'no element came back with rank 0 — the largest score has nothing ahead of it, ' +
                    'so its count is 0'
                  : `the element with rank 0 has score ${values[top]}, but the largest score in ` +
                    'this array is 249')
            );
            ctx.assert(
              bad < 0,
              hint ||
                `element ${bad} (score ${values[bad]}) came back with rank ${out[bad]}, expected ` +
                  `${want[bad]}`
            );
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            const values = makeScores(ctx.utils, 4096, 3141);
            const out = ctx.kernel(values);
            ctx.assert(out && out.length === 4096, 'expected 4096 ranks');
            const want = ranksOf(values);
            const bad = firstMismatch(out, want, 0.5);
            const hint = bad < 0 ? null : diagnoseArray(out, want, 0.5, rankAlternatives(values));
            ctx.assert(
              bad < 0,
              hint || `element ${bad} (score ${values[bad]}): expected rank ${want[bad]}, got ${out[bad]}`
            );
          },
        },
      ],
    },

    // ------------------------------------------------------------------ 2
    {
      slug: 'gather-the-winners',
      title: 'Gather the Winners',
      intro: `<p>Ranks are the answer in an unusable shape: the ten you want are scattered somewhere
        among 4,096 slots. What you want is packed — <code>top[0]</code> the biggest score,
        <code>top[9]</code> the tenth biggest.</p>
        <p>The obvious move is a <strong>scatter</strong>: element <code>i</code> writes itself into
        <code>top[ranks[i]]</code>. Kernels cannot do that — a thread writes one cell, its own. So
        turn it inside out, the way every scatter gets turned inside out. Instead of "where does my
        value go?", output slot <code>j</code> asks <strong>"who has rank <code>j</code>?"</strong>
        and goes looking. Ten threads, each scanning 4,096 ranks: a gather.</p>
        <p>Exactly one element answers each slot — which is what last task's tie-break bought you.
        (Turning a rank array into a packed result is a pattern in its own right, and the Stream
        Compaction module develops it properly, with the prefix sum that makes it O(n) instead of
        O(k·n). You do not need that here: <code>k</code> is ten.)</p>`,
      goal: `<strong>Goal:</strong> fill ten output slots with the ten largest scores, largest
        first — slot <code>j</code> holds the score of the element whose rank is <code>j</code>.`,
      requirements: [
        '<code>output: [10]</code> — one thread per result slot',
        'Scan all <code>this.constants.n</code> ranks for the one equal to <code>this.thread.x</code>',
        'Return that element\'s <em>score</em>, not its rank',
        '<code>top[0]</code> is the largest score and <code>top[9]</code> the tenth largest',
      ],
      hints: [
        {
          title: 'Hint 1 — which element is mine?',
          body: `<p>Thread <code>j</code> owns output slot <code>j</code>, and the element it wants
            is the one whose rank happens to be <code>j</code>. There is no way to know where that
            element sits, so look at all of them — a loop over the whole <code>ranks</code>
            array.</p>`,
        },
        {
          title: 'Hint 2 — the scan',
          body: `<pre><code>let best = 0;
for (let i = 0; i &lt; this.constants.n; i++) {
  if (ranks[i] === this.thread.x) best = scores[i];
}
return best;</code></pre>
<p>No <code>break</code> needed — exactly one <code>i</code> matches.</p>`,
        },
      ],
      transfer: `Gather-by-rank is the back half of every GPU sort: compute a destination for each
        element, then have each destination fetch its element — <code>thrust::gather</code>,
        <code>cub::DeviceRadixSort</code>'s final scatter pass, a WebGPU compute shader indexing a
        storage buffer. Production k-selection libraries (FAISS, RAFT's <code>select_k</code>) do
        exactly this once the candidates are down to a manageable few.`,
      starterCode: `// Ranks in, a packed top-10 out. Slot j goes looking for rank j.
const gpu = new GPU({ mode });

// Last task's kernel, unchanged.
const rankScores = gpu.createKernel(function (scores) {
  const mine = scores[this.thread.x];
  let ahead = 0;
  for (let j = 0; j < this.constants.n; j++) {
    const other = scores[j];
    if (j < this.thread.x) {
      if (other >= mine) ahead++;
    } else if (other > mine) {
      ahead++;
    }
  }
  return ahead;
}, { output: [4096], constants: { n: 4096 } });

const pickTop = gpu.createKernel(function (scores, ranks) {
  let best = 0;
  for (let i = 0; i < this.constants.n; i++) {
    // TODO: every slot is fetching rank 0. Slot this.thread.x wants
    // the element whose rank is this.thread.x.
    if (ranks[i] === 0) best = scores[i];
  }
  return best;
}, { output: [10], constants: { n: 4096 } });

const ranks = rankScores(scores);
const top = pickTop(scores, ranks);
console.log('top 10:', top);
`,
      solutionCode: `// Ranks in, a packed top-10 out. Slot j goes looking for rank j.
const gpu = new GPU({ mode });

// Last task's kernel, unchanged.
const rankScores = gpu.createKernel(function (scores) {
  const mine = scores[this.thread.x];
  let ahead = 0;
  for (let j = 0; j < this.constants.n; j++) {
    const other = scores[j];
    if (j < this.thread.x) {
      if (other >= mine) ahead++;
    } else if (other > mine) {
      ahead++;
    }
  }
  return ahead;
}, { output: [4096], constants: { n: 4096 } });

// The inverted scatter: slot j asks who has rank j, instead of
// element i asking where it should go.
const pickTop = gpu.createKernel(function (scores, ranks) {
  let best = 0;
  for (let i = 0; i < this.constants.n; i++) {
    if (ranks[i] === this.thread.x) best = scores[i];
  }
  return best;
}, { output: [10], constants: { n: 4096 } });

const ranks = rankScores(scores);
const top = pickTop(scores, ranks);
console.log('top 10:', top);
`,
      inputs: utils => ({ scores: makeScores(utils, 4096) }),
      publicTests: [
        {
          name: 'ten scores come back, largest first',
          run: async ctx => {
            const rank = kernelWithOutput(ctx, [4096]);
            const pick = kernelWithOutput(ctx, [10]);
            ctx.assert(rank, 'no kernel with output [4096] found — the ranking pass should still be there');
            ctx.assert(pick, 'no kernel with output [10] found — the picker owns one slot per result');
            const values = makeScores(ctx.utils, 4096);
            const out = pick(values, rank(values));
            ctx.assert(out && out.length === 10, `expected 10 values, got ${out && out.length}`);
            const want = topValues(values, 10);
            // "slot 0 came back 0" is ambiguous on its own — returning the RANK
            // also puts a 0 there — so each of those two only speaks once the
            // whole output has ruled the other out.
            const slots = Array.from({ length: 10 }, (unused, j) => j);
            let hint = null;
            if (matchesEvery(out, slots, 0.5)) {
              hint = 'you returned the rank instead of the score — the scan finds the element, then hands back scores[i]';
            } else if (out.every(v => Math.abs(v) <= 0.5) && want[0] > 0.5) {
              hint = 'no slot found an owner: no element reported a rank equal to this.thread.x, so every scan fell through to its starting value';
            } else {
              hint = diagnose(out[0], want[0], 0.5, [
                [want[9], 'that is the TENTH largest — the list came back upside down. Rank 0 is the largest, so slot 0 holds the maximum'],
              ]);
            }
            ctx.assertClose(out[0], want[0], 0.5, hint || 'the largest score belongs in slot 0');
          },
        },
        {
          name: 'the packed list matches the ten largest scores in order',
          run: async ctx => {
            const rank = kernelWithOutput(ctx, [4096]);
            const pick = kernelWithOutput(ctx, [10]);
            ctx.assert(rank && pick, 'expected a ranking kernel and a 10-slot picker');
            const values = makeScores(ctx.utils, 4096);
            const ranks = rank(values);
            const out = pick(values, ranks);
            const want = topValues(values, 10);
            const bad = firstMismatch(out, want, 0.5);
            const hint = bad < 0 ? null : diagnoseArray(out, want, 0.5, [
              [[...want].reverse(),
                'these are the right ten scores, upside down — rank 0 is the largest, so slot 0 holds the maximum'],
              [topValues(values.map(v => -v), 10).map(v => -v),
                'that is the BOTTOM ten, climbing — the ranking pass is counting the scores you beat instead of the scores that beat you'],
              [Array.from({ length: 10 }, (unused, j) => j),
                'you returned the rank instead of the score — the scan finds the element, then hands back scores[i]'],
            ]);
            ctx.assert(
              bad < 0,
              hint ||
                `slot ${bad} holds ${out[bad]}, expected ${want[bad]} (the top ten here are ` +
                  `${want.join(', ')})`
            );
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            const rank = kernelWithOutput(ctx, [4096]);
            const pick = kernelWithOutput(ctx, [10]);
            ctx.assert(rank && pick, 'expected a ranking kernel and a 10-slot picker');
            const values = makeScores(ctx.utils, 4096, 8081);
            const out = pick(values, rank(values));
            const want = topValues(values, 10);
            const bad = firstMismatch(out, want, 0.5);
            const hint = bad < 0 ? null : diagnoseArray(out, want, 0.5, [
              [[...want].reverse(),
                'these are the right ten scores, upside down — rank 0 is the largest, so slot 0 holds the maximum'],
              [topValues(values.map(v => -v), 10).map(v => -v),
                'that is the BOTTOM ten, climbing — the ranking pass is counting the scores you beat instead of the scores that beat you'],
            ]);
            ctx.assert(bad < 0, hint || `slot ${bad} holds ${out[bad]}, expected ${want[bad]}`);
          },
        },
      ],
    },

    // ------------------------------------------------------------------ 3
    {
      slug: 'brightest-pixels',
      title: 'The Brightest Pixels',
      intro: `<p>The same two moves, one dimension up. <code>brightness</code> is a 64×64 grid — a
        sensor frame, a heat map, a saliency map — and the question is which eight cells are
        brightest. Every thread now scans the grid with two loops instead of one, and "earlier in
        the array" means earlier in <strong>row-major order</strong>: the flat index of cell
        <code>[y][x]</code> is <code>y * 64 + x</code>. (Mind the inversion that catches everyone:
        the launch shape is given width-first, <code>output: [64, 64]</code>, but indexing runs
        row-first, <code>grid[this.thread.y][this.thread.x]</code>.)</p>
        <p>The other change is what comes back. The eight brightest <em>values</em> are rarely what
        anyone wants — you want to know <em>where</em> they are. So the picker returns the flat
        index rather than the value, and JavaScript decodes it: <code>y = Math.floor(idx / 64)</code>,
        <code>x = idx % 64</code>. Carry the index and you can always look the value back up; carry
        the value and the location is gone for good.</p>`,
      goal: `<strong>Goal:</strong> rank all 4,096 cells, then return the <strong>flat indices</strong>
        of the eight brightest, brightest first.`,
      requirements: [
        'The ranking kernel is 2D — <code>output: [64, 64]</code>, two loops over the whole grid',
        'Tie-break on the flat index <code>y * 64 + x</code>: an earlier cell wins a tie',
        'The picker returns the flat <em>index</em> of the cell whose rank is <code>this.thread.x</code>',
        'The brightest cell\'s value, row and column are logged (already wired up)',
      ],
      hints: [
        {
          title: 'Hint 1 — the same rule, flattened',
          body: `<p>Compute your own flat index once, before the loops:</p>
<pre><code>const myIndex = this.thread.y * this.constants.size + this.thread.x;</code></pre>
<p>Then compare each visited cell's flat index against it — that is exactly the
            <code>j &lt; this.thread.x</code> test from task 1, in two dimensions.</p>`,
        },
        {
          title: 'Hint 2 — the ranking body',
          body: `<pre><code>const other = grid[y][x];
if (y * this.constants.size + x &lt; myIndex) {
  if (other &gt;= mine) ahead++;
} else if (other &gt; mine) {
  ahead++;
}</code></pre>`,
        },
        {
          title: 'Hint 3 — returning a location',
          body: `<p>Track the coordinates as you scan and combine them at the end, so nothing has to
            be pulled apart again:</p>
<pre><code>if (ranks[y][x] === this.thread.x) {
  foundY = y;
  foundX = x;
}</code></pre>
<p>then <code>return foundY * this.constants.size + foundX;</code></p>`,
        },
      ],
      transfer: `Finding the brightest few cells of a grid is the last step of a stack of real
        pipelines: keypoint detection (SIFT/ORB pick local maxima of a response map), object
        detectors ranking anchor boxes before non-max suppression, astronomy source extraction.
        They all rank on the device and hand back <em>indices</em>, because the payload behind an
        index is usually far bigger than a float — the same reason CUDA's
        <code>cub::ArgMax</code> returns a <code>KeyValuePair</code> rather than a value.`,
      starterCode: `// Top-8 over a grid — and what comes back is WHERE, not what.
const gpu = new GPU({ mode });

const rankCells = gpu.createKernel(function (grid) {
  const mine = grid[this.thread.y][this.thread.x];
  const myIndex = this.thread.y * this.constants.size + this.thread.x;
  let ahead = 0;
  for (let y = 0; y < this.constants.size; y++) {
    for (let x = 0; x < this.constants.size; x++) {
      // TODO: this counts every cell. Count grid[y][x] only when it
      // outranks mine — brighter anywhere, or equally bright at a
      // lower flat index than myIndex.
      ahead++;
    }
  }
  return ahead;
}, { output: [64, 64], constants: { size: 64 } });

const pickBrightest = gpu.createKernel(function (ranks) {
  let foundY = 0;
  let foundX = 0;
  for (let y = 0; y < this.constants.size; y++) {
    for (let x = 0; x < this.constants.size; x++) {
      // TODO: every slot is fetching rank 0. Slot this.thread.x wants
      // the cell whose rank is this.thread.x.
      if (ranks[y][x] === 0) {
        foundY = y;
        foundX = x;
      }
    }
  }
  return foundY * this.constants.size + foundX;
}, { output: [8], constants: { size: 64 } });

const ranks = rankCells(brightness);
const spots = pickBrightest(ranks);

// Flat index back to a location — this part is plain JavaScript.
const row = Math.floor(spots[0] / 64);
const col = spots[0] % 64;
console.log('brightest:', brightness[row][col], 'at row', row, 'col', col);
console.log('all eight (flat indices):', spots);
`,
      solutionCode: `// Top-8 over a grid — and what comes back is WHERE, not what.
const gpu = new GPU({ mode });

const rankCells = gpu.createKernel(function (grid) {
  const mine = grid[this.thread.y][this.thread.x];
  const myIndex = this.thread.y * this.constants.size + this.thread.x;
  let ahead = 0;
  for (let y = 0; y < this.constants.size; y++) {
    for (let x = 0; x < this.constants.size; x++) {
      const other = grid[y][x];
      if (y * this.constants.size + x < myIndex) {
        if (other >= mine) ahead++;
      } else if (other > mine) {
        ahead++;
      }
    }
  }
  return ahead;
}, { output: [64, 64], constants: { size: 64 } });

const pickBrightest = gpu.createKernel(function (ranks) {
  let foundY = 0;
  let foundX = 0;
  for (let y = 0; y < this.constants.size; y++) {
    for (let x = 0; x < this.constants.size; x++) {
      if (ranks[y][x] === this.thread.x) {
        foundY = y;
        foundX = x;
      }
    }
  }
  return foundY * this.constants.size + foundX;
}, { output: [8], constants: { size: 64 } });

const ranks = rankCells(brightness);
const spots = pickBrightest(ranks);

// Flat index back to a location — this part is plain JavaScript.
const row = Math.floor(spots[0] / 64);
const col = spots[0] % 64;
console.log('brightest:', brightness[row][col], 'at row', row, 'col', col);
console.log('all eight (flat indices):', spots);
`,
      inputs: utils => ({ brightness: makeBrightness(utils, 64) }),
      publicTests: [
        {
          name: 'the 64×64 ranks are a permutation of 0…4095',
          run: async ctx => {
            const rank = kernelWithOutput(ctx, [64, 64]);
            ctx.assert(rank, 'no kernel with output [64, 64] found — the ranking pass is 2D');
            const grid = makeBrightness(ctx.utils, 64);
            const out = rank(grid);
            ctx.assert(out && out.length === 64 && out[0] && out[0].length === 64,
              'expected a 64×64 grid of ranks');
            const flat = flatten2D(out);
            const used = new Array(4096).fill(0);
            let strays = 0;
            for (let i = 0; i < 4096; i++) {
              const r = Math.round(flat[i]);
              if (r >= 0 && r < 4096) used[r]++;
              else strays++;
            }
            let shared = 0;
            let empty = 0;
            for (let r = 0; r < 4096; r++) {
              if (used[r] > 1) shared++;
              else if (used[r] === 0) empty++;
            }
            const values = flatten2D(grid);
            const broken = strays || shared || empty;
            const hint = broken
              ? diagnoseArray(flat, ranksOf(values), 0.5, rankAlternatives(values))
              : null;
            ctx.assert(
              !broken,
              hint ||
                `the ranks are not a permutation of 0…4095: ${shared} claimed by more than one ` +
                  `cell, ${empty} claimed by none` +
                  (strays ? `, ${strays} outside the range` : '') +
                  ' — every cell has to be ranked by the value it owns AND by its own flat index; ' +
                  'reading a value from one cell while tie-breaking with another cell\'s index is ' +
                  'not an ordering at all'
            );
          },
        },
        {
          name: 'the eight flat indices point at the eight brightest cells',
          run: async ctx => {
            const rank = kernelWithOutput(ctx, [64, 64]);
            const pick = kernelWithOutput(ctx, [8]);
            ctx.assert(rank && pick, 'expected a [64, 64] ranking kernel and an [8] picker');
            const grid = makeBrightness(ctx.utils, 64);
            const out = pick(rank(grid));
            ctx.assert(out && out.length === 8, `expected 8 indices, got ${out && out.length}`);
            const want = topIndices(grid, 8);
            const bad = firstMismatch(out, want, 0.5);
            const hint = bad < 0 ? null : diagnoseArray(out, want, 0.5, [
              [[...want].reverse(),
                'the right eight cells, in reverse — rank 0 is the brightest, so slot 0 holds it'],
              [want.map(idx => swapFlat(idx, 64)),
                'the right eight cells with their coordinates swapped — the flat index runs rows ' +
                'first, foundY * 64 + foundX'],
              [topIndices(grid.map(row => row.map(v => -v)), 8),
                'that is the DARKEST eight — the ranking pass is counting the cells you outshine ' +
                'instead of the cells that outshine you'],
              [want.map(idx => grid[Math.floor(idx / 64)][idx % 64]),
                'those are brightness values, not locations — return the flat index ' +
                'foundY * 64 + foundX instead'],
            ]);
            ctx.assert(
              bad < 0,
              hint ||
                `slot ${bad} holds index ${out[bad]}, expected ${want[bad]} (row ` +
                  `${Math.floor(want[bad] / 64)}, column ${want[bad] % 64})`
            );
          },
        },
        {
          name: 'the brightest cell is reported with its row and column',
          run: async ctx => {
            const grid = makeBrightness(ctx.utils, 64);
            const best = topIndices(grid, 1)[0];
            const row = Math.floor(best / 64);
            const col = best % 64;
            const text = lineWith(ctx.logs, 'brightest');
            ctx.assert(text, 'nothing was logged about the brightest cell — keep the console.log');
            const numbers = (text.match(/-?\d+(?:\.\d+)?/g) || []).map(parseFloat);
            ctx.assert(
              numbers.indexOf(grid[row][col]) !== -1,
              `the brightest cell holds ${grid[row][col]}, which is not in "${text}"`
            );
            ctx.assert(
              numbers.indexOf(row) !== -1 && numbers.indexOf(col) !== -1,
              `the brightest cell sits at row ${row}, column ${col} — "${text}" says otherwise`
            );
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            const rank = kernelWithOutput(ctx, [64, 64]);
            const pick = kernelWithOutput(ctx, [8]);
            ctx.assert(rank && pick, 'expected a [64, 64] ranking kernel and an [8] picker');
            const grid = makeBrightness(ctx.utils, 64, 4242);
            const out = pick(rank(grid));
            const want = topIndices(grid, 8);
            const bad = firstMismatch(out, want, 0.5);
            const hint = bad < 0 ? null : diagnoseArray(out, want, 0.5, [
              [want.map(idx => swapFlat(idx, 64)),
                'the right eight cells with their coordinates swapped — the flat index runs rows ' +
                'first, foundY * 64 + foundX'],
              [[...want].reverse(),
                'the right eight cells, in reverse — rank 0 is the brightest'],
            ]);
            ctx.assert(bad < 0, hint || `slot ${bad} holds index ${out[bad]}, expected ${want[bad]}`);
          },
        },
      ],
    },

    // ------------------------------------------------------------------ 4
    {
      slug: 'threshold-bisection',
      title: 'Find the Cutoff Instead',
      intro: `<p>O(n²) is fine at four thousand and hopeless at four million — ranking a million
        scores against each other is 10¹² comparisons. Production top-k does something else
        entirely: it goes looking for a <strong>threshold</strong>. Find a value <code>t</code> that
        exactly <code>k</code> scores exceed, and the top <code>k</code> is simply "everything above
        <code>t</code>". Counting how many scores clear a given <code>t</code> is <em>one linear
        pass</em>, and the whole problem collapses into a handful of them.</p>
        <p>Finding <code>t</code> is a <strong>bisection on the value axis</strong>. Bracket it:
        below <code>lo</code> at least <code>k</code> scores pass, above <code>hi</code> fewer than
        <code>k</code> do. Guess the middle, count, and throw away the half that cannot contain the
        answer. Eighteen halvings later the bracket is narrower than the gap between two whole
        numbers, and <code>Math.floor(lo)</code> is the cutoff. Each count is 65,536 elements shared
        across 256 threads — the same strided walk a reduction uses, where neighbouring threads read
        neighbouring elements.</p>
        <p>One condition, and it is a real one: the <code>k</code>-th and (<code>k</code>+1)-th
        scores must <em>differ</em>. If they are equal — which is exactly what task 1's data looked
        like, where the 10th and 11th scores were both 993 — then no threshold on earth separates
        them and you are back to the index tie-break. These scores are finer-grained on
        purpose.</p>`,
      goal: `<strong>Goal:</strong> count in parallel, bisect in JavaScript, and log the cutoff that
        exactly 10 of the 65,536 scores clear.`,
      requirements: [
        'The kernel counts a strided slice: element <code>i</code> of thread <code>x</code> is <code>values[i * 256 + x]</code>, and it is counted when it is <em>strictly above</em> <code>t</code>',
        'Total the 256 partial counts in plain JavaScript',
        'Bisect: <code>count &gt;= k</code> raises <code>lo</code> to <code>mid</code>, otherwise <code>hi</code> comes down to it',
        'Stop once <code>hi - lo</code> is 0.5 or less, then log <code>Math.floor(lo)</code> and how many scores clear it',
      ],
      hints: [
        {
          title: 'Hint 1 — the counting pass',
          body: `<p>It is a strided partial sum with a comparison in front of it:</p>
<pre><code>if (values[i * this.constants.threads + this.thread.x] &gt; t) hits++;</code></pre>
<p>Thread <code>x</code> walks <code>values[x]</code>, <code>values[x + 256]</code>,
            <code>values[x + 512]</code>, … so neighbouring threads touch neighbouring elements at
            every step.</p>`,
        },
        {
          title: 'Hint 2 — which half survives',
          body: `<p>Keep the invariant in your head: <em>at least <code>k</code> scores are above
            <code>lo</code>, fewer than <code>k</code> are above <code>hi</code></em>. So if the
            middle still lets <code>k</code> or more through, the cutoff is at or above it — raise
            <code>lo</code>. If it lets fewer through, the middle is too high — lower
            <code>hi</code>. Note the <code>&gt;=</code>: with <code>&gt;</code> the bracket keeps a
            value that <code>k + 1</code> scores clear.</p>`,
        },
        {
          title: 'Hint 3 — the whole driver',
          body: `<pre><code>while (hi - lo &gt; 0.5) {
  const mid = (lo + hi) / 2;
  if (total(countAbove(scores, mid)) &gt;= K) lo = mid;
  else hi = mid;
}
const cutoff = Math.floor(lo);</code></pre>
<p>The scores are whole numbers, so once the bracket is narrower than 1 there is nothing
            left to resolve.</p>`,
        },
      ],
      transfer: `Narrowing a value range with counting passes instead of sorting is what real
        device-side k-selection does: RAFT/cuML's <code>select_k</code>, FAISS's GPU k-selection and
        CUB's radix-select all count elements into buckets and recurse into the bucket that contains
        the boundary — a radix bisection rather than a binary one, but the same idea, and the same
        reason. Sorting a million things to look at ten of them is a bad trade on every platform.`,
      starterCode: `// A cutoff, not a ranking: 18 linear passes instead of 4 billion comparisons.
const gpu = new GPU({ mode });

const K = 10;

const countAbove = gpu.createKernel(function (values, t) {
  let hits = 0;
  for (let i = 0; i < this.constants.chunk; i++) {
    // TODO: count this thread's strided element when it is above t
  }
  return hits;
}, {
  output: [256],
  constants: { threads: 256, chunk: 256 },
});

function total(partials) {
  let sum = 0;
  for (let i = 0; i < partials.length; i++) sum += partials[i];
  return sum;
}

// Bracket the answer: everything is above lo, nothing is above hi.
let lo = scores[0];
let hi = scores[0];
for (let i = 1; i < scores.length; i++) {
  if (scores[i] < lo) lo = scores[i];
  if (scores[i] > hi) hi = scores[i];
}
lo = lo - 1;

// TODO: halve the bracket until it is narrower than 1, keeping the half
// that can still contain the cutoff.

const cutoff = Math.floor(lo);
console.log('cutoff:', cutoff);
console.log('above it:', total(countAbove(scores, cutoff)));
`,
      solutionCode: `// A cutoff, not a ranking: 18 linear passes instead of 4 billion comparisons.
const gpu = new GPU({ mode });

const K = 10;

const countAbove = gpu.createKernel(function (values, t) {
  let hits = 0;
  for (let i = 0; i < this.constants.chunk; i++) {
    if (values[i * this.constants.threads + this.thread.x] > t) hits++;
  }
  return hits;
}, {
  output: [256],
  constants: { threads: 256, chunk: 256 },
});

function total(partials) {
  let sum = 0;
  for (let i = 0; i < partials.length; i++) sum += partials[i];
  return sum;
}

// Bracket the answer: everything is above lo, nothing is above hi.
let lo = scores[0];
let hi = scores[0];
for (let i = 1; i < scores.length; i++) {
  if (scores[i] < lo) lo = scores[i];
  if (scores[i] > hi) hi = scores[i];
}
lo = lo - 1;

// At least K scores are above lo; fewer than K are above hi. Halve.
while (hi - lo > 0.5) {
  const mid = (lo + hi) / 2;
  if (total(countAbove(scores, mid)) >= K) lo = mid;
  else hi = mid;
}

const cutoff = Math.floor(lo);
console.log('cutoff:', cutoff);
console.log('above it:', total(countAbove(scores, cutoff)));
`,
      inputs: utils => ({ scores: makeFineScores(utils, 65536, 4801) }),
      publicTests: [
        {
          name: 'each thread counts its own strided slice',
          run: async ctx => {
            ctx.assert(ctx.kernels.length >= 1, 'no kernel was created — call gpu.createKernel()');
            // values[i] = i % 256, so thread x's strided slice is 256 copies
            // of x — and a contiguous walk would see 0…255 instead.
            const values = new Array(65536);
            for (let i = 0; i < 65536; i++) values[i] = i % 256;
            const out = ctx.kernel(values, 100);
            ctx.assert(out && out.length === 256, `expected 256 partial counts, got ${out && out.length}`);
            const want = partialCounts(values, 100, 256, 256);
            let anything = 0;
            for (let x = 0; x < 256; x++) anything += out[x];
            ctx.assert(anything > 0, 'every thread reported 0 — the counter never incremented');
            const bad = firstMismatch(out, want, 0.5);
            const contiguous = new Array(256).fill(0);
            for (let x = 0; x < 256; x++) {
              let n = 0;
              for (let i = 0; i < 256; i++) if (values[x * 256 + i] > 100) n++;
              contiguous[x] = n;
            }
            const hint = bad < 0 ? null : diagnoseArray(out, want, 0.5, [
              [contiguous,
                'that is the count over a contiguous block — the strided walk is ' +
                'values[i * this.constants.threads + this.thread.x], so neighbouring threads read ' +
                'neighbouring elements'],
              [partialCounts(values, 99.5, 256, 256),
                'the comparison is >= where it should be > — the cutoff itself does not clear the cutoff'],
              [new Array(256).fill(256),
                'every element was counted — the comparison against t never happened'],
            ]);
            ctx.assert(
              bad < 0,
              hint || `thread ${bad} counted ${out[bad]} of its 256 elements above 100, expected ${want[bad]}`
            );
          },
        },
        {
          name: 'the partial counts total correctly on the real scores',
          run: async ctx => {
            const values = makeFineScores(ctx.utils, 65536, 4801);
            for (const t of [0, 5000, 40000, 68327]) {
              const got = sumOf(ctx.kernel(values, t));
              const want = howManyAbove(values, t);
              const hint = diagnose(got, want, 0.5, [
                [65536 - want,
                  'that is how many scores fall BELOW t — the comparison is the wrong way round'],
                [256, 'every thread reported its whole slice as a hit — the comparison never ran'],
              ]);
              ctx.assertClose(got, want, 0.5, hint || `scores above ${t}`);
            }
          },
        },
        {
          name: 'the logged cutoff lets exactly 10 scores through',
          run: async ctx => {
            const values = makeFineScores(ctx.utils, 65536, 4801);
            const text = lineWith(ctx.logs, 'cutoff');
            ctx.assert(text, 'no cutoff was logged — keep the console.log at the end');
            const t = numberAfter(text, 'cutoff:');
            ctx.assert(t !== null, `could not read a cutoff out of "${text}"`);
            const hint = cutoffHint(values, t, 10);
            ctx.assert(hint === null, hint);
            const above = lineWith(ctx.logs, 'above it');
            ctx.assert(above, 'the second line, reporting how many scores clear the cutoff, is missing');
            ctx.assertClose(numberAfter(above, 'above it:'), 10, 0.5,
              'the count of scores above the cutoff');
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            const values = makeFineScores(ctx.utils, 65536, 2718);
            // The count has to be right everywhere, not just near the answer:
            // a bisection asks about the whole range on its way down.
            const [low, high] = bracketOf(values);
            for (let step = 0; step <= 8; step++) {
              const t = low + ((high - low) * step) / 8;
              const got = sumOf(ctx.kernel(values, t));
              const want = howManyAbove(values, t);
              const hint = diagnose(got, want, 0.5, [
                [65536 - want, 'that is how many scores fall BELOW t — the comparison is the wrong way round'],
              ]);
              ctx.assertClose(got, want, 0.5, hint || `scores above ${t.toFixed(1)}`);
            }
          },
        },
      ],
    },

    // ------------------------------------------------------------------ 5
    {
      slug: 'which-one-wins',
      title: 'Which One Wins?',
      // Sized to show the CROSSOVER rather than one side of it, which means the
      // big ranking pass (131,072² ≈ 17.2e9 comparisons) has to actually run.
      // Two consequences, both handled here rather than by the learner:
      //
      //   • The pre-flight guard extrapolates a run from a probe with the
      //     output clamped to ~4,096 threads, so a [512, 256] grid clamps to
      //     [64, 64] and the estimate is 32× the probe. (Authoring this task
      //     is what found the guard's old per-axis clamp of 64, under which a
      //     1D [131072] extrapolated 2,048× and was refused at any budget;
      //     sandbox.js now spreads the same thread count over however many
      //     axes there are. The grid here is kept because it is honest about
      //     what a wide launch already is, not to dodge the guard.)
      //
      //     Even so the estimate is wildly pessimistic, because most of what
      //     the probe times (four kernel compiles and 72 counting launches,
      //     each shipping the array to the GPU) does not scale with the clamp
      //     at all. Measured in the worker sandbox on an M1 Max: probe
      //     174-226 ms → an estimate of 6-7 s, against a real run of ~0.5 s.
      //     Hence 20000 rather than something near the truth: it keeps a GPU
      //     about three times slower than this one out of a refusal it does
      //     not deserve. It is a ceiling, not a cost — nothing waits for it.
      //     The run watchdog then allows twice the budget (see runner.js).
      //   • 17.2 billion comparisons single-threaded is about a minute, so the
      //     BIG ranking pass is measured on the gpu backend only. Everything
      //     else — both bisections and the 4,096 ranking pass — runs on both,
      //     which keeps `verify-learn --mode cpu` at ~0.4 s for this task and
      //     still lets the CPU backend make the same point (there the ranking
      //     pass already loses at 4,096). No size does both: the CPU backend
      //     costs ~3 ns per comparison, so anything large enough for the GPU
      //     crossover (~65,000 here) is already 13 s of JavaScript.
      //
      //     One knock-on effect, called out in the intro rather than hidden:
      //     ⏱ Benchmark runs the file once per backend and times the kernels
      //     each run invoked, so the cpu side is missing the big ranking pass
      //     and the chip reports ~1×. The intro says why, and uses it.
      budgetMs: 20000,
      intro: `<p>Two formulations, one answer, and a price that depends on <code>n</code>. Both are
        wired up below, both report the same thing so the comparison is honest — the score at the
        boundary — and both run <strong>twice</strong>: once on 4,096 scores, once on 131,072.
        Rank-by-counting reads 4,096² ≈ 16.8 <strong>million</strong> values at the small size and
        17.2 <strong>billion</strong> at the large one. The bisection reads 4,096 values eighteen
        times (about 74,000) and 131,072 eighteen times (about 2.4 million). Between the two sizes
        one of those grows 1,024×, the other 32×.</p>
        <p>So run it, and watch the winner change. At 4,096 the ranking pass <em>wins</em>, despite
        doing two hundred times the arithmetic: it is one dense, embarrassingly parallel launch,
        which is precisely what the hardware is for, while the bisection spends its life waiting for
        eighteen tiny kernels to come back — latency, not arithmetic. At 131,072 the arithmetic
        finally outgrows the latency and the order flips. On the machine this was written on (an M1
        Max) the small size measures about <strong>1 ms</strong> for the ranking against
        <strong>10 ms</strong> for the bisection, and the large one about <strong>48 ms</strong>
        against <strong>15 ms</strong>: 32× the data costs the bisection five milliseconds, because
        eighteen round trips are eighteen round trips whatever they carry, and costs the ranking
        pass everything. The crossover sits near 65,000, where the two trade places from run to run
        — and it will not sit there on your hardware, which is the point.</p>
        <p>Every kernel gets one untimed warm-up call before the clock starts: a kernel's first
        launch compiles it, and a shader compiler inside the timer measures nothing you asked about.
        Even so, one <code>performance.now()</code> sample is a shape, not a benchmark. Read the
        four lines, then read them again with <strong>Mode</strong> switched from Auto to CPU —
        there the ranking pass loses at 4,096 already (about 50 ms against 0.2 ms), and at 131,072
        it is not run at all, because a minute of single-threaded counting is the same lesson in a
        harsher form.</p>
        <p><strong>⏱ Benchmark</strong> answers a different question, and on this task it answers it
        badly — which is worth seeing once. It runs the whole file twice, once per backend, and on
        the CPU backend the file skips the big ranking pass; so it times a smaller job on one side
        and reports something near <strong>1×</strong>. That number means "the CPU backend got out
        of the work", not "the GPU is not helping" — timing two things that are not the same thing
        is the oldest way to get a benchmark wrong, and it is exactly what the four lines above go
        out of their way to avoid.</p>`,
      goal: `<strong>Goal:</strong> write one <code>bisect()</code> driver and one
        <code>boundaryOf()</code> scan, use them at both sizes, and read off the four timings.`,
      requirements: [
        'One <code>bisect(counter, values, k)</code> serving both the 4,096 and the 131,072 case',
        'It brackets from the data, halves while <code>hi - lo &gt; 0.5</code>, and returns <code>Math.floor(lo)</code>',
        'One <code>boundaryOf(ranks, values)</code> for the ranking side: the score of the element whose rank is <code>K - 1</code>, over flat ranks so the same scan serves the grid',
        'Every measurement the backend can afford runs and logs a time — four on the GPU, three on the CPU, where the 131,072-score ranking pass is reported instead of run (already written)',
      ],
      hints: [
        {
          title: 'Hint 1 — one driver, two counters',
          body: `<p><code>bisect</code> never mentions a size: it takes the counting kernel as an
            argument and gets its bracket from the values it was handed. That is why the same four
            lines serve 4,096 scores and 131,072.</p>`,
        },
        {
          title: 'Hint 2 — the boundary from ranks',
          body: `<p>The <code>K</code>-th largest score is the one whose rank is <code>K - 1</code>.
            One plain loop over the ranks finds it:</p>
<pre><code>for (let i = 0; i &lt; ranks.length; i++) {
  if (ranks[i] === K - 1) cut = values[i];
}</code></pre>
<p>The big ranking pass hands back a 512 × 256 grid, so the driver flattens it first
            (<code>utils.flatten</code>) — flat rank <code>i</code> then belongs to
            <code>values[i]</code> in both cases.</p>`,
        },
        {
          title: 'Hint 3 — reading the numbers',
          body: `<p>The two cutoffs at a size will not be the same number, and they should not be:
            the ranking pass reports the 10th largest <em>score</em>, the bisection reports the
            largest whole number strictly below it. Both describe the same boundary — exactly ten
            scores are above the bisection's cutoff, and the tenth of them is the ranking pass's
            answer.</p>
<p>Then compare the two <em>times</em> at 4,096 against the two at 131,072. The
            bisection barely notices the 32× more data; the ranking pass notices it 1,024 times
            over.</p>`,
        },
      ],
      transfer: `Picking a formulation by measurement rather than by asymptotics is the whole job.
        CUB ships several k-selection strategies and dispatches on size; cuDNN and cuBLAS carry
        multiple kernels per operation and choose at runtime; PyTorch's <code>topk</code> switches
        between a sorting path and a radix-select path on <code>k</code> and <code>n</code>. The
        crossovers are found the way you just found this one — by running both on both sides of it,
        warm, and reading the clock.`,
      starterCode: `// Same question, two formulations, two sizes. Time them, then ⏱ Benchmark.
const gpu = new GPU({ mode });

const K = 10;

// --- approach A: rank everything (tasks 1-2), at both sizes
const rankSmall = gpu.createKernel(function (values) {
  const mine = values[this.thread.x];
  let ahead = 0;
  for (let j = 0; j < this.constants.n; j++) {
    const other = values[j];
    if (j < this.thread.x) {
      if (other >= mine) ahead++;
    } else if (other > mine) {
      ahead++;
    }
  }
  return ahead;
}, { output: [4096], constants: { n: 4096 } });

// The same pass on 131,072 scores. A launch that wide is a 2D texture
// underneath whatever you call it, so this one says so — a 512 x 256
// grid, ranked by the flat index y * 512 + x, exactly like task 3.
const rankBig = gpu.createKernel(function (values) {
  const me = this.thread.y * this.constants.width + this.thread.x;
  const mine = values[me];
  let ahead = 0;
  for (let j = 0; j < this.constants.n; j++) {
    const other = values[j];
    if (j < me) {
      if (other >= mine) ahead++;
    } else if (other > mine) {
      ahead++;
    }
  }
  return ahead;
}, { output: [512, 256], constants: { n: 131072, width: 512 } });

// --- approach B: bisect for a cutoff (task 4), at both sizes
const countSmall = gpu.createKernel(function (values, t) {
  let hits = 0;
  for (let i = 0; i < this.constants.chunk; i++) {
    if (values[i * this.constants.threads + this.thread.x] > t) hits++;
  }
  return hits;
}, { output: [64], constants: { threads: 64, chunk: 64 } });

const countBig = gpu.createKernel(function (values, t) {
  let hits = 0;
  for (let i = 0; i < this.constants.chunk; i++) {
    if (values[i * this.constants.threads + this.thread.x] > t) hits++;
  }
  return hits;
}, { output: [512], constants: { threads: 512, chunk: 256 } });

function total(partials) {
  let sum = 0;
  for (let i = 0; i < partials.length; i++) sum += partials[i];
  return sum;
}

function bracket(values) {
  let lo = values[0];
  let hi = values[0];
  for (let i = 1; i < values.length; i++) {
    if (values[i] < lo) lo = values[i];
    if (values[i] > hi) hi = values[i];
  }
  return [lo - 1, hi];
}

function bisect(counter, values, k) {
  // TODO: bracket the values, halve while hi - lo > 0.5 keeping the half
  // that can still contain the cutoff, and return Math.floor(lo).
  return 0;
}

function boundaryOf(ranks, values) {
  // TODO: the K-th largest score is the one whose rank is K - 1. Ranks
  // arrive flat, so this same scan serves both sizes.
  return 0;
}

// A kernel's first launch compiles it, and a shader compiler inside the
// timer is not a measurement. One untimed warm-up call each.
rankSmall(scores);
bisect(countSmall, scores, K);
if (mode === 'gpu') rankBig(bigScores);
bisect(countBig, bigScores, K);

let t0 = performance.now();
const smallByRank = boundaryOf(rankSmall(scores), scores);
const rankSmallMs = performance.now() - t0;

t0 = performance.now();
const smallCut = bisect(countSmall, scores, K);
const bisectSmallMs = performance.now() - t0;

console.log('rank 4096:', rankSmallMs.toFixed(1), 'ms - 10th largest score is', smallByRank);
console.log('bisect 4096:', bisectSmallMs.toFixed(1), 'ms - cutoff', smallCut);

// 131,072 x 131,072 = 17.2 billion comparisons. The GPU eats them in tens
// of milliseconds; the CPU backend, one thread, would need about a minute,
// so there this measurement is reported rather than run.
if (mode === 'gpu') {
  t0 = performance.now();
  const bigByRank = boundaryOf(utils.flatten(rankBig(bigScores)), bigScores);
  const rankBigMs = performance.now() - t0;
  console.log('rank 131072:', rankBigMs.toFixed(1), 'ms - 10th largest score is', bigByRank);
} else {
  console.log('rank 131072: not run on the cpu backend - 17.2 billion comparisons, about a minute');
}

t0 = performance.now();
const bigCut = bisect(countBig, bigScores, K);
const bisectBigMs = performance.now() - t0;

console.log('bisect 131072:', bisectBigMs.toFixed(1), 'ms - cutoff', bigCut);
`,
      solutionCode: `// Same question, two formulations, two sizes. Time them, then ⏱ Benchmark.
const gpu = new GPU({ mode });

const K = 10;

// --- approach A: rank everything (tasks 1-2), at both sizes
const rankSmall = gpu.createKernel(function (values) {
  const mine = values[this.thread.x];
  let ahead = 0;
  for (let j = 0; j < this.constants.n; j++) {
    const other = values[j];
    if (j < this.thread.x) {
      if (other >= mine) ahead++;
    } else if (other > mine) {
      ahead++;
    }
  }
  return ahead;
}, { output: [4096], constants: { n: 4096 } });

// The same pass on 131,072 scores. A launch that wide is a 2D texture
// underneath whatever you call it, so this one says so — a 512 x 256
// grid, ranked by the flat index y * 512 + x, exactly like task 3.
const rankBig = gpu.createKernel(function (values) {
  const me = this.thread.y * this.constants.width + this.thread.x;
  const mine = values[me];
  let ahead = 0;
  for (let j = 0; j < this.constants.n; j++) {
    const other = values[j];
    if (j < me) {
      if (other >= mine) ahead++;
    } else if (other > mine) {
      ahead++;
    }
  }
  return ahead;
}, { output: [512, 256], constants: { n: 131072, width: 512 } });

// --- approach B: bisect for a cutoff (task 4), at both sizes
const countSmall = gpu.createKernel(function (values, t) {
  let hits = 0;
  for (let i = 0; i < this.constants.chunk; i++) {
    if (values[i * this.constants.threads + this.thread.x] > t) hits++;
  }
  return hits;
}, { output: [64], constants: { threads: 64, chunk: 64 } });

const countBig = gpu.createKernel(function (values, t) {
  let hits = 0;
  for (let i = 0; i < this.constants.chunk; i++) {
    if (values[i * this.constants.threads + this.thread.x] > t) hits++;
  }
  return hits;
}, { output: [512], constants: { threads: 512, chunk: 256 } });

function total(partials) {
  let sum = 0;
  for (let i = 0; i < partials.length; i++) sum += partials[i];
  return sum;
}

function bracket(values) {
  let lo = values[0];
  let hi = values[0];
  for (let i = 1; i < values.length; i++) {
    if (values[i] < lo) lo = values[i];
    if (values[i] > hi) hi = values[i];
  }
  return [lo - 1, hi];
}

// Size-free: the counting kernel is an argument, the bracket comes from
// the data. The same four lines serve 4,096 scores and 131,072.
function bisect(counter, values, k) {
  let [lo, hi] = bracket(values);
  while (hi - lo > 0.5) {
    const mid = (lo + hi) / 2;
    if (total(counter(values, mid)) >= k) lo = mid;
    else hi = mid;
  }
  return Math.floor(lo);
}

// Also size-free: flat ranks in, the K-th largest score out.
function boundaryOf(ranks, values) {
  let cut = 0;
  for (let i = 0; i < ranks.length; i++) {
    if (ranks[i] === K - 1) cut = values[i];
  }
  return cut;
}

// A kernel's first launch compiles it, and a shader compiler inside the
// timer is not a measurement. One untimed warm-up call each.
rankSmall(scores);
bisect(countSmall, scores, K);
if (mode === 'gpu') rankBig(bigScores);
bisect(countBig, bigScores, K);

let t0 = performance.now();
const smallByRank = boundaryOf(rankSmall(scores), scores);
const rankSmallMs = performance.now() - t0;

t0 = performance.now();
const smallCut = bisect(countSmall, scores, K);
const bisectSmallMs = performance.now() - t0;

console.log('rank 4096:', rankSmallMs.toFixed(1), 'ms - 10th largest score is', smallByRank);
console.log('bisect 4096:', bisectSmallMs.toFixed(1), 'ms - cutoff', smallCut);

// 131,072 x 131,072 = 17.2 billion comparisons. The GPU eats them in tens
// of milliseconds; the CPU backend, one thread, would need about a minute,
// so there this measurement is reported rather than run.
if (mode === 'gpu') {
  t0 = performance.now();
  const bigByRank = boundaryOf(utils.flatten(rankBig(bigScores)), bigScores);
  const rankBigMs = performance.now() - t0;
  console.log('rank 131072:', rankBigMs.toFixed(1), 'ms - 10th largest score is', bigByRank);
} else {
  console.log('rank 131072: not run on the cpu backend - 17.2 billion comparisons, about a minute');
}

t0 = performance.now();
const bigCut = bisect(countBig, bigScores, K);
const bisectBigMs = performance.now() - t0;

console.log('bisect 131072:', bisectBigMs.toFixed(1), 'ms - cutoff', bigCut);
`,
      inputs: utils => ({
        scores: makeFineScores(utils, 4096, 606),
        bigScores: makeFineScores(utils, 131072, 4801),
      }),
      publicTests: [
        {
          name: 'all four kernels still work: two rankers, two counters',
          run: async ctx => {
            const rankSmall = kernelWithOutput(ctx, [4096]);
            const rankBig = kernelWithOutput(ctx, [512, 256]);
            const small = kernelWithOutput(ctx, [64]);
            const big = kernelWithOutput(ctx, [512]);
            ctx.assert(rankSmall, 'no kernel with output [4096] found — the 4,096-score ranking pass');
            ctx.assert(rankBig, 'no kernel with output [512, 256] found — the 131,072-score ranking pass');
            ctx.assert(small, 'no kernel with output [64] found — the 4,096-score counter');
            ctx.assert(big, 'no kernel with output [512] found — the 131,072-score counter');
            const values = makeFineScores(ctx.utils, 4096, 606);
            const gotRanks = rankSmall(values);
            const wantRanks = ranksOf(values);
            const badRank = firstMismatch(gotRanks, wantRanks, 0.5);
            ctx.assert(badRank < 0, `the 4,096-score ranking pass is wrong at element ${badRank}`);
            ctx.assertClose(
              sumOf(small(values, 40000)),
              howManyAbove(values, 40000),
              0.5,
              'the 4,096-score counter'
            );
            // The big RANKING pass is 17.2 billion comparisons — invoked here
            // it would add a minute to a cpu-mode check, so its correctness is
            // read off the run's own log line instead (see the boundary test,
            // which only asks for it on the gpu backend). The big COUNTER is
            // one linear pass, so it is exercised directly on both.
            const bigValues = makeFineScores(ctx.utils, 131072, 4801);
            ctx.assertClose(
              sumOf(big(bigValues, 40000)),
              howManyAbove(bigValues, 40000),
              0.5,
              'the 131,072-score counter'
            );
          },
        },
        {
          name: 'every measurement this backend can afford reports a time',
          run: async ctx => {
            const timed = ['rank 4096', 'bisect 4096', 'bisect 131072'];
            if (ctx.resolvedMode === 'gpu') timed.push('rank 131072');
            for (const label of timed) {
              const text = lineWith(ctx.logs, label);
              ctx.assert(text, `nothing was logged for "${label}" — that measurement has to run`);
              const ms = millisecondsOn(text);
              ctx.assert(
                ms !== null && Number.isFinite(ms) && ms >= 0,
                `"${text}" does not report a duration in ms`
              );
            }
            if (ctx.resolvedMode !== 'gpu') {
              // Skipped, but not silently: the line still has to say so.
              ctx.assert(
                lineWith(ctx.logs, 'rank 131072'),
                'nothing was logged for "rank 131072" — on the cpu backend it reports why it did ' +
                  'not run, which is still a line'
              );
            }
          },
        },
        {
          name: 'both formulations agree about the boundary, at both sizes',
          run: async ctx => {
            const values = makeFineScores(ctx.utils, 4096, 606);
            const bigValues = makeFineScores(ctx.utils, 131072, 4801);
            const tenth = topValues(values, 10)[9];

            const rankLine = lineWith(ctx.logs, 'rank 4096');
            const byRank = numberAfter(rankLine, 'score is');
            ctx.assert(byRank !== null, `could not read a score out of "${rankLine}"`);
            const rankHint = diagnose(byRank, tenth, 0.5, [
              [topValues(values, 11)[10], 'that is the ELEVENTH largest — rank K − 1 is the K-th largest, because rank 0 is the first'],
              [topValues(values, 9)[8], 'that is the NINTH largest — rank K − 1, not rank K − 2'],
              [topValues(values, 1)[0], 'that is the largest score — you want the element whose rank is K − 1, not 0'],
              [0, 'no element reported rank K − 1, so the search fell through — ranks come back as floats, compare them to K - 1'],
            ]);
            ctx.assertClose(byRank, tenth, 0.5, rankHint || 'the 10th largest of the 4096 scores');

            const smallLine = lineWith(ctx.logs, 'bisect 4096');
            ctx.assert(smallLine, 'nothing was logged for "bisect 4096"');
            const smallCut = numberAfter(smallLine, 'cutoff');
            ctx.assert(smallCut !== null, `could not read a cutoff out of "${smallLine}"`);
            const smallHint = cutoffHint(values, smallCut, 10);
            ctx.assert(smallHint === null, smallHint);

            const bigLine = lineWith(ctx.logs, 'bisect 131072');
            ctx.assert(bigLine, 'nothing was logged for "bisect 131072"');
            const bigCut = numberAfter(bigLine, 'cutoff');
            ctx.assert(bigCut !== null, `could not read a cutoff out of "${bigLine}"`);
            const bigHint = cutoffHint(bigValues, bigCut, 10);
            ctx.assert(bigHint === null, bigHint);

            // The big ranking pass runs on the gpu backend only, so only there
            // is there a boundary of its own to agree with the big cutoff.
            if (ctx.resolvedMode === 'gpu') {
              const bigRankLine = lineWith(ctx.logs, 'rank 131072');
              const bigByRank = numberAfter(bigRankLine, 'score is');
              ctx.assert(bigByRank !== null, `could not read a score out of "${bigRankLine}"`);
              const bigTenth = topValues(bigValues, 10)[9];
              const gridHint = diagnose(bigByRank, bigTenth, 0.5, [
                [topValues(bigValues, 11)[10], 'that is the ELEVENTH largest of the 131,072 — rank K − 1 is the K-th largest'],
                [tenth, 'that is the 4,096-score answer — the big pass ranks bigScores, not scores'],
                [0, 'no cell reported rank K − 1 — the grid of ranks has to be flattened before the scan, utils.flatten(rankBig(bigScores))'],
              ]);
              ctx.assertClose(
                bigByRank,
                bigTenth,
                0.5,
                gridHint || 'the 10th largest of the 131,072 scores'
              );
              ctx.assertClose(
                bigCut,
                bigTenth - 1,
                0.5,
                'the two formulations disagree at 131,072: the bisection\'s cutoff should be the ' +
                  'whole number just below the ranking pass\'s 10th largest score'
              );
            }
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            // Drive the learner's own counter through a full bisection to make
            // sure the kernel — not just the driver — holds up at every guess.
            // 18 linear passes, so this costs the same on either backend.
            const big = kernelWithOutput(ctx, [512]);
            ctx.assert(big, 'no kernel with output [512] found');
            const values = makeFineScores(ctx.utils, 131072, 1204);
            let [lo, hi] = bracketOf(values);
            while (hi - lo > 0.5) {
              const mid = (lo + hi) / 2;
              const got = sumOf(big(values, mid));
              ctx.assertClose(got, howManyAbove(values, mid), 0.5, `scores above ${mid.toFixed(1)}`);
              if (got >= 10) lo = mid;
              else hi = mid;
            }
            ctx.assertClose(
              howManyAbove(values, Math.floor(lo)),
              10,
              0.5,
              'the bisection should land on a cutoff exactly 10 scores clear'
            );
          },
        },
      ],
    },
  ],
};
