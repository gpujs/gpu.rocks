// Module: Radix Sort — uuid fd3ff796-daed-4036-9202-987b374bb4d3 (short id fd3ff796).
// The file name is the uuid; identity lives in the exported object below,
// never in the path. No legacyId: this module postdates uuids, and it belongs
// to no track (it lands in "Others", which is unordered — so it stands alone).
//
// Radix Sort — the payoff module for the parallel primitives.
//
// Six tasks: the stable one-digit rank → a binary pass and the JavaScript
// scatter it needs → widening to 16 buckets with a histogram and an exclusive
// scan → the gather that replaces the scatter → the whole three-pass sort →
// the honest coda about keys that are not non-negative integers.
//
// Kernel-authoring rules (contract): no closures inside kernel functions,
// loop bounds come from this.constants (compile-time known), digits are
// computed by division and modulo (never by bit tricks — see task 6 for why),
// and every task passes in CPU mode. n stays ≤ 1024 so the O(n) per-thread
// counting loops stay fast in both backends.

const RADIX = 16;

// ---- deterministic inputs -------------------------------------------------

// n single digits, 0–9. With n > 10 repeats are guaranteed, which is the whole
// point: stability is only observable where two elements share a digit.
function makeDigits(utils, n, seed = 1024) {
  const rand = utils.seededRandom(seed);
  const data = new Array(n);
  for (let i = 0; i < n; i++) data[i] = Math.floor(rand() * 10);
  return data;
}

// n non-negative integer keys below `limit`.
function makeKeys(utils, n, limit, seed) {
  const rand = utils.seededRandom(seed);
  const data = new Array(n);
  for (let i = 0; i < n; i++) data[i] = Math.floor(rand() * limit);
  return data;
}

// n signed readings in −2048 … 2047, never exactly 0 — a logged 0 can then
// only mean the sort ran on unbiased negative keys (see task 6's diagnosis).
function makeReadings(utils, n, seed) {
  const rand = utils.seededRandom(seed);
  const data = new Array(n);
  for (let i = 0; i < n; i++) {
    const v = Math.floor(rand() * 4096) - 2048;
    data[i] = v === 0 ? 1 : v;
  }
  return data;
}

// ---- JS reference implementation ------------------------------------------
//
// Every expectation in this module is computed from these, never hardcoded.

function digitOf(key, place) {
  return Math.floor(key / place) % RADIX;
}

// The destination of each element in a STABLE pass over `digits`: everything
// with a smaller digit, plus the equal digits that started earlier.
function stableRank(digits) {
  const n = digits.length;
  const out = new Array(n);
  for (let i = 0; i < n; i++) {
    let before = 0;
    for (let j = 0; j < n; j++) {
      if (digits[j] < digits[i]) before++;
      else if (digits[j] === digits[i] && j < i) before++;
    }
    out[i] = before;
  }
  return out;
}

// Same, but ties broken the other way — sorted by digit, and backwards within
// every run of equal digits. The signature of an unstable pass.
function reversedRank(digits) {
  const n = digits.length;
  const out = new Array(n);
  for (let i = 0; i < n; i++) {
    let before = 0;
    for (let j = 0; j < n; j++) {
      if (digits[j] < digits[i]) before++;
      else if (digits[j] === digits[i] && j > i) before++;
    }
    out[i] = before;
  }
  return out;
}

// No tie-break at all: every element with the same digit claims one slot.
function collidingRank(digits) {
  return digits.map(d => digits.filter(other => other < d).length);
}

// `j <= this.thread.x` instead of `j < this.thread.x`: you count yourself.
function selfCountingRank(digits) {
  return stableRank(digits).map(v => v + 1);
}

function lowBits(keys) {
  return keys.map(k => k % 2);
}

function zeroCount(bits) {
  return bits.filter(b => b === 0).length;
}

// The binary pass: zeros keep their order at the front, ones behind them.
function binaryDestinations(bits) {
  const zeros = zeroCount(bits);
  let z = 0;
  let o = 0;
  return bits.map(b => (b === 0 ? z++ : zeros + o++));
}

// Ones never add the zero total — both halves start from 0 and collide.
function noOffsetDestinations(bits) {
  let z = 0;
  let o = 0;
  return bits.map(b => (b === 0 ? z++ : o++));
}

// The two branches swapped: ones first, zeros pushed behind them.
function swappedDestinations(bits) {
  const zeros = zeroCount(bits);
  let z = 0;
  let o = 0;
  return bits.map(b => (b === 0 ? zeros + z++ : o++));
}

// Every matching flag counted, not just the earlier ones: each element gets
// its whole bucket's size instead of its rank inside it.
function bucketSizeDestinations(bits) {
  const zeros = zeroCount(bits);
  return bits.map(b => (b === 0 ? zeros : zeros + (bits.length - zeros)));
}

function countsOf(values, place) {
  const counts = new Array(RADIX).fill(0);
  for (let i = 0; i < values.length; i++) counts[digitOf(values[i], place)]++;
  return counts;
}

// Exclusive scan: starts[b] = how many elements have a digit smaller than b.
function startsOf(counts) {
  const starts = new Array(RADIX).fill(0);
  for (let b = 1; b < RADIX; b++) starts[b] = starts[b - 1] + counts[b - 1];
  return starts;
}

// Inclusive scan — the classic off-by-one-bucket.
function inclusiveOf(counts) {
  const out = new Array(RADIX).fill(0);
  let running = 0;
  for (let b = 0; b < RADIX; b++) {
    running += counts[b];
    out[b] = running;
  }
  return out;
}

// A histogram whose digit forgot its `% radix`: bucket b then only matches
// keys whose whole quotient is exactly b.
function unmoddedCounts(values, place) {
  const counts = new Array(RADIX).fill(0);
  for (let i = 0; i < values.length; i++) {
    const q = Math.floor(values[i] / place);
    if (q >= 0 && q < RADIX) counts[q]++;
  }
  return counts;
}

function destinationsOf(values, place, starts) {
  const seen = new Array(RADIX).fill(0);
  const out = new Array(values.length);
  for (let i = 0; i < values.length; i++) {
    const d = digitOf(values[i], place);
    out[i] = starts[d] + seen[d];
    seen[d]++;
  }
  return out;
}

// The kernel gather's exact semantics: a slot nobody claims keeps 0, and a
// destination outside the array is simply lost (there is no cell to write).
function gatherBy(values, destinations) {
  const n = values.length;
  const out = new Array(n).fill(0);
  for (let i = 0; i < n; i++) {
    const d = destinations[i];
    if (d >= 0 && d < n) out[d] = values[i];
  }
  return out;
}

// One full pass, and the driver over a list of places. `hoist` reproduces the
// mistake of computing the histogram and offsets once and reusing them.
function radixDrive(values, places, hoist) {
  let v = values.slice();
  let starts = null;
  for (const place of places) {
    if (!hoist || starts === null) starts = startsOf(countsOf(v, place));
    v = gatherBy(v, destinationsOf(v, place, starts));
  }
  return v;
}

function ascending(values) {
  return values.slice().sort((a, b) => a - b);
}

// ---- test plumbing --------------------------------------------------------

// gpu.js locks an argument's TYPE on a kernel's first invocation, and every
// kernel in this module is fed another kernel's output somewhere in the
// pipeline — so a kernel a test wants to poke may already be locked to plain
// Array or to Float32Array depending on what the learner's own run happened to
// hand it first. On the WebGL backend the mismatch is fatal (argumentMismatch),
// so try both shapes rather than reporting a type error as a wrong answer. The
// retry is harmless: the kernel recovers, on both backends.
function call(kernel, args) {
  try {
    return kernel(...args);
  } catch (e) {
    try {
      return kernel(...args.map(a => (Array.isArray(a) ? Float32Array.from(a) : a)));
    } catch (e2) {
      return kernel(...args.map(a => (ArrayBuffer.isView(a) ? Array.from(a) : a)));
    }
  }
}

function plain(result) {
  return result ? Array.from(result) : [];
}

// Kernels are found by SHAPE — argument count plus output width — never by the
// name a learner happened to give them or the order they were created in.
function findKernel(ctx, arity, width) {
  return (
    ctx.kernels.find(k => {
      const built = k.kernel;
      if (!built) return false;
      const names = built.argumentNames || [];
      if (names.length !== arity) return false;
      if (width == null) return true;
      const output = built.output;
      return output && Number(output[0]) === width;
    }) || null
  );
}

// Every number that appeared in a console.log line (system lines excluded —
// "output 1024 · 1,024 threads" is not the learner's answer).
function loggedNumbers(logs) {
  const out = [];
  for (const line of logs) {
    if (line.type !== 'log' || !line.text) continue;
    const matches = line.text.match(/-?\d+(?:\.\d+)?/g);
    if (matches) for (const m of matches) out.push(parseFloat(m));
  }
  return out;
}

function hasNumber(nums, value, eps = 0.5) {
  return nums.some(v => Math.abs(v - value) <= eps);
}

// Does `nums` contain `seq` as a contiguous run? Robust to how the learner
// separated the values — join(' '), commas, or one log line per element.
function containsRun(nums, seq) {
  for (let i = 0; i + seq.length <= nums.length; i++) {
    let ok = true;
    for (let j = 0; j < seq.length; j++) {
      if (Math.abs(nums[i + j] - seq[j]) > 0.5) {
        ok = false;
        break;
      }
    }
    if (ok) return true;
  }
  return false;
}

// ---- near-miss diagnosis --------------------------------------------------
//
// A failing assert that reports only two numbers tells a learner nothing about
// WHICH slip produced them. A probe pairs the value one specific known mistake
// would produce with a sentence naming that mistake, and the probe speaks only
// when the observation matches it AND the correct answer does not — so a case
// where two candidates coincide stays silent, as do observations matching
// probes that disagree with each other. A wrong diagnosis is worse than none.
//
// Everything this module checks is an ARRAY, and one matching cell is worthless
// evidence about a permutation — element 0 of a stable rank and of an unstable
// one agree constantly. So a probe here must predict EVERY element of the
// observed array, and disagree with the right answer somewhere, before it is
// allowed to speak.
function diagnoseArray(got, expected, probes, eps = 0.5) {
  if (!got || got.length !== expected.length) return null;
  const hits = probes
    .filter(([candidate]) => {
      let differs = false;
      for (let i = 0; i < expected.length; i++) {
        if (!(Math.abs(got[i] - candidate[i]) <= eps)) return false;
        if (Math.abs(expected[i] - candidate[i]) > eps) differs = true;
      }
      return differs;
    })
    .map(p => p[1]);
  return hits.length && hits.every(m => m === hits[0]) ? hits[0] : null;
}

// The three ways the rank inside a bucket goes wrong. Tasks 1 and 5 both
// produce them, in different shapes, so the sentences live here once.
const UNSTABLE =
  'this pass is not stable — it is correctly ordered by digit, but every run of equal ' +
  'digits comes out reversed. Count an equal digit only when it started EARLIER: ' +
  'j < this.thread.x. Reverse a run here and the next pass has nothing left to preserve.';

const NO_TIE_BREAK =
  'every element sharing your digit got the SAME destination — the tie-break is missing. ' +
  'Add one for each equal digit that started before you.';

const COUNTS_ITSELF =
  'every destination is one too high — the loop counts your own element as coming before ' +
  'you. The comparison is j < this.thread.x, not j <= this.thread.x.';

// Task 1: the rank is over the whole array, so the three mistakes are whole
// alternative rankings.
function rankProbes(digits) {
  return [
    [reversedRank(digits), UNSTABLE],
    [collidingRank(digits), NO_TIE_BREAK],
    [selfCountingRank(digits), COUNTS_ITSELF],
  ];
}

// Task 5: the same three mistakes, but the kernel adds starts[digit] to a rank
// WITHIN the bucket — so each candidate is rebuilt in that shape rather than
// reusing task 1's global ordering. Plus the missing bucket offset itself.
function destinationProbes(values, place, starts) {
  const digits = values.map(v => digitOf(v, place));
  const build = tie => {
    const counters = new Array(RADIX).fill(0);
    const out = new Array(values.length);
    for (let i = 0; i < values.length; i++) {
      const d = digits[i];
      out[i] = starts[d] + tie(d, counters);
      counters[d]++;
    }
    return out;
  };
  const reversed = values.map((v, i) => {
    const d = digits[i];
    let after = 0;
    for (let j = i + 1; j < values.length; j++) if (digits[j] === d) after++;
    return starts[d] + after;
  });
  return [
    [reversed, UNSTABLE],
    [build(() => 0), NO_TIE_BREAK],
    [build((d, counters) => counters[d] + 1), COUNTS_ITSELF],
    [destinationsOf(values, place, new Array(RADIX).fill(0)),
      'the bucket offset never got added — starts[digit] is what keeps bucket 3 from ' +
      'landing on top of bucket 0. Return starts[digit] + your rank inside it.'],
  ];
}

// Task 4: the two ways an inversion goes wrong, plus doing nothing at all.
function gatherProbes(keys, destinations) {
  const n = keys.length;
  const inverse = new Array(n).fill(0);
  for (let i = 0; i < n; i++) inverse[destinations[i]] = i;
  return [
    [destinations.map(d => keys[d]),
      'the permutation was applied backwards — destinations[i] says where element i GOES, ' +
      'so it is not a source index. Slot x has to find the i whose destination is x.'],
    [inverse,
      'that is the index of the element that belongs here, not its value — return keys[i], ' +
      'not i.'],
    [keys.slice(),
      'nothing moved: every slot returned its own key. The loop has to compare ' +
      'destinations[i] against this.thread.x.'],
  ];
}

// Task 5: the driver. Its mistakes never touch a kernel, so they are read off
// the three landmarks the finished array logs. All three must match a
// candidate — one landmark agreeing means nothing, and two-passes and
// most-significant-first genuinely share both ends of the array.
function driverHint(nums, correct, candidates) {
  const hits = candidates
    .filter(([marks]) => {
      let differs = false;
      for (let i = 0; i < marks.length; i++) {
        if (!hasNumber(nums, marks[i])) return false;
        if (Math.abs(correct[i] - marks[i]) > 0.5) differs = true;
      }
      return differs;
    })
    .map(c => c[1]);
  return hits.length && hits.every(m => m === hits[0]) ? hits[0] : null;
}

function landmarks(values) {
  const n = values.length;
  return [values[0], values[n >> 1], values[n - 1]];
}

function driverCandidates(keys) {
  const places = [1, RADIX, RADIX * RADIX];
  return [
    [landmarks(radixDrive(keys, [1, RADIX], false)),
      'the high digit never got sorted — that result is ordered by the low eight bits only. ' +
      'Keys below 4096 have three hex digits, so the driver needs three passes: ' +
      'place = 1, 16 and 256.'],
    [landmarks(radixDrive(keys, [1], false)),
      'only one pass ran — that result is ordered by the ones digit and nothing else.'],
    [landmarks(radixDrive(keys, places.slice().reverse(), false)),
      'the passes ran most significant digit FIRST. Each pass overwrites the ordering of the ' +
      'one before it except where digits tie, so the last pass decides the primary key — ' +
      'which means the low digit must go LAST, not first.'],
    [landmarks(radixDrive(keys, places, true)),
      'the histogram and the offsets were computed once and reused. Every pass looks at a ' +
      'different digit of a differently ordered array, so both have to be recomputed inside ' +
      'the loop.'],
  ];
}

export default {
  uuid: 'fd3ff796-daed-4036-9202-987b374bb4d3',
  version: 1,
  slug: 'radix-sort',
  title: 'Radix Sort',
  blurb: 'A histogram, a scan and a gather assembled into the sort production GPU libraries actually run.',
  tasks: [
    // ------------------------------------------------------------------ 1
    {
      slug: 'one-digit-pass',
      title: 'Sort by One Digit',
      intro: `<p>Radix sort never compares two keys. It sorts by <strong>one digit at a time</strong>,
        starting with the least significant, and after enough passes the array is sorted — which
        reads like a card trick until you watch it happen:</p>
<pre><code>  start    by ones    by tens
    34        21         13
    21        13         21  ← tie
    13        34         27  ← tie
    27        27         34</code></pre>
        <p>The tens pass never looks at the ones digit. All it knows is that 21 and 27 both have
        a 2 — and the only reason 21 still comes out first is that the pass is
        <strong>stable</strong>: it leaves equal digits in the order it found them, and the ones
        pass had already put 21 first. Break that and the earlier pass's work is destroyed. An
        unstable tens pass may emit <code>13, 27, 21, 34</code>: perfectly ordered by tens digit,
        and not sorted.</p>
        <p>So each pass has to answer one question per element: <em>how many elements belong in
        front of me?</em> Everything with a smaller digit, plus everything with the same digit
        that started earlier. That second clause <strong>is</strong> stability.</p>`,
      goal: `<strong>Goal:</strong> for every element of <code>digits</code>, return the index it
        lands on in a stable one-digit pass.`,
      requirements: [
        'Loop over all <code>this.constants.n</code> digits — one pass over the array per thread',
        'Count every digit strictly smaller than yours',
        'Break ties by original position: count an equal digit only when its index is before <code>this.thread.x</code>',
        'Return the count — that count <em>is</em> the destination',
      ],
      hints: [
        {
          title: 'Hint 1 — two counts, one loop',
          body: `<p>Walk every <code>j</code> from 0 to <code>n − 1</code> and ask two questions
            about <code>digits[j]</code>: is it smaller than mine? and if it is <em>equal</em> to
            mine, did it start before me? Either one puts that element in front of you.</p>`,
        },
        {
          title: 'Hint 2 — the tie-break',
          body: `<pre><code>const other = digits[j];
if (other &lt; mine) {
  before++;
} else if (other === mine &amp;&amp; j &lt; this.thread.x) {
  before++;
}</code></pre>
<p>The <code>j &lt; this.thread.x</code> is the entire stability guarantee. Turn it
            round and the pass still sorts by digit — and still destroys everything the previous
            pass did.</p>`,
        },
      ],
      transfer: `Every production GPU radix sort is a <em>stable</em> sort, and not by accident:
        NVIDIA's CUB ranks each key inside its digit with <code>BlockRadixRank</code>, AMD's
        rocPRIM and Metal's sort primitives do the same. Stability is what makes multi-pass radix
        sorting work at all, and it is also what lets you sort key–value pairs, or sort by one
        field and then another, and trust the result.`,
      starterCode: `// A stable pass answers one question per element:
// how many elements belong in front of me?
const gpu = new GPU({ mode });

const destination = gpu.createKernel(function (digits) {
  const mine = digits[this.thread.x];
  let before = 0;
  for (let j = 0; j < this.constants.n; j++) {
    // TODO: count the digits that belong in front of this one —
    // everything smaller, plus the EQUAL digits that started earlier.
    if (digits[j] < mine) {
      before++;
    }
  }
  return before;
}, {
  output: [16],
  constants: { n: 16 },
});

const dest = destination(digits);
console.log('digits:      ', digits.join(' '));
console.log('destinations:', Array.from(dest).join(' '));
`,
      solutionCode: `// A stable pass answers one question per element:
// how many elements belong in front of me?
const gpu = new GPU({ mode });

const destination = gpu.createKernel(function (digits) {
  const mine = digits[this.thread.x];
  let before = 0;
  for (let j = 0; j < this.constants.n; j++) {
    const other = digits[j];
    if (other < mine) {
      before++;
    } else if (other === mine && j < this.thread.x) {
      before++;
    }
  }
  return before;
}, {
  output: [16],
  constants: { n: 16 },
});

const dest = destination(digits);
console.log('digits:      ', digits.join(' '));
console.log('destinations:', Array.from(dest).join(' '));
`,
      inputs: utils => ({ digits: makeDigits(utils, 16) }),
      publicTests: [
        {
          name: 'every element gets its own slot — the 16 destinations are a permutation',
          run: async ctx => {
            ctx.assert(ctx.kernels.length >= 1, 'no kernel was created — call gpu.createKernel()');
            const digits = makeDigits(ctx.utils, 16);
            const out = plain(call(ctx.kernel, [digits]));
            ctx.assert(out.length === 16, `expected 16 destinations, got ${out.length}`);
            const taken = new Array(16).fill(0);
            for (let i = 0; i < 16; i++) {
              const d = out[i];
              ctx.assert(
                Number.isFinite(d) && d >= 0 && d < 16,
                diagnoseArray(out, stableRank(digits), rankProbes(digits)) ||
                  `destination ${d} for element ${i} is outside 0…15`
              );
              taken[Math.round(d)]++;
            }
            const clash = taken.findIndex(count => count > 1);
            ctx.assert(
              clash === -1,
              diagnoseArray(out, stableRank(digits), rankProbes(digits)) ||
                `${taken[clash]} elements were all sent to slot ${clash} — every element needs its own`
            );
          },
        },
        {
          name: 'ties keep their original order — equal digits stay put',
          run: async ctx => {
            // Every digit equal: the only thing left to order by is the
            // original position, so a stable pass must be the identity.
            const flat = new Array(16).fill(5);
            const out = plain(call(ctx.kernel, [flat]));
            const expected = stableRank(flat);
            const hint = diagnoseArray(out, expected, rankProbes(flat));
            for (let i = 0; i < 16; i++) {
              ctx.assertClose(out[i], expected[i], 0.5, hint || `element ${i} of an all-equal input`);
            }
          },
        },
        {
          name: 'each destination counts the smaller digits plus the earlier equal ones',
          run: async ctx => {
            const digits = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9, 3];
            const out = plain(call(ctx.kernel, [digits]));
            const expected = stableRank(digits);
            const hint = diagnoseArray(out, expected, rankProbes(digits));
            for (let i = 0; i < 16; i++) {
              ctx.assertClose(out[i], expected[i], 0.5, hint || `element ${i} (digit ${digits[i]})`);
            }
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            for (const seed of [4242, 777, 90210]) {
              const digits = makeDigits(ctx.utils, 16, seed);
              const out = plain(call(ctx.kernel, [digits]));
              const expected = stableRank(digits);
              const hint = diagnoseArray(out, expected, rankProbes(digits));
              for (let i = 0; i < 16; i++) {
                ctx.assertClose(out[i], expected[i], 0.5, hint || `element ${i} (seed ${seed})`);
              }
            }
          },
        },
        {
          name: 'private test #2',
          run: async ctx => {
            // Two long runs of equal digits, deliberately out of order: a pass
            // that reverses ties cannot hide here.
            const digits = [7, 7, 7, 7, 2, 2, 2, 2, 7, 2, 7, 2, 2, 7, 2, 7];
            const out = plain(call(ctx.kernel, [digits]));
            const expected = stableRank(digits);
            const hint = diagnoseArray(out, expected, rankProbes(digits));
            for (let i = 0; i < 16; i++) {
              ctx.assertClose(out[i], expected[i], 0.5, hint || `element ${i} (digit ${digits[i]})`);
            }
          },
        },
      ],
    },

    // ------------------------------------------------------------------ 2
    {
      slug: 'binary-pass',
      title: 'One Bit at a Time',
      intro: `<p>Take the narrowest radix there is: <strong>2</strong>. One bit per pass, two
        buckets, and the whole bucket table collapses to a single number — how many zeros there
        are. Zeros go to the front in the order they appeared; ones go behind them, also in
        order. So element <code>i</code>'s destination is either <em>how many zeros are before
        me</em>, or <em>every zero, plus how many ones are before me</em>.</p>
        <p>That count of preceding flags is a running total over a 0/1 array — the same shape
        stream compaction uses to close its gaps, except here neither half gets thrown away.
        The zero total is 32 numbers coming back to JavaScript, which is cheap to finish there.</p>
        <p>Moving the data is one line of ordinary JavaScript: <code>out[dest[i]] = keys[i]</code>.
        Enjoy it while it lasts. That line is a <strong>scatter</strong>, and it is the one thing
        a kernel cannot do — task 4 is about turning it inside out.</p>`,
      goal: `<strong>Goal:</strong> split <code>keys</code> by their low bit — even keys first,
        odd keys behind them, each half keeping its original order — and log the result.`,
      requirements: [
        '<code>lowBit</code> returns <code>keys[this.thread.x] % 2</code> — a 0/1 flag per key',
        'Count the zeros in plain JavaScript and pass the total into the second kernel',
        'A zero\'s destination is how many zeros came before it; a one\'s is <code>zeros</code> plus how many ones came before it',
        '<code>console.log</code> the reordered array (the starter\'s last line already does)',
      ],
      hints: [
        {
          title: 'Hint 1 — count your own kind',
          body: `<p>Both halves need the same thing: how many <em>earlier</em> elements share your
            flag. One loop does it for either flag —</p>
<pre><code>if (bits[j] === mine &amp;&amp; j &lt; this.thread.x) before++;</code></pre>
<p>— and then only the starting point differs.</p>`,
        },
        {
          title: 'Hint 2 — the two starting points',
          body: `<p>The zero bucket starts at slot 0. The one bucket starts right after every
            zero, at slot <code>zeros</code>:</p>
<pre><code>if (mine === 0) return before;
return zeros + before;</code></pre>`,
        },
      ],
      transfer: `The one-bit split is where GPU radix sorting started — Satish, Harris and
        Garland's manycore sorting paper builds an entire sort from it, one bit at a time, with a
        prefix sum over the flag array supplying every destination. Modern hardware does the
        counting in a single instruction: CUDA's <code>__ballot_sync</code> + <code>__popc</code>
        and WGSL's <code>subgroupBallot</code> give a warp its flag ranks for free.`,
      starterCode: `// Radix 2: two buckets, and the whole bucket table is one number.
const gpu = new GPU({ mode });

const lowBit = gpu.createKernel(function (keys) {
  // TODO: return this key's low bit — 0 for even, 1 for odd
  return 0;
}, { output: [32] });

const destination = gpu.createKernel(function (bits, zeros) {
  const mine = bits[this.thread.x];
  let before = 0;
  for (let j = 0; j < this.constants.n; j++) {
    // TODO: count the EARLIER elements carrying the same flag
    before += 0;
  }
  // TODO: zeros start at slot 0; ones start after every zero
  return before;
}, {
  output: [32],
  constants: { n: 32 },
});

const bits = lowBit(keys);

let zeros = 0;
for (let i = 0; i < bits.length; i++) {
  if (bits[i] === 0) zeros++;
}
console.log('zeros:', zeros);

const dest = destination(bits, zeros);

// A scatter — fine in JavaScript, impossible inside a kernel. Task 4 fixes it.
const out = new Array(32);
for (let i = 0; i < 32; i++) out[dest[i]] = keys[i];
console.log('after the pass:', out.join(' '));
`,
      solutionCode: `// Radix 2: two buckets, and the whole bucket table is one number.
const gpu = new GPU({ mode });

const lowBit = gpu.createKernel(function (keys) {
  return keys[this.thread.x] % 2;
}, { output: [32] });

const destination = gpu.createKernel(function (bits, zeros) {
  const mine = bits[this.thread.x];
  let before = 0;
  for (let j = 0; j < this.constants.n; j++) {
    if (bits[j] === mine && j < this.thread.x) {
      before++;
    }
  }
  if (mine === 0) {
    return before;
  }
  return zeros + before;
}, {
  output: [32],
  constants: { n: 32 },
});

const bits = lowBit(keys);

let zeros = 0;
for (let i = 0; i < bits.length; i++) {
  if (bits[i] === 0) zeros++;
}
console.log('zeros:', zeros);

const dest = destination(bits, zeros);

// A scatter — fine in JavaScript, impossible inside a kernel. Task 4 fixes it.
const out = new Array(32);
for (let i = 0; i < 32; i++) out[dest[i]] = keys[i];
console.log('after the pass:', out.join(' '));
`,
      inputs: utils => ({ keys: makeKeys(utils, 32, 256, 2718) }),
      publicTests: [
        {
          name: 'the flag kernel returns each key\'s low bit',
          run: async ctx => {
            ctx.assert(ctx.kernels.length >= 2, `expected 2 kernels, found ${ctx.kernels.length}`);
            const flags = findKernel(ctx, 1, 32);
            ctx.assert(flags, 'no one-argument kernel with output [32] found — that is the flag kernel');
            const keys = makeKeys(ctx.utils, 32, 256, 2718);
            const out = plain(call(flags, [keys]));
            const expected = lowBits(keys);
            const hint = diagnoseArray(out, expected, [
              [keys.slice(), 'the key came back whole — the flag is keys[this.thread.x] % 2, one bit'],
              [keys.map(k => Math.floor(k / 2)), 'that divided by 2 instead of taking the remainder — % 2, not / 2'],
            ]);
            for (let i = 0; i < 32; i++) {
              ctx.assertClose(out[i], expected[i], 0.5, hint || `flag ${i} (key ${keys[i]})`);
            }
          },
        },
        {
          name: 'zeros keep their order at the front, ones keep theirs behind',
          run: async ctx => {
            const dest = findKernel(ctx, 2, 32);
            ctx.assert(dest, 'no two-argument kernel with output [32] found — that is the destination kernel');
            const keys = makeKeys(ctx.utils, 32, 256, 2718);
            const bits = lowBits(keys);
            const zeros = zeroCount(bits);
            const out = plain(call(dest, [bits, zeros]));
            const expected = binaryDestinations(bits);
            const hint = diagnoseArray(out, expected, [
              [noOffsetDestinations(bits),
                'the odd keys start from 0 as well, so both halves pile into the same slots — a ' +
                'one\'s destination is zeros + how many ones came before it'],
              [swappedDestinations(bits),
                'the two branches are the wrong way round: this puts the ODD keys first. Zeros ' +
                'take the front of the array'],
              [bucketSizeDestinations(bits),
                'every element got its whole bucket\'s size — the loop counts all matching flags ' +
                'instead of only the ones before this.thread.x'],
            ]);
            for (let i = 0; i < 32; i++) {
              ctx.assertClose(out[i], expected[i], 0.5, hint || `element ${i} (flag ${bits[i]})`);
            }
          },
        },
        {
          name: 'the reordered array is logged — evens first, in order',
          run: async ctx => {
            const keys = makeKeys(ctx.utils, 32, 256, 2718);
            const expected = gatherBy(keys, binaryDestinations(lowBits(keys)));
            const nums = loggedNumbers(ctx.logs);
            ctx.assert(
              containsRun(nums, expected),
              'log the reordered array — the starter\'s last line does it with ' +
                `out.join(' '), and it should start ${expected.slice(0, 4).join(', ')}, …`
            );
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            const dest = findKernel(ctx, 2, 32);
            ctx.assert(dest, 'expected a two-argument destination kernel');
            for (const seed of [909, 31337]) {
              const keys = makeKeys(ctx.utils, 32, 256, seed);
              const bits = lowBits(keys);
              const out = plain(call(dest, [bits, zeroCount(bits)]));
              const expected = binaryDestinations(bits);
              const hint = diagnoseArray(out, expected, [
                [noOffsetDestinations(bits),
                  'the odd keys start from 0 as well — a one\'s destination is zeros + how many ' +
                  'ones came before it'],
                [swappedDestinations(bits),
                  'the two branches are the wrong way round: this puts the ODD keys first'],
                [bucketSizeDestinations(bits),
                  'every element got its whole bucket\'s size — count only the matching flags ' +
                  'before this.thread.x'],
              ]);
              for (let i = 0; i < 32; i++) {
                ctx.assertClose(out[i], expected[i], 0.5, hint || `element ${i} (seed ${seed})`);
              }
            }
          },
        },
        {
          name: 'private test #2',
          run: async ctx => {
            // All-odd input: `zeros` is 0, so the one bucket starts at slot 0
            // and the destinations must still be the identity.
            const dest = findKernel(ctx, 2, 32);
            const bits = new Array(32).fill(1);
            const out = plain(call(dest, [bits, 0]));
            for (let i = 0; i < 32; i++) {
              ctx.assertClose(out[i], i, 0.5, `element ${i} of an all-odd input should stay put`);
            }
          },
        },
      ],
    },

    // ------------------------------------------------------------------ 3
    {
      slug: 'digit-histogram',
      title: 'Widen the Radix',
      intro: `<p>One bit per pass means 32 passes for a 32-bit key. Four bits per pass means
        <strong>16 buckets</strong> and eight passes — the same total work rearranged, with far
        fewer round trips. That is the real engineering trade in radix sorting, and every library
        picks a number here (4 and 8 bits are the usual answers).</p>
        <p>The price is that the bucket table stops being a single number. You need a
        <strong>histogram</strong> — how many keys carry each of the 16 digits — and then a
        running total across the buckets to turn those counts into <strong>starting offsets</strong>:
        bucket <code>b</code> begins after every key whose digit is smaller than <code>b</code>.
        Both of those are primitives in their own right (and each has a module of its own); at 16
        buckets they are small enough to write out in a loop.</p>
        <p>Note the word <em>smaller</em>. The scan is <strong>exclusive</strong>: bucket 0 starts
        at slot 0, and bucket <code>b</code>'s offset stops at <code>b − 1</code>. Include your own
        count and every bucket starts one whole bucket too far along.</p>`,
      goal: `<strong>Goal:</strong> write both kernels — <code>histogram</code> counts the keys in
        each of the 16 digit buckets at a given <code>place</code>, and <code>offsets</code> turns
        those counts into starting slots with an exclusive scan.`,
      requirements: [
        'The digit of a key at <code>place</code> is <code>Math.floor(key / place) % this.constants.radix</code>',
        '<code>histogram</code>: 16 threads, thread <code>b</code> counts the keys whose digit is <code>b</code>',
        '<code>offsets</code>: thread <code>b</code> totals <code>counts[0 … b−1]</code> — exclusive, so <code>offsets[0]</code> is <code>0</code>',
      ],
      hints: [
        {
          title: 'Hint 1 — extracting a digit',
          body: `<p><code>place</code> selects which digit you want: <code>1</code> for the ones
            digit, <code>16</code> for the sixteens, <code>256</code> for the next. Divide it away,
            then take what is left modulo the radix:</p>
<pre><code>const d = Math.floor(keys[i] / place) % this.constants.radix;</code></pre>
<p>Skip the <code>% 16</code> and <code>d</code> is the whole quotient, not a digit.</p>`,
        },
        {
          title: 'Hint 2 — the histogram is a gather, not a scatter',
          body: `<p>You cannot walk the keys and bump a counter — that is 64 threads fighting over
            16 cells. Invert it: each of the 16 threads owns one bucket and walks the whole key
            array counting its own digit. <code>if (d === this.thread.x) count++;</code></p>`,
        },
        {
          title: 'Hint 3 — exclusive means stop early',
          body: `<pre><code>for (let b = 0; b &lt; this.constants.radix; b++) {
  if (b &lt; this.thread.x) start += counts[b];
}</code></pre>
<p>Sixteen values is far too few to be worth a clever scan; the point is the
            <code>b &lt; this.thread.x</code>.</p>`,
        },
      ],
      transfer: `Count, scan, scatter is the skeleton of every real GPU radix sort: CUB and
        rocPRIM histogram each tile of keys locally, scan the per-tile histograms into global
        digit offsets, then move the keys. Choosing the radix is a genuine tuning knob — wider
        digits mean fewer passes over memory but a bigger bucket table to keep on chip, which is
        why 4 and 8 bits win in practice and 16 does not.`,
      starterCode: `// 16 buckets now, so the bucket table needs counting and scanning.
const gpu = new GPU({ mode });

const histogram = gpu.createKernel(function (keys, place) {
  let count = 0;
  for (let i = 0; i < this.constants.n; i++) {
    // TODO: this key's digit at \`place\`, then count it if it is MY bucket
    const d = 0;
    if (d === this.thread.x) {
      count++;
    }
  }
  return count;
}, {
  output: [16],
  constants: { n: 64, radix: 16 },
});

const offsets = gpu.createKernel(function (counts) {
  let start = 0;
  for (let b = 0; b < this.constants.radix; b++) {
    // TODO: add the buckets STRICTLY BEFORE this one
    start += 0;
  }
  return start;
}, {
  output: [16],
  constants: { radix: 16 },
});

const onesCounts = histogram(keys, 1);
console.log('ones-digit counts: ', Array.from(onesCounts).join(' '));
console.log('ones-digit offsets:', Array.from(offsets(onesCounts)).join(' '));

const sixteensCounts = histogram(keys, 16);
console.log('16s-digit counts:  ', Array.from(sixteensCounts).join(' '));
console.log('16s-digit offsets: ', Array.from(offsets(sixteensCounts)).join(' '));
`,
      solutionCode: `// 16 buckets now, so the bucket table needs counting and scanning.
const gpu = new GPU({ mode });

const histogram = gpu.createKernel(function (keys, place) {
  let count = 0;
  for (let i = 0; i < this.constants.n; i++) {
    const d = Math.floor(keys[i] / place) % this.constants.radix;
    if (d === this.thread.x) {
      count++;
    }
  }
  return count;
}, {
  output: [16],
  constants: { n: 64, radix: 16 },
});

const offsets = gpu.createKernel(function (counts) {
  let start = 0;
  for (let b = 0; b < this.constants.radix; b++) {
    if (b < this.thread.x) {
      start += counts[b];
    }
  }
  return start;
}, {
  output: [16],
  constants: { radix: 16 },
});

const onesCounts = histogram(keys, 1);
console.log('ones-digit counts: ', Array.from(onesCounts).join(' '));
console.log('ones-digit offsets:', Array.from(offsets(onesCounts)).join(' '));

const sixteensCounts = histogram(keys, 16);
console.log('16s-digit counts:  ', Array.from(sixteensCounts).join(' '));
console.log('16s-digit offsets: ', Array.from(offsets(sixteensCounts)).join(' '));
`,
      inputs: utils => ({ keys: makeKeys(utils, 64, 256, 6001) }),
      publicTests: [
        {
          name: 'the histogram counts all 64 keys into 16 buckets',
          run: async ctx => {
            ctx.assert(ctx.kernels.length >= 2, `expected 2 kernels, found ${ctx.kernels.length}`);
            const hist = findKernel(ctx, 2, 16);
            ctx.assert(hist, 'no two-argument kernel with output [16] found — that is the histogram');
            const keys = makeKeys(ctx.utils, 64, 256, 6001);
            const out = plain(call(hist, [keys, 1]));
            ctx.assert(out.length === 16, `expected 16 buckets, got ${out.length}`);
            const total = out.reduce((a, b) => a + b, 0);
            ctx.assertClose(
              total,
              64,
              0.5,
              diagnoseArray(out, countsOf(keys, 1), [
                [unmoddedCounts(keys, 1),
                  'the digit is missing its % this.constants.radix, so bucket b only matched keys ' +
                  'whose whole quotient is b — every key of 16 or more fell out of the histogram'],
              ]) || 'the 16 bucket counts should add up to all 64 keys'
            );
          },
        },
        {
          name: 'the histogram reads the digit at the requested <code>place</code>',
          run: async ctx => {
            const hist = findKernel(ctx, 2, 16);
            const keys = makeKeys(ctx.utils, 64, 256, 6001);
            for (const place of [1, 16]) {
              const out = plain(call(hist, [keys, place]));
              const expected = countsOf(keys, place);
              const hint = diagnoseArray(out, expected, [
                [countsOf(keys, place === 1 ? 16 : 1),
                  `those are the counts for the other digit — the pass asked for place ${place}, ` +
                  'so the divisor has to be the `place` argument'],
                [unmoddedCounts(keys, place),
                  'the digit is missing its % this.constants.radix — Math.floor(key / place) is a ' +
                  'quotient, not a digit'],
              ]);
              for (let b = 0; b < 16; b++) {
                ctx.assertClose(out[b], expected[b], 0.5, hint || `bucket ${b} at place ${place}`);
              }
            }
          },
        },
        {
          name: 'offsets are an <strong>exclusive</strong> scan — bucket 0 starts at 0',
          run: async ctx => {
            const scan = findKernel(ctx, 1, 16);
            ctx.assert(scan, 'no one-argument kernel with output [16] found — that is the offsets scan');
            const counts = [3, 0, 7, 2, 1, 9, 0, 4, 6, 1, 0, 8, 2, 5, 1, 15];
            const out = plain(call(scan, [counts]));
            const expected = startsOf(counts);
            const hint = diagnoseArray(out, expected, [
              [inclusiveOf(counts),
                'that is an INCLUSIVE scan — every bucket starts one whole bucket too far along. ' +
                'A bucket begins after the keys with a strictly SMALLER digit, so the sum has to ' +
                'stop at b − 1 and offsets[0] must be 0'],
              [counts.slice(),
                'the counts came back unchanged — offsets[b] is the running total of everything ' +
                'before bucket b, not bucket b\'s own count'],
            ]);
            for (let b = 0; b < 16; b++) {
              ctx.assertClose(out[b], expected[b], 0.5, hint || `offset ${b}`);
            }
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            const hist = findKernel(ctx, 2, 16);
            const scan = findKernel(ctx, 1, 16);
            ctx.assert(hist && scan, 'expected a histogram kernel and an offsets kernel');
            for (const seed of [4096, 55555]) {
              const keys = makeKeys(ctx.utils, 64, 256, seed);
              for (const place of [1, 16]) {
                const counts = plain(call(hist, [keys, place]));
                const expectedCounts = countsOf(keys, place);
                const countHint = diagnoseArray(counts, expectedCounts, [
                  [countsOf(keys, place === 1 ? 16 : 1),
                    'those are the counts for the other digit — divide by the `place` argument'],
                  [unmoddedCounts(keys, place),
                    'the digit is missing its % this.constants.radix'],
                ]);
                for (let b = 0; b < 16; b++) {
                  ctx.assertClose(counts[b], expectedCounts[b], 0.5,
                    countHint || `bucket ${b} at place ${place} (seed ${seed})`);
                }
                const starts = plain(call(scan, [expectedCounts]));
                const expectedStarts = startsOf(expectedCounts);
                const scanHint = diagnoseArray(starts, expectedStarts, [
                  [inclusiveOf(expectedCounts),
                    'that is an inclusive scan — the sum has to stop at b − 1'],
                  [expectedCounts.slice(), 'the counts came back unchanged — they were never totalled'],
                ]);
                for (let b = 0; b < 16; b++) {
                  ctx.assertClose(starts[b], expectedStarts[b], 0.5,
                    scanHint || `offset ${b} at place ${place} (seed ${seed})`);
                }
              }
            }
          },
        },
        {
          name: 'private test #2',
          run: async ctx => {
            // Empty buckets must not shift anything: two buckets holding every
            // key, fourteen holding none.
            const scan = findKernel(ctx, 1, 16);
            const counts = new Array(16).fill(0);
            counts[0] = 40;
            counts[9] = 24;
            const out = plain(call(scan, [counts]));
            const expected = startsOf(counts);
            const hint = diagnoseArray(out, expected, [
              [inclusiveOf(counts), 'that is an inclusive scan — the sum has to stop at b − 1'],
            ]);
            for (let b = 0; b < 16; b++) {
              ctx.assertClose(out[b], expected[b], 0.5, hint || `offset ${b}`);
            }
          },
        },
      ],
    },

    // ------------------------------------------------------------------ 4
    {
      slug: 'the-gather',
      title: 'Whose Value Lands Here?',
      intro: `<p>Everything so far produced a <strong>plan</strong>: for each element, the slot it
        belongs in. Executing the plan is the one move a kernel does not have.
        <code>out[destinations[i]] = keys[i]</code> is a <strong>scatter</strong> — a thread
        writing somewhere other than its own cell — and gpu.js has no such thing (Thinking in
        Parallel makes a whole module of why).</p>
        <p>So turn the question round, exactly as you would anywhere else on a GPU. Instead of
        <em>"where does my value go?"</em>, output slot <code>x</code> asks
        <em>"which element wants me?"</em> — sweep the destinations, find the one that equals
        <code>x</code>, and take that element's key. Every thread reads the whole plan and writes
        one cell. It looks wasteful and it is completely parallel, which on a GPU is the trade
        you take.</p>`,
      goal: `<strong>Goal:</strong> apply the permutation with a gather — output slot
        <code>x</code> holds the key of the element whose destination is <code>x</code>.`,
      requirements: [
        'No writes anywhere but your own cell — the answer is a <code>return</code>',
        'Sweep all <code>this.constants.n</code> destinations looking for <code>this.thread.x</code>',
        'Return that element\'s <em>key</em>, not its index',
      ],
      hints: [
        {
          title: 'Hint 1 — which comparison?',
          body: `<p><code>destinations[i]</code> is where element <code>i</code> is <em>going</em>.
            Your cell is <code>this.thread.x</code>. So the element you want is the one where
            those two are equal — never <code>keys[destinations[this.thread.x]]</code>, which
            applies the permutation backwards.</p>`,
        },
        {
          title: 'Hint 2 — the sweep',
          body: `<pre><code>let value = 0;
for (let i = 0; i &lt; this.constants.n; i++) {
  if (destinations[i] === this.thread.x) {
    value = keys[i];
  }
}
return value;</code></pre>`,
        },
      ],
      transfer: `Compute APIs do let you scatter — CUDA and WebGPU threads can store to any buffer
        address — and a production radix sort uses that: it writes keys straight to their computed
        offsets, which is why it also needs atomics and shared memory to arrange those offsets
        safely. Where you have no scatter, the inversion here is the standard replacement, and it
        is the same move a fragment shader has made since the beginning: every output pixel pulls
        what it needs.`,
      starterCode: `// The plan is done. Now move the data — without a scatter.
const gpu = new GPU({ mode });

const gather = gpu.createKernel(function (keys, destinations) {
  // TODO: find the element whose destination is THIS cell,
  // and return its key.
  return keys[this.thread.x];
}, {
  output: [64],
  constants: { n: 64 },
});

const sorted = gather(keys, destinations);
console.log('before:', keys.slice(0, 8).join(' '), '…');
console.log('after: ', Array.from(sorted).slice(0, 8).join(' '), '…');
`,
      solutionCode: `// The plan is done. Now move the data — without a scatter.
const gpu = new GPU({ mode });

const gather = gpu.createKernel(function (keys, destinations) {
  let value = 0;
  for (let i = 0; i < this.constants.n; i++) {
    if (destinations[i] === this.thread.x) {
      value = keys[i];
    }
  }
  return value;
}, {
  output: [64],
  constants: { n: 64 },
});

const sorted = gather(keys, destinations);
console.log('before:', keys.slice(0, 8).join(' '), '…');
console.log('after: ', Array.from(sorted).slice(0, 8).join(' '), '…');
`,
      inputs: utils => {
        const keys = makeKeys(utils, 64, 256, 1729);
        return {
          keys,
          destinations: destinationsOf(keys, 1, startsOf(countsOf(keys, 1))),
        };
      },
      publicTests: [
        {
          name: 'the gather applies the plan — 64 keys, reordered',
          run: async ctx => {
            ctx.assert(ctx.kernels.length >= 1, 'no kernel was created — call gpu.createKernel()');
            const keys = makeKeys(ctx.utils, 64, 256, 1729);
            const destinations = destinationsOf(keys, 1, startsOf(countsOf(keys, 1)));
            const out = plain(call(ctx.kernel, [keys, destinations]));
            ctx.assert(out.length === 64, `expected 64 values, got ${out.length}`);
            const expected = gatherBy(keys, destinations);
            const hint = diagnoseArray(out, expected, gatherProbes(keys, destinations));
            for (let i = 0; i < 64; i++) {
              ctx.assertClose(out[i], expected[i], 0.5, hint || `slot ${i}`);
            }
          },
        },
        {
          name: 'a reversal is a permutation too — <code>destinations[i] = n − 1 − i</code>',
          run: async ctx => {
            // A reversal is its own inverse, so applying the plan backwards
            // looks identical here — which is exactly why the probes above
            // have to predict every cell of a NON-involution before speaking.
            const keys = new Array(64);
            for (let i = 0; i < 64; i++) keys[i] = i * 3 + 7;
            const destinations = keys.map((v, i) => 63 - i);
            const out = plain(call(ctx.kernel, [keys, destinations]));
            for (let i = 0; i < 64; i++) {
              ctx.assertClose(out[i], keys[63 - i], 0.5, `slot ${i} of a reversal`);
            }
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            for (const seed of [808, 24601]) {
              const keys = makeKeys(ctx.utils, 64, 256, seed);
              for (const place of [1, 16]) {
                const destinations = destinationsOf(keys, place, startsOf(countsOf(keys, place)));
                const out = plain(call(ctx.kernel, [keys, destinations]));
                const expected = gatherBy(keys, destinations);
                const hint = diagnoseArray(out, expected, gatherProbes(keys, destinations));
                for (let i = 0; i < 64; i++) {
                  ctx.assertClose(out[i], expected[i], 0.5,
                    hint || `slot ${i} (seed ${seed}, place ${place})`);
                }
              }
            }
          },
        },
        {
          name: 'private test #2',
          run: async ctx => {
            // One pass at place 1 must leave the array sorted by its low digit,
            // with equal digits still in their original order.
            const keys = makeKeys(ctx.utils, 64, 256, 1729);
            const destinations = destinationsOf(keys, 1, startsOf(countsOf(keys, 1)));
            const out = plain(call(ctx.kernel, [keys, destinations]));
            for (let i = 1; i < 64; i++) {
              ctx.assert(
                digitOf(out[i - 1], 1) <= digitOf(out[i], 1),
                `slot ${i - 1} holds digit ${digitOf(out[i - 1], 1)} and slot ${i} holds ` +
                  `${digitOf(out[i], 1)} — after the gather the array must be ordered by digit`
              );
            }
          },
        },
      ],
    },

    // ------------------------------------------------------------------ 5
    {
      slug: 'full-sort',
      title: 'The Whole Sort',
      intro: `<p>Assemble it. 1,024 keys, all below 4,096 — three hex digits, so
        <strong>three passes</strong>. Each pass is the four kernels you have already written:
        histogram the digit, scan the counts into starting offsets, compute every element's
        destination, gather. The gathered array is the next pass's input.</p>
        <p>Only one piece is left: the destination rule at radix 16. It is task 1's stable rank
        with a bucket offset in front of it — <code>starts[digit]</code> puts you at the head of
        your bucket, and counting the earlier elements that share your digit places you inside it.
        Stability is still the whole game, and now you can see why: the last pass sorts by the
        <em>most</em> significant digit, and everything the first two passes achieved survives only
        inside its ties.</p>
        <p>Two things the driver must get right, both of them silent when they are wrong: the
        passes go <strong>least significant digit first</strong>, and the histogram and offsets are
        recomputed <em>every</em> pass — each one looks at a different digit of a differently
        ordered array.</p>`,
      goal: `<strong>Goal:</strong> finish the <code>destinations</code> kernel and drive three
        passes over <code>keys</code>, then log the smallest, middle and largest of the result.`,
      requirements: [
        '<code>destinations</code> returns <code>starts[digit] + </code> how many earlier elements share that digit',
        'Three passes with <code>place</code> = <code>1</code>, then <code>16</code>, then <code>256</code>',
        'Recompute the histogram and the offsets inside the loop — once per pass',
        '<code>console.log</code> the sorted array\'s first, middle and last values',
      ],
      hints: [
        {
          title: 'Hint 1 — the destination rule',
          body: `<p>Two halves. <code>starts[mine]</code> is where your bucket begins; the loop
            counts your rank inside it, exactly as in task 1 but restricted to your own digit:</p>
<pre><code>if (d === mine &amp;&amp; j &lt; this.thread.x) rank++;</code></pre>`,
        },
        {
          title: 'Hint 2 — the driver',
          body: `<pre><code>for (let place = 1; place &lt;= 256; place *= 16) {
  const counts = histogram(values, place);
  const starts = offsets(counts);
  const dest = destinations(values, place, starts);
  values = gather(values, dest);
}</code></pre>
<p>Three iterations, and every line of it inside the loop.</p>`,
        },
        {
          title: 'Hint 3 — why low digit first',
          body: `<p>Each pass makes its own digit the primary sort key and demotes everything the
            previous passes did to a tie-break. So the digit you want to dominate — the most
            significant one — has to be sorted <em>last</em>. Run the passes the other way and the
            array comes out ordered by its ones digit.</p>`,
        },
      ],
      transfer: `This is the shape of the real thing. <code>cub::DeviceRadixSort</code>,
        <code>thrust::sort</code> on integers, rocPRIM, and the Vulkan/WebGPU sort libraries all
        run this loop: per-pass digit histogram, scan to global offsets, stable scatter, repeat
        for as many digits as the key has. They beat this version on the two lines you did not
        write — the rank inside a bucket comes from a parallel scan instead of an O(n) sweep, and
        the move is a scatter into shared memory rather than a search — but the algorithm on the
        page is the algorithm they run.`,
      starterCode: `// Four kernels, three passes. Only the destination rule is missing.
const gpu = new GPU({ mode });

const histogram = gpu.createKernel(function (keys, place) {
  let count = 0;
  for (let i = 0; i < this.constants.n; i++) {
    const d = Math.floor(keys[i] / place) % this.constants.radix;
    if (d === this.thread.x) {
      count++;
    }
  }
  return count;
}, { output: [16], constants: { n: 1024, radix: 16 } });

const offsets = gpu.createKernel(function (counts) {
  let start = 0;
  for (let b = 0; b < this.constants.radix; b++) {
    if (b < this.thread.x) {
      start += counts[b];
    }
  }
  return start;
}, { output: [16], constants: { radix: 16 } });

const destinations = gpu.createKernel(function (keys, place, starts) {
  // TODO: your digit is Math.floor(keys[this.thread.x] / place) % this.constants.radix.
  // Return starts[digit], plus how many EARLIER elements carry the same digit.
  return 0;
}, { output: [1024], constants: { n: 1024, radix: 16 } });

const gather = gpu.createKernel(function (keys, dest) {
  let value = 0;
  for (let i = 0; i < this.constants.n; i++) {
    if (dest[i] === this.thread.x) {
      value = keys[i];
    }
  }
  return value;
}, { output: [1024], constants: { n: 1024 } });

// gpu.js locks an argument's type on a kernel's first call, and every kernel
// here is fed another kernel's output — so the chain starts as a Float32Array.
let values = Float32Array.from(keys);

// TODO: three passes, least significant digit first — place = 1, then 16,
// then 256. Each pass: counts → starts → destinations → gather, and the
// gathered array becomes the next pass's input.

console.log('smallest:', values[0], '| middle:', values[512], '| largest:', values[1023]);
`,
      solutionCode: `// Four kernels, three passes. Only the destination rule is missing.
const gpu = new GPU({ mode });

const histogram = gpu.createKernel(function (keys, place) {
  let count = 0;
  for (let i = 0; i < this.constants.n; i++) {
    const d = Math.floor(keys[i] / place) % this.constants.radix;
    if (d === this.thread.x) {
      count++;
    }
  }
  return count;
}, { output: [16], constants: { n: 1024, radix: 16 } });

const offsets = gpu.createKernel(function (counts) {
  let start = 0;
  for (let b = 0; b < this.constants.radix; b++) {
    if (b < this.thread.x) {
      start += counts[b];
    }
  }
  return start;
}, { output: [16], constants: { radix: 16 } });

const destinations = gpu.createKernel(function (keys, place, starts) {
  const mine = Math.floor(keys[this.thread.x] / place) % this.constants.radix;
  let rank = 0;
  for (let j = 0; j < this.constants.n; j++) {
    const d = Math.floor(keys[j] / place) % this.constants.radix;
    if (d === mine && j < this.thread.x) {
      rank++;
    }
  }
  return starts[mine] + rank;
}, { output: [1024], constants: { n: 1024, radix: 16 } });

const gather = gpu.createKernel(function (keys, dest) {
  let value = 0;
  for (let i = 0; i < this.constants.n; i++) {
    if (dest[i] === this.thread.x) {
      value = keys[i];
    }
  }
  return value;
}, { output: [1024], constants: { n: 1024 } });

// gpu.js locks an argument's type on a kernel's first call, and every kernel
// here is fed another kernel's output — so the chain starts as a Float32Array.
let values = Float32Array.from(keys);

for (let place = 1; place <= 256; place *= 16) {
  const counts = histogram(values, place);
  const starts = offsets(counts);
  const dest = destinations(values, place, starts);
  values = gather(values, dest);
}

console.log('smallest:', values[0], '| middle:', values[512], '| largest:', values[1023]);
`,
      inputs: utils => ({ keys: makeKeys(utils, 1024, 4096, 7717) }),
      publicTests: [
        {
          name: 'four kernels: histogram, offsets, destinations, gather',
          run: async ctx => {
            ctx.assert(
              findKernel(ctx, 2, 16),
              'no two-argument kernel with output [16] found — the histogram is missing'
            );
            ctx.assert(
              findKernel(ctx, 1, 16),
              'no one-argument kernel with output [16] found — the offsets scan is missing'
            );
            ctx.assert(
              findKernel(ctx, 3, 1024),
              'no three-argument kernel with output [1024] found — that is the destinations kernel'
            );
            ctx.assert(
              findKernel(ctx, 2, 1024),
              'no two-argument kernel with output [1024] found — the gather is missing'
            );
          },
        },
        {
          name: 'the destination rule: <code>starts[digit]</code> plus your rank inside the bucket',
          run: async ctx => {
            const dest = findKernel(ctx, 3, 1024);
            ctx.assert(dest, 'no three-argument destinations kernel found');
            const keys = makeKeys(ctx.utils, 1024, 4096, 7717);
            for (const place of [1, 256]) {
              const starts = startsOf(countsOf(keys, place));
              const out = plain(call(dest, [Float32Array.from(keys), place, Float32Array.from(starts)]));
              const expected = destinationsOf(keys, place, starts);
              const hint = diagnoseArray(out, expected, destinationProbes(keys, place, starts));
              for (let i = 0; i < 1024; i++) {
                ctx.assertClose(out[i], expected[i], 0.5, hint || `element ${i} at place ${place}`);
              }
            }
          },
        },
        {
          name: 'the four kernels, driven three passes, sort a fresh array',
          run: async ctx => {
            const hist = findKernel(ctx, 2, 16);
            const scan = findKernel(ctx, 1, 16);
            const dest = findKernel(ctx, 3, 1024);
            const move = findKernel(ctx, 2, 1024);
            ctx.assert(hist && scan && dest && move, 'expected all four kernels');
            const keys = makeKeys(ctx.utils, 1024, 4096, 31337);
            let values = Float32Array.from(keys);
            for (let place = 1; place <= 256; place *= 16) {
              values = call(move, [values, call(dest, [values, place, call(scan, [call(hist, [values, place])])])]);
            }
            const got = plain(values);
            const expected = ascending(keys);
            const hint = diagnoseArray(got, expected, [
              [radixDrive(keys, [1, RADIX], false),
                'the pipeline only ordered the low eight bits — with the driver fixed this is what ' +
                'three passes has to produce'],
            ]);
            for (let i = 0; i < 1024; i++) {
              ctx.assertClose(got[i], expected[i], 0.5, hint || `slot ${i} of the sorted array`);
            }
          },
        },
        {
          name: 'the finished array\'s first, middle and last values are logged',
          run: async ctx => {
            const keys = makeKeys(ctx.utils, 1024, 4096, 7717);
            const correct = landmarks(ascending(keys));
            const nums = loggedNumbers(ctx.logs);
            const missing = correct.filter(v => !hasNumber(nums, v));
            ctx.assert(
              missing.length === 0,
              driverHint(nums, correct, driverCandidates(keys)) ||
                `log the sorted array's smallest, middle and largest — expected ` +
                  `${correct.join(', ')} in the console output`
            );
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            const hist = findKernel(ctx, 2, 16);
            const scan = findKernel(ctx, 1, 16);
            const dest = findKernel(ctx, 3, 1024);
            const move = findKernel(ctx, 2, 1024);
            ctx.assert(hist && scan && dest && move, 'expected all four kernels');
            const keys = makeKeys(ctx.utils, 1024, 4096, 606);
            let values = Float32Array.from(keys);
            for (let place = 1; place <= 256; place *= 16) {
              values = call(move, [values, call(dest, [values, place, call(scan, [call(hist, [values, place])])])]);
            }
            const got = plain(values);
            const expected = ascending(keys);
            for (let i = 0; i < 1024; i++) {
              ctx.assertClose(got[i], expected[i], 0.5, `slot ${i} of the sorted array`);
            }
          },
        },
        {
          name: 'private test #2',
          run: async ctx => {
            // Every key sharing a digit somewhere: 1024 keys drawn from 32
            // distinct values, so all three passes are dense with ties.
            const dest = findKernel(ctx, 3, 1024);
            ctx.assert(dest, 'expected a destinations kernel');
            const keys = new Array(1024);
            for (let i = 0; i < 1024; i++) keys[i] = ((i * 37) % 32) * 129;
            for (const place of [1, 16, 256]) {
              const starts = startsOf(countsOf(keys, place));
              const out = plain(call(dest, [Float32Array.from(keys), place, Float32Array.from(starts)]));
              const expected = destinationsOf(keys, place, starts);
              const hint = diagnoseArray(out, expected, destinationProbes(keys, place, starts));
              for (let i = 0; i < 1024; i++) {
                ctx.assertClose(out[i], expected[i], 0.5, hint || `element ${i} at place ${place}`);
              }
            }
          },
        },
      ],
    },

    // ------------------------------------------------------------------ 6
    {
      slug: 'signed-keys',
      title: 'Keys That Aren\'t Plain Integers',
      intro: `<p>The sort has a requirement it never had to say out loud: the key must be a
        <strong>non-negative integer</strong>, because <code>Math.floor(key / place) % 16</code>
        is only a digit for those. Hand it <code>−5</code> and the "digit" is <code>−5</code>,
        <code>starts[−5]</code> is off the front of the bucket table, and the pass returns junk.</p>
        <p>Signed integers have a clean fix that costs one map each way: <strong>bias</strong>
        them. Add 2,048 and the range −2048…2047 becomes 0…4095 — same order, all non-negative.
        Sort, then subtract the 2,048 back off. Production libraries call this step encoding the
        key, and the rule is the only one that matters: any order-preserving, invertible map into
        the unsigned integers makes radix sort work on your type.</p>
        <p>Floats are the same idea and a harder map — and this is where gpu.js stops. A float's
        ordering <em>is</em> its bit pattern's ordering, for positives; IEEE-754 negatives carry a
        sign bit on top and sort backwards under an unsigned comparison, so real implementations
        reinterpret the 32 bits and flip them (<code>x ^ 0x80000000</code> for a positive,
        <code>~x</code> for a negative) before sorting and flip back after. gpu.js does have
        <code>&amp;</code>, <code>|</code>, <code>^</code>, <code>&lt;&lt;</code> and
        <code>&gt;&gt;</code> inside kernels, but the WebGL backend emulates them with GLSL integer
        loops and they part company with JavaScript the moment an operand goes negative
        (<code>-8 &amp; 15</code> is 8 in JavaScript and on the CPU backend, and 0 on WebGL). More
        to the point, there is no way to <em>see</em> a float's bits at all: GLSL ES 1.00 has no
        <code>floatBitsToInt</code> and gpu.js exposes none, so <code>key &amp; 15</code> truncates
        the value to an integer first — <code>3.5 &amp; 15</code> is 3, the number's integer part,
        never its bit pattern. This course therefore sorts non-negative integer keys, and signed
        ones through the bias below; a CUDA or WebGPU implementation runs the same six kernels with
        a bit-flipping encoder in front.</p>`,
      goal: `<strong>Goal:</strong> sort <code>readings</code>, which run from −2048 to 2047, by
        biasing them into non-negative integers, sorting, and taking the bias back off.`,
      requirements: [
        '<code>encode</code> adds <code>this.constants.bias</code> to every reading',
        '<code>decode</code> subtracts it again',
        'Run the given <code>radixSort</code> on the <em>encoded</em> values, and decode the result',
        '<code>console.log</code> the sorted readings\' smallest and largest values',
      ],
      hints: [
        {
          title: 'Hint 1 — the two maps',
          body: `<p>Both kernels are one-line maps over their own cell — one adds
            <code>this.constants.bias</code>, the other subtracts it. Nothing about the sort
            changes.</p>`,
        },
        {
          title: 'Hint 2 — the wiring',
          body: `<pre><code>const sorted = decode(radixSort(encode(readings)));</code></pre>
<p>Encode on the way in, decode on the way out. Miss the decode and every value
            comes back 2,048 too high; miss the encode and the negative keys index off the front
            of the bucket table.</p>`,
        },
      ],
      transfer: `Every serious sorting library has this seam. CUB twiddles a key's bits in and out
        around the sort so that floats, signed integers and custom types all reduce to unsigned
        digits; rocPRIM and Thrust do the same, and newer CUB versions let you hand it a
        <em>decomposer</em> for your own struct. The sort never changes — only the map into
        unsigned integers does.`,
      starterCode: `// The sort below is finished. It only accepts non-negative integer keys.
const gpu = new GPU({ mode });

const encode = gpu.createKernel(function (v) {
  // TODO: shift every reading up so the smallest one becomes 0
  return v[this.thread.x];
}, { output: [256], constants: { bias: 2048 } });

const decode = gpu.createKernel(function (v) {
  // TODO: undo the shift
  return v[this.thread.x];
}, { output: [256], constants: { bias: 2048 } });

const histogram = gpu.createKernel(function (keys, place) {
  let count = 0;
  for (let i = 0; i < this.constants.n; i++) {
    const d = Math.floor(keys[i] / place) % this.constants.radix;
    if (d === this.thread.x) {
      count++;
    }
  }
  return count;
}, { output: [16], constants: { n: 256, radix: 16 } });

const offsets = gpu.createKernel(function (counts) {
  let start = 0;
  for (let b = 0; b < this.constants.radix; b++) {
    if (b < this.thread.x) {
      start += counts[b];
    }
  }
  return start;
}, { output: [16], constants: { radix: 16 } });

const destinations = gpu.createKernel(function (keys, place, starts) {
  const mine = Math.floor(keys[this.thread.x] / place) % this.constants.radix;
  let rank = 0;
  for (let j = 0; j < this.constants.n; j++) {
    const d = Math.floor(keys[j] / place) % this.constants.radix;
    if (d === mine && j < this.thread.x) {
      rank++;
    }
  }
  return starts[mine] + rank;
}, { output: [256], constants: { n: 256, radix: 16 } });

const gather = gpu.createKernel(function (keys, dest) {
  let value = 0;
  for (let i = 0; i < this.constants.n; i++) {
    if (dest[i] === this.thread.x) {
      value = keys[i];
    }
  }
  return value;
}, { output: [256], constants: { n: 256 } });

function radixSort(values) {
  // gpu.js locks an argument's type on a kernel's first call, and every pass
  // feeds one kernel's output into the next — so the chain starts as a
  // Float32Array whatever it was handed.
  let v = Float32Array.from(values);
  for (let place = 1; place <= 256; place *= 16) {
    v = gather(v, destinations(v, place, offsets(histogram(v, place))));
  }
  return v;
}

// TODO: bias the readings on the way in, and take the bias off on the way out.
const sorted = radixSort(readings);
console.log('smallest:', sorted[0], '| largest:', sorted[255]);
`,
      solutionCode: `// The sort below is finished. It only accepts non-negative integer keys.
const gpu = new GPU({ mode });

const encode = gpu.createKernel(function (v) {
  return v[this.thread.x] + this.constants.bias;
}, { output: [256], constants: { bias: 2048 } });

const decode = gpu.createKernel(function (v) {
  return v[this.thread.x] - this.constants.bias;
}, { output: [256], constants: { bias: 2048 } });

const histogram = gpu.createKernel(function (keys, place) {
  let count = 0;
  for (let i = 0; i < this.constants.n; i++) {
    const d = Math.floor(keys[i] / place) % this.constants.radix;
    if (d === this.thread.x) {
      count++;
    }
  }
  return count;
}, { output: [16], constants: { n: 256, radix: 16 } });

const offsets = gpu.createKernel(function (counts) {
  let start = 0;
  for (let b = 0; b < this.constants.radix; b++) {
    if (b < this.thread.x) {
      start += counts[b];
    }
  }
  return start;
}, { output: [16], constants: { radix: 16 } });

const destinations = gpu.createKernel(function (keys, place, starts) {
  const mine = Math.floor(keys[this.thread.x] / place) % this.constants.radix;
  let rank = 0;
  for (let j = 0; j < this.constants.n; j++) {
    const d = Math.floor(keys[j] / place) % this.constants.radix;
    if (d === mine && j < this.thread.x) {
      rank++;
    }
  }
  return starts[mine] + rank;
}, { output: [256], constants: { n: 256, radix: 16 } });

const gather = gpu.createKernel(function (keys, dest) {
  let value = 0;
  for (let i = 0; i < this.constants.n; i++) {
    if (dest[i] === this.thread.x) {
      value = keys[i];
    }
  }
  return value;
}, { output: [256], constants: { n: 256 } });

function radixSort(values) {
  // gpu.js locks an argument's type on a kernel's first call, and every pass
  // feeds one kernel's output into the next — so the chain starts as a
  // Float32Array whatever it was handed.
  let v = Float32Array.from(values);
  for (let place = 1; place <= 256; place *= 16) {
    v = gather(v, destinations(v, place, offsets(histogram(v, place))));
  }
  return v;
}

const sorted = decode(radixSort(encode(readings)));
console.log('smallest:', sorted[0], '| largest:', sorted[255]);
`,
      inputs: utils => ({ readings: makeReadings(utils, 256, 8088) }),
      publicTests: [
        {
          name: 'one kernel biases the readings up, one takes the bias back off',
          run: async ctx => {
            const maps = ctx.kernels.filter(k => {
              const built = k.kernel;
              const names = built && built.argumentNames ? built.argumentNames : [];
              return names.length === 1 && built.output && Number(built.output[0]) === 256;
            });
            ctx.assert(
              maps.length >= 2,
              `expected an encode kernel and a decode kernel (one argument, output [256]) — found ${maps.length}`
            );
            const probe = [0, 100, -100, 7];
            const results = maps.map(k => plain(call(k, [new Array(256).fill(0).map((_, i) => probe[i % 4])])));
            const up = results.find(r => Math.abs(r[0] - 2048) <= 0.5);
            const down = results.find(r => Math.abs(r[0] + 2048) <= 0.5);
            ctx.assert(
              up,
              'no kernel adds the bias — encode should return v[this.thread.x] + this.constants.bias'
            );
            ctx.assert(
              down,
              'no kernel subtracts the bias — decode should return v[this.thread.x] - this.constants.bias'
            );
            for (let i = 0; i < 8; i++) {
              ctx.assertClose(up[i], probe[i % 4] + 2048, 0.5, `encode of ${probe[i % 4]}`);
              ctx.assertClose(down[i], probe[i % 4] - 2048, 0.5, `decode of ${probe[i % 4]}`);
            }
          },
        },
        {
          name: 'the biased pipeline sorts a fresh batch of signed readings',
          run: async ctx => {
            const hist = findKernel(ctx, 2, 16);
            const scan = findKernel(ctx, 1, 16);
            const dest = findKernel(ctx, 3, 256);
            const move = findKernel(ctx, 2, 256);
            ctx.assert(hist && scan && dest && move, 'expected the four sorting kernels from the starter');
            const readings = makeReadings(ctx.utils, 256, 4321);
            const maps = ctx.kernels.filter(k => {
              const built = k.kernel;
              const names = built && built.argumentNames ? built.argumentNames : [];
              return names.length === 1 && built.output && Number(built.output[0]) === 256;
            });
            let encoded = null;
            let decoder = null;
            for (const k of maps) {
              const probe = plain(call(k, [new Array(256).fill(0)]));
              if (Math.abs(probe[0] - 2048) <= 0.5) encoded = k;
              if (Math.abs(probe[0] + 2048) <= 0.5) decoder = k;
            }
            ctx.assert(encoded && decoder, 'expected an encode kernel and a decode kernel');
            let values = call(encoded, [readings]);
            for (let place = 1; place <= 256; place *= 16) {
              values = call(move, [values, call(dest, [values, place, call(scan, [call(hist, [values, place])])])]);
            }
            const got = plain(call(decoder, [values]));
            const expected = ascending(readings);
            for (let i = 0; i < 256; i++) {
              ctx.assertClose(got[i], expected[i], 0.5, `slot ${i} of the sorted readings`);
            }
          },
        },
        {
          name: 'the sorted readings\' smallest and largest values are logged',
          run: async ctx => {
            const readings = makeReadings(ctx.utils, 256, 8088);
            const expected = ascending(readings);
            const smallest = expected[0];
            const largest = expected[255];
            const nums = loggedNumbers(ctx.logs);
            const BIAS = 2048;
            const shifted = delta =>
              hasNumber(nums, smallest + delta) && hasNumber(nums, largest + delta);
            let hint = null;
            if (shifted(BIAS)) {
              hint = 'the bias never came off — decode has to subtract exactly what encode added';
            } else if (shifted(2 * BIAS)) {
              hint = 'decode added the bias instead of subtracting it — it goes on before the sort ' +
                'and comes off after';
            } else if (shifted(-BIAS)) {
              hint = 'the bias came off twice — encode adds it, decode subtracts it, once each';
            } else if (hasNumber(nums, 0) && smallest !== 0 && largest !== 0) {
              hint = 'a 0 in the result means the sort ran on the raw readings: a negative key makes ' +
                'Math.floor(key / place) % 16 negative, which indexes off the front of the offsets ' +
                'table and leaves output slots nobody claimed. Bias the readings first';
            }
            ctx.assert(
              hasNumber(nums, smallest) && hasNumber(nums, largest),
              hint || `log the sorted readings' smallest and largest — expected ${smallest} and ${largest}`
            );
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            const hist = findKernel(ctx, 2, 16);
            const scan = findKernel(ctx, 1, 16);
            const dest = findKernel(ctx, 3, 256);
            const move = findKernel(ctx, 2, 256);
            const maps = ctx.kernels.filter(k => {
              const built = k.kernel;
              const names = built && built.argumentNames ? built.argumentNames : [];
              return names.length === 1 && built.output && Number(built.output[0]) === 256;
            });
            let encoded = null;
            let decoder = null;
            for (const k of maps) {
              const probe = plain(call(k, [new Array(256).fill(0)]));
              if (Math.abs(probe[0] - 2048) <= 0.5) encoded = k;
              if (Math.abs(probe[0] + 2048) <= 0.5) decoder = k;
            }
            ctx.assert(hist && scan && dest && move && encoded && decoder,
              'expected the four sorting kernels plus encode and decode');
            for (const seed of [1234, 99991]) {
              const readings = makeReadings(ctx.utils, 256, seed);
              let values = call(encoded, [readings]);
              for (let place = 1; place <= 256; place *= 16) {
                values = call(move, [values, call(dest, [values, place, call(scan, [call(hist, [values, place])])])]);
              }
              const got = plain(call(decoder, [values]));
              const expected = ascending(readings);
              for (let i = 0; i < 256; i++) {
                ctx.assertClose(got[i], expected[i], 0.5, `slot ${i} (seed ${seed})`);
              }
              ctx.assert(got[0] < 0, 'the smallest reading is negative — the decode has to bring it back below zero');
            }
          },
        },
      ],
    },
  ],
};
