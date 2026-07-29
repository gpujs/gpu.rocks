// Module 2.2 — Reductions.
//
// Six tasks: the one-thread baseline → strided partial sums → one rung of
// the halving ladder → riding the ladder down to a scalar → min/max by
// swapping the fold operator → a fused mean + RMS pipeline as the payoff.
//
// Kernel-authoring rules (contract): no closures inside kernel functions,
// loop bounds come from this.constants (compile-time known), shrinking
// kernels need dynamicOutput/dynamicArguments, and every task passes in
// CPU mode. Array sizes stay ≤ 4096 so verification is fast.

// Deterministic values in 0–2 (3 decimal places) shared by inputs() and tests.
function makeValues(utils, n, seed = 2207) {
  const rand = utils.seededRandom(seed);
  const data = new Array(n);
  for (let i = 0; i < n; i++) data[i] = Math.round(rand() * 2000) / 1000;
  return data;
}

function sumOf(arr) {
  let total = 0;
  for (let i = 0; i < arr.length; i++) total += arr[i];
  return total;
}

function minOf(arr) {
  let m = arr[0];
  for (let i = 1; i < arr.length; i++) if (arr[i] < m) m = arr[i];
  return m;
}

function maxOf(arr) {
  let m = arr[0];
  for (let i = 1; i < arr.length; i++) if (arr[i] > m) m = arr[i];
  return m;
}

// Reference for the strided partial-sum pattern: thread x sums
// arr[x], arr[x + threads], arr[x + 2 * threads], …
function stridedPartial(arr, x, threads, chunk) {
  let s = 0;
  for (let i = 0; i < chunk; i++) s += arr[i * threads + x];
  return s;
}

// Drive a dynamic halving-ladder kernel from JS: n → n/2 → … → 1.
// gpu.js locks an argument's type on the kernel's first invocation, so the
// ladder must see Float32Arrays throughout — the same type its rungs return.
function runLadder(step, values) {
  let v = values instanceof Float32Array ? values : Float32Array.from(values);
  let n = v.length;
  while (n > 1) {
    n = n / 2;
    step.setOutput([n]);
    v = step(v);
  }
  return v[0];
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

// The kernel created with dynamicOutput: true (the ladder rung).
function findDynamicKernel(ctx) {
  return ctx.kernels.find(k => k.kernel && k.kernel.dynamicOutput) || null;
}

// Identify the two fixed partial-sum kernels by behaviour: on an all-2s input
// each partialSums thread yields 64 · 2 = 128, each partialSquares thread
// 64 · 4 = 256. The dynamic ladder kernel is excluded by its flag.
function findPartialKernels(ctx) {
  const twos = new Array(4096).fill(2);
  let sums = null;
  let squares = null;
  for (const k of ctx.kernels) {
    if (!k.kernel || k.kernel.dynamicOutput) continue;
    let out;
    try {
      out = k(twos);
    } catch (e) {
      continue;
    }
    if (!out || out.length !== 64) continue;
    if (Math.abs(out[0] - 128) <= 0.01) sums = k;
    else if (Math.abs(out[0] - 256) <= 0.01) squares = k;
  }
  return { sums, squares };
}

export default {
  id: '2-2',
  track: 2,
  title: 'Reductions',
  blurb: 'Sum, min, max and mean over millions of values — the ladder pattern every platform uses.',
  tasks: [
    // ------------------------------------------------------------------ 1
    {
      slug: 'one-thread-sum',
      title: 'The One-Thread Trap',
      intro: `<p>Meet the <strong>reduction</strong>: many values in, one value out — sum, min,
        max, mean. It's the awkward case in GPU land, because a kernel thread writes exactly
        <em>one</em> output cell. 4,096 inputs collapsing to 1 output means
        <code>output: [1]</code>… a single thread.</p>
        <p>You <em>can</em> do it — kernels may loop, as long as the bound is known at compile
        time, which is exactly what <code>this.constants</code> is for. But one thread grinding
        through 4,096 additions while thousands of its neighbours sit idle is the slowest
        possible way to use a GPU. Write it anyway: it's the baseline the rest of this module
        tears down.</p>`,
      goal: `<strong>Goal:</strong> make the single thread loop over all of <code>data</code>
        (bound: <code>this.constants.n</code>) and return the total.`,
      requirements: [
        'Keep <code>output: [1]</code> — one thread owns the one output cell',
        'Loop <code>for (let i = 0; i &lt; this.constants.n; i++)</code> — in gpu.js\'s WebGL backend, loop bounds must be compile-time constants',
        'Accumulate into a local <code>let sum</code> and return it',
      ],
      hints: [
        {
          title: 'Hint 1 — an accumulator',
          body: `<p>Declare <code>let sum = 0;</code> before the loop, add to it inside the loop,
            and <code>return sum;</code> after. Plain JavaScript — the transpiler handles it.</p>`,
        },
        {
          title: 'Hint 2 — the loop body',
          body: `<p>One statement: <code>sum += data[i];</code></p>`,
        },
      ],
      transfer: `This wall exists on every platform: a single CUDA thread summing a whole buffer
        is the textbook example of what <em>not</em> to do, and a naive WebGPU compute shader
        with one invocation hits it just the same. Everyone's escape route is the trick you
        build next — split the work, then combine.`,
      starterCode: `// 4096 values, ONE output cell — so exactly one thread does everything.
const gpu = new GPU({ mode });

const sumAll = gpu.createKernel(function (data) {
  // TODO: loop i from 0 to this.constants.n, accumulate data[i]
  // into a local sum, and return it.
  return 0;
}, {
  output: [1],
  constants: { n: 4096 },
});

console.log('total:', sumAll(data)[0]);
`,
      solutionCode: `// 4096 values, ONE output cell — so exactly one thread does everything.
const gpu = new GPU({ mode });

const sumAll = gpu.createKernel(function (data) {
  let sum = 0;
  for (let i = 0; i < this.constants.n; i++) {
    sum += data[i];
  }
  return sum;
}, {
  output: [1],
  constants: { n: 4096 },
});

console.log('total:', sumAll(data)[0]);
`,
      inputs: utils => ({ data: makeValues(utils, 4096) }),
      publicTests: [
        {
          name: 'one output cell holds the sum of 4096 ones',
          run: async ctx => {
            ctx.assert(ctx.kernels.length >= 1, 'no kernel was created — call gpu.createKernel()');
            const out = ctx.kernel(new Array(4096).fill(1));
            ctx.assert(out && out.length === 1, `expected 1 output value, got ${out && out.length}`);
            ctx.assertClose(out[0], 4096, 0.5, 'sum of 4096 ones');
          },
        },
        {
          name: 'the total matches on fresh data',
          run: async ctx => {
            const arr = new Array(4096);
            for (let i = 0; i < 4096; i++) arr[i] = ((i * 7) % 13) * 0.125;
            const out = ctx.kernel(arr);
            ctx.assertClose(out[0], sumOf(arr), 2, 'the total');
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            const data = makeValues(ctx.utils, 4096, 4242);
            const out = ctx.kernel(data);
            ctx.assert(out && out.length === 1, 'expected 1 output value');
            ctx.assertClose(out[0], sumOf(data), 2, 'the total');
          },
        },
      ],
    },

    // ------------------------------------------------------------------ 2
    {
      slug: 'partial-sums',
      title: 'Partial Sums: Divide the Work',
      intro: `<p>The fix: give <em>every</em> thread a slice. 64 threads, each summing 64 of the
        4,096 values, produce 64 <strong>partial sums</strong> — and 64 leftover numbers are
        cheap to finish off in plain JavaScript.</p>
        <p>Watch the reading pattern, though. Thread <code>x</code> does <em>not</em> take a
        contiguous block; it reads <code>data[x]</code>, <code>data[x + 64]</code>,
        <code>data[x + 128]</code>, … — a <strong>strided</strong> walk. At every step of the
        loop, neighbouring threads touch neighbouring elements, which is exactly the access
        pattern GPU memory hardware is built to serve in one go.</p>`,
      goal: `<strong>Goal:</strong> compute 64 strided partial sums on the GPU, then total the
        64 partials in JavaScript and log the grand total.`,
      requirements: [
        'Each of the 64 threads loops <code>this.constants.chunk</code> times',
        'Strided reads: element <code>i</code> of thread <code>x</code> is <code>data[i * this.constants.threads + this.thread.x]</code>',
        'Sum the 64 returned partials in plain JavaScript and <code>console.log</code> the total',
      ],
      hints: [
        {
          title: 'Hint 1 — which elements are mine?',
          body: `<p>Thread <code>x</code> owns elements <code>x</code>, <code>x + 64</code>,
            <code>x + 128</code>, … so its <code>i</code>-th element sits at index
            <code>i * 64 + x</code>.</p>`,
        },
        {
          title: 'Hint 2 — the loop body',
          body: `<p><code>sum += data[i * this.constants.threads + this.thread.x];</code></p>`,
        },
        {
          title: 'Hint 3 — finishing in JS',
          body: `<p>After <code>const partial = partials(data);</code> a plain loop does it:
            <code>let total = 0; for (let i = 0; i &lt; partial.length; i++) total += partial[i];</code></p>`,
        },
      ],
      transfer: `This is CUDA's <em>grid-stride loop</em>, almost line for line — every serious
        reduction in CUB and Thrust starts with per-thread partials accumulated in registers,
        and coalesced (strided-by-thread-count) reads are the whole reason for the pattern.
        WebGPU and Metal compute kernels stage the same partials into workgroup/threadgroup
        memory.`,
      starterCode: `// 64 threads, 64 values each. Strided reads keep the memory hardware happy.
const gpu = new GPU({ mode });

const partials = gpu.createKernel(function (data) {
  // TODO: loop this.constants.chunk times and accumulate this thread's
  // strided slice: data[i * this.constants.threads + this.thread.x]
  return 0;
}, {
  output: [64],
  constants: { threads: 64, chunk: 64 },
});

const partial = partials(data);
console.log('partials:', partial.length);

let total = 0;
for (let i = 0; i < partial.length; i++) total += partial[i];
console.log('total:', total);
`,
      solutionCode: `// 64 threads, 64 values each. Strided reads keep the memory hardware happy.
const gpu = new GPU({ mode });

const partials = gpu.createKernel(function (data) {
  let sum = 0;
  for (let i = 0; i < this.constants.chunk; i++) {
    sum += data[i * this.constants.threads + this.thread.x];
  }
  return sum;
}, {
  output: [64],
  constants: { threads: 64, chunk: 64 },
});

const partial = partials(data);
console.log('partials:', partial.length);

let total = 0;
for (let i = 0; i < partial.length; i++) total += partial[i];
console.log('total:', total);
`,
      inputs: utils => ({ data: makeValues(utils, 4096, 707) }),
      publicTests: [
        {
          name: '64 partial sums — all-ones input gives 64 everywhere',
          run: async ctx => {
            ctx.assert(ctx.kernels.length >= 1, 'no kernel was created — call gpu.createKernel()');
            const out = ctx.kernel(new Array(4096).fill(1));
            ctx.assert(out && out.length === 64, `expected 64 partial sums, got ${out && out.length}`);
            for (let x = 0; x < 64; x++) {
              ctx.assertClose(out[x], 64, 1e-3, `partial ${x} should sum 64 ones`);
            }
          },
        },
        {
          name: 'partials are strided — thread x sums <code>data[x], data[x + 64], …</code>',
          run: async ctx => {
            const arr = new Array(4096);
            for (let i = 0; i < 4096; i++) arr[i] = i;
            const out = ctx.kernel(arr);
            for (const x of [0, 1, 31, 63]) {
              // Σ over i of (i * 64 + x) = 129024 + 64x — contiguous chunks would differ.
              ctx.assertClose(out[x], 129024 + 64 * x, 0.5,
                `partial ${x} should sum data[${x}], data[${x} + 64], data[${x} + 128], …`);
            }
          },
        },
        {
          name: 'the grand total is computed and logged',
          run: async ctx => {
            const expected = sumOf(makeValues(ctx.utils, 4096, 707));
            const nums = loggedNumbers(ctx.logs);
            ctx.assert(
              nums.some(v => Math.abs(v - expected) <= 0.5),
              `log the total of the partials — expected ≈${expected.toFixed(2)} in the console output`
            );
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            const data = makeValues(ctx.utils, 4096, 555);
            const out = ctx.kernel(data);
            ctx.assert(out && out.length === 64, 'expected 64 partial sums');
            for (let x = 0; x < 64; x++) {
              ctx.assertClose(out[x], stridedPartial(data, x, 64, 64), 0.02, `partial ${x}`);
            }
            ctx.assertClose(sumOf(Array.from(out)), sumOf(data), 0.5, 'total of the partials');
          },
        },
      ],
    },

    // ------------------------------------------------------------------ 3
    {
      slug: 'halving-step',
      title: 'One Rung of the Ladder',
      intro: `<p>Sixty-four partials finished in JavaScript is fine. A million wouldn't be. To
        stay parallel all the way down, GPUs fold an array onto itself: add each element in the
        <em>top half</em> to its partner in the <em>bottom half</em>, and 512 values become 256
        in a single parallel step. That's one rung of the <strong>halving ladder</strong> —
        every reduction library on every platform is built from this move.</p>
        <p>One kernel invocation = one rung. Each thread adds exactly one pair:
        <code>data[x] + data[x + half]</code>. And <code>half</code> comes for free — the fold
        distance is just the output length, <code>this.output.x</code>.</p>`,
      goal: `<strong>Goal:</strong> write the rung kernel — fold 512 values into 256 pair sums,
        preserving the total.`,
      requirements: [
        '<code>output: [256]</code> — one thread per pair',
        'Each thread adds its own element to its partner one output-width away',
        'The fold preserves the total: the 256 outputs sum to the same value as the 512 inputs',
      ],
      hints: [
        {
          title: 'Hint 1 — how far away is my partner?',
          body: `<p>With 512 inputs and 256 outputs, thread <code>x</code> pairs with element
            <code>x + 256</code> — and 256 is exactly <code>this.output.x</code>, the width of
            the output.</p>`,
        },
        {
          title: 'Hint 2 — the one-liner',
          body: `<p><code>return data[this.thread.x] + data[this.thread.x + this.output.x];</code></p>`,
        },
      ],
      transfer: `The halving fold is the heart of every tree reduction: CUDA's classic
        shared-memory reduction halves its stride once per barrier, and WGSL subgroup ops or
        Metal's <code>simd_sum</code> are the same fold executed inside the hardware. One rung
        here equals one barrier-separated step there.`,
      starterCode: `// Fold the top half onto the bottom half: 512 values in, 256 out.
const gpu = new GPU({ mode });

const halve = gpu.createKernel(function (data) {
  // TODO: add this thread's element to its partner in the top half.
  // The fold distance is this.output.x.
  return data[this.thread.x];
}, {
  output: [256],
});

const folded = halve(data);
console.log('folded length:', folded.length);
console.log('first pair sum:', folded[0]);
`,
      solutionCode: `// Fold the top half onto the bottom half: 512 values in, 256 out.
const gpu = new GPU({ mode });

const halve = gpu.createKernel(function (data) {
  return data[this.thread.x] + data[this.thread.x + this.output.x];
}, {
  output: [256],
});

const folded = halve(data);
console.log('folded length:', folded.length);
console.log('first pair sum:', folded[0]);
`,
      inputs: utils => ({ data: makeValues(utils, 512, 1131) }),
      publicTests: [
        {
          name: 'one rung: 512 values fold to 256',
          run: async ctx => {
            ctx.assert(ctx.kernels.length >= 1, 'no kernel was created — call gpu.createKernel()');
            const out = ctx.kernel(makeValues(ctx.utils, 512, 1131));
            ctx.assert(out && out.length === 256, `expected 256 values after the fold, got ${out && out.length}`);
          },
        },
        {
          name: 'cell x holds <code>data[x] + data[x + 256]</code>',
          run: async ctx => {
            const arr = new Array(512);
            for (let i = 0; i < 512; i++) arr[i] = i;
            const out = ctx.kernel(arr);
            for (let x = 0; x < 256; x++) {
              ctx.assertClose(out[x], 2 * x + 256, 1e-3, `cell ${x}`);
            }
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            const data = makeValues(ctx.utils, 512, 9091);
            const out = ctx.kernel(data);
            ctx.assert(out && out.length === 256, 'expected 256 values after the fold');
            for (let x = 0; x < 256; x++) {
              ctx.assertClose(out[x], data[x] + data[x + 256], 1e-3, `cell ${x}`);
            }
            ctx.assertClose(sumOf(Array.from(out)), sumOf(data), 0.05, 'the fold must preserve the total');
          },
        },
      ],
    },

    // ------------------------------------------------------------------ 4
    {
      slug: 'ladder-to-scalar',
      title: 'Ride the Ladder Down',
      intro: `<p>Now ride it all the way: 1,024 → 512 → 256 → … → 1. Ten rungs and the array is
        a scalar. That means the <em>same</em> kernel has to run at a different size on every
        call — two options make that legal: <code>dynamicOutput: true</code> lets
        <code>setOutput()</code> shrink the thread grid between calls, and
        <code>dynamicArguments: true</code> lets the input shrink with it.</p>
        <p>The driving loop lives in JavaScript, but every rung of actual work stays parallel
        on the GPU: log₂(1024) = 10 launches instead of 1,023 serial additions. One real-world
        wrinkle, already wired into the driver: gpu.js locks an argument's <em>type</em> on the
        kernel's first call, so the ladder starts from a <code>Float32Array</code> — the same
        type every rung's output comes back as.</p>`,
      goal: `<strong>Goal:</strong> reduce the 1,024 values of <code>data</code> to a single
        total by iterating the halving rung, and log the result.`,
      requirements: [
        'Create the rung kernel with <code>dynamicOutput: true</code> and <code>dynamicArguments: true</code>',
        'Fold pairs with <code>this.output.x</code>, exactly like the last task',
        'Loop in JS: while <code>n &gt; 1</code>, halve <code>n</code>, <code>setOutput([n])</code>, re-invoke',
        '<code>console.log</code> the final scalar',
      ],
      hints: [
        {
          title: 'Hint 1 — resizing a kernel',
          body: `<p><code>halve.setOutput([n])</code> takes the new output shape as an array.
            Call it before each invocation, with <code>n</code> already halved.</p>`,
        },
        {
          title: 'Hint 2 — the driver skeleton',
          body: `<p><code>let n = values.length; while (n &gt; 1) { n = n / 2; … }</code> —
            inside the loop, resize, re-invoke, and keep the returned array for the next
            rung.</p>`,
        },
        {
          title: 'Hint 3 — the full driver',
          body: `<p><code>while (n &gt; 1) { n = n / 2; halve.setOutput([n]);
            values = halve(values); }</code> — then the answer is <code>values[0]</code>.</p>`,
        },
      ],
      transfer: `Multi-pass reduction is the production pattern everywhere: CUDA launches a
        shrinking sequence of grids (or grid-syncs with cooperative groups), WebGPU records
        repeated dispatches ping-ponging between two buffers, Metal encodes one compute pass
        per rung. The log₂(n) staircase is identical on all of them.`,
      starterCode: `// Same rung as before — but dynamic, so it can shrink call by call.
const gpu = new GPU({ mode });

const halve = gpu.createKernel(function (data) {
  // TODO: fold this thread's pair, exactly like the last task
  return data[this.thread.x];
}, {
  dynamicOutput: true,
  dynamicArguments: true,
});

// Start from a Float32Array: gpu.js locks an argument's type on the first
// call, and every rung's output comes back as a Float32Array.
let values = Float32Array.from(data);
let n = values.length;
while (n > 1) {
  n = n / 2;
  halve.setOutput([n]);
  values = halve(values);
}
console.log('total:', values[0]);
`,
      solutionCode: `// Same rung as before — but dynamic, so it can shrink call by call.
const gpu = new GPU({ mode });

const halve = gpu.createKernel(function (data) {
  return data[this.thread.x] + data[this.thread.x + this.output.x];
}, {
  dynamicOutput: true,
  dynamicArguments: true,
});

// Start from a Float32Array: gpu.js locks an argument's type on the first
// call, and every rung's output comes back as a Float32Array.
let values = Float32Array.from(data);
let n = values.length;
while (n > 1) {
  n = n / 2;
  halve.setOutput([n]);
  values = halve(values);
}
console.log('total:', values[0]);
`,
      inputs: utils => ({ data: makeValues(utils, 1024, 2024) }),
      publicTests: [
        {
          name: 'the rung is dynamic and folds pairs',
          run: async ctx => {
            const halve = findDynamicKernel(ctx);
            ctx.assert(halve, 'no kernel with dynamicOutput: true found — pass it in the kernel options');
            halve.setOutput([2]);
            const out = halve(Float32Array.from([1, 2, 3, 4]));
            ctx.assert(out && out.length === 2, `expected 2 values after one rung, got ${out && out.length}`);
            ctx.assertClose(out[0], 4, 1e-3, 'cell 0 should fold data[0] + data[2]');
            ctx.assertClose(out[1], 6, 1e-3, 'cell 1 should fold data[1] + data[3]');
          },
        },
        {
          name: 'the ladder reduces 1024 fresh values to their sum',
          run: async ctx => {
            const halve = findDynamicKernel(ctx);
            ctx.assert(halve, 'no kernel with dynamicOutput: true found');
            const arr = new Array(1024);
            for (let i = 0; i < 1024; i++) arr[i] = (i % 10) * 0.25;
            ctx.assertClose(runLadder(halve, arr), sumOf(arr), 0.1, 'the ladder total');
          },
        },
        {
          name: 'the final scalar is logged',
          run: async ctx => {
            const expected = sumOf(makeValues(ctx.utils, 1024, 2024));
            const nums = loggedNumbers(ctx.logs);
            ctx.assert(
              nums.some(v => Math.abs(v - expected) <= 0.5),
              `log the final total — expected ≈${expected.toFixed(2)} in the console output`
            );
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            const halve = findDynamicKernel(ctx);
            ctx.assert(halve, 'expected a dynamicOutput kernel');
            const data = makeValues(ctx.utils, 256, 40961);
            ctx.assertClose(runLadder(halve, data), sumOf(data), 0.1, 'ladder total on 256 values');
          },
        },
      ],
    },

    // ------------------------------------------------------------------ 5
    {
      slug: 'min-max',
      title: 'Min and Max: Change the Operator',
      intro: `<p>Here's the secret hiding inside the ladder: nothing about it is really about
        <em>addition</em>. Any operation that combines two values and doesn't care about order
        or grouping — associative and commutative — can ride the same ladder. Swap
        <code>+</code> for <code>Math.min</code> and the scalar at the bottom is the smallest
        value in the array. <code>Math.max</code> gives the largest.</p>
        <p>Two kernels, one driver. The structure doesn't change at all — only the fold
        rule.</p>`,
      goal: `<strong>Goal:</strong> find both the minimum and the maximum of <code>data</code>
        with two halving-ladder kernels, and log both.`,
      requirements: [
        '<code>minStep</code> folds with <code>Math.min</code>, <code>maxStep</code> with <code>Math.max</code>',
        'Both kernels use <code>dynamicOutput: true</code> and <code>dynamicArguments: true</code>',
        'Ride each ladder down to a scalar and <code>console.log</code> both results',
      ],
      hints: [
        {
          title: 'Hint 1 — Math inside kernels',
          body: `<p><code>Math.min(a, b)</code> and <code>Math.max(a, b)</code> both work inside
            kernel functions. The fold becomes
            <code>Math.min(data[this.thread.x], data[this.thread.x + this.output.x])</code>.</p>`,
        },
        {
          title: 'Hint 2 — one driver, two ladders',
          body: `<p>Wrap last task's while-loop in a plain JS function that takes the kernel as
            a parameter — <code>reduce(minStep, data)</code>, <code>reduce(maxStep, data)</code>
            — instead of writing it twice.</p>`,
        },
      ],
      transfer: `Pluggable operators are why every library ships reduce as a higher-order
        function: <code>thrust::reduce</code> and ROCm's rocPRIM accept any binary op plus an
        identity value, Metal Performance Shaders sells min/max reductions pre-built, and
        WGSL's <code>subgroupMin</code>/<code>subgroupMax</code> are this exact ladder burned
        into silicon.`,
      starterCode: `// Same ladder, new fold rule. Only the operator changes.
const gpu = new GPU({ mode });

const minStep = gpu.createKernel(function (data) {
  // TODO: keep the SMALLER of the pair, not the sum
  return data[this.thread.x] + data[this.thread.x + this.output.x];
}, { dynamicOutput: true, dynamicArguments: true });

const maxStep = gpu.createKernel(function (data) {
  // TODO: keep the LARGER of the pair
  return data[this.thread.x] + data[this.thread.x + this.output.x];
}, { dynamicOutput: true, dynamicArguments: true });

function reduce(step, values) {
  // Float32Array from the start — an argument's type is locked on first call.
  let v = Float32Array.from(values);
  let n = v.length;
  while (n > 1) {
    n = n / 2;
    step.setOutput([n]);
    v = step(v);
  }
  return v[0];
}

console.log('min:', reduce(minStep, data));
console.log('max:', reduce(maxStep, data));
`,
      solutionCode: `// Same ladder, new fold rule. Only the operator changes.
const gpu = new GPU({ mode });

const minStep = gpu.createKernel(function (data) {
  return Math.min(data[this.thread.x], data[this.thread.x + this.output.x]);
}, { dynamicOutput: true, dynamicArguments: true });

const maxStep = gpu.createKernel(function (data) {
  return Math.max(data[this.thread.x], data[this.thread.x + this.output.x]);
}, { dynamicOutput: true, dynamicArguments: true });

function reduce(step, values) {
  // Float32Array from the start — an argument's type is locked on first call.
  let v = Float32Array.from(values);
  let n = v.length;
  while (n > 1) {
    n = n / 2;
    step.setOutput([n]);
    v = step(v);
  }
  return v[0];
}

console.log('min:', reduce(minStep, data));
console.log('max:', reduce(maxStep, data));
`,
      inputs: utils => ({ data: makeValues(utils, 1024, 5150) }),
      publicTests: [
        {
          name: 'one ladder keeps the smaller value, one the larger',
          run: async ctx => {
            let minK = null;
            let maxK = null;
            for (const k of ctx.kernels) {
              if (!k.kernel || !k.kernel.dynamicOutput) continue;
              k.setOutput([1]);
              const a = k(Float32Array.from([3, 5]))[0];
              const b = k(Float32Array.from([8, 2]))[0];
              if (Math.abs(a - 3) < 1e-3 && Math.abs(b - 2) < 1e-3) minK = k;
              if (Math.abs(a - 5) < 1e-3 && Math.abs(b - 8) < 1e-3) maxK = k;
            }
            ctx.assert(minK, 'no min ladder found — one kernel should fold with Math.min');
            ctx.assert(maxK, 'no max ladder found — one kernel should fold with Math.max');
          },
        },
        {
          name: 'min and max of a fresh 512-value array',
          run: async ctx => {
            const rand = ctx.utils.seededRandom(88);
            const arr = new Array(512);
            for (let i = 0; i < 512; i++) arr[i] = Math.round(rand() * 4000) / 1000 - 2;
            let minK = null;
            let maxK = null;
            for (const k of ctx.kernels) {
              if (!k.kernel || !k.kernel.dynamicOutput) continue;
              k.setOutput([1]);
              const probe = k(Float32Array.from([3, 5]))[0];
              if (Math.abs(probe - 3) < 1e-3) minK = k;
              else if (Math.abs(probe - 5) < 1e-3) maxK = k;
            }
            ctx.assert(minK && maxK, 'expected a Math.min ladder and a Math.max ladder');
            ctx.assertClose(runLadder(minK, arr), minOf(arr), 1e-3, 'the minimum');
            ctx.assertClose(runLadder(maxK, arr), maxOf(arr), 1e-3, 'the maximum');
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            const data = makeValues(ctx.utils, 1024, 31337);
            let minK = null;
            let maxK = null;
            for (const k of ctx.kernels) {
              if (!k.kernel || !k.kernel.dynamicOutput) continue;
              k.setOutput([1]);
              const probe = k(Float32Array.from([-4, 9]))[0];
              if (Math.abs(probe - -4) < 1e-3) minK = k;
              else if (Math.abs(probe - 9) < 1e-3) maxK = k;
            }
            ctx.assert(minK && maxK, 'expected a Math.min ladder and a Math.max ladder');
            ctx.assertClose(runLadder(minK, data), minOf(data), 1e-3, 'the minimum');
            ctx.assertClose(runLadder(maxK, data), maxOf(data), 1e-3, 'the maximum');
          },
        },
      ],
    },

    // ------------------------------------------------------------------ 6
    {
      slug: 'fused-mean-rms',
      title: 'Payoff: Mean and RMS, Fused',
      intro: `<p>The payoff. Two statistics over 4,096 values: the <strong>mean</strong>
        (sum ÷ n) and the <strong>RMS</strong> — root-mean-square,
        √(sum&nbsp;of&nbsp;squares&nbsp;÷&nbsp;n) — the standard "how big is this signal"
        measure in audio and physics.</p>
        <p>RMS needs every value squared first. The rookie move is a separate squaring kernel —
        a whole extra pass over memory. The pro move is <strong>fusion</strong>: square each
        value in the same statement that reads it, inside the partial-sum kernel. Map and
        reduce, one pass over the data.</p>
        <p>Stack the whole module: strided partials (task 2) shrink 4,096 values to 64, then a
        single shared halving ladder (task 4) finishes <em>both</em> totals.</p>`,
      goal: `<strong>Goal:</strong> compute and log the mean and the RMS of <code>data</code> —
        two partial-sum kernels (one fused with squaring) plus one shared dynamic halving
        ladder.`,
      requirements: [
        '<code>partialSums</code>: 64 strided partial sums of <code>data</code>, as in task 2',
        '<code>partialSquares</code>: same shape, but square each value <em>as it is read</em> — no separate squaring pass',
        'One dynamic halving-ladder kernel rides both 64-value arrays down to scalars',
        '<code>mean = total / 4096</code>, <code>rms = Math.sqrt(totalSq / 4096)</code> — log both',
      ],
      hints: [
        {
          title: 'Hint 1 — the fused body',
          body: `<p>Read once, use twice:
            <code>const v = data[i * this.constants.threads + this.thread.x]; sum += v * v;</code></p>`,
        },
        {
          title: 'Hint 2 — one ladder, two rides',
          body: `<p>The ladder kernel doesn't care what its 64 inputs mean. Wrap the driver loop
            in a function and call it once with each partials array.</p>`,
        },
        {
          title: 'Hint 3 — the whole shape',
          body: `<p><code>const total = ladder(partialSums(data));</code>
            <code>const totalSq = ladder(partialSquares(data));</code> then divide, square-root,
            and log.</p>`,
        },
      ],
      transfer: `Fusing the map into the reduce is a marquee optimization on every platform:
        <code>thrust::transform_reduce</code> exists precisely for it, CUDA programmers
        hand-fuse to halve their memory traffic, and WebGPU/Metal kernels bake the transform
        into the accumulation loop. Memory bandwidth is the budget — fusion is the
        discount.`,
      starterCode: `// Everything in one pipeline: partials → shared ladder → two statistics.
const gpu = new GPU({ mode });

const partialSums = gpu.createKernel(function (data) {
  // TODO: strided partial sums, exactly like task 2
  return 0;
}, { output: [64], constants: { threads: 64, chunk: 64 } });

const partialSquares = gpu.createKernel(function (data) {
  // TODO: same walk, but square each value AS you read it (fusion!)
  return 0;
}, { output: [64], constants: { threads: 64, chunk: 64 } });

// One rung, reused for both reductions.
const halve = gpu.createKernel(function (data) {
  return data[this.thread.x] + data[this.thread.x + this.output.x];
}, { dynamicOutput: true, dynamicArguments: true });

function ladder(values) {
  let v = values;
  let n = v.length;
  while (n > 1) {
    n = n / 2;
    halve.setOutput([n]);
    v = halve(v);
  }
  return v[0];
}

const total = ladder(partialSums(data));
const totalSq = ladder(partialSquares(data));

const mean = total / 4096;
const rms = Math.sqrt(totalSq / 4096);
console.log('mean:', mean);
console.log('rms:', rms);
`,
      solutionCode: `// Everything in one pipeline: partials → shared ladder → two statistics.
const gpu = new GPU({ mode });

const partialSums = gpu.createKernel(function (data) {
  let sum = 0;
  for (let i = 0; i < this.constants.chunk; i++) {
    sum += data[i * this.constants.threads + this.thread.x];
  }
  return sum;
}, { output: [64], constants: { threads: 64, chunk: 64 } });

// Fused map + reduce: the square happens in the same statement as the read.
const partialSquares = gpu.createKernel(function (data) {
  let sum = 0;
  for (let i = 0; i < this.constants.chunk; i++) {
    const v = data[i * this.constants.threads + this.thread.x];
    sum += v * v;
  }
  return sum;
}, { output: [64], constants: { threads: 64, chunk: 64 } });

// One rung, reused for both reductions.
const halve = gpu.createKernel(function (data) {
  return data[this.thread.x] + data[this.thread.x + this.output.x];
}, { dynamicOutput: true, dynamicArguments: true });

function ladder(values) {
  let v = values;
  let n = v.length;
  while (n > 1) {
    n = n / 2;
    halve.setOutput([n]);
    v = halve(v);
  }
  return v[0];
}

const total = ladder(partialSums(data));
const totalSq = ladder(partialSquares(data));

const mean = total / 4096;
const rms = Math.sqrt(totalSq / 4096);
console.log('mean:', mean);
console.log('rms:', rms);
`,
      inputs: utils => ({ data: makeValues(utils, 4096, 6001) }),
      publicTests: [
        {
          name: 'three kernels: plain partials, fused squared partials, dynamic ladder',
          run: async ctx => {
            const { sums, squares } = findPartialKernels(ctx);
            ctx.assert(sums, 'no kernel producing 64 partial sums found (all-2s input should give 128 per thread)');
            ctx.assert(squares, 'no fused kernel producing 64 partial sums of squares found (all-2s input should give 256 per thread)');
            ctx.assert(findDynamicKernel(ctx), 'no dynamicOutput halving-ladder kernel found');
          },
        },
        {
          name: 'full pipeline: mean and RMS of a fresh array',
          run: async ctx => {
            const { sums, squares } = findPartialKernels(ctx);
            const halve = findDynamicKernel(ctx);
            ctx.assert(sums && squares && halve, 'expected partialSums, partialSquares and a dynamic ladder kernel');
            const arr = new Array(4096);
            for (let i = 0; i < 4096; i++) arr[i] = ((i % 8) + 1) / 4;
            let s = 0;
            let s2 = 0;
            for (let i = 0; i < 4096; i++) {
              s += arr[i];
              s2 += arr[i] * arr[i];
            }
            const total = runLadder(halve, sums(arr));
            const totalSq = runLadder(halve, squares(arr));
            ctx.assertClose(total / 4096, s / 4096, 1e-3, 'the mean');
            ctx.assertClose(Math.sqrt(totalSq / 4096), Math.sqrt(s2 / 4096), 1e-3, 'the RMS');
          },
        },
        {
          name: 'mean and RMS of <code>data</code> are logged',
          run: async ctx => {
            const data = makeValues(ctx.utils, 4096, 6001);
            let s = 0;
            let s2 = 0;
            for (let i = 0; i < data.length; i++) {
              s += data[i];
              s2 += data[i] * data[i];
            }
            const mean = s / 4096;
            const rms = Math.sqrt(s2 / 4096);
            const nums = loggedNumbers(ctx.logs);
            ctx.assert(
              nums.some(v => Math.abs(v - mean) <= 0.01),
              `log the mean — expected ≈${mean.toFixed(3)} in the console output`
            );
            ctx.assert(
              nums.some(v => Math.abs(v - rms) <= 0.01),
              `log the RMS — expected ≈${rms.toFixed(3)} in the console output`
            );
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            const { sums, squares } = findPartialKernels(ctx);
            const halve = findDynamicKernel(ctx);
            ctx.assert(sums && squares && halve, 'expected partialSums, partialSquares and a dynamic ladder kernel');
            const data = makeValues(ctx.utils, 4096, 909);
            let s = 0;
            let s2 = 0;
            for (let i = 0; i < data.length; i++) {
              s += data[i];
              s2 += data[i] * data[i];
            }
            const total = runLadder(halve, sums(data));
            const totalSq = runLadder(halve, squares(data));
            ctx.assertClose(total / 4096, s / 4096, 0.01, 'the mean');
            ctx.assertClose(Math.sqrt(totalSq / 4096), Math.sqrt(s2 / 4096), 0.01, 'the RMS');
          },
        },
      ],
    },
  ],
};
