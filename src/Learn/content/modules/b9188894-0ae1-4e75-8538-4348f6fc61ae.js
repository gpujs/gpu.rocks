// Module: Measuring Speed Honestly — uuid b9188894-0ae1-4e75-8538-4348f6fc61ae (short id b9188894).
// The file name is the uuid; identity lives in the exported object below,
// never in the path. Legacy id (pre-uuid URLs, localStorage migration): 1-5.
//
// Module 1.5 — Measuring Speed Honestly.
//
// Four tasks: the compile-time trap in first-call timings → transfer cost
// scaling with bytes, not math → float32 vs float64 and tolerant comparison
// → a tiny workload where the plain-JS loop legitimately wins.
//
// Timing numbers are for the learner's eyes only — tests assert computed
// values and setup (kernel outputs, logged labels/verdicts), never wall-clock
// durations, so verification stays deterministic in cpu and gpu modes.

// Seeded array of n values, 0–10 with 2 decimal places.
function makeValues(utils, n, seed) {
  const rand = utils.seededRandom(seed);
  const out = new Array(n);
  for (let i = 0; i < n; i++) out[i] = Math.round(rand() * 1000) / 100;
  return out;
}

// Float64 reference for task 3: sum of 1/(k + x) for k = 1…1000.
function partialSumRef(x) {
  let sum = 0;
  for (let k = 1; k <= 1000; k++) sum += 1 / (k + x);
  return sum;
}

// True when a console.log line containing `text` was captured.
function logged(ctx, text) {
  return ctx.logs.some(line => line.type === 'log' && line.text && line.text.includes(text));
}

// ---- near-miss diagnosis --------------------------------------------------
//
// A failing assert that reports only two numbers tells a learner nothing about
// WHICH slip produced them. A probe pairs the value one specific known mistake
// would produce with a sentence naming that mistake; diagnose() speaks only
// when the observed value matches a probe within the test's own tolerance AND
// the correct value does not — so elements where two candidates coincide stay
// silent, as do observations that match probes disagreeing with each other.
// A wrong diagnosis is worse than none.
function diagnose(got, expected, eps, probes) {
  const hits = probes
    .filter(p => Math.abs(got - p[0]) <= eps && Math.abs(expected - p[0]) > eps)
    .map(p => p[1]);
  return hits.length && hits.every(m => m === hits[0]) ? hits[0] : null;
}

// Task 1: Math.sin(x / 100) * 100, mis-scaled in each of the obvious ways.
function sineProbes(x) {
  return [
    [Math.sin(x / 100), 'the amplitude is missing — the sample is Math.sin(x / 100) * 100'],
    [Math.sin(x) * 100, 'you sampled Math.sin(this.thread.x) — the index has to be divided by 100 first'],
    [Math.sin(x * 100) * 100, 'the index is multiplied by 100 where it should be divided by it'],
  ];
}

// Tasks 2 and 4: a one-instruction kernel that never ran its one instruction.
function unchangedProbe(arr, i, operation) {
  return [arr[i], `that is the element unchanged — the ${operation} never happened`];
}

// Task 4: "twice the index" is computed from the thread id rather than from the
// data, and one matching cell would be weak evidence — this probe has to hold
// across all 16 cells (and differ from the right answer somewhere) to speak.
function doubleIndexHint(out, arr) {
  return diagnoseAll(16, i => out[i], i => arr[i] * 2, 1e-3, [
    [i => 2 * i,
      'every cell is twice the thread index, not twice the element — index the array with it: data[this.thread.x]'],
  ]);
}

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

// Task 3: the accumulator loop, three ways. The k = 0 probe is skipped for
// thread 0, where the extra term would be a division by zero rather than a
// number the test could match.
function partialSumProbes(x) {
  const probes = [
    [1 / (1 + x), 'each pass overwrote the running total — accumulate it with sum += inside the loop'],
    [partialSumRef(0) + 1000 * x, 'the parentheses are missing — each term is 1 / (k + this.thread.x), not 1 / k + this.thread.x'],
  ];
  if (x > 0) {
    probes.push([partialSumRef(x) + 1 / x, 'the loop started at k = 0 — that extra 1 / this.thread.x term does not belong to the sum']);
  }
  return probes;
}

export default {
  uuid: 'b9188894-0ae1-4e75-8538-4348f6fc61ae',
  version: 1,
  slug: 'measuring-speed-honestly',
  legacyId: '1-5',
  title: 'Measuring Speed Honestly',
  blurb: 'Warm-up, transfer costs, and precision — when the GPU wins, and when the CPU quietly beats it.',
  tasks: [
    // ------------------------------------------------------------------ 1
    {
      slug: 'first-call-lie',
      title: 'The First Call Is a Lie',
      intro: `<p>The first time you invoke a kernel, gpu.js does far more than run it: it
        <strong>transpiles</strong> your JavaScript function to shader code, hands it to the GPU
        driver to <strong>compile and link</strong>, allocates textures — and <em>then</em> runs it.
        The second call skips straight to the run.</p>
        <p>So timing the first call measures the compiler, not your kernel. It can be 100× slower
        than the steady state, and it happens exactly once. Every honest benchmark
        <strong>warms up first</strong> and throws that first measurement away.</p>`,
      goal: `<strong>Goal:</strong> finish the kernel, then use <code>Date.now()</code> to time the
        <em>first</em> call and the <em>warmed-up</em> average separately — and log both.`,
      requirements: [
        'Finish the kernel: return <code>Math.sin(x / 100) * 100</code> where <code>x</code> is this thread\'s index',
        'Time the first call with <code>Date.now()</code> and log it: <code>first call: N ms</code>',
        'Call the kernel 10 more times in one timed block and log the average: <code>warm call: N ms</code>',
      ],
      hints: [
        {
          title: 'Hint 1 — the stopwatch pattern',
          body: `<p>Snapshot the clock, do the work, subtract:</p>
<pre><code>const t0 = Date.now();
// … the work …
console.log('first call:', Date.now() - t0, 'ms');</code></pre>`,
        },
        {
          title: 'Hint 2 — averaging the warm calls',
          body: `<p>One stopwatch around a loop of 10 calls, then divide:</p>
<pre><code>t0 = Date.now();
for (let i = 0; i &lt; 10; i++) wave();
console.log('warm call:', (Date.now() - t0) / 10, 'ms');</code></pre>`,
        },
      ],
      transfer: `Every platform has a version of this pause: CUDA JIT-compiles PTX at first launch
        (then caches it), WebGPU builds the shader in <code>createComputePipeline</code>, Metal
        compiles MSL when the pipeline state is created. Benchmarking guides on all of them open
        with the same rule — discard the first iteration.`,
      starterCode: `// The first call compiles. The rest just run. Prove it.
const gpu = new GPU({ mode });

const wave = gpu.createKernel(function () {
  // TODO: return Math.sin(x / 100) * 100, where x is this thread's index
  return 0;
}, { output: [2048] });

// TODO: time the FIRST call with Date.now():
//   const t0 = Date.now();  ...call wave()...
//   console.log('first call:', Date.now() - t0, 'ms');
const result = wave();

// TODO: call wave() 10 more times inside one timed block, then log the
// average as:  console.log('warm call:', totalMs / 10, 'ms');

console.log('sample value:', result[100]);
`,
      solutionCode: `// The first call compiles. The rest just run. Prove it.
const gpu = new GPU({ mode });

const wave = gpu.createKernel(function () {
  return Math.sin(this.thread.x / 100) * 100;
}, { output: [2048] });

// First call: transpile + compile + allocate + run.
let t0 = Date.now();
const result = wave();
console.log('first call:', Date.now() - t0, 'ms');

// Steady state: the compiled program just runs.
t0 = Date.now();
for (let i = 0; i < 10; i++) wave();
console.log('warm call:', (Date.now() - t0) / 10, 'ms');

console.log('sample value:', result[100]);
`,
      publicTests: [
        {
          name: 'kernel computes <code>sin(x / 100) · 100</code> for all 2048 threads',
          run: async ctx => {
            ctx.assert(ctx.kernels.length >= 1, 'no kernel was created — call gpu.createKernel()');
            const out = ctx.kernel();
            ctx.assert(out && out.length === 2048, `expected 2048 output values, got ${out && out.length}`);
            for (const x of [0, 1, 100, 777, 1023, 2047]) {
              const expected = Math.sin(x / 100) * 100;
              const hint = diagnose(out[x], expected, 0.05, sineProbes(x));
              ctx.assertClose(out[x], expected, 0.05, hint || `element ${x}`);
            }
          },
        },
        {
          name: 'both timings are logged: <code>first call</code> and <code>warm call</code>',
          run: async ctx => {
            ctx.assert(
              logged(ctx, 'first call'),
              "time the first call and log it — console.log('first call:', ms, 'ms')"
            );
            ctx.assert(
              logged(ctx, 'warm call'),
              "time 10 warmed-up calls and log the average — console.log('warm call:', avg, 'ms')"
            );
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            const out = ctx.kernel();
            ctx.assert(out.length === 2048, 'expected 2048 output values');
            for (let x = 0; x < 2048; x++) {
              const expected = Math.sin(x / 100) * 100;
              const hint = diagnose(out[x], expected, 0.05, sineProbes(x));
              ctx.assertClose(out[x], expected, 0.05, hint || `element ${x}`);
            }
          },
        },
      ],
    },

    // ------------------------------------------------------------------ 2
    {
      slug: 'transfer-tax',
      title: 'Pay the Transfer Tax',
      intro: `<p>A kernel call isn't just compute. Every invocation ships your input array from
        JavaScript to GPU memory, runs, then ships the result back. For a one-instruction kernel
        like <code>value + 1</code>, the arithmetic is nearly free — <strong>the ride is the whole
        bill</strong>.</p>
        <p>Below, the same trivial kernel runs on 1,024 values and on 65,536 values — 64× the data,
        identical math per thread. If compute were the cost, both would time about the same. Warm
        up first (task 1!), then measure: the per-call cost tracks <strong>bytes moved</strong>,
        not operations performed.</p>`,
      goal: `<strong>Goal:</strong> finish the <code>+ 1</code> kernel and the
        <code>timeKernel</code> helper — warm up, then average 20 timed calls — and log the
        per-call cost for both payload sizes.`,
      requirements: [
        'Kernel returns <code>data[this.thread.x] + 1</code> — one instruction, on purpose',
        'In <code>timeKernel</code>: call the kernel once <em>untimed</em> to warm it up',
        'Then time 20 calls with <code>Date.now()</code> and return the average ms per call',
        'Log both costs (the <code>small:</code>/<code>big:</code> lines are already wired up)',
      ],
      hints: [
        {
          title: 'Hint 1 — why warm up here too?',
          body: `<p><code>makePlusOne</code> builds <em>two separate kernels</em>, and each one
            compiles on its own first call. Without the warm-up, the big kernel's timing would
            include a compile — task 1's lie all over again.</p>`,
        },
        {
          title: 'Hint 2 — the helper body',
          body: `<pre><code>kernel(arg);
const t0 = Date.now();
for (let i = 0; i &lt; 20; i++) kernel(arg);
return (Date.now() - t0) / 20;</code></pre>`,
        },
      ],
      transfer: `The bus is the bottleneck everywhere: <code>cudaMemcpy</code> across PCIe is the
        classic hot spot in CUDA and ROCm profiles, WebGPU makes you stage the copies explicitly
        with <code>writeBuffer</code> and <code>mapAsync</code>, and Apple's unified memory exists
        precisely to shrink this tax. Arithmetic is cheap; moving bytes is not.`,
      starterCode: `// One-instruction kernel, two payload sizes. Cost tracks bytes, not math.
const gpu = new GPU({ mode });

function makePlusOne(n) {
  return gpu.createKernel(function (data) {
    // TODO: return this thread's element, plus one
    return data[this.thread.x];
  }, { output: [n] });
}

const smallKernel = makePlusOne(1024);   // small = 1,024 values
const bigKernel = makePlusOne(65536);    // big = 65,536 values

function timeKernel(kernel, arg) {
  // TODO: warm up with one untimed call (task 1!),
  // then time 20 calls and return the average ms per call
  return 0;
}

console.log('small:', timeKernel(smallKernel, small), 'ms/call');
console.log('big:', timeKernel(bigKernel, big), 'ms/call');
`,
      solutionCode: `// One-instruction kernel, two payload sizes. Cost tracks bytes, not math.
const gpu = new GPU({ mode });

function makePlusOne(n) {
  return gpu.createKernel(function (data) {
    return data[this.thread.x] + 1;
  }, { output: [n] });
}

const smallKernel = makePlusOne(1024);   // small = 1,024 values
const bigKernel = makePlusOne(65536);    // big = 65,536 values

function timeKernel(kernel, arg) {
  kernel(arg); // warm up — never time the compile (task 1)
  const t0 = Date.now();
  for (let i = 0; i < 20; i++) kernel(arg);
  return (Date.now() - t0) / 20;
}

console.log('small:', timeKernel(smallKernel, small), 'ms/call');
console.log('big:', timeKernel(bigKernel, big), 'ms/call');
`,
      inputs: utils => ({
        small: makeValues(utils, 1024, 1101),
        big: makeValues(utils, 65536, 1102),
      }),
      publicTests: [
        {
          name: 'two kernels exist: output sizes <code>1024</code> and <code>65536</code>',
          run: async ctx => {
            ctx.assert(ctx.kernels.length >= 2, `expected 2 kernels, found ${ctx.kernels.length}`);
            const small = ctx.kernels.find(k => k.kernel && k.kernel.output && k.kernel.output[0] === 1024);
            const big = ctx.kernels.find(k => k.kernel && k.kernel.output && k.kernel.output[0] === 65536);
            ctx.assert(small, 'no kernel with output [1024] found');
            ctx.assert(big, 'no kernel with output [65536] found');
          },
        },
        {
          name: 'every element comes back as <code>value + 1</code>',
          run: async ctx => {
            const small = ctx.kernels.find(k => k.kernel && k.kernel.output && k.kernel.output[0] === 1024);
            const big = ctx.kernels.find(k => k.kernel && k.kernel.output && k.kernel.output[0] === 65536);
            ctx.assert(small && big, 'expected kernels with outputs [1024] and [65536]');
            const smallIn = makeValues(ctx.utils, 1024, 2201);
            const smallOut = small(smallIn);
            for (let i = 0; i < 1024; i++) {
              const hint = diagnose(smallOut[i], smallIn[i] + 1, 1e-3, [
                unchangedProbe(smallIn, i, '+ 1'),
              ]);
              ctx.assertClose(smallOut[i], smallIn[i] + 1, 1e-3, hint || `small element ${i}`);
            }
            const bigIn = makeValues(ctx.utils, 65536, 2202);
            const bigOut = big(bigIn);
            for (let i = 0; i < 65536; i += 271) {
              const hint = diagnose(bigOut[i], bigIn[i] + 1, 1e-3, [
                unchangedProbe(bigIn, i, '+ 1'),
              ]);
              ctx.assertClose(bigOut[i], bigIn[i] + 1, 1e-3, hint || `big element ${i}`);
            }
          },
        },
        {
          name: 'per-call cost logged for both payloads (<code>ms/call</code>)',
          run: async ctx => {
            ctx.assert(logged(ctx, 'small:'), "log the small kernel's cost — the console.log is in the starter");
            ctx.assert(logged(ctx, 'big:'), "log the big kernel's cost — the console.log is in the starter");
            ctx.assert(logged(ctx, 'ms/call'), 'timeKernel should return ms per call (did it return 0 forever?)');
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            const small = ctx.kernels.find(k => k.kernel && k.kernel.output && k.kernel.output[0] === 1024);
            const big = ctx.kernels.find(k => k.kernel && k.kernel.output && k.kernel.output[0] === 65536);
            ctx.assert(small && big, 'expected kernels with outputs [1024] and [65536]');
            const smallIn = makeValues(ctx.utils, 1024, 3301);
            const smallOut = small(smallIn);
            ctx.assert(smallOut.length === 1024, 'small kernel should produce 1024 values');
            for (let i = 0; i < 1024; i++) {
              const hint = diagnose(smallOut[i], smallIn[i] + 1, 1e-3, [
                unchangedProbe(smallIn, i, '+ 1'),
              ]);
              ctx.assertClose(smallOut[i], smallIn[i] + 1, 1e-3, hint || `small element ${i}`);
            }
            const bigIn = makeValues(ctx.utils, 65536, 3302);
            const bigOut = big(bigIn);
            ctx.assert(bigOut.length === 65536, 'big kernel should produce 65536 values');
            for (let i = 0; i < 65536; i += 97) {
              const hint = diagnose(bigOut[i], bigIn[i] + 1, 1e-3, [
                unchangedProbe(bigIn, i, '+ 1'),
              ]);
              ctx.assertClose(bigOut[i], bigIn[i] + 1, 1e-3, hint || `big element ${i}`);
            }
          },
        },
      ],
    },

    // ------------------------------------------------------------------ 3
    {
      slug: 'two-answers',
      title: 'Two Machines, Two Answers',
      intro: `<p>JavaScript numbers are 64-bit floats — about 16 significant digits. GPU shaders
        compute in <strong>32-bit floats</strong> — about 7. Run the <em>same</em> arithmetic on
        both machines and the answers drift apart, a little more with every operation.</p>
        <p>The kernel below adds 1,000 fractions per thread; a plain JavaScript loop computes the
        identical sum in float64. The two results will disagree somewhere around the sixth decimal
        place — which means <code>===</code> is the wrong question. The right question is:
        <strong>are they within a tolerance that matters for your problem?</strong></p>`,
      goal: `<strong>Goal:</strong> finish the kernel — each thread sums
        <code>1 / (k + this.thread.x)</code> for <code>k = 1…1000</code> — then fix the final
        comparison to use a tolerance instead of <code>===</code>.`,
      requirements: [
        'Kernel: accumulate <code>1 / (k + this.thread.x)</code> over <code>k = 1…1000</code> in a loop',
        'Keep the float64 reference sum for thread 0 (already wired up)',
        'Log the verdict with a tolerance: <code>Math.abs(result[0] - ref) &lt; 1e-3</code>, not <code>===</code>',
      ],
      hints: [
        {
          title: 'Hint 1 — loops inside kernels',
          body: `<p>Fixed-bound loops are fine in kernel code:</p>
<pre><code>for (let k = 1; k &lt;= 1000; k++) {
  sum += 1 / (k + this.thread.x);
}</code></pre>`,
        },
        {
          title: 'Hint 2 — the tolerant verdict',
          body: `<p>Replace the <code>===</code> comparison in the last line with
            <code>Math.abs(result[0] - ref) &lt; 1e-3</code>. Exact equality across float32 and
            float64 is a coin you will almost never win.</p>`,
        },
      ],
      transfer: `float32-by-default is universal shader behavior — and production GPU code often
        trades away <em>more</em> precision on purpose: CUDA's <code>--use_fast_math</code>, TF32
        on tensor cores, half-precision inference. That's why numerical toolkits ship
        <code>allclose</code>-style comparisons, and why this course's tests use
        <code>assertClose</code> instead of <code>==</code>.`,
      starterCode: `// Same math, two machines: your GPU adds in float32, JavaScript in float64.
const gpu = new GPU({ mode });

const partialSums = gpu.createKernel(function () {
  let sum = 0;
  // TODO: add up 1 / (k + this.thread.x) for k = 1 ... 1000
  sum = 1 / (1 + this.thread.x);
  return sum;
}, { output: [64] });

const result = partialSums();

// The same sum for thread 0, computed in float64 JavaScript:
let ref = 0;
for (let k = 1; k <= 1000; k++) ref += 1 / k;

console.log('kernel says:', result[0]);
console.log('js says:    ', ref);
console.log('difference:', Math.abs(result[0] - ref));
// TODO: '===' is the wrong question — compare with a tolerance instead:
console.log('close enough:', result[0] === ref);
`,
      solutionCode: `// Same math, two machines: your GPU adds in float32, JavaScript in float64.
const gpu = new GPU({ mode });

const partialSums = gpu.createKernel(function () {
  let sum = 0;
  for (let k = 1; k <= 1000; k++) {
    sum += 1 / (k + this.thread.x);
  }
  return sum;
}, { output: [64] });

const result = partialSums();

// The same sum for thread 0, computed in float64 JavaScript:
let ref = 0;
for (let k = 1; k <= 1000; k++) ref += 1 / k;

console.log('kernel says:', result[0]);
console.log('js says:    ', ref);
console.log('difference:', Math.abs(result[0] - ref));
// The right question: within a tolerance that matters for this problem?
console.log('close enough:', Math.abs(result[0] - ref) < 1e-3);
`,
      publicTests: [
        {
          name: 'all 64 partial sums match the float64 reference within <code>1e-3</code>',
          run: async ctx => {
            ctx.assert(ctx.kernels.length >= 1, 'no kernel was created — call gpu.createKernel()');
            const out = ctx.kernel();
            ctx.assert(out && out.length === 64, `expected 64 output values, got ${out && out.length}`);
            for (let x = 0; x < 64; x++) {
              const hint = diagnose(out[x], partialSumRef(x), 1e-3, partialSumProbes(x));
              ctx.assertClose(out[x], partialSumRef(x), 1e-3, hint || `partial sum for thread ${x}`);
            }
          },
        },
        {
          name: 'verdict uses a tolerance — <code>close enough: true</code> is logged',
          run: async ctx => {
            ctx.assert(logged(ctx, 'difference:'), 'keep the difference log — it shows the float32/float64 drift');
            ctx.assert(
              logged(ctx, 'close enough: true'),
              "compare with Math.abs(result[0] - ref) < 1e-3, not === — the verdict should log true"
            );
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            const out = ctx.kernel();
            // Values must track the reference AND carry its shape: each partial
            // sum starts one term later, so the sequence strictly decreases.
            for (let x = 0; x < 64; x++) {
              const hint = diagnose(out[x], partialSumRef(x), 1e-3, partialSumProbes(x));
              ctx.assertClose(out[x], partialSumRef(x), 1e-3, hint || `partial sum for thread ${x}`);
            }
            for (let x = 0; x < 63; x++) {
              ctx.assert(out[x] > out[x + 1], `sum for thread ${x} should exceed thread ${x + 1}`);
            }
          },
        },
      ],
    },

    // ------------------------------------------------------------------ 4
    {
      slug: 'when-cpu-wins',
      title: 'When the CPU Wins',
      intro: `<p>Sixteen numbers, doubled. The GPU <em>can</em> do it — but every kernel call pays
        a fixed toll before any math happens: dispatch through the graphics API, upload 16 values,
        read 16 back. A plain JavaScript loop finishes the whole job in nanoseconds, before the
        GPU has cleared its throat.</p>
        <p>This is the module's payoff — the full honest-measurement checklist in one run:
        <strong>warm up first</strong> (task 1), <strong>remember the transfer toll</strong>
        (task 2), <strong>compare results with a tolerance</strong> (task 3), and then
        <strong>declare the true winner</strong> — even when it isn't the GPU. Parallel hardware
        pays off on big workloads; on tiny ones, the honest answer is a for-loop.</p>`,
      goal: `<strong>Goal:</strong> double <code>tiny</code> both ways — kernel and plain loop —
        verify they agree within a tolerance, time both fairly, and log the winner.`,
      requirements: [
        'Kernel returns <code>data[this.thread.x] * 2</code> for all 16 threads',
        'Compare <code>fromKernel</code> to <code>fromLoop</code> element-wise with tolerance <code>1e-4</code> and log <code>match: true</code>',
        'Time 200 warmed-up rounds of each contender and log both as <code>ms/round</code>',
        'Log <code>winner:</code> with whichever contender was faster',
      ],
      hints: [
        {
          title: 'Hint 1 — the tolerant match',
          body: `<p>Task 3's move, in a loop: start with <code>let allMatch = true;</code> and flip
            it to <code>false</code> whenever <code>Math.abs(fromKernel[i] - fromLoop[i]) &gt; 1e-4</code>.</p>`,
        },
        {
          title: 'Hint 2 — a fair fight',
          body: `<p>The first <code>doubleTiny(tiny)</code> call already warmed the kernel up, so
            both timed loops measure steady state. Time 200 rounds of <code>doubleTiny(tiny)</code>,
            then 200 rounds of the JS loop, and divide each total by 200.</p>`,
        },
        {
          title: 'Hint 3 — declaring the winner',
          body: `<pre><code>console.log('winner:', kernelMs &lt; loopMs ? 'gpu kernel' : 'plain js');</code></pre>
<p>On a job this small, expect the loop to take it. That's the honest answer.</p>`,
        },
      ],
      transfer: `Kernel-launch overhead runs to microseconds on CUDA and ROCm — thousands of
        CPU instructions' worth per launch. It's why serious frameworks batch and fuse tiny
        operations instead of dispatching them one at a time, and why "is this workload big
        enough?" is the first question asked in any GPU port.`,
      starterCode: `// 16 numbers. The GPU CAN double them — but should it?
const gpu = new GPU({ mode });

const doubleTiny = gpu.createKernel(function (data) {
  // TODO: return double this thread's element
  return data[this.thread.x];
}, { output: [16] });

const fromKernel = doubleTiny(tiny); // also serves as the warm-up call

// The same job, plain JavaScript:
const fromLoop = new Array(16);
for (let i = 0; i < 16; i++) fromLoop[i] = tiny[i] * 2;

// TODO: compare fromKernel and fromLoop element-wise with tolerance 1e-4
// (task 3!) and log:  console.log('match:', allMatch);

// TODO: time 200 rounds of each contender with Date.now(), then log:
//   console.log('kernel:  ', kernelMs, 'ms/round');
//   console.log('plain js:', loopMs, 'ms/round');
//   console.log('winner:', kernelMs < loopMs ? 'gpu kernel' : 'plain js');
`,
      solutionCode: `// 16 numbers. The GPU CAN double them — but should it?
const gpu = new GPU({ mode });

const doubleTiny = gpu.createKernel(function (data) {
  return data[this.thread.x] * 2;
}, { output: [16] });

const fromKernel = doubleTiny(tiny); // also serves as the warm-up call

// The same job, plain JavaScript:
const fromLoop = new Array(16);
for (let i = 0; i < 16; i++) fromLoop[i] = tiny[i] * 2;

// Same answer? Tolerance, not === (task 3).
let allMatch = true;
for (let i = 0; i < 16; i++) {
  if (Math.abs(fromKernel[i] - fromLoop[i]) > 1e-4) allMatch = false;
}
console.log('match:', allMatch);

// A fair fight: both warmed up, both averaged over many rounds.
const ROUNDS = 200;
let t0 = Date.now();
for (let r = 0; r < ROUNDS; r++) doubleTiny(tiny);
const kernelMs = (Date.now() - t0) / ROUNDS;

t0 = Date.now();
for (let r = 0; r < ROUNDS; r++) {
  for (let i = 0; i < 16; i++) fromLoop[i] = tiny[i] * 2;
}
const loopMs = (Date.now() - t0) / ROUNDS;

console.log('kernel:  ', kernelMs, 'ms/round');
console.log('plain js:', loopMs, 'ms/round');
console.log('winner:', kernelMs < loopMs ? 'gpu kernel' : 'plain js');
`,
      inputs: utils => ({ tiny: makeValues(utils, 16, 4404) }),
      publicTests: [
        {
          name: 'kernel doubles all 16 values',
          run: async ctx => {
            ctx.assert(ctx.kernels.length >= 1, 'no kernel was created — call gpu.createKernel()');
            const arr = new Array(16);
            for (let i = 0; i < 16; i++) arr[i] = i * 1.25 - 3;
            const out = ctx.kernel(arr);
            ctx.assert(out && out.length === 16, `expected 16 output values, got ${out && out.length}`);
            const series = doubleIndexHint(out, arr);
            for (let i = 0; i < 16; i++) {
              const hint = series ||
                diagnose(out[i], arr[i] * 2, 1e-3, [unchangedProbe(arr, i, 'doubling')]);
              ctx.assertClose(out[i], arr[i] * 2, 1e-3, hint || `element ${i}`);
            }
          },
        },
        {
          name: 'results agree within tolerance — <code>match: true</code> is logged',
          run: async ctx => {
            ctx.assert(
              logged(ctx, 'match: true'),
              "compare fromKernel and fromLoop with Math.abs(a - b) <= 1e-4 and log the verdict — expected 'match: true'"
            );
          },
        },
        {
          name: 'both contenders timed and a winner declared',
          run: async ctx => {
            ctx.assert(logged(ctx, 'ms/round'), 'time both contenders and log each as ms/round');
            ctx.assert(logged(ctx, 'kernel:'), "log the kernel's time — console.log('kernel:  ', kernelMs, 'ms/round')");
            ctx.assert(logged(ctx, 'plain js:'), "log the loop's time — console.log('plain js:', loopMs, 'ms/round')");
            ctx.assert(logged(ctx, 'winner:'), "declare the faster contender with a 'winner:' log");
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            const data = makeValues(ctx.utils, 16, 5505);
            const out = ctx.kernel(data);
            ctx.assert(out.length === 16, 'expected exactly 16 output values');
            const series = doubleIndexHint(out, data);
            for (let i = 0; i < 16; i++) {
              const hint = series ||
                diagnose(out[i], data[i] * 2, 1e-3, [unchangedProbe(data, i, 'doubling')]);
              ctx.assertClose(out[i], data[i] * 2, 1e-3, hint || `element ${i}`);
            }
          },
        },
      ],
    },
  ],
};
