// Module 1.1 — Hello, Kernel (the very first module of the course).
//
// Five tasks: the first kernel ever (createKernel + output size) → the
// thread index as identity → computing a formula from the index → the 2D
// thread grid → passing a first argument in and reusing one kernel for
// many calls.
//
// Kernel-authoring rules (contract): no closures inside kernel functions,
// only numbers / nested number arrays as inputs, this.thread.* for
// indexing. Every task passes in CPU mode. Sizes are tiny on purpose —
// this module is about the mental model, not throughput.

const TWO_PI = Math.PI * 2;

// Expected 64-sample sine wave for task 3 (shared by both test tiers).
function expectedWave(count = 64) {
  const wave = new Array(count);
  for (let i = 0; i < count; i++) wave[i] = Math.sin((i / count) * TWO_PI);
  return wave;
}

export default {
  id: '1-1',
  track: 1,
  title: 'Hello, Kernel',
  blurb: 'What a kernel is, what a thread is, and why <code>this.thread.x</code> replaces your for-loop.',
  tasks: [
    // ------------------------------------------------------------------ 1
    {
      slug: 'first-kernel',
      title: 'Your First Kernel',
      intro: `<p>A <strong>kernel</strong> is an ordinary-looking JavaScript function with one twist:
        it doesn't run once. gpu.js compiles it and launches it <strong>once per output cell</strong>,
        all in parallel — each launch is called a <strong>thread</strong>. You never call the function
        in a loop; you tell the GPU how many cells you want, and it runs that many copies.</p>
        <p>That cell count is the <code>output</code> option: <code>output: [16]</code> means
        &ldquo;give me 16 cells&rdquo;, so 16 threads run and their 16 return values come back to you
        collected into one array.</p>`,
      goal: `<strong>Goal:</strong> finish the kernel so that <strong>16 threads</strong> each return
        the number <code>42</code> — your first parallel program.`,
      requirements: [
        'Set <code>output</code> to <code>[16]</code> so 16 threads run',
        'Return <code>42</code> from the kernel body',
        'Call the kernel and log the result (already wired up)',
      ],
      hints: [
        {
          title: 'Hint 1 — where does the 16 go?',
          body: `<p><code>output</code> lives in the options object — the second argument to
            <code>createKernel</code>. It's an array because output can have more than one
            dimension (that's task 4).</p>`,
        },
        {
          title: 'Hint 2 — the whole thing',
          body: `<p><code>gpu.createKernel(function () { return 42; }, { output: [16] })</code> —
            then calling it returns an array of sixteen 42s.</p>`,
        },
      ],
      transfer: `Launching N copies of one function is <em>the</em> primitive of every GPU API:
        CUDA spells it <code>kernel&lt;&lt;&lt;blocks, threads&gt;&gt;&gt;()</code>, WebGPU calls it
        a compute <code>dispatch</code>, Metal dispatches threadgroups. gpu.js just hides the
        ceremony behind <code>output</code>.`,
      starterCode: `// A kernel runs once per output cell — in parallel, not in a loop.
const gpu = new GPU({ mode });

const answer = gpu.createKernel(function () {
  // TODO: every thread should return the same number: 42
  return 0;
}, {
  // TODO: give the kernel 16 output cells, not 1
  output: [1],
});

const result = answer();
console.log(result);
`,
      solutionCode: `// A kernel runs once per output cell — in parallel, not in a loop.
const gpu = new GPU({ mode });

const answer = gpu.createKernel(function () {
  return 42;
}, {
  output: [16],
});

const result = answer();
console.log(result);
`,
      publicTests: [
        {
          name: 'kernel runs 16 threads — the result has 16 values',
          run: async ctx => {
            ctx.assert(ctx.kernels.length >= 1, 'no kernel was created — call gpu.createKernel()');
            const out = ctx.kernel();
            ctx.assert(out && out.length === 16, `expected 16 output values, got ${out && out.length}`);
          },
        },
        {
          name: 'every thread returns <code>42</code>',
          run: async ctx => {
            const out = ctx.kernel();
            for (let i = 0; i < 16; i++) {
              ctx.assertClose(out[i], 42, 1e-3, `value from thread ${i}`);
            }
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            // Re-invoke: same kernel, same result — 16 forty-twos, every time.
            const out = ctx.kernel();
            ctx.assert(out.length === 16, 'expected 16 output values');
            let sum = 0;
            for (let i = 0; i < out.length; i++) sum += out[i];
            ctx.assertClose(sum, 42 * 16, 1e-2, 'the 16 values should total 672');
          },
        },
      ],
    },

    // ------------------------------------------------------------------ 2
    {
      slug: 'thread-identity',
      title: 'Who Am I? this.thread.x',
      intro: `<p>Sixteen identical 42s prove the launch works, but parallel code is only useful if
        each thread can do something <em>different</em>. The trick: every thread knows which output
        cell it owns. That number is <code>this.thread.x</code> — 0 for the first cell, 1 for the
        next, up to <code>output − 1</code>.</p>
        <p>Same function, same arguments, different <code>this.thread.x</code> — that one number is
        the only thing telling the threads apart, and it's how each one finds its own work.</p>`,
      goal: `<strong>Goal:</strong> make each of the 32 threads return <strong>its own index</strong>,
        so the result counts <code>0, 1, 2, … 31</code>.`,
      requirements: [
        'Keep <code>output: [32]</code> — 32 threads',
        'Return <code>this.thread.x</code> from the kernel body',
        'No loops, no counters — the index is handed to you',
      ],
      hints: [
        {
          title: 'Hint 1 — it’s already there',
          body: `<p>You don't compute the index and you don't pass it in. Inside the kernel body,
            <code>this.thread.x</code> is simply available — gpu.js fills it in per thread.</p>`,
        },
        {
          title: 'Hint 2 — the one-liner',
          body: `<p>The entire kernel body: <code>return this.thread.x;</code></p>`,
        },
      ],
      transfer: `Every platform hands threads this same self-identity, just under a different name:
        <code>threadIdx</code>/<code>blockIdx</code> in CUDA and ROCm/HIP,
        <code>global_invocation_id</code> in WebGPU's WGSL,
        <code>thread_position_in_grid</code> in Metal.`,
      starterCode: `// Every thread runs the same body — this.thread.x is what differs.
const gpu = new GPU({ mode });

const whoAmI = gpu.createKernel(function () {
  // TODO: return this thread's own index
  return 0;
}, { output: [32] });

const result = whoAmI();
console.log(result);
`,
      solutionCode: `// Every thread runs the same body — this.thread.x is what differs.
const gpu = new GPU({ mode });

const whoAmI = gpu.createKernel(function () {
  return this.thread.x;
}, { output: [32] });

const result = whoAmI();
console.log(result);
`,
      publicTests: [
        {
          name: 'result holds 32 values',
          run: async ctx => {
            ctx.assert(ctx.kernels.length >= 1, 'no kernel was created — call gpu.createKernel()');
            const out = ctx.kernel();
            ctx.assert(out && out.length === 32, `expected 32 output values, got ${out && out.length}`);
          },
        },
        {
          name: 'cell <code>i</code> holds <code>i</code> — each thread reports its index',
          run: async ctx => {
            const out = ctx.kernel();
            for (let i = 0; i < 32; i++) {
              ctx.assertClose(out[i], i, 1e-3, `cell ${i}`);
            }
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            const out = ctx.kernel();
            // 0 + 1 + … + 31, and spot-check both ends and the middle.
            let sum = 0;
            for (let i = 0; i < out.length; i++) sum += out[i];
            ctx.assertClose(sum, (31 * 32) / 2, 1e-2, 'the indices should total 496');
            ctx.assertClose(out[0], 0, 1e-3, 'first thread is index 0');
            ctx.assertClose(out[17], 17, 1e-3, 'thread 17');
            ctx.assertClose(out[31], 31, 1e-3, 'last thread is index 31');
          },
        },
      ],
    },

    // ------------------------------------------------------------------ 3
    {
      slug: 'index-formula',
      title: 'From For-Loop to Formula',
      intro: `<p>Here's the payoff of the thread index. On the CPU you'd sample a sine wave like
        this:</p>
        <p><code>for (let i = 0; i &lt; 64; i++) wave[i] = Math.sin(i / 64 * 2 * Math.PI);</code></p>
        <p>On the GPU, the loop <strong>disappears</strong> — the 64 iterations become 64 threads,
        and the loop variable <code>i</code> becomes <code>this.thread.x</code>. The body of the loop
        is your kernel body, unchanged. (<code>Math.sin</code> and <code>Math.PI</code> work inside
        kernels, along with most of <code>Math</code>.)</p>`,
      goal: `<strong>Goal:</strong> sample one full sine cycle across 64 threads — thread
        <code>x</code> returns <code>Math.sin(x / 64 * 2 * Math.PI)</code>.`,
      requirements: [
        'Keep <code>output: [64]</code> — one thread per sample',
        'Use <code>this.thread.x</code> where the CPU loop used <code>i</code>',
        'Return one sine sample per thread — the CPU loop body, unchanged except for the index',
      ],
      hints: [
        {
          title: 'Hint 1 — the translation rule',
          body: `<p>Take the CPU loop body, delete the loop, and substitute
            <code>this.thread.x</code> for <code>i</code>. That mechanical rewrite is how most
            for-loops become kernels.</p>`,
        },
        {
          title: 'Hint 2 — the body',
          body: `<p><code>return Math.sin(this.thread.x / 64 * 2 * Math.PI);</code></p>`,
        },
      ],
      transfer: `This loop-body-becomes-kernel-body rewrite is called an <em>embarrassingly
        parallel map</em>, and it's the bread and butter of GPGPU: the same move turns a pixel loop
        into a Metal fragment shader, a physics update into a CUDA kernel, or an array transform
        into a WebGPU compute pass.`,
      starterCode: `// The for-loop is gone — 64 threads each compute one sample.
const gpu = new GPU({ mode });

// CPU version, for reference:
//   for (let i = 0; i < 64; i++) wave[i] = Math.sin(i / 64 * 2 * Math.PI);

const wave = gpu.createKernel(function () {
  // TODO: one sample of a sine wave — i is this.thread.x now
  return 0;
}, { output: [64] });

const samples = wave();
console.log(samples);
`,
      solutionCode: `// The for-loop is gone — 64 threads each compute one sample.
const gpu = new GPU({ mode });

const wave = gpu.createKernel(function () {
  return Math.sin(this.thread.x / 64 * 2 * Math.PI);
}, { output: [64] });

const samples = wave();
console.log(samples);
`,
      publicTests: [
        {
          name: 'kernel produces 64 samples',
          run: async ctx => {
            ctx.assert(ctx.kernels.length >= 1, 'no kernel was created — call gpu.createKernel()');
            const out = ctx.kernel();
            ctx.assert(out && out.length === 64, `expected 64 samples, got ${out && out.length}`);
          },
        },
        {
          name: 'samples trace <code>sin(x / 64 · 2π)</code> — starts at 0, peaks at thread 16',
          run: async ctx => {
            const out = ctx.kernel();
            const expected = expectedWave(64);
            ctx.assertClose(out[0], 0, 1e-3, 'thread 0: sin(0) = 0');
            ctx.assertClose(out[16], 1, 1e-3, 'thread 16: quarter cycle, sin = 1');
            ctx.assertClose(out[48], -1, 1e-3, 'thread 48: three-quarter cycle, sin = -1');
            for (let i = 0; i < 64; i += 7) {
              ctx.assertClose(out[i], expected[i], 1e-3, `sample ${i}`);
            }
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            const out = ctx.kernel();
            const expected = expectedWave(64);
            for (let i = 0; i < 64; i++) {
              ctx.assertClose(out[i], expected[i], 1e-3, `sample ${i}`);
            }
          },
        },
      ],
    },

    // ------------------------------------------------------------------ 4
    {
      slug: 'checkerboard',
      title: 'A Second Dimension: this.thread.y',
      intro: `<p>Threads don't have to line up in a row. Give <code>output</code> two numbers —
        <code>output: [8, 8]</code> — and gpu.js launches an 8×8 <strong>grid</strong> of 64
        threads. Each one now has two coordinates: <code>this.thread.x</code> is its column and
        <code>this.thread.y</code> is its row, and the result comes back as an array of rows you
        read as <code>result[y][x]</code>.</p>
        <p>To prove both coordinates are live, paint a classic: a checkerboard. A cell is
        &ldquo;black&rdquo; or &ldquo;white&rdquo; depending on whether <code>x + y</code> is even
        or odd — which is just <code>(x + y) % 2</code>.</p>`,
      goal: `<strong>Goal:</strong> launch an 8×8 grid where each cell holds
        <code>(x + y) % 2</code> — an alternating pattern of 0s and 1s.`,
      requirements: [
        'Change <code>output</code> to a grid: <code>[8, 8]</code>',
        'Use <code>this.thread.x</code> <em>and</em> <code>this.thread.y</code>',
        'Return 0 or 1 in a checkerboard — the parity of the two coordinates added together',
      ],
      hints: [
        {
          title: 'Hint 1 — what changes with 2D?',
          body: `<p>Two things: <code>output</code> gets a second number
            (<code>[width, height]</code>), and <code>this.thread.y</code> starts meaning
            something. Nothing else about the kernel changes.</p>`,
        },
        {
          title: 'Hint 2 — the pattern',
          body: `<p><code>return (this.thread.x + this.thread.y) % 2;</code> — neighbours differ
            by one in <code>x</code> or <code>y</code>, so the parity flips checkerboard-style.</p>`,
        },
      ],
      transfer: `GPUs are built around 2D grids because images are 2D: ROCm and CUDA launch
        <code>dim3</code>-shaped blocks, WebGPU dispatches workgroups across x/y/z, and Metal's
        grids are up to three-dimensional. One thread per pixel — the idea module 1.2 runs with —
        starts exactly here.`,
      starterCode: `// output: [width, height] launches a whole grid of threads.
const gpu = new GPU({ mode });

const board = gpu.createKernel(function () {
  // TODO: return (x + y) % 2 using BOTH thread coordinates
  return this.thread.x % 2;
}, {
  // TODO: make this an 8×8 grid, not an 8-cell line
  output: [8],
});

const result = board();
console.log(result);
`,
      solutionCode: `// output: [width, height] launches a whole grid of threads.
const gpu = new GPU({ mode });

const board = gpu.createKernel(function () {
  return (this.thread.x + this.thread.y) % 2;
}, {
  output: [8, 8],
});

const result = board();
console.log(result);
`,
      publicTests: [
        {
          name: 'result is an 8×8 grid — 8 rows of 8 values',
          run: async ctx => {
            ctx.assert(ctx.kernels.length >= 1, 'no kernel was created — call gpu.createKernel()');
            const out = ctx.kernel();
            ctx.assert(out && out.length === 8, `expected 8 rows, got ${out && out.length}`);
            ctx.assert(
              out[0] && typeof out[0] !== 'number' && out[0].length === 8,
              'each row should hold 8 values — is your output still 1D?'
            );
          },
        },
        {
          name: 'cells alternate like a checkerboard: <code>(x + y) % 2</code>',
          run: async ctx => {
            const out = ctx.kernel();
            ctx.assertClose(out[0][0], 0, 1e-3, 'corner [0][0] is 0');
            ctx.assertClose(out[0][1], 1, 1e-3, 'its neighbour [0][1] is 1');
            ctx.assertClose(out[1][0], 1, 1e-3, 'its neighbour [1][0] is 1');
            ctx.assertClose(out[7][7], 0, 1e-3, 'far corner [7][7] is 0 (7 + 7 is even)');
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            const out = ctx.kernel();
            for (let y = 0; y < 8; y++) {
              for (let x = 0; x < 8; x++) {
                ctx.assertClose(out[y][x], (x + y) % 2, 1e-3, `cell [${y}][${x}]`);
              }
            }
          },
        },
      ],
    },

    // ------------------------------------------------------------------ 5
    {
      slug: 'first-argument',
      title: 'Pass Something In',
      intro: `<p>So far every kernel has conjured its output from thread coordinates alone. Real
        kernels also take <strong>arguments</strong> — declare a parameter on the kernel function,
        pass a value when you call it, and every thread sees that same value. Combine it with
        <code>this.thread.x</code> and each thread computes something different from shared
        input.</p>
        <p>Here's the payoff: a compiled kernel is <strong>reusable</strong>. Build it once, call it
        with <code>2.5</code>, call it again with <code>0.5</code> — two parallel launches, zero
        recompiles. That build-once/call-many rhythm is how all real GPU code is structured.
        One gpu.js habit to pick up now: <code>this.thread.x</code> is an <em>integer</em>, and
        gpu.js's transpiler types a <code>*</code>, <code>+</code> or <code>-</code> expression
        from its <strong>left</strong> operand (division always produces a float) — so write
        <code>scale * this.thread.x</code> (float first) to get float math. That's a quirk of
        this framework, not a GPU law: CUDA promotes mixed int/float math to float, and WGSL or
        GLSL refuse to compile the mix outright.</p>`,
      goal: `<strong>Goal:</strong> make <code>ramp</code> return <code>scale * this.thread.x</code>,
        then call it twice — once with <code>2.5</code>, once with <code>0.5</code>.`,
      requirements: [
        'Give the kernel function a <code>scale</code> parameter',
        'Multiply the shared argument by this thread\'s index — shared argument × thread identity',
        'Keep the float on the left so gpu.js compiles a float multiply, not an integer one',
        'Call the kernel twice with different scales (already wired up)',
      ],
      hints: [
        {
          title: 'Hint 1 — where arguments come from',
          body: `<p>Kernel arguments are ordinary function parameters:
            <code>function (scale) { … }</code>, called as <code>ramp(3)</code>. Every one of the
            64 threads receives the same <code>3</code>.</p>`,
        },
        {
          title: 'Hint 2 — the body',
          body: `<p><code>return scale * this.thread.x;</code> — the argument is shared, the
            index is per-thread, the product is different in every cell. Written the other way
            round (<code>this.thread.x * scale</code>) gpu.js's GL backend compiles an
            <em>integer</em> multiplication and truncates <code>scale</code> — a transpiler
            gotcha specific to gpu.js (CUDA would promote to float; WGSL would refuse the
            mixed types at compile time).</p>`,
        },
      ],
      transfer: `A value shared by all threads is a <em>uniform</em>: WebGPU binds it as a uniform
        buffer, CUDA and ROCm pass it as a kernel launch parameter, Metal hands it over with
        <code>setBytes</code>. And build-once/dispatch-many is universal too — shader and kernel
        compilation is expensive everywhere, so it's paid once up front.`,
      starterCode: `// Arguments are shared by all threads; this.thread.x stays per-thread.
const gpu = new GPU({ mode });

const ramp = gpu.createKernel(function (scale) {
  // TODO: scale this thread's index by the argument
  // (keep the float argument on the LEFT of the multiply)
  return this.thread.x;
}, { output: [64] });

// One kernel, two launches — no recompilation between calls.
console.log('scale 2.5:', ramp(2.5));
console.log('scale 0.5:', ramp(0.5));
`,
      solutionCode: `// Arguments are shared by all threads; this.thread.x stays per-thread.
const gpu = new GPU({ mode });

const ramp = gpu.createKernel(function (scale) {
  // float on the left → float math on the GPU
  return scale * this.thread.x;
}, { output: [64] });

// One kernel, two launches — no recompilation between calls.
console.log('scale 2.5:', ramp(2.5));
console.log('scale 0.5:', ramp(0.5));
`,
      publicTests: [
        {
          name: 'called with <code>2.5</code>, cell <code>i</code> holds <code>i * 2.5</code>',
          run: async ctx => {
            ctx.assert(ctx.kernels.length >= 1, 'no kernel was created — call gpu.createKernel()');
            const out = ctx.kernel(2.5);
            ctx.assert(out && out.length === 64, `expected 64 output values, got ${out && out.length}`);
            for (let i = 0; i < 64; i++) {
              ctx.assertClose(out[i], i * 2.5, 1e-2, `cell ${i} with scale 2.5`);
            }
          },
        },
        {
          name: 'the same kernel re-launches with <code>0.5</code> — no rebuild needed',
          run: async ctx => {
            const out = ctx.kernel(0.5);
            for (let i = 0; i < 64; i++) {
              ctx.assertClose(out[i], i * 0.5, 1e-2, `cell ${i} with scale 0.5`);
            }
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            // A scale the public tests never use — hardcoding their answers fails here.
            const scale = -2.25;
            const out = ctx.kernel(scale);
            ctx.assert(out.length === 64, 'expected 64 output values');
            for (let i = 0; i < 64; i++) {
              ctx.assertClose(out[i], i * scale, 1e-2, `cell ${i} with scale ${scale}`);
            }
          },
        },
      ],
    },
  ],
};
