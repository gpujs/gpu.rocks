// Module 1.3 — Thinking in Parallel.
//
// Six tasks: the map pattern → the gather pattern (reverse) → why scatter is
// impossible and how to invert it (circular shift) → boundary clamping →
// a 5-tap moving average → the payoff: a two-pass separable box blur.
//
// Kernel-authoring rules (contract): no closures inside kernel functions,
// only numbers / nested number arrays as inputs, this.thread.* for indexing,
// statically bounded loops, Math.* allowed. Every task passes in CPU mode.

// 64 deterministic temperature readings in °C, −10 … 40, 2 dp.
function makeCelsius(utils, seed = 1303) {
  const rand = utils.seededRandom(seed);
  const data = new Array(64);
  for (let i = 0; i < 64; i++) data[i] = Math.round((rand() * 50 - 10) * 100) / 100;
  return data;
}

// n deterministic samples in 1 … 9, 2 dp — never near zero, so edge tests
// can tell "clamped correctly" apart from "read garbage / zero-fill".
function makeSamples(utils, n, seed = 2718) {
  const rand = utils.seededRandom(seed);
  const data = new Array(n);
  for (let i = 0; i < n; i++) data[i] = Math.round((1 + rand() * 8) * 100) / 100;
  return data;
}

// size×size deterministic heightmap, values 0–1, 3 dp.
function makeGrid(utils, size, seed = 917) {
  const rand = utils.seededRandom(seed);
  const grid = new Array(size);
  for (let y = 0; y < size; y++) {
    const row = new Array(size);
    for (let x = 0; x < size; x++) row[x] = Math.round(rand() * 1000) / 1000;
    grid[y] = row;
  }
  return grid;
}

function clampIndex(i, n) {
  return Math.max(0, Math.min(n - 1, i));
}

// JS reference: 5-tap moving average with clamped edges.
function movingAverageRef(data) {
  const n = data.length;
  const out = new Array(n);
  for (let i = 0; i < n; i++) {
    let sum = 0;
    for (let d = -2; d <= 2; d++) sum += data[clampIndex(i + d, n)];
    out[i] = sum / 5;
  }
  return out;
}

// JS references: 3-tap box blur along one axis with clamped edges.
function blurRowsRef(grid) {
  const n = grid.length;
  return grid.map(row => {
    const out = new Array(n);
    for (let x = 0; x < n; x++) {
      out[x] = (row[clampIndex(x - 1, n)] + row[x] + row[clampIndex(x + 1, n)]) / 3;
    }
    return out;
  });
}

function blurColsRef(grid) {
  const n = grid.length;
  const out = new Array(n);
  for (let y = 0; y < n; y++) {
    const row = new Array(n);
    for (let x = 0; x < n; x++) {
      row[x] = (grid[clampIndex(y - 1, n)][x] + grid[y][x] + grid[clampIndex(y + 1, n)][x]) / 3;
    }
    out[y] = row;
  }
  return out;
}

// Run the learner's two kernels in creation order on a fresh grid.
function composeBlur(ctx, grid) {
  const pass1 = ctx.kernels[0](grid);
  return ctx.kernels[1](pass1);
}

export default {
  id: '1-3',
  track: 1,
  title: 'Thinking in Parallel',
  blurb: 'Map and gather patterns, why kernels write only their own cell, and how to design around it.',
  tasks: [
    // ------------------------------------------------------------------ 1
    {
      slug: 'map-pattern',
      title: 'Map: One Thread, One Value',
      intro: `<p>Nearly every GPU program you'll ever write is built from a handful of patterns,
        and the first one has a name: <strong>map</strong>. Each output cell is a pure function of
        the input cell <em>at the same index</em> — no neighbors, no shared state, no
        "first do cell 3, then cell 4". That independence is exactly what lets the GPU run all
        the cells at once.</p>
        <p>Here <code>celsius</code> holds 64 temperature readings. Converting them to Fahrenheit
        is a textbook map: reading 7 becomes output 7, and nothing else matters to thread 7.</p>`,
      goal: `<strong>Goal:</strong> map every Celsius reading to Fahrenheit —
        <code>°F = °C × 9/5 + 32</code> — one thread per reading.`,
      requirements: [
        'Read only <em>this thread\'s</em> element: <code>celsius[this.thread.x]</code>',
        'Apply the formula <code>c * 9 / 5 + 32</code>',
        'No loops over the array — the grid of threads <em>is</em> the loop',
      ],
      hints: [
        {
          title: 'Hint 1 — the shape of a map',
          body: `<p>A map kernel touches exactly one input cell and one output cell, both at
            index <code>this.thread.x</code>. If you find yourself reading any other index,
            it's not a map any more.</p>`,
        },
        {
          title: 'Hint 2 — the one-liner',
          body: `<pre><code>return celsius[this.thread.x] * 9 / 5 + 32;</code></pre>`,
        },
      ],
      transfer: `Map is the hello-world of every GPU API: a CUDA grid where each thread transforms
        one element of a device array, a WebGPU compute shader dispatched once per buffer entry, a
        Metal compute encoder doing the same. If a problem is a pure map, it parallelizes for free.`,
      starterCode: `// Map: output[i] depends ONLY on input[i]. One thread per reading.
const gpu = new GPU({ mode });

const toFahrenheit = gpu.createKernel(function (celsius) {
  // TODO: convert THIS thread's reading — °F = °C × 9/5 + 32
  return celsius[this.thread.x];
}, { output: [64] });

const result = toFahrenheit(celsius);
console.log('first reading:', celsius[0], '°C →', result[0], '°F');
`,
      solutionCode: `// Map: output[i] depends ONLY on input[i]. One thread per reading.
const gpu = new GPU({ mode });

const toFahrenheit = gpu.createKernel(function (celsius) {
  return celsius[this.thread.x] * 9 / 5 + 32;
}, { output: [64] });

const result = toFahrenheit(celsius);
console.log('first reading:', celsius[0], '°C →', result[0], '°F');
`,
      inputs: utils => ({ celsius: makeCelsius(utils) }),
      publicTests: [
        {
          name: 'converts all 64 readings — one output per thread',
          run: async ctx => {
            ctx.assert(ctx.kernels.length >= 1, 'no kernel was created — call gpu.createKernel()');
            const out = ctx.kernel(makeCelsius(ctx.utils));
            ctx.assert(out && out.length === 64, `expected 64 output values, got ${out && out.length}`);
          },
        },
        {
          name: 'each cell is <code>c × 9/5 + 32</code> — 0 °C → 32 °F, 100 °C → 212 °F',
          run: async ctx => {
            const arr = new Array(64);
            for (let i = 0; i < 64; i++) arr[i] = i * 2 - 20; // includes 0 and 100
            const out = ctx.kernel(arr);
            for (let i = 0; i < 64; i++) {
              ctx.assertClose(out[i], arr[i] * 9 / 5 + 32, 1e-2, `reading ${i} (${arr[i]} °C)`);
            }
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            const data = makeCelsius(ctx.utils, 999);
            const out = ctx.kernel(data);
            ctx.assert(out.length === 64, 'expected 64 output values');
            for (let i = 0; i < 64; i++) {
              ctx.assertClose(out[i], data[i] * 9 / 5 + 32, 1e-2, `reading ${i}`);
            }
          },
        },
      ],
    },

    // ------------------------------------------------------------------ 2
    {
      slug: 'gather-pattern',
      title: 'Gather: Read Anywhere',
      intro: `<p>A map reads its own cell. A <strong>gather</strong> reads <em>any</em> cell —
        the thread computes <strong>where to read from</strong> using its own index. Reads are
        random-access and cheap; it's only <em>writes</em> that are pinned to your own cell
        (the next task is all about that).</p>
        <p>The cleanest possible gather: reverse an array. Thread 0 pulls the last element,
        thread 63 pulls the first — every thread reads exactly one cell, just not its own.
        The array length is wired in as <code>this.constants.n</code>, so the kernel doesn't
        hardcode 64.</p>`,
      goal: `<strong>Goal:</strong> make the kernel return the element from the
        <em>mirrored</em> position, so the output is <code>data</code> reversed.`,
      requirements: [
        'Compute the read index from <code>this.thread.x</code> and <code>this.constants.n</code>',
        'Thread <code>i</code> reads <code>data[n − 1 − i]</code>',
        'No loops, no temporary arrays — one read per thread',
      ],
      hints: [
        {
          title: 'Hint 1 — mirror arithmetic',
          body: `<p>The mirror of index <code>i</code> in an <code>n</code>-element array is
            <code>n − 1 − i</code>: index 0 ↔ index 63, index 1 ↔ index 62, …</p>`,
        },
        {
          title: 'Hint 2 — the one-liner',
          body: `<pre><code>return data[this.constants.n - 1 - this.thread.x];</code></pre>`,
        },
      ],
      transfer: `Gather is why GPUs have texture units: shaders sample textures at arbitrary
        coordinates, CUDA routes scattered reads through <code>__ldg</code> and texture memory,
        WebGPU compute shaders index storage buffers freely. Hardware is built to make "read from
        anywhere" fast.`,
      starterCode: `// A gather kernel computes WHERE to read from its own thread id.
const gpu = new GPU({ mode });

const reverse = gpu.createKernel(function (data) {
  // TODO: read the element from the OTHER end of the array.
  // The array length is available as this.constants.n.
  return data[this.thread.x];
}, {
  output: [64],
  constants: { n: 64 },
});

const result = reverse(data);
console.log('first:', result[0], '(should be the old last:', data[63] + ')');
`,
      solutionCode: `// A gather kernel computes WHERE to read from its own thread id.
const gpu = new GPU({ mode });

const reverse = gpu.createKernel(function (data) {
  return data[this.constants.n - 1 - this.thread.x];
}, {
  output: [64],
  constants: { n: 64 },
});

const result = reverse(data);
console.log('first:', result[0], '(should be the old last:', data[63] + ')');
`,
      inputs: utils => ({ data: makeSamples(utils, 64, 1101) }),
      publicTests: [
        {
          name: 'the ends swap places — <code>out[0] = data[63]</code>, <code>out[63] = data[0]</code>',
          run: async ctx => {
            ctx.assert(ctx.kernels.length >= 1, 'no kernel was created — call gpu.createKernel()');
            const arr = new Array(64);
            for (let i = 0; i < 64; i++) arr[i] = i + 1;
            const out = ctx.kernel(arr);
            ctx.assert(out && out.length === 64, `expected 64 output values, got ${out && out.length}`);
            ctx.assertClose(out[0], 64, 1e-3, 'out[0] should hold the last input value');
            ctx.assertClose(out[63], 1, 1e-3, 'out[63] should hold the first input value');
          },
        },
        {
          name: 'every cell mirrors: <code>out[i] = data[63 − i]</code>',
          run: async ctx => {
            const arr = new Array(64);
            for (let i = 0; i < 64; i++) arr[i] = i * 1.5 + 3;
            const out = ctx.kernel(arr);
            for (let i = 0; i < 64; i++) {
              ctx.assertClose(out[i], arr[63 - i], 1e-3, `element ${i}`);
            }
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            const data = makeSamples(ctx.utils, 64, 4242);
            const out = ctx.kernel(data);
            ctx.assert(out.length === 64, 'expected 64 output values');
            for (let i = 0; i < 64; i++) {
              ctx.assertClose(out[i], data[63 - i], 1e-3, `element ${i}`);
            }
          },
        },
      ],
    },

    // ------------------------------------------------------------------ 3
    {
      slug: 'invert-the-scatter',
      title: 'No Scatter Allowed',
      intro: `<p>Here's the rule that shapes gpu.js kernels (and any fragment shader): a thread
        can read anywhere but can only write <strong>one place — its own cell</strong>, via
        <code>return</code>. There is no <code>out[i + 1] = value</code> here, because 4096
        simultaneous writers into shared cells would be chaos (who wins? in what order?).</p>
        <p>So the "push my value over there" plan — a <strong>scatter</strong> — must be turned
        inside out. Don't ask <em>"where does my value go?"</em>; ask
        <em>"whose value lands in <strong>my</strong> cell?"</em> — a gather. Try it on a rotation:
        every value moves one slot to the <em>right</em>, and the last wraps around to slot 0.</p>`,
      goal: `<strong>Goal:</strong> rotate <code>ring</code> one slot to the right by gathering:
        each thread pulls the value that belongs in its cell.`,
      requirements: [
        'No writes to other cells — express the shift purely as a read',
        'Thread <code>i</code> pulls from index <code>i − 1</code>',
        'Thread 0 wraps around and pulls the <em>last</em> element',
      ],
      hints: [
        {
          title: 'Hint 1 — invert the direction',
          body: `<p>If every value moves right by one, then the value in <em>my</em> cell came
            from my <em>left</em>: index <code>this.thread.x - 1</code>. The starter currently
            pulls from the right — that rotates the wrong way.</p>`,
        },
        {
          title: 'Hint 2 — wrapping without an if',
          body: `<p>Adding <code>n</code> before the modulo keeps the index positive:</p>
<pre><code>(this.thread.x - 1 + this.constants.n) % this.constants.n</code></pre>
<p>That turns <code>-1</code> into <code>63</code> and leaves 1…63 alone.</p>`,
        },
      ],
      transfer: `Compute APIs relax this ban: CUDA, WebGPU and ROCm threads <em>can</em> store to
        any buffer address (scatter), and neighbours in a block cooperate through workgroup
        memory. But two threads storing to the <em>same</em> address is still a data race, and
        the escape hatch — atomics like <code>atomicAdd</code> — serializes threads and costs
        dearly. That's why GPU folklore compresses this lesson into four words:
        <em>turn scatter into gather</em>.`,
      starterCode: `// There is no out[i + 1] = value on a GPU. Threads only fill their OWN cell.
const gpu = new GPU({ mode });

// Wanted: every value moves one slot RIGHT, the last wraps to slot 0.
// You can't push your value right — so pull the right value in.
const rotate = gpu.createKernel(function (ring) {
  // TODO: this pulls from the wrong side — it rotates LEFT. Fix the
  // gather so each thread pulls the value that belongs in its cell.
  return ring[(this.thread.x + 1) % this.constants.n];
}, {
  output: [64],
  constants: { n: 64 },
});

const result = rotate(ring);
console.log('ring[0] was', ring[0], '— it should now sit at result[1]:', result[1]);
`,
      solutionCode: `// There is no out[i + 1] = value on a GPU. Threads only fill their OWN cell.
const gpu = new GPU({ mode });

// "Whose value lands in MY cell?" — the one from my left, wrapping at 0.
const rotate = gpu.createKernel(function (ring) {
  return ring[(this.thread.x - 1 + this.constants.n) % this.constants.n];
}, {
  output: [64],
  constants: { n: 64 },
});

const result = rotate(ring);
console.log('ring[0] was', ring[0], '— it should now sit at result[1]:', result[1]);
`,
      inputs: utils => ({ ring: makeSamples(utils, 64, 3301) }),
      publicTests: [
        {
          name: 'values move one slot right: <code>out[i] = ring[i − 1]</code>',
          run: async ctx => {
            ctx.assert(ctx.kernels.length >= 1, 'no kernel was created — call gpu.createKernel()');
            const arr = new Array(64);
            for (let i = 0; i < 64; i++) arr[i] = i * 2 + 5;
            const out = ctx.kernel(arr);
            ctx.assert(out && out.length === 64, `expected 64 output values, got ${out && out.length}`);
            for (let i = 1; i < 64; i++) {
              ctx.assertClose(out[i], arr[i - 1], 1e-3, `element ${i} should hold ring[${i - 1}]`);
            }
          },
        },
        {
          name: 'the first cell wraps around: <code>out[0] = ring[63]</code>',
          run: async ctx => {
            const arr = new Array(64);
            for (let i = 0; i < 64; i++) arr[i] = i + 10;
            const out = ctx.kernel(arr);
            ctx.assertClose(out[0], arr[63], 1e-3, 'out[0] should hold the last input value');
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            const ring = makeSamples(ctx.utils, 64, 8088);
            const out = ctx.kernel(ring);
            ctx.assert(out.length === 64, 'expected 64 output values');
            for (let i = 0; i < 64; i++) {
              ctx.assertClose(out[i], ring[(i - 1 + 64) % 64], 1e-3, `element ${i}`);
            }
          },
        },
      ],
    },

    // ------------------------------------------------------------------ 4
    {
      slug: 'edges-and-clamps',
      title: 'Life on the Edge',
      intro: `<p>The moment a gather reads a <em>neighbor</em>, the edges bite. Take the forward
        difference — <code>out[i] = signal[i+1] − signal[i]</code>, "how much does the signal jump
        here?". Thread 63 asks for <code>signal[64]</code>, which doesn't exist. On the CPU that
        read is <code>NaN</code>; on the GPU it's whatever the hardware feels like. Either way,
        garbage.</p>
        <p>The standard fix is <strong>clamping</strong>: pin the index inside
        <code>0 … n−1</code> with <code>Math.min</code>/<code>Math.max</code> before reading.
        The clamped edge cell reads itself, so its difference is exactly 0 — a defined,
        deliberate answer instead of an accident.</p>`,
      goal: `<strong>Goal:</strong> compute the forward difference with a clamped read, so the
        last cell returns exactly <code>0</code> instead of garbage.`,
      requirements: [
        'Clamp the neighbor index: <code>Math.min(this.thread.x + 1, this.constants.n - 1)</code>',
        'Interior cells still return <code>signal[i+1] − signal[i]</code>',
        'The last cell returns exactly <code>0</code>',
      ],
      hints: [
        {
          title: 'Hint 1 — where the garbage comes from',
          body: `<p>Only thread 63 misbehaves: <code>this.thread.x + 1</code> is 64, one past the
            end. Every other thread's read is fine — the fix should change <em>only</em> that
            one thread's read.</p>`,
        },
        {
          title: 'Hint 2 — the clamped read',
          body: `<pre><code>const j = Math.min(this.thread.x + 1, this.constants.n - 1);
return signal[j] - signal[this.thread.x];</code></pre>
<p>For thread 63, <code>j</code> is 63, and <code>signal[63] - signal[63]</code> is 0.</p>`,
        },
      ],
      transfer: `Graphics hardware ships this as a sampler setting — <code>clamp-to-edge</code>
        address mode in WebGPU and Metal, <code>cudaAddressModeClamp</code> on CUDA texture
        objects. Reading a raw buffer instead of a texture? Then you clamp by hand, exactly like
        here.`,
      starterCode: `// Forward difference: out[i] = signal[i + 1] - signal[i].
const gpu = new GPU({ mode });

const delta = gpu.createKernel(function (signal) {
  // TODO: thread 63 reads signal[64] — one past the end. Clamp the
  // neighbor index so the last cell returns 0 instead of garbage.
  return signal[this.thread.x + 1] - signal[this.thread.x];
}, {
  output: [64],
  constants: { n: 64 },
});

const result = delta(signal);
console.log('last delta (should be exactly 0):', result[63]);
`,
      solutionCode: `// Forward difference: out[i] = signal[i + 1] - signal[i].
const gpu = new GPU({ mode });

const delta = gpu.createKernel(function (signal) {
  const j = Math.min(this.thread.x + 1, this.constants.n - 1);
  return signal[j] - signal[this.thread.x];
}, {
  output: [64],
  constants: { n: 64 },
});

const result = delta(signal);
console.log('last delta (should be exactly 0):', result[63]);
`,
      inputs: utils => ({ signal: makeSamples(utils, 64, 5150) }),
      publicTests: [
        {
          name: 'interior cells hold the jump: <code>out[i] = signal[i+1] − signal[i]</code>',
          run: async ctx => {
            ctx.assert(ctx.kernels.length >= 1, 'no kernel was created — call gpu.createKernel()');
            const data = makeSamples(ctx.utils, 64, 5150);
            const out = ctx.kernel(data);
            ctx.assert(out && out.length === 64, `expected 64 output values, got ${out && out.length}`);
            for (let i = 0; i < 63; i++) {
              ctx.assertClose(out[i], data[i + 1] - data[i], 1e-3, `element ${i}`);
            }
          },
        },
        {
          name: 'the last cell clamps to <code>0</code> — no read past the end',
          run: async ctx => {
            const data = makeSamples(ctx.utils, 64, 5150);
            const out = ctx.kernel(data);
            ctx.assert(Number.isFinite(out[63]), `out[63] is ${out[63]} — an out-of-bounds read`);
            ctx.assertClose(out[63], 0, 1e-4, 'the clamped edge cell should be exactly 0');
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            const data = makeSamples(ctx.utils, 64, 6006);
            const out = ctx.kernel(data);
            for (let i = 0; i < 63; i++) {
              ctx.assertClose(out[i], data[i + 1] - data[i], 1e-3, `element ${i}`);
            }
            ctx.assertClose(out[63], 0, 1e-4, 'last cell should clamp to 0');
          },
        },
      ],
    },

    // ------------------------------------------------------------------ 5
    {
      slug: 'moving-average',
      title: 'Smooth a Signal',
      intro: `<p>Time to combine everything: a <strong>5-tap moving average</strong>. Each output
        cell is the mean of <code>signal[x−2 … x+2]</code> — a gather over a small
        <em>window</em> of neighbors, with clamping where the window hangs off either end. This
        shape — loop over a fixed window, clamp, accumulate — is called a
        <strong>stencil</strong>, and it powers blurs, edge detectors, and physics simulations
        alike.</p>
        <p>Yes, a loop <em>inside</em> the kernel is fine: it's 5 iterations of private
        arithmetic per thread, not a loop over the data. 128 threads each averaging 5 numbers
        is still one parallel pass.</p>`,
      goal: `<strong>Goal:</strong> each cell returns the average of the five values centered on
        it, with window indexes clamped to <code>0 … n−1</code>.`,
      requirements: [
        'Loop over the window: <code>for (let d = 0; d < 5; d++)</code> with offset <code>d − 2</code>',
        'Clamp every read with <code>Math.max(0, Math.min(n − 1, …))</code>',
        'Return the sum divided by <code>5</code>',
      ],
      hints: [
        {
          title: 'Hint 1 — the window',
          body: `<p>The five indexes are <code>this.thread.x + d - 2</code> for
            <code>d = 0…4</code>: two to the left, itself, two to the right.</p>`,
        },
        {
          title: 'Hint 2 — clamp inside the loop',
          body: `<p>Each iteration:</p>
<pre><code>const j = Math.max(0, Math.min(this.constants.n - 1, this.thread.x + d - 2));
sum += signal[j];</code></pre>`,
        },
        {
          title: 'Hint 3 — sanity-check the edge',
          body: `<p>Cell 0's clamped window reads indexes <code>0, 0, 0, 1, 2</code> — so
            <code>out[0]</code> should equal <code>(3·signal[0] + signal[1] + signal[2]) / 5</code>.</p>`,
        },
      ],
      transfer: `Windowed sums over neighbors are stencil computations — the bread and butter of
        scientific codes on CUDA and ROCm, where entire papers are devoted to tiling stencils into
        shared memory so the window reads come from fast on-chip storage instead of DRAM.`,
      starterCode: `// A 5-tap stencil: mean of signal[x-2 ... x+2], edges clamped.
const gpu = new GPU({ mode });

const smooth = gpu.createKernel(function (signal) {
  let sum = 0;
  for (let d = 0; d < 5; d++) {
    // TODO: read the window neighbor at offset d - 2,
    // clamped to 0 ... this.constants.n - 1
    sum += signal[this.thread.x];
  }
  return sum / 5;
}, {
  output: [128],
  constants: { n: 128 },
});

const result = smooth(signal);
console.log('raw:', signal[64], '→ smoothed:', result[64]);
`,
      solutionCode: `// A 5-tap stencil: mean of signal[x-2 ... x+2], edges clamped.
const gpu = new GPU({ mode });

const smooth = gpu.createKernel(function (signal) {
  let sum = 0;
  for (let d = 0; d < 5; d++) {
    const j = Math.max(0, Math.min(this.constants.n - 1, this.thread.x + d - 2));
    sum += signal[j];
  }
  return sum / 5;
}, {
  output: [128],
  constants: { n: 128 },
});

const result = smooth(signal);
console.log('raw:', signal[64], '→ smoothed:', result[64]);
`,
      inputs: utils => ({ signal: makeSamples(utils, 128, 7203) }),
      publicTests: [
        {
          name: 'mid-signal cells average their five neighbors',
          run: async ctx => {
            ctx.assert(ctx.kernels.length >= 1, 'no kernel was created — call gpu.createKernel()');
            const data = makeSamples(ctx.utils, 128, 7203);
            const out = ctx.kernel(data);
            ctx.assert(out && out.length === 128, `expected 128 output values, got ${out && out.length}`);
            const ref = movingAverageRef(data);
            for (const i of [2, 17, 64, 99, 125]) {
              ctx.assertClose(out[i], ref[i], 1e-3, `element ${i}`);
            }
          },
        },
        {
          name: 'edge cells clamp — <code>out[0]</code> averages indexes 0, 0, 0, 1, 2',
          run: async ctx => {
            const data = makeSamples(ctx.utils, 128, 7203);
            const out = ctx.kernel(data);
            const n = data.length;
            ctx.assertClose(out[0], (3 * data[0] + data[1] + data[2]) / 5, 1e-3, 'left edge');
            ctx.assertClose(
              out[n - 1],
              (3 * data[n - 1] + data[n - 2] + data[n - 3]) / 5,
              1e-3,
              'right edge'
            );
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            const data = makeSamples(ctx.utils, 128, 9090);
            const out = ctx.kernel(data);
            const ref = movingAverageRef(data);
            for (let i = 0; i < 128; i++) {
              ctx.assertClose(out[i], ref[i], 1e-3, `element ${i}`);
            }
          },
        },
      ],
    },

    // ------------------------------------------------------------------ 6
    {
      slug: 'two-pass-blur',
      title: 'The Two-Pass Blur',
      intro: `<p>The payoff. A 3×3 box blur of a 2D grid needs nine reads per cell — but the box
        blur is <strong>separable</strong>: blurring horizontally and then blurring that result
        vertically gives the <em>identical</em> answer with just three reads per cell per pass.
        Bigger blurs win bigger: a 9×9 blur drops from 81 reads to 18.</p>
        <p>This is also how you design around the no-communication rule at scale: since threads
        can't share work <em>within</em> a pass, you split the algorithm into passes — each pass
        a clean parallel gather, each handoff a finished grid. Kernel one blurs along
        <code>x</code>; its output feeds kernel two, which blurs along <code>y</code>. Both are
        3-tap clamped stencils — task 5, twice, at right angles.</p>`,
      goal: `<strong>Goal:</strong> finish both kernels — <code>blurX</code> averages each cell
        with its left/right neighbors, <code>blurY</code> with its up/down neighbors — edges
        clamped, so the composition equals a full 3×3 box blur.`,
      requirements: [
        '<code>blurX</code>: 3-tap average along the row — clamp <code>x + d − 1</code>, read <code>grid[this.thread.y][j]</code>',
        '<code>blurY</code>: 3-tap average down the column — clamp <code>y + d − 1</code>, read <code>grid[j][this.thread.x]</code>',
        'Both kernels divide their sum by <code>3</code>',
        'Feed <code>blurX</code>\'s output into <code>blurY</code> (already wired up)',
      ],
      hints: [
        {
          title: 'Hint 1 — task 5, rotated',
          body: `<p>Each kernel is the moving-average pattern with a 3-wide window. The only new
            move: in 2D you clamp the coordinate along the blur axis and keep the other
            coordinate fixed.</p>`,
        },
        {
          title: 'Hint 2 — the x pass',
          body: `<pre><code>for (let d = 0; d &lt; 3; d++) {
  const j = Math.max(0, Math.min(this.constants.n - 1, this.thread.x + d - 1));
  sum += grid[this.thread.y][j];
}
return sum / 3;</code></pre>
<p>The y pass swaps which coordinate is clamped: <code>grid[j][this.thread.x]</code>.</p>`,
        },
      ],
      transfer: `Separable filtering is a classic GPU optimization you'll meet everywhere: game
        engines render Gaussian blurs as two fullscreen passes, WebGPU and Metal chain compute
        encoder passes the same way, and CUDA image pipelines launch one kernel per axis. Two
        cheap 1D passes beating one fat 2D pass — O(k) taps instead of O(k²) — never stops
        being true.`,
      starterCode: `// Two passes at right angles = one 3×3 box blur, for 6 reads instead of 9.
const gpu = new GPU({ mode });

const blurX = gpu.createKernel(function (grid) {
  // TODO: average grid[y][x-1], grid[y][x], grid[y][x+1] — clamp x
  return grid[this.thread.y][this.thread.x];
}, { output: [48, 48], constants: { n: 48 } });

const blurY = gpu.createKernel(function (grid) {
  // TODO: average grid[y-1][x], grid[y][x], grid[y+1][x] — clamp y
  return grid[this.thread.y][this.thread.x];
}, { output: [48, 48], constants: { n: 48 } });

const pass1 = blurX(heightmap);
const smooth = blurY(pass1);
console.log('corner before → after:', heightmap[0][0], '→', smooth[0][0]);
`,
      solutionCode: `// Two passes at right angles = one 3×3 box blur, for 6 reads instead of 9.
const gpu = new GPU({ mode });

const blurX = gpu.createKernel(function (grid) {
  let sum = 0;
  for (let d = 0; d < 3; d++) {
    const j = Math.max(0, Math.min(this.constants.n - 1, this.thread.x + d - 1));
    sum += grid[this.thread.y][j];
  }
  return sum / 3;
}, { output: [48, 48], constants: { n: 48 } });

const blurY = gpu.createKernel(function (grid) {
  let sum = 0;
  for (let d = 0; d < 3; d++) {
    const j = Math.max(0, Math.min(this.constants.n - 1, this.thread.y + d - 1));
    sum += grid[j][this.thread.x];
  }
  return sum / 3;
}, { output: [48, 48], constants: { n: 48 } });

const pass1 = blurX(heightmap);
const smooth = blurY(pass1);
console.log('corner before → after:', heightmap[0][0], '→', smooth[0][0]);
`,
      inputs: utils => ({ heightmap: makeGrid(utils, 48) }),
      publicTests: [
        {
          name: 'two passes compose into a 48×48 grid',
          run: async ctx => {
            ctx.assert(ctx.kernels.length >= 2, `expected 2 kernels, found ${ctx.kernels.length}`);
            const out = composeBlur(ctx, makeGrid(ctx.utils, 48));
            ctx.assert(out && out.length === 48, `expected 48 rows, got ${out && out.length}`);
            ctx.assert(out[0] && out[0].length === 48, 'each row should hold 48 values');
          },
        },
        {
          name: 'interior cells equal the full 3×3 box average',
          run: async ctx => {
            const grid = makeGrid(ctx.utils, 48);
            const out = composeBlur(ctx, grid);
            const ref = blurColsRef(blurRowsRef(grid));
            for (const [y, x] of [[1, 1], [10, 30], [24, 24], [40, 7], [46, 46]]) {
              ctx.assertClose(out[y][x], ref[y][x], 2e-3, `cell [${y}][${x}]`);
            }
          },
        },
        {
          name: 'edges and corners clamp — no zero-padding creeping in',
          run: async ctx => {
            const grid = makeGrid(ctx.utils, 48);
            const out = composeBlur(ctx, grid);
            const ref = blurColsRef(blurRowsRef(grid));
            for (const [y, x] of [[0, 0], [0, 47], [47, 0], [47, 47], [0, 20], [20, 0]]) {
              ctx.assertClose(out[y][x], ref[y][x], 2e-3, `cell [${y}][${x}]`);
            }
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            const grid = makeGrid(ctx.utils, 48, 555);
            const out = composeBlur(ctx, grid);
            const ref = blurColsRef(blurRowsRef(grid));
            for (let y = 0; y < 48; y++) {
              for (let x = 0; x < 48; x++) {
                ctx.assertClose(out[y][x], ref[y][x], 2e-3, `cell [${y}][${x}]`);
              }
            }
          },
        },
      ],
    },
  ],
};
