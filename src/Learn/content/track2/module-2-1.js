// Module 2.1 — Matrix Multiply.
//
// Five tasks: one dot product in one thread → the full 2D matmul grid →
// rectangular shapes → transpose → a size-agnostic kernel using
// dynamicOutput + dynamicArguments + loopMaxIterations.
//
// Kernel-authoring rules (contract): no closures inside kernel functions,
// only numbers / nested number arrays as inputs, this.thread.* for indexing,
// loops statically bounded unless loopMaxIterations is set. Every task
// passes in CPU mode; sizes stay ≤ 64×64 so CPU verification is fast.

// Deterministic vector/matrix builders (shared by inputs() and tests).
// Values are -5…5 with one decimal so float32 sums stay tight.
function makeVector(utils, length, seed) {
  const rand = utils.seededRandom(seed);
  const v = new Array(length);
  for (let i = 0; i < length; i++) v[i] = Math.round(rand() * 100 - 50) / 10;
  return v;
}

function makeMatrix(utils, rows, cols, seed) {
  const rand = utils.seededRandom(seed);
  const m = new Array(rows);
  for (let y = 0; y < rows; y++) {
    const row = new Array(cols);
    for (let x = 0; x < cols; x++) row[x] = Math.round(rand() * 100 - 50) / 10;
    m[y] = row;
  }
  return m;
}

function identityMatrix(n) {
  const m = new Array(n);
  for (let y = 0; y < n; y++) {
    const row = new Array(n).fill(0);
    row[y] = 1;
    m[y] = row;
  }
  return m;
}

function dotRef(a, b) {
  let sum = 0;
  for (let k = 0; k < a.length; k++) sum += a[k] * b[k];
  return sum;
}

// CPU reference: A (n×k) times B (k×m) → n×m.
function matmulRef(a, b) {
  const rows = a.length;
  const inner = b.length;
  const cols = b[0].length;
  const c = new Array(rows);
  for (let y = 0; y < rows; y++) {
    const row = new Array(cols);
    for (let x = 0; x < cols; x++) {
      let sum = 0;
      for (let k = 0; k < inner; k++) sum += a[y][k] * b[k][x];
      row[x] = sum;
    }
    c[y] = row;
  }
  return c;
}

export default {
  id: '2-1',
  track: 2,
  title: 'Matrix Multiply',
  blurb: 'The canonical GPGPU workload: from naive triple loop to a kernel that scales.',
  tasks: [
    // ------------------------------------------------------------------ 1
    {
      slug: 'dot-product',
      title: 'One Cell, One Dot Product',
      intro: `<p>Matrix multiply is the workload GPUs were born for, and every cell of the result
        is the same small machine: a <strong>dot product</strong>. Multiply matching elements of
        two vectors, add the products up, one number comes out. Get one cell right before
        launching a grid of them.</p>
        <p>Notice what is parallel and what is not. The loop over <code>k</code> runs
        <em>sequentially inside one thread</em> — GPUs don't parallelize the sum, they parallelize
        the thousands of <em>independent</em> sums a full matrix needs. This task needs exactly
        one, so the launch is a single thread: <code>output: [1]</code>.</p>`,
      goal: `<strong>Goal:</strong> make the kernel return the dot product of the 16-vectors
        <code>a</code> and <code>b</code> — one output cell holding
        <code>a[0]·b[0] + a[1]·b[1] + … + a[15]·b[15]</code>.`,
      requirements: [
        'Change <code>output</code> to a single cell: <code>[1]</code>',
        'Loop <code>k</code> from 0 to 15 <em>inside</em> the kernel — statically bounded loops are allowed',
        'Accumulate <code>a[k] * b[k]</code> into a running sum and return it',
      ],
      hints: [
        {
          title: 'Hint 1 — a loop? inside a kernel?',
          body: `<p>Yes — as long as the bound is a compile-time constant:
            <code>for (let k = 0; k &lt; 16; k++) { … }</code>. The loop belongs to one thread;
            the parallelism (next task) comes from launching many threads that each own a loop.</p>`,
        },
        {
          title: 'Hint 2 — the whole body',
          body: `<pre><code>let sum = 0;
for (let k = 0; k &lt; 16; k++) {
  sum += a[k] * b[k];
}
return sum;</code></pre>
<p>— and <code>output: [1]</code> so only one thread runs it.</p>`,
        },
      ],
      transfer: `Every GPU linear-algebra library — cuBLAS on CUDA, rocBLAS on ROCm, Metal
        Performance Shaders — bottoms out in this exact shape: one output element, one
        multiply-accumulate loop. All their sophistication goes into feeding that loop faster.`,
      starterCode: `// A dot product folds two 16-vectors into ONE number.
const gpu = new GPU({ mode });

const dot = gpu.createKernel(function (a, b) {
  // TODO: one thread owns the whole sum. Loop k = 0..15,
  // multiply matching elements, add them up, return the total.
  return a[this.thread.x] * b[this.thread.x];
}, {
  // TODO: how many output cells does a dot product have?
  output: [16],
});

console.log(dot(a, b));
`,
      solutionCode: `// A dot product folds two 16-vectors into ONE number.
const gpu = new GPU({ mode });

const dot = gpu.createKernel(function (a, b) {
  let sum = 0;
  for (let k = 0; k < 16; k++) {
    sum += a[k] * b[k];
  }
  return sum;
}, { output: [1] });

console.log(dot(a, b));
`,
      inputs: utils => ({
        a: makeVector(utils, 16, 1101),
        b: makeVector(utils, 16, 1102),
      }),
      publicTests: [
        {
          name: 'output is a single cell — 1 value, not 16',
          run: async ctx => {
            ctx.assert(ctx.kernels.length >= 1, 'no kernel was created — call gpu.createKernel()');
            const ones = new Array(16).fill(1);
            const out = ctx.kernel(ones, ones);
            ctx.assert(
              out && out.length === 1,
              `expected 1 output value, got ${out && out.length} — a dot product is one number`
            );
            ctx.assertClose(out[0], 16, 1e-2, 'dot of two all-ones vectors should be 16');
          },
        },
        {
          name: 'the sum is right: <code>Σ a[k]·b[k]</code>',
          run: async ctx => {
            const a = makeVector(ctx.utils, 16, 1101);
            const b = makeVector(ctx.utils, 16, 1102);
            const out = ctx.kernel(a, b);
            ctx.assertClose(out[0], dotRef(a, b), 1e-2, 'dot product of the provided vectors');
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            // Fresh vectors, plus a basis vector that picks out one element.
            const a = makeVector(ctx.utils, 16, 1177);
            const b = makeVector(ctx.utils, 16, 1178);
            ctx.assertClose(ctx.kernel(a, b)[0], dotRef(a, b), 1e-2, 'dot of fresh vectors');
            const basis = new Array(16).fill(0);
            basis[11] = 1;
            ctx.assertClose(ctx.kernel(a, basis)[0], a[11], 1e-2, 'dot with a basis vector picks a[11]');
          },
        },
      ],
    },

    // ------------------------------------------------------------------ 2
    {
      slug: 'full-matmul',
      title: 'The Full Grid: Matrix × Matrix',
      intro: `<p>On the CPU, <code>C = A × B</code> is the classic triple loop: over rows, over
        columns, over <code>k</code>. On the GPU the outer two loops <strong>vanish into the
        launch</strong> — <code>output: [16, 16]</code> starts 256 threads, one per cell of
        <code>C</code>, and only the innermost loop survives inside the kernel.</p>
        <p>Cell <code>C[y][x]</code> is the dot product of <strong>row y of A</strong> with
        <strong>column x of B</strong>: walk <code>k</code> across the row
        <code>a[y][k]</code> and down the column <code>b[k][x]</code>. Same loop as task 1 —
        now every thread aims it at its own row/column pair.</p>`,
      goal: `<strong>Goal:</strong> compute the 16×16 product <code>matA × matB</code> — each
        thread returns the dot product of its row of <code>a</code> with its column of
        <code>b</code>.`,
      requirements: [
        'Keep <code>output: [16, 16]</code> — one thread per cell of C',
        'Loop <code>k</code> over the 16 shared elements',
        'Accumulate <code>a[this.thread.y][k] * b[k][this.thread.x]</code> and return the sum',
      ],
      hints: [
        {
          title: 'Hint 1 — row and column',
          body: `<p><code>this.thread.y</code> picks the row of <code>a</code>,
            <code>this.thread.x</code> picks the column of <code>b</code>, and <code>k</code> is
            the only index that moves during the loop.</p>`,
        },
        {
          title: 'Hint 2 — the inner loop',
          body: `<pre><code>let sum = 0;
for (let k = 0; k &lt; 16; k++) {
  sum += a[this.thread.y][k] * b[k][this.thread.x];
}
return sum;</code></pre>`,
        },
      ],
      transfer: `This one-thread-per-output-cell matmul is the "naive kernel" every WebGPU and
        CUDA tutorial starts from — and the baseline that tiled, shared-memory versions are
        measured against. The structure you just wrote is their starting point too.`,
      starterCode: `// output: [16, 16] launches 256 threads — one per cell of C.
const gpu = new GPU({ mode });

const multiply = gpu.createKernel(function (a, b) {
  // TODO: this is the ELEMENTWISE product — one term, no loop.
  // C[y][x] needs the whole dot product: loop k over the 16
  // shared elements, walking a's row and b's column.
  return a[this.thread.y][this.thread.x] * b[this.thread.y][this.thread.x];
}, { output: [16, 16] });

const c = multiply(matA, matB);
console.log('C[0][0] =', c[0][0]);
`,
      solutionCode: `// output: [16, 16] launches 256 threads — one per cell of C.
const gpu = new GPU({ mode });

const multiply = gpu.createKernel(function (a, b) {
  let sum = 0;
  for (let k = 0; k < 16; k++) {
    sum += a[this.thread.y][k] * b[k][this.thread.x];
  }
  return sum;
}, { output: [16, 16] });

const c = multiply(matA, matB);
console.log('C[0][0] =', c[0][0]);
`,
      inputs: utils => ({
        matA: makeMatrix(utils, 16, 16, 2101),
        matB: makeMatrix(utils, 16, 16, 2102),
      }),
      publicTests: [
        {
          name: 'result is a 16×16 grid',
          run: async ctx => {
            ctx.assert(ctx.kernels.length >= 1, 'no kernel was created — call gpu.createKernel()');
            const a = makeMatrix(ctx.utils, 16, 16, 2101);
            const b = makeMatrix(ctx.utils, 16, 16, 2102);
            const out = ctx.kernel(a, b);
            ctx.assert(out && out.length === 16, `expected 16 rows, got ${out && out.length}`);
            ctx.assert(out[0] && out[0].length === 16, 'each row should hold 16 values');
          },
        },
        {
          name: 'cells match the dot product of row × column',
          run: async ctx => {
            const a = makeMatrix(ctx.utils, 16, 16, 2101);
            const b = makeMatrix(ctx.utils, 16, 16, 2102);
            const out = ctx.kernel(a, b);
            const ref = matmulRef(a, b);
            const cases = [[0, 0], [3, 12], [8, 8], [15, 1], [15, 15]];
            for (const [y, x] of cases) {
              ctx.assertClose(out[y][x], ref[y][x], 1e-2, `cell [${y}][${x}]`);
            }
          },
        },
        {
          name: 'multiplying by the identity gives A back',
          run: async ctx => {
            const a = makeMatrix(ctx.utils, 16, 16, 2101);
            const out = ctx.kernel(a, identityMatrix(16));
            for (let y = 0; y < 16; y++) {
              for (let x = 0; x < 16; x++) {
                ctx.assertClose(out[y][x], a[y][x], 1e-2, `cell [${y}][${x}] of A × I`);
              }
            }
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            const a = makeMatrix(ctx.utils, 16, 16, 2777);
            const b = makeMatrix(ctx.utils, 16, 16, 2778);
            const out = ctx.kernel(a, b);
            const ref = matmulRef(a, b);
            for (let y = 0; y < 16; y++) {
              for (let x = 0; x < 16; x++) {
                ctx.assertClose(out[y][x], ref[y][x], 1e-2, `cell [${y}][${x}]`);
              }
            }
          },
        },
      ],
    },

    // ------------------------------------------------------------------ 3
    {
      slug: 'rectangular',
      title: 'Rectangular: Three Different Sizes',
      intro: `<p>Square matrices hide a trap: every dimension is 16, so any loop bound "works".
        Real matmuls are rectangular — here <code>rectA</code> is 8×32 (8 rows, 32 columns) and
        <code>rectB</code> is 32×12, so the product is <strong>8×12</strong>. Suddenly there are
        three different sizes and each belongs somewhere specific.</p>
        <p>Two of them shape the launch: <code>output: [width, height]</code> = [columns of B,
        rows of A] = <code>[12, 8]</code> — already set up below. The third, 32, is the
        <strong>shared dimension</strong>: A's columns must equal B's rows, and that's the only
        dimension the loop is allowed to run over.</p>`,
      goal: `<strong>Goal:</strong> compute the 8×12 product <code>rectA × rectB</code> — fix the
        inner loop so it covers the full shared dimension of 32.`,
      requirements: [
        'Keep <code>output: [12, 8]</code> — columns of B across, rows of A down',
        'Loop <code>k</code> over the <em>shared</em> dimension: all 32 of it',
        'Sum <code>a[this.thread.y][k] * b[k][this.thread.x]</code> as before',
      ],
      hints: [
        {
          title: 'Hint 1 — which size does the loop get?',
          body: `<p>The loop walks <em>across</em> a row of A (32 long) and <em>down</em> a column
            of B (also 32 long — that's why the shapes are compatible). Neither 8 nor 12 appears
            in the loop at all.</p>`,
        },
        {
          title: 'Hint 2 — the fix',
          body: `<p>The starter loop stops at 12 — it sums only the first 12 of 32 terms. Change
            the bound: <code>for (let k = 0; k &lt; 32; k++)</code>.</p>`,
        },
      ],
      transfer: `BLAS calls this M, N, K — <code>sgemm(M, N, K, …)</code> in cuBLAS and rocBLAS
        keeps the three sizes as separate parameters for exactly this reason. Mixing them up is
        the classic GEMM bug on every platform, not just here.`,
      starterCode: `// (8×32) times (32×12) → 8×12. Three sizes, three different jobs.
const gpu = new GPU({ mode });

const multiply = gpu.createKernel(function (a, b) {
  let sum = 0;
  // TODO: this loop stops too early — it covers 12 of the 32
  // shared elements. Which of the three sizes does the loop own?
  for (let k = 0; k < 12; k++) {
    sum += a[this.thread.y][k] * b[k][this.thread.x];
  }
  return sum;
}, {
  // [width, height] = [columns of B, rows of A]
  output: [12, 8],
});

const c = multiply(rectA, rectB);
console.log('rows:', c.length, 'cols:', c[0].length);
`,
      solutionCode: `// (8×32) times (32×12) → 8×12. Three sizes, three different jobs.
const gpu = new GPU({ mode });

const multiply = gpu.createKernel(function (a, b) {
  let sum = 0;
  // k runs over the SHARED dimension: A's columns = B's rows = 32.
  for (let k = 0; k < 32; k++) {
    sum += a[this.thread.y][k] * b[k][this.thread.x];
  }
  return sum;
}, {
  // [width, height] = [columns of B, rows of A]
  output: [12, 8],
});

const c = multiply(rectA, rectB);
console.log('rows:', c.length, 'cols:', c[0].length);
`,
      inputs: utils => ({
        rectA: makeMatrix(utils, 8, 32, 3101),
        rectB: makeMatrix(utils, 32, 12, 3102),
      }),
      publicTests: [
        {
          name: 'result is 8 rows × 12 columns',
          run: async ctx => {
            ctx.assert(ctx.kernels.length >= 1, 'no kernel was created — call gpu.createKernel()');
            const a = makeMatrix(ctx.utils, 8, 32, 3101);
            const b = makeMatrix(ctx.utils, 32, 12, 3102);
            const out = ctx.kernel(a, b);
            ctx.assert(out && out.length === 8, `expected 8 rows, got ${out && out.length}`);
            ctx.assert(out[0] && out[0].length === 12, `expected 12 columns, got ${out[0] && out[0].length}`);
          },
        },
        {
          name: 'every term counted — all 32 of the shared dimension',
          run: async ctx => {
            const a = makeMatrix(ctx.utils, 8, 32, 3101);
            const b = makeMatrix(ctx.utils, 32, 12, 3102);
            const out = ctx.kernel(a, b);
            const ref = matmulRef(a, b);
            const cases = [[0, 0], [2, 11], [5, 6], [7, 0], [7, 11]];
            for (const [y, x] of cases) {
              ctx.assertClose(out[y][x], ref[y][x], 2e-2, `cell [${y}][${x}]`);
            }
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            const a = makeMatrix(ctx.utils, 8, 32, 3777);
            const b = makeMatrix(ctx.utils, 32, 12, 3778);
            const out = ctx.kernel(a, b);
            const ref = matmulRef(a, b);
            for (let y = 0; y < 8; y++) {
              for (let x = 0; x < 12; x++) {
                ctx.assertClose(out[y][x], ref[y][x], 2e-2, `cell [${y}][${x}]`);
              }
            }
          },
        },
      ],
    },

    // ------------------------------------------------------------------ 4
    {
      slug: 'transpose',
      title: 'Transpose: Swap the Axes',
      intro: `<p>Look back at the matmul loop: <code>b[k][x]</code> walks <em>down a column</em> —
        each step jumps a whole row of memory. GPUs hate that; neighbouring threads reading
        neighbouring addresses is where their bandwidth comes from. The standard fix is to
        <strong>transpose</strong> B first, turning column walks into row walks.</p>
        <p>A transpose kernel is one line of insight: the thread that owns output cell
        <code>[y][x]</code> reads input cell <code>[x][y]</code>. With a rectangular 24×40 input
        the flip is visible in the shapes too — the result is 40×24, so
        <code>output: [24, 40]</code>.</p>`,
      goal: `<strong>Goal:</strong> transpose the 24×40 matrix <code>matWide</code> — output cell
        <code>[y][x]</code> holds <code>matWide[x][y]</code>, giving a 40×24 result.`,
      requirements: [
        'Keep <code>output: [24, 40]</code> — the transposed width and height',
        'Each thread reads exactly one input cell: indices <em>swapped</em>',
        'No loops — a transpose moves data, it computes nothing',
      ],
      hints: [
        {
          title: 'Hint 1 — who reads what',
          body: `<p>The thread writing output cell <code>[y][x]</code> must read the input cell
            whose row and column are swapped. Both <code>this.thread.x</code> and
            <code>this.thread.y</code> appear — just not in their usual seats.</p>`,
        },
        {
          title: 'Hint 2 — the one-liner',
          body: `<p><code>return m[this.thread.x][this.thread.y];</code></p>`,
        },
      ],
      transfer: `Memory-coalescing is why cuBLAS and rocBLAS pick a different tiled kernel for
        each setting of GEMM's <code>transA/transB</code> flags — whichever layout you pass,
        threads must still read side by side — and why Metal and WebGPU matmul kernels
        pre-stage tiles in threadgroup memory. Reordering data for coalesced access is half of
        GPU performance work.`,
      starterCode: `// The thread for output [y][x] reads input... where?
const gpu = new GPU({ mode });

const transpose = gpu.createKernel(function (m) {
  // TODO: return the input cell with row and column swapped.
  return 0;
}, {
  // input is 24 rows × 40 cols → output is 40 rows × 24 cols
  output: [24, 40],
});

const t = transpose(matWide);
console.log('rows:', t.length, 'cols:', t[0].length);
`,
      solutionCode: `// The thread for output [y][x] reads input... where?
const gpu = new GPU({ mode });

const transpose = gpu.createKernel(function (m) {
  return m[this.thread.x][this.thread.y];
}, {
  // input is 24 rows × 40 cols → output is 40 rows × 24 cols
  output: [24, 40],
});

const t = transpose(matWide);
console.log('rows:', t.length, 'cols:', t[0].length);
`,
      inputs: utils => ({ matWide: makeMatrix(utils, 24, 40, 4101) }),
      publicTests: [
        {
          name: 'shape flips: 24×40 in, 40×24 out',
          run: async ctx => {
            ctx.assert(ctx.kernels.length >= 1, 'no kernel was created — call gpu.createKernel()');
            const m = makeMatrix(ctx.utils, 24, 40, 4101);
            const out = ctx.kernel(m);
            ctx.assert(out && out.length === 40, `expected 40 rows, got ${out && out.length}`);
            ctx.assert(out[0] && out[0].length === 24, `expected 24 columns, got ${out[0] && out[0].length}`);
          },
        },
        {
          name: 'cell [y][x] equals input [x][y]',
          run: async ctx => {
            const m = makeMatrix(ctx.utils, 24, 40, 4101);
            const out = ctx.kernel(m);
            const cases = [[0, 0], [0, 23], [39, 0], [17, 5], [39, 23]];
            for (const [y, x] of cases) {
              ctx.assertClose(out[y][x], m[x][y], 1e-3, `cell [${y}][${x}] should hold input [${x}][${y}]`);
            }
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            const m = makeMatrix(ctx.utils, 24, 40, 4777);
            const out = ctx.kernel(m);
            for (let y = 0; y < 40; y++) {
              for (let x = 0; x < 24; x++) {
                ctx.assertClose(out[y][x], m[x][y], 1e-3, `cell [${y}][${x}]`);
              }
            }
          },
        },
      ],
    },

    // ------------------------------------------------------------------ 5
    {
      slug: 'any-size',
      title: 'One Kernel, Any Size',
      intro: `<p>Every kernel so far had its size welded on: <code>output: [16, 16]</code>, loop
        to 16. Real code multiplies whatever matrices show up. gpu.js has three switches for
        that: <code>dynamicOutput: true</code> lets you call <code>kernel.setOutput([n, n])</code>
        before each run, <code>dynamicArguments: true</code> lets the input arrays change size
        between calls, and <code>loopMaxIterations</code> raises the safety cap so the loop bound
        can be a <em>runtime argument</em> instead of a constant.</p>
        <p>Pass the size in as a plain number, loop <code>k &lt; size</code>, and one kernel
        object serves an 8×8 and a 48×48 multiply back to back. This is the payoff of the module:
        the naive triple loop from task 2, now packaged as a function that scales.</p>`,
      goal: `<strong>Goal:</strong> make <code>multiply(a, b)</code> work for any square size up
        to 64 using a <em>single</em> kernel — verify it on the 8×8 and 48×48 pairs provided.`,
      requirements: [
        'Kernel options: <code>dynamicOutput</code>, <code>dynamicArguments</code>, and <code>loopMaxIterations: 64</code>',
        'Take <code>size</code> as a third kernel argument and loop <code>k &lt; size</code>',
        'In <code>multiply</code>, call <code>matmul.setOutput([n, n])</code> before invoking',
        'Exactly one <code>createKernel</code> call serves both sizes',
      ],
      hints: [
        {
          title: 'Hint 1 — why the cap?',
          body: `<p>On the GPU backend a loop bound that isn't a compile-time constant becomes</p>
<pre><code>for (i = 0; i &lt; LOOP_MAX; i++) {
  if (!(i &lt; size)) break;
  // …
}</code></pre>
<p>in the shader — <code>loopMaxIterations</code> <em>is</em> that LOOP_MAX. Set it to the
            largest size you'll ever pass: 64 here.</p>`,
        },
        {
          title: 'Hint 2 — sizing per call',
          body: `<p>Inside <code>multiply</code>:</p>
<pre><code>const n = a.length;
matmul.setOutput([n, n]);
return matmul(a, b, n);</code></pre>
<p>— set the launch shape first,
            then invoke with the size as the last argument.</p>`,
        },
        {
          title: 'Hint 3 — the kernel',
          body: `<pre><code>function (a, b, size) {
  let sum = 0;
  for (let k = 0; k &lt; size; k++) {
    sum += a[this.thread.y][k] * b[k][this.thread.x];
  }
  return sum;
}</code></pre>
<p>with options</p>
<pre><code>{
  dynamicOutput: true,
  dynamicArguments: true,
  loopMaxIterations: 64,
}</code></pre>`,
        },
      ],
      transfer: `Shipping one kernel that covers a size range is standard practice everywhere:
        CUDA kernels take M, N, K as launch parameters and pick grid dimensions at call time,
        WebGPU dispatches a runtime-computed number of workgroups, and Metal binds sizes through
        a constant buffer. Compile once, launch at any size — exactly what you just built.`,
      starterCode: `// One kernel, any size — no rebuilding between calls.
const gpu = new GPU({ mode });

// TODO: this kernel is welded to 8×8. Free it: dynamicOutput,
// dynamicArguments, loopMaxIterations: 64, and a size argument.
const matmul = gpu.createKernel(function (a, b) {
  let sum = 0;
  for (let k = 0; k < 8; k++) {
    sum += a[this.thread.y][k] * b[k][this.thread.x];
  }
  return sum;
}, { output: [8, 8] });

function multiply(a, b) {
  const n = a.length;
  // TODO: point the kernel at an n×n launch before invoking,
  // and pass n in so the loop knows where to stop.
  return matmul(a, b);
}

console.log('8×8  C[0][0] =', multiply(smallA, smallB)[0][0]);
console.log('48×48 C[0][0] =', multiply(bigA, bigB)[0][0]);
`,
      solutionCode: `// One kernel, any size — no rebuilding between calls.
const gpu = new GPU({ mode });

const matmul = gpu.createKernel(function (a, b, size) {
  let sum = 0;
  for (let k = 0; k < size; k++) {
    sum += a[this.thread.y][k] * b[k][this.thread.x];
  }
  return sum;
}, {
  dynamicOutput: true,
  dynamicArguments: true,
  loopMaxIterations: 64,
});

function multiply(a, b) {
  const n = a.length;
  matmul.setOutput([n, n]);
  return matmul(a, b, n);
}

console.log('8×8  C[0][0] =', multiply(smallA, smallB)[0][0]);
console.log('48×48 C[0][0] =', multiply(bigA, bigB)[0][0]);
`,
      inputs: utils => ({
        smallA: makeMatrix(utils, 8, 8, 5101),
        smallB: makeMatrix(utils, 8, 8, 5102),
        bigA: makeMatrix(utils, 48, 48, 5103),
        bigB: makeMatrix(utils, 48, 48, 5104),
      }),
      publicTests: [
        {
          name: 'one kernel serves both sizes',
          run: async ctx => {
            ctx.assert(
              ctx.kernels.length === 1,
              `expected exactly 1 kernel to handle every size, found ${ctx.kernels.length}`
            );
          },
        },
        {
          name: '8×8 product is correct',
          run: async ctx => {
            const a = makeMatrix(ctx.utils, 8, 8, 5101);
            const b = makeMatrix(ctx.utils, 8, 8, 5102);
            ctx.kernel.setOutput([8, 8]);
            const out = ctx.kernel(a, b, 8);
            const ref = matmulRef(a, b);
            for (let y = 0; y < 8; y++) {
              for (let x = 0; x < 8; x++) {
                ctx.assertClose(out[y][x], ref[y][x], 2e-2, `cell [${y}][${x}]`);
              }
            }
          },
        },
        {
          name: '48×48 product is correct — same kernel, bigger launch',
          run: async ctx => {
            const a = makeMatrix(ctx.utils, 48, 48, 5103);
            const b = makeMatrix(ctx.utils, 48, 48, 5104);
            ctx.kernel.setOutput([48, 48]);
            const out = ctx.kernel(a, b, 48);
            ctx.assert(out.length === 48 && out[0].length === 48, 'expected a 48×48 result');
            const ref = matmulRef(a, b);
            const cases = [[0, 0], [7, 33], [24, 24], [40, 3], [47, 47]];
            for (const [y, x] of cases) {
              ctx.assertClose(out[y][x], ref[y][x], 5e-2, `cell [${y}][${x}]`);
            }
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            // A size never used in the visible tests: 32×32.
            const a = makeMatrix(ctx.utils, 32, 32, 5777);
            const b = makeMatrix(ctx.utils, 32, 32, 5778);
            ctx.kernel.setOutput([32, 32]);
            const out = ctx.kernel(a, b, 32);
            const ref = matmulRef(a, b);
            for (let y = 0; y < 32; y++) {
              for (let x = 0; x < 32; x++) {
                ctx.assertClose(out[y][x], ref[y][x], 5e-2, `cell [${y}][${x}]`);
              }
            }
          },
        },
        {
          name: 'private test #2',
          run: async ctx => {
            // Identity at yet another size — 16×16 × I = the matrix itself.
            const a = makeMatrix(ctx.utils, 16, 16, 5888);
            ctx.kernel.setOutput([16, 16]);
            const out = ctx.kernel(a, identityMatrix(16), 16);
            for (let y = 0; y < 16; y++) {
              for (let x = 0; x < 16; x++) {
                ctx.assertClose(out[y][x], a[y][x], 2e-2, `cell [${y}][${x}] of A × I`);
              }
            }
          },
        },
      ],
    },
  ],
};
