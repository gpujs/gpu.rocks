// Module: Escape-Time Fractals — uuid 0de4764c-e40f-4966-9014-05e3a26c0eec (short id 0de4764c).
// The file name is the uuid; identity lives in the exported object below,
// never in the path. Legacy id (pre-uuid URLs, localStorage migration): 3-2.
//
// Escape-Time Fractals.
//
// Five tasks: mapping pixels onto the complex plane → the Mandelbrot
// escape-time loop → coloring by iteration count → smooth (fractional)
// coloring → Julia sets with c as live kernel arguments.
//
// Kernel-authoring rules (contract): no closures inside kernel functions,
// only numbers / nested number arrays as inputs, this.thread.* for indexing,
// graphical kernels use this.color(). Loops are statically bounded with a
// literal 100. Every task passes in CPU mode.

const MAX_ITER = 100;

// Float64 twin of the kernels' guarded escape loop, shared by all tests.
// z starts at (zr0, zi0); each of MAX_ITER passes advances z → z² + c only
// while |z|² < 4, counting the passes that ran. z freezes at escape.
function escapeOrbit(zr0, zi0, cr, ci, maxIter = MAX_ITER) {
  let zr = zr0;
  let zi = zi0;
  let count = 0;
  for (let i = 0; i < maxIter; i++) {
    if (zr * zr + zi * zi < 4) {
      const zrNext = zr * zr - zi * zi + cr;
      zi = 2 * zr * zi + ci;
      zr = zrNext;
      count += 1;
    }
  }
  return { count, zr, zi };
}

// Smooth (fractional) escape value: count + 1 − log2(log2|z|), or maxIter
// for points that never escape. Matches the kernel formula in tasks 4–5.
function smoothEscape(zr0, zi0, cr, ci, maxIter = MAX_ITER) {
  const { count, zr, zi } = escapeOrbit(zr0, zi0, cr, ci, maxIter);
  if (count >= maxIter) return maxIter;
  return count + 1 - Math.log2(0.5 * Math.log2(zr * zr + zi * zi));
}

// The module's palette, in canvas bytes: shade t → [r, g, b].
function paletteBytes(t) {
  return [
    Math.round(t * 255),
    Math.round(t * t * 255),
    Math.round((0.5 + 0.5 * t) * 255),
  ];
}

// ---- near-miss diagnosis --------------------------------------------------
//
// A failing assert that reports only two numbers tells a learner nothing about
// WHICH slip produced them. A probe pairs the value one specific known mistake
// would produce with a sentence naming that mistake; diagnose() speaks only
// when the observed value matches a probe within the test's own tolerance AND
// the correct value does not — so cells where a candidate happens to equal the
// right answer stay silent, as do observations matching probes that disagree
// with each other. A wrong diagnosis is worse than none.
function diagnose(got, expected, eps, probes) {
  const hits = probes
    .filter(p => Math.abs(got - p[0]) <= eps && Math.abs(expected - p[0]) > eps)
    .map(p => p[1]);
  return hits.length && hits.every(m => m === hits[0]) ? hits[0] : null;
}

// Task 1: the plane mapping. (Until gpu.js 2.20 the GL backend typed a mixed
// multiply from its left operand, so an integer thread id on the left silently
// truncated step to 0 and every cell reported the view's corner; that is fixed
// upstream and the probe for it has been retired.)
function planeProbes(xMin, yMin, step, x, y) {
  const cr = xMin + x * step;
  const ci = yMin + y * step;
  return [
    [Math.sqrt(cr * cr + ci * ci),
      'that is |c|, not |c|² — return cr * cr + ci * ci without the square root'],
    [x * step * (x * step) + y * step * (y * step),
      'the view offsets never got added — cr is xMin + x * step, ci is yMin + y * step'],
  ];
}

// Task 2: the escape guard, missing or inverted. 100 everywhere means counting
// continued after z escaped; 0 everywhere means no pass ever ran.
function escapeProbes() {
  return [
    [MAX_ITER,
      'every point reached the 100 cap — counting continued after z escaped, so the guard zr * zr + zi * zi < 4 is either missing or not wrapping the count'],
    [0,
      'count came back 0 — no guarded pass ever ran; the guard admits z while zr * zr + zi * zi is BELOW 4'],
  ];
}

// Task 3: interior pixels painted with the escape shade instead of black. At
// count = 100 the shade is t = 1, so a white interior names its own mistake.
function interiorHint(r, g, b) {
  return r >= 253 && g >= 253 && b >= 253
    ? 'the interior came out white — count = 100 pixels are falling into the shade branch; shade only when count < 100 and paint the rest black'
    : null;
}

// Task 4: the normalized iteration count, one term at a time.
function smoothProbes(cr, ci) {
  const orbit = escapeOrbit(cr, ci);
  const r2 = orbit.zr * orbit.zr + orbit.zi * orbit.zi;
  return [
    [orbit.count,
      'that is the raw integer count — the fractional correction 1 − log2(0.5 · log2|z|²) is missing'],
    [orbit.count + 1 - Math.log(0.5 * Math.log(r2)),
      'Math.log is the natural logarithm — the normalized iteration count takes log2 twice'],
    [orbit.count + 1 - Math.log2(Math.log2(r2)),
      'the halving is missing — log2|z| is 0.5 * Math.log2(zr * zr + zi * zi)'],
  ];
}

export default {
  uuid: '0de4764c-e40f-4966-9014-05e3a26c0eec',
  version: 1,
  slug: 'escape-time-fractals',
  legacyId: '3-2',
  title: 'Escape-Time Fractals',
  blurb: 'Mandelbrot and Julia sets with smooth coloring — infinite detail from a ten-line kernel.',
  tasks: [
    // ------------------------------------------------------------------ 1
    {
      slug: 'pixel-to-plane',
      title: 'Map Pixels to the Complex Plane',
      intro: `<p>A fractal isn't drawn — it's <strong>evaluated</strong>. There is a function
        defined on the complex plane, and every pixel asks: what does that function do
        <em>at my point</em>? So before any fractal math, each thread must know which complex
        number it owns.</p>
        <p>Three numbers describe the camera: <code>xMin</code> and <code>yMin</code> pin the
        bottom-left corner of the view, and <code>step</code> is the width of one pixel in plane
        units. Thread <code>(x, y)</code> then sits at <code>c = (xMin + x·step) +
        (yMin + y·step)·i</code>. Change the three numbers and the same kernel becomes a zoom lens.</p>`,
      goal: `<strong>Goal:</strong> map each thread to its point <code>(cr, ci)</code> on the
        complex plane and return the squared magnitude <code>cr² + ci²</code> — a distance field
        we can sanity-check before iterating anything.`,
      requirements: [
        'Hoist the thread ids into consts: <code>const x = this.thread.x</code> — as a const it becomes a float you can scale',
        'Map to the plane: <code>cr = xMin + x * step</code>, <code>ci = yMin + y * step</code>',
        'Return <code>cr * cr + ci * ci</code> — the squared distance from the origin',
      ],
      hints: [
        {
          title: 'Hint 1 — pixels are integers, planes are not',
          body: `<p><code>this.thread.x</code> counts 0…63 — and it's an <em>integer</em>. Assign it
            to a const first (<code>const x = this.thread.x;</code>) so the GPU treats it as a float;
            then <code>x * step</code> turns pixel counts into plane distance, and adding
            <code>xMin</code> slides the view into place. Same story for y.</p>`,
        },
        {
          title: 'Hint 2 — the whole body',
          body: `<pre><code>const x = this.thread.x;
const y = this.thread.y;
const cr = xMin + x * step;
const ci = yMin + y * step;
return cr * cr + ci * ci;</code></pre>`,
        },
      ],
      transfer: `Index-to-domain mapping is step one of nearly every GPU program: fragment
        shaders scale normalized uv coordinates into world space, CUDA turns
        <code>blockIdx * blockDim + threadIdx</code> into a grid coordinate, WebGPU does the same
        with <code>global_invocation_id</code>. Integer id in, domain point out.`,
      starterCode: `// Which complex number does THIS pixel own?
const gpu = new GPU({ mode });

const distanceField = gpu.createKernel(function (xMin, yMin, step) {
  // TODO: map this thread onto the complex plane:
  //   const x = this.thread.x;   ← hoisting makes it a float
  //   cr = xMin + x * step   (and the same for ci with y)
  // then return the squared magnitude cr² + ci².
  return this.thread.x;
}, { output: [64, 64] });

const field = await distanceField(-2, -2, 4 / 64);
console.log('cell [32][32] sits at the origin:', field[32][32]);
console.log('corner cell [0][0]:', field[0][0]);
`,
      solutionCode: `// Which complex number does THIS pixel own?
const gpu = new GPU({ mode });

const distanceField = gpu.createKernel(function (xMin, yMin, step) {
  const x = this.thread.x;
  const y = this.thread.y;
  const cr = xMin + x * step;
  const ci = yMin + y * step;
  return cr * cr + ci * ci;
}, { output: [64, 64] });

const field = await distanceField(-2, -2, 4 / 64);
console.log('cell [32][32] sits at the origin:', field[32][32]);
console.log('corner cell [0][0]:', field[0][0]);
`,
      publicTests: [
        {
          name: 'the view <code>(-2, -2, 4/64)</code> puts the origin at cell [32][32]',
          run: async ctx => {
            ctx.assert(ctx.kernels.length >= 1, 'no kernel was created — call gpu.createKernel()');
            const out = await ctx.kernel(-2, -2, 4 / 64);
            ctx.assert(out && out.length === 64, `expected 64 rows, got ${out && out.length}`);
            ctx.assert(out[0] && out[0].length === 64, 'each row should hold 64 values');
            const step = 4 / 64;
            const planeHint = (y, x, expected, eps) =>
              diagnose(out[y][x], expected, eps, planeProbes(-2, -2, step, x, y));
            ctx.assertClose(out[32][32], 0, 1e-3, planeHint(32, 32, 0, 1e-3) ||
              'cell [32][32] should be the origin, |c|² = 0');
            ctx.assertClose(out[32][48], 1, 1e-3, planeHint(32, 48, 1, 1e-3) ||
              'cell [32][48] sits at c = 1 + 0i, so |c|² = 1');
            ctx.assertClose(out[0][0], 8, 1e-2, planeHint(0, 0, 8, 1e-2) ||
              'cell [0][0] sits at c = -2 - 2i, so |c|² = 8');
          },
        },
        {
          name: 'a different camera — <code>(0, 0, 0.5)</code> — moves every cell',
          run: async ctx => {
            const out = await ctx.kernel(0, 0, 0.5);
            const cases = [[0, 0], [2, 3], [7, 7], [63, 1]];
            for (const [y, x] of cases) {
              const expected = 0.25 * (x * x + y * y);
              const hint = diagnose(out[y][x], expected, 1e-2, planeProbes(0, 0, 0.5, x, y));
              ctx.assertClose(out[y][x], expected, 1e-2, hint || `cell [${y}][${x}]`);
            }
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            const out = await ctx.kernel(-1, 2, 0.25);
            for (let y = 0; y < 64; y += 3) {
              for (let x = 0; x < 64; x += 3) {
                const cr = -1 + x * 0.25;
                const ci = 2 + y * 0.25;
                const hint = diagnose(out[y][x], cr * cr + ci * ci, 1e-2,
                  planeProbes(-1, 2, 0.25, x, y));
                ctx.assertClose(out[y][x], cr * cr + ci * ci, 1e-2, hint || `cell [${y}][${x}]`);
              }
            }
          },
        },
      ],
    },

    // ------------------------------------------------------------------ 2
    {
      slug: 'escape-time',
      title: 'The Escape-Time Loop',
      intro: `<p>The Mandelbrot set asks one question at every point <code>c</code>: start
        <code>z = 0</code> and repeat <code>z → z² + c</code> — does <code>z</code> stay near the
        origin forever, or fly off to infinity? Points that stay bounded are <em>in</em> the set;
        for the rest, the interesting number is <strong>how many iterations</strong> they survived.</p>
        <p>Two facts make this computable. Once <code>|z| &gt; 2</code>, escape is guaranteed — so
        we can stop watching. And we cap the loop at 100 passes: anything still bounded by then we
        declare "inside". With <code>z = zr + zi·i</code>, one step is
        <code>zr² − zi² + cr</code> for the new real part and <code>2·zr·zi + ci</code> for the new
        imaginary part.</p>`,
      goal: `<strong>Goal:</strong> iterate <code>z → z² + c</code> up to 100 times, but only
        while <code>zr² + zi² &lt; 4</code>, and return how many iterations actually ran.`,
      requirements: [
        'Start at <code>zr = 0, zi = 0, count = 0</code> (already wired up)',
        'Loop a fixed 100 times, guarding each pass with <code>zr² + zi² &lt; 4</code>',
        'Inside the guard: update z via a temporary — <code>zr</code> is read by both formulas',
        'Return <code>count</code>: 100 means "never escaped", small means "escaped fast"',
      ],
      hints: [
        {
          title: 'Hint 1 — the shape of the loop',
          body: `<p>gpu.js's WebGL backend needs a fixed loop bound, so instead of breaking out
            we guard the body:</p>
<pre><code>for (let i = 0; i &lt; 100; i++) {
  if (zr * zr + zi * zi &lt; 4) {
    // …step and count…
  }
}</code></pre>
            <p>After escape the guard fails on every remaining pass, so z freezes and count stops.</p>`,
        },
        {
          title: 'Hint 2 — don\'t clobber zr',
          body: `<p>Both formulas read the <em>old</em> <code>zr</code>, so stash the new real part
            first:</p>
<pre><code>const zrNext = zr * zr - zi * zi + cr;
zi = 2 * zr * zi + ci;
zr = zrNext;
count = count + 1;</code></pre>`,
        },
      ],
      transfer: `Data-dependent loops like this are where <em>divergence</em> lives: in CUDA and
        ROCm, threads of a warp that escape early still march in lockstep with their slowest
        neighbor, so a tile renders at the speed of its deepest pixel. WGSL and Metal shading
        language allow exactly this kind of bounded loop in fragment and compute stages.`,
      starterCode: `// z → z² + c, over and over. Count how long z stays near the origin.
const gpu = new GPU({ mode });

const mandelbrot = gpu.createKernel(function (xMin, yMin, step) {
  const x = this.thread.x;
  const y = this.thread.y;
  const cr = xMin + x * step;
  const ci = yMin + y * step;
  let zr = 0;
  let zi = 0;
  let count = 0;
  // TODO: loop 100 times; on each pass, ONLY while zr² + zi² < 4:
  //   new real part:      zr² - zi² + cr   (stash it in a temporary!)
  //   new imaginary part: 2 * zr * zi + ci
  //   and add 1 to count.
  return count;
}, { output: [64, 64] });

const counts = await mandelbrot(-2.2, -1.6, 3.2 / 64);
console.log('c = 0, deep inside the set:', counts[32][44]);
console.log('far corner, escapes at once:', counts[0][0]);
`,
      solutionCode: `// z → z² + c, over and over. Count how long z stays near the origin.
const gpu = new GPU({ mode });

const mandelbrot = gpu.createKernel(function (xMin, yMin, step) {
  const x = this.thread.x;
  const y = this.thread.y;
  const cr = xMin + x * step;
  const ci = yMin + y * step;
  let zr = 0;
  let zi = 0;
  let count = 0;
  for (let i = 0; i < 100; i++) {
    // guarded instead of break: after escape, z and count just freeze
    if (zr * zr + zi * zi < 4) {
      const zrNext = zr * zr - zi * zi + cr;
      zi = 2 * zr * zi + ci;
      zr = zrNext;
      count = count + 1;
    }
  }
  return count;
}, { output: [64, 64] });

const counts = await mandelbrot(-2.2, -1.6, 3.2 / 64);
console.log('c = 0, deep inside the set:', counts[32][44]);
console.log('far corner, escapes at once:', counts[0][0]);
`,
      publicTests: [
        {
          name: 'interior points never escape — <code>c = 0</code> and <code>c = -1</code> hit the 100 cap',
          run: async ctx => {
            ctx.assert(ctx.kernels.length >= 1, 'no kernel was created — call gpu.createKernel()');
            const out = await ctx.kernel(-2.2, -1.6, 0.05);
            ctx.assert(out && out.length === 64 && out[0].length === 64, 'expected a 64×64 grid');
            const countHint = (y, x, expected) => diagnose(out[y][x], expected, 1e-3, escapeProbes());
            ctx.assertClose(out[32][44], 100, 1e-3, countHint(32, 44, 100) ||
              'cell [32][44] is c = 0 — it never escapes');
            ctx.assertClose(out[32][24], 100, 1e-3, countHint(32, 24, 100) ||
              'cell [32][24] is c = -1 — a stable 2-cycle');
            ctx.assertClose(out[0][0], 1, 1e-3, countHint(0, 0, 1) ||
              'cell [0][0] is c = -2.2 - 1.6i, |c| > 2 — gone in one step');
          },
        },
        {
          name: 'everything with <code>|c| &gt; 2</code> escapes on the very first pass',
          run: async ctx => {
            const out = await ctx.kernel(2.5, 0.5, 0.01);
            for (let y = 0; y < 64; y++) {
              for (let x = 0; x < 64; x++) {
                const hint = diagnose(out[y][x], 1, 1e-3, escapeProbes());
                ctx.assertClose(out[y][x], 1, 1e-3,
                  hint || `cell [${y}][${x}] has |c| > 2 — count must be 1`);
              }
            }
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            // A window strictly inside the main cardioid: everything bounded.
            const inside = await ctx.kernel(-0.2, -0.1, 0.003);
            for (let y = 0; y < 64; y += 5) {
              for (let x = 0; x < 64; x += 5) {
                const hint = diagnose(inside[y][x], 100, 1e-3, escapeProbes());
                ctx.assertClose(inside[y][x], 100, 1e-3, hint || `cardioid cell [${y}][${x}]`);
              }
            }
            // A window far above the set: |c| > 2 everywhere.
            const above = await ctx.kernel(-0.32, 2.5, 0.01);
            for (let y = 0; y < 64; y += 5) {
              for (let x = 0; x < 64; x += 5) {
                const hint = diagnose(above[y][x], 1, 1e-3, escapeProbes());
                ctx.assertClose(above[y][x], 1, 1e-3, hint || `exterior cell [${y}][${x}]`);
              }
            }
          },
        },
      ],
    },

    // ------------------------------------------------------------------ 3
    {
      slug: 'paint-by-count',
      title: 'Paint by Iteration Count',
      intro: `<p>Those counts <em>are</em> the picture. Make the kernel graphical and let every
        thread color its own pixel: points that hit the 100 cap are inside the set — paint them
        <strong>black</strong> — and everything else gets a shade from its count. That's the whole
        recipe behind every Mandelbrot poster ever printed.</p>
        <p>This module's palette maps <code>t = count / 100</code> to
        <code>this.color(t, t·t, 0.5 + 0.5·t, 1)</code> — fast escapes glow deep blue, slow ones
        burn toward white near the boundary, where all the detail hides.</p>`,
      goal: `<strong>Goal:</strong> same escape-time loop, but <code>graphical: true</code> —
        interior pixels black, escaped pixels shaded <code>this.color(t, t*t, 0.5 + 0.5*t, 1)</code>
        with <code>t = count / 100</code>.`,
      requirements: [
        'Keep the guarded 100-pass loop from the last task (already in place)',
        'If <code>count</code> reached 100, paint black: <code>this.color(0, 0, 0, 1)</code>',
        'Otherwise compute <code>t = count / 100</code> and paint <code>this.color(t, t * t, 0.5 + 0.5 * t, 1)</code>',
      ],
      hints: [
        {
          title: 'Hint 1 — two kinds of pixel',
          body: `<p>Branch on the cap: <code>if (count &lt; 100) { …shade… } else { …black… }</code>.
            Both branches must call <code>this.color()</code> — a graphical thread always paints
            exactly one pixel.</p>`,
        },
        {
          title: 'Hint 2 — the shade branch',
          body: `<pre><code>const t = count / 100;
this.color(t, t * t, 0.5 + 0.5 * t, 1);</code></pre>`,
        },
      ],
      transfer: `Mapping a scalar to a color is a <em>transfer function</em> — in scientific
        visualization and medical imaging it's usually a 1D texture the fragment shader samples
        by value; here the colormap is three inline formulas. Same trick, WebGPU to Metal.`,
      starterCode: `// The counts become the picture: one thread paints one pixel.
const gpu = new GPU({ mode });

const paint = gpu.createKernel(function (xMin, yMin, step) {
  const x = this.thread.x;
  const y = this.thread.y;
  const cr = xMin + x * step;
  const ci = yMin + y * step;
  let zr = 0;
  let zi = 0;
  let count = 0;
  for (let i = 0; i < 100; i++) {
    if (zr * zr + zi * zi < 4) {
      const zrNext = zr * zr - zi * zi + cr;
      zi = 2 * zr * zi + ci;
      zr = zrNext;
      count = count + 1;
    }
  }
  // TODO: paint this pixel.
  //   count reached 100  → inside the set → black
  //   escaped            → t = count / 100 → this.color(t, t*t, 0.5 + 0.5*t, 1)
  this.color(1, 0, 1, 1);
}, { output: [128, 128], graphical: true });

await paint(-2.2, -1.6, 3.2 / 128);
render(paint.canvas);
`,
      solutionCode: `// The counts become the picture: one thread paints one pixel.
const gpu = new GPU({ mode });

const paint = gpu.createKernel(function (xMin, yMin, step) {
  const x = this.thread.x;
  const y = this.thread.y;
  const cr = xMin + x * step;
  const ci = yMin + y * step;
  let zr = 0;
  let zi = 0;
  let count = 0;
  for (let i = 0; i < 100; i++) {
    if (zr * zr + zi * zi < 4) {
      const zrNext = zr * zr - zi * zi + cr;
      zi = 2 * zr * zi + ci;
      zr = zrNext;
      count = count + 1;
    }
  }
  if (count < 100) {
    const t = count / 100;
    this.color(t, t * t, 0.5 + 0.5 * t, 1);
  } else {
    this.color(0, 0, 0, 1);
  }
}, { output: [128, 128], graphical: true });

await paint(-2.2, -1.6, 3.2 / 128);
render(paint.canvas);
`,
      publicTests: [
        {
          name: 'the classic view shows both worlds — black interior AND shaded exterior',
          run: async ctx => {
            ctx.assert(ctx.kernels.length >= 1, 'no kernel was created — call gpu.createKernel()');
            ctx.assert(ctx.canvas, 'no canvas — is the kernel graphical: true, and did you call render()?');
            ctx.assert(
              ctx.canvas.width === 128 && ctx.canvas.height === 128,
              `expected a 128×128 canvas, got ${ctx.canvas.width}×${ctx.canvas.height}`
            );
            const pixels = ctx.getPixels();
            let black = 0;
            let shaded = 0;
            for (let i = 0; i < pixels.length; i += 4) {
              if (pixels[i] + pixels[i + 1] + pixels[i + 2] <= 3) black++;
              else if (pixels[i + 2] > 100) shaded++;
            }
            ctx.assert(black > 300, `expected a black interior — found only ${black} black pixels`);
            ctx.assert(shaded > 300, `expected a shaded exterior — found only ${shaded} blue-ish pixels`);
          },
        },
        {
          name: 'a window inside the set is pure black',
          run: async ctx => {
            await ctx.kernel(-0.2, -0.05, 0.001);
            const pixels = ctx.getPixels();
            for (let i = 0; i < pixels.length; i += 401 * 4) {
              ctx.assert(
                pixels[i] <= 2 && pixels[i + 1] <= 2 && pixels[i + 2] <= 2,
                interiorHint(pixels[i], pixels[i + 1], pixels[i + 2]) ||
                  `interior pixel at byte ${i} should be black, got rgb(${pixels[i]}, ${pixels[i + 1]}, ${pixels[i + 2]})`
              );
            }
          },
        },
        {
          name: 'far outside, every pixel wears the count-1 shade',
          run: async ctx => {
            await ctx.kernel(2.5, 2.5, 0.001);
            const pixels = ctx.getPixels();
            const [r, g, b] = paletteBytes(1 / MAX_ITER);
            for (let i = 0; i < pixels.length; i += 401 * 4) {
              ctx.assertClose(pixels[i], r, 2, `red at byte ${i}`);
              ctx.assertClose(pixels[i + 1], g, 2, `green at byte ${i}`);
              ctx.assertClose(pixels[i + 2], b, 2, `blue at byte ${i}`);
            }
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            // Inside the period-2 bulb around c = -1: all black.
            await ctx.kernel(-1.05, -0.05, 0.0008);
            let pixels = ctx.getPixels();
            for (let i = 0; i < pixels.length; i += 293 * 4) {
              ctx.assert(
                pixels[i] <= 2 && pixels[i + 1] <= 2 && pixels[i + 2] <= 2,
                interiorHint(pixels[i], pixels[i + 1], pixels[i + 2]) ||
                  `bulb pixel at byte ${i} should be black, got rgb(${pixels[i]}, ${pixels[i + 1]}, ${pixels[i + 2]})`
              );
            }
            // Far left of the set: uniform count-1 shade.
            await ctx.kernel(-9, 0, 0.001);
            pixels = ctx.getPixels();
            const [r, g, b] = paletteBytes(1 / MAX_ITER);
            for (let i = 0; i < pixels.length; i += 293 * 4) {
              ctx.assertClose(pixels[i], r, 2, `red at byte ${i}`);
              ctx.assertClose(pixels[i + 1], g, 2, `green at byte ${i}`);
              ctx.assertClose(pixels[i + 2], b, 2, `blue at byte ${i}`);
            }
          },
        },
      ],
    },

    // ------------------------------------------------------------------ 4
    {
      slug: 'smooth-coloring',
      title: 'Smooth Out the Bands',
      intro: `<p>Look closely at task 3's exterior and you'll see hard rings: iteration counts are
        integers, so neighboring pixels jump from shade 6 straight to shade 7. But the kernel knows
        more than the count — it knows <strong>how far past the escape radius</strong> z flew on its
        final step. A barely-escaped z and one that rocketed to |z| = 50 both count the same pass;
        that overshoot is the missing fraction.</p>
        <p>The classic fix is the <em>normalized iteration count</em>:
        <code>count + 1 − log2(log2|z|)</code>. When z barely clears the radius the correction is
        near 1, when it overshoots hugely it's near 0 — and the bands blend into a continuous ramp.
        (With <code>|z|² = zr² + zi²</code> in hand, use <code>log2|z| = 0.5 · log2(zr² + zi²)</code>
        and skip the square root.)</p>`,
      goal: `<strong>Goal:</strong> return a <em>fractional</em> escape value — interior points
        return exactly 100, escaped points return
        <code>count + 1 − Math.log2(0.5 * Math.log2(zr² + zi²))</code>.`,
      requirements: [
        'Keep the guarded loop; after it, branch on <code>count &lt; 100</code>',
        'Escaped: return the fractional escape value — the normalized iteration count from the intro',
        'Interior: return <code>100</code> exactly — no correction for points that never escaped',
      ],
      hints: [
        {
          title: 'Hint 1 — why z is still usable after the loop',
          body: `<p>The guard freezes z the moment it escapes, so after the loop <code>zr, zi</code>
            hold the <em>first</em> value with <code>|z|² ≥ 4</code> — exactly the overshoot the
            formula needs. Math.log2 works inside kernels on both backends.</p>`,
        },
        {
          title: 'Hint 2 — the ending',
          body: `<pre><code>if (count &lt; 100) {
  return count + 1
    - Math.log2(0.5 * Math.log2(zr * zr + zi * zi));
}
return 100;</code></pre>`,
        },
      ],
      transfer: `Fighting quantization with a fractional correction is a graphics evergreen:
        trilinear blending between mipmap levels, <code>smoothstep</code> edges, ordered dithering.
        The same normalized-iteration formula runs unchanged in a CUDA kernel or a WGSL fragment
        shader — it's pure float math.`,
      starterCode: `// Counts are integers — that's why the shading shows rings.
// Return a FRACTIONAL escape value instead.
const gpu = new GPU({ mode });

const smoothField = gpu.createKernel(function (xMin, yMin, step) {
  const x = this.thread.x;
  const y = this.thread.y;
  const cr = xMin + x * step;
  const ci = yMin + y * step;
  let zr = 0;
  let zi = 0;
  let count = 0;
  for (let i = 0; i < 100; i++) {
    if (zr * zr + zi * zi < 4) {
      const zrNext = zr * zr - zi * zi + cr;
      zi = 2 * zr * zi + ci;
      zr = zrNext;
      count = count + 1;
    }
  }
  // TODO: escaped pixels (count < 100) should return
  //   count + 1 - Math.log2(0.5 * Math.log2(zr² + zi²))
  // interior pixels return 100 exactly.
  return count;
}, { output: [64, 64] });

const field = await smoothField(3, 1, 0.01);
console.log('a fractional escape value:', field[0][0]);
`,
      solutionCode: `// Counts are integers — that's why the shading shows rings.
// Return a FRACTIONAL escape value instead.
const gpu = new GPU({ mode });

const smoothField = gpu.createKernel(function (xMin, yMin, step) {
  const x = this.thread.x;
  const y = this.thread.y;
  const cr = xMin + x * step;
  const ci = yMin + y * step;
  let zr = 0;
  let zi = 0;
  let count = 0;
  for (let i = 0; i < 100; i++) {
    if (zr * zr + zi * zi < 4) {
      const zrNext = zr * zr - zi * zi + cr;
      zi = 2 * zr * zi + ci;
      zr = zrNext;
      count = count + 1;
    }
  }
  if (count < 100) {
    // z froze at its first escaped value — its overshoot is the fraction
    return count + 1 - Math.log2(0.5 * Math.log2(zr * zr + zi * zi));
  }
  return 100;
}, { output: [64, 64] });

const field = await smoothField(3, 1, 0.01);
console.log('a fractional escape value:', field[0][0]);
`,
      publicTests: [
        {
          name: 'interior cells still return exactly 100',
          run: async ctx => {
            ctx.assert(ctx.kernels.length >= 1, 'no kernel was created — call gpu.createKernel()');
            const out = await ctx.kernel(-0.2, -0.05, 0.002);
            ctx.assert(out && out.length === 64 && out[0].length === 64, 'expected a 64×64 grid');
            for (let y = 0; y < 64; y += 7) {
              for (let x = 0; x < 64; x += 7) {
                ctx.assertClose(out[y][x], 100, 1e-3, `interior cell [${y}][${x}]`);
              }
            }
          },
        },
        {
          name: 'escaped cells carry a fraction that matches the formula',
          run: async ctx => {
            const out = await ctx.kernel(3, 1, 0.01);
            let sawFraction = false;
            const cases = [[0, 0], [10, 20], [33, 7], [63, 63]];
            for (const [y, x] of cases) {
              const expected = smoothEscape(0, 0, 3 + x * 0.01, 1 + y * 0.01);
              const hint = diagnose(out[y][x], expected, 0.02,
                smoothProbes(3 + x * 0.01, 1 + y * 0.01));
              ctx.assertClose(out[y][x], expected, 0.02, hint || `cell [${y}][${x}]`);
              if (Math.abs(out[y][x] - Math.round(out[y][x])) > 0.05) sawFraction = true;
            }
            ctx.assert(
              sawFraction,
              'every sampled value is a whole number — are you still returning the raw count?'
            );
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            // Count-1 territory, far from the set.
            const far = await ctx.kernel(-4, 2, 0.005);
            for (let y = 0; y < 64; y += 9) {
              for (let x = 0; x < 64; x += 9) {
                const expected = smoothEscape(0, 0, -4 + x * 0.005, 2 + y * 0.005);
                const hint = diagnose(far[y][x], expected, 0.02,
                  smoothProbes(-4 + x * 0.005, 2 + y * 0.005));
                ctx.assertClose(far[y][x], expected, 0.02, hint || `far cell [${y}][${x}]`);
              }
            }
            // Count-2 territory on the positive real side.
            const near = await ctx.kernel(1.5, -0.032, 0.001);
            for (let y = 0; y < 64; y += 9) {
              for (let x = 0; x < 64; x += 9) {
                const expected = smoothEscape(0, 0, 1.5 + x * 0.001, -0.032 + y * 0.001);
                const hint = diagnose(near[y][x], expected, 0.02,
                  smoothProbes(1.5 + x * 0.001, -0.032 + y * 0.001));
                ctx.assertClose(near[y][x], expected, 0.02, hint || `near cell [${y}][${x}]`);
              }
            }
          },
        },
      ],
    },

    // ------------------------------------------------------------------ 5
    {
      slug: 'julia-dial',
      title: 'Julia Sets: Turn the Dial',
      intro: `<p>Here's the payoff. Take the exact loop you've built and <strong>flip the
        roles</strong>: in a Julia set, z starts <em>at the pixel</em> and <code>c</code> is one
        fixed complex number shared by every thread. Each choice of c is a different fractal —
        <code>c = 0</code> gives a plain disk, <code>−0.7269 + 0.1889i</code> a galaxy of spirals —
        and the Mandelbrot set turns out to be the map of which c values give connected Julias.</p>
        <p>Because c arrives as <strong>kernel arguments</strong>, changing it costs one function
        call — no recompiling. That's what makes those mesmerizing morphing-Julia animations:
        nudge c, redraw, repeat.</p>`,
      goal: `<strong>Goal:</strong> a graphical Julia kernel over the fixed view −1.6…1.6: seed
        <code>z</code> from the pixel, add the arguments <code>cRe, cIm</code> each step, and keep
        task 4's smooth shading (interior black).`,
      requirements: [
        'Seed z from the pixel: <code>zr = xMin + x·step</code>, <code>zi = yMin + y·step</code> (constants are wired up)',
        'Inside the loop, add <code>cRe</code> and <code>cIm</code> — not the pixel coordinates',
        'Escaped: shade with <code>t = smooth / 100</code> via <code>this.color(t, t * t, 0.5 + 0.5 * t, 1)</code>; interior: black',
        'Call the kernel with a c of your choice and <code>render()</code> it',
      ],
      hints: [
        {
          title: 'Hint 1 — what actually changes',
          body: `<p>Two lines. Mandelbrot: z starts at 0 and c is the pixel. Julia: z starts at the
            pixel and c is the argument pair. The loop body, the guard, the shading — all identical.</p>`,
        },
        {
          title: 'Hint 2 — the exact edits',
          body: `<p>Seed with</p>
<pre><code>let zr = this.constants.xMin + x * this.constants.step;</code></pre>
<p>(and likewise <code>zi</code> from y), then inside the loop use
            <code>… + cRe</code> and <code>… + cIm</code> instead of <code>px</code> / <code>py</code>.</p>`,
        },
      ],
      transfer: `A per-launch value broadcast to every thread is what other APIs call a
        <em>uniform</em>: a WGSL uniform buffer, a Metal constant buffer, a plain CUDA kernel
        parameter. Animating one uniform per frame — exactly your c — is how every shader-toy
        Julia morph is driven.`,
      starterCode: `// Same loop, roles flipped: the pixel is z₀, and c is a knob you turn.
const gpu = new GPU({ mode });

const julia = gpu.createKernel(function (cRe, cIm) {
  const x = this.thread.x;
  const y = this.thread.y;
  const px = this.constants.xMin + x * this.constants.step;
  const py = this.constants.yMin + y * this.constants.step;
  // TODO: this is still the Mandelbrot arrangement — z from 0, pixel as c.
  // Flip it: seed z from (px, py), and add cRe / cIm inside the loop.
  let zr = 0;
  let zi = 0;
  let count = 0;
  for (let i = 0; i < 100; i++) {
    if (zr * zr + zi * zi < 4) {
      const zrNext = zr * zr - zi * zi + px;
      zi = 2 * zr * zi + py;
      zr = zrNext;
      count = count + 1;
    }
  }
  if (count < 100) {
    const smooth = count + 1 - Math.log2(0.5 * Math.log2(zr * zr + zi * zi));
    const t = smooth / 100;
    this.color(t, t * t, 0.5 + 0.5 * t, 1);
  } else {
    this.color(0, 0, 0, 1);
  }
}, {
  output: [128, 128],
  graphical: true,
  constants: { xMin: -1.6, yMin: -1.6, step: 0.025 },
});

// A real dial. slider() returns the value this run is using and puts a control
// under the console; moving it re-runs the whole program, so the set redraws as
// you drag. Defaults are the classic dendrite c = -0.7269 + 0.1889i.
const cRe = slider('c real', { min: -1, max: 0.4, value: -0.7269, step: 0.001 });
const cIm = slider('c imag', { min: -0.8, max: 0.8, value: 0.1889, step: 0.001 });

await julia(cRe, cIm);
render(julia.canvas);
`,
      solutionCode: `// Same loop, roles flipped: the pixel is z₀, and c is a knob you turn.
const gpu = new GPU({ mode });

const julia = gpu.createKernel(function (cRe, cIm) {
  // z starts AT the pixel; c is shared by every thread
  const x = this.thread.x;
  const y = this.thread.y;
  let zr = this.constants.xMin + x * this.constants.step;
  let zi = this.constants.yMin + y * this.constants.step;
  let count = 0;
  for (let i = 0; i < 100; i++) {
    if (zr * zr + zi * zi < 4) {
      const zrNext = zr * zr - zi * zi + cRe;
      zi = 2 * zr * zi + cIm;
      zr = zrNext;
      count = count + 1;
    }
  }
  if (count < 100) {
    const smooth = count + 1 - Math.log2(0.5 * Math.log2(zr * zr + zi * zi));
    const t = smooth / 100;
    this.color(t, t * t, 0.5 + 0.5 * t, 1);
  } else {
    this.color(0, 0, 0, 1);
  }
}, {
  output: [128, 128],
  graphical: true,
  constants: { xMin: -1.6, yMin: -1.6, step: 0.025 },
});

// A real dial. slider() returns the value this run is using and puts a control
// under the console; moving it re-runs the whole program, so the set redraws as
// you drag. Defaults are the classic dendrite c = -0.7269 + 0.1889i.
const cRe = slider('c real', { min: -1, max: 0.4, value: -0.7269, step: 0.001 });
const cIm = slider('c imag', { min: -0.8, max: 0.8, value: 0.1889, step: 0.001 });

await julia(cRe, cIm);
render(julia.canvas);
`,
      publicTests: [
        {
          name: 'with <code>c = 0</code> the Julia set is the unit disk — inside black, outside shaded',
          run: async ctx => {
            ctx.assert(ctx.kernels.length >= 1, 'no kernel was created — call gpu.createKernel()');
            ctx.assert(ctx.canvas, 'no canvas — is the kernel graphical: true, and did you call render()?');
            ctx.assert(
              ctx.canvas.width === 128 && ctx.canvas.height === 128,
              `expected a 128×128 canvas, got ${ctx.canvas.width}×${ctx.canvas.height}`
            );
            await ctx.kernel(0, 0);
            const pixels = ctx.getPixels();
            for (let y = 4; y < 128; y += 8) {
              for (let x = 4; x < 128; x += 8) {
                const zr0 = -1.6 + x * 0.025;
                const zi0 = -1.6 + y * 0.025;
                const r2 = zr0 * zr0 + zi0 * zi0;
                const i = (y * 128 + x) * 4;
                if (r2 < 0.9) {
                  ctx.assert(
                    pixels[i] + pixels[i + 1] + pixels[i + 2] <= 3,
                    `pixel (${x}, ${y}) is inside the unit disk — expected black, got rgb(${pixels[i]}, ${pixels[i + 1]}, ${pixels[i + 2]})`
                  );
                } else if (r2 > 4.25) {
                  ctx.assert(
                    pixels[i + 2] > 100,
                    `pixel (${x}, ${y}) is far outside the disk — expected a blue-ish shade, got rgb(${pixels[i]}, ${pixels[i + 1]}, ${pixels[i + 2]})`
                  );
                }
              }
            }
          },
        },
        {
          name: 'c is a live argument — turn the dial and the center pixel flips',
          run: async ctx => {
            await ctx.kernel(0, 0);
            let pixels = ctx.getPixels();
            const center = (64 * 128 + 64) * 4;
            ctx.assert(
              pixels[center] + pixels[center + 1] + pixels[center + 2] <= 3,
              'with c = 0 the center pixel (z₀ = 0) never escapes — it should be black'
            );
            await ctx.kernel(-2.5, 0);
            pixels = ctx.getPixels();
            ctx.assert(
              pixels[center + 2] > 100,
              'with c = -2.5 the center pixel escapes in one step — it should be shaded, not black. Is c actually used in the loop?'
            );
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            await ctx.kernel(0, 0);
            const pixels = ctx.getPixels();
            // z₀ = (0.5, 0.5): inside the unit disk → black.
            let i = (84 * 128 + 84) * 4;
            ctx.assert(
              pixels[i] + pixels[i + 1] + pixels[i + 2] <= 3,
              `pixel (84, 84) lies inside the unit disk — expected black, got rgb(${pixels[i]}, ${pixels[i + 1]}, ${pixels[i + 2]})`
            );
            // z₀ = (-1.5, 0): escapes; color must match the smooth palette.
            i = (64 * 128 + 4) * 4;
            const t = smoothEscape(-1.5, 0, 0, 0) / MAX_ITER;
            const [r, g, b] = paletteBytes(t);
            ctx.assertClose(pixels[i], r, 4, 'red at pixel (4, 64)');
            ctx.assertClose(pixels[i + 1], g, 4, 'green at pixel (4, 64)');
            ctx.assertClose(pixels[i + 2], b, 4, 'blue at pixel (4, 64)');
          },
        },
      ],
    },
  ],
};
