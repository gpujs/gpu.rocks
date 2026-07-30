// Module 3.1 — Pixels from Scratch (Track 3: Computational Graphics).
//
// Four tasks: first graphical kernel (a coordinate gradient) → checkerboard
// via modular arithmetic → plotting y = f(x) as a per-pixel distance test →
// radial ripples in polar coordinates.
//
// Kernel-authoring rules (contract): no closures inside kernel functions,
// graphical kernels paint with this.color(r, g, b, a) with channels 0–1,
// this.thread.y counts rows bottom-up (GL convention). All canvases are
// 128×128 so CPU verification stays fast. Pixel tests are robust to
// getPixels() row order: they test x-only properties, y-symmetric patterns,
// or detect the orientation from the data before checking exact values.

// Byte offset of pixel (x, row) in a size×size getPixels() buffer.
function at(size, x, row) {
  return (row * size + x) * 4;
}

// Task 3's curve: one full sine period across a 128-px canvas (thread coords).
function curveHeight(x) {
  return 64 + 40 * Math.sin((x * 2 * Math.PI) / 128);
}

// Buffer rows (0-based) of lit pixels (red > 128) in one column.
function litRowsInColumn(pixels, size, x) {
  const rows = [];
  for (let row = 0; row < size; row++) {
    if (pixels[at(size, x, row)] > 128) rows.push(row);
  }
  return rows;
}

function mean(values) {
  let total = 0;
  for (let i = 0; i < values.length; i++) total += values[i];
  return total / values.length;
}

// Task 4's ripple color at (x, y) — orientation-free because it depends only
// on the distance to the canvas center. Returns [r, g, b] in 0–255.
function rippleRGB(x, y) {
  const dx = x - 63.5;
  const dy = y - 63.5;
  const r = Math.sqrt(dx * dx + dy * dy);
  const wave = 0.5 + 0.5 * Math.cos(r * 0.35);
  const fade = Math.max(0, 1 - r / 96);
  const v = wave * fade;
  return [0.4 * v * 255, 0.75 * v * 255, v * 255];
}

// ---- near-miss diagnosis --------------------------------------------------
//
// A failing assert that reports only two numbers tells a learner nothing about
// WHICH slip produced them. A probe pairs the value one specific known mistake
// would produce with a sentence naming that mistake; diagnose() speaks only
// when the observed value matches a probe within the test's own tolerance AND
// the correct value does not — so pixels where two candidates coincide stay
// silent, as do observations that match probes disagreeing with each other.
// A wrong diagnosis is worse than none. getPixels() row order is unknown, so
// probes that depend on the row list both orientations under one message.
function diagnose(got, expected, eps, probes) {
  const hits = probes
    .filter(p => Math.abs(got - p[0]) <= eps && Math.abs(expected - p[0]) > eps)
    .map(p => p[1]);
  return hits.length && hits.every(m => m === hits[0]) ? hits[0] : null;
}

// Task 1: red following the wrong coordinate, or never divided by the canvas
// size at all (a 0–1 channel saturates the moment this.thread.x reaches 1).
function rampProbes(x, row) {
  const swapped = 'red is following the row instead of the column — the horizontal ramp is this.thread.x / 128';
  return [
    [(255 * row) / 128, swapped],
    [(255 * (127 - row)) / 128, swapped],
    [Math.min(255, x * 255),
      'color channels run 0–1, so an undivided this.thread.x saturates every column past the first — divide it by 128'],
  ];
}

// Task 1: green frozen down the canvas AND sitting exactly where this column's
// x ramp would put it means the two coordinates are swapped.
function flatGreenHint(g0, g127, column) {
  return Math.abs(g0 - g127) <= 2 && Math.abs(g0 - (255 * column) / 128) <= 3
    ? 'green is constant down the canvas and matches this column\'s x ramp — this.thread.x and this.thread.y are swapped; green rises with this.thread.y'
    : null;
}

// Task 2: a learner who divides by the wrong number still paints a perfectly
// regular board — just with the wrong cell size, which the length of the first
// run of equal values along a row gives away. A whole uniform row says nothing.
function cellWidthHint(pixels, row) {
  const first = pixels[at(128, 0, row)];
  let width = 1;
  while (width < 128 && pixels[at(128, width, row)] === first) width++;
  return width === 16 || width === 128
    ? null
    : `your cells are ${width} pixels wide, not 16 — the cell index is Math.floor(coordinate / 16)`;
}

// Task 3: mis-scaling x before Math.sin. `orient` maps a thread-space height
// onto the buffer row the caller is reading, so the probes hold under either
// row order.
function curveProbes(x, orient) {
  return [
    [orient(64 + 40 * Math.sin(x)),
      'that is Math.sin() of the raw pixel count — one period across 128 px needs x * 2 * Math.PI / 128'],
    [orient(64 + 40 * Math.sin((x * 2 * Math.PI) / 64)),
      'two periods fit the canvas — divide x by 128, the full width, for one'],
    [orient(64 + 40 * Math.sin((x * 360) / 128)),
      'Math.sin takes radians, not degrees — the scale is 2 * Math.PI / 128'],
    [orient(64),
      'the curve is still the constant 64 — it never became a function of x'],
  ];
}

// Task 4: v assembled from only half the recipe. Each candidate v is weighted
// by the channel under test (0.4 red, 0.75 green, 1 blue) and clamped the way
// this.color() clamps, so the probe matches the byte the test read.
function rippleProbes(x, y, channel) {
  const weight = [0.4, 0.75, 1][channel];
  const dx = x - 63.5;
  const dy = y - 63.5;
  const r = Math.sqrt(dx * dx + dy * dy);
  const wave = 0.5 + 0.5 * Math.cos(r * 0.35);
  const fade = Math.max(0, 1 - r / 96);
  return [
    [wave, 'the fade never got multiplied in — v = wave * fade'],
    [fade, 'that is the bare fade — the cosine ripple is missing from v'],
    [Math.max(0, Math.cos(r * 0.35)) * fade,
      'the cosine still swings negative — remap it with 0.5 + 0.5 * Math.cos(r * 0.35)'],
  ].map(pair => [Math.min(1, pair[0]) * weight * 255, pair[1]]);
}

export default {
  id: '3-1',
  track: 3,
  title: 'Pixels from Scratch',
  blurb: 'Graphical kernels and <code>this.color()</code>: gradients, patterns and plots, one thread per pixel.',
  tasks: [
    // ------------------------------------------------------------------ 1
    {
      slug: 'coordinate-gradient',
      title: 'Paint with Coordinates',
      intro: `<p>Set <code>graphical: true</code> and a kernel stops returning numbers — instead
        every thread paints <strong>exactly one pixel</strong> by calling
        <code>this.color(r, g, b, a)</code>, channels 0–1. The output shape becomes the canvas:
        <code>output: [128, 128]</code> is a 128×128 picture, 16,384 threads, one per pixel.</p>
        <p>A solid color is one line — and the starter already paints one. The interesting part is
        that each thread knows <em>where</em> it is: <code>this.thread.x</code> counts columns from
        the left, <code>this.thread.y</code> counts rows from the <strong>bottom</strong> (GL
        convention). Divide either by the canvas size and you get a smooth 0–1 ramp, ready to feed
        straight into a color channel.</p>`,
      goal: `<strong>Goal:</strong> turn the flat gray into a two-axis gradient — red rising with
        <code>x</code>, green rising with <code>y</code>, blue fixed at <code>0.5</code>.`,
      requirements: [
        'Keep <code>graphical: true</code> and <code>output: [128, 128]</code>',
        'Red channel = <code>this.thread.x / 128</code>',
        'Green channel = <code>this.thread.y / 128</code>',
        'Blue stays <code>0.5</code>, alpha stays <code>1</code>',
      ],
      hints: [
        {
          title: 'Hint 1 — where am I?',
          body: `<p><code>this.thread.x</code> runs 0…127 here, so
            <code>this.thread.x / 128</code> runs 0…0.992 — a ready-made red ramp.
            Same move with <code>this.thread.y</code> for green.</p>`,
        },
        {
          title: 'Hint 2 — the one-liner',
          body: `<p>The whole kernel body:</p>
<pre><code>this.color(this.thread.x / 128, this.thread.y / 128, 0.5, 1);</code></pre>`,
        },
      ],
      transfer: `Normalized pixel coordinates are the <em>uv</em> every shader language starts
        from: WebGPU and Metal fragment shaders derive them from the fragment position, and CUDA
        image kernels divide thread indices by the image width the same way. The famous red-green
        "uv debug gradient" is exactly this kernel.`,
      starterCode: `// graphical: true turns a kernel into a painter — one thread per pixel.
const gpu = new GPU({ mode });

const gradient = gpu.createKernel(function () {
  // Right now all 16,384 threads paint the SAME color.
  // TODO: mix this thread's coordinates into the color —
  //   red   = this.thread.x / 128
  //   green = this.thread.y / 128
  //   blue  = 0.5
  this.color(0.2, 0.2, 0.2, 1);
}, {
  output: [128, 128],
  graphical: true,
});

gradient();
render(gradient.canvas);
`,
      solutionCode: `// graphical: true turns a kernel into a painter — one thread per pixel.
const gpu = new GPU({ mode });

const gradient = gpu.createKernel(function () {
  this.color(this.thread.x / 128, this.thread.y / 128, 0.5, 1);
}, {
  output: [128, 128],
  graphical: true,
});

gradient();
render(gradient.canvas);
`,
      publicTests: [
        {
          name: 'paints a graphical <code>128×128</code> canvas',
          run: async ctx => {
            ctx.assert(ctx.kernels.length >= 1, 'no kernel was created — call gpu.createKernel()');
            ctx.assert(ctx.canvas, 'no canvas — is the kernel graphical: true, and did you call render()?');
            ctx.assert(
              ctx.canvas.width === 128 && ctx.canvas.height === 128,
              `expected a 128×128 canvas, got ${ctx.canvas.width}×${ctx.canvas.height}`
            );
            ctx.assert(ctx.getPixels().length === 128 * 128 * 4, 'pixel buffer should hold 128×128 RGBA values');
          },
        },
        {
          name: 'red rises left to right: <code>this.thread.x / 128</code>',
          run: async ctx => {
            const pixels = ctx.getPixels();
            // red depends only on x, so this check is row-order independent —
            // verify it on a row near each edge of the buffer.
            for (const row of [3, 64, 124]) {
              for (let x = 0; x < 128; x += 7) {
                const got = pixels[at(128, x, row)];
                const expected = (255 * x) / 128;
                const hint = diagnose(got, expected, 2.5, rampProbes(x, row));
                ctx.assertClose(
                  got,
                  expected,
                  2.5,
                  hint || `red at column ${x} (buffer row ${row})`
                );
              }
            }
          },
        },
        {
          name: 'green rises with <code>this.thread.y</code>; blue holds at 0.5',
          run: async ctx => {
            const pixels = ctx.getPixels();
            // Row order may be top-down or bottom-up — the green ramp just has
            // to span ~0 → ~252 from one edge of the buffer to the other.
            const g0 = pixels[at(128, 20, 0) + 1];
            const g127 = pixels[at(128, 20, 127) + 1];
            ctx.assert(
              Math.min(g0, g127) <= 4 && Math.max(g0, g127) >= 248,
              flatGreenHint(g0, g127, 20) ||
                `green should ramp 0 → 252 across the canvas, got edge values ${g0} and ${g127}`
            );
            for (const [x, row] of [[10, 10], [90, 40], [64, 100]]) {
              ctx.assertClose(pixels[at(128, x, row) + 2], 127.5, 2.5, `blue at (${x}, row ${row})`);
            }
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            const pixels = ctx.getPixels();
            // Detect buffer orientation from the green ramp, then check every
            // sampled pixel exactly in thread coordinates.
            const rowIsThreadY = pixels[at(128, 5, 0) + 1] < pixels[at(128, 5, 127) + 1];
            for (let row = 0; row < 128; row += 5) {
              const y = rowIsThreadY ? row : 127 - row;
              for (let x = 0; x < 128; x += 5) {
                const i = at(128, x, row);
                const redHint = diagnose(pixels[i], (255 * x) / 128, 2.5, rampProbes(x, row));
                ctx.assertClose(pixels[i], (255 * x) / 128, 2.5, redHint || `red at (${x}, y=${y})`);
                ctx.assertClose(pixels[i + 1], (255 * y) / 128, 2.5, `green at (${x}, y=${y})`);
                ctx.assertClose(pixels[i + 2], 127.5, 2.5, `blue at (${x}, y=${y})`);
                ctx.assert(pixels[i + 3] === 255, `alpha at (${x}, y=${y}) should be 255`);
              }
            }
          },
        },
      ],
    },

    // ------------------------------------------------------------------ 2
    {
      slug: 'checkerboard',
      title: 'Checkerboard Logic',
      intro: `<p>Smooth ramps become hard-edged patterns with two tools:
        <code>Math.floor</code> to chop coordinates into cells, and the remainder operator
        <code>%</code> to make the cells repeat. <code>Math.floor(this.thread.x / 16)</code> asks
        <em>"which 16-pixel band am I in?"</em> — and <code>% 2</code> answers
        <em>"odd or even?"</em>.</p>
        <p>The starter already draws vertical stripes with exactly that trick. A checkerboard is
        the same idea in both axes at once: compute a cell index for x <em>and</em> y, add them,
        and take the parity of the sum — cells that touch on an edge always disagree.</p>`,
      goal: `<strong>Goal:</strong> upgrade the stripes to an 8×8 checkerboard of 16-pixel cells —
        paint <code>(cellX + cellY) % 2</code> into all three color channels.`,
      requirements: [
        'Keep the cells 16 pixels: <code>Math.floor(coordinate / 16)</code>',
        'Combine both axes: parity of <code>cellX + cellY</code>',
        'Pure black and white only — the parity (0 or 1) is the color',
      ],
      hints: [
        {
          title: 'Hint 1 — the second axis',
          body: `<p>Mirror the existing line for y:</p>
<pre><code>const cellY = Math.floor(this.thread.y / 16);</code></pre>`,
        },
        {
          title: 'Hint 2 — why the sum?',
          body: `<p>Moving one cell right changes <code>cellX</code> by 1; moving one cell up
            changes <code>cellY</code> by 1. Either move flips the parity of
            <code>cellX + cellY</code> — which is exactly what a checkerboard does. So:
            <code>const v = (cellX + cellY) % 2;</code></p>`,
        },
      ],
      transfer: `Procedural patterns are a GPU staple: GLSL and WGSL shaders build checkers,
        stripes and grids from <code>floor()</code> and <code>mod()</code> with no texture in
        sight, and CUDA kernels lean on the same modular arithmetic on thread ids to stripe work
        across blocks.`,
      starterCode: `// Modular arithmetic turns smooth coordinates into repeating patterns.
const gpu = new GPU({ mode });

const board = gpu.createKernel(function () {
  // Stripes: which 16-pixel column band is this thread in — odd or even?
  const cellX = Math.floor(this.thread.x / 16);
  const v = cellX % 2;
  // TODO: bring this.thread.y into it. A checkerboard flips parity every
  // 16 pixels vertically too — (cellX + cellY) is the trick.
  this.color(v, v, v, 1);
}, {
  output: [128, 128],
  graphical: true,
});

board();
render(board.canvas);
`,
      solutionCode: `// Modular arithmetic turns smooth coordinates into repeating patterns.
const gpu = new GPU({ mode });

const board = gpu.createKernel(function () {
  const cellX = Math.floor(this.thread.x / 16);
  const cellY = Math.floor(this.thread.y / 16);
  const v = (cellX + cellY) % 2;
  this.color(v, v, v, 1);
}, {
  output: [128, 128],
  graphical: true,
});

board();
render(board.canvas);
`,
      publicTests: [
        {
          name: 'every pixel is pure black or pure white',
          run: async ctx => {
            ctx.assert(ctx.kernels.length >= 1, 'no kernel was created — call gpu.createKernel()');
            const pixels = ctx.getPixels();
            ctx.assert(pixels.length === 128 * 128 * 4, 'expected a 128×128 canvas');
            for (let i = 0; i < pixels.length; i += 4) {
              const r = pixels[i];
              ctx.assert(
                r <= 1 || r >= 254,
                `pixel at byte ${i} is gray (${r}) — the parity should be exactly 0 or 1`
              );
              ctx.assert(
                Math.abs(pixels[i + 1] - r) <= 1 && Math.abs(pixels[i + 2] - r) <= 1,
                `pixel at byte ${i} is tinted — use the same value for r, g and b`
              );
            }
          },
        },
        {
          name: 'cells are 16 pixels wide and alternate along x',
          run: async ctx => {
            const pixels = ctx.getPixels();
            for (const row of [8, 40, 100]) {
              const inCellA = pixels[at(128, 2, row)];
              const inCellB = pixels[at(128, 13, row)];
              ctx.assert(
                Math.abs(inCellA - inCellB) <= 1,
                cellWidthHint(pixels, row) ||
                  `columns 2 and 13 share a 16-px cell but differ on buffer row ${row}`
              );
              const left = pixels[at(128, 8, row)];
              const right = pixels[at(128, 24, row)];
              ctx.assert(
                Math.abs(left - right) >= 250,
                cellWidthHint(pixels, row) ||
                  `columns 8 and 24 are in adjacent cells but match on buffer row ${row} — still stripes?`
              );
            }
          },
        },
        {
          name: 'cells alternate along y too — that\'s what makes it a checkerboard',
          run: async ctx => {
            const pixels = ctx.getPixels();
            // 16-px row bands map onto 16-px row bands whichever way the
            // buffer is oriented, so these checks are row-order independent.
            for (const x of [8, 40, 100]) {
              const inCellA = pixels[at(128, x, 2)];
              const inCellB = pixels[at(128, x, 13)];
              ctx.assert(
                Math.abs(inCellA - inCellB) <= 1,
                `rows 2 and 13 share a 16-px cell but differ in column ${x}`
              );
              const near = pixels[at(128, x, 8)];
              const far = pixels[at(128, x, 24)];
              ctx.assert(
                Math.abs(near - far) >= 250,
                `rows 8 and 24 are in adjacent cells but match in column ${x} — did you use this.thread.y?`
              );
            }
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            const pixels = ctx.getPixels();
            // Full structural check, independent of row order and of which
            // corner is black: read the corner, then demand perfect parity.
            const base = pixels[0] >= 254 ? 255 : 0;
            let white = 0;
            for (let row = 0; row < 128; row++) {
              for (let x = 0; x < 128; x++) {
                const parity = (Math.floor(x / 16) + Math.floor(row / 16)) % 2;
                const expected = parity === 0 ? base : 255 - base;
                const got = pixels[at(128, x, row)];
                ctx.assert(
                  Math.abs(got - expected) <= 1,
                  `pixel (${x}, row ${row}) breaks the checkerboard: got ${got}`
                );
                if (got >= 254) white++;
              }
            }
            ctx.assert(white === 128 * 128 / 2, `expected exactly half the pixels white, got ${white}`);
          },
        },
      ],
    },

    // ------------------------------------------------------------------ 3
    {
      slug: 'plot-a-wave',
      title: 'Plot a Function',
      intro: `<p>How do you plot <code>y = f(x)</code> when no thread can draw a line? Flip the
        question: every pixel decides <em>for itself</em> whether it lies on the curve. Thread
        <code>(x, y)</code> evaluates the function at its own x, measures the vertical distance to
        that height, and paints amber if the distance is under 2 pixels — background otherwise.</p>
        <p>This per-pixel <em>"how far am I from the shape?"</em> question is one of the great
        tricks of computer graphics. Today it draws a sine wave; the same idea, pushed further,
        draws the fractals of module 3.2 and the ray-marched scenes of module 3.5.</p>`,
      goal: `<strong>Goal:</strong> plot one full period of
        <code>y = 64 + 40 · sin(2πx / 128)</code> as a thin amber curve on the dark background.`,
      requirements: [
        'Compute the curve height for this thread\'s x: <code>64 + 40 * Math.sin(x * 2 * Math.PI / 128)</code>',
        'Light the pixel when <code>Math.abs(this.thread.y - curveY) &lt; 2</code>',
        'Keep the amber-on-dark colors from the starter',
      ],
      hints: [
        {
          title: 'Hint 1 — one line changes',
          body: `<p>The distance test and both colors are already written. Only
            <code>curveY</code> is wrong: it's a constant, so you get a flat line instead of a
            wave.</p>`,
        },
        {
          title: 'Hint 2 — the curve',
          body: `<pre><code>const curveY = 64 + 40 * Math.sin(x * 2 * Math.PI / 128);</code></pre>
<p><code>Math.sin</code> and <code>Math.PI</code> both work inside kernels.</p>`,
        },
      ],
      transfer: `Distance-to-shape rendering is how GPUs draw crisp text and vector art at any
        zoom (signed distance fields), and it's the engine behind every Shadertoy graph you've
        seen: a WGSL or Metal fragment shader evaluating <code>f(x)</code> per fragment, exactly
        as here.`,
      starterCode: `// A plot is a per-pixel question: how far am I from the curve?
const gpu = new GPU({ mode });

const plot = gpu.createKernel(function () {
  const x = this.thread.x;
  // TODO: make this a real curve —
  //   y = 64 + 40 * Math.sin(x * 2 * Math.PI / 128)
  const curveY = 64;
  if (Math.abs(this.thread.y - curveY) < 2) {
    this.color(1, 0.85, 0.3, 1);      // on the curve — amber
  } else {
    this.color(0.06, 0.07, 0.1, 1);   // background — near black
  }
}, {
  output: [128, 128],
  graphical: true,
});

plot();
render(plot.canvas);
`,
      solutionCode: `// A plot is a per-pixel question: how far am I from the curve?
const gpu = new GPU({ mode });

const plot = gpu.createKernel(function () {
  const x = this.thread.x;
  const curveY = 64 + 40 * Math.sin(x * 2 * Math.PI / 128);
  if (Math.abs(this.thread.y - curveY) < 2) {
    this.color(1, 0.85, 0.3, 1);      // on the curve — amber
  } else {
    this.color(0.06, 0.07, 0.1, 1);   // background — near black
  }
}, {
  output: [128, 128],
  graphical: true,
});

plot();
render(plot.canvas);
`,
      publicTests: [
        {
          name: 'a thin curve on a dark background',
          run: async ctx => {
            ctx.assert(ctx.kernels.length >= 1, 'no kernel was created — call gpu.createKernel()');
            const pixels = ctx.getPixels();
            ctx.assert(pixels.length === 128 * 128 * 4, 'expected a 128×128 canvas');
            let lit = 0;
            for (let i = 0; i < pixels.length; i += 4) {
              if (pixels[i] > 128) lit++;
              else ctx.assert(pixels[i] < 40, `background pixel at byte ${i} is not dark (red ${pixels[i]})`);
            }
            const fraction = lit / (128 * 128);
            ctx.assert(
              fraction > 0.01 && fraction < 0.15,
              `expected a thin curve (1–15% of pixels lit), got ${(fraction * 100).toFixed(1)}%`
            );
          },
        },
        {
          name: 'every column crosses the curve exactly once',
          run: async ctx => {
            const pixels = ctx.getPixels();
            for (let x = 0; x < 128; x += 4) {
              const rows = litRowsInColumn(pixels, 128, x);
              ctx.assert(rows.length >= 1, `column ${x} has no lit pixels`);
              ctx.assert(
                rows.length <= 14,
                `column ${x} has ${rows.length} lit pixels — the band should stay thin`
              );
              ctx.assert(
                rows[rows.length - 1] - rows[0] === rows.length - 1,
                `column ${x} lights two separate bands — the curve should cross it once`
              );
            }
          },
        },
        {
          name: 'the curve follows <code>64 + 40·sin(2πx/128)</code>',
          run: async ctx => {
            const pixels = ctx.getPixels();
            // Detect buffer orientation where the wave peaks (x=32 → y≈104),
            // then hold every sampled column to the formula.
            const peak = mean(litRowsInColumn(pixels, 128, 32));
            const up = Math.abs(peak - curveHeight(32)) <= 4;
            const down = Math.abs(peak - (127 - curveHeight(32))) <= 4;
            ctx.assert(
              up || down,
              `at x=32 the curve should sit ~40 px from the middle (y≈104), found its center at buffer row ${peak.toFixed(1)}`
            );
            const orient = h => (up ? h : 127 - h);
            for (const x of [0, 8, 16, 32, 48, 64, 80, 96, 112, 120]) {
              const center = mean(litRowsInColumn(pixels, 128, x));
              const expected = orient(curveHeight(x));
              const hint = diagnose(center, expected, 3, curveProbes(x, orient));
              ctx.assertClose(center, expected, 3, hint || `curve center in column ${x}`);
            }
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            const pixels = ctx.getPixels();
            const peak = mean(litRowsInColumn(pixels, 128, 32));
            const up = Math.abs(peak - curveHeight(32)) <= 4;
            ctx.assert(
              up || Math.abs(peak - (127 - curveHeight(32))) <= 4,
              'curve peak is not where sin() puts it'
            );
            // Every single column, not just the public samples.
            for (let x = 0; x < 128; x++) {
              const rows = litRowsInColumn(pixels, 128, x);
              ctx.assert(rows.length >= 1 && rows.length <= 14, `column ${x}: ${rows.length} lit pixels`);
              const orient = h => (up ? h : 127 - h);
              const expected = orient(curveHeight(x));
              const hint = diagnose(mean(rows), expected, 3, curveProbes(x, orient));
              ctx.assertClose(mean(rows), expected, 3, hint || `curve center in column ${x}`);
            }
          },
        },
      ],
    },

    // ------------------------------------------------------------------ 4
    {
      slug: 'radial-ripples',
      title: 'Ripples: Think in Polar',
      intro: `<p>Gradients, cells and curves all thought in x and y. The last move of this module
        is to change coordinate systems <em>inside the kernel</em>: subtract the canvas center
        (63.5, 63.5 — halfway between the two middle rows and columns), and Pythagoras turns the
        thread's position into a <strong>radius</strong>. Anything you compute from that radius is
        automatically a perfect circle.</p>
        <p>Feed the radius into a cosine and you get concentric ripples; multiply by a fade so
        they die out toward the edge; tint the channels and the flat canvas turns into water.
        One gotcha, handled in the starter: <code>this.thread.x</code> is an <em>integer</em> on
        the GPU, so promote it to a float (<code>this.thread.x / 1</code>) before subtracting the
        fractional center — otherwise the GPU rounds your 63.5 away.</p>`,
      goal: `<strong>Goal:</strong> finish the ripple kernel — a cosine wave over the radius,
        faded toward the edge, tinted blue: <code>this.color(0.4v, 0.75v, v, 1)</code>.`,
      requirements: [
        'Radius from the center: <code>Math.sqrt(dx*dx + dy*dy)</code> with dx, dy relative to (63.5, 63.5)',
        'Ripple: <code>wave = 0.5 + 0.5 * Math.cos(r * 0.35)</code>',
        'Fade: <code>fade = Math.max(0, 1 - r / 96)</code>, then <code>v = wave * fade</code>',
        'Blue tint: channels <code>0.4*v</code>, <code>0.75*v</code>, <code>v</code>',
      ],
      hints: [
        {
          title: 'Hint 1 — the ripple',
          body: `<p><code>Math.cos(r * 0.35)</code> swings between −1 and 1 as r grows —
            <code>0.5 + 0.5 * cos</code> remaps that to 0…1, a crest roughly every 18 pixels.</p>`,
        },
        {
          title: 'Hint 2 — the last three lines',
          body: `<pre><code>const wave = 0.5 + 0.5 * Math.cos(r * 0.35);
const v = wave * Math.max(0, 1 - r / 96);
this.color(0.4 * v, 0.75 * v, v, 1);</code></pre>`,
        },
      ],
      transfer: `Radius-and-angle reasoning is everywhere in GPU code: vignette and
        lens-distortion passes in Metal and WebGPU post-processing, CUDA and ROCm image warps
        that resample in polar space, every "tunnel" demo ever shipped. Center the coordinates,
        transform them, color by the result — that opening move never changes.`,
      starterCode: `// Change coordinates INSIDE the kernel: position → radius from center.
const gpu = new GPU({ mode });

const ripples = gpu.createKernel(function () {
  // thread ids are integers — "/ 1" promotes them to floats so the
  // half-pixel center stays exact on the GPU
  const dx = this.thread.x / 1 - 63.5;
  const dy = this.thread.y / 1 - 63.5;
  const r = Math.sqrt(dx * dx + dy * dy);
  const fade = Math.max(0, 1 - r / 96);
  // TODO: 1) ripple — wave = 0.5 + 0.5 * Math.cos(r * 0.35)
  //       2) combine — v = wave * fade
  //       3) tint    — this.color(0.4 * v, 0.75 * v, v, 1)
  this.color(fade, fade, fade, 1);
}, {
  output: [128, 128],
  graphical: true,
});

ripples();
render(ripples.canvas);
`,
      solutionCode: `// Change coordinates INSIDE the kernel: position → radius from center.
const gpu = new GPU({ mode });

const ripples = gpu.createKernel(function () {
  // thread ids are integers — "/ 1" promotes them to floats so the
  // half-pixel center stays exact on the GPU
  const dx = this.thread.x / 1 - 63.5;
  const dy = this.thread.y / 1 - 63.5;
  const r = Math.sqrt(dx * dx + dy * dy);
  const wave = 0.5 + 0.5 * Math.cos(r * 0.35);  // crest every ~18 px
  const fade = Math.max(0, 1 - r / 96);         // dim toward the edge
  const v = wave * fade;
  this.color(0.4 * v, 0.75 * v, v, 1);
}, {
  output: [128, 128],
  graphical: true,
});

ripples();
render(ripples.canvas);
`,
      publicTests: [
        {
          name: 'the picture is radially symmetric about the center',
          run: async ctx => {
            ctx.assert(ctx.kernels.length >= 1, 'no kernel was created — call gpu.createKernel()');
            const pixels = ctx.getPixels();
            ctx.assert(pixels.length === 128 * 128 * 4, 'expected a 128×128 canvas');
            // A radius-only pattern must survive both mirror flips exactly —
            // and mirror checks are immune to row-order questions.
            for (const [x, row] of [[10, 30], [45, 8], [70, 100], [120, 60], [33, 33]]) {
              const i = at(128, x, row);
              const h = at(128, 127 - x, row);
              const v = at(128, x, 127 - row);
              for (let c = 0; c < 3; c++) {
                ctx.assert(
                  Math.abs(pixels[i + c] - pixels[h + c]) <= 2,
                  `pixel (${x}, row ${row}) and its horizontal mirror disagree — is the center at (63.5, 63.5)?`
                );
                ctx.assert(
                  Math.abs(pixels[i + c] - pixels[v + c]) <= 2,
                  `pixel (${x}, row ${row}) and its vertical mirror disagree — is the center at (63.5, 63.5)?`
                );
              }
            }
          },
        },
        {
          name: 'crests and troughs land where <code>cos(0.35r)</code> puts them',
          run: async ctx => {
            const pixels = ctx.getPixels();
            // Buffer row 63 maps to thread y 63 or 64 — either way |dy| = 0.5,
            // so the expected color is identical. x=64: bright center.
            // x=73 (r≈9.5): first trough. x=81 (r≈17.5): first ring.
            for (const x of [64, 73, 81, 99, 120]) {
              const i = at(128, x, 63);
              const expected = rippleRGB(x, 63.5 + 0.5);
              const hint = diagnose(pixels[i + 2], expected[2], 3, rippleProbes(x, 64, 2));
              ctx.assertClose(pixels[i + 2], expected[2], 3,
                hint || `blue in column ${x} of the center row`);
            }
            const center = pixels[at(128, 64, 63) + 2];
            const trough = pixels[at(128, 73, 63) + 2];
            ctx.assert(
              center - trough > 200,
              `the first trough should be nearly black next to the bright center (got ${center} vs ${trough})`
            );
          },
        },
        {
          name: 'blue tint and edge fade: <code>b &gt; g &gt; r</code>, corners dark',
          run: async ctx => {
            const pixels = ctx.getPixels();
            // On the first bright ring (x≈81 on the center row) the tint
            // ordering must hold, with the exact 0.4 / 0.75 ratios.
            const i = at(128, 81, 63);
            const [er, eg, eb] = rippleRGB(81, 64);
            ctx.assertClose(pixels[i], er, 3,
              diagnose(pixels[i], er, 3, rippleProbes(81, 64, 0)) || 'red on the first ring');
            ctx.assertClose(pixels[i + 1], eg, 3,
              diagnose(pixels[i + 1], eg, 3, rippleProbes(81, 64, 1)) || 'green on the first ring');
            ctx.assertClose(pixels[i + 2], eb, 3,
              diagnose(pixels[i + 2], eb, 3, rippleProbes(81, 64, 2)) || 'blue on the first ring');
            ctx.assert(
              pixels[i + 2] > pixels[i + 1] && pixels[i + 1] > pixels[i],
              'ring pixels should be tinted blue: b > g > r'
            );
            // Corners sit at r ≈ 89.8, where the fade leaves only ~6% —
            // they must be far dimmer than the first ring.
            for (const [x, row] of [[0, 0], [127, 0], [0, 127], [127, 127]]) {
              const j = at(128, x, row);
              const [cr, cg, cb] = rippleRGB(x < 64 ? 0 : 127, row < 64 ? 0 : 127);
              ctx.assertClose(pixels[j], cr, 3, `red in corner (${x}, row ${row})`);
              ctx.assertClose(pixels[j + 1], cg, 3, `green in corner (${x}, row ${row})`);
              ctx.assertClose(pixels[j + 2], cb, 3, `blue in corner (${x}, row ${row})`);
              ctx.assert(
                pixels[i + 2] - pixels[j + 2] > 150,
                `corner (${x}, row ${row}) should be far dimmer than the first ring`
              );
            }
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            const pixels = ctx.getPixels();
            // Full-grid check against the formula. rippleRGB depends only on
            // the distance to (63.5, 63.5), so it is flip-invariant and the
            // buffer row can stand in for thread y directly.
            for (let row = 0; row < 128; row += 3) {
              for (let x = 0; x < 128; x += 3) {
                const i = at(128, x, row);
                const [er, eg, eb] = rippleRGB(x, row);
                ctx.assertClose(pixels[i], er, 3,
                  diagnose(pixels[i], er, 3, rippleProbes(x, row, 0)) || `red at (${x}, row ${row})`);
                ctx.assertClose(pixels[i + 1], eg, 3,
                  diagnose(pixels[i + 1], eg, 3, rippleProbes(x, row, 1)) || `green at (${x}, row ${row})`);
                ctx.assertClose(pixels[i + 2], eb, 3,
                  diagnose(pixels[i + 2], eb, 3, rippleProbes(x, row, 2)) || `blue at (${x}, row ${row})`);
              }
            }
          },
        },
      ],
    },
  ],
};
