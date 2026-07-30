// Module 3.5 — Ray-Marched Metaballs.
//
// Six tasks: a sphere SDF sampled per pixel → smooth-min merges two spheres
// into a metaball field → the sphere-tracing loop finds the surface → finite-
// difference normals → Lambert diffuse → soft shadows with an A/B switch.
//
// Every scene lives on a fixed 64×64 orthographic canvas: pixel (ix, iy) maps
// to world (wx, wy) = ((ix - 32) / 16, (iy - 32) / 16), so the view spans
// roughly [-2, 2] and the center pixel is exactly the origin. All scenes are
// symmetric in y (lights keep ly = 0), which makes every test robust to
// getPixels() row order: exact center-row probes check buffer rows 32 AND 31
// (flip partners) and accept either. Every task passes in CPU mode.

// ---- plain-JS mirrors of the kernel math (used by tests) -------------------

function worldCoord(i) {
  return (i - 32) / 16;
}

// Polynomial smooth minimum (Inigo Quilez) — same formula the kernels use.
function sminJS(a, b, k) {
  const h = Math.max(k - Math.abs(a - b), 0) / k;
  return Math.min(a, b) - h * h * k * 0.25;
}

// Two metaballs at (±sep, 0, 0), radius r, blended with smooth-min k.
function sceneDistJS(x, y, z, sep, r, k) {
  const d1 = Math.sqrt((x + sep) * (x + sep) + y * y + z * z) - r;
  const d2 = Math.sqrt((x - sep) * (x - sep) + y * y + z * z) - r;
  return sminJS(d1, d2, k);
}

// ---- pixel-probe helpers ---------------------------------------------------

function pixelAt(pixels, x, y) {
  const i = (y * 64 + x) * 4;
  return [pixels[i], pixels[i + 1], pixels[i + 2], pixels[i + 3]];
}

// Exact-value probe on the center row, robust to row order: buffer rows 32
// and 31 are flip partners, so one of them is always the true center row.
function assertCenterRow(ctx, pixels, x, channel, expected, tol, label) {
  const a = pixelAt(pixels, x, 32)[channel];
  const b = pixelAt(pixels, x, 31)[channel];
  ctx.assert(
    Math.abs(a - expected) <= tol || Math.abs(b - expected) <= tol,
    `${label} — expected ≈${expected} ±${tol}, got ${a} (row 32) / ${b} (row 31)`
  );
}

function redRow32(pixels, x) {
  return pixelAt(pixels, x, 32)[0];
}

export default {
  id: '3-5',
  track: 3,
  title: 'Ray-Marched Metaballs',
  blurb: 'Signed distance fields and soft shadows — a real-time 3D scene with no triangles at all.',
  tasks: [
    // ------------------------------------------------------------------ 1
    {
      slug: 'sphere-sdf',
      title: 'The Sphere as a Number',
      intro: `<p>A <strong>signed distance field</strong> describes a shape with one function:
        for any point in space it returns the distance to the nearest surface — positive outside,
        <em>negative</em> inside, exactly zero on the skin. A whole sphere collapses into one line:
        <code>length(p - center) - radius</code>. No vertices, no triangles, just math.</p>
        <p>That's a perfect fit for a kernel: one thread per sample point, each evaluating the same
        tiny function. Here every pixel owns a point on the <code>z = 0</code> slice through the
        scene — pixel <code>(ix, iy)</code> maps to world <code>((ix - 32) / 16, (iy - 32) / 16)</code>,
        so the canvas spans −2…2 and the center pixel sits exactly at the origin.</p>`,
      goal: `<strong>Goal:</strong> make the kernel return the signed distance from this thread's
        world point to a sphere at <code>(cx, cy)</code> with radius <code>r</code>.`,
      requirements: [
        'Map the thread to world space: <code>(this.thread.x - 32) / 16</code> (already wired up)',
        'Measure the offset from the sphere center: <code>(wx - cx, wy - cy)</code>',
        'Return its length via <code>Math.sqrt</code>, <strong>minus</strong> <code>r</code>',
      ],
      hints: [
        {
          title: 'Hint 1 — what should the numbers look like?',
          body: `<p>For the unit sphere at the origin: the center pixel is <em>inside</em>, distance
            <code>-1</code>. A pixel one unit from the center sits exactly on the surface —
            distance <code>0</code>. The far corner at (−2, −2) reads <code>√8 − 1 ≈ 1.83</code>.</p>`,
        },
        {
          title: 'Hint 2 — the whole thing',
          body: `<pre><code>const dx = wx - cx;
const dy = wy - cy;
return Math.sqrt(dx * dx + dy * dy) - r;</code></pre>`,
        },
      ],
      transfer: `Distance fields are a production technique, not a toy: Valve renders crisp text
        from SDF textures, and every WebGPU fragment shader that draws rounded rectangles is
        evaluating exactly this per-pixel field — one invocation per pixel, one signed distance out.`,
      starterCode: `// A sphere in one line of math: length(p - center) - radius.
const gpu = new GPU({ mode });

const sliceSDF = gpu.createKernel(function (cx, cy, r) {
  // This thread's point on the z = 0 slice: center pixel = origin.
  const wx = (this.thread.x - 32) / 16;
  const wy = (this.thread.y - 32) / 16;
  // TODO: return the signed distance from (wx, wy) to the sphere:
  // the length of (wx - cx, wy - cy), minus r.
  return 0;
}, { output: [64, 64] });

const field = sliceSDF(0, 0, 1);
console.log('center (inside, should be -1):', field[32][32]);
console.log('far corner (outside):', field[0][0]);
`,
      solutionCode: `// A sphere in one line of math: length(p - center) - radius.
const gpu = new GPU({ mode });

const sliceSDF = gpu.createKernel(function (cx, cy, r) {
  const wx = (this.thread.x - 32) / 16;
  const wy = (this.thread.y - 32) / 16;
  const dx = wx - cx;
  const dy = wy - cy;
  return Math.sqrt(dx * dx + dy * dy) - r;
}, { output: [64, 64] });

const field = sliceSDF(0, 0, 1);
console.log('center (inside, should be -1):', field[32][32]);
console.log('far corner (outside):', field[0][0]);
`,
      publicTests: [
        {
          name: 'unit sphere: center reads −1, surface reads 0, corner reads ≈1.83',
          run: async ctx => {
            ctx.assert(ctx.kernels.length >= 1, 'no kernel was created — call gpu.createKernel()');
            const out = ctx.kernel(0, 0, 1);
            ctx.assert(out && out.length === 64 && out[0].length === 64, 'expected a 64×64 field');
            ctx.assertClose(out[32][32], -1, 2e-3, 'center of the sphere (inside)');
            ctx.assertClose(out[32][48], 0, 2e-3, 'one unit right of center (on the surface)');
            ctx.assertClose(out[0][0], Math.sqrt(8) - 1, 2e-3, 'far corner (outside)');
          },
        },
        {
          name: 'the field is radially symmetric around the sphere center',
          run: async ctx => {
            const out = ctx.kernel(0, 0, 1);
            for (const d of [5, 10, 15]) {
              ctx.assertClose(out[32][32 + d], out[32][32 - d], 2e-3, `left/right at offset ${d}`);
              ctx.assertClose(out[32][32 + d], out[32 + d][32], 2e-3, `x/y at offset ${d}`);
            }
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            // An off-center, smaller sphere — hardcoding the unit sphere fails here.
            const out = ctx.kernel(0.5, -0.25, 0.75);
            const cases = [[3, 7], [20, 44], [32, 32], [50, 12], [10, 58], [63, 63]];
            for (const [y, x] of cases) {
              const dx = worldCoord(x) - 0.5;
              const dy = worldCoord(y) + 0.25;
              const expected = Math.sqrt(dx * dx + dy * dy) - 0.75;
              ctx.assertClose(out[y][x], expected, 3e-3, `cell [${y}][${x}]`);
            }
          },
        },
      ],
    },

    // ------------------------------------------------------------------ 2
    {
      slug: 'smooth-min',
      title: 'Two Spheres Melt Into One',
      intro: `<p>Combining two SDFs is just <code>Math.min</code> — the nearest surface wins.
        But <code>min</code> leaves a hard crease where the shapes meet. Swap it for a
        <strong>smooth minimum</strong> and the fields <em>blend</em>: wherever the two distances
        are within <code>k</code> of each other, the result dips below both, bulging the surfaces
        toward each other. That bulge <em>is</em> a metaball.</p>
        <p>Since every task from here on needs this helper, register it once with
        <code>gpu.addFunction()</code> — gpu.js transpiles it alongside the kernel, and any kernel
        on that GPU instance can call it by name.</p>`,
      goal: `<strong>Goal:</strong> implement the polynomial smooth minimum and use it to blend
        two sphere fields into one metaball field.`,
      requirements: [
        'Register <code>smin(a, b, k)</code> with <code>gpu.addFunction()</code>',
        'Blend amount: <code>h = Math.max(k - Math.abs(a - b), 0) / k</code>',
        'Return <code>Math.min(a, b) - h * h * k * 0.25</code>',
        'Call <code>smin(d1, d2, k)</code> in the kernel instead of <code>Math.min</code>',
      ],
      hints: [
        {
          title: 'Hint 1 — what should change?',
          body: `<p>Far from the seam, <code>smin</code> equals plain <code>min</code>. Exactly
            between the spheres the two distances are equal, so <code>h = 1</code> and the field
            dips by <code>k / 4</code>. With the starter's arguments the midpoint should read
            <code>0.2 − 0.1 = 0.1</code>.</p>`,
        },
        {
          title: 'Hint 2 — the function, verbatim',
          body: `<pre><code>gpu.addFunction(function smin(a, b, k) {
  const h = Math.max(k - Math.abs(a - b), 0.0) / k;
  return Math.min(a, b) - h * h * k * 0.25;
});</code></pre>`,
        },
      ],
      transfer: `Smooth blends of implicit surfaces are how molecular-surface renderers in CUDA
        draw proteins and how Metal-based sculpting apps merge clay-like blobs — the union operator
        is soft everywhere, and the GPU evaluates it millions of times per frame without blinking.`,
      starterCode: `// min() gives a hard crease. smin() gives a blend. Metaballs are just smin.
const gpu = new GPU({ mode });

// TODO: register smin(a, b, k) with gpu.addFunction():
//   const h = Math.max(k - Math.abs(a - b), 0.0) / k;
//   return Math.min(a, b) - h * h * k * 0.25;

const metaField = gpu.createKernel(function (sep, r, k) {
  const wx = (this.thread.x - 32) / 16;
  const wy = (this.thread.y - 32) / 16;
  // one sphere at (-sep, 0), one at (+sep, 0)
  const d1 = Math.sqrt((wx + sep) * (wx + sep) + wy * wy) - r;
  const d2 = Math.sqrt((wx - sep) * (wx - sep) + wy * wy) - r;
  // TODO: blend with smin(d1, d2, k) instead of the hard minimum
  return Math.min(d1, d2);
}, { output: [64, 64] });

const field = metaField(0.7, 0.5, 0.4);
console.log('midpoint (should dip to 0.1):', field[32][32]);
`,
      solutionCode: `// min() gives a hard crease. smin() gives a blend. Metaballs are just smin.
const gpu = new GPU({ mode });

gpu.addFunction(function smin(a, b, k) {
  const h = Math.max(k - Math.abs(a - b), 0.0) / k;
  return Math.min(a, b) - h * h * k * 0.25;
});

const metaField = gpu.createKernel(function (sep, r, k) {
  const wx = (this.thread.x - 32) / 16;
  const wy = (this.thread.y - 32) / 16;
  const d1 = Math.sqrt((wx + sep) * (wx + sep) + wy * wy) - r;
  const d2 = Math.sqrt((wx - sep) * (wx - sep) + wy * wy) - r;
  return smin(d1, d2, k);
}, { output: [64, 64] });

const field = metaField(0.7, 0.5, 0.4);
console.log('midpoint (should dip to 0.1):', field[32][32]);
`,
      publicTests: [
        {
          name: 'midpoint dips below the plain minimum by <code>k / 4</code>',
          run: async ctx => {
            ctx.assert(ctx.kernels.length >= 1, 'no kernel was created — call gpu.createKernel()');
            const out = ctx.kernel(0.7, 0.5, 0.4);
            // both distances are 0.7 - 0.5 = 0.2 at the midpoint; smin dips by 0.4 / 4
            ctx.assertClose(out[32][32], 0.1, 3e-3, 'field at the midpoint');
            ctx.assert(out[32][32] < 0.2 - 0.05, 'the blend should dip clearly below plain min (0.2)');
          },
        },
        {
          name: 'field stays symmetric and returns to plain min far from the seam',
          run: async ctx => {
            const out = ctx.kernel(0.7, 0.5, 0.4);
            for (const d of [6, 12, 20]) {
              ctx.assertClose(out[32][32 + d], out[32][32 - d], 3e-3, `mirror pair at offset ${d}`);
            }
            // far corner: distances differ by more than k, so smin == min
            const wx = worldCoord(0);
            const wy = worldCoord(0);
            const d1 = Math.sqrt((wx + 0.7) * (wx + 0.7) + wy * wy) - 0.5;
            const d2 = Math.sqrt((wx - 0.7) * (wx - 0.7) + wy * wy) - 0.5;
            ctx.assertClose(out[0][0], Math.min(d1, d2), 3e-3, 'far corner (outside the blend zone)');
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            const [sep, r, k] = [0.9, 0.45, 0.3];
            const out = ctx.kernel(sep, r, k);
            for (let y = 1; y < 64; y += 7) {
              for (let x = 1; x < 64; x += 7) {
                const wx = worldCoord(x);
                const wy = worldCoord(y);
                const d1 = Math.sqrt((wx + sep) * (wx + sep) + wy * wy) - r;
                const d2 = Math.sqrt((wx - sep) * (wx - sep) + wy * wy) - r;
                ctx.assertClose(out[y][x], sminJS(d1, d2, k), 3e-3, `cell [${y}][${x}]`);
              }
            }
          },
        },
      ],
    },

    // ------------------------------------------------------------------ 3
    {
      slug: 'ray-march',
      title: 'March Until You Hit Something',
      intro: `<p>Now the third dimension. Every pixel fires a ray straight into the screen
        (orthographic: origin <code>(wx, wy, -2.5)</code>, direction <code>(0, 0, 1)</code>), and
        the SDF turns finding the surface into a beautiful trick called <strong>sphere tracing</strong>:
        the distance at your current point is a <em>guaranteed-safe step size</em> — nothing can be
        closer than that. So step exactly that far, re-evaluate, repeat.</p>
        <p>Near a surface the distance shrinks toward zero, so the march converges right onto the
        skin. When <code>d</code> drops below a small epsilon, that ray has hit. Rays that miss just
        keep flying — after a fixed number of steps you paint them background. GPUs need that fixed
        bound: every thread runs the same loop, so give it <code>48</code> iterations and let hits
        simply stop making progress.</p>`,
      goal: `<strong>Goal:</strong> write the ray-marching loop — 48 steps of
        <code>t += d</code> — and paint hit pixels pink, misses dark blue.`,
      requirements: [
        'Loop a fixed 48 times; sample the field at <code>(wx, wy, -2.5 + t)</code>',
        'Flag a hit when <code>d &lt; 0.01</code>',
        'Step forward by the distance itself: <code>t += d</code>',
        'Hits get <code>this.color(0.98, 0.63, 0.89, 1)</code>, misses the background color',
      ],
      hints: [
        {
          title: 'Hint 1 — the loop shape',
          body: `<p>Two state variables before the loop: <code>let t = 0.0;</code> and
            <code>let hit = 0.0;</code>. Inside: evaluate <code>sceneDist</code>, set
            <code>hit = 1.0</code> when close enough, then advance <code>t</code>. After the loop,
            color by <code>hit</code>.</p>`,
        },
        {
          title: 'Hint 2 — the loop, spelled out',
          body: `<pre><code>for (let i = 0; i &lt; 48; i++) {
  const d = sceneDist(wx, wy, -2.5 + t, sep, r, k);
  if (d &lt; 0.01) hit = 1.0;
  t += d;
}</code></pre>`,
        },
      ],
      transfer: `The fixed bound is a gpu.js/WebGL constraint — shader loop bounds must be
        static, so we guard instead of <code>break</code>. Real marchers on CUDA, WGSL or
        Shadertoy <em>do</em> break on a hit, and it pays off whenever a whole warp hits or
        misses together. The durable lesson is about <em>divergence</em>: warps execute in
        lockstep, so a warp runs as long as its slowest thread.`,
      starterCode: `// Sphere tracing: the SDF value IS a safe step size. Step, sample, repeat.
const gpu = new GPU({ mode });

gpu.addFunction(function smin(a, b, k) {
  const h = Math.max(k - Math.abs(a - b), 0.0) / k;
  return Math.min(a, b) - h * h * k * 0.25;
});

// The full 3D metaball field: two spheres at (±sep, 0, 0), blended by k.
gpu.addFunction(function sceneDist(x, y, z, sep, r, k) {
  const d1 = Math.sqrt((x + sep) * (x + sep) + y * y + z * z) - r;
  const d2 = Math.sqrt((x - sep) * (x - sep) + y * y + z * z) - r;
  return smin(d1, d2, k);
});

const marchScene = gpu.createKernel(function (sep, r, k) {
  const wx = (this.thread.x - 32) / 16;
  const wy = (this.thread.y - 32) / 16;
  // TODO: march! Start at t = 0 and repeat 48 times:
  //   d = sceneDist(wx, wy, -2.5 + t, sep, r, k)
  //   if d < 0.01 → this ray has hit the surface
  //   step forward: t += d
  // This only checks the starting plane — nothing is that close, so
  // every pixel comes out background until you write the loop:
  let hit = 0.0;
  if (sceneDist(wx, wy, -2.5, sep, r, k) < 0.01) hit = 1.0;

  if (hit > 0.5) this.color(0.98, 0.63, 0.89, 1);
  else this.color(0.02, 0.03, 0.06, 1);
}, { output: [64, 64], graphical: true });

marchScene(0.55, 0.5, 0.3);
render(marchScene.canvas);
`,
      solutionCode: `// Sphere tracing: the SDF value IS a safe step size. Step, sample, repeat.
const gpu = new GPU({ mode });

gpu.addFunction(function smin(a, b, k) {
  const h = Math.max(k - Math.abs(a - b), 0.0) / k;
  return Math.min(a, b) - h * h * k * 0.25;
});

// The full 3D metaball field: two spheres at (±sep, 0, 0), blended by k.
gpu.addFunction(function sceneDist(x, y, z, sep, r, k) {
  const d1 = Math.sqrt((x + sep) * (x + sep) + y * y + z * z) - r;
  const d2 = Math.sqrt((x - sep) * (x - sep) + y * y + z * z) - r;
  return smin(d1, d2, k);
});

const marchScene = gpu.createKernel(function (sep, r, k) {
  const wx = (this.thread.x - 32) / 16;
  const wy = (this.thread.y - 32) / 16;
  let t = 0.0;
  let hit = 0.0;
  for (let i = 0; i < 48; i++) {
    const d = sceneDist(wx, wy, -2.5 + t, sep, r, k);
    if (d < 0.01) hit = 1.0;
    t += d;
  }
  if (hit > 0.5) this.color(0.98, 0.63, 0.89, 1);
  else this.color(0.02, 0.03, 0.06, 1);
}, { output: [64, 64], graphical: true });

marchScene(0.55, 0.5, 0.3);
render(marchScene.canvas);
`,
      publicTests: [
        {
          name: 'a 64×64 canvas whose corners are background',
          run: async ctx => {
            ctx.assert(ctx.kernels.length >= 1, 'no kernel was created — call gpu.createKernel()');
            ctx.assert(ctx.canvas, 'no canvas — is the kernel graphical: true, and did you call render()?');
            ctx.assert(
              ctx.canvas.width === 64 && ctx.canvas.height === 64,
              `expected a 64×64 canvas, got ${ctx.canvas.width}×${ctx.canvas.height}`
            );
            const pixels = ctx.getPixels();
            for (const [x, y] of [[0, 0], [63, 0], [0, 63], [63, 63]]) {
              const [r] = pixelAt(pixels, x, y);
              ctx.assert(r < 40, `corner (${x}, ${y}) should be background, got red ${r}`);
            }
          },
        },
        {
          name: 'rays through the blob hit: center and both lobes come back pink',
          run: async ctx => {
            ctx.kernel(0.55, 0.5, 0.3);
            const pixels = ctx.getPixels();
            for (const x of [22, 32, 42]) {
              const r = redRow32(pixels, x);
              ctx.assert(r > 180, `pixel (${x}, 32) should be a hit (red > 180), got ${r}`);
            }
          },
        },
        {
          name: 'the silhouette is left-right symmetric',
          run: async ctx => {
            ctx.kernel(0.55, 0.5, 0.3);
            const pixels = ctx.getPixels();
            for (const d of [6, 10, 14, 18]) {
              const left = redRow32(pixels, 32 - d) > 128;
              const right = redRow32(pixels, 32 + d) > 128;
              ctx.assert(
                left === right,
                `pixels (${32 - d}, 32) and (${32 + d}, 32) should both hit or both miss`
              );
            }
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            // Pull the spheres apart: the center ray now passes clean between them.
            ctx.kernel(1.2, 0.45, 0.15);
            const pixels = ctx.getPixels();
            ctx.assert(
              redRow32(pixels, 32) < 40,
              `separated spheres: the center ray should miss, got red ${redRow32(pixels, 32)}`
            );
            for (const x of [13, 51]) {
              const r = redRow32(pixels, x);
              ctx.assert(r > 180, `pixel (${x}, 32) is inside a lobe and should hit, got red ${r}`);
            }
          },
        },
      ],
    },

    // ------------------------------------------------------------------ 4
    {
      slug: 'normals',
      title: 'Normals Without Geometry',
      intro: `<p>Lighting needs surface normals, and a mesh would hand them to you per-vertex. We
        have no mesh — but we have something better. The normal of an implicit surface is the
        <strong>gradient</strong> of its distance field: the direction in which distance grows
        fastest is exactly "straight off the surface".</p>
        <p>Estimate it with <strong>central differences</strong>: nudge the hit point by a tiny
        <code>e</code> along each axis, sample the field on both sides, subtract. Six extra field
        evaluations, then normalize. The classic way to sanity-check normals is to paint them:
        <code>n * 0.5 + 0.5</code> maps each component into color range — a head-on surface
        (normal <code>(0, 0, -1)</code>, pointing at the camera) renders as
        <code>rgb(0.5, 0.5, 0)</code>, that mustard-olive tone every graphics programmer knows.</p>`,
      goal: `<strong>Goal:</strong> at each hit point, compute the finite-difference normal of
        <code>sceneDist</code> and paint each component as <code>n * 0.5 + 0.5</code> —
        hint 2 has the exact <code>this.color</code> call.`,
      requirements: [
        'Remember the hit distance: record <code>tHit</code> at the <em>first</em> hit',
        'Sample ± <code>e = 0.01</code> along x, y and z around the hit point',
        'Normalize the three differences with <code>Math.sqrt</code>',
        'Paint <code>n * 0.5 + 0.5</code>; misses keep the background color',
      ],
      hints: [
        {
          title: 'Hint 1 — one axis at a time',
          body: `<p>The x component before normalizing is</p>
<pre><code>sceneDist(wx + e, wy, pz, …) - sceneDist(wx - e, wy, pz, …)</code></pre>
<p>where <code>pz = -2.5 + tHit</code>. Same pattern for y and z.</p>`,
        },
        {
          title: 'Hint 2 — normalize and paint',
          body: `<pre><code>const len = Math.sqrt(nx * nx + ny * ny + nz * nz);
this.color(nx / len * 0.5 + 0.5,
  ny / len * 0.5 + 0.5,
  nz / len * 0.5 + 0.5, 1);</code></pre>`,
        },
      ],
      transfer: `Gradient-by-central-differences is the same stencil you'd write in a CUDA fluid
        solver or a ROCm heightfield pipeline, and Metal deferred renderers reconstruct normals
        from depth buffers with exactly this two-sided sampling.`,
      starterCode: `// The normal of an SDF surface is its gradient. Six samples buy it.
const gpu = new GPU({ mode });

gpu.addFunction(function smin(a, b, k) {
  const h = Math.max(k - Math.abs(a - b), 0.0) / k;
  return Math.min(a, b) - h * h * k * 0.25;
});

gpu.addFunction(function sceneDist(x, y, z, sep, r, k) {
  const d1 = Math.sqrt((x + sep) * (x + sep) + y * y + z * z) - r;
  const d2 = Math.sqrt((x - sep) * (x - sep) + y * y + z * z) - r;
  return smin(d1, d2, k);
});

const showNormals = gpu.createKernel(function (sep, r, k) {
  const wx = (this.thread.x - 32) / 16;
  const wy = (this.thread.y - 32) / 16;
  let t = 0.0;
  let hit = 0.0;
  let tHit = 0.0;
  for (let i = 0; i < 48; i++) {
    const d = sceneDist(wx, wy, -2.5 + t, sep, r, k);
    if (hit < 0.5) {
      if (d < 0.01) {
        hit = 1.0;
        tHit = t;
      }
    }
    t += d;
  }
  if (hit > 0.5) {
    const pz = -2.5 + tHit;
    // TODO: central differences with e = 0.01 around (wx, wy, pz),
    // normalize (nx, ny, nz), then paint n * 0.5 + 0.5.
    this.color(1, 1, 1, 1);
  } else {
    this.color(0.02, 0.03, 0.06, 1);
  }
}, { output: [64, 64], graphical: true });

showNormals(0.55, 0.5, 0.3);
render(showNormals.canvas);
`,
      solutionCode: `// The normal of an SDF surface is its gradient. Six samples buy it.
const gpu = new GPU({ mode });

gpu.addFunction(function smin(a, b, k) {
  const h = Math.max(k - Math.abs(a - b), 0.0) / k;
  return Math.min(a, b) - h * h * k * 0.25;
});

gpu.addFunction(function sceneDist(x, y, z, sep, r, k) {
  const d1 = Math.sqrt((x + sep) * (x + sep) + y * y + z * z) - r;
  const d2 = Math.sqrt((x - sep) * (x - sep) + y * y + z * z) - r;
  return smin(d1, d2, k);
});

const showNormals = gpu.createKernel(function (sep, r, k) {
  const wx = (this.thread.x - 32) / 16;
  const wy = (this.thread.y - 32) / 16;
  let t = 0.0;
  let hit = 0.0;
  let tHit = 0.0;
  for (let i = 0; i < 48; i++) {
    const d = sceneDist(wx, wy, -2.5 + t, sep, r, k);
    if (hit < 0.5) {
      if (d < 0.01) {
        hit = 1.0;
        tHit = t;
      }
    }
    t += d;
  }
  if (hit > 0.5) {
    const pz = -2.5 + tHit;
    const e = 0.01;
    const nx = sceneDist(wx + e, wy, pz, sep, r, k) - sceneDist(wx - e, wy, pz, sep, r, k);
    const ny = sceneDist(wx, wy + e, pz, sep, r, k) - sceneDist(wx, wy - e, pz, sep, r, k);
    const nz = sceneDist(wx, wy, pz + e, sep, r, k) - sceneDist(wx, wy, pz - e, sep, r, k);
    const len = Math.sqrt(nx * nx + ny * ny + nz * nz);
    this.color(nx / len * 0.5 + 0.5, ny / len * 0.5 + 0.5, nz / len * 0.5 + 0.5, 1);
  } else {
    this.color(0.02, 0.03, 0.06, 1);
  }
}, { output: [64, 64], graphical: true });

showNormals(0.55, 0.5, 0.3);
render(showNormals.canvas);
`,
      publicTests: [
        {
          name: 'the head-on center pixel paints <code>rgb(0.5, 0.5, 0)</code> — normal (0, 0, −1)',
          run: async ctx => {
            ctx.assert(ctx.kernels.length >= 1, 'no kernel was created — call gpu.createKernel()');
            ctx.kernel(0.55, 0.5, 0.3);
            const pixels = ctx.getPixels();
            assertCenterRow(ctx, pixels, 32, 0, 128, 14, 'center red (nx = 0)');
            assertCenterRow(ctx, pixels, 32, 1, 128, 14, 'center green (ny = 0)');
            const b32 = pixelAt(pixels, 32, 32)[2];
            const b31 = pixelAt(pixels, 32, 31)[2];
            ctx.assert(
              b32 < 40 && b31 < 40,
              `center blue should be near 0 (nz = -1, facing the camera), got ${b32}/${b31}`
            );
          },
        },
        {
          name: 'mirrored hit pixels have mirrored normals: red channels sum to ≈255',
          run: async ctx => {
            ctx.kernel(0.55, 0.5, 0.3);
            const pixels = ctx.getPixels();
            for (const d of [8, 10]) {
              const left = pixelAt(pixels, 32 - d, 32);
              const right = pixelAt(pixels, 32 + d, 32);
              ctx.assert(left[1] > 60 && right[1] > 60, `both probes at ±${d} should be hits`);
              ctx.assert(
                Math.abs(left[0] + right[0] - 255) <= 24,
                `red(${32 - d}) + red(${32 + d}) should be ≈255 (nx antisymmetric), got ${left[0] + right[0]}`
              );
            }
          },
        },
        {
          name: 'misses keep the background color',
          run: async ctx => {
            ctx.kernel(0.55, 0.5, 0.3);
            const pixels = ctx.getPixels();
            for (const [x, y] of [[0, 0], [63, 0], [0, 63], [63, 63]]) {
              const [, g] = pixelAt(pixels, x, y);
              ctx.assert(g < 40, `corner (${x}, ${y}) should be background, got green ${g}`);
            }
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            // Separated spheres: center misses; a lobe-center pixel faces the camera.
            ctx.kernel(1.2, 0.45, 0.15);
            const pixels = ctx.getPixels();
            ctx.assert(pixelAt(pixels, 32, 32)[1] < 40, 'center should be background now');
            assertCenterRow(ctx, pixels, 51, 0, 124, 16, 'right lobe red (nx ≈ 0)');
            assertCenterRow(ctx, pixels, 51, 1, 128, 16, 'right lobe green (ny = 0)');
            const left = pixelAt(pixels, 13, 32);
            const right = pixelAt(pixels, 51, 32);
            ctx.assert(
              Math.abs(left[0] + right[0] - 255) <= 26,
              `mirrored lobe pixels should have mirrored nx, got ${left[0] + right[0]}`
            );
          },
        },
      ],
    },

    // ------------------------------------------------------------------ 5
    {
      slug: 'diffuse-lighting',
      title: 'Turn On the Light',
      intro: `<p>With normals in hand, lighting is one dot product. <strong>Lambert's law</strong>:
        a surface facing a light head-on catches full brightness; tilt it away and brightness falls
        with the cosine of the angle — which is exactly <code>n · l</code>, the dot product of the
        unit normal and the unit direction <em>toward</em> the light. Clamp it at zero so surfaces
        facing away go dark instead of negative.</p>
        <p>Two finishing touches make it look right: an <strong>ambient floor</strong> of
        <code>0.15</code> so shadowed sides stay readable, and an albedo tint — multiply the final
        brightness into the metaball's pink <code>(1.0, 0.62, 0.86)</code>.</p>`,
      goal: `<strong>Goal:</strong> light each hit point with
        <code>c = 0.15 + 0.85 * Math.max(nx * lx + ny * ly + nz * lz, 0)</code> and paint
        <code>this.color(c, c * 0.62, c * 0.86, 1)</code>.`,
      requirements: [
        'Take the dot product of the normal with the light direction <code>(lx, ly, lz)</code>',
        'Clamp with <code>Math.max(dot, 0.0)</code> — no negative light',
        'Apply the ambient floor: <code>c = 0.15 + 0.85 * diff</code>',
        'Tint by the albedo: <code>this.color(c, c * 0.62, c * 0.86, 1)</code>',
      ],
      hints: [
        {
          title: 'Hint 1 — sanity-check the center',
          body: `<p>The center pixel's normal is <code>(0, 0, -1)</code> and the light is
            <code>(-0.6, 0, -0.8)</code>, so the dot product is <code>0.8</code> and
            <code>c = 0.15 + 0.85 × 0.8 = 0.83</code> — a bright, not-quite-white pink.</p>`,
        },
        {
          title: 'Hint 2 — the two lines',
          body: `<pre><code>const diff = Math.max(nx / len * lx + ny / len * ly
  + nz / len * lz, 0.0);
const c = 0.15 + 0.85 * diff;</code></pre>`,
        },
      ],
      transfer: `<code>max(dot(n, l), 0.0)</code> is character-for-character the same in GLSL,
        WGSL, HLSL and Metal Shading Language — Lambert diffuse may be the single most portable
        line of shading code in existence.`,
      starterCode: `// Lighting is a dot product: brightness = how squarely you face the light.
const gpu = new GPU({ mode });

gpu.addFunction(function smin(a, b, k) {
  const h = Math.max(k - Math.abs(a - b), 0.0) / k;
  return Math.min(a, b) - h * h * k * 0.25;
});

gpu.addFunction(function sceneDist(x, y, z, sep, r, k) {
  const d1 = Math.sqrt((x + sep) * (x + sep) + y * y + z * z) - r;
  const d2 = Math.sqrt((x - sep) * (x - sep) + y * y + z * z) - r;
  return smin(d1, d2, k);
});

const shadeScene = gpu.createKernel(function (sep, r, k, lx, ly, lz) {
  const wx = (this.thread.x - 32) / 16;
  const wy = (this.thread.y - 32) / 16;
  let t = 0.0;
  let hit = 0.0;
  let tHit = 0.0;
  for (let i = 0; i < 48; i++) {
    const d = sceneDist(wx, wy, -2.5 + t, sep, r, k);
    if (hit < 0.5) {
      if (d < 0.01) {
        hit = 1.0;
        tHit = t;
      }
    }
    t += d;
  }
  if (hit > 0.5) {
    const pz = -2.5 + tHit;
    const e = 0.01;
    const nx = sceneDist(wx + e, wy, pz, sep, r, k) - sceneDist(wx - e, wy, pz, sep, r, k);
    const ny = sceneDist(wx, wy + e, pz, sep, r, k) - sceneDist(wx, wy - e, pz, sep, r, k);
    const nz = sceneDist(wx, wy, pz + e, sep, r, k) - sceneDist(wx, wy, pz - e, sep, r, k);
    const len = Math.sqrt(nx * nx + ny * ny + nz * nz);
    // TODO: Lambert — dot the unit normal with (lx, ly, lz), clamp at 0,
    // then c = 0.15 + 0.85 * diff. Flat 1.0 means "fully lit everywhere":
    const c = 1.0;
    this.color(c, c * 0.62, c * 0.86, 1);
  } else {
    this.color(0.02, 0.03, 0.06, 1);
  }
}, { output: [64, 64], graphical: true });

// light direction: up-left of the camera, pointing at the scene
shadeScene(0.55, 0.5, 0.3, -0.6, 0, -0.8);
render(shadeScene.canvas);
`,
      solutionCode: `// Lighting is a dot product: brightness = how squarely you face the light.
const gpu = new GPU({ mode });

gpu.addFunction(function smin(a, b, k) {
  const h = Math.max(k - Math.abs(a - b), 0.0) / k;
  return Math.min(a, b) - h * h * k * 0.25;
});

gpu.addFunction(function sceneDist(x, y, z, sep, r, k) {
  const d1 = Math.sqrt((x + sep) * (x + sep) + y * y + z * z) - r;
  const d2 = Math.sqrt((x - sep) * (x - sep) + y * y + z * z) - r;
  return smin(d1, d2, k);
});

const shadeScene = gpu.createKernel(function (sep, r, k, lx, ly, lz) {
  const wx = (this.thread.x - 32) / 16;
  const wy = (this.thread.y - 32) / 16;
  let t = 0.0;
  let hit = 0.0;
  let tHit = 0.0;
  for (let i = 0; i < 48; i++) {
    const d = sceneDist(wx, wy, -2.5 + t, sep, r, k);
    if (hit < 0.5) {
      if (d < 0.01) {
        hit = 1.0;
        tHit = t;
      }
    }
    t += d;
  }
  if (hit > 0.5) {
    const pz = -2.5 + tHit;
    const e = 0.01;
    const nx = sceneDist(wx + e, wy, pz, sep, r, k) - sceneDist(wx - e, wy, pz, sep, r, k);
    const ny = sceneDist(wx, wy + e, pz, sep, r, k) - sceneDist(wx, wy - e, pz, sep, r, k);
    const nz = sceneDist(wx, wy, pz + e, sep, r, k) - sceneDist(wx, wy, pz - e, sep, r, k);
    const len = Math.sqrt(nx * nx + ny * ny + nz * nz);
    const diff = Math.max((nx * lx + ny * ly + nz * lz) / len, 0.0);
    const c = 0.15 + 0.85 * diff;
    this.color(c, c * 0.62, c * 0.86, 1);
  } else {
    this.color(0.02, 0.03, 0.06, 1);
  }
}, { output: [64, 64], graphical: true });

// light direction: up-left of the camera, pointing at the scene
shadeScene(0.55, 0.5, 0.3, -0.6, 0, -0.8);
render(shadeScene.canvas);
`,
      publicTests: [
        {
          name: 'center pixel brightness matches Lambert: <code>0.15 + 0.85 × 0.8</code>',
          run: async ctx => {
            ctx.assert(ctx.kernels.length >= 1, 'no kernel was created — call gpu.createKernel()');
            ctx.kernel(0.55, 0.5, 0.3, -0.6, 0, -0.8);
            const pixels = ctx.getPixels();
            // n = (0, 0, -1), l = (-0.6, 0, -0.8) → diff 0.8 → c 0.83 → red ≈ 212
            assertCenterRow(ctx, pixels, 32, 0, 212, 14, 'center red');
            assertCenterRow(ctx, pixels, 32, 1, 131, 14, 'center green (red × 0.62)');
          },
        },
        {
          name: 'the side facing the light is brighter than the side facing away',
          run: async ctx => {
            ctx.kernel(0.55, 0.5, 0.3, -0.6, 0, -0.8);
            const pixels = ctx.getPixels();
            const lit = redRow32(pixels, 22);
            const unlit = redRow32(pixels, 42);
            ctx.assert(
              lit > unlit + 20,
              `light comes from the left: red(22) = ${lit} should exceed red(42) = ${unlit} by > 20`
            );
          },
        },
        {
          name: 'misses keep the background color',
          run: async ctx => {
            ctx.kernel(0.55, 0.5, 0.3, -0.6, 0, -0.8);
            const pixels = ctx.getPixels();
            for (const [x, y] of [[0, 0], [63, 63]]) {
              const [r] = pixelAt(pixels, x, y);
              ctx.assert(r < 40, `corner (${x}, ${y}) should be background, got red ${r}`);
            }
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            // Move the light to the right: the gradient must flip with it.
            ctx.kernel(0.55, 0.5, 0.3, 0.6, 0, -0.8);
            const pixels = ctx.getPixels();
            assertCenterRow(ctx, pixels, 32, 0, 212, 14, 'center red (same head-on dot product)');
            const left = redRow32(pixels, 22);
            const right = redRow32(pixels, 42);
            ctx.assert(
              right > left + 20,
              `light from the right: red(42) = ${right} should exceed red(22) = ${left} by > 20`
            );
          },
        },
      ],
    },

    // ------------------------------------------------------------------ 6
    {
      slug: 'soft-shadows',
      title: 'Soft Shadows, Full Scene',
      intro: `<p>The payoff. A point is in shadow when something sits between it and the light —
        and you already own the tool that answers that: <em>march again</em>, from the hit point
        toward the light. Here's Inigo Quilez's beautiful upgrade: instead of a binary blocked/clear,
        track how <em>closely</em> the shadow ray grazes the scene. The running minimum of
        <code>3 · d / t</code> (distance over travel) is ≈1 when the ray stays clear, 0 when it's
        blocked, and slides smoothly between when it grazes — a free penumbra.</p>
        <p>The kernel takes a <code>shadowOn</code> switch so you can A/B it: with the light swung
        low to the left, the left lobe casts a soft-edged shadow across the neck of the blob. One
        more march, and a scene with no triangles anywhere gets real cinematography.</p>`,
      goal: `<strong>Goal:</strong> add the shadow march — 32 steps from the hit point toward the
        light, penumbra factor <code>sh = Math.min(sh, 3.0 * d / st)</code> — and scale the diffuse
        term by it when <code>shadowOn</code> is 1.`,
      requirements: [
        'Start the shadow ray at <code>st = 0.06</code> so it clears its own surface',
        'Sample at <code>(wx + lx·st, wy + ly·st, pz + lz·st)</code>, 32 steps',
        'Keep the running minimum <code>sh = Math.min(sh, 3.0 * d / st)</code>, clamped ≥ 0',
        'Advance with <code>st += Math.max(d, 0.02)</code>, then shade <code>c = 0.15 + 0.85 * diff * sh</code>',
      ],
      hints: [
        {
          title: 'Hint 1 — why 3 · d / t?',
          body: `<p>At travel distance <code>st</code>, a field value <code>d</code> means the ray
            passes within <code>d</code> of an occluder. The ratio <code>d / st</code> is the sine
            of the "clearance angle" from the surface point — small angle, deep penumbra. The 3
            just sets how sharp the shadow edge is.</p>`,
        },
        {
          title: 'Hint 2 — the loop, spelled out',
          body: `<pre><code>let sh = 1.0;
let st = 0.06;
for (let j = 0; j &lt; 32; j++) {
  const d = sceneDist(wx + lx * st, wy + ly * st,
    pz + lz * st, sep, r, k);
  sh = Math.min(sh, 3.0 * d / st);
  st += Math.max(d, 0.02);
}
sh = Math.max(sh, 0.0);</code></pre>
<p>And remember to only apply it when
            <code>shadowOn &gt; 0.5</code>.</p>`,
        },
      ],
      transfer: `Secondary rays are the moment ray-marching meets real-time ray tracing: shadow
        rays are exactly what RTX/DXR hardware accelerates, and this penumbra estimate ships in
        countless WGSL and GLSL engines as the cheap alternative when you can't afford one.`,
      starterCode: `// One more march — from the surface toward the light — buys shadows.
const gpu = new GPU({ mode });

gpu.addFunction(function smin(a, b, k) {
  const h = Math.max(k - Math.abs(a - b), 0.0) / k;
  return Math.min(a, b) - h * h * k * 0.25;
});

gpu.addFunction(function sceneDist(x, y, z, sep, r, k) {
  const d1 = Math.sqrt((x + sep) * (x + sep) + y * y + z * z) - r;
  const d2 = Math.sqrt((x - sep) * (x - sep) + y * y + z * z) - r;
  return smin(d1, d2, k);
});

const finalScene = gpu.createKernel(function (sep, r, k, lx, ly, lz, shadowOn) {
  const wx = (this.thread.x - 32) / 16;
  const wy = (this.thread.y - 32) / 16;
  let t = 0.0;
  let hit = 0.0;
  let tHit = 0.0;
  for (let i = 0; i < 48; i++) {
    const d = sceneDist(wx, wy, -2.5 + t, sep, r, k);
    if (hit < 0.5) {
      if (d < 0.01) {
        hit = 1.0;
        tHit = t;
      }
    }
    t += d;
  }
  if (hit > 0.5) {
    const pz = -2.5 + tHit;
    const e = 0.01;
    const nx = sceneDist(wx + e, wy, pz, sep, r, k) - sceneDist(wx - e, wy, pz, sep, r, k);
    const ny = sceneDist(wx, wy + e, pz, sep, r, k) - sceneDist(wx, wy - e, pz, sep, r, k);
    const nz = sceneDist(wx, wy, pz + e, sep, r, k) - sceneDist(wx, wy, pz - e, sep, r, k);
    const len = Math.sqrt(nx * nx + ny * ny + nz * nz);
    const diff = Math.max((nx * lx + ny * ly + nz * lz) / len, 0.0);

    let sh = 1.0;
    if (shadowOn > 0.5) {
      // TODO: march toward the light from st = 0.06, 32 steps:
      //   d = sceneDist(wx + lx * st, wy + ly * st, pz + lz * st, sep, r, k)
      //   sh = Math.min(sh, 3.0 * d / st)
      //   st += Math.max(d, 0.02)
      // then clamp: sh = Math.max(sh, 0.0)
    }

    const c = 0.15 + 0.85 * diff * sh;
    this.color(c, c * 0.62, c * 0.86, 1);
  } else {
    this.color(0.02, 0.03, 0.06, 1);
  }
}, { output: [64, 64], graphical: true });

// light swung low to the left — the left lobe should shade the neck
finalScene(0.55, 0.5, 0.3, -0.86, 0, -0.51, 1);
render(finalScene.canvas);
`,
      solutionCode: `// One more march — from the surface toward the light — buys shadows.
const gpu = new GPU({ mode });

gpu.addFunction(function smin(a, b, k) {
  const h = Math.max(k - Math.abs(a - b), 0.0) / k;
  return Math.min(a, b) - h * h * k * 0.25;
});

gpu.addFunction(function sceneDist(x, y, z, sep, r, k) {
  const d1 = Math.sqrt((x + sep) * (x + sep) + y * y + z * z) - r;
  const d2 = Math.sqrt((x - sep) * (x - sep) + y * y + z * z) - r;
  return smin(d1, d2, k);
});

const finalScene = gpu.createKernel(function (sep, r, k, lx, ly, lz, shadowOn) {
  const wx = (this.thread.x - 32) / 16;
  const wy = (this.thread.y - 32) / 16;
  let t = 0.0;
  let hit = 0.0;
  let tHit = 0.0;
  for (let i = 0; i < 48; i++) {
    const d = sceneDist(wx, wy, -2.5 + t, sep, r, k);
    if (hit < 0.5) {
      if (d < 0.01) {
        hit = 1.0;
        tHit = t;
      }
    }
    t += d;
  }
  if (hit > 0.5) {
    const pz = -2.5 + tHit;
    const e = 0.01;
    const nx = sceneDist(wx + e, wy, pz, sep, r, k) - sceneDist(wx - e, wy, pz, sep, r, k);
    const ny = sceneDist(wx, wy + e, pz, sep, r, k) - sceneDist(wx, wy - e, pz, sep, r, k);
    const nz = sceneDist(wx, wy, pz + e, sep, r, k) - sceneDist(wx, wy, pz - e, sep, r, k);
    const len = Math.sqrt(nx * nx + ny * ny + nz * nz);
    const diff = Math.max((nx * lx + ny * ly + nz * lz) / len, 0.0);

    let sh = 1.0;
    if (shadowOn > 0.5) {
      let st = 0.06;
      for (let j = 0; j < 32; j++) {
        const d = sceneDist(wx + lx * st, wy + ly * st, pz + lz * st, sep, r, k);
        sh = Math.min(sh, 3.0 * d / st);
        st += Math.max(d, 0.02);
      }
      sh = Math.max(sh, 0.0);
    }

    const c = 0.15 + 0.85 * diff * sh;
    this.color(c, c * 0.62, c * 0.86, 1);
  } else {
    this.color(0.02, 0.03, 0.06, 1);
  }
}, { output: [64, 64], graphical: true });

// light swung low to the left — the left lobe should shade the neck
finalScene(0.55, 0.5, 0.3, -0.86, 0, -0.51, 1);
render(finalScene.canvas);
`,
      publicTests: [
        {
          name: 'a 64×64 canvas: hits keep an ambient floor, corners stay background',
          run: async ctx => {
            ctx.assert(ctx.kernels.length >= 1, 'no kernel was created — call gpu.createKernel()');
            ctx.assert(ctx.canvas, 'no canvas — is the kernel graphical: true, and did you call render()?');
            ctx.assert(
              ctx.canvas.width === 64 && ctx.canvas.height === 64,
              `expected a 64×64 canvas, got ${ctx.canvas.width}×${ctx.canvas.height}`
            );
            ctx.kernel(0.55, 0.5, 0.3, -0.86, 0, -0.51, 1);
            const pixels = ctx.getPixels();
            // even a fully shadowed hit keeps the 0.15 ambient floor (red ≈ 38)
            ctx.assert(
              redRow32(pixels, 32) >= 28,
              `center should be a hit with at least the ambient floor, got red ${redRow32(pixels, 32)}`
            );
            for (const [x, y] of [[0, 0], [63, 63]]) {
              const [r] = pixelAt(pixels, x, y);
              ctx.assert(r <= 18, `corner (${x}, ${y}) should be pure background, got red ${r}`);
            }
          },
        },
        {
          name: 'toggling <code>shadowOn</code> darkens the neck of the blob by > 60',
          run: async ctx => {
            ctx.kernel(0.55, 0.5, 0.3, -0.86, 0, -0.51, 1);
            const withShadow = Array.from(ctx.getPixels());
            ctx.kernel(0.55, 0.5, 0.3, -0.86, 0, -0.51, 0);
            const noShadow = Array.from(ctx.getPixels());
            let maxDrop = 0;
            for (let x = 28; x <= 40; x++) {
              maxDrop = Math.max(maxDrop, redRow32(noShadow, x) - redRow32(withShadow, x));
            }
            ctx.assert(
              maxDrop > 60,
              `expected the shadow to darken some neck pixel by > 60, biggest drop was ${maxDrop}`
            );
          },
        },
        {
          name: 'shadows only ever darken — and the lit flank is untouched',
          run: async ctx => {
            ctx.kernel(0.55, 0.5, 0.3, -0.86, 0, -0.51, 1);
            const withShadow = Array.from(ctx.getPixels());
            ctx.kernel(0.55, 0.5, 0.3, -0.86, 0, -0.51, 0);
            const noShadow = Array.from(ctx.getPixels());
            for (let x = 0; x < 64; x++) {
              ctx.assert(
                redRow32(withShadow, x) <= redRow32(noShadow, x) + 10,
                `pixel (${x}, 32) got BRIGHTER with shadows on — sh must stay ≤ 1`
              );
            }
            const litOn = redRow32(withShadow, 18);
            const litOff = redRow32(noShadow, 18);
            ctx.assert(
              Math.abs(litOn - litOff) <= 10,
              `pixel (18, 32) faces the light with a clear path — it should not change (${litOff} → ${litOn})`
            );
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            // New geometry AND a mirrored light: the shadow must move to the
            // other flank, and the now-separated center ray must miss.
            ctx.kernel(0.6, 0.48, 0.25, 0.86, 0, -0.51, 1);
            const withShadow = Array.from(ctx.getPixels());
            ctx.kernel(0.6, 0.48, 0.25, 0.86, 0, -0.51, 0);
            const noShadow = Array.from(ctx.getPixels());
            ctx.assert(redRow32(withShadow, 32) < 20, 'center ray should miss the separated blobs');
            let maxDrop = 0;
            for (let x = 24; x <= 36; x++) {
              maxDrop = Math.max(maxDrop, redRow32(noShadow, x) - redRow32(withShadow, x));
            }
            ctx.assert(
              maxDrop > 60,
              `mirrored light: expected a shadow drop > 60 on the left flank, got ${maxDrop}`
            );
            const litOn = redRow32(withShadow, 46);
            const litOff = redRow32(noShadow, 46);
            ctx.assert(
              Math.abs(litOn - litOff) <= 10,
              `pixel (46, 32) faces the mirrored light — it should not change (${litOff} → ${litOn})`
            );
          },
        },
      ],
    },
  ],
};
