// Module: Progressive Path Tracing — uuid c99efc67-071e-441a-91ac-62cb82009ca4
// (short id c99efc67). The file name is the uuid; identity lives in the
// exported object below, never in the path. No legacyId: this module postdates
// the pre-uuid URLs.
//
// Five tasks: a per-thread random stream hashed out of the thread's own
// coordinates → analytic ray/sphere intersection → one path per pixel, one
// unusably noisy frame → temporal accumulation in a texture that outlives the
// frame (the payoff: a scrubber you drag from static to a picture) → the noise
// measured against 1/√n.
//
// THE SCENE, once, so every task can share it. Camera at (0, 0.3, 1.5) looking
// down −z through a plane 1.7 in front of it, 64×64 pixels spanning u, v ∈
// [−1, 1]. Two spheres — A at (−0.45, 0, −0.4) r 0.5 albedo 0.85, B at
// (0.5, −0.2, −0.1) r 0.3 albedo 0.3 — standing on an infinite floor at
// y = −0.5, albedo 0.55. The only light is the sky: a dome whose radiance is
// 0.3 at the horizon rising to 1.0 straight up. Radiance is one channel, not
// three: this module is about where the noise goes, and grey noise says it as
// well as coloured noise for a third of the work.
//
// DETERMINISM. Kernels never call Math.random — WebGPU refuses to compile it,
// and it would make every run and every test irreproducible besides. Every
// random number comes from nextRandom(), a pure function of a state the thread
// derives from this.thread.x, this.thread.y and the frame counter. Same code,
// same machine, same picture, every time.
//
// WHAT THE TESTS CAN AND CANNOT ASSERT. The hash is evaluated in float32 on a
// GPU and float64 on the CPU backend, so the two backends draw genuinely
// different random numbers and no per-pixel value is portable. What IS portable
// is the thing the estimator estimates: region means converge to the same
// answer on every backend, and the tests check those against a float64 mirror
// of these exact kernels run at 8,000 samples per pixel. Tolerances below are
// sized from the measured spread of those means, not guessed.
//
// Kernel-authoring rules per the contract: no closures, arguments + literals +
// this.thread.* only, statically bounded loops (the samples-per-frame loop is
// bounded by loopMaxIterations: 8 instead). Every task passes in CPU mode.

// ---------------------------------------------------------------------------
// Shared kernel source. The scene is long enough that retyping it in five
// starters and five solutions would guarantee they drift apart; these strings
// are the single copy, interpolated into both.
// ---------------------------------------------------------------------------

const RNG_FN = `// The whole random-number generator. It is a pure function: same state in,
// same number out, on every backend and every run.
gpu.addFunction(function nextRandom(state) {
  let p = state * 78.233 + 0.7213;
  p = p - Math.floor(p);
  p = p * (p + 61.7);
  return p - Math.floor(p);
});`;

const SEED_LINES = `  // This thread's own stream: start from the frame, stir in the column,
  // then the row. Every stir goes through nextRandom, so two threads one
  // pixel apart come out completely unrelated.
  let seed = 0.1237 + frame * 0.0173;
  seed = nextRandom(seed + this.thread.x * 0.0713);
  seed = nextRandom(seed + this.thread.y * 0.0917);`;

const SPHERE_FN = `// Nearest hit of a unit-direction ray against a sphere, or -1 for a miss.
gpu.addFunction(function sphereT(ox, oy, oz, dx, dy, dz, cx, cy, cz, r) {
  const ex = ox - cx;
  const ey = oy - cy;
  const ez = oz - cz;
  const b = ex * dx + ey * dy + ez * dz;
  const c = ex * ex + ey * ey + ez * ez - r * r;
  const disc = b * b - c;
  if (disc <= 0) {
    return -1;
  }
  const t = -b - Math.sqrt(disc);
  if (t <= 0.001) {
    return -1;
  }
  return t;
});`;

const REST_OF_SCENE = `// The floor: one division, no quadratic.
gpu.addFunction(function planeT(oy, dy, planeY) {
  if (dy > -0.0001) {
    return -1;
  }
  const t = (planeY - oy) / dy;
  if (t <= 0.001) {
    return -1;
  }
  return t;
});

// The only light in the scene is the sky itself.
gpu.addFunction(function skyLight(dy) {
  if (dy < 0) {
    return 0.3;
  }
  return 0.3 + 0.7 * dy;
});`;

const SCATTER_SOLVED = `      through = through * albedo;
      dx = nx + sx;
      dy = ny + sy;
      dz = nz + sz;
      len = Math.sqrt(dx * dx + dy * dy + dz * dz);
      dx = dx / len;
      dy = dy / len;
      dz = dz / len;`;

const SCATTER_TODO = `      // TODO: absorb, then bounce.
      //   1. the surface keeps some of the light: multiply \`through\` by \`albedo\`
      //   2. the new direction is the NORMAL plus that random unit vector,
      //      normalized — nx + sx, ny + sy, nz + sz, divided by its own length
      through = through * 1;
      dx = sx;
      dy = sy;
      dz = sz;
      len = Math.sqrt(dx * dx + dy * dy + dz * dz);
      dx = dx / len;
      dy = dy / len;
      dz = dz / len;`;

// The path tracer itself. `scatter` is the one block a task may leave open.
function traceKernel(scatter, settings) {
  return `const trace = gpu.createKernel(function (frame, spp) {
${SEED_LINES}

  let total = 0;
  for (let s = 0; s < spp; s++) {
    // One camera ray, jittered inside this pixel — so the edges antialias
    // themselves for free as the frames pile up.
    seed = nextRandom(seed);
    const u = ((this.thread.x + seed) / 64) * 2 - 1;
    seed = nextRandom(seed);
    const v = ((this.thread.y + seed) / 64) * 2 - 1;
    let dx = u;
    let dy = v;
    let dz = -1.7;
    let len = Math.sqrt(dx * dx + dy * dy + dz * dz);
    dx = dx / len;
    dy = dy / len;
    dz = dz / len;
    let ox = 0;
    let oy = 0.3;
    let oz = 1.5;
    let through = 1;

    // Up to three segments: camera → surface → surface → sky.
    for (let d = 0; d < 3; d++) {
      const tA = sphereT(ox, oy, oz, dx, dy, dz, -0.45, 0, -0.4, 0.5);
      const tB = sphereT(ox, oy, oz, dx, dy, dz, 0.5, -0.2, -0.1, 0.3);
      const tP = planeT(oy, dy, -0.5);
      let t = 999;
      let which = -1;
      if (tA > 0 && tA < t) {
        t = tA;
        which = 0;
      }
      if (tB > 0 && tB < t) {
        t = tB;
        which = 1;
      }
      if (tP > 0 && tP < t) {
        t = tP;
        which = 2;
      }
      if (which < 0) {
        // Nothing in the way — the path ends in the sky, which is the light.
        total = total + through * skyLight(dy);
        break;
      }

      const hx = ox + t * dx;
      const hy = oy + t * dy;
      const hz = oz + t * dz;
      let nx = 0;
      let ny = 1;
      let nz = 0;
      let albedo = 0.55;
      if (which === 0) {
        nx = (hx + 0.45) / 0.5;
        ny = hy / 0.5;
        nz = (hz + 0.4) / 0.5;
        albedo = 0.85;
      }
      if (which === 1) {
        nx = (hx - 0.5) / 0.3;
        ny = (hy + 0.2) / 0.3;
        nz = (hz + 0.1) / 0.3;
        albedo = 0.3;
      }

      // A uniformly random point on the unit sphere, out of this thread's
      // stream: z is uniform on [-1, 1] and phi uniform around the axis.
      seed = nextRandom(seed);
      const z = 2 * seed - 1;
      seed = nextRandom(seed);
      const phi = 6.2831853 * seed;
      const rr = Math.sqrt(Math.max(1 - z * z, 0));
      const sx = rr * Math.cos(phi);
      const sy = rr * Math.sin(phi);
      const sz = z;

${scatter}

      // Step off the surface so the next ray does not hit it at t = 0.
      ox = hx + 0.001 * nx;
      oy = hy + 0.001 * ny;
      oz = hz + 0.001 * nz;
    }
  }
  return total / spp;
}, { output: [64, 64], loopMaxIterations: 8${settings || ''} });`;
}

const PAINT_KERNEL = `// Radiance in, pixels out. The square root is a gamma curve: it is what
// makes a 0.2 and a 0.4 look as different as they are.
const paint = gpu.createKernel(function (image) {
  const g = Math.sqrt(Math.max(image[this.thread.y][this.thread.x], 0));
  this.color(g, g, g, 1);
}, { output: [64, 64], graphical: true });`;

const BLACK_KERNEL = `// A buffer of zeros to start the average from — the same trick Pipelines &
// Textures uses to get its seed onto the card as a texture.
const black = gpu.createKernel(function () {
  return 0;
}, { output: [64, 64], pipeline: true });`;

// ---------------------------------------------------------------------------
// The scene again, in plain float64 JavaScript. Tests use this to predict what
// a ray does; nothing here is ever shown to the learner.
// ---------------------------------------------------------------------------

const CAM = { x: 0, y: 0.3, z: 1.5 };
const FOCAL = 1.7;

function rayFor(px, py) {
  const u = ((px + 0.5) / 64) * 2 - 1;
  const v = ((py + 0.5) / 64) * 2 - 1;
  const len = Math.sqrt(u * u + v * v + FOCAL * FOCAL);
  return { dx: u / len, dy: v / len, dz: -FOCAL / len };
}

// The quadratic, spelled out so the tests can also ask for the pieces a
// near-miss probe needs (the far root, the discriminant, b on its own).
function sphereParts(o, d, c, r) {
  const ex = o.x - c.x;
  const ey = o.y - c.y;
  const ez = o.z - c.z;
  const b = ex * d.dx + ey * d.dy + ez * d.dz;
  const cc = ex * ex + ey * ey + ez * ez - r * r;
  const disc = b * b - cc;
  return { b, c: cc, disc, dist: Math.sqrt(ex * ex + ey * ey + ez * ez) };
}

function sphereTRef(o, d, c, r) {
  const { b, disc } = sphereParts(o, d, c, r);
  if (disc <= 0) return -1;
  const t = -b - Math.sqrt(disc);
  return t <= 0.001 ? -1 : t;
}

// ---------------------------------------------------------------------------
// Ground truth. Region means of the converged image, from a float64 mirror of
// these exact kernels at 2,000 frames × 4 samples = 8,000 samples per pixel.
// Regions rather than pixels: a Monte Carlo image is only ever right ON
// AVERAGE, and averaging 1,024 pixels of it is what turns "roughly" into a
// number a test can hold to three decimals.
// ---------------------------------------------------------------------------

const REGIONS = {
  whole: [0, 0, 64, 64],
  sky: [0, 48, 64, 64], // above everything — primary rays hit nothing at all
  floor: [0, 0, 30, 8], // near floor, clear of both spheres
  ballA: [14, 18, 25, 29], // inside the big pale sphere
  ballB: [45, 11, 54, 20], // inside the small dark one
};
const REGION_NAMES = Object.keys(REGIONS);

const TRUTH = { whole: 0.4037, sky: 0.5685, floor: 0.2179, ballA: 0.4481, ballB: 0.1698 };

// Measured spread of each region mean at ONE sample per pixel (24 trials):
// whole 0.0019, sky 0.00008, floor 0.0126, ballA 0.0152, ballB 0.0068. A test
// that averages k frames divides those by √k; the tolerances below are five to
// ten of the resulting sigmas, which is wide enough that a correct answer
// never trips and narrow enough that every wrong scatter measured during
// authoring does.
const TOL_4 = { whole: 0.012, sky: 0.01, floor: 0.035, ballA: 0.04, ballB: 0.022 };
const TOL_12 = { whole: 0.008, sky: 0.008, floor: 0.022, ballA: 0.026, ballB: 0.014 };
const TOL_ACC = { whole: 0.006, sky: 0.006, floor: 0.016, ballA: 0.018, ballB: 0.01 };

// The same region means for four scatters that are wrong in a specific way,
// each measured at 300 samples per pixel with the same float64 mirror. A probe
// may only speak when it predicts EVERY region — one region agreeing proves
// nothing when four candidates all sit within a few hundredths of each other.
const WRONG_SCATTERS = [
  {
    means: { whole: 0.3292, sky: 0.5685, floor: 0.1299, ballA: 0.2607, ballB: 0.0844 },
    message:
      'the new direction IS the random unit vector — half of those directions point straight ' +
      'into the surface, so half your light is lost underground. Add it to the normal: ' +
      'dx = nx + sx, dy = ny + sy, dz = nz + sz.',
  },
  {
    means: { whole: 0.5525, sky: 0.5685, floor: 0.4068, ballA: 0.6641, ballB: 0.6775 },
    message:
      'nothing in this scene is absorbing any light — the throughput never picked up the ' +
      'surface albedo. One multiply per bounce: through = through * albedo.',
  },
  {
    means: { whole: 0.429, sky: 0.5685, floor: 0.1781, ballA: 0.4037, ballB: 0.1593 },
    message:
      'every bounce left along the normal itself — the random vector never reached the ' +
      'direction, so this image is not a random estimate of anything and no amount of ' +
      'accumulating will improve it.',
  },
  {
    means: { whole: 0.425, sky: 0.5685, floor: 0.1699, ballA: 0.4886, ballB: 0.1939 },
    message:
      'the bounce direction was never normalized — sphereT and planeT both assume a unit ' +
      'direction, so t stops meaning a distance the moment the length drifts off 1.',
  },
];

// ---------------------------------------------------------------------------
// Test plumbing
// ---------------------------------------------------------------------------

// Mode-safe read of a pipeline result: a Texture on WebGL, a buffer result on
// WebGPU, a plain array on the CPU backend — and toArray() is itself a Promise
// under the async contract, so the await goes in front of the call it guards.
async function toArr(value) {
  return value && typeof value.toArray === 'function' ? await value.toArray() : value;
}

// A COPY of that, and the difference matters exactly once. On the GPU backends
// toArray() builds a fresh array every time; on the CPU backend a pipelined
// kernel hands back its own output storage, so the numbers a test is still
// holding change the next time that kernel runs. Anything that keeps several
// frames at once — the four-step sequence below is the only such place — has
// to freeze each one as it arrives. (Found the hard way: without this the
// accumulate probes fell silent on cpu and spoke on webgl, which is the worst
// possible way for a diagnosis to behave.)
async function frozen(value) {
  const rows = await toArr(value);
  const out = [];
  for (let y = 0; y < rows.length; y++) out.push(Array.from(rows[y]));
  return out;
}

function meanOfRegion(img, [x0, y0, x1, y1]) {
  let sum = 0;
  let n = 0;
  for (let y = y0; y < y1; y++) {
    for (let x = x0; x < x1; x++) {
      sum += img[y][x];
      n++;
    }
  }
  return sum / n;
}

function regionMeans(img) {
  const out = {};
  for (const name of REGION_NAMES) out[name] = meanOfRegion(img, REGIONS[name]);
  return out;
}

// How much does a pixel disagree with the one beside it? On a converged image
// that is the picture's own detail (~0.022); on a one-sample frame it is
// almost entirely noise (~0.16). It is the cheapest honest noise meter there
// is, and it needs no reference image.
function neighbourRms(img) {
  let sum = 0;
  let n = 0;
  for (let y = 0; y < 64; y++) {
    for (let x = 0; x + 1 < 64; x++) {
      const d = img[y][x] - img[y][x + 1];
      sum += d * d;
      n++;
    }
  }
  return Math.sqrt(sum / n);
}

// ---- near-miss diagnosis --------------------------------------------------
//
// A failing assert that reports only two numbers tells a learner nothing about
// WHICH slip produced them. A probe pairs the value one specific known mistake
// would produce with a sentence naming that mistake; diagnose() speaks only
// when the observed value matches a probe within the test's own tolerance AND
// the correct value does not — so a ray where two candidates coincide stays
// silent, as do observations that match probes disagreeing with each other. A
// wrong diagnosis is worse than none.
function diagnose(got, expected, eps, probes) {
  const hits = probes
    .filter(p => Math.abs(got - p[0]) <= eps && Math.abs(expected - p[0]) > eps)
    .map(p => p[1]);
  return hits.length && hits.every(m => m === hits[0]) ? hits[0] : null;
}

// Every candidate weighting of an accumulation buffer agrees with some other
// one at some particular step: while the previous buffer is still all zeros, an
// exponential blend and an n that counts one too high are the SAME number. So a
// probe has to predict the whole sequence of steps — every step, every sampled
// cell — before it is allowed to name anything.
function sequenceHint(steps, eps, probes) {
  const hits = probes
    .filter(([value]) => {
      let differs = false;
      for (const step of steps) {
        for (let y = 0; y < 64; y += 7) {
          for (let x = 0; x < 64; x += 5) {
            const c = value(step.before[y][x], step.sample[y][x], step.n);
            if (!(Math.abs(step.got[y][x] - c) <= eps)) return false;
            if (Math.abs(step.expected[y][x] - c) > eps) differs = true;
          }
        }
      }
      return differs;
    })
    .map(p => p[1]);
  return hits.length && hits.every(m => m === hits[0]) ? hits[0] : null;
}

// The same idea for a single buffer: a candidate has to predict every sampled
// cell before it may speak, because three different weightings of the same two
// numbers agree at particular cells all the time.
function gridHint(out, expected, eps, probes) {
  const hits = probes
    .filter(([value]) => {
      let differs = false;
      for (let y = 0; y < 64; y += 7) {
        for (let x = 0; x < 64; x += 5) {
          const c = value(y, x);
          if (!(out[y] && Math.abs(out[y][x] - c) <= eps)) return false;
          if (Math.abs(expected(y, x) - c) > eps) differs = true;
        }
      }
      return differs;
    })
    .map(p => p[1]);
  return hits.length && hits.every(m => m === hits[0]) ? hits[0] : null;
}

// And for the scatter: five region means at once. Three of the four wrong
// scatters land within a couple of hundredths of each other on the image as a
// whole, so demanding all five regions is what lets any of them speak.
function scatterHint(got, tol) {
  const hits = WRONG_SCATTERS.filter(
    v =>
      REGION_NAMES.every(n => Math.abs(got[n] - v.means[n]) <= tol[n]) &&
      REGION_NAMES.some(n => Math.abs(TRUTH[n] - v.means[n]) > tol[n])
  ).map(v => v.message);
  return hits.length && hits.every(m => m === hits[0]) ? hits[0] : null;
}

// Task 1: the three ways a seed can forget where it lives.
function sameEverywhere(out) {
  const first = out[0][0];
  for (let y = 0; y < 64; y++) {
    for (let x = 0; x < 64; x++) {
      if (out[y][x] !== first) return false;
    }
  }
  return true;
}

function rowsAreCopies(out) {
  for (let y = 1; y < 64; y++) {
    for (let x = 0; x < 64; x++) {
      if (out[y][x] !== out[0][x]) return false;
    }
  }
  return true;
}

function columnsAreCopies(out) {
  for (let y = 0; y < 64; y++) {
    for (let x = 1; x < 64; x++) {
      if (out[y][x] !== out[y][0]) return false;
    }
  }
  return true;
}

function fractionDifferent(a, b) {
  let n = 0;
  for (let y = 0; y < 64; y++) {
    for (let x = 0; x < 64; x++) {
      if (a[y][x] !== b[y][x]) n++;
    }
  }
  return n / 4096;
}

// Task 2: the roots of the quadratic that are NOT the answer.
function sphereProbes(o, d, c, r) {
  const parts = sphereParts(o, d, c, r);
  const probes = [
    [
      parts.dist,
      'that is the distance to the sphere\'s CENTRE, not to its surface — the quadratic is ' +
        'what turns one into the other',
    ],
    [
      -parts.b,
      'that is -b on its own: the distance to the point where the ray passes closest to the ' +
        'centre. The surface is √(b² − c) nearer than that.',
    ],
  ];
  if (parts.disc > 0) {
    probes.push([
      -parts.b + Math.sqrt(parts.disc),
      'that is the FAR root — where the ray leaves the sphere out the back. The one you can ' +
        'see is the near root, -b − Math.sqrt(disc).',
    ]);
    const t = -parts.b - Math.sqrt(parts.disc);
    if (t <= 0.001) {
      probes.push([
        t,
        'that root is at or behind the ray\'s origin: the intersection you found is in the ray\'s ' +
          'past, or is the surface the ray started on. Return -1 for anything at t <= 0.001.',
      ]);
    }
  }
  // Dropping the −r² from c leaves a different quadratic with a different root
  // — on the rays where that quadratic still has one.
  const bad = parts.b * parts.b - (parts.c + r * r);
  if (bad > 0) {
    probes.push([
      -parts.b - Math.sqrt(bad),
      'the radius never made it into c — it is c = e·e − r * r, and without the r * r the ' +
        'sphere has no size at all',
    ]);
  }
  return probes;
}

// A quadratic that has lost a piece usually has no real root at all, so what
// comes back is the perfectly ordinary -1 of a miss and no probe may speak for
// it. On a ray aimed straight through a sphere's centre, though, a miss is
// impossible, and that much IS diagnosable: the near root never got returned.
// Which piece went missing is not — an unwritten function, a flipped
// comparison, a sign, a stray 4 all arrive here as the same -1 — so this hands
// back the checklist rather than naming a culprit it cannot identify.
function centreMissHint(got, expected) {
  return expected > 0 && got < 0
    ? 'this ray goes straight through the middle of the sphere and still came back a miss, so ' +
        'the near root never got returned at all. Check the pieces one at a time: ' +
        'e = o - centre, b = e·d, c = e·e - r * r, disc = b * b - c (no 4 and no a — a unit ' +
        'direction makes both of those 1), and the answer is -b - Math.sqrt(disc).'
    : null;
}

// Task 5: the slope of a log–log fit, which is the whole point of the plot.
function logLogSlope(values) {
  let sx = 0;
  let sy = 0;
  let sxx = 0;
  let sxy = 0;
  const n = values.length;
  for (let i = 0; i < n; i++) {
    const lx = Math.log(i + 1);
    const ly = Math.log(values[i]);
    sx += lx;
    sy += ly;
    sxx += lx * lx;
    sxy += lx * ly;
  }
  return (n * sxy - sx * sy) / (n * sxx - sx * sx);
}

export default {
  uuid: 'c99efc67-071e-441a-91ac-62cb82009ca4',
  version: 1,
  slug: 'path-tracing',
  title: 'Progressive Path Tracing: Noise Melting Into an Image',
  blurb:
    'Every pixel fires its own random rays; a buffer that outlives the frame turns the static into a picture.',
  tasks: [
    // ------------------------------------------------------------------ 1
    {
      slug: 'per-thread-dice',
      title: 'Dice Every Thread Can Roll',
      intro: `<p>A path tracer needs random numbers by the million — every bounce picks a direction
        out of a hat. It cannot ask <code>Math.random()</code> for them: WebGPU refuses to compile a
        kernel containing it, and even where it compiles, thousands of threads drawing from
        <em>one</em> generator is a fiction, because a GPU has no shared state and no defined order
        in which the threads would take their turns. <strong>The Ising Model</strong> makes that
        argument in full and builds the generator from first principles; if you have not met it
        there, the short version is what follows.</p>
        <p>The fix is to make <strong>the thread's own coordinates the seed</strong>. Hash
        <code>this.thread.x</code>, <code>this.thread.y</code> and a frame counter into a starting
        state, then walk that state forward. Every thread gets a private stream, nothing is shared,
        and — because a hash is a pure function — the same run produces the same picture every
        single time. What is new <em>here</em> is the frame counter: Ising re-seeds per sweep to
        keep a simulation honest, and a path tracer re-seeds per frame so that each frame's noise is
        independent of the last, which is the only reason averaging them helps.</p>
        <p>The generator is written for you. Notice how small the numbers stay:</p>
<pre><code>gpu.addFunction(function nextRandom(state) {
  let p = state * 78.233 + 0.7213;
  p = p - Math.floor(p);   // wrap into 0…1
  p = p * (p + 61.7);
  return p - Math.floor(p);
});</code></pre>
        <p>That restraint is deliberate. A GPU float has 24 bits of mantissa, and taking the
        fractional part of a big number throws away exactly the bits the integer part is using — so
        the famous <code>fract(sin(x) * 43758.5453)</code> one-liner, whose intermediate reaches
        40,000, has about eight bits of randomness left by the time you see it. Keep every value you
        wrap under a hundred and you keep seventeen.</p>`,
      goal: `<strong>Goal:</strong> build this thread's seed out of <code>frame</code>,
        <code>this.thread.x</code> and <code>this.thread.y</code>, so that no two threads and no two
        frames ever share a stream.`,
      requirements: [
        'The seed must depend on all three of <code>frame</code>, <code>this.thread.x</code> and <code>this.thread.y</code>',
        'Run the mix through <code>nextRandom</code> — a raw sum of coordinates is not a random number',
        'No <code>Math.random</code>: pressing ▶ Run twice must produce the identical static',
      ],
      hints: [
        {
          title: 'Hint 1 — what is wrong with the starter',
          body: `<p>Run it. Every pixel is the same shade, because every thread computed the same
            seed from the same constant. A seed that does not mention <code>this.thread.x</code> is
            a seed 4,096 threads agree on.</p>`,
        },
        {
          title: 'Hint 2 — stir one coordinate at a time',
          body: `<p>Add a coordinate, hash, add the next, hash again. Each hash smears whatever
            went in across the whole 0…1 range, so two seeds that started 0.0713 apart end up with
            nothing in common:</p>
<pre><code>let seed = 0.1237 + frame * 0.0173;
seed = nextRandom(seed + this.thread.x * 0.0713);
seed = nextRandom(seed + this.thread.y * 0.0917);</code></pre>
<p>The constants are arbitrary; what matters is that all three numbers get in.</p>`,
        },
        {
          title: 'Hint 3 — why the frame belongs in there',
          body: `<p>Leave <code>frame</code> out and every frame draws the same numbers. The next
            tasks average frames together to kill noise — and averaging a value with a perfect copy
            of itself removes nothing at all.</p>`,
        },
      ],
      transfer: `Counter-based, per-thread generators are how every production GPU renderer does
        this: CUDA's cuRAND is initialised as <code>curand_init(seed, thread_id, 0, &amp;state)</code>,
        and WebGPU and Metal shaders hash <code>global_invocation_id</code> with a frame index using
        exactly this shape. It is the same tool <strong>The Ising Model</strong> reaches for, which
        is worth noticing: a renderer and a statistical-mechanics simulation have almost nothing in
        common except that both need a private stream per thread, and both get it the same way.`,
      starterCode: `// Static, on purpose: 4,096 threads each rolling their own die.
const gpu = new GPU({ mode });

${RNG_FN}

const dice = gpu.createKernel(function (frame) {
  // TODO: build THIS thread's seed. It has to depend on the frame, on
  // this.thread.x and on this.thread.y — otherwise threads share a stream.
  let seed = 0.1237;
  return nextRandom(seed);
}, { output: [64, 64] });

const paint = gpu.createKernel(function (values) {
  const v = values[this.thread.y][this.thread.x];
  this.color(v, v, v, 1);
}, { output: [64, 64], graphical: true });

// Four frames of static. Consecutive render() calls collapse into one
// scrubber, so you can drag between them and watch the dice re-roll.
for (let f = 0; f < 4; f++) {
  await paint(await dice(f));
  render(paint.canvas);
}

const first = await dice(0);
let mean = 0;
for (let y = 0; y < 64; y++) {
  for (let x = 0; x < 64; x++) mean += first[y][x];
}
console.log('mean of 4,096 draws:', mean / 4096, '— should sit near 0.5');
`,
      solutionCode: `// Static, on purpose: 4,096 threads each rolling their own die.
const gpu = new GPU({ mode });

${RNG_FN}

const dice = gpu.createKernel(function (frame) {
  // Start from the frame, stir in the column, then the row.
  let seed = 0.1237 + frame * 0.0173;
  seed = nextRandom(seed + this.thread.x * 0.0713);
  seed = nextRandom(seed + this.thread.y * 0.0917);
  return nextRandom(seed);
}, { output: [64, 64] });

const paint = gpu.createKernel(function (values) {
  const v = values[this.thread.y][this.thread.x];
  this.color(v, v, v, 1);
}, { output: [64, 64], graphical: true });

// Four frames of static. Consecutive render() calls collapse into one
// scrubber, so you can drag between them and watch the dice re-roll.
for (let f = 0; f < 4; f++) {
  await paint(await dice(f));
  render(paint.canvas);
}

const first = await dice(0);
let mean = 0;
for (let y = 0; y < 64; y++) {
  for (let x = 0; x < 64; x++) mean += first[y][x];
}
console.log('mean of 4,096 draws:', mean / 4096, '— should sit near 0.5');
`,
      publicTests: [
        {
          name: 'every thread draws its own number — no shared rows, columns or constants',
          run: async ctx => {
            ctx.assert(ctx.kernels.length >= 1, 'no kernel was created — call gpu.createKernel()');
            const dice = ctx.kernels.find(k => k.kernel && !k.kernel.graphical);
            ctx.assert(dice, 'no numeric kernel found — keep the dice kernel');
            const out = await dice(0);
            ctx.assert(out && out.length === 64, `expected a 64×64 grid, got ${out && out.length} rows`);
            ctx.assert(
              !sameEverywhere(out),
              'every one of the 4,096 threads returned the identical number — the seed never ' +
                'mentions this.thread.x or this.thread.y, so all of them computed the same one'
            );
            ctx.assert(
              !rowsAreCopies(out),
              'every row is an exact copy of row 0 — this.thread.y never reached the seed'
            );
            ctx.assert(
              !columnsAreCopies(out),
              'every column is an exact copy of column 0 — this.thread.x never reached the seed'
            );
          },
        },
        {
          name: 'the draws are uniform on <code>[0, 1)</code>',
          run: async ctx => {
            const dice = ctx.kernels.find(k => k.kernel && !k.kernel.graphical);
            const out = await dice(0);
            let mean = 0;
            const bins = new Array(8).fill(0);
            for (let y = 0; y < 64; y++) {
              for (let x = 0; x < 64; x++) {
                const v = out[y][x];
                ctx.assert(
                  v >= 0 && v < 1,
                  `draw at [${y}][${x}] is ${v} — nextRandom returns a fraction, so every value ` +
                    'belongs in 0 … 1'
                );
                mean += v;
                bins[Math.min(7, Math.floor(v * 8))]++;
              }
            }
            ctx.assertClose(mean / 4096, 0.5, 0.03, 'mean of 4,096 draws');
            // 4,096 draws in 8 bins: 512 expected, sigma 21 — ±35% is seven
            // sigmas, so a flat generator never trips it and a lumpy one does.
            for (let b = 0; b < 8; b++) {
              ctx.assert(
                bins[b] > 332 && bins[b] < 692,
                `bin ${b} of 8 holds ${bins[b]} of 4,096 draws (512 expected) — the draws are ` +
                  'not spread evenly over 0 … 1'
              );
            }
          },
        },
        {
          name: 'a new frame is a new roll',
          run: async ctx => {
            const dice = ctx.kernels.find(k => k.kernel && !k.kernel.graphical);
            const a = await dice(0);
            const b = await dice(7);
            const changed = fractionDifferent(a, b);
            ctx.assert(
              changed > 0.95,
              changed === 0
                ? 'frame 7 drew exactly the same 4,096 numbers as frame 0 — the frame never ' +
                  'reached the seed. Accumulating a frame with a perfect copy of itself cancels ' +
                  'no noise whatsoever, which is what the rest of this module is about.'
                : `only ${Math.round(changed * 100)}% of pixels changed between frame 0 and ` +
                  'frame 7 — the frame is barely reaching the seed'
            );
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            const dice = ctx.kernels.find(k => k.kernel && !k.kernel.graphical);
            // Same frame, twice: a hash is a pure function, so this has to be
            // identical to the last bit. Math.random would not survive it.
            const a = await dice(3);
            const b = await dice(3);
            ctx.assert(
              fractionDifferent(a, b) === 0,
              'two calls with the same frame gave different numbers — the stream has to be a ' +
                'pure function of (frame, x, y). Math.random() is not, and a run nobody can ' +
                'reproduce is a run nobody can debug.'
            );
            ctx.assert(
              fractionDifferent(await dice(3), await dice(4)) > 0.95,
              'frames 3 and 4 draw the same numbers — the frame is not reaching the seed'
            );
          },
        },
        {
          name: 'private test #2',
          run: async ctx => {
            const dice = ctx.kernels.find(k => k.kernel && !k.kernel.graphical);
            // Six frames pooled: 24,576 draws should sit very close to 0.5,
            // and neighbouring threads must not be correlated with each other.
            let mean = 0;
            let cov = 0;
            let varA = 0;
            let varB = 0;
            const pairs = [];
            for (let f = 0; f < 6; f++) {
              const out = await dice(f);
              for (let y = 0; y < 64; y++) {
                for (let x = 0; x < 64; x++) mean += out[y][x];
                for (let x = 0; x + 1 < 64; x++) pairs.push([out[y][x], out[y][x + 1]]);
              }
            }
            mean /= 6 * 4096;
            ctx.assertClose(mean, 0.5, 0.012, 'mean over 24,576 draws');
            const ma = pairs.reduce((s, p) => s + p[0], 0) / pairs.length;
            const mb = pairs.reduce((s, p) => s + p[1], 0) / pairs.length;
            for (const [a, b] of pairs) {
              cov += (a - ma) * (b - mb);
              varA += (a - ma) * (a - ma);
              varB += (b - mb) * (b - mb);
            }
            const r = cov / Math.sqrt(varA * varB);
            ctx.assert(
              Math.abs(r) < 0.05,
              `neighbouring threads' draws correlate at r = ${r.toFixed(3)} — side-by-side ` +
                'pixels are sharing a stream, which shows up as stripes in every image built ' +
                'on top of this'
            );
          },
        },
      ],
    },

    // ------------------------------------------------------------------ 2
    {
      slug: 'sphere-hit',
      title: 'Where the Ray Hits',
      intro: `<p>Before anything can bounce, a ray has to find a surface. The sibling module
        <em>Ray-Marched Metaballs</em> walks its rays forward in safe hops, because a signed
        distance field is all it has to go on. A sphere gives you something better: an exact answer,
        one square root long.</p>
        <p>Write the ray as <code>o + t·d</code> with <code>d</code> a <strong>unit</strong> vector,
        put <code>e = o − centre</code>, and asking "where does the ray meet the surface" is asking
        when <code>|e + t·d|² = r²</code>. Because <code>d</code> is unit, the <code>t²</code>
        coefficient is exactly 1 and the whole thing collapses to</p>
<pre><code>b = e · d
c = e · e - r * r
t = -b ± Math.sqrt(b * b - c)</code></pre>
        <p>Three cases, and all three matter. A negative discriminant means the ray misses. Two
        roots mean it goes in one side and out the other — the one you can <em>see</em> is the
        nearer, <code>−b − √(b² − c)</code>. And a root at or behind <code>t = 0</code> is in the
        ray's past: the sphere is behind the camera, and it must not be reported as a hit. Reject
        anything up to <code>t = 0.001</code> rather than just the negatives — the next task fires
        rays that start <em>on</em> a surface, and in float32 such a ray finds that surface again
        at a <code>t</code> a hair above zero.</p>`,
      goal: `<strong>Goal:</strong> finish <code>sphereT</code> so it returns the distance to the
        nearest visible hit, or <code>-1</code> when there isn't one.`,
      requirements: [
        'Compute <code>b = e·d</code> and <code>c = e·e − r * r</code> for <code>e = o − centre</code>',
        'Return <code>-1</code> when the discriminant <code>b * b - c</code> is not positive',
        'Otherwise return the NEAR root <code>-b - Math.sqrt(disc)</code> — and <code>-1</code> instead when that root is at or behind the ray\'s origin (<code>t &lt;= 0.001</code>)',
      ],
      hints: [
        {
          title: 'Hint 1 — the shape of the function',
          body: `<p>Three early exits and one answer:</p>
<pre><code>const disc = b * b - c;
if (disc &lt;= 0) {
  return -1;          // the ray misses this sphere entirely
}
const t = -b - Math.sqrt(disc);
if (t &lt;= 0.001) {
  return -1;          // the hit is behind us (or right on top of us)
}
return t;</code></pre>`,
        },
        {
          title: 'Hint 2 — why 0.001 and not 0',
          body: `<p>The next task fires rays that start <em>on</em> a surface. In float32 the
            starting point is never quite on it, and a root at <code>t = 0.000001</code> is that
            same surface hitting itself. The small epsilon is what stops a bounce from being
            immediately swallowed by the thing it bounced off.</p>`,
        },
        {
          title: 'Hint 3 — what the picture should look like',
          body: `<p>Nearer is brighter. You should get a pale sphere sitting on a floor that fades
            with distance, and the flat background colour everywhere the ray found nothing.</p>`,
        },
      ],
      transfer: `An RTX or RDNA ray-tracing core is silicon that does exactly this — traversal plus
        an intersection test — a few billion times a second, and the API shape around it (DXR's
        closest-hit shader, OptiX's <code>rtTrace</code>, Vulkan's ray query) is still one thread,
        one ray, one nearest hit. Analytic primitives never went away either: spheres, boxes and
        capsules still get intersected in closed form because closed form is faster than a mesh.`,
      starterCode: `// One ray per pixel, and the exact distance to the sphere it hits.
const gpu = new GPU({ mode });

// Nearest hit of a unit-direction ray against a sphere, or -1 for a miss.
gpu.addFunction(function sphereT(ox, oy, oz, dx, dy, dz, cx, cy, cz, r) {
  const ex = ox - cx;
  const ey = oy - cy;
  const ez = oz - cz;
  // TODO: b = e·d, c = e·e - r*r, disc = b*b - c.
  // Return -1 if disc <= 0, else the near root -b - Math.sqrt(disc),
  // and -1 again if that root is <= 0.001 (behind the ray).
  return -1;
});

gpu.addFunction(function planeT(oy, dy, planeY) {
  if (dy > -0.0001) {
    return -1;
  }
  const t = (planeY - oy) / dy;
  if (t <= 0.001) {
    return -1;
  }
  return t;
});

const depth = gpu.createKernel(function (cx, cy, cz, r) {
  // this pixel's camera ray
  const u = ((this.thread.x + 0.5) / 64) * 2 - 1;
  const v = ((this.thread.y + 0.5) / 64) * 2 - 1;
  let dx = u;
  let dy = v;
  let dz = -1.7;
  const len = Math.sqrt(dx * dx + dy * dy + dz * dz);
  dx = dx / len;
  dy = dy / len;
  dz = dz / len;

  const tS = sphereT(0, 0.3, 1.5, dx, dy, dz, cx, cy, cz, r);
  const tP = planeT(0.3, dy, -0.5);
  if (tS > 0 && (tP < 0 || tS < tP)) {
    return tS;
  }
  return tP;
}, { output: [64, 64] });

const paint = gpu.createKernel(function (t) {
  const d = t[this.thread.y][this.thread.x];
  if (d < 0) {
    this.color(0.10, 0.12, 0.17, 1);
  } else {
    const g = 1 - Math.min(Math.max((d - 1.2) / 1.6, 0), 1);
    this.color(g, g, g, 1);
  }
}, { output: [64, 64], graphical: true });

const ts = await depth(-0.45, 0, -0.4, 0.5);
await paint(ts);
render(paint.canvas);

let hits = 0;
let nearest = 999;
for (let y = 0; y < 64; y++) {
  for (let x = 0; x < 64; x++) {
    if (ts[y][x] > 0) {
      hits++;
      nearest = Math.min(nearest, ts[y][x]);
    }
  }
}
console.log(hits, 'of 4,096 rays hit something; nearest hit at t =', nearest);
`,
      solutionCode: `// One ray per pixel, and the exact distance to the sphere it hits.
const gpu = new GPU({ mode });

${SPHERE_FN}

gpu.addFunction(function planeT(oy, dy, planeY) {
  if (dy > -0.0001) {
    return -1;
  }
  const t = (planeY - oy) / dy;
  if (t <= 0.001) {
    return -1;
  }
  return t;
});

const depth = gpu.createKernel(function (cx, cy, cz, r) {
  // this pixel's camera ray
  const u = ((this.thread.x + 0.5) / 64) * 2 - 1;
  const v = ((this.thread.y + 0.5) / 64) * 2 - 1;
  let dx = u;
  let dy = v;
  let dz = -1.7;
  const len = Math.sqrt(dx * dx + dy * dy + dz * dz);
  dx = dx / len;
  dy = dy / len;
  dz = dz / len;

  const tS = sphereT(0, 0.3, 1.5, dx, dy, dz, cx, cy, cz, r);
  const tP = planeT(0.3, dy, -0.5);
  if (tS > 0 && (tP < 0 || tS < tP)) {
    return tS;
  }
  return tP;
}, { output: [64, 64] });

const paint = gpu.createKernel(function (t) {
  const d = t[this.thread.y][this.thread.x];
  if (d < 0) {
    this.color(0.10, 0.12, 0.17, 1);
  } else {
    const g = 1 - Math.min(Math.max((d - 1.2) / 1.6, 0), 1);
    this.color(g, g, g, 1);
  }
}, { output: [64, 64], graphical: true });

const ts = await depth(-0.45, 0, -0.4, 0.5);
await paint(ts);
render(paint.canvas);

let hits = 0;
let nearest = 999;
for (let y = 0; y < 64; y++) {
  for (let x = 0; x < 64; x++) {
    if (ts[y][x] > 0) {
      hits++;
      nearest = Math.min(nearest, ts[y][x]);
    }
  }
}
console.log(hits, 'of 4,096 rays hit something; nearest hit at t =', nearest);
`,
      publicTests: [
        {
          name: 'a sphere planted straight ahead is hit at the right distance',
          run: async ctx => {
            ctx.assert(ctx.kernels.length >= 1, 'no kernel was created — call gpu.createKernel()');
            const depth = ctx.kernels.find(k => k.kernel && !k.kernel.graphical);
            ctx.assert(depth, 'no numeric kernel found — keep the depth kernel');
            // Put a sphere of radius 0.5 exactly 2.0 along the centre pixel's
            // own ray: the near root is then exactly 1.5, whatever the camera.
            const d = rayFor(32, 32);
            const c = { x: CAM.x + 2 * d.dx, y: CAM.y + 2 * d.dy, z: CAM.z + 2 * d.dz };
            const out = await depth(c.x, c.y, c.z, 0.5);
            ctx.assert(out && out.length === 64, `expected a 64×64 grid, got ${out && out.length} rows`);
            const hint =
              centreMissHint(out[32][32], 1.5) ||
              diagnose(out[32][32], 1.5, 3e-3, sphereProbes(CAM, d, c, 0.5));
            ctx.assertClose(
              out[32][32],
              1.5,
              3e-3,
              hint || 'centre pixel, sphere of radius 0.5 at distance 2 along that very ray'
            );
          },
        },
        {
          name: 'misses, spheres behind the camera and roots at the origin all report <code>-1</code>',
          run: async ctx => {
            const depth = ctx.kernels.find(k => k.kernel && !k.kernel.graphical);
            // Behind: the same sphere, mirrored to the ray's past. Both roots
            // are negative, so the only right answer is "no hit".
            const d = rayFor(32, 32);
            const behind = { x: CAM.x - 2 * d.dx, y: CAM.y - 2 * d.dy, z: CAM.z - 2 * d.dz };
            const out = await depth(behind.x, behind.y, behind.z, 0.5);
            const hint = diagnose(out[32][32], -1, 3e-3, sphereProbes(CAM, d, behind, 0.5));
            ctx.assert(
              Math.abs(out[32][32] - (-1)) <= 3e-3,
              hint ||
                `the sphere is behind the camera, so the centre pixel should read -1, got ${out[32][32]}`
            );
            // Miss: far off to one side, nothing anywhere near the frustum.
            const away = await depth(-40, 20, -0.4, 0.5);
            ctx.assert(
              away[48][32] === -1,
              `a sphere 40 units off to the side was still reported as hit at t = ${away[48][32]} ` +
                '— a discriminant of zero or less means the ray misses'
            );
            // The other half of the same guard — and the half this kernel can
            // actually show you. A properly NEGATIVE root is invisible from
            // out here: `depth` drops anything ≤ 0 before it races the floor,
            // so a sphereT that forgot the check reads identically. A root
            // that is positive but tiny does not hide: park the sphere so its
            // surface is 0.0005 in front of the camera and the near root is
            // 0.0005 — inside the epsilon, so still no hit. That epsilon is
            // exactly what stops a bounce in the next task from being
            // swallowed by the surface it just left. Measured at 0.0005000 in
            // float64 and 0.0004999936 in float32: half the threshold on
            // every backend, and a thousand times clear of zero.
            const onIt = {
              x: CAM.x + 0.5005 * d.dx,
              y: CAM.y + 0.5005 * d.dy,
              z: CAM.z + 0.5005 * d.dz,
            };
            const grazing = await depth(onIt.x, onIt.y, onIt.z, 0.5);
            ctx.assert(
              grazing[32][32] < 0,
              `the sphere's surface sits 0.0005 in front of the camera, so the near root is ` +
                `0.0005 — at the origin, for all practical purposes — and the centre pixel ` +
                `should read -1. It read ${grazing[32][32]}. The guard has to reject anything ` +
                'with t <= 0.001, not merely anything negative: in float32 a ray that starts ' +
                'ON a surface finds that surface again at a t just above zero, and the next ' +
                'task fires exactly such rays by the thousand.'
            );
          },
        },
        {
          name: 'the whole depth image matches the quadratic, pixel by pixel',
          run: async ctx => {
            const depth = ctx.kernels.find(k => k.kernel && !k.kernel.graphical);
            const c = { x: -0.45, y: 0, z: -0.4 };
            const out = await depth(c.x, c.y, c.z, 0.5);
            let checked = 0;
            for (let y = 2; y < 64; y += 5) {
              for (let x = 2; x < 64; x += 5) {
                const d = rayFor(x, y);
                const parts = sphereParts(CAM, d, c, 0.5);
                // Skip the silhouette: where the discriminant is within a
                // whisker of zero, float32 and float64 legitimately disagree
                // about whether the ray grazed the sphere at all.
                if (Math.abs(parts.disc) < 0.02) continue;
                const expected = sphereTRef(CAM, d, c, 0.5);
                if (expected < 0) continue; // floor pixels are planeT's job
                const hint = diagnose(out[y][x], expected, 3e-3, sphereProbes(CAM, d, c, 0.5));
                ctx.assertClose(out[y][x], expected, 3e-3, hint || `pixel [${y}][${x}]`);
                checked++;
              }
            }
            ctx.assert(checked > 20, `only ${checked} sphere pixels were checked — that is a bug in the test`);
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            // A different sphere in a different place, so nothing about the
            // public cases can be hardcoded.
            const depth = ctx.kernels.find(k => k.kernel && !k.kernel.graphical);
            const c = { x: 0.35, y: 0.15, z: -0.9 };
            const r = 0.62;
            const out = await depth(c.x, c.y, c.z, r);
            let hitCount = 0;
            for (let y = 0; y < 64; y += 3) {
              for (let x = 0; x < 64; x += 3) {
                const d = rayFor(x, y);
                const parts = sphereParts(CAM, d, c, r);
                if (Math.abs(parts.disc) < 0.02) continue;
                const expected = sphereTRef(CAM, d, c, r);
                if (expected < 0) continue;
                hitCount++;
                const hint = diagnose(out[y][x], expected, 3e-3, sphereProbes(CAM, d, c, r));
                ctx.assertClose(out[y][x], expected, 3e-3, hint || `pixel [${y}][${x}]`);
              }
            }
            ctx.assert(hitCount > 30, `expected the sphere to cover more of the frame (got ${hitCount} samples)`);
          },
        },
        {
          name: 'private test #2',
          run: async ctx => {
            // The sphere has to WIN against the floor exactly where it should
            // and nowhere else: a far-root answer sits metres behind the true
            // surface, which both changes the value and loses races it should
            // have won. So check the whole selection, and count the silhouette.
            const depth = ctx.kernels.find(k => k.kernel && !k.kernel.graphical);
            const c = { x: -0.2, y: 0.25, z: -0.2 };
            const r = 0.45;
            const out = await depth(c.x, c.y, c.z, r);
            let sphereOwned = 0;
            let expectedOwned = 0;
            for (let y = 0; y < 64; y++) {
              for (let x = 0; x < 64; x++) {
                const d = rayFor(x, y);
                const parts = sphereParts(CAM, d, c, r);
                const tS = sphereTRef(CAM, d, c, r);
                const tP = d.dy > -0.0001 ? -1 : (-0.5 - CAM.y) / d.dy;
                const wins = tS > 0 && (tP < 0 || tS < tP);
                if (wins) expectedOwned++;
                if (out[y][x] > 0 && (tP < 0 || out[y][x] < tP - 1e-3)) sphereOwned++;
                if (Math.abs(parts.disc) < 0.02) continue; // the silhouette is a coin toss
                const expected = wins ? tS : tP;
                const hint = diagnose(out[y][x], expected, 3e-3, sphereProbes(CAM, d, c, r));
                ctx.assertClose(out[y][x], expected, 3e-3, hint || `pixel [${y}][${x}]`);
              }
            }
            ctx.assertClose(
              sphereOwned,
              expectedOwned,
              24,
              `the sphere covers ${sphereOwned} pixels where it should cover ${expectedOwned}`
            );
          },
        },
      ],
    },

    // ------------------------------------------------------------------ 3
    {
      slug: 'one-noisy-frame',
      title: 'One Path Per Pixel, and It Looks Terrible',
      intro: `<p>Here is the whole idea of path tracing. Fire a ray from the eye through a pixel.
        Where it lands, pick a <em>random</em> new direction and carry on. Multiply the surviving
        light by each surface's albedo as you go, and when the path finally escapes into the sky,
        that sky's brightness — scaled by everything it passed through — is your estimate of what
        the pixel should be.</p>
        <p>One estimate. It is <em>unbiased</em>: average enough of them and you get the exact
        answer, no approximation anywhere. It is also, on its own, nearly worthless — one path is a
        single sample of an integral over every direction light could possibly have arrived from,
        and it is wrong by tens of percent.</p>
        <p>The bounce itself is one line of geometry. Take a uniformly random point on the unit
        sphere, add it to the surface normal, and normalize: that gives you a
        <strong>cosine-weighted</strong> direction on the hemisphere — more likely to leave
        straight up than sideways, which is exactly how a matte surface actually scatters. Because
        the sampling density already matches the physics, no correction factor is needed; the
        throughput just picks up the albedo and moves on.</p>`,
      goal: `<strong>Goal:</strong> finish the bounce — multiply the throughput by the surface's
        <code>albedo</code>, then send the ray off along <code>normal + random unit vector</code>,
        normalized.`,
      requirements: [
        'Multiply <code>through</code> by <code>albedo</code>, once per bounce',
        'Set the new direction to <code>nx + sx</code>, <code>ny + sy</code>, <code>nz + sz</code>',
        'Normalize it — <code>sphereT</code> and <code>planeT</code> both assume a unit direction',
      ],
      hints: [
        {
          title: 'Hint 1 — why add the normal at all',
          body: `<p><code>(sx, sy, sz)</code> is a random point on the whole sphere, so half of
            those directions point <em>into</em> the surface. Adding the unit normal shifts the
            sphere so it sits tangent to the surface: every direction then points outward, and the
            density of directions comes out proportional to the cosine of the angle from the
            normal — which is Lambert's law, for free.</p>`,
        },
        {
          title: 'Hint 2 — the six lines',
          body: `<pre><code>through = through * albedo;
dx = nx + sx;
dy = ny + sy;
dz = nz + sz;
len = Math.sqrt(dx * dx + dy * dy + dz * dz);
dx = dx / len;
dy = dy / len;
dz = dz / len;</code></pre>`,
        },
        {
          title: 'Hint 3 — what "right" looks like here',
          body: `<p>Grainy. Every pixel should be visibly wrong by itself, with soft shadows and
            the contact between the spheres and the floor only <em>hinted</em> at through the
            noise. If your image comes out smooth, the bounce is not random and the next task has
            nothing to average.</p>`,
        },
      ],
      transfer: `This loop — generate a ray, find the closest hit, sample a new direction, multiply
        the throughput, repeat — is the core of every production renderer, from Cycles and Arnold to
        the DXR sample code. The kernels get bigger (materials, textures, lights sampled directly)
        but the shape is this, and the reason it is on a GPU is the reason it is here: every pixel's
        path is independent of every other pixel's.`,
      starterCode: `// A whole path tracer. One sample per pixel, one frame — and it shows.
const gpu = new GPU({ mode });

${RNG_FN}

${SPHERE_FN}

${REST_OF_SCENE}

${traceKernel(SCATTER_TODO)}

${PAINT_KERNEL}

const frame = await trace(0, 1);
await paint(frame);
render(paint.canvas);

let mean = 0;
for (let y = 0; y < 64; y++) {
  for (let x = 0; x < 64; x++) mean += frame[y][x];
}
console.log('mean radiance over the frame:', mean / 4096);
`,
      solutionCode: `// A whole path tracer. One sample per pixel, one frame — and it shows.
const gpu = new GPU({ mode });

${RNG_FN}

${SPHERE_FN}

${REST_OF_SCENE}

${traceKernel(SCATTER_SOLVED)}

${PAINT_KERNEL}

const frame = await trace(0, 1);
await paint(frame);
render(paint.canvas);

let mean = 0;
for (let y = 0; y < 64; y++) {
  for (let x = 0; x < 64; x++) mean += frame[y][x];
}
console.log('mean radiance over the frame:', mean / 4096);
`,
      publicTests: [
        {
          name: 'the frame is a plausible image: finite, non-negative, and painted',
          run: async ctx => {
            ctx.assert(ctx.kernels.length >= 2, `expected 2 kernels, found ${ctx.kernels.length}`);
            const trace = ctx.kernels.find(k => k.kernel && !k.kernel.graphical);
            ctx.assert(trace, 'no numeric kernel found — keep the trace kernel');
            const out = await trace(0, 1);
            ctx.assert(out && out.length === 64, `expected a 64×64 grid, got ${out && out.length} rows`);
            for (let y = 0; y < 64; y += 3) {
              for (let x = 0; x < 64; x += 3) {
                const v = out[y][x];
                ctx.assert(
                  Number.isFinite(v),
                  `pixel [${y}][${x}] is ${v} — a non-finite radiance usually means a direction ` +
                    'was divided by a length of zero'
                );
                ctx.assert(
                  v >= 0 && v <= 1.4,
                  `pixel [${y}][${x}] is ${v.toFixed(3)} — the sky tops out at 1.0 and every ` +
                    'surface only ever absorbs, so nothing can exceed it'
                );
              }
            }
            ctx.assert(ctx.canvas, 'no canvas — did you call render(paint.canvas)?');
            ctx.assert(
              ctx.canvas.width === 64 && ctx.canvas.height === 64,
              `expected a 64×64 canvas, got ${ctx.canvas.width}×${ctx.canvas.height}`
            );
          },
        },
        {
          name: 'the estimate is unbiased — four frames already land on the right picture',
          run: async ctx => {
            const trace = ctx.kernels.find(k => k.kernel && !k.kernel.graphical);
            const acc = [];
            for (let y = 0; y < 64; y++) acc.push(new Array(64).fill(0));
            for (let f = 0; f < 4; f++) {
              const out = await trace(f, 1);
              for (let y = 0; y < 64; y++) {
                for (let x = 0; x < 64; x++) acc[y][x] += out[y][x] / 4;
              }
            }
            const got = regionMeans(acc);
            const hint = scatterHint(got, TOL_4);
            for (const name of REGION_NAMES) {
              ctx.assertClose(
                got[name],
                TRUTH[name],
                TOL_4[name],
                hint || `mean radiance over the ${name} region`
              );
            }
          },
        },
        {
          name: 'and it is loud — a single frame is mostly noise',
          run: async ctx => {
            const trace = ctx.kernels.find(k => k.kernel && !k.kernel.graphical);
            const out = await trace(1, 1);
            const rms = neighbourRms(out);
            // The converged picture's own detail measures about 0.022; one
            // sample per pixel measures about 0.16. Anything under 0.09 is a
            // frame with no randomness in it at all.
            ctx.assert(
              rms > 0.09,
              `neighbouring pixels disagree by only ${rms.toFixed(3)} — this frame is smooth, ` +
                'which means the bounce direction is not actually random. A path tracer buys its ' +
                'correctness with noise; a frame with no noise is not sampling anything.'
            );
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            // Twelve unseen frames: the tolerances shrink with √12, so a
            // scatter that is subtly wrong has nowhere left to hide.
            const trace = ctx.kernels.find(k => k.kernel && !k.kernel.graphical);
            const acc = [];
            for (let y = 0; y < 64; y++) acc.push(new Array(64).fill(0));
            for (let f = 20; f < 32; f++) {
              const out = await trace(f, 1);
              for (let y = 0; y < 64; y++) {
                for (let x = 0; x < 64; x++) acc[y][x] += out[y][x] / 12;
              }
            }
            const got = regionMeans(acc);
            const hint = scatterHint(got, TOL_12);
            for (const name of REGION_NAMES) {
              ctx.assertClose(
                got[name],
                TRUTH[name],
                TOL_12[name],
                hint || `mean radiance over the ${name} region, on unseen frames`
              );
            }
            ctx.assert(
              got.ballA > got.ballB + 0.2,
              `the pale sphere (${got.ballA.toFixed(3)}) should be far brighter than the dark ` +
                `one (${got.ballB.toFixed(3)}) — their albedos are 0.85 and 0.3`
            );
          },
        },
        {
          name: 'private test #2',
          run: async ctx => {
            // Four samples inside ONE frame must be worth the same as four
            // frames of one sample — the spp loop has to keep drawing new
            // numbers rather than repeating the first path four times.
            const trace = ctx.kernels.find(k => k.kernel && !k.kernel.graphical);
            const one = await trace(5, 1);
            const four = await trace(5, 4);
            const rms1 = neighbourRms(one);
            const rms4 = neighbourRms(four);
            ctx.assert(
              rms4 < rms1 * 0.75,
              `four samples per frame are no quieter than one (${rms4.toFixed(3)} against ` +
                `${rms1.toFixed(3)}) — the extra samples are drawing the same path over again`
            );
            const got = regionMeans(four);
            const hint = scatterHint(got, TOL_12);
            for (const name of REGION_NAMES) {
              ctx.assertClose(got[name], TRUTH[name], TOL_4[name], hint || `${name} region at 4 spp`);
            }
          },
        },
      ],
    },

    // ------------------------------------------------------------------ 4
    {
      slug: 'accumulate',
      title: 'The Buffer That Outlives the Frame',
      // Measured, not assumed. In auto this chain comes out mixed — the three
      // numeric kernels upgrade to WebGPU and the graphical paint cannot, so
      // the accumulation buffer crosses backends once per frame, 48 times.
      // Median of three runs in the browser: 70.9 ms mixed against 24.9 ms
      // pinned to WebGL. The readback the pin removes is precisely the one
      // this task exists to teach away, so the lesson and the number agree.
      backend: 'webgl',
      intro: `<p>Nothing in this scene moves. So every frame you trace is another independent
        estimate of the <em>same</em> integral — and the way to spend them is not to throw the last
        one away, but to keep a running average of all of them in a buffer that survives from frame
        to frame. Frame 48 is not 48 times more work per frame; it is the same work, added to what
        you already had.</p>
        <p>The average has to be exact, not exponential. <em>Video Filters</em> blends each new
        frame in with a fixed <code>alpha</code> because its scene is moving and old frames go
        stale; here nothing goes stale, so every frame gets an equal vote:</p>
<pre><code>next = (previous * n + sample) / (n + 1)</code></pre>
        <p>And the buffer lives on the card. This is the ping-pong from <em>Pipelines &amp;
        Textures</em> doing real work: the kernel reads the very texture it is about to replace, so
        <code>immutable: true</code> is what makes the loop legal — each call renders into a
        <em>fresh</em> texture and last frame's numbers stay safe to read while this frame's are
        being written. Leave it off and gpu.js stops you with the reason. (Auto runs this one on
        WebGL throughout, deliberately: a graphical kernel can never be a WebGPU kernel, so
        letting the maths upgrade would hand the buffer across backends once a frame — measured
        here at 71 ms against 25 ms for the whole 48-frame loop.)</p>
        <p>Then render every pass. Consecutive <code>render()</code> calls collapse into a
        scrubber, so you get to drag the noise away by hand.</p>`,
      goal: `<strong>Goal:</strong> make <code>accumulate</code> a running mean, and make the
        feedback loop legal, so that 48 traced frames melt into one clean image.`,
      requirements: [
        'Return <code>(old * n + now) / (n + 1)</code> — an exact mean of every frame so far, not a blend',
        'Add <code>immutable: true</code> to <code>accumulate</code> (keep <code>pipeline: true</code>)',
        'Call <code>render(paint.canvas)</code> <em>inside</em> the loop, once per frame, for the scrubber',
      ],
      hints: [
        {
          title: 'Hint 1 — run the starter and read the error',
          body: `<p>gpu.js refuses the second pass of the loop: the kernel's input is its own
            output storage. WebGL names the fix in the message; WebGPU only tells you the buffer
            belongs to the kernel. Either way <code>immutable: true</code> is it — a fresh output
            texture per call, so reading last frame's is safe.</p>`,
        },
        {
          title: 'Hint 2 — where the n comes from',
          body: `<p><code>n</code> is how many frames are <em>already</em> in the buffer, which is
            the loop counter. At <code>n = 0</code> the formula returns the new sample untouched,
            which is exactly right: the zero buffer must not get a vote.</p>`,
        },
        {
          title: 'Hint 3 — the two edits',
          body: `<p>In the kernel body:</p>
<pre><code>return (old * n + now) / (n + 1);</code></pre>
<p>and in its settings, beside <code>pipeline: true</code>:</p>
<pre><code>immutable: true,</code></pre>`,
        },
      ],
      transfer: `Every "progressive" viewport you have ever watched resolve — Cycles, Arnold's IPR,
        a Substance or Unreal path-traced preview — is this loop: a persistent accumulation buffer
        plus a sample count. On the GPU it is always two textures being ping-ponged, because a
        shader may not read the surface it writes; WebGPU makes you bind two storage textures and
        swap them, CUDA makes you swap two device pointers, and <code>immutable: true</code> is
        gpu.js doing that bookkeeping for you. Real-time renderers do the same trick within a
        <em>moving</em> scene by reprojecting the history with motion vectors — that is what the
        "TA" in TAA stands for.`,
      starterCode: `const gpu = new GPU({ mode });

${RNG_FN}

${SPHERE_FN}

${REST_OF_SCENE}

${traceKernel(SCATTER_SOLVED, ', pipeline: true')}

${BLACK_KERNEL}

const accumulate = gpu.createKernel(function (previous, sample, n) {
  const old = previous[this.thread.y][this.thread.x];
  const now = sample[this.thread.y][this.thread.x];
  // TODO: the running mean of every frame so far — (old * n + now) / (n + 1).
  return now;
}, {
  output: [64, 64],
  pipeline: true,
  // TODO: this kernel reads its own previous output. Run it as-is and let
  // gpu.js tell you the setting it wants.
});

${PAINT_KERNEL}

// A real dial: moving it re-runs the whole program.
const spp = slider('samples per frame', { min: 1, max: 4, value: 2, step: 1 });
const FRAMES = 48;

let acc = await black();
for (let f = 0; f < FRAMES; f++) {
  const sample = await trace(f, spp);
  acc = await accumulate(acc, sample, f);
  await paint(acc);
  render(paint.canvas);
}

console.log('total samples per pixel:', FRAMES * spp);
`,
      solutionCode: `const gpu = new GPU({ mode });

${RNG_FN}

${SPHERE_FN}

${REST_OF_SCENE}

${traceKernel(SCATTER_SOLVED, ', pipeline: true')}

${BLACK_KERNEL}

const accumulate = gpu.createKernel(function (previous, sample, n) {
  const old = previous[this.thread.y][this.thread.x];
  const now = sample[this.thread.y][this.thread.x];
  return (old * n + now) / (n + 1);
}, {
  output: [64, 64],
  pipeline: true,
  immutable: true, // fresh texture per call — reading last frame's is now safe
});

${PAINT_KERNEL}

// A real dial: moving it re-runs the whole program.
const spp = slider('samples per frame', { min: 1, max: 4, value: 2, step: 1 });
const FRAMES = 48;

let acc = await black();
for (let f = 0; f < FRAMES; f++) {
  const sample = await trace(f, spp);
  acc = await accumulate(acc, sample, f);
  await paint(acc);
  render(paint.canvas);
}

console.log('total samples per pixel:', FRAMES * spp);
`,
      publicTests: [
        {
          name: 'the accumulator ping-pongs: <code>pipeline</code> plus <code>immutable</code>',
          run: async ctx => {
            ctx.assert(ctx.kernels.length >= 4, `expected 4 kernels, found ${ctx.kernels.length}`);
            const acc = ctx.kernels.find(k => k.kernel && k.kernel.immutable);
            ctx.assert(
              acc,
              'no immutable kernel found — accumulate reads the texture it is about to replace, ' +
                'and immutable: true is what makes that legal'
            );
            ctx.assert(acc.kernel.pipeline === true, 'accumulate should keep pipeline: true as well');
            const paint = ctx.kernels.find(k => k.kernel && k.kernel.graphical);
            ctx.assert(paint, 'no graphical kernel found — keep paint');
            if (ctx.resolvedMode !== 'cpu') {
              ctx.assert(
                acc.lastArgs && acc.lastArgs[0] && typeof acc.lastArgs[0].toArray === 'function',
                'accumulate should be handed the previous texture itself — no .toArray() in the loop'
              );
            }
          },
        },
        {
          name: 'every pass is rendered, so the frames become a scrubber',
          run: async ctx => {
            const frames = ctx.logs.filter(l => l.type === 'canvas').length;
            ctx.assert(
              frames >= 40,
              `only ${frames} frame(s) were rendered — call render(paint.canvas) inside the loop, ` +
                'once per pass. Consecutive renders collapse into one scrubber, which is the ' +
                'whole payoff: you can drag from static to picture.'
            );
          },
        },
        {
          name: 'the running mean is exact at every <code>n</code>',
          run: async ctx => {
            const trace = ctx.kernels.find(
              k => k.kernel && !k.kernel.graphical && !k.kernel.immutable && k.lastArgs && k.lastArgs.length === 2
            );
            const black = ctx.kernels.find(
              k => k.kernel && !k.kernel.graphical && !k.kernel.immutable && k.lastArgs && k.lastArgs.length === 0
            );
            const accumulate = ctx.kernels.find(k => k.kernel && k.kernel.immutable);
            ctx.assert(trace && black && accumulate, 'expected the trace, black and accumulate kernels');

            // Walk four steps first, THEN diagnose. Each step is judged
            // against the buffer the kernel ACTUALLY held going in — a wrong
            // formula's buffer diverges from the right one immediately, and
            // comparing against the ideal history would make every candidate
            // miss from step two onwards.
            let acc = await black();
            let prev = [];
            for (let y = 0; y < 64; y++) prev.push(new Array(64).fill(0));
            const steps = [];
            for (let n = 0; n < 4; n++) {
              const sample = await trace(n + 60, 1);
              // frozen(), not toArr(): four steps are held at once, and on the
              // CPU backend trace's output array is reused by its next call.
              const s = await frozen(sample);
              acc = await accumulate(acc, sample, n);
              const got = await frozen(acc);
              const expected = [];
              for (let y = 0; y < 64; y++) {
                const row = new Array(64);
                for (let x = 0; x < 64; x++) row[x] = (prev[y][x] * n + s[y][x]) / (n + 1);
                expected.push(row);
              }
              steps.push({ n, before: prev, sample: s, got, expected });
              prev = got;
            }
            const hint = sequenceHint(steps, 2e-3, [
              [(old, now) => now, 'the buffer is just the newest frame — the previous average never entered the answer'],
              [
                (old, now) => (old + now) / 2,
                'that is a half-and-half blend: an exponential moving average, which is what you ' +
                  'want when old frames go stale. Nothing in this scene moves, so every frame ' +
                  'deserves an equal vote — weight the old average by n, not by 1/2.',
              ],
              [(old, now) => old + now, 'the frames are being summed and never divided — this is a total, not a mean'],
              [
                (old, now, n) => (old * n + now) / Math.max(n, 1),
                'the divisor is off by one: after folding in the new sample there are n + 1 ' +
                  'frames in the buffer, not n',
              ],
              [
                (old, now, n) => (old * (n + 1) + now) / (n + 2),
                'n is one too large — it counts the frames ALREADY in the buffer, so it starts at 0',
              ],
            ]);
            for (const step of steps) {
              for (let y = 0; y < 64; y += 7) {
                for (let x = 0; x < 64; x += 5) {
                  ctx.assertClose(
                    step.got[y][x],
                    step.expected[y][x],
                    2e-3,
                    hint || `cell [${y}][${x}] after ${step.n + 1} frame(s)`
                  );
                }
              }
            }
          },
        },
        {
          name: 'sixteen accumulated frames are far quieter than one, and still the same picture',
          run: async ctx => {
            const trace = ctx.kernels.find(
              k => k.kernel && !k.kernel.graphical && !k.kernel.immutable && k.lastArgs && k.lastArgs.length === 2
            );
            const black = ctx.kernels.find(
              k => k.kernel && !k.kernel.graphical && !k.kernel.immutable && k.lastArgs && k.lastArgs.length === 0
            );
            const accumulate = ctx.kernels.find(k => k.kernel && k.kernel.immutable);
            ctx.assert(trace && black && accumulate, 'expected the trace, black and accumulate kernels');
            const single = neighbourRms(await toArr(await trace(0, 2)));
            let acc = await black();
            for (let f = 0; f < 16; f++) acc = await accumulate(acc, await trace(f, 2), f);
            const img = await toArr(acc);
            const many = neighbourRms(img);
            ctx.assert(
              many < single * 0.5,
              `accumulating 16 frames left the noise at ${many.toFixed(3)}, against ` +
                `${single.toFixed(3)} for a single frame — averaging n frames should divide it ` +
                'by about √n'
            );
            const got = regionMeans(img);
            for (const name of REGION_NAMES) {
              ctx.assertClose(got[name], TRUTH[name], TOL_ACC[name], `mean radiance over the ${name} region`);
            }
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            // Unseen frames, and twice as many of them: the accumulated image
            // has to keep converging rather than settle on something close.
            const trace = ctx.kernels.find(
              k => k.kernel && !k.kernel.graphical && !k.kernel.immutable && k.lastArgs && k.lastArgs.length === 2
            );
            const black = ctx.kernels.find(
              k => k.kernel && !k.kernel.graphical && !k.kernel.immutable && k.lastArgs && k.lastArgs.length === 0
            );
            const accumulate = ctx.kernels.find(k => k.kernel && k.kernel.immutable);
            ctx.assert(trace && black && accumulate, 'expected the trace, black and accumulate kernels');
            let acc = await black();
            for (let f = 200; f < 232; f++) acc = await accumulate(acc, await trace(f, 2), f - 200);
            const img = await toArr(acc);
            const got = regionMeans(img);
            for (const name of REGION_NAMES) {
              ctx.assertClose(got[name], TRUTH[name], TOL_ACC[name], `${name} region after 32 unseen frames`);
            }
            ctx.assert(
              neighbourRms(img) < 0.05,
              `the accumulated image still measures ${neighbourRms(img).toFixed(3)} between ` +
                'neighbouring pixels; the converged picture measures about 0.022'
            );
          },
        },
        {
          name: 'private test #2',
          run: async ctx => {
            // The slider is the program's only input, so its declared default
            // is what every reader of this task will actually see.
            const control = (ctx.controls || []).find(c => /sample/i.test(c.name));
            ctx.assert(
              control,
              'no samples-per-frame slider was declared — slider() is what puts the control under ' +
                'the console and re-runs the program when it moves'
            );
            ctx.assert(
              control.min === 1 && control.max === 4 && control.step === 1,
              `the slider should run 1…4 in steps of 1, got ${control.min}…${control.max} step ${control.step}`
            );
            const logged = ctx.logs.some(l => l.type === 'log' && l.text && l.text.includes('96'));
            ctx.assert(
              logged,
              'expected the total sample count (48 frames × 2 samples = 96) in the console — ' +
                'that product, not the frame count, is what sets the noise'
            );
          },
        },
      ],
    },

    // ------------------------------------------------------------------ 5
    {
      slug: 'watch-it-fall',
      title: 'The Noise Falls Like 1/√n',
      intro: `<p>You have watched the noise go. Now measure it, because the rate it goes at is the
        single most important number in rendering — and it is the same <code>1/√n</code> that
        <em>Monte Carlo Methods</em> derives for a dart-throwing estimate of π. A path tracer is
        that estimator with a much more interesting integrand.</p>
        <p>Measuring it needs a reference, and the honest trick is to build one out of thin air:
        run the accumulation <strong>twice</strong>, with two different slices of the random stream.
        Neither run is the truth, but their errors are independent, so the RMS gap between them is
        √2 times either one's own error. Halve the square and you have measured your own noise with
        no ground truth anywhere in sight.</p>
        <p>Plot that against <code>errs[0] / √n</code> on a log axis. If the two curves lie on top
        of each other, the estimator is behaving exactly as theory says — and the shape of them is
        the bad news in the theory: halving the noise costs <em>four times</em> the samples. That
        single fact is why production renderers spend their effort on importance sampling and
        denoisers instead of simply waiting longer.</p>
        <p>The <code>samples</code> dial is that trade in your hand. Drag it from 24 to 96 and the
        whole run happens again with four times the paths per pixel: both curves grow, the picture
        quietens, and the noise at the far end lands on half what 24 frames managed. Four times the
        work for one halving — and this is the cheap end of the curve, where the frames are still
        making a visible difference.</p>`,
      goal: `<strong>Goal:</strong> finish <code>sqDiff</code> — half the squared difference between
        the two runs at this pixel — and build the <code>1/√n</code> reference curve to plot
        against it.`,
      requirements: [
        'In <code>sqDiff</code>, return <code>0.5 * d * d</code> for <code>d = a - b</code> at this pixel',
        'Push <code>errs[0] / Math.sqrt(f + 1)</code> into <code>ideal</code> each pass',
        'Both curves go to <code>plot</code> on a log axis (already wired)',
      ],
      hints: [
        {
          title: 'Hint 1 — where the half comes from',
          body: `<p>Run A is off by <code>eₐ</code> and run B by <code>e_b</code>, independently.
            The gap between them has variance <code>var(eₐ) + var(e_b)</code> = twice one run's.
            So the square of the gap, halved, estimates the square of one run's error — and the
            square root of the mean of that is the RMS error you want.</p>`,
        },
        {
          title: 'Hint 2 — the kernel',
          body: `<pre><code>const d = a[this.thread.y][this.thread.x] - b[this.thread.y][this.thread.x];
return 0.5 * d * d;</code></pre>
<p>Squaring in the kernel and square-rooting once in JavaScript is the usual split: the
            per-pixel work is parallel, the final scalar is not.</p>`,
        },
        {
          title: 'Hint 3 — the reference curve',
          body: `<p>Anchor it to the measurement you already have, so the two lines start together
            and any drift between them is real:</p>
<pre><code>ideal.push(errs[0] / Math.sqrt(f + 1));</code></pre>`,
        },
      ],
      transfer: `√n is why every serious renderer stopped brute-forcing: importance sampling,
        next-event estimation and multiple importance sampling all attack the <em>constant</em> in
        front of the <code>1/√n</code> because the exponent itself cannot be moved; low-discrepancy
        (quasi-Monte Carlo) sequences buy a slightly better exponent on smooth integrands; and
        OptiX's and OIDN's neural denoisers give up unbiasedness altogether to get the picture
        twenty times sooner. The measurement you just made — two independent runs, no reference
        image — is also how those denoisers are evaluated in practice.`,
      starterCode: `const gpu = new GPU({ mode });

${RNG_FN}

${SPHERE_FN}

${REST_OF_SCENE}

${traceKernel(SCATTER_SOLVED, ', pipeline: true')}

${BLACK_KERNEL}

const accumulate = gpu.createKernel(function (previous, sample, n) {
  const old = previous[this.thread.y][this.thread.x];
  const now = sample[this.thread.y][this.thread.x];
  return (old * n + now) / (n + 1);
}, { output: [64, 64], pipeline: true, immutable: true });

const sqDiff = gpu.createKernel(function (a, b) {
  // TODO: half the squared difference between the two runs at this pixel.
  // Half, because two independent runs disagree by √2 times either one's
  // own error.
  return 0;
}, { output: [64, 64] });

${PAINT_KERNEL}

// A dial, not a constant: slider() re-runs the whole program when you drag it,
// so this is the sample budget, and 24 → 96 is exactly the four times theory
// says buys you half the noise. The plot is where you check that it does.
const FRAMES = slider('samples', { min: 24, max: 96, value: 24, step: 4 });
let accA = await black();
let accB = await black();
const errs = [];
const ideal = [];

for (let f = 0; f < FRAMES; f++) {
  // The same accumulation twice, on two different slices of the stream.
  accA = await accumulate(accA, await trace(f, 1), f);
  accB = await accumulate(accB, await trace(f + 500, 1), f);

  const d = await sqDiff(accA, accB);
  let sum = 0;
  for (let y = 0; y < 64; y++) {
    for (let x = 0; x < 64; x++) sum += d[y][x];
  }
  errs.push(Math.sqrt(sum / 4096));
  // TODO: the theory line — errs[0] / Math.sqrt(f + 1)
  ideal.push(errs[0]);
}

await paint(accA);
render(paint.canvas);
console.log('RMS noise per frame:', errs);
plot({ measured: errs, 'errs[0] / √n': ideal }, {
  title: 'RMS noise vs frames accumulated',
  log: true,
});
`,
      solutionCode: `const gpu = new GPU({ mode });

${RNG_FN}

${SPHERE_FN}

${REST_OF_SCENE}

${traceKernel(SCATTER_SOLVED, ', pipeline: true')}

${BLACK_KERNEL}

const accumulate = gpu.createKernel(function (previous, sample, n) {
  const old = previous[this.thread.y][this.thread.x];
  const now = sample[this.thread.y][this.thread.x];
  return (old * n + now) / (n + 1);
}, { output: [64, 64], pipeline: true, immutable: true });

const sqDiff = gpu.createKernel(function (a, b) {
  const d = a[this.thread.y][this.thread.x] - b[this.thread.y][this.thread.x];
  return 0.5 * d * d;
}, { output: [64, 64] });

${PAINT_KERNEL}

// A dial, not a constant: slider() re-runs the whole program when you drag it,
// so this is the sample budget, and 24 → 96 is exactly the four times theory
// says buys you half the noise. The plot is where you check that it does.
const FRAMES = slider('samples', { min: 24, max: 96, value: 24, step: 4 });
let accA = await black();
let accB = await black();
const errs = [];
const ideal = [];

for (let f = 0; f < FRAMES; f++) {
  // The same accumulation twice, on two different slices of the stream.
  accA = await accumulate(accA, await trace(f, 1), f);
  accB = await accumulate(accB, await trace(f + 500, 1), f);

  const d = await sqDiff(accA, accB);
  let sum = 0;
  for (let y = 0; y < 64; y++) {
    for (let x = 0; x < 64; x++) sum += d[y][x];
  }
  errs.push(Math.sqrt(sum / 4096));
  ideal.push(errs[0] / Math.sqrt(f + 1));
}

await paint(accA);
render(paint.canvas);
console.log('RMS noise per frame:', errs);
plot({ measured: errs, 'errs[0] / √n': ideal }, {
  title: 'RMS noise vs frames accumulated',
  log: true,
});
`,
      publicTests: [
        {
          name: '<code>sqDiff</code> is half the squared gap, cell by cell',
          run: async ctx => {
            const sqDiff = ctx.kernels.find(
              k => k.kernel && !k.kernel.graphical && !k.kernel.pipeline && k.lastArgs && k.lastArgs.length === 2
            );
            ctx.assert(sqDiff, 'no plain two-argument kernel found — keep sqDiff');
            const trace = ctx.kernels.find(
              k => k.kernel && k.kernel.pipeline && !k.kernel.immutable && k.lastArgs && k.lastArgs.length === 2
            );
            const black = ctx.kernels.find(
              k => k.kernel && k.kernel.pipeline && !k.kernel.immutable && k.lastArgs && k.lastArgs.length === 0
            );
            const accumulate = ctx.kernels.find(k => k.kernel && k.kernel.immutable);
            ctx.assert(trace && black && accumulate, 'expected the trace, black and accumulate kernels');

            // Two short, genuinely different accumulations to compare.
            let a = await black();
            let b = await black();
            for (let f = 0; f < 3; f++) {
              a = await accumulate(a, await trace(f + 700, 1), f);
              b = await accumulate(b, await trace(f + 900, 1), f);
            }
            const av = await toArr(a);
            const bv = await toArr(b);
            const out = await sqDiff(a, b);
            const expected = (y, x) => 0.5 * (av[y][x] - bv[y][x]) * (av[y][x] - bv[y][x]);
            const hint = gridHint(out, expected, 1e-4, [
              [
                (y, x) => (av[y][x] - bv[y][x]) * (av[y][x] - bv[y][x]),
                'that is the full squared gap between the two runs, which is twice one run\'s own ' +
                  'squared error — the 0.5 is what turns a comparison into a measurement',
              ],
              [
                (y, x) => Math.abs(av[y][x] - bv[y][x]),
                'that is |a − b|, not its square — averaging absolute differences and square-rooting ' +
                  'the result is not an RMS of anything',
              ],
              [(y, x) => av[y][x] - bv[y][x], 'the difference was never squared, so positive and negative errors cancel in the sum'],
              [
                (y, x) => 0.5 * (av[x][y] - bv[x][y]) * (av[x][y] - bv[x][y]),
                'this cell holds the value belonging to [x][y] — this.thread.x and this.thread.y ' +
                  'are swapped. Rows come first: a[this.thread.y][this.thread.x]',
              ],
            ]);
            for (let y = 0; y < 64; y += 7) {
              for (let x = 0; x < 64; x += 5) {
                ctx.assertClose(out[y][x], expected(y, x), 1e-4, hint || `cell [${y}][${x}]`);
              }
            }
          },
        },
        {
          name: 'the run plots two curves, both of them falling',
          run: async ctx => {
            const entry = ctx.logs.find(l => l.plot);
            ctx.assert(entry, 'no chart in the console — the plot() call at the end draws it');
            const series = entry.plot.series;
            ctx.assert(
              series.length === 2,
              `expected two series (the measurement and the 1/√n reference), got ${series.length}`
            );
            ctx.assert(entry.plot.log, 'the chart should use a log axis — { log: true } — or the falls all look the same');
            for (const s of series) {
              ctx.assert(
                s.values.length >= 20,
                `series "${s.name}" holds ${s.values.length} points — expected one per accumulated frame`
              );
              ctx.assert(
                s.values.every(v => Number.isFinite(v) && v > 0),
                `series "${s.name}" contains a zero or a non-number — sqDiff returning 0 gives a ` +
                  'flat line at zero, which no log axis can draw and no theory line can match'
              );
            }
          },
        },
        {
          name: 'the measured noise falls like <code>1/√n</code>',
          run: async ctx => {
            const entry = ctx.logs.find(l => l.plot);
            ctx.assert(entry, 'no chart in the console');
            const measured = entry.plot.series[0].values;
            const reference = entry.plot.series[1].values;
            const slope = logLogSlope(measured);
            ctx.assert(
              slope < -0.38 && slope > -0.62,
              `the measured curve falls as n^${slope.toFixed(2)}, and Monte Carlo error falls as ` +
                'n^-0.5. A slope near 0 means the two runs are not independent (both accumulations ' +
                'drawing the same stream); a much steeper one means the error is not what is ' +
                'being measured.'
            );
            const ratio = measured[0] / measured[measured.length - 1];
            ctx.assertClose(
              ratio,
              Math.sqrt(measured.length),
              0.35 * Math.sqrt(measured.length),
              `noise at frame 1 divided by noise at frame ${measured.length}`
            );
            // The reference has to BE 1/√n, anchored to the first measurement.
            for (let i = 0; i < reference.length; i++) {
              const want = measured[0] / Math.sqrt(i + 1);
              ctx.assertClose(
                reference[i],
                want,
                Math.max(1e-4, want * 0.02),
                diagnose(reference[i], want, Math.max(1e-4, want * 0.02), [
                  [measured[0], 'the reference curve is flat — it needs the / Math.sqrt(f + 1)'],
                  [measured[0] / (i + 1), 'that reference falls as 1/n, not 1/√n — the square root is the whole point'],
                ]) || `the 1/√n reference at frame ${i + 1}`
              );
            }
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            // Re-run the identical experiment against the learner's own
            // kernels: the program is deterministic, so the plotted numbers
            // must be the numbers this measurement produces.
            const sqDiff = ctx.kernels.find(
              k => k.kernel && !k.kernel.graphical && !k.kernel.pipeline && k.lastArgs && k.lastArgs.length === 2
            );
            const trace = ctx.kernels.find(
              k => k.kernel && k.kernel.pipeline && !k.kernel.immutable && k.lastArgs && k.lastArgs.length === 2
            );
            const black = ctx.kernels.find(
              k => k.kernel && k.kernel.pipeline && !k.kernel.immutable && k.lastArgs && k.lastArgs.length === 0
            );
            const accumulate = ctx.kernels.find(k => k.kernel && k.kernel.immutable);
            ctx.assert(sqDiff && trace && black && accumulate, 'expected the four kernels of this task');
            let a = await black();
            let b = await black();
            const errs = [];
            for (let f = 0; f < 24; f++) {
              a = await accumulate(a, await trace(f, 1), f);
              b = await accumulate(b, await trace(f + 500, 1), f);
              const d = await sqDiff(a, b);
              let sum = 0;
              for (let y = 0; y < 64; y++) {
                for (let x = 0; x < 64; x++) sum += d[y][x];
              }
              errs.push(Math.sqrt(sum / 4096));
            }
            const entry = ctx.logs.find(l => l.plot);
            ctx.assert(entry, 'no chart in the console');
            const measured = entry.plot.series[0].values;
            // The samples dial only ever ADDS frames, and frame f's error does
            // not depend on how many come after it, so the first 24 plotted
            // points are these 24 numbers whatever the dial is set to.
            ctx.assert(
              measured.length >= errs.length,
              `the plotted series holds ${measured.length} points; the loop accumulates at least ` +
                `${errs.length} frames and plots one point per frame`
            );
            for (let i = 0; i < errs.length; i++) {
              ctx.assertClose(
                measured[i],
                errs[i],
                Math.max(2e-3, errs[i] * 0.05),
                `the plotted noise at frame ${i + 1} is not what these kernels measure`
              );
            }
            // …and the number itself is right: one sample per pixel of this
            // scene has an RMS error near 0.11.
            ctx.assertClose(errs[0], 0.112, 0.03, 'RMS noise after a single frame');
          },
        },
        {
          name: 'private test #2',
          run: async ctx => {
            // Four times the samples, half the noise — the same law read the
            // other way round, and the reason quality is expensive.
            const sqDiff = ctx.kernels.find(
              k => k.kernel && !k.kernel.graphical && !k.kernel.pipeline && k.lastArgs && k.lastArgs.length === 2
            );
            const trace = ctx.kernels.find(
              k => k.kernel && k.kernel.pipeline && !k.kernel.immutable && k.lastArgs && k.lastArgs.length === 2
            );
            const black = ctx.kernels.find(
              k => k.kernel && k.kernel.pipeline && !k.kernel.immutable && k.lastArgs && k.lastArgs.length === 0
            );
            const accumulate = ctx.kernels.find(k => k.kernel && k.kernel.immutable);
            ctx.assert(sqDiff && trace && black && accumulate, 'expected the four kernels of this task');
            const measure = async frames => {
              let a = await black();
              let b = await black();
              for (let f = 0; f < frames; f++) {
                a = await accumulate(a, await trace(f + 300, 1), f);
                b = await accumulate(b, await trace(f + 800, 1), f);
              }
              const d = await sqDiff(a, b);
              let sum = 0;
              for (let y = 0; y < 64; y++) {
                for (let x = 0; x < 64; x++) sum += d[y][x];
              }
              return Math.sqrt(sum / 4096);
            };
            const at4 = await measure(4);
            const at16 = await measure(16);
            ctx.assertClose(at4 / at16, 2, 0.6, 'noise at 4 frames divided by noise at 16 frames');
          },
        },
      ],
    },
  ],
};
