// Module 2.5 — N-Body Gravity.
//
// Five tasks: inverse-square pull from one star → the O(n) inner loop that
// makes it O(n²) → Plummer softening → a semi-implicit Euler tick → a full
// 128-body, ten-tick simulation.
//
// Kernel-authoring rules (contract): no closures inside kernel functions,
// only numbers / nested number arrays as inputs, this.thread.* for indexing,
// loop bounds via this.constants.*, kernels may return [x, y] pairs
// (Array(2) — single precision on GL, automatic CPU fallback otherwise).
// Every task passes in CPU mode. G = 1 throughout.

// Deterministic ring of bodies: evenly spaced angles, jittered radius —
// pairwise distances are bounded below, so unsoftened forces stay finite.
function ringBodies(utils, n, seed) {
  const rand = utils.seededRandom(seed);
  const posX = new Array(n);
  const posY = new Array(n);
  const mass = new Array(n);
  for (let i = 0; i < n; i++) {
    const angle = (2 * Math.PI * i) / n;
    const radius = 0.7 + 0.6 * rand();
    posX[i] = Math.round(radius * Math.cos(angle) * 1e4) / 1e4;
    posY[i] = Math.round(radius * Math.sin(angle) * 1e4) / 1e4;
    mass[i] = Math.round((0.5 + rand()) * 100) / 100;
  }
  return { posX, posY, mass };
}

// Deterministic scattered cloud with velocities — and bodies 0 and 1 rammed
// 0.001 apart on purpose, so unsoftened kernels visibly explode.
function scatterBodies(utils, n, seed) {
  const rand = utils.seededRandom(seed);
  const posX = new Array(n);
  const posY = new Array(n);
  const velX = new Array(n);
  const velY = new Array(n);
  const mass = new Array(n);
  for (let i = 0; i < n; i++) {
    posX[i] = Math.round((rand() * 2 - 1) * 1e4) / 1e4;
    posY[i] = Math.round((rand() * 2 - 1) * 1e4) / 1e4;
    velX[i] = Math.round((rand() - 0.5) * 0.2 * 1e4) / 1e4;
    velY[i] = Math.round((rand() - 0.5) * 0.2 * 1e4) / 1e4;
    mass[i] = Math.round((0.5 + rand()) * 100) / 100;
  }
  posX[1] = posX[0] + 0.001;
  posY[1] = posY[0];
  return { posX, posY, velX, velY, mass };
}

// Reference acceleration on body i. soft = 0 means "skip self, no softening"
// (task 2 physics); soft > 0 means Plummer softening, self term is zero.
function accelOn(i, posX, posY, mass, soft) {
  let ax = 0;
  let ay = 0;
  for (let j = 0; j < posX.length; j++) {
    if (soft === 0 && j === i) continue;
    const dx = posX[j] - posX[i];
    const dy = posY[j] - posY[i];
    const r2 = dx * dx + dy * dy + soft * soft;
    if (r2 === 0) continue;
    const w = mass[j] / (r2 * Math.sqrt(r2));
    ax += dx * w;
    ay += dy * w;
  }
  return [ax, ay];
}

function accelArrays(bodies, soft) {
  const accX = new Array(bodies.posX.length);
  const accY = new Array(bodies.posX.length);
  for (let i = 0; i < bodies.posX.length; i++) {
    const a = accelOn(i, bodies.posX, bodies.posY, bodies.mass, soft);
    accX[i] = a[0];
    accY[i] = a[1];
  }
  return { accX, accY };
}

// Reference semi-implicit Euler simulation (float64), mirroring the kernel
// pipeline exactly: ALL accelerations from the step-start positions first,
// then every body updates velocity and moves with the NEW velocity.
function simulateExact(bodies, steps, dt, soft) {
  const state = {
    posX: bodies.posX.slice(),
    posY: bodies.posY.slice(),
    velX: bodies.velX.slice(),
    velY: bodies.velY.slice(),
  };
  for (let s = 0; s < steps; s++) {
    const acc = accelArrays({ posX: state.posX, posY: state.posY, mass: bodies.mass }, soft);
    for (let i = 0; i < state.posX.length; i++) {
      state.velX[i] += acc.accX[i] * dt;
      state.velY[i] += acc.accY[i] * dt;
      state.posX[i] += state.velX[i] * dt;
      state.posY[i] += state.velY[i] * dt;
    }
  }
  return state;
}

// Split an array of [x, y] pairs (kernel Array(2) output) into two arrays.
function unzip(pairs) {
  const xs = new Array(pairs.length);
  const ys = new Array(pairs.length);
  for (let i = 0; i < pairs.length; i++) {
    xs[i] = pairs[i][0];
    ys[i] = pairs[i][1];
  }
  return [xs, ys];
}

// Drive one full tick through live kernels (accel → stepVel → stepPos).
function kernelTick(accelK, velK, posK, state, mass, dt, soft) {
  const [ax, ay] = unzip(accelK(state.posX, state.posY, mass, soft));
  const [vx, vy] = unzip(velK(state.velX, state.velY, ax, ay, dt));
  const [px, py] = unzip(posK(state.posX, state.posY, vx, vy, dt));
  return { posX: px, posY: py, velX: vx, velY: vy };
}

// Tolerance that scales with the expected value (GPU float32 vs float64 ref).
function closeish(ctx, got, expected, scale, message) {
  ctx.assertClose(got, expected, scale * (1 + Math.abs(expected)), message);
}

// ---- near-miss diagnosis --------------------------------------------------
//
// A failing assert that reports only two numbers tells a learner nothing about
// WHICH slip produced them. A probe pairs the value one specific known mistake
// would produce with a sentence naming that mistake; diagnose() speaks only
// when the observed value matches a probe within the test's own tolerance AND
// the correct value does not — so bodies where two candidates coincide stay
// silent, as do observations that match probes disagreeing with each other.
// A wrong diagnosis is worse than none. `looseEps` mirrors closeish, so probes
// are judged by exactly the yardstick the assert next to them uses.
function looseEps(scale, expected) {
  return scale * (1 + Math.abs(expected));
}

function diagnose(got, expected, eps, probes) {
  const hits = probes
    .filter(p => Math.abs(got - p[0]) <= eps && Math.abs(expected - p[0]) > eps)
    .map(p => p[1]);
  return hits.length && hits.every(m => m === hits[0]) ? hits[0] : null;
}

// Task 1: the inverse-square law, over- and under-rooted.
function pullProbes(mass, r2) {
  return [
    [mass / Math.sqrt(r2),
      'that is M / r — dx * dx + dy * dy is already r², so there is no square root to take'],
    [1 / r2,
      "the star's mass never entered the result — the pull is starMass / r²"],
    [mass * r2,
      'that multiplies by r² where the law divides by it'],
  ];
}

// Task 2 reference with the wrong power in the denominator: what mass[j] * dx
// / r2 gives, with the direction dx / r never normalised.
function accelUnnormalized(i, posX, posY, mass) {
  let ax = 0;
  let ay = 0;
  for (let j = 0; j < posX.length; j++) {
    if (j === i) continue;
    const dx = posX[j] - posX[i];
    const dy = posY[j] - posY[i];
    const r2 = dx * dx + dy * dy;
    ax += (mass[j] * dx) / r2;
    ay += (mass[j] * dy) / r2;
  }
  return [ax, ay];
}

// Task 2: a missing factor of r, or an offset measured backwards.
function accelProbes(i, b, ref, component) {
  return [
    [accelUnnormalized(i, b.posX, b.posY, b.mass)[component],
      'that is mass[j]·d / r², one factor of r short — the unit direction is d / r, so each term is mass[j]·d / r³'],
    [-ref,
      'the offset points the wrong way — dx is posX[j] minus your OWN x, so the pull points at the other body'],
  ];
}

// Task 3: softening added linearly instead of squared, or not added at all.
// accelOn squares whatever ε it is handed, so √soft reproduces r² + soft.
function softProbes(i, b, soft, component) {
  return [
    [accelOn(i, b.posX, b.posY, b.mass, Math.sqrt(soft))[component],
      'that adds soft where it should add soft · soft — Plummer softening replaces r² with r² + ε²'],
    [accelOn(i, b.posX, b.posY, b.mass, 0)[component],
      'that is the unsoftened sum — ε never reached the denominator'],
  ];
}

// Task 4: the integrator without its time step, or without its update at all.
// Takes the same scale the closeish beside it uses, so both judge by one
// tolerance.
function stepHint(got, value, rate, dt, scale) {
  const expected = value + rate * dt;
  return diagnose(got, expected, looseEps(scale, expected), [
    [value + rate, 'the time step is missing — the update is value + rate · dt'],
    [value, 'that value came back unchanged — nothing was added to it'],
  ]);
}

export default {
  id: '2-5',
  track: 2,
  title: 'N-Body Gravity',
  blurb: 'Every particle pulls on every other: an O(n²) problem the GPU eats for breakfast.',
  tasks: [
    // ------------------------------------------------------------------ 1
    {
      slug: 'one-star-pull',
      title: 'The Pull of One Star',
      intro: `<p>Newton, in one line: the gravitational pull between two bodies is
        <code>G · m₁ · m₂ / r²</code>. Divide out the mass being pulled and you get its
        <strong>acceleration</strong> — <code>a = G · M / r²</code> — which only depends on the
        <em>other</em> body. In this course <code>G = 1</code> (astrophysicists rescale units to
        do exactly this, so you're in good company).</p>
        <p>Here 64 bodies drift around one star. Each thread owns one body — its position is
        <code>posX[this.thread.x]</code>, <code>posY[this.thread.x]</code> — and answers a single
        question: <em>how hard does the star pull on me?</em> No loops yet; that's next.</p>`,
      goal: `<strong>Goal:</strong> make the kernel return the strength of the star's pull on this
        thread's body: <code>starMass / r²</code>.`,
      requirements: [
        'Use the <code>dx</code>, <code>dy</code> offsets to the star (already wired up)',
        'Compute the squared distance: <code>r² = dx·dx + dy·dy</code>',
        'Return <code>starMass / r²</code> — inverse-square, with <code>G = 1</code>',
      ],
      hints: [
        {
          title: 'Hint 1 — no square root needed',
          body: `<p>The law wants <code>r²</code>, and <code>dx*dx + dy*dy</code> <em>is</em>
            <code>r²</code>. Taking <code>Math.sqrt</code> just to square it again is the most
            popular way to waste GPU cycles.</p>`,
        },
        {
          title: 'Hint 2 — the one-liner',
          body: `<p><code>return starMass / (dx * dx + dy * dy);</code></p>`,
        },
      ],
      transfer: `One-thread-per-body is the opening move of GPU physics everywhere: the CUDA SDK's
        classic <code>nbody</code> sample assigns body <em>i</em> to thread <em>i</em> exactly like
        this, and its HIP port runs the identical mapping on ROCm.`,
      starterCode: `// 64 bodies, one star. Each thread owns one body and asks:
// how hard does the star pull on ME?
const gpu = new GPU({ mode });

const pull = gpu.createKernel(function (posX, posY, starX, starY, starMass) {
  const dx = starX - posX[this.thread.x];
  const dy = starY - posY[this.thread.x];
  // TODO: inverse-square law — return starMass / r²,
  // where r² = dx·dx + dy·dy. (G = 1 here.)
  return 0;
}, { output: [64] });

const strength = pull(posX, posY, 0, 0, 100);
console.log('pull on body 0:', strength[0]);
`,
      solutionCode: `// 64 bodies, one star. Each thread owns one body and asks:
// how hard does the star pull on ME?
const gpu = new GPU({ mode });

const pull = gpu.createKernel(function (posX, posY, starX, starY, starMass) {
  const dx = starX - posX[this.thread.x];
  const dy = starY - posY[this.thread.x];
  return starMass / (dx * dx + dy * dy);
}, { output: [64] });

const strength = pull(posX, posY, 0, 0, 100);
console.log('pull on body 0:', strength[0]);
`,
      inputs: utils => {
        const b = ringBodies(utils, 64, 901);
        return { posX: b.posX, posY: b.posY };
      },
      publicTests: [
        {
          name: 'one pull strength per body — 64 positive numbers',
          run: async ctx => {
            ctx.assert(ctx.kernels.length >= 1, 'no kernel was created — call gpu.createKernel()');
            const b = ringBodies(ctx.utils, 64, 901);
            const out = ctx.kernel(b.posX, b.posY, 0, 0, 100);
            ctx.assert(out && out.length === 64, `expected 64 values, got ${out && out.length}`);
            for (let i = 0; i < 64; i++) {
              ctx.assert(
                Number.isFinite(out[i]) && out[i] > 0,
                `body ${i}: a star of mass 100 should pull with positive strength, got ${out[i]}`
              );
            }
          },
        },
        {
          name: 'doubling the distance quarters the pull — <code>M / r²</code>',
          run: async ctx => {
            // bodies lined up on the x-axis at distances 1, 2, 3, … from the star
            const posX = new Array(64);
            const posY = new Array(64);
            for (let i = 0; i < 64; i++) {
              posX[i] = i + 1;
              posY[i] = 0;
            }
            const out = ctx.kernel(posX, posY, 0, 0, 100);
            const pullHint = (i, expected) =>
              diagnose(out[i], expected, 1e-2, pullProbes(100, (i + 1) * (i + 1)));
            ctx.assertClose(out[0], 100, 1e-2, 'body at distance 1');
            ctx.assertClose(out[1], 25, 1e-2, pullHint(1, 25) || 'body at distance 2 (quarter the pull)');
            ctx.assertClose(out[3], 6.25, 1e-2, pullHint(3, 6.25) || 'body at distance 4 (a sixteenth)');
            ctx.assertClose(out[9], 1, 1e-2, pullHint(9, 1) || 'body at distance 10');
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            const b = ringBodies(ctx.utils, 64, 4242);
            const out = ctx.kernel(b.posX, b.posY, -1.5, 2.5, 77);
            for (let i = 0; i < 64; i++) {
              const dx = -1.5 - b.posX[i];
              const dy = 2.5 - b.posY[i];
              const r2 = dx * dx + dy * dy;
              const expected = 77 / r2;
              const hint = diagnose(out[i], expected, looseEps(1e-3, expected), pullProbes(77, r2));
              closeish(ctx, out[i], expected, 1e-3, hint || `body ${i}`);
            }
          },
        },
      ],
    },

    // ------------------------------------------------------------------ 2
    {
      slug: 'sum-the-sky',
      title: 'Every Body Pulls on Every Body',
      intro: `<p>Real gravity has no star at the center — <strong>every body pulls on every
        other</strong>. For 64 bodies that's 64 × 63 interactions; for a million, half a trillion.
        On the GPU the shape is beautiful: the <em>outer</em> loop over bodies becomes 64 parallel
        threads, and each thread keeps a small <em>inner</em> loop over the other 63. O(n²) work,
        O(n) time per thread, all at once.</p>
        <p>One wrinkle: pulls are <strong>vectors</strong> now, not strengths. The unit direction
        from you to body <em>j</em> is <code>(dx / r, dy / r)</code>, and the strength is
        <code>mass[j] / r²</code> — multiply them and the x-component of each contribution is
        <code>mass[j] · dx / r³</code>. This kernel sums just the x-components; skip yourself, or
        you'll divide by zero.</p>`,
      goal: `<strong>Goal:</strong> complete the inner loop so each thread returns the net
        x-acceleration on its body: the sum of <code>mass[j] · dx / r³</code> over every other body.`,
      requirements: [
        'Loop <code>j</code> over all <code>this.constants.n</code> bodies',
        'Skip yourself — the <code>j !== this.thread.x</code> guard is already there',
        'Accumulate <code>mass[j] * dx / (r² · r)</code> into <code>ax</code>',
      ],
      hints: [
        {
          title: 'Hint 1 — where does r³ come from?',
          body: `<p>Direction <code>dx / r</code> times strength <code>1 / r²</code> is
            <code>dx / r³</code>. With <code>r2 = dx*dx + dy*dy</code> in hand, that's
            <code>r2 * Math.sqrt(r2)</code> — one square root per pair.</p>`,
        },
        {
          title: 'Hint 2 — the loop body',
          body: `<pre><code>const dx = posX[j] - myX;
const dy = posY[j] - myY;
const r2 = dx * dx + dy * dy;
ax += mass[j] * dx / (r2 * Math.sqrt(r2));</code></pre>`,
        },
      ],
      transfer: `This loop-inside-a-thread is the canonical O(n²) GPU pattern. Fast CUDA and ROCm
        n-body codes keep exactly this loop but <em>tile</em> it: a thread block stages a chunk of
        bodies in shared memory so all threads reuse the loads — WebGPU's
        <code>var&lt;workgroup&gt;</code> and Metal's threadgroup memory exist for the same trick.`,
      starterCode: `// Newton, vectorised: this thread's body feels EVERY other body.
// The inner loop is O(n) — but all 64 of them run at once.
const gpu = new GPU({ mode });

const accelX = gpu.createKernel(function (posX, posY, mass) {
  const myX = posX[this.thread.x];
  const myY = posY[this.thread.x];
  let ax = 0;
  for (let j = 0; j < this.constants.n; j++) {
    if (j !== this.thread.x) {
      // TODO: dx, dy → r² → accumulate mass[j] * dx / r³
      // (dx / r is the direction, 1 / r² is the strength.)
      ax += 0;
    }
  }
  return ax;
}, { output: [64], constants: { n: 64 } });

const ax = accelX(posX, posY, mass);
console.log('net x-pull on body 0:', ax[0]);
`,
      solutionCode: `// Newton, vectorised: this thread's body feels EVERY other body.
// The inner loop is O(n) — but all 64 of them run at once.
const gpu = new GPU({ mode });

const accelX = gpu.createKernel(function (posX, posY, mass) {
  const myX = posX[this.thread.x];
  const myY = posY[this.thread.x];
  let ax = 0;
  for (let j = 0; j < this.constants.n; j++) {
    if (j !== this.thread.x) {
      const dx = posX[j] - myX;
      const dy = posY[j] - myY;
      const r2 = dx * dx + dy * dy;
      ax += mass[j] * dx / (r2 * Math.sqrt(r2));
    }
  }
  return ax;
}, { output: [64], constants: { n: 64 } });

const ax = accelX(posX, posY, mass);
console.log('net x-pull on body 0:', ax[0]);
`,
      inputs: utils => ringBodies(utils, 64, 1702),
      publicTests: [
        {
          name: "pulls are real — and Newton's third law holds",
          run: async ctx => {
            ctx.assert(ctx.kernels.length >= 1, 'no kernel was created — call gpu.createKernel()');
            const b = ringBodies(ctx.utils, 64, 1702);
            const out = ctx.kernel(b.posX, b.posY, b.mass);
            ctx.assert(out && out.length === 64, `expected 64 values, got ${out && out.length}`);
            let any = false;
            let momentum = 0;
            for (let i = 0; i < 64; i++) {
              if (Math.abs(out[i]) > 1e-3) any = true;
              momentum += b.mass[i] * out[i];
            }
            ctx.assert(any, 'every net pull came out ~0 — is the loop body still empty?');
            // equal and opposite forces: mass-weighted accelerations cancel
            ctx.assertClose(momentum, 0, 0.05, 'Σ mass[i]·ax[i] should cancel to ~0');
          },
        },
        {
          name: 'body-by-body against a reference O(n²) loop',
          run: async ctx => {
            const b = ringBodies(ctx.utils, 64, 1702);
            const out = ctx.kernel(b.posX, b.posY, b.mass);
            for (const i of [0, 17, 40, 63]) {
              const ref = accelOn(i, b.posX, b.posY, b.mass, 0);
              const hint = diagnose(out[i], ref[0], looseEps(2e-3, ref[0]), accelProbes(i, b, ref[0], 0));
              closeish(ctx, out[i], ref[0], 2e-3, hint || `net x-acceleration on body ${i}`);
            }
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            const b = ringBodies(ctx.utils, 64, 555);
            const out = ctx.kernel(b.posX, b.posY, b.mass);
            for (let i = 0; i < 64; i++) {
              const ref = accelOn(i, b.posX, b.posY, b.mass, 0);
              const hint = diagnose(out[i], ref[0], looseEps(2e-3, ref[0]), accelProbes(i, b, ref[0], 0));
              closeish(ctx, out[i], ref[0], 2e-3, hint || `net x-acceleration on body ${i}`);
            }
          },
        },
      ],
    },

    // ------------------------------------------------------------------ 3
    {
      slug: 'softening',
      title: 'Softening the Singularity',
      intro: `<p>Two of this task's bodies sit <code>0.001</code> apart. Plug that into
        <code>1 / r²</code> and their mutual pull is about a <em>million</em> — one tick of the
        clock later they're flung out of the galaxy. That's not physics; it's what happens when a
        point-mass model meets a finite time step.</p>
        <p>The standard fix is <strong>Plummer softening</strong>: replace <code>r²</code> with
        <code>r² + ε²</code>. Far away, <code>ε</code> changes nothing; up close, the force
        flattens out instead of diverging. Bonus: the <code>j !== i</code> self-check becomes dead
        weight — your own term has <code>dx = dy = 0</code>, so it contributes exactly zero. Drop
        the branch; GPUs run happiest when every thread takes the same path.</p>`,
      goal: `<strong>Goal:</strong> soften the kernel — use <code>r² + soft²</code>, drop the
        self-check, and return the full <code>[ax, ay]</code> pair.`,
      requirements: [
        'Squared distance becomes <code>dx·dx + dy·dy + soft·soft</code>',
        'Remove the <code>j !== this.thread.x</code> guard — the self term is now zero',
        'Accumulate <em>both</em> components and return <code>[ax, ay]</code>',
      ],
      hints: [
        {
          title: 'Hint 1 — why the guard can go',
          body: `<p>For <code>j === i</code>: <code>dx</code> and <code>dy</code> are 0, so the
            contribution is <code>0 · something</code>. With <code>soft² &gt; 0</code> the
            denominator is never zero, so that something is a plain finite number.</p>`,
        },
        {
          title: 'Hint 2 — share the weight',
          body: `<p>Compute <code>const w = mass[j] / (r2 * Math.sqrt(r2));</code> once, then
            <code>ax += dx * w; ay += dy * w;</code> — one denominator, two components.</p>`,
        },
      ],
      transfer: `Softening appears verbatim in production astrophysics codes (GADGET, Bonsai) on
        CUDA and ROCm clusters. It's also a lesson in GPU numerics generally: shader float math
        never throws — a divide-by-zero silently mints <code>Infinity</code> and then
        <code>NaN</code>s spread through every sum they touch, on Metal and WebGPU alike.`,
      starterCode: `// Bodies 0 and 1 sit 0.001 apart. Unsoftened, their mutual pull
// is ~a million — one bad pair and the whole simulation explodes.
const gpu = new GPU({ mode });

const accel = gpu.createKernel(function (posX, posY, mass, soft) {
  const myX = posX[this.thread.x];
  const myY = posY[this.thread.x];
  let ax = 0;
  let ay = 0;
  for (let j = 0; j < this.constants.n; j++) {
    if (j !== this.thread.x) {
      const dx = posX[j] - myX;
      const dy = posY[j] - myY;
      // TODO: soften — add soft·soft to r² so close encounters stay
      // finite. Then the j !== i guard above is dead weight: delete it.
      const r2 = dx * dx + dy * dy;
      const w = mass[j] / (r2 * Math.sqrt(r2));
      ax += dx * w;
      ay += dy * w;
    }
  }
  return [ax, ay];
}, { output: [64], constants: { n: 64 } });

const acc = accel(posX, posY, mass, 0.1);
console.log('acceleration on body 0:', acc[0][0], acc[0][1]);
`,
      solutionCode: `// Plummer softening: r² → r² + ε². Close encounters flatten out
// instead of diverging, and the self term is exactly zero — no branch.
const gpu = new GPU({ mode });

const accel = gpu.createKernel(function (posX, posY, mass, soft) {
  const myX = posX[this.thread.x];
  const myY = posY[this.thread.x];
  let ax = 0;
  let ay = 0;
  for (let j = 0; j < this.constants.n; j++) {
    const dx = posX[j] - myX;
    const dy = posY[j] - myY;
    const r2 = dx * dx + dy * dy + soft * soft;
    const w = mass[j] / (r2 * Math.sqrt(r2));
    ax += dx * w;
    ay += dy * w;
  }
  return [ax, ay];
}, { output: [64], constants: { n: 64 } });

const acc = accel(posX, posY, mass, 0.1);
console.log('acceleration on body 0:', acc[0][0], acc[0][1]);
`,
      inputs: utils => {
        const b = scatterBodies(utils, 64, 33);
        return { posX: b.posX, posY: b.posY, mass: b.mass };
      },
      publicTests: [
        {
          name: 'the close pair no longer explodes — every value stays finite and small',
          run: async ctx => {
            ctx.assert(ctx.kernels.length >= 1, 'no kernel was created — call gpu.createKernel()');
            const b = scatterBodies(ctx.utils, 64, 33);
            const out = ctx.kernel(b.posX, b.posY, b.mass, 0.1);
            ctx.assert(out && out.length === 64, `expected 64 [ax, ay] pairs, got ${out && out.length}`);
            let any = false;
            for (let i = 0; i < 64; i++) {
              ctx.assert(out[i] && out[i].length === 2, `body ${i}: expected an [ax, ay] pair`);
              const magnitude = Math.abs(out[i][0]) + Math.abs(out[i][1]);
              ctx.assert(
                Number.isFinite(magnitude) && magnitude < 1000,
                `body ${i}: |acceleration| ≈ ${magnitude.toFixed(1)} — the 0.001-apart pair is still unsoftened`
              );
              if (magnitude > 1e-3) any = true;
            }
            ctx.assert(any, 'every acceleration came out ~0 — did the loop body survive?');
          },
        },
        {
          name: 'matches the softened reference — <code>mass · d / (r² + ε²)^{3/2}</code>',
          run: async ctx => {
            const b = scatterBodies(ctx.utils, 64, 33);
            const out = ctx.kernel(b.posX, b.posY, b.mass, 0.1);
            for (const i of [0, 1, 7, 63]) {
              const ref = accelOn(i, b.posX, b.posY, b.mass, 0.1);
              const hintX = diagnose(out[i][0], ref[0], looseEps(2e-3, ref[0]), softProbes(i, b, 0.1, 0));
              const hintY = diagnose(out[i][1], ref[1], looseEps(2e-3, ref[1]), softProbes(i, b, 0.1, 1));
              closeish(ctx, out[i][0], ref[0], 2e-3, hintX || `ax on body ${i}`);
              closeish(ctx, out[i][1], ref[1], 2e-3, hintY || `ay on body ${i}`);
            }
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            // different cloud AND different ε — a hardcoded 0.1 fails here
            const b = scatterBodies(ctx.utils, 64, 909);
            const out = ctx.kernel(b.posX, b.posY, b.mass, 0.25);
            for (let i = 0; i < 64; i++) {
              const ref = accelOn(i, b.posX, b.posY, b.mass, 0.25);
              const hintX = diagnose(out[i][0], ref[0], looseEps(2e-3, ref[0]), softProbes(i, b, 0.25, 0));
              const hintY = diagnose(out[i][1], ref[1], looseEps(2e-3, ref[1]), softProbes(i, b, 0.25, 1));
              closeish(ctx, out[i][0], ref[0], 2e-3, hintX || `ax on body ${i}`);
              closeish(ctx, out[i][1], ref[1], 2e-3, hintY || `ay on body ${i}`);
            }
          },
        },
      ],
    },

    // ------------------------------------------------------------------ 4
    {
      slug: 'euler-step',
      title: 'One Tick of the Clock',
      intro: `<p>Accelerations are just numbers until an integrator turns them into motion. The
        simplest scheme that doesn't wreck orbits is <strong>semi-implicit Euler</strong>: update
        the velocity <em>first</em>, then move the body with the <em>new</em> velocity —
        <code>v′ = v + a·dt</code>, then <code>x′ = x + v′·dt</code>. Do it in the other order
        (plain Euler) and orbits visibly spiral outward, gaining energy from nowhere.</p>
        <p>Both updates are embarrassingly parallel — body <em>i</em> never looks at body
        <em>j</em> — so they're two tiny kernels. Between them, the <code>[vx, vy]</code> pairs
        come back to JavaScript and get unpacked into plain arrays for the next kernel. Clunky?
        Yes. Instructive? Also yes — and track 2's pipeline module shows how to skip the round
        trip.</p>`,
      goal: `<strong>Goal:</strong> finish both kernels — <code>stepVel</code> returns
        <code>[v + a·dt]</code> per component, <code>stepPos</code> returns
        <code>[x + v·dt]</code> — and feed the position step the <em>new</em> velocities.`,
      requirements: [
        '<code>stepVel</code> returns <code>[vx + ax·dt, vy + ay·dt]</code> for its body',
        '<code>stepPos</code> returns <code>[x + vx·dt, y + vy·dt]</code> for its body',
        'The position step must receive the <em>updated</em> velocities (semi-implicit, already wired up)',
      ],
      hints: [
        {
          title: 'Hint 1 — the same index four times',
          body: `<p>Everything in both kernels is indexed by <code>this.thread.x</code>:
            this body's velocity, this body's acceleration, this body's position.</p>`,
        },
        {
          title: 'Hint 2 — the velocity kernel',
          body: `<pre><code>return [velX[this.thread.x] + accX[this.thread.x] * dt,
        velY[this.thread.x] + accY[this.thread.x] * dt];</code></pre>
<p>— the position kernel is the
            same shape with <code>pos</code> and <code>vel</code>.</p>`,
        },
      ],
      transfer: `Splitting an integrator into per-buffer passes is exactly how GPU engines ship it:
        WebGPU dispatches one compute pass per update with position/velocity buffers ping-ponging
        between bind groups, and Metal encodes the same thing as back-to-back compute command
        encoders. The math stays this small; the choreography is the product.`,
      starterCode: `// Numbers → motion. Semi-implicit Euler: update velocity FIRST,
// then move with the NEW velocity — it keeps orbits stable.
const gpu = new GPU({ mode });

const stepVel = gpu.createKernel(function (velX, velY, accX, accY, dt) {
  // TODO: return [new vx, new vy] — old velocity plus acceleration · dt
  return [velX[this.thread.x], velY[this.thread.x]];
}, { output: [64] });

const stepPos = gpu.createKernel(function (posX, posY, velX, velY, dt) {
  // TODO: return [new x, new y] — old position plus velocity · dt
  return [posX[this.thread.x], posY[this.thread.x]];
}, { output: [64] });

const DT = 0.01;
const newVel = stepVel(velX, velY, accX, accY, DT);

// unpack the [vx, vy] pairs so the position kernel gets plain arrays
const newVelX = [];
const newVelY = [];
for (let i = 0; i < 64; i++) {
  newVelX.push(newVel[i][0]);
  newVelY.push(newVel[i][1]);
}

const newPos = stepPos(posX, posY, newVelX, newVelY, DT);
console.log('body 0 moved to', newPos[0][0], newPos[0][1]);
`,
      solutionCode: `// Numbers → motion. Semi-implicit Euler: update velocity FIRST,
// then move with the NEW velocity — it keeps orbits stable.
const gpu = new GPU({ mode });

const stepVel = gpu.createKernel(function (velX, velY, accX, accY, dt) {
  return [velX[this.thread.x] + accX[this.thread.x] * dt,
          velY[this.thread.x] + accY[this.thread.x] * dt];
}, { output: [64] });

const stepPos = gpu.createKernel(function (posX, posY, velX, velY, dt) {
  return [posX[this.thread.x] + velX[this.thread.x] * dt,
          posY[this.thread.x] + velY[this.thread.x] * dt];
}, { output: [64] });

const DT = 0.01;
const newVel = stepVel(velX, velY, accX, accY, DT);

// unpack the [vx, vy] pairs so the position kernel gets plain arrays
const newVelX = [];
const newVelY = [];
for (let i = 0; i < 64; i++) {
  newVelX.push(newVel[i][0]);
  newVelY.push(newVel[i][1]);
}

const newPos = stepPos(posX, posY, newVelX, newVelY, DT);
console.log('body 0 moved to', newPos[0][0], newPos[0][1]);
`,
      inputs: utils => {
        const b = scatterBodies(utils, 64, 74);
        const acc = accelArrays(b, 0.1);
        return {
          posX: b.posX, posY: b.posY,
          velX: b.velX, velY: b.velY,
          accX: acc.accX, accY: acc.accY,
        };
      },
      publicTests: [
        {
          // runs FIRST — later tests re-invoke the kernels and would clobber lastArgs
          name: 'the position step consumed the NEW velocities (semi-implicit)',
          run: async ctx => {
            ctx.assert(ctx.kernels.length >= 2, `expected 2 kernels (stepVel, stepPos), found ${ctx.kernels.length}`);
            const posK = ctx.kernels[1];
            ctx.assert(Array.isArray(posK.lastArgs), 'stepPos was never called');
            const b = scatterBodies(ctx.utils, 64, 74);
            const acc = accelArrays(b, 0.1);
            const seenVelX = posK.lastArgs[2];
            const seenVelY = posK.lastArgs[3];
            for (let i = 0; i < 64; i++) {
              closeish(ctx, seenVelX[i], b.velX[i] + acc.accX[i] * 0.01, 1e-3,
                `stepPos got a stale vx for body ${i} — did stepVel add a·dt?`);
              closeish(ctx, seenVelY[i], b.velY[i] + acc.accY[i] * 0.01, 1e-3,
                `stepPos got a stale vy for body ${i}`);
            }
          },
        },
        {
          name: "velocity kernel: <code>v' = v + a·dt</code>",
          run: async ctx => {
            const velK = ctx.kernels[0];
            const vx = new Array(64);
            const vy = new Array(64);
            const ax = new Array(64);
            const ay = new Array(64);
            for (let i = 0; i < 64; i++) {
              vx[i] = i * 0.1 - 3;
              vy[i] = 2 - i * 0.05;
              ax[i] = Math.sin(i) * 4;
              ay[i] = Math.cos(i) * 4;
            }
            const out = velK(vx, vy, ax, ay, 0.5);
            for (let i = 0; i < 64; i++) {
              closeish(ctx, out[i][0], vx[i] + ax[i] * 0.5, 1e-3,
                stepHint(out[i][0], vx[i], ax[i], 0.5, 1e-3) || `vx of body ${i}`);
              closeish(ctx, out[i][1], vy[i] + ay[i] * 0.5, 1e-3,
                stepHint(out[i][1], vy[i], ay[i], 0.5, 1e-3) || `vy of body ${i}`);
            }
          },
        },
        {
          name: "position kernel: <code>x' = x + v·dt</code>",
          run: async ctx => {
            const posK = ctx.kernels[1];
            const px = new Array(64);
            const py = new Array(64);
            const vx = new Array(64);
            const vy = new Array(64);
            for (let i = 0; i < 64; i++) {
              px[i] = i * 0.25;
              py[i] = -i * 0.125;
              vx[i] = 1 + i * 0.02;
              vy[i] = -2 + i * 0.03;
            }
            const out = posK(px, py, vx, vy, 0.2);
            for (let i = 0; i < 64; i++) {
              closeish(ctx, out[i][0], px[i] + vx[i] * 0.2, 1e-3,
                stepHint(out[i][0], px[i], vx[i], 0.2, 1e-3) || `x of body ${i}`);
              closeish(ctx, out[i][1], py[i] + vy[i] * 0.2, 1e-3,
                stepHint(out[i][1], py[i], vy[i], 0.2, 1e-3) || `y of body ${i}`);
            }
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            const rand = ctx.utils.seededRandom(31);
            const arrays = [];
            for (let k = 0; k < 6; k++) {
              const a = new Array(64);
              for (let i = 0; i < 64; i++) a[i] = rand() * 4 - 2;
              arrays.push(a);
            }
            const [vx, vy, ax, ay, px, py] = arrays;
            const vel = ctx.kernels[0](vx, vy, ax, ay, 0.025);
            const pos = ctx.kernels[1](px, py, vx, vy, 0.025);
            for (let i = 0; i < 64; i++) {
              closeish(ctx, vel[i][0], vx[i] + ax[i] * 0.025, 1e-3,
                stepHint(vel[i][0], vx[i], ax[i], 0.025, 1e-3) || `vx of body ${i}`);
              closeish(ctx, vel[i][1], vy[i] + ay[i] * 0.025, 1e-3,
                stepHint(vel[i][1], vy[i], ay[i], 0.025, 1e-3) || `vy of body ${i}`);
              closeish(ctx, pos[i][0], px[i] + vx[i] * 0.025, 1e-3,
                stepHint(pos[i][0], px[i], vx[i], 0.025, 1e-3) || `x of body ${i}`);
              closeish(ctx, pos[i][1], py[i] + vy[i] * 0.025, 1e-3,
                stepHint(pos[i][1], py[i], vy[i], 0.025, 1e-3) || `y of body ${i}`);
            }
          },
        },
      ],
    },

    // ------------------------------------------------------------------ 5
    {
      slug: 'full-simulation',
      title: 'Put It Together: 128 Bodies',
      intro: `<p>Everything from this module, running as one machine. The three kernels below are
        your last three tasks — softened O(n²) acceleration, the velocity tick, the position tick.
        What's missing is the <strong>conductor</strong>: a JavaScript loop that runs ten full
        ticks, feeding each kernel's output into the next and carrying the new state into the next
        tick.</p>
        <p>Notice who does what: JavaScript never touches a single interaction — it just passes
        arrays around. The GPU grinds through 128 × 128 = 16,384 interactions per tick, 163,840
        across the run. Swap 128 for 100,000 and this exact structure is a galaxy simulator; the
        loop you're about to write wouldn't change by a character.</p>`,
      goal: `<strong>Goal:</strong> write the simulation loop — ten ticks of
        <code>accel → stepVel → stepPos</code>, carrying the new arrays forward each time.`,
      requirements: [
        'Each tick: accelerations first — <code>accel(px, py, mass, SOFT)</code>',
        'Unpack the pairs, then <code>stepVel(vx, vy, ax, ay, DT)</code>, then <code>stepPos</code> with the <em>new</em> velocities',
        'Reassign <code>px, py, vx, vy</code> so the next tick starts from the new state',
        'Run exactly <code>STEPS</code> ticks, then log body 0\'s final position',
      ],
      hints: [
        {
          title: 'Hint 1 — the shape of one tick',
          body: `<p>Inside the loop: call <code>accel</code>, unpack its pairs into
            <code>ax, ay</code> arrays (the <code>unpack</code> helper is right there), call
            <code>stepVel</code>, unpack, call <code>stepPos</code>, unpack.</p>`,
        },
        {
          title: 'Hint 2 — carrying the state',
          body: `<p>End every tick by overwriting the state:</p>
<pre><code>vx = newVx;
vy = newVy;
px = newPx;
py = newPy;</code></pre>
<p>— next tick's
            <code>accel</code> must see the moved bodies, or time never advances.</p>`,
        },
        {
          title: 'Hint 3 — the whole loop',
          body: `<pre><code>for (let step = 0; step &lt; STEPS; step++) {
  const [ax, ay] = unpack(accel(px, py, mass, SOFT));
  const [nvx, nvy] = unpack(stepVel(vx, vy, ax, ay, DT));
  const [npx, npy] = unpack(stepPos(px, py, nvx, nvy, DT));
  px = npx; py = npy; vx = nvx; vy = nvy;
}</code></pre>`,
        },
      ],
      transfer: `A host loop launching device kernels in sequence is the universal skeleton of GPU
        simulation: CUDA streams queueing kernel after kernel per timestep, WebGPU building one
        command encoder per frame, Metal committing a command buffer per tick. Production codes
        differ mainly in never reading the arrays back between passes — that's what track 2's
        pipeline textures are for.`,
      starterCode: `// Three kernels from the last three tasks — and a conductor's podium.
const gpu = new GPU({ mode });
const N = 128;
const DT = 0.01;
const SOFT = 0.1;
const STEPS = 10;

const accel = gpu.createKernel(function (posX, posY, mass, soft) {
  const myX = posX[this.thread.x];
  const myY = posY[this.thread.x];
  let ax = 0;
  let ay = 0;
  for (let j = 0; j < this.constants.n; j++) {
    const dx = posX[j] - myX;
    const dy = posY[j] - myY;
    const r2 = dx * dx + dy * dy + soft * soft;
    const w = mass[j] / (r2 * Math.sqrt(r2));
    ax += dx * w;
    ay += dy * w;
  }
  return [ax, ay];
}, { output: [N], constants: { n: N } });

const stepVel = gpu.createKernel(function (velX, velY, accX, accY, dt) {
  return [velX[this.thread.x] + accX[this.thread.x] * dt,
          velY[this.thread.x] + accY[this.thread.x] * dt];
}, { output: [N] });

const stepPos = gpu.createKernel(function (posX, posY, velX, velY, dt) {
  return [posX[this.thread.x] + velX[this.thread.x] * dt,
          posY[this.thread.x] + velY[this.thread.x] * dt];
}, { output: [N] });

// [x, y] pairs → two plain arrays
function unpack(pairs) {
  const xs = [];
  const ys = [];
  for (let i = 0; i < pairs.length; i++) {
    xs.push(pairs[i][0]);
    ys.push(pairs[i][1]);
  }
  return [xs, ys];
}

let px = posX;
let py = posY;
let vx = velX;
let vy = velY;

for (let step = 0; step < STEPS; step++) {
  // TODO — one full tick:
  //   1. pairs = accel(px, py, mass, SOFT), unpack into ax, ay
  //   2. stepVel with DT → unpack into the NEW vx, vy
  //   3. stepPos with the NEW velocities → unpack into the new px, py
  //   4. reassign px, py, vx, vy for the next tick
}

console.log('after', STEPS, 'ticks, body 0 is at', px[0], py[0]);
`,
      solutionCode: `// Three kernels from the last three tasks — and a conductor's podium.
const gpu = new GPU({ mode });
const N = 128;
const DT = 0.01;
const SOFT = 0.1;
const STEPS = 10;

const accel = gpu.createKernel(function (posX, posY, mass, soft) {
  const myX = posX[this.thread.x];
  const myY = posY[this.thread.x];
  let ax = 0;
  let ay = 0;
  for (let j = 0; j < this.constants.n; j++) {
    const dx = posX[j] - myX;
    const dy = posY[j] - myY;
    const r2 = dx * dx + dy * dy + soft * soft;
    const w = mass[j] / (r2 * Math.sqrt(r2));
    ax += dx * w;
    ay += dy * w;
  }
  return [ax, ay];
}, { output: [N], constants: { n: N } });

const stepVel = gpu.createKernel(function (velX, velY, accX, accY, dt) {
  return [velX[this.thread.x] + accX[this.thread.x] * dt,
          velY[this.thread.x] + accY[this.thread.x] * dt];
}, { output: [N] });

const stepPos = gpu.createKernel(function (posX, posY, velX, velY, dt) {
  return [posX[this.thread.x] + velX[this.thread.x] * dt,
          posY[this.thread.x] + velY[this.thread.x] * dt];
}, { output: [N] });

// [x, y] pairs → two plain arrays
function unpack(pairs) {
  const xs = [];
  const ys = [];
  for (let i = 0; i < pairs.length; i++) {
    xs.push(pairs[i][0]);
    ys.push(pairs[i][1]);
  }
  return [xs, ys];
}

let px = posX;
let py = posY;
let vx = velX;
let vy = velY;

for (let step = 0; step < STEPS; step++) {
  const [ax, ay] = unpack(accel(px, py, mass, SOFT));
  const [nvx, nvy] = unpack(stepVel(vx, vy, ax, ay, DT));
  const [npx, npy] = unpack(stepPos(px, py, nvx, nvy, DT));
  px = npx;
  py = npy;
  vx = nvx;
  vy = nvy;
}

console.log('after', STEPS, 'ticks, body 0 is at', px[0], py[0]);
`,
      inputs: utils => scatterBodies(utils, 128, 55),
      publicTests: [
        {
          // runs FIRST — later tests re-invoke kernels, which would clobber lastArgs
          name: 'all ten ticks ran — the final tick saw step-nine positions',
          run: async ctx => {
            ctx.assert(ctx.kernels.length >= 3, `expected 3 kernels (accel, stepVel, stepPos), found ${ctx.kernels.length}`);
            const accelK = ctx.kernels[0];
            ctx.assert(Array.isArray(accelK.lastArgs), 'the accel kernel was never called — is the loop wired up?');
            const bodies = scatterBodies(ctx.utils, 128, 55);
            const ref = simulateExact(bodies, 9, 0.01, 0.1);
            const seenX = accelK.lastArgs[0];
            const seenY = accelK.lastArgs[1];
            for (let i = 0; i < 128; i += 7) {
              closeish(ctx, seenX[i], ref.posX[i], 5e-3,
                `tick 10 saw a wrong x for body ${i} — is the new state carried between ticks?`);
              closeish(ctx, seenY[i], ref.posY[i], 5e-3, `tick 10 saw a wrong y for body ${i}`);
            }
          },
        },
        {
          name: 'momentum is conserved across the whole run',
          run: async ctx => {
            const velK = ctx.kernels[1];
            ctx.assert(Array.isArray(velK.lastArgs), 'stepVel was never called — is the loop wired up?');
            // re-running the last velocity tick reproduces the final velocities
            const finalVel = velK(...velK.lastArgs);
            const bodies = scatterBodies(ctx.utils, 128, 55);
            let px0 = 0, py0 = 0, px1 = 0, py1 = 0;
            for (let i = 0; i < 128; i++) {
              px0 += bodies.mass[i] * bodies.velX[i];
              py0 += bodies.mass[i] * bodies.velY[i];
              px1 += bodies.mass[i] * finalVel[i][0];
              py1 += bodies.mass[i] * finalVel[i][1];
            }
            ctx.assertClose(px1, px0, 0.05, 'total x-momentum drifted — forces should cancel pairwise');
            ctx.assertClose(py1, py0, 0.05, 'total y-momentum drifted');
          },
        },
        {
          name: 'one tick, rebuilt from scratch, matches the physics',
          run: async ctx => {
            const bodies = scatterBodies(ctx.utils, 128, 55);
            const state = kernelTick(
              ctx.kernels[0], ctx.kernels[1], ctx.kernels[2],
              bodies, bodies.mass, 0.01, 0.1
            );
            const ref = simulateExact(bodies, 1, 0.01, 0.1);
            for (const i of [0, 1, 42, 127]) {
              closeish(ctx, state.posX[i], ref.posX[i], 2e-3, `x of body ${i} after one tick`);
              closeish(ctx, state.posY[i], ref.posY[i], 2e-3, `y of body ${i} after one tick`);
              closeish(ctx, state.velX[i], ref.velX[i], 2e-3, `vx of body ${i} after one tick`);
            }
          },
        },
      ],
      privateTests: [
        {
          name: 'private test #1',
          run: async ctx => {
            // a different cloud, five ticks, every body checked
            const bodies = scatterBodies(ctx.utils, 128, 991);
            let state = bodies;
            for (let s = 0; s < 5; s++) {
              state = kernelTick(
                ctx.kernels[0], ctx.kernels[1], ctx.kernels[2],
                state, bodies.mass, 0.01, 0.1
              );
            }
            const ref = simulateExact(bodies, 5, 0.01, 0.1);
            for (let i = 0; i < 128; i++) {
              closeish(ctx, state.posX[i], ref.posX[i], 5e-3, `x of body ${i} after five ticks`);
              closeish(ctx, state.posY[i], ref.posY[i], 5e-3, `y of body ${i} after five ticks`);
            }
          },
        },
      ],
    },
  ],
};
