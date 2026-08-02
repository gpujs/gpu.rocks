/**
 * Sphere-traced signed distance field — 1024 × 1024 primary rays, up to 48
 * marching steps each, then a normal, then a soft shadow ray of up to 24 more.
 *
 * This row prices a loop whose LENGTH IS DATA. Sphere tracing advances a ray by
 * the distance to the nearest surface, so a ray pointed at open sky doubles its
 * clearance every step and is finished in a handful; a ray landing square on the
 * ground plane converges in a dozen; a ray grazing the edge of a sphere creeps
 * along it and burns all 48 without ever hitting anything. On this frame the
 * average is 13.3 steps, 4.3% of pixels run to the cap, and only the 61% that
 * hit something go on to trace a shadow ray — 19 more steps on average. A GPU
 * executes a warp in lockstep, so every lane pays for the slowest in its group.
 * A CPU simply stops.
 *
 * ── THE INTERESTING PART: A RENDERED FRAME IS MOSTLY COHERENT ───────────────
 *
 * The obvious conclusion from the paragraph above is that this row should be
 * badly hurt by divergence. It is not, and the reason is worth more than the
 * assumption was. Counting SDF evaluations per pixel and grouping pixels into
 * the 8 × 4 blocks a 32-lane warp actually covers, the frame needs 28.7 M
 * evaluations and a lockstep machine must issue 29.4 M: a divergence tax of
 * 2.5%. The median warp has a spread of ZERO — every lane in it does the same
 * number of steps — and only 1.1% of warps span a range of 20 or more.
 *
 * That is what a picture looks like. Sky is uniformly sky, a floor is uniformly
 * a floor, the inside of a sphere is uniformly sphere; the divergence lives on
 * silhouettes and shadow edges, which are one-dimensional features of a
 * two-dimensional image and so are a vanishing fraction of it. Run the same
 * measurement on `escape-time`'s Julia dendrite and the tax is 10.6%, four times
 * as much, because a fractal boundary has no interior for warps to be coherent
 * inside — it is boundary everywhere at every scale.
 *
 * So the two rows are not duplicates and should be read together: escape-time is
 * the pathological case for a variable-length loop, and this one is the ordinary
 * case. The ordinary case is 2.5%. A reader who takes "GPUs hate branches" from
 * the fractal row and applies it to rendering will be wrong by a factor of four,
 * and the point of putting both on the page is to make that visible instead of
 * arguable.
 *
 * The other difference between them is arithmetic intensity: escape-time's inner
 * loop is six flops over two registers, this one is about 60 flops and three
 * square roots, called again four times for the normal and again down the shadow
 * ray. So whatever divergence does cost here has much more real work to hide
 * behind — which is the second reason to expect this row to look better than the
 * fractal, and the second reason not to read either number as "what branching
 * costs".
 *
 * ── HOW THE CHECKSUM SURVIVES A HIT/MISS TEST ──────────────────────────────
 *
 * A ray marcher is built out of two knife edges, and both were designed around
 * rather than hoped about.
 *
 * FIRST, THE HIT TEST. It is `d < 0.002` — a TOLERANCE, and a deliberately
 * generous one: 0.002 is about 2000 times the fp32 noise floor of the distances
 * being compared, so which side of it a given ray falls on is decided by
 * geometry, not by rounding. What rounding can still move is the silhouette: a
 * ray that grazes a sphere hits it in fp64 and slides past in fp32 if the two
 * evaluations disagree by more than the ray's clearance. That is unavoidable —
 * a silhouette IS a discontinuity — so the question is only how many pixels sit
 * inside the fp32 uncertainty band, and the answer is measured, not argued:
 * recomputing the whole frame with Math.fround on every single operation moves
 * the checksum by 4.6e-7 relative, about 200 times inside the runner's 1e-4, and
 * exactly ONE pixel in 1,048,576 changes by more than 0.01.
 *
 * SECOND, THE SHADOW. A hard shadow test — "did the ray to the light hit
 * anything" — would put a step discontinuity along every shadow boundary in the
 * frame, which is a second set of long contours on top of the silhouettes and
 * would roughly double the exposure. So the shadow is the standard sphere-traced
 * SOFT shadow: it tracks min(12·d/t) along the ray, a quantity that is CONTINUOUS in
 * the geometry. A ray that passes a hair from an occluder returns a hair less
 * light instead of flipping from lit to black. This is both the idiomatic thing
 * to write and the thing that makes the row checkable; a hard shadow would have
 * been cheaper and would have made this row report noise.
 *
 * THE HORIZON IS FOGGED FOR THE SAME REASON. A ground plane meets the sky at a
 * line where the hit distance runs off to `maxDist`, and the colour either side
 * of it would otherwise be unrelated. Shading is mixed toward the sky colour by
 * t/maxDist, so at the far plane the surface colour EQUALS the sky colour and
 * the horizon is continuous. That is also just what fog is.
 *
 * ── WHAT MIGHT MISLEAD ─────────────────────────────────────────────────────
 *
 * The scene is three spheres smooth-unioned with a ground plane, and a smooth
 * minimum returns less than the true distance, never more — so the march is
 * conservative and cannot overshoot. It is a small scene: a real renderer
 * evaluates hundreds of primitives through a bounding hierarchy, and the balance
 * between divergence and arithmetic would shift with it. What this row shows is
 * the shape of the effect, not a number to quote about renderers in general.
 */

const W = 1024;
const H = 1024;
const MAX_STEPS = 48;
const SHADOW_STEPS = 24;
const MAX_DIST = 24;
// The hit tolerance. See the header: a tolerance, three orders of magnitude
// above fp32 noise, not a test for equality with zero.
const EPS = 0.002;
// Smooth-union radius. Also the reason the field is C1 rather than creased,
// which keeps the marcher from taking a hard corner's word for the distance.
const KS = 0.45;
// Normal probe offset, and the shading constants.
const NEPS = 0.0015;
const SHADOW_K = 12;
const FOCAL = 1.6;

// Camera. Computed once here so that the plain-JS baseline, the gpu.js
// constants and the WGSL literals are all fed from the same three numbers.
const EYE = [0, 1.35, 4.6];
const TARGET = [0, 0.15, 0];
const CAM = (() => {
  let fx = TARGET[0] - EYE[0];
  let fy = TARGET[1] - EYE[1];
  let fz = TARGET[2] - EYE[2];
  let l = Math.sqrt(fx * fx + fy * fy + fz * fz);
  fx /= l;
  fy /= l;
  fz /= l;
  // right = normalize(cross(forward, worldUp)), up = cross(right, forward)
  let rx = fy * 0 - fz * 1;
  let ry = fz * 0 - fx * 0;
  let rz = fx * 1 - fy * 0;
  l = Math.sqrt(rx * rx + ry * ry + rz * rz);
  rx /= l;
  ry /= l;
  rz /= l;
  return {
    fx, fy, fz, rx, ry, rz,
    ux: ry * fz - rz * fy,
    uy: rz * fx - rx * fz,
    uz: rx * fy - ry * fx,
  };
})();

const LIGHT = (() => {
  const [x, y, z] = [0.55, 0.75, 0.38];
  const l = Math.sqrt(x * x + y * y + z * z);
  return { lx: x / l, ly: y / l, lz: z / l };
})();

/**
 * The scene, as a scalar function of three scalars.
 *
 * THIS FUNCTION EXISTS THREE TIMES — here, as a gpu.js added function, and as
 * WGSL — and the three must stay identical to the last digit. There is no way
 * around that: a gpu.js added function is transpiled from its own source and
 * cannot see module scope or `this.constants`, so its numbers have to be
 * literals. If you edit one, edit all three; the checksum will tell you if you
 * did not, which is exactly what a checksum is for.
 *
 * A plane at y = -1 smooth-unioned with three spheres. Polynomial smooth-min:
 * it returns a value at or below the true minimum, so the march understeps and
 * can never tunnel through a surface.
 */
function sdfScene(px, py, pz) {
  let d = py + 1;

  let ax = px + 1.6;
  let ay = py + 0.15;
  let az = pz - 0.35;
  let s = Math.sqrt(ax * ax + ay * ay + az * az) - 0.85;
  let h = 0.5 + (0.5 * (s - d)) / KS;
  if (h < 0) h = 0;
  if (h > 1) h = 1;
  d = s + (d - s) * h - KS * h * (1 - h);

  ax = px - 0.35;
  ay = py - 0.25;
  az = pz + 0.6;
  s = Math.sqrt(ax * ax + ay * ay + az * az) - 1.25;
  h = 0.5 + (0.5 * (s - d)) / KS;
  if (h < 0) h = 0;
  if (h > 1) h = 1;
  d = s + (d - s) * h - KS * h * (1 - h);

  ax = px - 1.95;
  ay = py + 0.35;
  az = pz - 0.9;
  s = Math.sqrt(ax * ax + ay * ay + az * az) - 0.65;
  h = 0.5 + (0.5 * (s - d)) / KS;
  if (h < 0) h = 0;
  if (h > 1) h = 1;
  return s + (d - s) * h - KS * h * (1 - h);
}

export default {
  id: 'sdf-march',
  name: 'SDF ray march',
  params: `${W} × ${H} px · ≤ ${MAX_STEPS} steps + ≤ ${SHADOW_STEPS} shadow, fp32`,
  tag: 'variable-length loop',
  group: 'render',
  size: { w: W, h: H, maxSteps: MAX_STEPS, shadowSteps: SHADOW_STEPS },

  // No make(): the input is the pixel grid and the grid is the size. A scene
  // uploaded as a buffer would be a table the GPU reads instead of arithmetic it
  // has to do, which is the opposite of what this row measures.

  /**
   * The oracle, and a fair baseline. Flat typed array out, scalar locals rather
   * than vector objects (three numbers in registers, no allocation per ray), the
   * per-row screen coordinate hoisted, and the same early exits the kernels
   * have — the `break` out of the march is the whole point of the row, and a
   * baseline that always ran 48 steps would hand every GPU column a speed-up it
   * had not earned.
   *
   * It is not vectorised and does not try to be. Nothing on this page is: the
   * comparison is one straightforward implementation per runtime.
   */
  js({ w, h, maxSteps, shadowSteps }) {
    const out = new Float32Array(w * h);
    const half = h / 2;
    const hw = w / 2;
    const { fx, fy, fz, rx, ry, rz, ux, uy, uz } = CAM;
    const { lx, ly, lz } = LIGHT;
    const ex = EYE[0];
    const ey = EYE[1];
    const ez = EYE[2];

    for (let y = 0; y < h; y++) {
      const sv = (half - y - 0.5) / half;

      for (let x = 0; x < w; x++) {
        const su = (x + 0.5 - hw) / half;
        // Left to right, exactly as the two shaders spell it. The two multiplies
        // that could be hoisted out of this loop are worth nothing against the
        // thousands of flops below them, and hoisting them would re-associate
        // the sum and put an avoidable ulp between the columns.
        let dx = su * rx + sv * ux + FOCAL * fx;
        let dy = su * ry + sv * uy + FOCAL * fy;
        let dz = su * rz + sv * uz + FOCAL * fz;
        const il = 1 / Math.sqrt(dx * dx + dy * dy + dz * dz);
        dx *= il;
        dy *= il;
        dz *= il;

        // ── the march. Length varies per pixel; that is the workload ───────
        let t = 0;
        let hit = 0;
        for (let i = 0; i < maxSteps; i++) {
          const d = sdfScene(ex + dx * t, ey + dy * t, ez + dz * t);
          if (d < EPS) {
            hit = 1;
            break;
          }
          t += d;
          if (t > MAX_DIST) break;
        }

        const sky = 0.55 + 0.3 * dy;
        let col = sky;
        if (hit === 1) {
          const px = ex + dx * t;
          const py = ey + dy * t;
          const pz = ez + dz * t;

          // Tetrahedron normal: four probes at the corners of a tetrahedron,
          // rather than the six a central-difference gradient would need.
          const d1 = sdfScene(px + NEPS, py - NEPS, pz - NEPS);
          const d2 = sdfScene(px - NEPS, py - NEPS, pz + NEPS);
          const d3 = sdfScene(px - NEPS, py + NEPS, pz - NEPS);
          const d4 = sdfScene(px + NEPS, py + NEPS, pz + NEPS);
          let nx = d1 - d2 - d3 + d4;
          let ny = -d1 - d2 + d3 + d4;
          let nz = -d1 + d2 - d3 + d4;
          const nl = 1 / Math.sqrt(nx * nx + ny * ny + nz * nz);
          nx *= nl;
          ny *= nl;
          nz *= nl;

          let diff = nx * lx + ny * ly + nz * lz;
          if (diff < 0) diff = 0;

          // ── the shadow ray. Soft, so the shadow edge is continuous ───────
          let res = 1;
          let st = 0.035;
          const ox = px + 0.01 * nx;
          const oy = py + 0.01 * ny;
          const oz = pz + 0.01 * nz;
          for (let i = 0; i < shadowSteps; i++) {
            const d = sdfScene(ox + lx * st, oy + ly * st, oz + lz * st);
            const q = (SHADOW_K * d) / st;
            if (q < res) res = q;
            if (res < 0.005) break;
            let step = d;
            if (step < 0.03) step = 0.03;
            if (step > 0.6) step = 0.6;
            st += step;
            if (st > 9) break;
          }
          if (res < 0) res = 0;
          if (res > 1) res = 1;

          const amb = 0.28 + 0.12 * ny;
          const lit = amb + 0.85 * diff * res;
          const f = t / MAX_DIST;
          col = lit * (1 - f) + sky * f;
        }
        out[y * w + x] = col;
      }
    }
    return out;
  },

  gpujs(gpu, { w, h, maxSteps, shadowSteps }) {
    // The scene, as a gpu.js added function so the kernel body holds ONE copy of
    // it rather than three. Argument and return types are declared rather than
    // inferred: the inference is good, but a wrong guess here would fail at
    // shader-compile time on one backend and not another.
    gpu.addFunction(
      function sdfScene(px, py, pz) {
        let d = py + 1;

        let ax = px + 1.6;
        let ay = py + 0.15;
        let az = pz - 0.35;
        let s = Math.sqrt(ax * ax + ay * ay + az * az) - 0.85;
        let h = 0.5 + (0.5 * (s - d)) / 0.45;
        if (h < 0) h = 0;
        if (h > 1) h = 1;
        d = s + (d - s) * h - 0.45 * h * (1 - h);

        ax = px - 0.35;
        ay = py - 0.25;
        az = pz + 0.6;
        s = Math.sqrt(ax * ax + ay * ay + az * az) - 1.25;
        h = 0.5 + (0.5 * (s - d)) / 0.45;
        if (h < 0) h = 0;
        if (h > 1) h = 1;
        d = s + (d - s) * h - 0.45 * h * (1 - h);

        ax = px - 1.95;
        ay = py + 0.35;
        az = pz - 0.9;
        s = Math.sqrt(ax * ax + ay * ay + az * az) - 0.65;
        h = 0.5 + (0.5 * (s - d)) / 0.45;
        if (h < 0) h = 0;
        if (h > 1) h = 1;
        return s + (d - s) * h - 0.45 * h * (1 - h);
      },
      // ARRAY-form types because the minifier renames this expression's
      // parameters, so an object form keyed by px/py/pz stops matching
      // (gpu.js#863).
      //
      // And the name is READ OFF the module-scope sdfScene rather than written
      // as the string 'sdfScene'. The kernel below calls sdfScene(), which
      // resolves lexically to that module-scope declaration — so the minifier
      // renames the declaration to something like `q` AND rewrites the call
      // site to match, while a hard-coded string registered the added function
      // under 'sdfScene'. Every gpu.js backend then failed identically with
      // "Identifier is not defined: q". Taking the name from the same binding
      // the call site resolves to is correct minified and unminified, because
      // it is by construction whatever that binding ended up being called.
      { name: sdfScene.name, argumentTypes: ['Number', 'Number', 'Number'], returnType: 'Number' }
    );

    // No kernel arguments at all: nothing is uploaded, so this row is pure
    // arithmetic plus one read-back.
    const kernel = gpu
      .createKernel(function () {
        const su = (this.thread.x + 0.5 - this.constants.hw) / this.constants.half;
        const sv = (this.constants.hh - this.thread.y - 0.5) / this.constants.half;
        let dx = su * this.constants.rx + sv * this.constants.ux + this.constants.focal * this.constants.fx;
        let dy = su * this.constants.ry + sv * this.constants.uy + this.constants.focal * this.constants.fy;
        let dz = su * this.constants.rz + sv * this.constants.uz + this.constants.focal * this.constants.fz;
        const il = 1 / Math.sqrt(dx * dx + dy * dy + dz * dz);
        dx = dx * il;
        dy = dy * il;
        dz = dz * il;

        let t = 0;
        let hit = 0;
        for (let i = 0; i < this.constants.maxSteps; i++) {
          const d = sdfScene(
            this.constants.ex + dx * t,
            this.constants.ey + dy * t,
            this.constants.ez + dz * t
          );
          if (d < this.constants.eps) {
            hit = 1;
            break;
          }
          t = t + d;
          if (t > this.constants.maxDist) break;
        }

        const sky = 0.55 + 0.3 * dy;
        let col = sky;
        if (hit > 0.5) {
          const px = this.constants.ex + dx * t;
          const py = this.constants.ey + dy * t;
          const pz = this.constants.ez + dz * t;
          const e = this.constants.neps;
          const d1 = sdfScene(px + e, py - e, pz - e);
          const d2 = sdfScene(px - e, py - e, pz + e);
          const d3 = sdfScene(px - e, py + e, pz - e);
          const d4 = sdfScene(px + e, py + e, pz + e);
          let nx = d1 - d2 - d3 + d4;
          let ny = -d1 - d2 + d3 + d4;
          let nz = -d1 + d2 - d3 + d4;
          const nl = 1 / Math.sqrt(nx * nx + ny * ny + nz * nz);
          nx = nx * nl;
          ny = ny * nl;
          nz = nz * nl;

          let diff = nx * this.constants.lx + ny * this.constants.ly + nz * this.constants.lz;
          if (diff < 0) diff = 0;

          let res = 1;
          let st = 0.035;
          const ox = px + 0.01 * nx;
          const oy = py + 0.01 * ny;
          const oz = pz + 0.01 * nz;
          for (let i = 0; i < this.constants.shadowSteps; i++) {
            const d = sdfScene(
              ox + this.constants.lx * st,
              oy + this.constants.ly * st,
              oz + this.constants.lz * st
            );
            const q = (this.constants.shadowK * d) / st;
            if (q < res) res = q;
            if (res < 0.005) break;
            let step = d;
            if (step < 0.03) step = 0.03;
            if (step > 0.6) step = 0.6;
            st = st + step;
            if (st > 9) break;
          }
          if (res < 0) res = 0;
          if (res > 1) res = 1;

          const amb = 0.28 + 0.12 * ny;
          const lit = amb + 0.85 * diff * res;
          const f = t / this.constants.maxDist;
          col = lit * (1 - f) + sky * f;
        }
        return col;
      })
      .setConstants({
        hw: W / 2,
        hh: H / 2,
        half: H / 2,
        focal: FOCAL,
        ex: EYE[0],
        ey: EYE[1],
        ez: EYE[2],
        ...CAM,
        ...LIGHT,
        maxSteps,
        shadowSteps,
        maxDist: MAX_DIST,
        eps: EPS,
        neps: NEPS,
        shadowK: SHADOW_K,
      })
      // Shaded values run past 1 and distances past 20, both outside what
      // gpu.js's 'unsigned' fallback encoding can carry. Asking for single
      // precision makes an unsupported machine fail loudly instead of quietly.
      .setPrecision('single')
      .setOutput([w, h]);

    return {
      async run() {
        return await kernel();
      },
      backend: () => kernel.kernel && kernel.kernel.constructor.mode,
      destroy: () => kernel.destroy && kernel.destroy(),
    };
  },

  /**
   * Hand-written WGSL, borrowing nothing from gpu.js. One entry point, one
   * dispatch, one read-back — deliberately the simplest possible shape, because
   * on this row the only thing separating this cell from the gpu.js WebGPU cell
   * beside it is the runtime's price for the same shader.
   *
   * The two `break`s are left exactly as they are. A real implementation would
   * not try to be clever about them, neither backend has anything better to
   * offer, and pretending the loop is fixed-length would delete the finding.
   *
   * Note that the vectors are real vec3s here and three separate scalars in the
   * other two columns. That is not a difference in arithmetic — a vec3 add is
   * three adds — it is a difference in what the language lets you write, and
   * the normalize is spelled out as 1/sqrt(dot) rather than `normalize()` so
   * that all three columns perform the same operations in the same order.
   */
  async webgpu(device, { w, h, maxSteps, shadowSteps }) {
    const bytes = w * h * 4;
    const bufOut = device.createBuffer({ size: bytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const read = device.createBuffer({ size: bytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });

    const TILE = 8;
    // WGSL will not take a bare `0` where an f32 is declared, and several of
    // these camera components are exactly 0 or 1. Every float literal goes
    // through here so none of them can come out as an integer.
    const f = v => (Number.isInteger(v) ? v.toFixed(1) : String(v));
    const module = device.createShaderModule({
      code: `
@group(0) @binding(0) var<storage, read_write> out: array<f32>;

const W: u32 = ${w}u;
const H: u32 = ${h}u;
const HALF: f32 = ${f(H / 2)};
const HW: f32 = ${f(W / 2)};
const FOCAL: f32 = ${f(FOCAL)};
const MAX_STEPS: i32 = ${maxSteps};
const SHADOW_STEPS: i32 = ${shadowSteps};
const MAX_DIST: f32 = ${f(MAX_DIST)};
const EPS: f32 = ${f(EPS)};
const KS: f32 = ${f(KS)};
const NEPS: f32 = ${f(NEPS)};
const SHADOW_K: f32 = ${f(SHADOW_K)};
const EYE = vec3<f32>(${f(EYE[0])}, ${f(EYE[1])}, ${f(EYE[2])});
const FWD = vec3<f32>(${f(CAM.fx)}, ${f(CAM.fy)}, ${f(CAM.fz)});
const RIGHT = vec3<f32>(${f(CAM.rx)}, ${f(CAM.ry)}, ${f(CAM.rz)});
const UP = vec3<f32>(${f(CAM.ux)}, ${f(CAM.uy)}, ${f(CAM.uz)});
const LIGHT = vec3<f32>(${f(LIGHT.lx)}, ${f(LIGHT.ly)}, ${f(LIGHT.lz)});

// The third copy of the scene. See the header of the plain-JS one: these
// numbers and those must agree exactly, and the checksum is what enforces it.
fn sdfScene(p: vec3<f32>) -> f32 {
  var d = p.y + 1.0;

  var a = p - vec3<f32>(-1.60, -0.15, 0.35);
  var s = sqrt(dot(a, a)) - 0.85;
  var h = clamp(0.5 + (0.5 * (s - d)) / KS, 0.0, 1.0);
  d = s + (d - s) * h - KS * h * (1.0 - h);

  a = p - vec3<f32>(0.35, 0.25, -0.60);
  s = sqrt(dot(a, a)) - 1.25;
  h = clamp(0.5 + (0.5 * (s - d)) / KS, 0.0, 1.0);
  d = s + (d - s) * h - KS * h * (1.0 - h);

  a = p - vec3<f32>(1.95, -0.35, 0.90);
  s = sqrt(dot(a, a)) - 0.65;
  h = clamp(0.5 + (0.5 * (s - d)) / KS, 0.0, 1.0);
  return s + (d - s) * h - KS * h * (1.0 - h);
}

@compute @workgroup_size(${TILE}, ${TILE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= W || gid.y >= H) { return; }
  let su = (f32(gid.x) + 0.5 - HW) / HALF;
  let sv = (HALF - f32(gid.y) - 0.5) / HALF;
  var rd = su * RIGHT + sv * UP + FOCAL * FWD;
  rd = rd * (1.0 / sqrt(dot(rd, rd)));

  var t = 0.0;
  var hit = false;
  for (var i = 0; i < MAX_STEPS; i++) {
    let d = sdfScene(EYE + rd * t);
    if (d < EPS) { hit = true; break; }
    t = t + d;
    if (t > MAX_DIST) { break; }
  }

  let sky = 0.55 + 0.3 * rd.y;
  var col = sky;
  if (hit) {
    let p = EYE + rd * t;
    let d1 = sdfScene(p + vec3<f32>( NEPS, -NEPS, -NEPS));
    let d2 = sdfScene(p + vec3<f32>(-NEPS, -NEPS,  NEPS));
    let d3 = sdfScene(p + vec3<f32>(-NEPS,  NEPS, -NEPS));
    let d4 = sdfScene(p + vec3<f32>( NEPS,  NEPS,  NEPS));
    var n = vec3<f32>(d1 - d2 - d3 + d4, -d1 - d2 + d3 + d4, -d1 + d2 - d3 + d4);
    n = n * (1.0 / sqrt(dot(n, n)));

    var diff = dot(n, LIGHT);
    if (diff < 0.0) { diff = 0.0; }

    var res = 1.0;
    var st = 0.035;
    let o = p + 0.01 * n;
    for (var i = 0; i < SHADOW_STEPS; i++) {
      let d = sdfScene(o + LIGHT * st);
      let q = (SHADOW_K * d) / st;
      if (q < res) { res = q; }
      if (res < 0.005) { break; }
      st = st + clamp(d, 0.03, 0.6);
      if (st > 9.0) { break; }
    }
    res = clamp(res, 0.0, 1.0);

    let amb = 0.28 + 0.12 * n.y;
    let lit = amb + 0.85 * diff * res;
    let f = t / MAX_DIST;
    col = lit * (1.0 - f) + sky * f;
  }
  out[gid.y * W + gid.x] = col;
}`,
    });

    const pipeline = device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: 'main' } });
    const bind = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: bufOut } }],
    });
    const gx = Math.ceil(w / TILE);
    const gy = Math.ceil(h / TILE);

    return {
      async run() {
        const enc = device.createCommandEncoder();
        const pass = enc.beginComputePass();
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bind);
        pass.dispatchWorkgroups(gx, gy);
        pass.end();
        enc.copyBufferToBuffer(bufOut, 0, read, 0, bytes);
        device.queue.submit([enc.finish()]);
        // The read-back is the only thing that proves the dispatch finished.
        await read.mapAsync(GPUMapMode.READ);
        const out = new Float32Array(read.getMappedRange()).slice();
        read.unmap();
        return out;
      },
      destroy() {
        [bufOut, read].forEach(b => b.destroy && b.destroy());
      },
    };
  },

  /**
   * Every pixel, index-weighted, in the same row-major order whichever shape the
   * backend handed back — a gpu.js 2-D kernel returns an array of rows and the
   * other two return one flat buffer, and the weight has to line up or the row
   * would report WRONG for a formatting difference.
   *
   * Values are positive and bounded by about 1.4, so nothing in the sum cancels
   * and the total cannot lose its leading digits. The measured fp32-versus-fp64
   * spread on this checksum is 4.6e-7, against a 1e-4 tolerance.
   */
  reduce(out, { w, h }) {
    let acc = 0;
    if (ArrayBuffer.isView(out)) {
      for (let i = 0; i < out.length; i++) acc += out[i] * (1 + (i % 17));
      return acc / (w * h);
    }
    let i = 0;
    for (let y = 0; y < out.length; y++) {
      const row = out[y];
      for (let x = 0; x < row.length; x++, i++) acc += row[x] * (1 + (i % 17));
    }
    return acc / (w * h);
  },
};
