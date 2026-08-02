/**
 * Hydraulic erosion on a heightfield — four kernels per tick.
 *
 * Every other simulation row here is one kernel run many times. Real simulation
 * code is not shaped like that: a step is a PIPELINE of small kernels, each one
 * needing the previous one finished across the whole grid before it can start,
 * because it reads its neighbours' new values and not just its own. This row
 * exists to price that shape. The arithmetic per cell is unremarkable; what is
 * being measured is four dependent dispatches per tick instead of one, and
 * three fields of state that have to survive between them without ever coming
 * back to the CPU.
 *
 * The four stages, all 5-point stencils over the same 1024² grid:
 *
 *   1. FLOW      rain, evaporation, and water redistributed down the gradient
 *                of the water SURFACE (terrain + water), not the terrain.
 *   2. SPEED     the surface gradient of the new water, scaled by depth — a
 *                proxy for how much the flow can carry.
 *   3. ERODE     terrain rises or falls toward the sediment capacity implied by
 *                that speed.
 *   4. TRANSPORT sediment diffuses, and gains exactly what the terrain lost.
 *
 * Stage 3 is purely local, so a hand-written implementation could fold it into
 * stage 4 and run three kernels instead of four. Every column here keeps all
 * four, because fusing in one column and not another would mean the columns
 * were doing different work — and the four-stage pipeline is the thing this row
 * is supposed to be measuring. Stages 1, 2 and 4 genuinely cannot be fused:
 * each reads the previous stage's output at its NEIGHBOURS, so the whole grid
 * has to finish before the next one starts.
 *
 * ── HONESTY ABOUT THE MODEL ────────────────────────────────────────────────
 *
 * This is a simplified diffusive erosion model, not the Mei-style virtual-pipes
 * model a graphics paper would use. The real thing carries four per-cell flux
 * components and a velocity vector, which would need several more kernels and
 * more state than the point of the row justifies. What is kept is the part that
 * matters for a benchmark: four dependent stencil passes over three coupled
 * scalar fields, each stage reading the previous stage's fresh output.
 *
 * The specific corner cut is in stage 1. Water is diffused down the surface
 * gradient and then clamped at zero, rather than each cell's outflow being
 * limited to the water it actually has — limiting it properly means knowing a
 * neighbour's outflow, which means a two-ring stencil. So the clamp quietly
 * creates a little water on ridges. It stays bounded and it pools in the
 * valleys, which is what the erosion needs, but it is not mass-conserving and
 * should not be mistaken for a physics result. It is arithmetic with the right
 * shape and the right cost, which is all a benchmark row needs it to be.
 *
 * ── WHY THE CHECKSUM SURVIVES ──────────────────────────────────────────────
 *
 * Both diffusion coefficients are inside the 2-D explicit-Euler limit of 1/4
 * (FLOW = 0.15, SEDIMENT_DIFF = 0.2), so nothing here can grow without bound,
 * and the erosion exchange is clamped per tick. That leaves the question
 * stability does not answer — whether small differences AMPLIFY — and that was
 * measured, not assumed. Running the same model with fp64 intermediates and
 * with every intermediate rounded to fp32, which is what a GPU does, the two
 * checksums differ by 7e-9 relative after the full 128 ticks. Repeating it with
 * sqrt() also nudged by 3e-7 — WGSL does not promise a correctly rounded
 * sqrt — moved the answer by 2e-9. The model is dissipative: diffusion and the
 * clamped exchange damp perturbations rather than growing them, so the row is
 * four orders of magnitude inside the runner's 1e-4 tolerance and a column that
 * disagrees here disagrees because it is wrong.
 */

const N = 1024;

// 128 ticks, not the "several hundred" the model would like. Four passes over a
// megapixel is about 18 ms of plain JS, so even 192 ticks is a 3.6 s baseline —
// past the top of the sizing band. The grid stays at 1024²: shrinking it would
// pull the working set (six 4 MB fields plus a scratch) into cache and quietly
// turn a bandwidth-bound row into an arithmetic one, so the tick count is what
// gives. 128 ticks is still 512 dependent dispatches, which is the point.
const TICKS = 128;

const RAIN = 0.008; // added to every cell each tick
const DECAY = 0.995; // evaporation
const FLOW = 0.15; // water diffusion; <= 0.25 for 2-D explicit Euler
const WCAP = 1.0; // depth beyond which extra water does not carry more
const CAPACITY = 0.6; // sediment the flow can hold per unit of activity
const EXCHANGE = 0.12; // how fast terrain moves toward that capacity
const MAX_EXCHANGE = 0.05; // per-tick clamp on how much terrain can move
const SEDIMENT_DIFF = 0.2; // sediment diffusion; <= 0.25, same limit

export default {
  id: 'erosion',
  name: 'Hydraulic erosion',
  params: `${N} × ${N} heightfield · ${TICKS} ticks × 4 kernels`,
  tag: 'multi-kernel step',
  group: 'sim',
  size: { n: N, ticks: TICKS },

  make({ n }) {
    // Octaves of smoothed value noise — a terrain with both broad valleys for
    // water to collect in and fine detail for the erosion to bite on. Seeded,
    // because two columns must start from the same landscape.
    let s = 0x1b873593 >>> 0;
    const rnd = () => {
      s = (s * 1664525 + 1013904223) >>> 0;
      return (s >>> 8) / 0x1000000;
    };

    const h = new Float32Array(n * n);
    let amp = 32;
    for (let freq = 4; freq < n; freq *= 2) {
      const side = freq + 1;
      const g = new Float32Array(side * side);
      for (let i = 0; i < g.length; i++) g[i] = rnd();
      const cell = n / freq;
      for (let y = 0; y < n; y++) {
        const gy = y / cell;
        const y0 = Math.floor(gy);
        const ty = gy - y0;
        const fy = ty * ty * (3 - 2 * ty); // smoothstep, so octaves blend without creases
        for (let x = 0; x < n; x++) {
          const gx = x / cell;
          const x0 = Math.floor(gx);
          const tx = gx - x0;
          const fx = tx * tx * (3 - 2 * tx);
          const a = g[y0 * side + x0];
          const b = g[y0 * side + x0 + 1];
          const c = g[(y0 + 1) * side + x0];
          const d = g[(y0 + 1) * side + x0 + 1];
          const top = a + (b - a) * fx;
          h[y * n + x] += amp * (top + (c + (d - c) * fx - top) * fy);
        }
      }
      amp *= 0.5;
    }

    const w = new Float32Array(n * n); // dry to begin with; the rain does the rest
    const sed = new Float32Array(n * n);

    // Row views for gpu.js's 2-D indexing. Views, not copies.
    const rows = arr => {
      const out = [];
      for (let y = 0; y < n; y++) out.push(arr.subarray(y * n, y * n + n));
      return out;
    };
    return { h, w, sed, hRows: rows(h), wRows: rows(w), sRows: rows(sed) };
  },

  /**
   * The oracle. Four passes per tick, in the same order and with the same
   * decomposition as the four kernels, so the columns are comparable — a fused
   * plain-JS version would be faster and would be measuring a different
   * algorithm from the one the GPU columns run.
   *
   * Within a pass this is ordinary tuned JS: flat Float32Arrays, buffers
   * swapped by reference rather than copied, and the three row bases hoisted
   * out of the inner loop.
   */
  js({ n, ticks }, { h, w, sed }) {
    // Fresh state each run: a run that carried the landscape forward would do
    // different work on every repetition and the median would mean nothing.
    let ch = new Float32Array(h);
    let cw = new Float32Array(w);
    let cs = new Float32Array(sed);
    let nh = new Float32Array(n * n);
    let nw = new Float32Array(n * n);
    let ns = new Float32Array(n * n);
    const vel = new Float32Array(n * n);

    for (let t = 0; t < ticks; t++) {
      // 1. FLOW — rain, evaporation, and water down the surface gradient.
      for (let y = 0; y < n; y++) {
        const up = (y === 0 ? n - 1 : y - 1) * n;
        const row = y * n;
        const down = (y === n - 1 ? 0 : y + 1) * n;
        for (let x = 0; x < n; x++) {
          const left = x === 0 ? n - 1 : x - 1;
          const right = x === n - 1 ? 0 : x + 1;
          const i = row + x;
          const surf = ch[i] + cw[i];
          const lap =
            ch[up + x] + cw[up + x] + ch[down + x] + cw[down + x] +
            ch[row + left] + cw[row + left] + ch[row + right] + cw[row + right] -
            4 * surf;
          const next = (cw[i] + RAIN) * DECAY + FLOW * lap;
          nw[i] = next > 0 ? next : 0;
        }
      }

      // 2. SPEED — gradient of the NEW water surface, scaled by depth. Needs
      //    the whole of stage 1 finished, at the neighbours as well as here.
      for (let y = 0; y < n; y++) {
        const up = (y === 0 ? n - 1 : y - 1) * n;
        const row = y * n;
        const down = (y === n - 1 ? 0 : y + 1) * n;
        for (let x = 0; x < n; x++) {
          const left = x === 0 ? n - 1 : x - 1;
          const right = x === n - 1 ? 0 : x + 1;
          const i = row + x;
          const gx = (ch[row + right] + nw[row + right] - ch[row + left] - nw[row + left]) * 0.5;
          const gy = (ch[down + x] + nw[down + x] - ch[up + x] - nw[up + x]) * 0.5;
          const depth = nw[i] < WCAP ? nw[i] : WCAP;
          vel[i] = Math.sqrt(gx * gx + gy * gy) * depth;
        }
      }

      // 3. ERODE — terrain moves toward the capacity the flow implies. Local,
      //    and clamped so no single tick can move an unbounded amount of rock.
      for (let i = 0; i < nh.length; i++) {
        let dh = EXCHANGE * (cs[i] - CAPACITY * vel[i]);
        if (dh > MAX_EXCHANGE) dh = MAX_EXCHANGE;
        else if (dh < -MAX_EXCHANGE) dh = -MAX_EXCHANGE;
        const next = ch[i] + dh;
        nh[i] = next > 0 ? next : 0;
      }

      // 4. TRANSPORT — sediment spreads, and gains exactly what the terrain
      //    lost, so the pair stays conservative apart from the clamps.
      for (let y = 0; y < n; y++) {
        const up = (y === 0 ? n - 1 : y - 1) * n;
        const row = y * n;
        const down = (y === n - 1 ? 0 : y + 1) * n;
        for (let x = 0; x < n; x++) {
          const left = x === 0 ? n - 1 : x - 1;
          const right = x === n - 1 ? 0 : x + 1;
          const i = row + x;
          const c = cs[i];
          const lap = cs[up + x] + cs[down + x] + cs[row + left] + cs[row + right] - 4 * c;
          const next = c + SEDIMENT_DIFF * lap - (nh[i] - ch[i]);
          ns[i] = next > 0 ? next : 0;
        }
      }

      let t2 = ch;
      ch = nh;
      nh = t2;
      t2 = cw;
      cw = nw;
      nw = t2;
      t2 = cs;
      cs = ns;
      ns = t2;
    }
    return { h: ch, w: cw, s: cs };
  },

  gpujs(gpu, { n, ticks }, { hRows, wRows, sRows }) {
    const constants = {
      n,
      rain: RAIN,
      decay: DECAY,
      flow: FLOW,
      wcap: WCAP,
      capacity: CAPACITY,
      exchange: EXCHANGE,
      maxExchange: MAX_EXCHANGE,
      sdiff: SEDIMENT_DIFF,
    };
    const build = fn =>
      gpu.createKernel(fn).setConstants(constants).setPipeline(true).setPrecision('single').setOutput([n, n]);

    // Stage 1. Two copies, because a pipelined mutable kernel reuses one output
    // texture: the even tick's flow kernel must not be the odd tick's, or it
    // would be asked to read the texture it is about to overwrite.
    const flowFn = function (h, w) {
      // `dim`, not `n`: a kernel-local named after one of the kernel's own
      // constants collides with the CPU backend's generated `constants_n`.
      const dim = this.constants.n;
      const x = this.thread.x;
      const y = this.thread.y;
      let left = x - 1;
      if (left < 0) left = dim - 1;
      let right = x + 1;
      if (right >= dim) right = 0;
      let up = y - 1;
      if (up < 0) up = dim - 1;
      let down = y + 1;
      if (down >= dim) down = 0;

      const surf = h[y][x] + w[y][x];
      const lap =
        h[up][x] + w[up][x] + h[down][x] + w[down][x] +
        h[y][left] + w[y][left] + h[y][right] + w[y][right] -
        4 * surf;
      const next = (w[y][x] + this.constants.rain) * this.constants.decay + this.constants.flow * lap;
      if (next > 0) return next;
      return 0;
    };
    const flowEven = build(flowFn);
    const flowOdd = build(flowFn);

    // Stage 2. Only one copy is needed: it writes the velocity field, which is
    // never one of its own inputs, so there is nothing to ping-pong.
    const speed = build(function (h, w) {
      const dim = this.constants.n;
      const x = this.thread.x;
      const y = this.thread.y;
      let left = x - 1;
      if (left < 0) left = dim - 1;
      let right = x + 1;
      if (right >= dim) right = 0;
      let up = y - 1;
      if (up < 0) up = dim - 1;
      let down = y + 1;
      if (down >= dim) down = 0;

      const gx = (h[y][right] + w[y][right] - h[y][left] - w[y][left]) * 0.5;
      const gy = (h[down][x] + w[down][x] - h[up][x] - w[up][x]) * 0.5;
      let depth = w[y][x];
      if (depth > this.constants.wcap) depth = this.constants.wcap;
      return Math.sqrt(gx * gx + gy * gy) * depth;
    });

    // Stage 3.
    const erodeFn = function (h, s, v) {
      const x = this.thread.x;
      const y = this.thread.y;
      let dh = this.constants.exchange * (s[y][x] - this.constants.capacity * v[y][x]);
      if (dh > this.constants.maxExchange) dh = this.constants.maxExchange;
      if (dh < -this.constants.maxExchange) dh = -this.constants.maxExchange;
      const next = h[y][x] + dh;
      if (next > 0) return next;
      return 0;
    };
    const erodeEven = build(erodeFn);
    const erodeOdd = build(erodeFn);

    // Stage 4. Takes both the old and the new terrain, so the amount the
    // terrain moved is a difference rather than a recomputation.
    const transportFn = function (s, h, hNext) {
      const dim = this.constants.n;
      const x = this.thread.x;
      const y = this.thread.y;
      let left = x - 1;
      if (left < 0) left = dim - 1;
      let right = x + 1;
      if (right >= dim) right = 0;
      let up = y - 1;
      if (up < 0) up = dim - 1;
      let down = y + 1;
      if (down >= dim) down = 0;

      const c = s[y][x];
      const lap = s[up][x] + s[down][x] + s[y][left] + s[y][right] - 4 * c;
      const next = c + this.constants.sdiff * lap - (hNext[y][x] - h[y][x]);
      if (next > 0) return next;
      return 0;
    };
    const transportEven = build(transportFn);
    const transportOdd = build(transportFn);

    // Uploads the initial fields once per run, and keeps the step kernels'
    // argument types constant — a plain array on tick 0 and a Texture
    // afterwards would make gpu.js recompile inside the timed region.
    const copy = () =>
      gpu
        .createKernel(function (a) {
          return a[this.thread.y][this.thread.x];
        })
        .setPipeline(true)
        .setPrecision('single')
        .setOutput([n, n]);
    const seedH = copy();
    const seedW = copy();
    const seedS = copy();

    const kernels = [
      flowEven, flowOdd, speed, erodeEven, erodeOdd,
      transportEven, transportOdd, seedH, seedW, seedS,
    ];

    return {
      async run() {
        let h = await seedH(hRows);
        let w = await seedW(wRows);
        let s = await seedS(sRows);
        for (let t = 0; t < ticks; t++) {
          const even = t % 2 === 0;
          const wNext = await (even ? flowEven : flowOdd)(h, w);
          const v = await speed(h, wNext);
          const hNext = await (even ? erodeEven : erodeOdd)(h, s, v);
          const sNext = await (even ? transportEven : transportOdd)(s, h, hNext);
          h = hNext;
          w = wNext;
          s = sNext;
        }
        // The read-backs, which are the only proof that the four dispatches per
        // tick ran rather than merely being queued.
        return {
          h: h.toArray ? await h.toArray() : h,
          w: w.toArray ? await w.toArray() : w,
          s: s.toArray ? await s.toArray() : s,
        };
      },
      backend: () => flowEven.kernel && flowEven.kernel.constructor.mode,
      destroy() {
        kernels.forEach(k => k.destroy && k.destroy());
      },
    };
  },

  /**
   * Hand-written WebGPU: four entry points in one module, one explicit bind
   * group layout, and two bind groups that swap the three state buffers.
   *
   * The whole tick loop — 4 × ticks dispatches — goes into a single compute
   * pass. WebGPU orders dispatches within a pass and makes each one's writes
   * visible to the next, which is exactly the guarantee this pipeline needs and
   * exactly why it needs no submit per stage. That is the interesting number in
   * this cell: how cheap a dependent dispatch is when nothing has to go back to
   * the CPU between them.
   */
  async webgpu(device, { n, ticks }, { h, w, sed }) {
    const cells = n * n;
    const bytes = cells * 4;
    const S = GPUBufferUsage.STORAGE;

    const upload = (data, usage) => {
      const buf = device.createBuffer({ size: bytes, usage, mappedAtCreation: true });
      new Float32Array(buf.getMappedRange()).set(data);
      buf.unmap();
      return buf;
    };
    // Pristine copies stay on the device so a run resets with a device-side
    // copy rather than 12 MB of upload inside the timed region.
    const initH = upload(h, GPUBufferUsage.COPY_SRC);
    const initW = upload(w, GPUBufferUsage.COPY_SRC);
    const initS = upload(sed, GPUBufferUsage.COPY_SRC);
    const mk = () => device.createBuffer({ size: bytes, usage: S | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
    const hA = mk();
    const wA = mk();
    const sA = mk();
    const hB = mk();
    const wB = mk();
    const sB = mk();
    const vel = device.createBuffer({ size: bytes, usage: S });
    const reads = [0, 1, 2].map(() =>
      device.createBuffer({ size: bytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ })
    );

    const TILE = 16;
    const module = device.createShaderModule({
      code: `
@group(0) @binding(0) var<storage, read> hSrc: array<f32>;
@group(0) @binding(1) var<storage, read> wSrc: array<f32>;
@group(0) @binding(2) var<storage, read> sSrc: array<f32>;
@group(0) @binding(3) var<storage, read_write> hDst: array<f32>;
@group(0) @binding(4) var<storage, read_write> wDst: array<f32>;
@group(0) @binding(5) var<storage, read_write> sDst: array<f32>;
@group(0) @binding(6) var<storage, read_write> vel: array<f32>;

const N: u32 = ${n}u;
const RAIN: f32 = ${RAIN};
const DECAY: f32 = ${DECAY};
const FLOW: f32 = ${FLOW};
const WCAP: f32 = ${WCAP};
const CAPACITY: f32 = ${CAPACITY};
const EXCHANGE: f32 = ${EXCHANGE};
const MAX_EXCHANGE: f32 = ${MAX_EXCHANGE};
const SDIFF: f32 = ${SEDIMENT_DIFF};

struct Nb { i: u32, up: u32, down: u32, left: u32, right: u32, row: u32 };
fn neighbours(x: u32, y: u32) -> Nb {
  var nb: Nb;
  nb.row   = y * N;
  nb.i     = nb.row + x;
  nb.up    = select(y - 1u, N - 1u, y == 0u) * N;
  nb.down  = select(y + 1u, 0u, y == N - 1u) * N;
  nb.left  = select(x - 1u, N - 1u, x == 0u);
  nb.right = select(x + 1u, 0u, x == N - 1u);
  return nb;
}

@compute @workgroup_size(${TILE}, ${TILE})
fn flow(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= N || gid.y >= N) { return; }
  let nb = neighbours(gid.x, gid.y);
  let surf = hSrc[nb.i] + wSrc[nb.i];
  let lap = hSrc[nb.up + gid.x] + wSrc[nb.up + gid.x]
          + hSrc[nb.down + gid.x] + wSrc[nb.down + gid.x]
          + hSrc[nb.row + nb.left] + wSrc[nb.row + nb.left]
          + hSrc[nb.row + nb.right] + wSrc[nb.row + nb.right]
          - 4.0 * surf;
  wDst[nb.i] = max(0.0, (wSrc[nb.i] + RAIN) * DECAY + FLOW * lap);
}

@compute @workgroup_size(${TILE}, ${TILE})
fn speed(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= N || gid.y >= N) { return; }
  let nb = neighbours(gid.x, gid.y);
  let gx = (hSrc[nb.row + nb.right] + wDst[nb.row + nb.right]
          - hSrc[nb.row + nb.left]  - wDst[nb.row + nb.left]) * 0.5;
  let gy = (hSrc[nb.down + gid.x] + wDst[nb.down + gid.x]
          - hSrc[nb.up + gid.x]    - wDst[nb.up + gid.x]) * 0.5;
  vel[nb.i] = sqrt(gx * gx + gy * gy) * min(wDst[nb.i], WCAP);
}

@compute @workgroup_size(${TILE}, ${TILE})
fn erode(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= N || gid.y >= N) { return; }
  let i = gid.y * N + gid.x;
  let dh = clamp(EXCHANGE * (sSrc[i] - CAPACITY * vel[i]), -MAX_EXCHANGE, MAX_EXCHANGE);
  hDst[i] = max(0.0, hSrc[i] + dh);
}

@compute @workgroup_size(${TILE}, ${TILE})
fn transport(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= N || gid.y >= N) { return; }
  let nb = neighbours(gid.x, gid.y);
  let c = sSrc[nb.i];
  let lap = sSrc[nb.up + gid.x] + sSrc[nb.down + gid.x]
          + sSrc[nb.row + nb.left] + sSrc[nb.row + nb.right] - 4.0 * c;
  sDst[nb.i] = max(0.0, c + SDIFF * lap - (hDst[nb.i] - hSrc[nb.i]));
}`,
    });

    const ro = { type: 'read-only-storage' };
    const rw = { type: 'storage' };
    const layout = device.createBindGroupLayout({
      entries: [ro, ro, ro, rw, rw, rw, rw].map((buffer, binding) => ({
        binding,
        visibility: GPUShaderStage.COMPUTE,
        buffer,
      })),
    });
    const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [layout] });
    const stage = entryPoint => device.createComputePipeline({ layout: pipelineLayout, compute: { module, entryPoint } });
    const pipelines = [stage('flow'), stage('speed'), stage('erode'), stage('transport')];

    const bindGroup = (hs, ws, ss, hd, wd, sd) =>
      device.createBindGroup({
        layout,
        entries: [hs, ws, ss, hd, wd, sd, vel].map((buffer, binding) => ({ binding, resource: { buffer } })),
      });
    const aToB = bindGroup(hA, wA, sA, hB, wB, sB);
    const bToA = bindGroup(hB, wB, sB, hA, wA, sA);
    const groups = Math.ceil(n / TILE);
    const final = ticks % 2 === 0 ? [hA, wA, sA] : [hB, wB, sB];

    return {
      async run() {
        const enc = device.createCommandEncoder();
        enc.copyBufferToBuffer(initH, 0, hA, 0, bytes);
        enc.copyBufferToBuffer(initW, 0, wA, 0, bytes);
        enc.copyBufferToBuffer(initS, 0, sA, 0, bytes);
        const pass = enc.beginComputePass();
        for (let t = 0; t < ticks; t++) {
          const bind = t % 2 === 0 ? aToB : bToA;
          for (const pipeline of pipelines) {
            pass.setPipeline(pipeline);
            pass.setBindGroup(0, bind);
            pass.dispatchWorkgroups(groups, groups);
          }
        }
        pass.end();
        final.forEach((buf, i) => enc.copyBufferToBuffer(buf, 0, reads[i], 0, bytes));
        device.queue.submit([enc.finish()]);
        await Promise.all(reads.map(r => r.mapAsync(GPUMapMode.READ)));
        const [outH, outW, outS] = reads.map(r => {
          const copy = new Float32Array(r.getMappedRange()).slice();
          r.unmap();
          return copy;
        });
        return { h: outH, w: outW, s: outS };
      },
      destroy() {
        [initH, initW, initS, hA, wA, sA, hB, wB, sB, vel, ...reads].forEach(b => b.destroy && b.destroy());
      },
    };
  },

  /**
   * All three fields, every cell, with a different weight period each so a
   * backend that crossed two of them over does not pass. The three are strongly
   * coupled — sediment feeds straight into the terrain exchange and water into
   * the speed — so an error in any one stage shows up in all three within a
   * tick, which is what makes this a real check on a four-kernel pipeline
   * rather than on its last kernel.
   *
   * Divided by the cell count so a million-term sum stays where fp32 and fp64
   * agree far inside the runner's tolerance. gpu.js hands back rows and the
   * other two columns hand back flat arrays; both are walked in the same order.
   */
  reduce(out, { n }) {
    const fold = (field, period) => {
      let acc = 0;
      let i = 0;
      if (ArrayBuffer.isView(field)) {
        for (; i < field.length; i++) acc += field[i] * (1 + (i % period));
      } else {
        for (let y = 0; y < field.length; y++) {
          const row = field[y];
          for (let x = 0; x < row.length; x++, i++) acc += row[x] * (1 + (i % period));
        }
      }
      return acc;
    };
    return (fold(out.h, 17) + fold(out.w, 13) + fold(out.s, 11)) / (n * n);
  },
};
