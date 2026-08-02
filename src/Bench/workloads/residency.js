/**
 * Resident pipeline — thirty intermediate planes that never cross the bus.
 *
 * A three-stage kernel chain over a 1536 x 1536 grid, run at ten spatial
 * scales. Stage 1 takes a dilated five-point Laplacian of the source image,
 * stage 2 measures local activity as the 3x3 sum of |Laplacian|, stage 3 adds
 * the Laplacian into a running accumulator with a gain that is throttled where
 * the neighbourhood is already busy. Thirty dispatches, twenty-nine
 * intermediates, one image in and one image out.
 *
 * This row is not here because the arithmetic is interesting. It is here to
 * price the thing every other row quietly assumes: that a GPU pipeline keeps its
 * intermediates ON the GPU. In gpu.js that is one method call, `.setPipeline(
 * true)`, which changes a kernel's return type from a JavaScript array to a
 * texture handle. In hand-written WebGPU it is the default — a storage buffer
 * only comes back if you copy it back. Both are so easy to get right that it is
 * worth stating what getting it wrong costs.
 *
 * ── THE NUMBER ─────────────────────────────────────────────────────────────
 *
 * One plane here is 1536 * 1536 * 4 = 9.0 MiB. As written, a run moves 18 MiB
 * across the bus: the source up, the answer down. If every stage returned its
 * result to JavaScript and the next stage uploaded it again — which is exactly
 * what happens if you forget `.setPipeline(true)`, and it still WORKS and still
 * gives the right answer — the same run would move 540 MiB. Thirty times the
 * traffic, for a pipeline whose arithmetic is five adds, nine absolute values
 * and three multiplies per pixel per round.
 *
 * That ratio is why residency is not a micro-optimisation. A kernel small
 * enough that one dispatch's worth of transfer IS the entire cost is the
 * degenerate case of this row. Here there are thirty of those round trips
 * waiting to be paid, each of a 9 MiB plane, and the only reason they are not
 * paid is a boolean.
 *
 * ── WHAT STAYS RESIDENT, AND WHAT MUST NOT ─────────────────────────────────
 *
 * The subtlety that makes this row a measurement rather than a decoration: the
 * FINAL result is still read back, and awaited, on every column. It has to be.
 * A pipeline whose last stage also stays on the GPU has not finished — it has
 * only been queued, and timing it reports the cost of filling a command buffer.
 * What stays resident here is the twenty-nine INTERMEDIATES and the source
 * plane; the thirtieth output comes home, and its arrival is the only evidence
 * that the other twenty-nine were ever computed.
 *
 * ── WHY TEN ROUNDS, AND WHY THEY CANNOT BE SKIPPED ─────────────────────────
 *
 * One three-stage chain over this grid is 28 ms of plain JavaScript, under the
 * 200 ms floor the sizing script enforces, so a run is ten of them at dilations
 * 1 through 10 — a multi-scale local-contrast operator, which is a real thing to
 * want and not a loop bolted on for the clock's benefit.
 *
 * Every round reads the ORIGINAL image and adds its contribution to a shared
 * accumulator. Two consequences, both deliberate. The chain cannot converge to a
 * fixed point, because no round's Laplacian is ever taken of a previous round's
 * output — a naive feedback loop settles after three rounds and then measures
 * nothing. And every round's contribution survives into the answer, so a backend
 * that ran nine of the ten is caught: the smallest single-round contribution
 * moves the checksum by 3.7e-3, thirty-seven times the runner's tolerance. Most
 * rows in this table that repeat work can only be checked on their last
 * repetition. This one is checked on all ten.
 *
 * ── EVERY VALUE IN THE CHAIN IS EXACT IN fp32 ──────────────────────────────
 *
 * The source is 8-bit greyscale held as floats, so the Laplacian is an integer
 * in [-1020, 1020] and the activity an integer in [0, 9180]. The gain is
 * 1024 - activity clamped at zero, an integer in [0, 1024], and the correction
 * is Laplacian * gain / 2048 — a product of two integers below 2^21, scaled by a
 * power of two, so it is an exact multiple of 2^-11 no larger than 510. The
 * accumulator is clamped to [0, 255] and therefore stays an exact multiple of
 * 2^-11 as well. Nothing anywhere in the chain rounds, in fp32 or in fp64, and
 * every backend on this row must agree with the oracle BIT FOR BIT.
 *
 * That is not decoration either. Stage 3 contains a clamp, which is a decision,
 * and a decision at a tie goes one way in fp64 and the other in fp32 for no
 * better reason than which unit rounded first. Made inexact, this row would
 * report correct backends as WRONG a few pixels at a time.
 *
 * ── WHAT MIGHT MISLEAD ─────────────────────────────────────────────────────
 *
 * The speed-up printed here is the speed-up of a RESIDENT pipeline over plain
 * JavaScript. It is not an upper bound on what residency is worth, because the
 * comparison it implies — the same pipeline without residency — is not a column
 * on this page. Read the 30x traffic figure above as the size of the mistake
 * being avoided, and read the row itself as what a chain of cheap kernels is
 * worth once the mistake is not being made.
 */

const W = 1536;
const H = 1536;
const ROUNDS = 10;

// Gain ceiling for the activity throttle, and the shift applied to the
// correction. Both powers of two so nothing in stage 3 rounds; see the header.
const AMAX = 1024;
const INV = 1 / 2048;

// Round r works at dilation r+1, so no two rounds are the same computation and
// the ten of them together are a multi-scale operator rather than one operator
// repeated.
const dilationOf = r => r + 1;

/**
 * An 8-bit greyscale image with structure at several scales, which is what a
 * multi-scale operator needs to have anything to do: hard-edged discs and bars
 * over a smooth low-frequency background, plus a little noise.
 *
 * Built to be cheap as well as deterministic — the sizing script warns when
 * make() costs more than js(), and rightly, since the inputs are built once and
 * shared by six columns. Hence the separable background (one sine per column,
 * one cosine per row) and discs rasterised over their bounding boxes rather than
 * tested against every pixel.
 */
function image(w, h, seed) {
  const a = new Float32Array(w * h);
  let s = seed >>> 0;
  const rnd = () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return (s >>> 8) / 0x1000000;
  };

  const sx = new Float64Array(w);
  const cy = new Float64Array(h);
  for (let i = 0; i < w; i++) sx[i] = Math.sin(i * 0.011);
  for (let i = 0; i < h; i++) cy[i] = Math.cos(i * 0.007);

  for (let y = 0; y < h; y++) {
    const row = y * w;
    const c = 50 * cy[y];
    const band = (y >> 7) % 4 === 1;
    for (let x = 0; x < w; x++) {
      let v = 120 + c * sx[x];
      if ((x >> 6) % 5 === 0) v = 210;
      if (band) v = v * 0.5 + 20;
      a[row + x] = v;
    }
  }

  for (let k = 0; k < 16; k++) {
    const ccx = rnd() * w;
    const ccy = rnd() * h;
    const r = 20 + rnd() * 130;
    const level = 20 + rnd() * 210;
    const r2 = r * r;
    const y0 = Math.max(0, Math.ceil(ccy - r));
    const y1 = Math.min(h - 1, Math.floor(ccy + r));
    for (let y = y0; y <= y1; y++) {
      const dy = y - ccy;
      const half = Math.sqrt(Math.max(0, r2 - dy * dy));
      const lo = Math.max(0, Math.ceil(ccx - half));
      const hi = Math.min(w - 1, Math.floor(ccx + half));
      const row = y * w;
      for (let x = lo; x <= hi; x++) a[row + x] = level;
    }
  }

  for (let i = 0; i < a.length; i++) {
    const v = a[i] + (rnd() - 0.5) * 8;
    a[i] = v < 0 ? 0 : v > 255 ? 255 : Math.round(v);
  }
  return a;
}

// Rows of a flat grid as a 2-D array, which is what a gpu.js kernel indexes.
// subarray, not slice: these are views, so nothing is copied.
function rows(flat, w, h) {
  const out = [];
  for (let y = 0; y < h; y++) out.push(flat.subarray(y * w, y * w + w));
  return out;
}

export default {
  id: 'residency',
  name: 'Resident 3-stage pipeline',
  params: `${W} × ${H}, 3 stages × ${ROUNDS} scales, fp32`,
  tag: 'residency',
  group: 'movement',
  size: { w: W, h: H, rounds: ROUNDS, amax: AMAX, inv: INV },

  make({ w, h }) {
    const u0 = image(w, h, 0x5bd1e995);
    return { u0, u0rows: rows(u0, w, h) };
  },

  /**
   * The oracle, and a fair baseline: the three stages in sequence, exactly as
   * the kernels do them, over flat typed arrays with the row offset hoisted out
   * of the inner loop and neighbours addressed as i ± d*w and i ± d so every
   * stage streams cache lines forwards.
   *
   * The two scratch planes are allocated once and reused across rounds rather
   * than per round — a version that allocated 30 planes would be timing the
   * garbage collector. `lap` and `act` are cleared each round because only their
   * interiors are written and the borders must be zero; the clear is one pass
   * over 9 MiB against the seventeen reads per pixel that follow it.
   */
  js({ w, h, rounds, amax, inv }, { u0 }) {
    const n = w * h;
    let acc = new Float32Array(u0);
    let out = new Float32Array(n);
    const lap = new Float32Array(n);
    const act = new Float32Array(n);

    for (let r = 0; r < rounds; r++) {
      const d = dilationOf(r);
      const dw = d * w;

      lap.fill(0);
      for (let y = d; y < h - d; y++) {
        const row = y * w;
        for (let x = d; x < w - d; x++) {
          const i = row + x;
          lap[i] = 4 * u0[i] - u0[i - dw] - u0[i + dw] - u0[i - d] - u0[i + d];
        }
      }

      act.fill(0);
      for (let y = 1; y < h - 1; y++) {
        const row = y * w;
        for (let x = 1; x < w - 1; x++) {
          const i = row + x;
          act[i] =
            Math.abs(lap[i - w - 1]) +
            Math.abs(lap[i - w]) +
            Math.abs(lap[i - w + 1]) +
            Math.abs(lap[i - 1]) +
            Math.abs(lap[i]) +
            Math.abs(lap[i + 1]) +
            Math.abs(lap[i + w - 1]) +
            Math.abs(lap[i + w]) +
            Math.abs(lap[i + w + 1]);
        }
      }

      for (let i = 0; i < n; i++) {
        let g = amax - act[i];
        if (g < 0) g = 0;
        let v = acc[i] + lap[i] * g * inv;
        if (v < 0) v = 0;
        else if (v > 255) v = 255;
        out[i] = v;
      }

      const t = acc;
      acc = out;
      out = t;
    }
    return acc;
  },

  gpujs(gpu, { w, h, rounds, amax, inv }, { u0rows }) {
    const settings = { constants: { w1: w - 1, h1: h - 1, amax, inv }, output: [w, h], pipeline: true };

    // One upload per run, and exactly one: ten rounds read the source and none
    // of them writes it. Without this the stage-1 kernel would take the raw
    // rows as an argument and gpu.js would re-upload 9 MiB on every scale, which
    // would be a perfectly good measurement of an upload path and a useless one
    // of a pipeline. It is inside run() rather than here because it is the one
    // transfer this row actually pays, and a row about transfers should not
    // hide it outside the clock.
    const upload = gpu.createKernel(
      function (v) {
        return v[this.thread.y][this.thread.x];
      },
      { output: [w, h], pipeline: true }
    );

    /**
     * Stage 1. `d` arrives as an argument rather than a constant so that one
     * compiled kernel serves all ten scales; a constant would mean ten kernels
     * and ten shader compilations, and the runner's warm-up would be timing a
     * compiler.
     *
     * The band of width d around the edge is zero. That is a branch the whole
     * workgroup takes together everywhere except at the border, so it costs the
     * GPU essentially nothing, and it keeps stage 3 from needing its own edge
     * case: where lap is zero the correction is zero and the accumulator passes
     * through unchanged.
     */
    const lapK = gpu
      .createKernel(function (u, d) {
        const x = this.thread.x;
        const y = this.thread.y;
        if (x < d || y < d || x > this.constants.w1 - d || y > this.constants.h1 - d) return 0;
        return 4 * u[y][x] - u[y - d][x] - u[y + d][x] - u[y][x - d] - u[y][x + d];
      }, settings);

    // Stage 2. Nine absolute values, one write. The lowest arithmetic intensity
    // in the chain and the reason this row is a transfer story rather than a
    // compute story: there is nothing here for a GPU to be clever about except
    // not moving the data.
    const actK = gpu
      .createKernel(function (l) {
        const x = this.thread.x;
        const y = this.thread.y;
        if (x < 1 || y < 1 || x > this.constants.w1 - 1 || y > this.constants.h1 - 1) return 0;
        return (
          Math.abs(l[y - 1][x - 1]) +
          Math.abs(l[y - 1][x]) +
          Math.abs(l[y - 1][x + 1]) +
          Math.abs(l[y][x - 1]) +
          Math.abs(l[y][x]) +
          Math.abs(l[y][x + 1]) +
          Math.abs(l[y + 1][x - 1]) +
          Math.abs(l[y + 1][x]) +
          Math.abs(l[y + 1][x + 1])
        );
      }, settings);

    // Stage 3, in two instances. gpu.js's default immutable:false has a kernel
    // reuse its own output texture, so one kernel cannot both read the previous
    // accumulator and overwrite it; they alternate. Round 0 reads the pristine
    // source, so `src` survives every repetition the runner asks for.
    const mkCombine = () =>
      gpu.createKernel(function (a, l, c) {
        const x = this.thread.x;
        const y = this.thread.y;
        let g = this.constants.amax - c[y][x];
        if (g < 0) g = 0;
        let v = a[y][x] + l[y][x] * g * this.constants.inv;
        if (v < 0) v = 0;
        if (v > 255) v = 255;
        return v;
      }, settings);
    const combineA = mkCombine();
    const combineB = mkCombine();

    return {
      async run() {
        const src = await upload(u0rows);
        let acc = src;
        for (let r = 0; r < rounds; r++) {
          // Every one of these is a texture handle, not an array. Thirty
          // dispatches, twenty-nine intermediates, and nothing between here and
          // the toArray below ever touches JavaScript memory. Round 0 reads the
          // source as its accumulator, which is why `src` is never written.
          // eslint-disable-next-line no-await-in-loop
          const l = await lapK(src, dilationOf(r));
          // eslint-disable-next-line no-await-in-loop
          const c = await actK(l);
          // eslint-disable-next-line no-await-in-loop
          acc = await (r % 2 === 0 ? combineA : combineB)(acc, l, c);
        }
        // The read-back, and awaiting it is the only thing that proves all
        // thirty dispatches finished. The CPU backend's pipeline result is
        // already a plain array of rows and has no toArray to call.
        return acc.toArray ? await acc.toArray() : acc;
      },
      backend: () => lapK.kernel && lapK.kernel.constructor.mode,
      destroy() {
        [upload, lapK, actK, combineA, combineB].forEach(k => k.destroy && k.destroy());
      },
    };
  },

  /**
   * Hand-written WebGPU. Five storage buffers, three entry points, and all
   * thirty dispatches recorded into ONE compute pass before anything is
   * submitted. WebGPU orders dispatches within a pass and inserts the barrier
   * itself, so stage 2 genuinely sees stage 1's writes without a round trip
   * through the host — the same residency the gpu.js column gets from
   * `.setPipeline(true)`, expressed as the absence of a copyBufferToBuffer.
   *
   * There is exactly one copy back, at the end, of exactly one plane. If that
   * copy were removed the whole run would report microseconds and would have
   * measured a command encoder.
   *
   * Same stencils, same order of operations, no workgroup-memory halo. A tiled
   * version would read each cell once per workgroup instead of once per thread
   * and would be meaningfully faster — and would turn this column into a measure
   * of how well the kernel was written rather than of what the runtime costs.
   */
  async webgpu(device, { w, h, rounds, amax, inv }, { u0 }) {
    const bytes = w * h * 4;
    const S = GPUBufferUsage.STORAGE;
    const mk = usage => device.createBuffer({ size: bytes, usage });
    // Uploaded inside run(), not here, so that the one transfer this row pays is
    // inside the clock on both GPU columns rather than only on one of them.
    const src = mk(S | GPUBufferUsage.COPY_DST);
    const lap = mk(S);
    const act = mk(S);
    const accA = mk(S | GPUBufferUsage.COPY_SRC);
    const accB = mk(S | GPUBufferUsage.COPY_SRC);
    const read = device.createBuffer({ size: bytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });

    // amax and inv are emitted from the constants at the top of this file, so
    // the shader multiplies by the same bits the baseline multiplies by.
    const TILE = 16;
    const CONST = `
const W: i32 = ${w};
const H: i32 = ${h};
const AMAX: f32 = ${amax}.0;
const INV: f32 = ${inv};
struct P { d: i32 };`;

    // One module per entry point: `layout: "auto"` builds a bind group layout
    // from the bindings an entry point actually touches, so sharing a module
    // between entry points with different binding sets is a quiet way to get
    // layouts that do not match the bind groups written against them.
    const shader = code => device.createShaderModule({ code });

    const mLap = shader(`${CONST}
@group(0) @binding(0) var<storage, read> u: array<f32>;
@group(0) @binding(1) var<storage, read_write> lap: array<f32>;
@group(0) @binding(2) var<uniform> p: P;
@compute @workgroup_size(${TILE}, ${TILE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = i32(gid.x);
  let y = i32(gid.y);
  if (x >= W || y >= H) { return; }
  let i = y * W + x;
  let d = p.d;
  if (x < d || y < d || x >= W - d || y >= H - d) { lap[i] = 0.0; return; }
  let dw = d * W;
  lap[i] = 4.0 * u[i] - u[i - dw] - u[i + dw] - u[i - d] - u[i + d];
}`);

    const mAct = shader(`${CONST}
@group(0) @binding(0) var<storage, read> lap: array<f32>;
@group(0) @binding(1) var<storage, read_write> act: array<f32>;
@compute @workgroup_size(${TILE}, ${TILE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = i32(gid.x);
  let y = i32(gid.y);
  if (x >= W || y >= H) { return; }
  let i = y * W + x;
  if (x < 1 || y < 1 || x >= W - 1 || y >= H - 1) { act[i] = 0.0; return; }
  act[i] =
    abs(lap[i - W - 1]) + abs(lap[i - W]) + abs(lap[i - W + 1]) +
    abs(lap[i - 1])     + abs(lap[i])     + abs(lap[i + 1]) +
    abs(lap[i + W - 1]) + abs(lap[i + W]) + abs(lap[i + W + 1]);
}`);

    const mCombine = shader(`${CONST}
@group(0) @binding(0) var<storage, read> acc: array<f32>;
@group(0) @binding(1) var<storage, read> lap: array<f32>;
@group(0) @binding(2) var<storage, read> act: array<f32>;
@group(0) @binding(3) var<storage, read_write> dst: array<f32>;
@compute @workgroup_size(${TILE}, ${TILE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = i32(gid.x);
  let y = i32(gid.y);
  if (x >= W || y >= H) { return; }
  let i = y * W + x;
  var g = AMAX - act[i];
  if (g < 0.0) { g = 0.0; }
  var v = acc[i] + lap[i] * g * INV;
  if (v < 0.0) { v = 0.0; }
  if (v > 255.0) { v = 255.0; }
  dst[i] = v;
}`);

    const pipe = module => device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: 'main' } });
    const cLap = pipe(mLap);
    const cAct = pipe(mAct);
    const cCombine = pipe(mCombine);

    // WebGPU has no push constants and queue.writeBuffer cannot be interleaved
    // between dispatches inside one command buffer, so the ten dilations live in
    // one uniform buffer at the 256-byte alignment with a bind group each, built
    // once here at zero per-run cost.
    const STRIDE = 256;
    const params = device.createBuffer({
      size: rounds * STRIDE,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const staging = new Int32Array((rounds * STRIDE) / 4);
    for (let r = 0; r < rounds; r++) staging[(r * STRIDE) / 4] = dilationOf(r);
    device.queue.writeBuffer(params, 0, staging);

    const bind = (pipeline, buffers, round) =>
      device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: buffers
          .map((buffer, binding) => ({ binding, resource: { buffer } }))
          .concat(
            round === undefined
              ? []
              : [{ binding: buffers.length, resource: { buffer: params, offset: round * STRIDE, size: 16 } }]
          ),
      });

    const lapBinds = [];
    for (let r = 0; r < rounds; r++) lapBinds.push(bind(cLap, [src, lap], r));
    const actBind = bind(cAct, [lap, act]);
    // Three accumulator bind groups, not two: round 0 reads the pristine source,
    // so A and B are free to alternate afterwards and the input is never lost.
    const first = bind(cCombine, [src, lap, act, accA]);
    const ab = bind(cCombine, [accA, lap, act, accB]);
    const ba = bind(cCombine, [accB, lap, act, accA]);
    const gx = Math.ceil(w / TILE);
    const gy = Math.ceil(h / TILE);
    // Round r writes A when r is even, B when r is odd.
    const last = (rounds - 1) % 2 === 0 ? accA : accB;

    return {
      async run() {
        device.queue.writeBuffer(src, 0, u0);
        const enc = device.createCommandEncoder();
        const pass = enc.beginComputePass();
        for (let r = 0; r < rounds; r++) {
          pass.setPipeline(cLap);
          pass.setBindGroup(0, lapBinds[r]);
          pass.dispatchWorkgroups(gx, gy);
          pass.setPipeline(cAct);
          pass.setBindGroup(0, actBind);
          pass.dispatchWorkgroups(gx, gy);
          pass.setPipeline(cCombine);
          pass.setBindGroup(0, r === 0 ? first : r % 2 === 1 ? ab : ba);
          pass.dispatchWorkgroups(gx, gy);
        }
        pass.end();
        // The one copy home, and the map that follows it, are the only proof the
        // pass ran at all.
        enc.copyBufferToBuffer(last, 0, read, 0, bytes);
        device.queue.submit([enc.finish()]);
        await read.mapAsync(GPUMapMode.READ);
        const out = new Float32Array(read.getMappedRange()).slice();
        read.unmap();
        return out;
      },
      destroy() {
        [src, lap, act, accA, accB, read, params].forEach(b => b.destroy && b.destroy());
      },
    };
  },

  /**
   * Field energy, index-weighted.
   *
   * Squares and not the plain mean, for the reason `heat` gives in reverse: this
   * operator barely moves the mean — it adds signed contrast around edges and
   * takes as much away as it puts in — so a mean-based checksum would be nearly
   * the same number whether the ten rounds ran or not. The energy rises from
   * 169,994 to 191,837 over a run, 12.8%, so a backend that did nothing is out
   * by 1,100 times the tolerance, and the smallest single round moves it by
   * 3.7e-3, thirty-seven times.
   *
   * The index weight catches a backend that processed only part of the grid, and
   * the average keeps the magnitude somewhere a relative tolerance is meaningful.
   * A 2-D gpu.js kernel resolves to an array of rows and the other two columns
   * to one flat array, so both shapes are walked rather than flattened —
   * concatenating 2.4 million floats to compute a checksum would cost more than
   * some of the cells being checked.
   */
  reduce(out, { w, h }) {
    let acc = 0;
    if (ArrayBuffer.isView(out)) {
      for (let i = 0; i < out.length; i++) acc += out[i] * out[i] * (1 + (i % 17));
    } else {
      let i = 0;
      for (let y = 0; y < out.length; y++) {
        const row = out[y];
        for (let x = 0; x < row.length; x++, i++) acc += row[x] * row[x] * (1 + (i % 17));
      }
    }
    return acc / (w * h);
  },
};
