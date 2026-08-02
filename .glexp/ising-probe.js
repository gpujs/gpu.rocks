/**
 * Ising model, Metropolis updates, red-black checkerboard.
 *
 * This row exists because of the write race. A Metropolis sweep flips one spin
 * at a time using its four neighbours, so two neighbouring sites may not be
 * decided at the same instant — whichever went second was supposed to see the
 * other's new value. A GPU decides a million sites at once, so the naive
 * parallel sweep is simply the wrong algorithm.
 *
 * The fix is a colouring. On a square lattice every site's four neighbours have
 * the opposite parity of (x + y), so the red sites and the black sites form two
 * independent sets: all reds can be updated simultaneously because none of them
 * is a neighbour of another. One sweep therefore becomes two half-passes, and
 * that doubling of the dispatch count — not the arithmetic — is what this row
 * prices. It is the cheapest honest example of an ordered dependency: the GPU
 * does not get to flatten it, it only gets to do half of it at a time.
 *
 * ── WHY THE RANDOMNESS IS A TABLE AND NOT Math.random ───────────────────────
 *
 * Metropolis needs a uniform draw per site per sweep. Math.random would make
 * this row uncheckable: every backend would take different decisions, every
 * checksum would differ, and there would be no way to tell a broken kernel from
 * a differently-seeded one. So the draw must be a pure function of (site,
 * sweep) that every backend evaluates identically.
 *
 * A bitwise integer hash is the usual way to write that, and it is the wrong
 * way here. gpu.js's WebGL1 backend has no integer type: `^`, `&` and `<<`
 * compile to 32-iteration emulation loops over floats. A hash would then cost
 * more than the physics and this row would be measuring gpu.js's bitwise
 * polyfill. Worse, its exactness depends on float division rounding the same
 * way on every driver, and a single flipped decision in a near-critical Ising
 * lattice diverges — the checksum would disagree for a reason that is not a bug.
 *
 * So: make() builds one seeded random field, shared by every column (rule 2),
 * and each sweep reads it through a whole-lattice shift. Site (x, y) at sweep t
 * draws rnd[(y + 577t) mod n][(x + 373t) mod n]. The shifts are odd, so each of
 * the 256 sweeps uses a different one; within a sweep the draws are independent
 * by construction, and the values are small integers that a float32 texture
 * carries exactly on every backend.
 *
 * That exactness is the point. The acceptance test compares two integers —
 * a draw in [0, 65536) against a precomputed threshold — never two floats, so
 * every backend takes bit-for-bit identical decisions and the final lattice is
 * identical, not merely close. The checksum is a sum of ±1 and must match
 * EXACTLY; if it ever does not, something is genuinely wrong.
 */

const N = 1024;
const SWEEPS = 256;

// 2.2 sits just below the 2D critical temperature (2.269), where domains are
// large and the dynamics are interesting rather than frozen or scrambled.
const TEMP = 2.2;

// The Metropolis test, in integers. A spin s with neighbour sum S has
// dE = 2*s*S, and s*S can only be one of {-4,-2,0,2,4} because S is a sum of
// four ±1. So there are exactly two thresholds to precompute, and the draw is
// compared against them as an integer in [0, LEVELS). No exp() on the GPU, no
// float comparison, nothing whose last bit can differ between backends.
const LEVELS = 65536;
const ACCEPT_4 = Math.round(Math.exp(-4 / TEMP) * LEVELS); // dE = +4
const ACCEPT_8 = Math.round(Math.exp(-8 / TEMP) * LEVELS); // dE = +8

const SHIFT_X = 373;
const SHIFT_Y = 577;

// Deterministic and cheap; the same generator matmul uses. Every column is
// handed the same bytes, which a seeded sequence is the only way to promise.
//
// Callers take the HIGH bits of the state. Bit k of a power-of-two-modulus LCG
// repeats with period 2^(k+1), so a spin drawn from bit 8 would repeat every
// 512 sites — which on a 1024-wide lattice means every row starts out
// identical, and the "random" initial condition has a stripe pattern baked into
// it. Bit 31 has the full period.
function lcg(seed) {
  let s = seed >>> 0;
  return () => (s = (s * 1664525 + 1013904223) >>> 0);
}

export default {
  id: 'ising',
  name: 'Ising model, Metropolis',
  params: `${N} × ${N} lattice · ${SWEEPS} red-black sweeps · T = ${TEMP}`,
  tag: 'ordered dependency',
  group: 'sim',
  size: { n: N, sweeps: SWEEPS },

  make({ n, sweeps }) {
    const next = lcg(0xc2b2ae35);
    const spins = new Float32Array(n * n);
    for (let i = 0; i < spins.length; i++) spins[i] = next() >>> 31 ? 1 : -1;

    // One random value per site, reused across sweeps through a shift. Held as
    // exact small integers so that "draw < threshold" is an integer comparison
    // on every backend, whatever the texture format underneath.
    const rnd = new Float32Array(n * n);
    for (let i = 0; i < rnd.length; i++) rnd[i] = next() >>> 16; // top 16 bits: [0, LEVELS)

    // Rows, for the 2-D indexing a gpu.js kernel does. Views, not copies.
    const rows = [];
    for (let y = 0; y < n; y++) rows.push(rnd.subarray(y * n, y * n + n));
    const spinRows = [];
    for (let y = 0; y < n; y++) spinRows.push(spins.subarray(y * n, y * n + n));

    return { spins, rnd, rows, spinRows, sweeps };
  },

  /**
   * The oracle. Updates IN PLACE, which is the whole payoff of the colouring:
   * a red pass reads only black sites, so nothing it writes is read again until
   * the pass is over. Rows are walked in order and each colour is reached by
   * stepping x by 2, which is the natural way to write this and the fast one.
   *
   * Note the asymmetry this creates with the gpu.js column, which is real and
   * is part of what the row reports: a gpu.js kernel cannot write into its
   * input, so it must emit a whole new lattice per half-pass and copy across
   * the colour it did not touch. Same arithmetic, twice the memory traffic,
   * because the runtime has no in-place kernel. The bare-WebGPU column updates
   * in place exactly like this one does.
   */
  js({ n, sweeps }, { spins, rnd }) {
    const s = new Float32Array(spins); // every run starts from the same lattice
    const mask = n - 1; // n is a power of two, so the wrap is a mask

    for (let t = 0; t < sweeps; t++) {
      const sx = (t * SHIFT_X) & mask;
      const sy = (t * SHIFT_Y) & mask;
      for (let parity = 0; parity < 2; parity++) {
        for (let y = 0; y < n; y++) {
          const row = y * n;
          const up = (y === 0 ? n - 1 : y - 1) * n;
          const down = (y === n - 1 ? 0 : y + 1) * n;
          const rrow = ((y + sy) & mask) * n;
          for (let x = (y + parity) & 1; x < n; x += 2) {
            const c = s[row + x];
            const left = x === 0 ? n - 1 : x - 1;
            const right = x === n - 1 ? 0 : x + 1;
            // e = s * (sum of four neighbours); dE = 2e, so e <= 0 is downhill.
            const e = c * (s[up + x] + s[down + x] + s[row + left] + s[row + right]);
            if (e <= 0) {
              s[row + x] = -c;
            } else {
              const r = rnd[rrow + ((x + sx) & mask)];
              if (r < (e < 3 ? ACCEPT_4 : ACCEPT_8)) s[row + x] = -c;
            }
          }
        }
      }
    }
    return s;
  },

  gpujs(gpu, { n, sweeps }, { rows, spinRows }) {
    const constants = {
      n,
      rnd: rows, // a constant, so it uploads once — an argument would re-upload 4 MB per dispatch
      accept4: ACCEPT_4,
      accept8: ACCEPT_8,
      parity: 0,
    };

    // One kernel per colour. Both are pipelined and mutable, so each owns a
    // single output texture that it reuses: red reads black's texture and
    // writes its own, black reads red's and writes its own, and neither ever
    // writes a texture that is currently an input. That is the ping-pong, and
    // it costs no allocation per sweep.
    const half = parity =>
      gpu
        .createKernel(function (state, sx, sy) {
          // `dim`, not `n`: a kernel-local named after one of its own constants
          // collides with the CPU backend's generated `constants_n`.
          const dim = this.constants.n;
          const x = this.thread.x;
          const y = this.thread.y;
          const c = state[y][x];

          // (x + y) & 1 without a bitwise op, which WebGL1 would emulate in a
          // loop. Division by two is exact in binary floating point.
          const p = x + y;
          if (p - 2 * Math.floor(p * 0.5) !== this.constants.parity) return c;

          // Periodic neighbours by branch rather than by %, so nothing depends
          // on how a driver rounds a float division.
          let left = x - 1;
          if (left < 0) left = dim - 1;
          let right = x + 1;
          if (right >= dim) right = 0;
          let up = y - 1;
          if (up < 0) up = dim - 1;
          let down = y + 1;
          if (down >= dim) down = 0;

          const e = c * (state[up][x] + state[down][x] + state[y][left] + state[y][right]);
          if (e <= 0) return -c;

          let rx = x + sx;
          if (rx >= dim) rx = rx - dim;
          let ry = y + sy;
          if (ry >= dim) ry = ry - dim;
          const r = this.constants.rnd[ry][rx];
          if (e < 3) {
            if (r < this.constants.accept4) return -c;
            return c;
          }
          if (r < this.constants.accept8) return -c;
          return c;
        })
        .setConstants({ ...constants, parity })
        .setPipeline(true)
        .setPrecision('single')
        .setOutput([n, n]);

    const red = half(0);
    const black = half(1);

    // Uploads the pristine lattice into a texture. It exists so the ping-pong
    // kernels always see a Texture argument: handing them a plain array on the
    // first sweep of every run would make gpu.js recompile for the new argument
    // type, and the row would be timing a shader compiler.
    const seed = gpu
      .createKernel(function (s) {
        return s[this.thread.y][this.thread.x];
      })
      .setPipeline(true)
      .setPrecision('single')
      .setOutput([n, n]);

    const mask = n - 1;

    const sum = a => {
      let s = 0;
      if (ArrayBuffer.isView(a)) { for (let i = 0; i < a.length; i++) s += a[i]; return s; }
      for (const r of a) for (let i = 0; i < r.length; i++) s += r[i];
      return s;
    };
    const peek = async (tag, tex) => {
      const v = tex.toArray ? await tex.toArray() : tex;
      (globalThis.__probe = globalThis.__probe || []).push([tag, sum(v)]);
    };
    let runNo = 0;

    return {
      async run() {
        runNo++;
        let state = await seed(spinRows);
        await peek(`run${runNo}.seed`, state);
        for (let t = 0; t < sweeps; t++) {
          const sx = (t * SHIFT_X) & mask;
          const sy = (t * SHIFT_Y) & mask;
          state = await red(state, sx, sy);
          if (t < 3) await peek(`run${runNo}.red${t}`, state);
          state = await black(state, sx, sy);
          if (t < 3) await peek(`run${runNo}.black${t}`, state);
        }
        // The read-back. Without it this function would return while 512
        // dispatches were still queued, and the row would report the time it
        // takes to fill a command buffer.
        return state.toArray ? await state.toArray() : state;
      },
      backend: () => red.kernel && red.kernel.constructor.mode,
      destroy() {
        [red, black, seed].forEach(k => k.destroy && k.destroy());
      },
    };
  },

  /**
   * Hand-written WebGPU. No gpu.js anywhere in it, so the gap between this cell
   * and the WebGPU cell to its left is the runtime's price.
   *
   * Two things differ from the gpu.js column on purpose, and both are the
   * runtime's doing rather than the algorithm's:
   *
   *   - the update is IN PLACE, as the plain-JS oracle's is. Red invocations
   *     write only red sites and read only black ones, so there is no race
   *     inside a dispatch; that is what the colouring buys, and a gpu.js kernel
   *     cannot express it because its output is always a fresh texture.
   *   - all 512 half-passes go into ONE compute pass. WebGPU orders dispatches
   *     within a pass and makes each one's writes visible to the next, so the
   *     sweeps need no per-pass submit.
   *
   * The per-sweep shifts live in one pre-filled uniform buffer, read through a
   * dynamic offset. Rewriting a uniform between dispatches would mean a submit
   * per half-pass, which would measure the queue rather than the lattice.
   */
  async webgpu(device, { n, sweeps }, { spins, rnd }) {
    const cells = n * n;
    const bytes = cells * 4;
    const S = GPUBufferUsage.STORAGE;

    const upload = (data, usage) => {
      const buf = device.createBuffer({ size: data.byteLength, usage, mappedAtCreation: true });
      new Float32Array(buf.getMappedRange()).set(data);
      buf.unmap();
      return buf;
    };
    // Pristine copy, kept on the GPU: each run resets from it with a device-side
    // copy rather than a 4 MB upload, so the timing is the sweeps and not PCIe.
    const bufInit = upload(spins, GPUBufferUsage.COPY_SRC);
    const bufSpins = device.createBuffer({ size: bytes, usage: S | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
    const bufRnd = upload(rnd, S);
    const read = device.createBuffer({ size: bytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });

    // One 256-byte slot per half-pass — 256 is the dynamic-offset alignment.
    const STRIDE = 256;
    const plan = new Uint32Array((sweeps * 2 * STRIDE) / 4);
    for (let t = 0; t < sweeps; t++) {
      for (let parity = 0; parity < 2; parity++) {
        const base = ((t * 2 + parity) * STRIDE) / 4;
        plan[base] = (t * SHIFT_X) & (n - 1);
        plan[base + 1] = (t * SHIFT_Y) & (n - 1);
        plan[base + 2] = parity;
      }
    }
    const bufPlan = device.createBuffer({
      size: plan.byteLength,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(bufPlan, 0, plan);

    const TILE = 16;
    const module = device.createShaderModule({
      code: `
struct Pass { sx: u32, sy: u32, parity: u32, pad: u32 };
@group(0) @binding(0) var<storage, read_write> spins: array<f32>;
@group(0) @binding(1) var<storage, read> rnd: array<f32>;
@group(0) @binding(2) var<uniform> p: Pass;

const N: u32 = ${n}u;
const MASK: u32 = ${n - 1}u;

@compute @workgroup_size(${TILE}, ${TILE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x;
  let y = gid.y;
  if (x >= N || y >= N) { return; }
  // The colouring. Every neighbour of this site has the other parity, so the
  // sites this dispatch writes are disjoint from the ones it reads.
  if (((x + y) & 1u) != p.parity) { return; }

  let i = y * N + x;
  let c = spins[i];
  let left  = select(x - 1u, N - 1u, x == 0u);
  let right = select(x + 1u, 0u, x == N - 1u);
  let up    = select(y - 1u, N - 1u, y == 0u);
  let down  = select(y + 1u, 0u, y == N - 1u);

  let e = c * (spins[up * N + x] + spins[down * N + x] + spins[y * N + left] + spins[y * N + right]);
  if (e <= 0.0) {
    spins[i] = -c;
    return;
  }
  let r = rnd[((y + p.sy) & MASK) * N + ((x + p.sx) & MASK)];
  // Integer thresholds: r and the bound are both exact whole numbers below
  // 2^16, so this comparison cannot disagree with the plain-JS oracle.
  let bound = select(${ACCEPT_8}.0, ${ACCEPT_4}.0, e < 3.0);
  if (r < bound) { spins[i] = -c; }
}`,
    });

    const layout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform', hasDynamicOffset: true } },
      ],
    });
    const pipeline = device.createComputePipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }),
      compute: { module, entryPoint: 'main' },
    });
    const bind = device.createBindGroup({
      layout,
      entries: [
        { binding: 0, resource: { buffer: bufSpins } },
        { binding: 1, resource: { buffer: bufRnd } },
        { binding: 2, resource: { buffer: bufPlan, offset: 0, size: 16 } },
      ],
    });
    const groups = Math.ceil(n / TILE);

    return {
      async run() {
        const enc = device.createCommandEncoder();
        enc.copyBufferToBuffer(bufInit, 0, bufSpins, 0, bytes);
        const pass = enc.beginComputePass();
        pass.setPipeline(pipeline);
        for (let i = 0; i < sweeps * 2; i++) {
          pass.setBindGroup(0, bind, [i * STRIDE]);
          pass.dispatchWorkgroups(groups, groups);
        }
        pass.end();
        enc.copyBufferToBuffer(bufSpins, 0, read, 0, bytes);
        device.queue.submit([enc.finish()]);
        await read.mapAsync(GPUMapMode.READ);
        const out = new Float32Array(read.getMappedRange()).slice();
        read.unmap();
        return out;
      },
      destroy() {
        [bufInit, bufSpins, bufRnd, bufPlan, read].forEach(b => b.destroy && b.destroy());
      },
    };
  },

  /**
   * Index-weighted magnetisation. Every site contributes, and the weight stops
   * a lattice that is right in one half and stale in the other from summing to
   * the same number as the correct one.
   *
   * Spins are ±1 and the weights are small integers, so this sum is an exact
   * integer in every backend's arithmetic. Two columns that ran the same
   * decisions agree to the last bit; a column that differs at all is wrong, and
   * the 1e-4 tolerance the runner applies is slack this row never needs.
   *
   * gpu.js hands back an array of rows and the other two columns hand back a
   * flat array; both are walked here in the same order.
   */
  reduce(out, { n }) {
    let acc = 0;
    let i = 0;
    if (ArrayBuffer.isView(out)) {
      for (; i < out.length; i++) acc += out[i] * (1 + (i % 17));
    } else {
      for (let y = 0; y < out.length; y++) {
        const row = out[y];
        for (let x = 0; x < row.length; x++, i++) acc += row[x] * (1 + (i % 17));
      }
    }
    return acc / (n * n);
  },
};
