/**
 * Gray–Scott reaction–diffusion — two fields that need each other.
 *
 * Life next door is one buffer ping-ponged; this is two, and the two are
 * coupled: the u update needs v and the v update needs u, both at the OLD time
 * level. So a step cannot be split into "do u, then do v" in place — u would
 * have moved on before v read it — and it cannot be fused into one buffer
 * either. Four textures, alternating in pairs, every step.
 *
 * That coupling is where the runtimes separate. A gpu.js kernel returns exactly
 * one value per thread, so a step is two dispatches that each load both fields:
 * the same neighbourhood is fetched twice. A hand-written WGSL kernel binds two
 * output buffers and writes both fields from one dispatch, halving both the
 * dispatch count and the loads. Identical arithmetic, and the difference
 * between those two cells is a straightforward statement of what the runtime's
 * one-output-per-kernel model costs on a coupled system.
 *
 * ── WHY THIS ROW IS STILL CHECKABLE AFTER THOUSANDS OF STEPS ────────────────
 *
 * An iterated float simulation is the easiest way to make a benchmark row that
 * cannot be verified: run something mildly chaotic for long enough and fp32 and
 * fp64 disagree in the fourth digit, at which point every GPU column is
 * "WRONG" and the row says nothing. Two choices avoid that here.
 *
 * STABILITY. Explicit Euler on a 5-point Laplacian is stable only while
 * D*dt/h^2 <= 1/4 in two dimensions. With h = 1, dt = 1 and the larger
 * diffusion rate Du = 0.16, that ratio is 0.16 — inside the limit with room to
 * spare, and Dv = 0.08 is further inside it. Push dt past 1.5 here and the
 * scheme does not merely lose accuracy, it blows up, differently on each
 * backend and at a different step.
 *
 * CONTRACTION. Stability is not enough on its own: a stable scheme can still be
 * chaotic, and then a one-ulp difference grows exponentially. So this was
 * measured rather than assumed. Running the same simulation with fp64
 * intermediates and with every intermediate rounded to fp32 — which is what a
 * GPU does — the two checksums differ by 2e-9 after 64 steps and 2e-7 after
 * 2048, growing roughly linearly rather than exponentially. F = 0.037,
 * k = 0.060 is a growth regime, not a turbulent one; perturbations get damped
 * by the F*(1-u) feed term rather than amplified. 2e-7 is three orders of
 * magnitude inside the runner's 1e-4 tolerance, so a column that disagrees here
 * disagrees because it is wrong.
 */

const N = 1024;

// 384, not the 2048 this model is usually run for. At 1024 x 1024 a plain-JS
// step is about 5 ms, so 2048 steps is a ten-second baseline — three times the
// top of the sizing band. The grid is what makes the row a real bandwidth test
// (four 4 MB fields, none of which fits in cache), so the step count is what
// gives. 384 steps is still hundreds of dependent dispatches, which is the
// shape this row is here to price, and fewer steps only makes the fp32/fp64
// agreement above better.
const STEPS = 384;

// Diffusion rates, feed and kill. Du*dt = 0.16 <= 0.25, the 2-D explicit-Euler
// stability limit for a 5-point Laplacian on a unit grid. See the header.
const DU = 0.16;
const DV = 0.08;
const FEED = 0.037;
const KILL = 0.06;
const DT = 1.0;

export default {
  id: 'gray-scott',
  name: 'Gray–Scott reaction–diffusion',
  params: `${N} × ${N} · ${STEPS} steps · F = ${FEED}, k = ${KILL}, dt = ${DT}`,
  tag: 'coupled ping-pong',
  group: 'sim',
  size: { n: N, steps: STEPS },

  make({ n }) {
    // The classic seeding: u saturated everywhere, v absent, and a scatter of
    // square patches where the reaction is kicked off. Seeded, because two
    // columns have to start from the same field for their checksums to mean
    // anything.
    const u = new Float32Array(n * n).fill(1);
    const v = new Float32Array(n * n);
    let s = 0x2545f491 >>> 0;
    const rnd = () => {
      s = (s * 1664525 + 1013904223) >>> 0;
      return (s >>> 8) / 0x1000000;
    };
    for (let p = 0; p < 60; p++) {
      const cx = Math.floor(rnd() * n);
      const cy = Math.floor(rnd() * n);
      const r = 6 + Math.floor(rnd() * 12);
      for (let y = cy - r; y <= cy + r; y++) {
        for (let x = cx - r; x <= cx + r; x++) {
          const i = ((y + n) % n) * n + ((x + n) % n);
          u[i] = 0.5;
          v[i] = 0.25;
        }
      }
    }

    // Row views for gpu.js's 2-D indexing. Views, not copies.
    const uRows = [];
    const vRows = [];
    for (let y = 0; y < n; y++) {
      uRows.push(u.subarray(y * n, y * n + n));
      vRows.push(v.subarray(y * n, y * n + n));
    }
    return { u, v, uRows, vRows };
  },

  /**
   * The oracle. Four flat Float32Arrays swapped by reference, row bases hoisted
   * out of the inner loop, and both fields updated in the same pass so the
   * neighbourhood is loaded once — which is exactly what the hand-written WGSL
   * kernel does and what a gpu.js kernel cannot.
   *
   * The state is Float32Array rather than Float64Array deliberately. Every
   * backend then rounds to fp32 at the same points in the iteration, so the
   * only precision difference left is in the intermediates, which is the 2e-7
   * measured in the header.
   */
  js({ n, steps }, { u, v }) {
    let cu = new Float32Array(u); // every run starts from the same field
    let cv = new Float32Array(v);
    let nu = new Float32Array(n * n);
    let nv = new Float32Array(n * n);

    for (let s = 0; s < steps; s++) {
      for (let y = 0; y < n; y++) {
        const up = (y === 0 ? n - 1 : y - 1) * n;
        const row = y * n;
        const down = (y === n - 1 ? 0 : y + 1) * n;
        for (let x = 0; x < n; x++) {
          const left = x === 0 ? n - 1 : x - 1;
          const right = x === n - 1 ? 0 : x + 1;
          const i = row + x;
          const a = cu[i];
          const b = cv[i];
          const lapU = cu[up + x] + cu[down + x] + cu[row + left] + cu[row + right] - 4 * a;
          const lapV = cv[up + x] + cv[down + x] + cv[row + left] + cv[row + right] - 4 * b;
          const uvv = a * b * b;
          nu[i] = a + DT * (DU * lapU - uvv + FEED * (1 - a));
          nv[i] = b + DT * (DV * lapV + uvv - (FEED + KILL) * b);
        }
      }
      let t = cu;
      cu = nu;
      nu = t;
      t = cv;
      cv = nv;
      nv = t;
    }
    return { u: cu, v: cv };
  },

  gpujs(gpu, { n, steps }, { uRows, vRows }) {
    const constants = { n, du: DU, dv: DV, feed: FEED, kill: KILL, dt: DT };

    // Two kernels per field, not one each. A pipelined mutable kernel reuses a
    // single output texture, so the even step's u kernel and the odd step's u
    // kernel must be different objects — otherwise a kernel would be asked to
    // write the texture it is reading. Four kernels, alternating in pairs, is
    // the ping-pong for a coupled pair.
    const uStep = () =>
      gpu
        .createKernel(function (u, v) {
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

          const a = u[y][x];
          const b = v[y][x];
          const lap = u[up][x] + u[down][x] + u[y][left] + u[y][right] - 4 * a;
          return a + this.constants.dt * (this.constants.du * lap - a * b * b + this.constants.feed * (1 - a));
        })
        .setConstants(constants)
        .setPipeline(true)
        .setPrecision('single')
        .setOutput([n, n]);

    const vStep = () =>
      gpu
        .createKernel(function (u, v) {
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

          const a = u[y][x];
          const b = v[y][x];
          const lap = v[up][x] + v[down][x] + v[y][left] + v[y][right] - 4 * b;
          return (
            b +
            this.constants.dt *
              (this.constants.dv * lap + a * b * b - (this.constants.feed + this.constants.kill) * b)
          );
        })
        .setConstants(constants)
        .setPipeline(true)
        .setPrecision('single')
        .setOutput([n, n]);

    const uEven = uStep();
    const uOdd = uStep();
    const vEven = vStep();
    const vOdd = vStep();

    // Uploads the initial fields once per run. Its real job is to keep the step
    // kernels' argument types constant — a plain array on step 0 and a Texture
    // afterwards would make gpu.js recompile inside the timed region.
    const copy = () =>
      gpu
        .createKernel(function (a) {
          return a[this.thread.y][this.thread.x];
        })
        .setPipeline(true)
        .setPrecision('single')
        .setOutput([n, n]);
    const seedU = copy();
    const seedV = copy();

    return {
      async run() {
        let u = await seedU(uRows);
        let v = await seedV(vRows);
        for (let s = 0; s < steps; s++) {
          const even = s % 2 === 0;
          // Both reads happen against the OLD pair, which is why the new u is
          // not assigned until the new v has been dispatched.
          const nu = await (even ? uEven : uOdd)(u, v);
          const nv = await (even ? vEven : vOdd)(u, v);
          u = nu;
          v = nv;
        }
        // The read-backs. Until these resolve, the dispatches are only queued.
        return {
          u: u.toArray ? await u.toArray() : u,
          v: v.toArray ? await v.toArray() : v,
        };
      },
      backend: () => uEven.kernel && uEven.kernel.constructor.mode,
      destroy() {
        [uEven, uOdd, vEven, vOdd, seedU, seedV].forEach(k => k.destroy && k.destroy());
      },
    };
  },

  /**
   * Hand-written WebGPU, with nothing borrowed from gpu.js.
   *
   * The one structural difference from the column to its left, and it is the
   * point of this column: ONE kernel writes BOTH fields. A compute shader can
   * bind two read-write storage buffers, so the ten loads a step needs are done
   * once instead of twice and there is one dispatch per step instead of two.
   * The arithmetic is line-for-line the same as the two gpu.js kernels.
   *
   * The fields stay in separate buffers rather than being interleaved into a
   * vec2, which would be faster again — that would change the memory layout
   * relative to every other column, and this cell is meant to isolate the
   * runtime's cost, not to win.
   */
  async webgpu(device, { n, steps }, { u, v }) {
    const cells = n * n;
    const bytes = cells * 4;
    const S = GPUBufferUsage.STORAGE;

    const upload = (data, usage) => {
      const buf = device.createBuffer({ size: bytes, usage, mappedAtCreation: true });
      new Float32Array(buf.getMappedRange()).set(data);
      buf.unmap();
      return buf;
    };
    // Pristine copies kept on the device, so a run resets with a device-side
    // copy rather than 8 MB of upload inside the timed region.
    const initU = upload(u, GPUBufferUsage.COPY_SRC);
    const initV = upload(v, GPUBufferUsage.COPY_SRC);
    const mk = () => device.createBuffer({ size: bytes, usage: S | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
    const uA = mk();
    const vA = mk();
    const uB = mk();
    const vB = mk();
    const readU = device.createBuffer({ size: bytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const readV = device.createBuffer({ size: bytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });

    const TILE = 16;
    const module = device.createShaderModule({
      code: `
@group(0) @binding(0) var<storage, read> uSrc: array<f32>;
@group(0) @binding(1) var<storage, read> vSrc: array<f32>;
@group(0) @binding(2) var<storage, read_write> uDst: array<f32>;
@group(0) @binding(3) var<storage, read_write> vDst: array<f32>;

const N: u32 = ${n}u;
const DU: f32 = ${DU};
const DV: f32 = ${DV};
const FEED: f32 = ${FEED};
const KILL: f32 = ${KILL};
const DT: f32 = ${DT};

@compute @workgroup_size(${TILE}, ${TILE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x;
  let y = gid.y;
  if (x >= N || y >= N) { return; }

  let left  = select(x - 1u, N - 1u, x == 0u);
  let right = select(x + 1u, 0u, x == N - 1u);
  let up    = select(y - 1u, N - 1u, y == 0u) * N;
  let down  = select(y + 1u, 0u, y == N - 1u) * N;
  let row   = y * N;
  let i     = row + x;

  let a = uSrc[i];
  let b = vSrc[i];
  let lapU = uSrc[up + x] + uSrc[down + x] + uSrc[row + left] + uSrc[row + right] - 4.0 * a;
  let lapV = vSrc[up + x] + vSrc[down + x] + vSrc[row + left] + vSrc[row + right] - 4.0 * b;
  let uvv = a * b * b;

  uDst[i] = a + DT * (DU * lapU - uvv + FEED * (1.0 - a));
  vDst[i] = b + DT * (DV * lapV + uvv - (FEED + KILL) * b);
}`,
    });

    const pipeline = device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: 'main' } });
    const layout = pipeline.getBindGroupLayout(0);
    const bindGroup = (us, vs, ud, vd) =>
      device.createBindGroup({
        layout,
        entries: [
          { binding: 0, resource: { buffer: us } },
          { binding: 1, resource: { buffer: vs } },
          { binding: 2, resource: { buffer: ud } },
          { binding: 3, resource: { buffer: vd } },
        ],
      });
    const aToB = bindGroup(uA, vA, uB, vB);
    const bToA = bindGroup(uB, vB, uA, vA);
    const groups = Math.ceil(n / TILE);
    const finalU = steps % 2 === 0 ? uA : uB;
    const finalV = steps % 2 === 0 ? vA : vB;

    return {
      async run() {
        const enc = device.createCommandEncoder();
        enc.copyBufferToBuffer(initU, 0, uA, 0, bytes);
        enc.copyBufferToBuffer(initV, 0, vA, 0, bytes);
        const pass = enc.beginComputePass();
        pass.setPipeline(pipeline);
        for (let s = 0; s < steps; s++) {
          // Dispatches inside one compute pass are ordered and each one's
          // writes are visible to the next, so the whole simulation is a single
          // submit rather than one per step.
          pass.setBindGroup(0, s % 2 === 0 ? aToB : bToA);
          pass.dispatchWorkgroups(groups, groups);
        }
        pass.end();
        enc.copyBufferToBuffer(finalU, 0, readU, 0, bytes);
        enc.copyBufferToBuffer(finalV, 0, readV, 0, bytes);
        device.queue.submit([enc.finish()]);
        await Promise.all([readU.mapAsync(GPUMapMode.READ), readV.mapAsync(GPUMapMode.READ)]);
        const outU = new Float32Array(readU.getMappedRange()).slice();
        const outV = new Float32Array(readV.getMappedRange()).slice();
        readU.unmap();
        readV.unmap();
        return { u: outU, v: outV };
      },
      destroy() {
        [initU, initV, uA, vA, uB, vB, readU, readV].forEach(b => b.destroy && b.destroy());
      },
    };
  },

  /**
   * Both fields, every cell, index-weighted so that a field which is right in
   * one region and stale in another cannot land on the correct total.
   *
   * The two fields get different weight periods, so a backend that swapped them
   * — which a coupled system makes an easy mistake — does not pass. gpu.js
   * hands back rows and the other two columns hand back flat arrays; both are
   * walked here in the same order, and the result is divided by the cell count
   * so a million-term sum stays in a range where fp32 and fp64 agree far inside
   * the runner's tolerance.
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
    return (fold(out.u, 17) + fold(out.v, 13)) / (n * n);
  },
};
