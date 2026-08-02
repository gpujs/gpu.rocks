/**
 * All-pairs gravity — one force evaluation over 16384 bodies.
 *
 * Matmul is the case a GPU is built for: the operands are read in a regular,
 * shareable pattern and every value is reused n times. This row is the other
 * shape of the same n^2 story. Every thread reads EVERY body, so the gather is
 * total — 16384 threads each stream 16384 * 4 floats, and nothing a thread
 * reads belongs to it. That is a broadcast, not a tile. It is worth sitting
 * next to matmul because the two are indistinguishable on a flop counter and
 * look nothing alike in the memory system.
 *
 * The arithmetic is a Plummer-softened Newtonian sum:
 *
 *     a_i = G * sum_j  m_j * (r_j - r_i) / (|r_j - r_i|^2 + eps^2)^(3/2)
 *
 * and what each thread writes is |a_i|. The softening is what makes this
 * measurable rather than a lottery: without eps^2 the j == i term is 0/0, and
 * near-coincident pairs produce accelerations whose fp32 and fp64 values have
 * nothing to do with each other. With eps = 0.1 — about one body spacing — and
 * bodies spread over a unit cube, the denominator never approaches zero and
 * every backend evaluates the same well-conditioned expression.
 *
 * WHY THERE IS NO INTEGRATION STEP. The first draft did the leapfrog kick,
 * v' = v + G*dt*a, and returned |v'|. That is three flops on top of an n^2 sum,
 * so it cost nothing to run — but it wrecked the checksum. A kick whose
 * direction is uncorrelated with v changes the MEAN speed only at second order:
 * with the kick at 5% of |v|, the mean moved by 0.13%, so a backend that had
 * computed no forces at all would have landed within 0.13% of the right answer.
 * Still caught by the 1e-4 tolerance, but only 13x clear of it — a checksum that
 * barely notices the one thing the row spends all its time on. |a_i| is
 * first-order in the sum, strictly positive, and 100% of it is the gather.
 *
 * WHY THE OUTPUT IS A SCALAR AND NOT A VECTOR. A gpu.js kernel returns one
 * number per thread. Three components would mean either a vec3 return type that
 * is not equally supported across webgl/webgl2/webgpu/cpu, or three passes —
 * and three passes would triple the O(n^2) gather, which is the entire cost of
 * the row. All three components are accumulated inside the thread; only the
 * store is collapsed.
 *
 * FP32 VS FP64. Each acceleration is a sum of 16384 signed terms that largely
 * cancel, which is exactly where a narrow mantissa usually bites. It survives
 * for a reason worth stating: naive summation of N terms in fp32 carries a
 * rounding error of roughly eps32*sqrt(N/2) RELATIVE TO THE SUM when the
 * partial sums random-walk — about 5e-6 here. The cancellation is already
 * priced into that, because the partial sums are themselves small. The checksum
 * then averages over 16384 bodies whose errors are signed and independent, so
 * it lands far below even that. Measured against an operation-by-operation
 * fp32-emulated baseline the checksum moves by 1.2e-8, a ~8000x margin on the
 * runner's 1e-4.
 *
 * SIZE. 16384 bodies rather than the more usual 8192: at 8192 the plain-JS
 * baseline is 141 ms on the development machine, under the 200 ms floor the
 * sizing rule sets, and a row that short is measuring the clock as much as the
 * kernel. n^2 meant one doubling of n was enough.
 */

const N = 16384;
const SOFT2 = 0.01; // eps^2, eps = 0.1 — about one body spacing
const G = 1; // units: total mass is 1 and the cube is 2 across, so |a| is O(1)

// Deterministic, and not Math.random: every column must be handed the same
// bytes for their checksums to mean anything.
function lcg(seed) {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return (s >>> 8) / 0x1000000;
  };
}

export default {
  id: 'nbody',
  name: 'All-pairs gravity',
  params: `${N} bodies, all pairs, fp32`,
  tag: 'O(n^2) gather',
  group: 'sim',
  size: { n: N, soft2: SOFT2, g: G },

  // Structure-of-arrays, which is what the JS baseline and a GPU both want: the
  // inner loop walks x, y, z and m forwards, one stream each.
  make({ n }) {
    const rnd = lcg(0xc2b2ae35);
    const x = new Float32Array(n);
    const y = new Float32Array(n);
    const z = new Float32Array(n);
    const m = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      x[i] = rnd() * 2 - 1;
      y[i] = rnd() * 2 - 1;
      z[i] = rnd() * 2 - 1;
      m[i] = (0.5 + rnd()) / n; // masses in [0.5, 1.5]/n — total mass ~ 1
    }
    return { x, y, z, m };
  },

  // The oracle. Flat typed arrays, the body's own coordinates hoisted out of the
  // inner loop, and one reciprocal per pair rather than three divides — the
  // shape any competent hand-written version has. A baseline that re-read x[i]
  // 16384 times, or divided three times per pair, would be roughly twice as
  // slow and would hand every GPU column a free doubling.
  js({ n, soft2, g }, { x, y, z, m }) {
    const out = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      const xi = x[i];
      const yi = y[i];
      const zi = z[i];
      let ax = 0;
      let ay = 0;
      let az = 0;
      for (let j = 0; j < n; j++) {
        const dx = x[j] - xi;
        const dy = y[j] - yi;
        const dz = z[j] - zi;
        const r2 = dx * dx + dy * dy + dz * dz + soft2;
        // j === i falls out on its own: dx = dy = dz = 0 makes the term zero,
        // and the softened denominator keeps it finite. No branch in the hot
        // loop, and so no branch for the GPU to diverge on either.
        const inv = m[j] / (r2 * Math.sqrt(r2));
        ax += dx * inv;
        ay += dy * inv;
        az += dz * inv;
      }
      out[i] = g * Math.sqrt(ax * ax + ay * ay + az * az);
    }
    return out;
  },

  gpujs(gpu, { n, soft2, g }, { x, y, z, m }) {
    const kernel = gpu
      .createKernel(function (px, py, pz, pm) {
        const xi = px[this.thread.x];
        const yi = py[this.thread.x];
        const zi = pz[this.thread.x];
        let ax = 0;
        let ay = 0;
        let az = 0;
        for (let j = 0; j < this.constants.n; j++) {
          const dx = px[j] - xi;
          const dy = py[j] - yi;
          const dz = pz[j] - zi;
          const r2 = dx * dx + dy * dy + dz * dz + this.constants.soft2;
          const inv = pm[j] / (r2 * Math.sqrt(r2));
          ax += dx * inv;
          ay += dy * inv;
          az += dz * inv;
        }
        return this.constants.g * Math.sqrt(ax * ax + ay * ay + az * az);
      })
      .setConstants({ n, soft2, g })
      .setOutput([n]);

    return {
      // await, not fire-and-forget: on the WebGPU backend the result is a
      // promise, and returning early would time the dispatch call, not the sum.
      async run() {
        return await kernel(x, y, z, m);
      },
      backend: () => kernel.kernel && kernel.kernel.constructor.mode,
      destroy: () => kernel.destroy && kernel.destroy(),
    };
  },

  /**
   * Hand-written WebGPU, borrowing nothing from gpu.js.
   *
   * Deliberately the SAME algorithm as the columns to its left: one thread per
   * body, one flat loop over every body, no workgroup-memory tiling. Staging a
   * block of bodies in workgroup memory so the block is read once per workgroup
   * instead of once per thread is THE n-body optimisation and is worth a large
   * factor here. It is left out on purpose: this column exists to price the
   * gpu.js runtime, and an optimisation gpu.js cannot express would quietly
   * fold "we wrote a better kernel" into that price.
   *
   * The four input arrays are packed into one buffer purely to keep the binding
   * count low; each component still occupies a contiguous block, so the shader
   * performs exactly the structure-of-arrays walk the JS baseline performs.
   */
  async webgpu(device, { n, soft2, g }, { x, y, z, m }) {
    const pos = new Float32Array(n * 4);
    pos.set(x, 0);
    pos.set(y, n);
    pos.set(z, 2 * n);
    pos.set(m, 3 * n);

    const bufPos = device.createBuffer({
      size: pos.byteLength,
      usage: GPUBufferUsage.STORAGE,
      mappedAtCreation: true,
    });
    new Float32Array(bufPos.getMappedRange()).set(pos);
    bufPos.unmap();

    const outBytes = n * 4;
    const bufOut = device.createBuffer({
      size: outBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    const read = device.createBuffer({
      size: outBytes,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    // n, eps^2 and G are constants of the workload, so they are baked into the
    // source rather than bound as a uniform: one fewer binding, and the shader
    // compiler gets to see the trip count.
    const WG = 64;
    const module = device.createShaderModule({
      code: `
const N: u32 = ${n}u;
const SOFT2: f32 = ${soft2};
const G: f32 = ${g};

@group(0) @binding(0) var<storage, read> pos: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= N) { return; }
  let xi = pos[i];
  let yi = pos[N + i];
  let zi = pos[2u * N + i];
  var ax = 0.0;
  var ay = 0.0;
  var az = 0.0;
  for (var j: u32 = 0u; j < N; j = j + 1u) {
    let dx = pos[j] - xi;
    let dy = pos[N + j] - yi;
    let dz = pos[2u * N + j] - zi;
    let r2 = dx * dx + dy * dy + dz * dz + SOFT2;
    // sqrt() and a divide, not inverseSqrt(): inverseSqrt is allowed a far
    // looser error bound, and the baseline computes m / (r2 * sqrt(r2)).
    let inv = pos[3u * N + j] / (r2 * sqrt(r2));
    ax = ax + dx * inv;
    ay = ay + dy * inv;
    az = az + dz * inv;
  }
  out[i] = G * sqrt(ax * ax + ay * ay + az * az);
}`,
    });

    const pipeline = device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: 'main' } });
    const bind = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: bufPos } },
        { binding: 1, resource: { buffer: bufOut } },
      ],
    });
    const groups = Math.ceil(n / WG);

    return {
      async run() {
        const enc = device.createCommandEncoder();
        const pass = enc.beginComputePass();
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bind);
        pass.dispatchWorkgroups(groups);
        pass.end();
        enc.copyBufferToBuffer(bufOut, 0, read, 0, outBytes);
        device.queue.submit([enc.finish()]);
        // The map is the only thing that proves the dispatch actually ran.
        await read.mapAsync(GPUMapMode.READ);
        const out = new Float32Array(read.getMappedRange()).slice();
        read.unmap();
        return out;
      },
      destroy() {
        [bufPos, bufOut, read].forEach(b => b.destroy && b.destroy());
      },
    };
  },

  // Every body, index-weighted so a backend that filled half the output cannot
  // match by luck, and averaged so the number stays O(1) — a raw sum of 16384
  // magnitudes would put the fp32 and fp64 totals further apart than the value
  // being compared. Accelerations are strictly positive, so the checksum sits
  // near 4.4 and the runner's RELATIVE tolerance means what it says.
  reduce(out, { n }) {
    const flat = ArrayBuffer.isView(out) ? out : Float32Array.from(out);
    let acc = 0;
    for (let i = 0; i < flat.length; i++) acc += flat[i] * (1 + (i % 17));
    return acc / n;
  },
};
