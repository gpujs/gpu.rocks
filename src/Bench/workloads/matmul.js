/**
 * Dense matrix multiply — the golden example.
 *
 * This is the reference every other workload is written against, so it is
 * deliberately the plainest possible case: no textures, no pipelines, one
 * kernel, one output, and arithmetic a reader can check by eye.
 *
 * It is also the honest headline. Dense fp32 matmul is where a GPU looks best —
 * n^3 multiply-adds over n^2 memory, so the arithmetic intensity is high enough
 * that bandwidth stops being the limit. Every GPU claim starts here, which is
 * exactly why the table should not stop here.
 */

const N = 1024;

// Deterministic and cheap. Not Math.random: two columns must be handed the same
// bytes, and a seeded generator is the only way to say that and mean it.
function fill(a, seed) {
  let s = seed >>> 0;
  for (let i = 0; i < a.length; i++) {
    s = (s * 1664525 + 1013904223) >>> 0;
    a[i] = (s >>> 8) / 0x1000000 - 0.5;
  }
  return a;
}

export default {
  id: 'matmul',
  name: 'Dense matrix multiply',
  params: `${N} × ${N} · ${N} × ${N}, fp32`,
  tag: 'arithmetic intensity',
  group: 'dense',
  size: { n: N },

  make({ n }) {
    return { a: fill(new Float32Array(n * n), 0x9e3779b9), b: fill(new Float32Array(n * n), 0x85ebca6b) };
  },

  // The oracle. Flat typed arrays and an i-k-j loop order, which keeps the
  // inner loop walking both b and out forwards — a j-k order is three times
  // slower on the same arithmetic and would flatter every GPU column.
  js({ n }, { a, b }) {
    const out = new Float32Array(n * n);
    for (let i = 0; i < n; i++) {
      const ai = i * n;
      const oi = i * n;
      for (let k = 0; k < n; k++) {
        const aik = a[ai + k];
        if (aik === 0) continue;
        const bk = k * n;
        for (let j = 0; j < n; j++) out[oi + j] += aik * b[bk + j];
      }
    }
    return out;
  },

  gpujs(gpu, { n }, { a, b }) {
    // 2-D arrays, because that is what a gpu.js kernel indexes. Built once here
    // rather than per run: reshaping inside run() would time the reshape.
    const A = [];
    const B = [];
    for (let i = 0; i < n; i++) {
      A.push(a.subarray(i * n, i * n + n));
      B.push(b.subarray(i * n, i * n + n));
    }
    const kernel = gpu
      .createKernel(function (m, p) {
        let sum = 0;
        for (let k = 0; k < this.constants.n; k++) {
          sum += m[this.thread.y][k] * p[k][this.thread.x];
        }
        return sum;
      })
      .setConstants({ n })
      .setOutput([n, n]);

    return {
      // await, not fire-and-forget: on the async/WebGPU path the result is a
      // promise, and returning before it settles would time the dispatch call
      // rather than the multiply.
      async run() {
        return await kernel(A, B);
      },
      backend: () => kernel.kernel && kernel.kernel.constructor.mode,
      destroy: () => kernel.destroy && kernel.destroy(),
    };
  },

  /**
   * Hand-written WebGPU, with no gpu.js anywhere in it. This column exists to
   * separate "the GPU is fast" from "gpu.js is fast", so it must not borrow so
   * much as a helper — the difference between this cell and the WebGPU cell to
   * its left IS the runtime's price.
   */
  async webgpu(device, { n }, { a, b }) {
    const bytes = n * n * 4;
    const mk = (data, usage) => {
      const buf = device.createBuffer({ size: bytes, usage, mappedAtCreation: Boolean(data) });
      if (data) {
        new Float32Array(buf.getMappedRange()).set(data);
        buf.unmap();
      }
      return buf;
    };
    const S = GPUBufferUsage.STORAGE;
    const bufA = mk(a, S | GPUBufferUsage.COPY_DST);
    const bufB = mk(b, S | GPUBufferUsage.COPY_DST);
    const bufOut = mk(null, S | GPUBufferUsage.COPY_SRC);
    const read = device.createBuffer({ size: bytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });

    const TILE = 16;
    const module = device.createShaderModule({
      code: `
struct Dim { n: u32 };
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> dim: Dim;

@compute @workgroup_size(${TILE}, ${TILE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let n = dim.n;
  if (gid.x >= n || gid.y >= n) { return; }
  var sum = 0.0;
  for (var k: u32 = 0u; k < n; k = k + 1u) {
    sum = sum + a[gid.y * n + k] * b[k * n + gid.x];
  }
  out[gid.y * n + gid.x] = sum;
}`,
    });

    const dim = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(dim, 0, new Uint32Array([n, 0, 0, 0]));

    const pipeline = device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: 'main' } });
    const bind = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: bufA } },
        { binding: 1, resource: { buffer: bufB } },
        { binding: 2, resource: { buffer: bufOut } },
        { binding: 3, resource: { buffer: dim } },
      ],
    });
    const groups = Math.ceil(n / TILE);

    return {
      async run() {
        const enc = device.createCommandEncoder();
        const pass = enc.beginComputePass();
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bind);
        pass.dispatchWorkgroups(groups, groups);
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
        [bufA, bufB, bufOut, read, dim].forEach(x => x.destroy && x.destroy());
      },
    };
  },

  // Index-weighted so a backend that fills only part of the output cannot match
  // it by luck, and scaled down so the sum of a million floats stays in a range
  // where fp32 and fp64 still agree to the tolerance the runner applies.
  reduce(out, { n }) {
    // Flatten row by row. Not [].concat(...out): concat only spreads real
    // Arrays, so Float32Array rows (what gpu.js returns for a 2-D kernel) ride
    // through as single elements and Float32Array.from turns each into NaN.
    let flat;
    if (ArrayBuffer.isView(out)) {
      flat = out;
    } else {
      flat = new Float32Array(out.length * out[0].length);
      for (let i = 0; i < out.length; i++) flat.set(out[i], i * out[0].length);
    }
    let acc = 0;
    for (let i = 0; i < flat.length; i++) acc += flat[i] * (1 + (i % 17));
    return acc / flat.length;
  },
};
