/**
 * Separable box blur — arithmetic intensity near the floor.
 *
 * Two passes, horizontal then vertical, radius 8, so seventeen taps and one
 * multiply per output pixel per pass. That is about two floating-point
 * operations for every byte that has to cross a memory bus. `matmul` is two
 * thousand. This row is the other end of the same axis, and it is where most
 * real image work actually lives: convolutions, resamples, colour transforms,
 * compositing — all of them move far more data than they compute on.
 *
 * WHAT THAT MEANS FOR THE NUMBERS. A GPU's advantage over a CPU in raw
 * bandwidth is roughly one order of magnitude; its advantage in raw arithmetic
 * is closer to three. A workload that can only spend bandwidth therefore cannot
 * collect the other two orders no matter how well it is written, and on this
 * row the 16 MB going up and the 16 MB coming back are a large fraction of the
 * whole. Expect a real speed-up and expect it to be small compared to the dense
 * row. If a reader takes one thing from this table, it should be that the gap
 * between those two rows is a property of the problem, not of the programmer.
 *
 * WHY 3072 AND NOT 2048. The natural size for this kernel is 2048 x 2048, but
 * a fair plain-JS baseline for it is about 155 ms on the machine this was
 * written on — under the 200 ms floor the sizing script enforces, which would
 * make the row a measurement of the clock. Radius is the wrong knob to turn to
 * fix that: more taps means more arithmetic per byte moved, which is precisely
 * the property this row is here to hold constant. Image size is the right knob,
 * because it scales the work and the traffic together and leaves the intensity
 * exactly where it was. Hence 3072 x 3072 at radius 8.
 *
 * ON THE BASELINE. The vertical pass in js() walks a whole row of the
 * intermediate at a time and accumulates into a row of the output, instead of
 * gathering seventeen strided taps per pixel the way the kernels do. Both do
 * seventeen adds and one multiply per output pixel — identical arithmetic — but
 * the row-major order is about 45% faster because it streams cache lines
 * instead of jumping seventeen rows for every pixel. That is the same reasoning
 * matmul gives for choosing an i-k-j loop: the baseline gets the good loop
 * order, because a baseline that does not is a gift to every other column.
 */

const W = 3072;
const H = 3072;
const R = 8;
const TAPS = 2 * R + 1;
const INV = 1 / TAPS;

function fill(a, seed) {
  let s = seed >>> 0;
  for (let i = 0; i < a.length; i++) {
    s = (s * 1664525 + 1013904223) >>> 0;
    a[i] = (s >>> 8) / 0x1000000;
  }
  return a;
}

export default {
  id: 'blur-separable',
  name: 'Separable box blur',
  params: `${W} × ${H} fp32, radius ${R}, 2 passes`,
  tag: 'bandwidth-bound',
  group: 'movement',
  size: { w: W, h: H, r: R, taps: TAPS, inv: INV },

  make({ w, h }) {
    return { src: fill(new Float32Array(w * h), 0x6c078965) };
  },

  // The oracle. Interior and edges are split in the horizontal pass so the
  // common case has no clamp at all, and the vertical pass accumulates row by
  // row. Same seventeen taps either way; this order just does not fight the
  // cache. Both scratch buffers are allocated per call rather than kept on the
  // inputs, so nothing here is shared with the columns that follow.
  js({ w, h, r, inv }, { src }) {
    const tmp = new Float32Array(w * h);
    const out = new Float32Array(w * h);

    for (let y = 0; y < h; y++) {
      const row = y * w;
      for (let x = 0; x < r; x++) {
        let s = 0;
        for (let k = -r; k <= r; k++) s += src[row + (x + k < 0 ? 0 : x + k)];
        tmp[row + x] = s * inv;
      }
      for (let x = r; x < w - r; x++) {
        let s = 0;
        for (let k = -r; k <= r; k++) s += src[row + x + k];
        tmp[row + x] = s * inv;
      }
      for (let x = w - r; x < w; x++) {
        let s = 0;
        for (let k = -r; k <= r; k++) s += src[row + (x + k >= w ? w - 1 : x + k)];
        tmp[row + x] = s * inv;
      }
    }

    for (let y = 0; y < h; y++) {
      const orow = y * w;
      for (let k = -r; k <= r; k++) {
        let yy = y + k;
        if (yy < 0) yy = 0;
        else if (yy >= h) yy = h - 1;
        const srow = yy * w;
        for (let x = 0; x < w; x++) out[orow + x] += tmp[srow + x];
      }
      for (let x = 0; x < w; x++) out[orow + x] *= inv;
    }
    return out;
  },

  gpujs(gpu, { w, h, r, taps, inv }, { src }) {
    // 2-D rows, built once. Reshaping inside run() would time the reshape.
    const S = [];
    for (let y = 0; y < h; y++) S.push(src.subarray(y * w, y * w + w));

    const consts = { taps, r, inv, wm1: w - 1, hm1: h - 1 };

    // Pipeline mode on the horizontal pass: the intermediate is 37 MB and has
    // no reason to visit the CPU. Reading it back and uploading it again would
    // triple this row's traffic and turn it into a transfer benchmark.
    const horizontal = gpu
      .createKernel(function (a) {
        let s = 0;
        for (let k = 0; k < this.constants.taps; k++) {
          let xx = this.thread.x + k - this.constants.r;
          if (xx < 0) xx = 0;
          if (xx > this.constants.wm1) xx = this.constants.wm1;
          s += a[this.thread.y][xx];
        }
        return s * this.constants.inv;
      })
      .setConstants(consts)
      .setOutput([w, h])
      .setPipeline(true);

    const vertical = gpu
      .createKernel(function (a) {
        let s = 0;
        for (let k = 0; k < this.constants.taps; k++) {
          let yy = this.thread.y + k - this.constants.r;
          if (yy < 0) yy = 0;
          if (yy > this.constants.hm1) yy = this.constants.hm1;
          s += a[yy][this.thread.x];
        }
        return s * this.constants.inv;
      })
      .setConstants(consts)
      .setOutput([w, h]);

    return {
      async run() {
        const mid = await horizontal(S);
        return await vertical(mid);
      },
      backend: () => vertical.kernel && vertical.kernel.constructor.mode,
      destroy() {
        [horizontal, vertical].forEach(k => k.destroy && k.destroy());
      },
    };
  },

  /**
   * Hand-written WebGPU. Two entry points over one flat storage buffer, both
   * dispatches recorded into a single compute pass — WebGPU orders and
   * synchronises dispatches within a pass, so the vertical pass sees the
   * horizontal pass's writes without a round trip through the host.
   */
  async webgpu(device, { w, h, r, inv }, { src }) {
    const bytes = w * h * 4;
    const S = GPUBufferUsage.STORAGE;
    // Uploaded inside run(), not here. On a row whose entire subject is the
    // cost of moving data, a bare column that started with the image already
    // resident while the gpu.js column beside it re-uploaded 37 MB every run
    // would not be measuring the same thing at all.
    const bufSrc = device.createBuffer({ size: bytes, usage: S | GPUBufferUsage.COPY_DST });
    const bufTmp = device.createBuffer({ size: bytes, usage: S });
    const bufOut = device.createBuffer({ size: bytes, usage: S | GPUBufferUsage.COPY_SRC });
    const read = device.createBuffer({ size: bytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });

    const TILE = 16;
    const module = device.createShaderModule({
      code: `
struct P { w: u32, h: u32, r: u32, inv: f32 };
@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<uniform> p: P;

@compute @workgroup_size(${TILE}, ${TILE})
fn hpass(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= p.w || gid.y >= p.h) { return; }
  let taps = 2u * p.r + 1u;
  let last = i32(p.w) - 1;
  var s = 0.0;
  for (var k: u32 = 0u; k < taps; k = k + 1u) {
    let xx = max(0, min(last, i32(gid.x) + i32(k) - i32(p.r)));
    s = s + src[gid.y * p.w + u32(xx)];
  }
  dst[gid.y * p.w + gid.x] = s * p.inv;
}

@compute @workgroup_size(${TILE}, ${TILE})
fn vpass(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= p.w || gid.y >= p.h) { return; }
  let taps = 2u * p.r + 1u;
  let last = i32(p.h) - 1;
  var s = 0.0;
  for (var k: u32 = 0u; k < taps; k = k + 1u) {
    let yy = max(0, min(last, i32(gid.y) + i32(k) - i32(p.r)));
    s = s + src[u32(yy) * p.w + gid.x];
  }
  dst[gid.y * p.w + gid.x] = s * p.inv;
}`,
    });

    const uni = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(uni, 0, new Uint32Array([w, h, r, 0]));
    device.queue.writeBuffer(uni, 12, new Float32Array([inv]));

    const mkPipe = entryPoint => device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint } });
    const pipeH = mkPipe('hpass');
    const pipeV = mkPipe('vpass');
    const bind = (pipeline, from, to) =>
      device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: from } },
          { binding: 1, resource: { buffer: to } },
          { binding: 2, resource: { buffer: uni } },
        ],
      });
    const bgH = bind(pipeH, bufSrc, bufTmp);
    const bgV = bind(pipeV, bufTmp, bufOut);
    const gx = Math.ceil(w / TILE);
    const gy = Math.ceil(h / TILE);

    return {
      async run() {
        device.queue.writeBuffer(bufSrc, 0, src);
        const enc = device.createCommandEncoder();
        const pass = enc.beginComputePass();
        pass.setPipeline(pipeH);
        pass.setBindGroup(0, bgH);
        pass.dispatchWorkgroups(gx, gy);
        pass.setPipeline(pipeV);
        pass.setBindGroup(0, bgV);
        pass.dispatchWorkgroups(gx, gy);
        pass.end();
        enc.copyBufferToBuffer(bufOut, 0, read, 0, bytes);
        device.queue.submit([enc.finish()]);
        await read.mapAsync(GPUMapMode.READ);
        const out = new Float32Array(read.getMappedRange()).slice();
        read.unmap();
        return out;
      },
      destroy() {
        [bufSrc, bufTmp, bufOut, read, uni].forEach(x => x.destroy && x.destroy());
      },
    };
  },

  // A 2-D gpu.js kernel resolves to an array of rows, plain JS to one flat
  // array, so this walks both shapes rather than flattening — nine million
  // floats is not a thing to build a temporary copy of just to add them up.
  // Index-weighted across the whole image, because a blur that only ran on the
  // top half would otherwise be indistinguishable from one that ran everywhere.
  reduce(out, { w, h }) {
    let acc = 0;
    if (ArrayBuffer.isView(out)) {
      for (let i = 0; i < out.length; i++) acc += out[i] * (1 + (i % 17));
    } else {
      let i = 0;
      for (let y = 0; y < out.length; y++) {
        const row = out[y];
        for (let x = 0; x < row.length; x++, i++) acc += row[x] * (1 + (i % 17));
      }
    }
    return acc / (w * h);
  },
};
