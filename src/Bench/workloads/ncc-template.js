/**
 * Normalised cross-correlation of a template over an image — read amplification.
 *
 * Slide a 64 x 64 patch over every valid position of a 512 x 512 image and score
 * the match. 449 x 449 positions, 4096 samples each: 826 million reads out of an
 * image that is one megabyte. Every pixel of that megabyte is read about three
 * thousand times, by three thousand different output positions, none of which
 * knows about the others.
 *
 * That ratio is the whole point of the row. `matmul` also reuses its inputs
 * heavily, but it reuses them in a shape a tiled kernel can exploit; template
 * matching reuses them in overlapping windows that slide by one pixel, so the
 * reuse is entirely the memory system's problem. A CPU handles it with an L1
 * cache and a well-behaved stride. A GPU handles it with a texture cache and a
 * thousand threads whose windows overlap almost completely. Both are doing the
 * same arithmetic; what differs is which piece of hardware absorbs the reads.
 *
 * ── THE SIZE, AND WHY IT IS NOT 2048 x 2048 ────────────────────────────────
 *
 * The natural framing of this workload is a 64 x 64 template over a 2048 x 2048
 * image. That is 1985^2 x 4096 = 1.6e10 multiply-accumulates, which is about
 * twenty seconds of plain JS — six times the top of the sizing band and a cell
 * nobody would wait for. The template is the wrong knob to shrink, because the
 * template size IS the amplification factor and shrinking it is exactly the
 * property this row exists to hold. So the image shrinks instead: 512 x 512
 * keeps the 4096-sample window, keeps the ~3000x reuse of every input pixel,
 * and lands the baseline near a second.
 *
 * ── THE FORMULA, AND WHY IT IS WRITTEN THIS WAY ────────────────────────────
 *
 *   ncc = sum((I - Ibar)(T - Tbar)) / sqrt(sum((I - Ibar)^2) sum((T - Tbar)^2))
 *
 * The template is mean-subtracted once, in make(), so sum(T - Tbar) is zero and
 * the numerator collapses to sum(I * t'): no second pass over the window to find
 * the window mean first. The denominator needs the window's variance, which
 * comes from the same single pass as sum(I^2) - (sum I)^2/n.
 *
 * That form has a well-known failure — catastrophic cancellation when the mean
 * is large compared to the spread — so the image is generated with zero mean
 * and unit-ish spread. With a DC offset of, say, 1000, sum(I)^2/n and sum(I^2)
 * would agree to five digits and the fp32 columns would disagree with the fp64
 * baseline for reasons that have nothing whatever to do with the GPU. The
 * benchmark would then be reporting a numerical-methods choice as a hardware
 * result, which is the kind of lie this table is built to avoid.
 *
 * ── CHECKSUM ────────────────────────────────────────────────────────────────
 *
 * No branches, no ties, no argmax: every output is a smooth function of its
 * window, so nothing here can be decided differently by two backends. What is
 * left is ordinary rounding — a 4096-term sum accumulated in fp32 on the GPU
 * and fp64 in the baseline, which random-walks to about 4e-6 relative — and the
 * checksum averages 200k of those, so it lands three or four orders inside the
 * runner's 1e-4. The one guard that IS a branch, `den > EPS`, only fires on a
 * perfectly flat window, and this image has none; it is there so the row cannot
 * emit a NaN if the image generator is ever changed.
 */

const W = 512;
const H = 512;
const T = 64;
const OW = W - T + 1;
const OH = H - T + 1;

// 1/4096. A power of two, so multiplying by it is exact everywhere and the
// baseline and the kernels cannot disagree about the window mean. Written as a
// multiply rather than a divide because gpu.js's WebGL backends route division
// through an accuracy wrapper on some drivers.
const INV_N = 1 / (T * T);

// Only reached by a window with literally zero variance. See the header.
const EPS = 1e-12;

/**
 * A zero-mean image with structure at several scales: smooth bands, a handful
 * of blobs, and broadband noise. The structure matters — a pure-noise image
 * makes every window statistically identical, so the correlation surface is
 * flat and a backend that got the arithmetic subtly wrong would still land on
 * a plausible-looking checksum.
 */
function image(w, h, seed) {
  const a = new Float32Array(w * h);
  let s = seed >>> 0;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      s = (s * 1664525 + 1013904223) >>> 0;
      const r = (s >>> 8) / 0x1000000 - 0.5;
      a[y * w + x] =
        0.30 * Math.sin(x * 0.05 + y * 0.02) +
        0.20 * Math.cos(x * 0.011 - y * 0.017) +
        0.15 * Math.sin((x * x + y * y) * 0.0004) +
        0.35 * r;
    }
  }
  return a;
}

export default {
  id: 'ncc-template',
  name: 'Template match (NCC)',
  params: `${W} × ${H} image · ${T} × ${T} template · ${OW} × ${OH} positions`,
  tag: 'read amplification',
  group: 'image',
  size: { w: W, h: H, t: T, ow: OW, oh: OH, invN: INV_N, eps: EPS },

  make({ w, h, t }) {
    const img = image(w, h, 0x1d872b41);

    // The template is a real crop of the image, taken from a known position,
    // with a little independent noise on top. A crop means the correlation
    // surface has a genuine peak at (px, py) rather than being a field of
    // near-zero scores, so the numbers on this row describe a problem someone
    // would actually pose. The noise keeps that peak below a perfect 1.0.
    const px = 137;
    const py = 211;
    const tpl = new Float32Array(t * t);
    let s = 0xb5297a4d >>> 0;
    for (let ty = 0; ty < t; ty++) {
      for (let tx = 0; tx < t; tx++) {
        s = (s * 1664525 + 1013904223) >>> 0;
        tpl[ty * t + tx] = img[(py + ty) * w + px + tx] + ((s >>> 8) / 0x1000000 - 0.5) * 0.08;
      }
    }

    // Mean-subtracted once, here, so every column starts from the same bytes
    // and none of them pays for the centring. `ss` is sum(t'^2), the constant
    // half of the denominator.
    let mean = 0;
    for (let i = 0; i < tpl.length; i++) mean += tpl[i];
    mean /= tpl.length;
    let ss = 0;
    for (let i = 0; i < tpl.length; i++) {
      tpl[i] -= mean;
      ss += tpl[i] * tpl[i];
    }

    const imgRows = [];
    for (let y = 0; y < h; y++) imgRows.push(img.subarray(y * w, y * w + w));
    const tplRows = [];
    for (let y = 0; y < t; y++) tplRows.push(tpl.subarray(y * t, y * t + t));

    return { img, tpl, ss, imgRows, tplRows, peak: [px, py] };
  },

  /**
   * The oracle, and a fair baseline. One pass per window accumulating three
   * sums, with the window walked row by row so both the image and the template
   * are read forwards — the same 4096 multiply-accumulates the kernels do, in
   * the order that does not fight the cache. Walking the window column-major
   * instead costs about 3x on the same arithmetic, and a baseline that did that
   * would hand every GPU column a speed-up it had not earned.
   */
  js({ w, t, ow, oh, invN, eps }, { img, tpl, ss }) {
    const out = new Float32Array(ow * oh);
    for (let oy = 0; oy < oh; oy++) {
      for (let ox = 0; ox < ow; ox++) {
        let sI = 0;
        let sI2 = 0;
        let sIT = 0;
        for (let ty = 0; ty < t; ty++) {
          const ir = (oy + ty) * w + ox;
          const tr = ty * t;
          for (let tx = 0; tx < t; tx++) {
            const v = img[ir + tx];
            sI += v;
            sI2 += v * v;
            sIT += v * tpl[tr + tx];
          }
        }
        const varI = sI2 - sI * sI * invN;
        const den = Math.sqrt(varI * ss);
        out[oy * ow + ox] = den > eps ? sIT / den : 0;
      }
    }
    return out;
  },

  gpujs(gpu, { t, ow, oh, invN, eps }, { imgRows, tplRows, ss }) {
    const kernel = gpu
      .createKernel(function (img, tpl) {
        const ox = this.thread.x;
        const oy = this.thread.y;
        let sI = 0;
        let sI2 = 0;
        let sIT = 0;
        for (let ty = 0; ty < this.constants.t; ty++) {
          for (let tx = 0; tx < this.constants.t; tx++) {
            const v = img[oy + ty][ox + tx];
            sI = sI + v;
            sI2 = sI2 + v * v;
            sIT = sIT + v * tpl[ty][tx];
          }
        }
        const varI = sI2 - sI * sI * this.constants.invN;
        const den = Math.sqrt(varI * this.constants.ss);
        let out = 0;
        if (den > this.constants.eps) out = sIT / den;
        return out;
      })
      .setConstants({ t, invN, ss, eps })
      // `t` is an integer constant, so gpu.js inlines it as a GLSL `const int`
      // and both loops keep constant bounds. A loop bound that arrived as a
      // uniform would be rewritten into gpu.js's LOOP_MAX form, whose default
      // ceiling is 1000 — it would silently stop at 1000 of the 4096 samples
      // and the row would report a fast, wrong answer.
      .setPrecision('single')
      .setTactic('precision')
      .setOutput([ow, oh]);

    return {
      async run() {
        return await kernel(imgRows, tplRows);
      },
      backend: () => kernel.kernel && kernel.kernel.constructor.mode,
      destroy: () => kernel.destroy && kernel.destroy(),
    };
  },

  /**
   * Hand-written WebGPU, borrowing nothing from gpu.js. One dispatch, both
   * inputs uploaded once at build time and left resident: only the 449 x 449
   * correlation surface crosses the bus per run, so this row is one of the few
   * where the transfer really is negligible next to the arithmetic.
   */
  async webgpu(device, { w, t, ow, oh, invN, eps }, { img, tpl, ss }) {
    const S = GPUBufferUsage.STORAGE;
    const upload = data => {
      const buf = device.createBuffer({
        size: data.byteLength,
        usage: S | GPUBufferUsage.COPY_DST,
        mappedAtCreation: true,
      });
      new Float32Array(buf.getMappedRange()).set(data);
      buf.unmap();
      return buf;
    };
    const bufImg = upload(img);
    const bufTpl = upload(tpl);
    const outBytes = ow * oh * 4;
    const bufOut = device.createBuffer({ size: outBytes, usage: S | GPUBufferUsage.COPY_SRC });
    const read = device.createBuffer({
      size: outBytes,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    const TILE = 8;
    const module = device.createShaderModule({
      code: `
@group(0) @binding(0) var<storage, read> img: array<f32>;
@group(0) @binding(1) var<storage, read> tpl: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

const W: u32 = ${w}u;
const T: u32 = ${t}u;
const OW: u32 = ${ow}u;
const OH: u32 = ${oh}u;
const INV_N: f32 = ${invN};
const SS: f32 = ${ss};
const EPS: f32 = ${eps};

@compute @workgroup_size(${TILE}, ${TILE})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let ox = gid.x;
  let oy = gid.y;
  if (ox >= OW || oy >= OH) { return; }

  var sI = 0.0;
  var sI2 = 0.0;
  var sIT = 0.0;
  for (var ty: u32 = 0u; ty < T; ty = ty + 1u) {
    let ir = (oy + ty) * W + ox;
    let tr = ty * T;
    for (var tx: u32 = 0u; tx < T; tx = tx + 1u) {
      let v = img[ir + tx];
      sI = sI + v;
      sI2 = sI2 + v * v;
      sIT = sIT + v * tpl[tr + tx];
    }
  }
  let varI = sI2 - sI * sI * INV_N;
  let den = sqrt(varI * SS);
  var r = 0.0;
  if (den > EPS) { r = sIT / den; }
  out[oy * OW + ox] = r;
}`,
    });

    const pipeline = device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: 'main' } });
    const bind = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: bufImg } },
        { binding: 1, resource: { buffer: bufTpl } },
        { binding: 2, resource: { buffer: bufOut } },
      ],
    });
    const gx = Math.ceil(ow / TILE);
    const gy = Math.ceil(oh / TILE);

    return {
      async run() {
        const enc = device.createCommandEncoder();
        const pass = enc.beginComputePass();
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bind);
        pass.dispatchWorkgroups(gx, gy);
        pass.end();
        enc.copyBufferToBuffer(bufOut, 0, read, 0, outBytes);
        device.queue.submit([enc.finish()]);
        // The read-back is the only thing that proves the dispatch finished.
        await read.mapAsync(GPUMapMode.READ);
        const out = new Float32Array(read.getMappedRange()).slice();
        read.unmap();
        return out;
      },
      destroy() {
        [bufImg, bufTpl, bufOut, read].forEach(b => b.destroy && b.destroy());
      },
    };
  },

  /**
   * Index-weighted mean of the correlation surface. Scores live in [-1, 1] and
   * the surface has one real peak, so a backend that computed only part of it
   * cannot match by luck: the weight makes position count as well as value.
   */
  reduce(out, { ow, oh }) {
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
    return acc / (ow * oh);
  },
};
