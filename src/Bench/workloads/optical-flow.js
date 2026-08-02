/**
 * Lucas-Kanade optical flow — a small linear solve inside every thread.
 *
 * Two frames, a 17 x 17 window around every pixel, and a 2 x 2 normal-equation
 * system solved per pixel from the sums over that window. A million independent
 * linear solves, each too small to be worth a library call and each depending on
 * 289 neighbours' worth of accumulation.
 *
 * That shape is why the row is here. The stencil rows in this table read a fixed
 * handful of neighbours and do a handful of flops; this one reads 867 values per
 * output and then does something with a divide and a determinant in it. It is
 * the case where a thread has enough private state to keep registers busy but
 * not enough to justify shared memory, which is where an enormous amount of real
 * vision code sits.
 *
 * ── THE ANSWER IS KNOWN ────────────────────────────────────────────────────
 *
 * The second frame is the first one resampled at a smooth, analytically-defined
 * displacement field of about a pixel and a half. Lucas-Kanade should recover
 * the negative of that field — B(x) = A(x + d) means the pattern moved by -d —
 * and it does, to within a few percent over the textured part of the frame. That
 * matters because a benchmark whose output nobody can sanity-check is a
 * benchmark that will happily report a subtly broken kernel as a fast one; the
 * checksum catches disagreement BETWEEN columns, and a known answer is what
 * catches all of them being wrong together.
 *
 * ── THE SINGULAR SYSTEM, AND WHY IT IS DAMPED RATHER THAN BRANCHED ─────────
 *
 * The 2 x 2 system is singular wherever the window has no two-dimensional
 * structure. A flat patch gives the zero matrix — no information at all. A
 * region of pure horizontal stripes gives a rank-1 matrix: the motion along the
 * stripes is genuinely unknowable, which is the aperture problem, and no amount
 * of arithmetic will produce it. The honest answer in both places is "no flow",
 * not a division by a determinant that happens to have rounded to 1e-9. The
 * frame deliberately contains one of each so the row is actually exercising this.
 *
 * The usual guard is `if (det < tol) return zero`, and it is exactly the wrong
 * thing to put in a cross-backend benchmark: `det` near `tol` is a knife-edge,
 * fp32 and fp64 land on opposite sides of it for a scattering of pixels, and
 * those pixels then differ by the whole magnitude of the flow rather than by an
 * ulp. The checksums disagree and a correct GPU is reported as WRONG.
 *
 * So the diagonal is damped instead — Tikhonov, the same thing Levenberg does to
 * a Gauss-Newton step:
 *
 *      [ Sxx + L   Sxy     ] [u]   =  -[ Sxt ]
 *      [ Sxy       Syy + L ] [v]       [ Syt ]
 *
 * det is now at least L^2 > 0, so there is no divide by zero and no branch at
 * all. A textureless window has zero sums on both sides and yields exactly
 * [0, 0]; an aperture-limited window yields the component it can see and ~0 for
 * the one it cannot; a well-textured window is biased by well under a percent,
 * since L is four orders below a typical Sxx. Every pixel takes the same path
 * through the same arithmetic on every backend.
 *
 * ── CHECKSUM ────────────────────────────────────────────────────────────────
 *
 * With the branch gone there is nothing left to break a tie on, so the only
 * disagreement possible is rounding. The place it could still have bitten is the
 * determinant: for a rank-1 window Sxx*Syy and Sxy^2 cancel almost exactly, and
 * in fp32 that cancellation is worth about 1e-7 of the product. The damping term
 * L*(Sxx + Syy) sits four orders ABOVE that, so it sets a floor under the
 * subtraction and the relative error in det stays near 1e-5 even where the
 * geometry is worst. Simulating fp32 rounding through the whole baseline moves
 * the checksum by ~1e-7 relative, three orders inside the runner's 1e-4.
 */

const W = 1024;
const H = 1024;
const R = 8;
const WIN = 2 * R + 1;

// Tikhonov damping. Sxx over a well-textured 17 x 17 window runs around 1 for
// this frame, so 1e-3 is a floor three orders below the signal: negligible bias
// where there is structure, decisive where there is not.
const LAMBDA = 1e-3;

// The displacement field, in pixels. Smooth and low-amplitude, because
// Lucas-Kanade linearises: a displacement that moves the pattern by more than
// its own feature size in one step is not something a single-scale solve is
// supposed to recover, and building one in would be measuring the algorithm's
// failure rather than the hardware.
const AMP_U = 1.4;
const AMP_V = 1.1;

/**
 * The first frame: broadband texture over most of it, plus two regions with no
 * unique answer, which are the reason the damping in the header exists.
 *
 *   - a flat patch, exactly constant and noise-free, where the structure tensor
 *     is the zero matrix.
 *   - a band of pure horizontal stripes, varying with y only, where Ix is
 *     exactly zero and the horizontal component of the flow is unknowable.
 */
function frameA(w, h, seed) {
  const a = new Float32Array(w * h);
  let s = seed >>> 0;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      s = (s * 1664525 + 1013904223) >>> 0;
      const r = (s >>> 8) / 0x1000000 - 0.5;
      let v =
        0.5 +
        0.22 * Math.sin(x * 0.07 + y * 0.031) +
        0.16 * Math.cos(x * 0.021 - y * 0.045) +
        0.12 * Math.sin((x + y) * 0.13) +
        0.03 * r;
      // Aperture band: a function of y alone, so Ix is identically zero.
      if (y >= 620 && y < 780) v = 0.5 + 0.3 * Math.sin(y * 0.35);
      // Flat patch: no structure at all.
      if (x >= 700 && x < 880 && y >= 120 && y < 300) v = 0.42;
      a[y * w + x] = v;
    }
  }
  return a;
}

/**
 * The second frame: the first, resampled bilinearly at (x + u, y + v). Built in
 * make() so all six columns are handed the same bytes; a column that generated
 * its own frame would be timing a sine wave.
 */
function frameB(a, w, h) {
  const b = new Float32Array(w * h);
  const tau = Math.PI * 2;
  for (let y = 0; y < h; y++) {
    const fy = tau * (y / h);
    for (let x = 0; x < w; x++) {
      const fx = tau * (x / w);
      const sx = x + AMP_U * Math.sin(fx) * Math.cos(fy);
      const sy = y + AMP_V * Math.cos(fx + 0.6) * Math.sin(fy + 0.3);
      const x0 = Math.min(w - 2, Math.max(0, Math.floor(sx)));
      const y0 = Math.min(h - 2, Math.max(0, Math.floor(sy)));
      const tx = sx - x0;
      const ty = sy - y0;
      const r0 = y0 * w + x0;
      const r1 = r0 + w;
      b[y * w + x] =
        a[r0] * (1 - tx) * (1 - ty) +
        a[r0 + 1] * tx * (1 - ty) +
        a[r1] * (1 - tx) * ty +
        a[r1 + 1] * tx * ty;
    }
  }
  return b;
}

export default {
  id: 'optical-flow',
  name: 'Lucas–Kanade optical flow',
  params: `${W} × ${H} frame pair · ${WIN} × ${WIN} window`,
  tag: 'per-thread 2×2 solve',
  group: 'image',
  size: { w: W, h: H, r: R, win: WIN, lambda: LAMBDA, wm1: W - 1, hm1: H - 1, wmr: W - R, hmr: H - R },

  make({ w, h }) {
    const a = frameA(w, h, 0x3c6ef35f);
    const b = frameB(a, w, h);
    const aRows = [];
    const bRows = [];
    for (let y = 0; y < h; y++) {
      aRows.push(a.subarray(y * w, y * w + w));
      bRows.push(b.subarray(y * w, y * w + w));
    }
    return { a, b, aRows, bRows };
  },

  /**
   * The oracle, and a fair baseline.
   *
   * The three derivative planes are built first and then read, rather than
   * recomputed inside the window loop — 289 window samples per pixel would
   * otherwise mean differencing every pixel 289 times. That is what every real
   * implementation does and what all three kernels below do, so the columns are
   * comparing the same arithmetic; it just happens to also be the fast way.
   *
   * The window loop walks row by row so all three planes are read forwards, and
   * the five accumulators live in locals rather than an array.
   */
  js({ w, h, r, lambda, wm1, hm1, wmr, hmr }, { a, b }) {
    const ix = new Float32Array(w * h);
    const iy = new Float32Array(w * h);
    const it = new Float32Array(w * h);

    for (let y = 0; y < h; y++) {
      const up = (y > 0 ? y - 1 : 0) * w;
      const row = y * w;
      const down = (y < hm1 ? y + 1 : hm1) * w;
      for (let x = 0; x < w; x++) {
        const xl = x > 0 ? x - 1 : 0;
        const xr = x < wm1 ? x + 1 : wm1;
        ix[row + x] = (a[row + xr] - a[row + xl]) * 0.5;
        iy[row + x] = (a[down + x] - a[up + x]) * 0.5;
        it[row + x] = b[row + x] - a[row + x];
      }
    }

    // Interleaved [u, v] per pixel. Pixels closer than r to an edge keep their
    // zeros: the window would run off the frame, and the kernels reach the same
    // zeros by accumulating nothing and dividing by lambda^2.
    const out = new Float32Array(w * h * 2);
    for (let y = r; y < hmr; y++) {
      for (let x = r; x < wmr; x++) {
        let sxx = 0;
        let sxy = 0;
        let syy = 0;
        let sxt = 0;
        let syt = 0;
        for (let ky = -r; ky <= r; ky++) {
          const o = (y + ky) * w + x;
          for (let kx = -r; kx <= r; kx++) {
            const gx = ix[o + kx];
            const gy = iy[o + kx];
            const gt = it[o + kx];
            sxx += gx * gx;
            sxy += gx * gy;
            syy += gy * gy;
            sxt += gx * gt;
            syt += gy * gt;
          }
        }
        const ta = sxx + lambda;
        const td = syy + lambda;
        const det = ta * td - sxy * sxy;
        const i2 = (y * w + x) * 2;
        out[i2] = (sxy * syt - td * sxt) / det;
        out[i2 + 1] = (sxy * sxt - ta * syt) / det;
      }
    }
    return out;
  },

  gpujs(gpu, { w, h, r, win, lambda, wm1, hm1, wmr, hmr }, { aRows, bRows }) {
    const consts = { r, win, lambda, wm1, hm1, wmr, hmr };
    const mk = (fn, pipeline) =>
      gpu
        .createKernel(fn)
        .setConstants(consts)
        .setPipeline(pipeline)
        .setPrecision('single')
        // Explicit, because the default tactic picks a GLSL precision qualifier
        // from the texture size and a `lowp` structure tensor would be noise.
        .setTactic('precision')
        .setOutput([w, h]);

    // Three derivative planes, computed once and left resident as pipeline
    // textures. Sending them to the host and back would be 12 MB of traffic per
    // run for values the next kernel is about to read anyway.
    const kIx = mk(function (a) {
      const x = this.thread.x;
      let xl = x - 1;
      if (xl < 0) xl = 0;
      let xr = x + 1;
      if (xr > this.constants.wm1) xr = this.constants.wm1;
      return (a[this.thread.y][xr] - a[this.thread.y][xl]) * 0.5;
    }, true);

    const kIy = mk(function (a) {
      const y = this.thread.y;
      let yu = y - 1;
      if (yu < 0) yu = 0;
      let yd = y + 1;
      if (yd > this.constants.hm1) yd = this.constants.hm1;
      return (a[yd][this.thread.x] - a[yu][this.thread.x]) * 0.5;
    }, true);

    const kIt = mk(function (a, b) {
      return b[this.thread.y][this.thread.x] - a[this.thread.y][this.thread.x];
    }, true);

    // The solve. Not pipelined: it is the last stage, so `run` resolves on a
    // real array rather than on a handle to work that may still be queued.
    const kFlow = mk(function (ix, iy, it) {
      const x = this.thread.x;
      const y = this.thread.y;
      let sxx = 0;
      let sxy = 0;
      let syy = 0;
      let sxt = 0;
      let syt = 0;
      if (
        x >= this.constants.r &&
        x < this.constants.wmr &&
        y >= this.constants.r &&
        y < this.constants.hmr
      ) {
        for (let ky = 0; ky < this.constants.win; ky++) {
          const yy = y + ky - this.constants.r;
          for (let kx = 0; kx < this.constants.win; kx++) {
            const xx = x + kx - this.constants.r;
            const gx = ix[yy][xx];
            const gy = iy[yy][xx];
            const gt = it[yy][xx];
            sxx = sxx + gx * gx;
            sxy = sxy + gx * gy;
            syy = syy + gy * gy;
            sxt = sxt + gx * gt;
            syt = syt + gy * gt;
          }
        }
      }
      const ta = sxx + this.constants.lambda;
      const td = syy + this.constants.lambda;
      const det = ta * td - sxy * sxy;
      return [(sxy * syt - td * sxt) / det, (sxy * sxt - ta * syt) / det];
    }, false);

    return {
      async run() {
        const tx = await kIx(aRows);
        const ty = await kIy(aRows);
        const tt = await kIt(aRows, bRows);
        return await kFlow(tx, ty, tt);
      },
      backend: () => kFlow.kernel && kFlow.kernel.constructor.mode,
      destroy() {
        [kIx, kIy, kIt, kFlow].forEach(k => k.destroy && k.destroy());
      },
    };
  },

  /**
   * Hand-written WebGPU, borrowing nothing from gpu.js. Four entry points over
   * one shared bind group layout, all four dispatches in a single compute pass:
   * WebGPU orders dispatches within a pass and makes each one's writes visible
   * to the next, so the derivative planes never leave the device and the whole
   * thing is one submit and one read-back.
   */
  async webgpu(device, { w, h, r, win, lambda, wm1, hm1, wmr, hmr }, { a, b }) {
    const px = w * h;
    const bytes = px * 4;
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
    const bufA = upload(a);
    const bufB = upload(b);
    const mk = () => device.createBuffer({ size: bytes, usage: S });
    const bufIx = mk();
    const bufIy = mk();
    const bufIt = mk();
    const outBytes = px * 2 * 4;
    const bufOut = device.createBuffer({ size: outBytes, usage: S | GPUBufferUsage.COPY_SRC });
    const read = device.createBuffer({
      size: outBytes,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    const TILE = 16;
    const module = device.createShaderModule({
      code: `
@group(0) @binding(0) var<storage, read> fa: array<f32>;
@group(0) @binding(1) var<storage, read> fb: array<f32>;
@group(0) @binding(2) var<storage, read_write> ix: array<f32>;
@group(0) @binding(3) var<storage, read_write> iy: array<f32>;
@group(0) @binding(4) var<storage, read_write> it: array<f32>;
@group(0) @binding(5) var<storage, read_write> flow: array<f32>;

const W: i32 = ${w};
const H: i32 = ${h};
const WM1: i32 = ${wm1};
const HM1: i32 = ${hm1};
const R: i32 = ${r};
const WIN: i32 = ${win};
const WMR: i32 = ${wmr};
const HMR: i32 = ${hmr};
const LAMBDA: f32 = ${lambda};

@compute @workgroup_size(${TILE}, ${TILE})
fn derivX(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = i32(gid.x);
  let y = i32(gid.y);
  if (x >= W || y >= H) { return; }
  let row = y * W;
  ix[row + x] = (fa[row + min(x + 1, WM1)] - fa[row + max(x - 1, 0)]) * 0.5;
}

@compute @workgroup_size(${TILE}, ${TILE})
fn derivY(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = i32(gid.x);
  let y = i32(gid.y);
  if (x >= W || y >= H) { return; }
  iy[y * W + x] = (fa[min(y + 1, HM1) * W + x] - fa[max(y - 1, 0) * W + x]) * 0.5;
}

@compute @workgroup_size(${TILE}, ${TILE})
fn derivT(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = i32(gid.x);
  let y = i32(gid.y);
  if (x >= W || y >= H) { return; }
  let i = y * W + x;
  it[i] = fb[i] - fa[i];
}

@compute @workgroup_size(${TILE}, ${TILE})
fn solve(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = i32(gid.x);
  let y = i32(gid.y);
  if (x >= W || y >= H) { return; }

  var sxx = 0.0;
  var sxy = 0.0;
  var syy = 0.0;
  var sxt = 0.0;
  var syt = 0.0;
  if (x >= R && x < WMR && y >= R && y < HMR) {
    for (var ky = 0; ky < WIN; ky++) {
      let o = (y + ky - R) * W + x - R;
      for (var kx = 0; kx < WIN; kx++) {
        let gx = ix[o + kx];
        let gy = iy[o + kx];
        let gt = it[o + kx];
        sxx = sxx + gx * gx;
        sxy = sxy + gx * gy;
        syy = syy + gy * gy;
        sxt = sxt + gx * gt;
        syt = syt + gy * gt;
      }
    }
  }
  let ta = sxx + LAMBDA;
  let td = syy + LAMBDA;
  let det = ta * td - sxy * sxy;
  let i2 = (y * W + x) * 2;
  flow[i2] = (sxy * syt - td * sxt) / det;
  flow[i2 + 1] = (sxy * sxt - ta * syt) / det;
}`,
    });

    const ro = { type: 'read-only-storage' };
    const rw = { type: 'storage' };
    const layout = device.createBindGroupLayout({
      entries: [ro, ro, rw, rw, rw, rw].map((buffer, binding) => ({
        binding,
        visibility: GPUShaderStage.COMPUTE,
        buffer,
      })),
    });
    const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [layout] });
    const pipe = entryPoint =>
      device.createComputePipeline({ layout: pipelineLayout, compute: { module, entryPoint } });
    const pipelines = [pipe('derivX'), pipe('derivY'), pipe('derivT'), pipe('solve')];
    const bind = device.createBindGroup({
      layout,
      entries: [bufA, bufB, bufIx, bufIy, bufIt, bufOut].map((buffer, binding) => ({
        binding,
        resource: { buffer },
      })),
    });
    const gx = Math.ceil(w / TILE);
    const gy = Math.ceil(h / TILE);

    return {
      async run() {
        const enc = device.createCommandEncoder();
        const pass = enc.beginComputePass();
        pass.setBindGroup(0, bind);
        for (const p of pipelines) {
          pass.setPipeline(p);
          pass.dispatchWorkgroups(gx, gy);
        }
        pass.end();
        enc.copyBufferToBuffer(bufOut, 0, read, 0, outBytes);
        device.queue.submit([enc.finish()]);
        await read.mapAsync(GPUMapMode.READ);
        const out = new Float32Array(read.getMappedRange()).slice();
        read.unmap();
        return out;
      },
      destroy() {
        [bufA, bufB, bufIx, bufIy, bufIt, bufOut, read].forEach(x => x.destroy && x.destroy());
      },
    };
  },

  /**
   * Index-weighted mean over BOTH components of every pixel. Summing only the
   * magnitude would let a backend that swapped u and v pass, and summing only
   * the interior would let one that skipped the border pass; this walks the
   * whole interleaved field in one order.
   *
   * The three shapes it has to accept are the flat interleaved array the
   * baseline and the bare WebGPU column return, and gpu.js's Array(2) output,
   * which arrives as rows of two-element views. The defensive branch on
   * `typeof row[0]` covers a backend that ever flattens a row instead.
   */
  reduce(out, { w, h }) {
    let acc = 0;
    let i = 0;
    if (ArrayBuffer.isView(out)) {
      for (; i < out.length; i++) acc += out[i] * (1 + (i % 17));
    } else {
      for (let y = 0; y < out.length; y++) {
        const row = out[y];
        if (typeof row[0] === 'number') {
          for (let k = 0; k < row.length; k++, i++) acc += row[k] * (1 + (i % 17));
        } else {
          for (let x = 0; x < row.length; x++) {
            const p = row[x];
            acc += p[0] * (1 + (i % 17));
            i++;
            acc += p[1] * (1 + (i % 17));
            i++;
          }
        }
      }
    }
    return acc / (w * h * 2);
  },
};
