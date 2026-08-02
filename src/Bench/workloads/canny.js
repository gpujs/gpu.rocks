/**
 * Canny edge detection — five kernels, one chain, nothing returning to the host.
 *
 * Blur, gradient, non-maximum suppression, double threshold, hysteresis. Every
 * stage reads the whole plane the stage before it wrote, and none of the four
 * intermediates has any business being copied to the CPU and back. That is what
 * this row prices: not one kernel, but the SHAPE of a real pipeline, where the
 * win comes from what you did not transfer as much as from what you computed.
 *
 * Put beside `sobel` — one stencil of about the same cost as this row's second
 * stage — it says something the single-kernel rows cannot. If a five-stage
 * chain is not roughly five times a one-stage row, the difference is dispatch
 * and residency, and this table exists to make that visible rather than
 * arguable.
 *
 * ── EVERY NUMBER IN THIS PIPELINE IS AN EXACT INTEGER, ON PURPOSE ───────────
 *
 * Canny is made of decisions: is this pixel a local maximum along the gradient,
 * is it above the high threshold, does it touch something strong. A decision is
 * a comparison, and a comparison at a tie goes one way in fp64 and the other in
 * fp32 for no better reason than the order two hardware units happened to
 * round. One flipped tie in non-maximum suppression removes a pixel; hysteresis
 * then propagates the absence; the checksums disagree and the row reports a
 * GPU as WRONG when it was only differently rounded. That failure would be the
 * benchmark's fault, not the backend's.
 *
 * So the whole chain is arranged to carry only integers that fp32 represents
 * exactly (everything below 2^24 = 16777216):
 *
 *   - the source is 8-bit greyscale, 0..255, stored as floats.
 *   - the 5x5 binomial blur is left UNNORMALISED — weights [1,4,6,4,1] outer
 *     product, summing to 256 — so its output is an integer <= 65280 rather
 *     than a rounded fraction.
 *   - the gradient magnitude is |gx| + |gy|, not sqrt(gx^2 + gy^2). The L1
 *     approximation is the one Canny's own paper suggests for cheapness, and
 *     here it also keeps the magnitude an exact integer <= 522240.
 *   - the gradient direction is quantised by comparing 2|gy| vs |gx| and
 *     2|gx| vs |gy| — thresholds of 1/2 and 2 rather than the textbook
 *     tan(22.5 deg) = 0.414 and tan(67.5 deg) = 2.414. Rational thresholds make
 *     the sector an exact integer comparison; irrational ones would put a
 *     knife-edge back in.
 *   - magnitude and sector travel as ONE float, mag * 4 + sector, maximum
 *     2088963. Two channels would mean an Array(2) pipeline texture, which
 *     gpu.js's WebGPU backend cannot yet take as a kernel argument, and
 *     recomputing the direction in the NMS stage would mean doing the Sobel
 *     twice.
 *
 * The result is that every backend must agree BIT FOR BIT. The runner's 1e-4
 * tolerance is slack this row never spends.
 *
 * ── HYSTERESIS IS A FIXED NUMBER OF PASSES ─────────────────────────────────
 *
 * Textbook hysteresis is a flood fill: keep promoting weak pixels that touch
 * strong ones until nothing changes. "Until nothing changes" is data-dependent
 * iteration, which on a GPU means either a read-back per pass to test a flag —
 * pricing the round trip, not the algorithm — or a serial scan no kernel can
 * do. Eight passes, always, in every column: an edge is joined across up to
 * eight pixels of weak chain, every backend does exactly the same work, and the
 * answer is a function of the input alone. Eight rather than two also because
 * the passes are nearly free on the CPU and are eight more dispatches on the
 * GPU, which is precisely the asymmetry this row is here to price.
 *
 * ── THE BLUR IS NOT SEPARATED ──────────────────────────────────────────────
 *
 * A 5x5 binomial is separable and a competent implementation would run it as
 * two 1-D passes for 10 taps instead of 25. It is left as a direct 2-D
 * convolution here for two reasons: it keeps the chain at the five stages Canny
 * is actually described as, and `blur-separable` already exists to show what
 * separating one buys. Both columns pay the same 25 taps, so the ratio is
 * untouched; only the absolute numbers are 2.5x what a separated blur would be.
 */

const N = 2048;

// Weight-256 binomial blur, so a pixel that is 255 everywhere blurs to 65280.
const W5 = [1, 4, 6, 4, 1];

// Thresholds on the unnormalised L1 magnitude, chosen from the histogram of
// this exact image: the 99.5th percentile is about 66000 and the 97th about
// 5000, so these leave roughly 0.3% of pixels strong and 0.35% weak after
// suppression — enough weak pixels for hysteresis to have real work to do, few
// enough that the map is edges rather than texture.
const HI = 60000;
const LO = 8000;

// Passes of weak-to-strong propagation. See the header: fixed, not to
// convergence.
const HYST = 8;

/**
 * An 8-bit greyscale image with things in it that are actually edges: filled
 * discs and axis-aligned bars with hard boundaries over a smooth low-frequency
 * background, plus a couple of levels of noise. Canny on white noise is a
 * uniform mush and would exercise none of the branches this row is here for;
 * the bars in particular are what make the sector-0 and sector-2 arms of the
 * non-maximum suppression fire on something other than an accident of noise.
 *
 * Built to be cheap as well as deterministic. The sizing script warns when
 * `make` costs more than `js`, and rightly: the inputs are built once and
 * shared by six columns, so a builder that outweighs the work it feeds is a
 * sign the work is in the wrong place. Hence the separable background (one
 * sine per column, one cosine per row) and discs rasterised over their bounding
 * boxes rather than tested against every pixel.
 */
function image(n, seed) {
  const a = new Float32Array(n * n);
  let s = seed >>> 0;
  const rnd = () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return (s >>> 8) / 0x1000000;
  };

  const sx = new Float64Array(n);
  const cy = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    sx[i] = Math.sin(i * 0.004);
    cy[i] = Math.cos(i * 0.003);
  }

  const barTop = Math.round(n * 0.1);
  const barBottom = Math.round(n * 0.9);
  const barLeft = Math.round(n * 0.05);
  const barRight = Math.round(n * 0.95);
  for (let y = 0; y < n; y++) {
    const row = y * n;
    const c = 40 * cy[y];
    const vertBar = (y >> 8) % 4 === 1;
    for (let x = 0; x < n; x++) {
      let v = 110 + c * sx[x];
      if ((x >> 7) % 5 === 0 && y > barTop && y < barBottom) v = 230;
      if (vertBar && x > barLeft && x < barRight) v = v * 0.4 + 20;
      a[row + x] = v;
    }
  }

  for (let k = 0; k < 24; k++) {
    const ccx = rnd() * n;
    const ccy = rnd() * n;
    const r = 40 + rnd() * 220;
    const level = 30 + rnd() * 200;
    const r2 = r * r;
    const y0 = Math.max(0, Math.ceil(ccy - r));
    const y1 = Math.min(n - 1, Math.floor(ccy + r));
    const x0 = Math.max(0, Math.ceil(ccx - r));
    const x1 = Math.min(n - 1, Math.floor(ccx + r));
    for (let y = y0; y <= y1; y++) {
      const dy = y - ccy;
      const row = y * n;
      const half = Math.sqrt(Math.max(0, r2 - dy * dy));
      const lo = Math.max(x0, Math.ceil(ccx - half));
      const hi = Math.min(x1, Math.floor(ccx + half));
      for (let x = lo; x <= hi; x++) a[row + x] = level;
    }
  }

  for (let i = 0; i < a.length; i++) {
    const v = a[i] + (rnd() - 0.5) * 6;
    a[i] = v < 0 ? 0 : v > 255 ? 255 : Math.round(v);
  }
  return a;
}

const clamp = (v, hi) => (v < 0 ? 0 : v > hi ? hi : v);

/** The clamped 5x5 tap, used only on the four border columns of each row. */
function blurClamped(src, n, nm1, y, x) {
  let acc = 0;
  for (let ky = 0; ky < 5; ky++) {
    const r = clamp(y + ky - 2, nm1) * n;
    const wy = W5[ky];
    for (let kx = 0; kx < 5; kx++) acc += wy * W5[kx] * src[r + clamp(x + kx - 2, nm1)];
  }
  return acc;
}

export default {
  id: 'canny',
  name: 'Canny edge detector',
  params: `${N} × ${N} 8-bit, 5 stages, ${HYST} hysteresis passes`,
  tag: 'multi-kernel chain',
  group: 'image',
  size: { n: N, nm1: N - 1, hi: HI, lo: LO, hyst: HYST },

  make({ n }) {
    const src = image(n, 0x7f4a7c15);
    const rows = [];
    for (let y = 0; y < n; y++) rows.push(src.subarray(y * n, y * n + n));
    return { src, rows };
  },

  /**
   * The oracle, and a fair baseline. Five passes over five flat planes, in the
   * same order and with the same arithmetic as the five kernels.
   *
   * The blur splits its interior from its border so the 4.2 M interior pixels
   * pay no clamping at all, and factors the separable weights per row —
   * w[ky] * (p-2 + 4p-1 + 6p + 4p+1 + p+2) — instead of 25 independent
   * multiplies. Both are exactly the arithmetic the kernel does, reassociated;
   * on integers this small the reassociation is exact, so the two agree to the
   * bit. A baseline that walked 25 clamped taps per pixel would be 40% slower
   * and would hand every GPU column a speed-up it had not earned.
   */
  js({ n, hi, lo, hyst }, { src }) {
    const nm1 = n - 1;
    const blur = new Float32Array(n * n);
    const mag = new Float32Array(n * n);
    const nms = new Float32Array(n * n);

    // ── 1. blur ─────────────────────────────────────────────────────────────
    for (let y = 0; y < n; y++) {
      const r0 = clamp(y - 2, nm1) * n;
      const r1 = clamp(y - 1, nm1) * n;
      const r2 = y * n;
      const r3 = clamp(y + 1, nm1) * n;
      const r4 = clamp(y + 2, nm1) * n;
      const out = y * n;

      for (let x = 0; x < 2; x++) blur[out + x] = blurClamped(src, n, nm1, y, x);
      for (let x = n - 2; x < n; x++) blur[out + x] = blurClamped(src, n, nm1, y, x);

      for (let x = 2; x < n - 2; x++) {
        const a = src[r0 + x - 2] + 4 * src[r0 + x - 1] + 6 * src[r0 + x] + 4 * src[r0 + x + 1] + src[r0 + x + 2];
        const b = src[r1 + x - 2] + 4 * src[r1 + x - 1] + 6 * src[r1 + x] + 4 * src[r1 + x + 1] + src[r1 + x + 2];
        const c = src[r2 + x - 2] + 4 * src[r2 + x - 1] + 6 * src[r2 + x] + 4 * src[r2 + x + 1] + src[r2 + x + 2];
        const d = src[r3 + x - 2] + 4 * src[r3 + x - 1] + 6 * src[r3 + x] + 4 * src[r3 + x + 1] + src[r3 + x + 2];
        const e = src[r4 + x - 2] + 4 * src[r4 + x - 1] + 6 * src[r4 + x] + 4 * src[r4 + x + 1] + src[r4 + x + 2];
        blur[out + x] = a + 4 * b + 6 * c + 4 * d + e;
      }
    }

    // ── 2. gradient: L1 magnitude and a quantised direction, packed ─────────
    for (let y = 0; y < n; y++) {
      const up = clamp(y - 1, nm1) * n;
      const row = y * n;
      const down = clamp(y + 1, nm1) * n;
      for (let x = 0; x < n; x++) {
        const xl = clamp(x - 1, nm1);
        const xr = clamp(x + 1, nm1);
        const p00 = blur[up + xl];
        const p01 = blur[up + x];
        const p02 = blur[up + xr];
        const p10 = blur[row + xl];
        const p12 = blur[row + xr];
        const p20 = blur[down + xl];
        const p21 = blur[down + x];
        const p22 = blur[down + xr];
        const gx = p02 + 2 * p12 + p22 - (p00 + 2 * p10 + p20);
        const gy = p20 + 2 * p21 + p22 - (p00 + 2 * p01 + p02);
        const ax = gx < 0 ? -gx : gx;
        const ay = gy < 0 ? -gy : gy;
        let sector = 3;
        if (2 * ay < ax) sector = 0;
        else if (2 * ax < ay) sector = 2;
        else if (gx * gy < 0) sector = 1;
        mag[row + x] = (ax + ay) * 4 + sector;
      }
    }

    // ── 3. non-maximum suppression ─────────────────────────────────────────
    for (let y = 0; y < n; y++) {
      const row = y * n;
      for (let x = 0; x < n; x++) {
        const v = mag[row + x];
        const m = Math.floor(v * 0.25);
        const sector = v - 4 * m;
        let xa = x + 1;
        let ya = y;
        let xb = x - 1;
        let yb = y;
        if (sector === 1) {
          ya = y - 1;
          yb = y + 1;
        } else if (sector === 2) {
          xa = x;
          ya = y + 1;
          xb = x;
          yb = y - 1;
        } else if (sector === 3) {
          ya = y + 1;
          yb = y - 1;
        }
        const ma = Math.floor(mag[clamp(ya, nm1) * n + clamp(xa, nm1)] * 0.25);
        const mb = Math.floor(mag[clamp(yb, nm1) * n + clamp(xb, nm1)] * 0.25);
        nms[row + x] = m >= ma && m >= mb ? m : 0;
      }
    }

    // ── 4. double threshold ────────────────────────────────────────────────
    let cur = new Float32Array(n * n);
    for (let i = 0; i < nms.length; i++) {
      const v = nms[i];
      cur[i] = v > hi ? 2 : v > lo ? 1 : 0;
    }

    // ── 5. hysteresis, a fixed number of passes ────────────────────────────
    let next = new Float32Array(n * n);
    for (let p = 0; p < hyst; p++) {
      for (let y = 0; y < n; y++) {
        const up = clamp(y - 1, nm1) * n;
        const row = y * n;
        const down = clamp(y + 1, nm1) * n;
        for (let x = 0; x < n; x++) {
          const v = cur[row + x];
          if (v !== 1) {
            next[row + x] = v;
            continue;
          }
          const xl = clamp(x - 1, nm1);
          const xr = clamp(x + 1, nm1);
          const strong =
            cur[up + xl] === 2 || cur[up + x] === 2 || cur[up + xr] === 2 ||
            cur[row + xl] === 2 || cur[row + xr] === 2 ||
            cur[down + xl] === 2 || cur[down + x] === 2 || cur[down + xr] === 2;
          next[row + x] = strong ? 2 : 1;
        }
      }
      const t = cur;
      cur = next;
      next = t;
    }
    return cur;
  },

  gpujs(gpu, { n, nm1, hi, lo, hyst }, { rows }) {
    const consts = { nm1, hi, lo };
    const mk = (fn, pipeline) =>
      gpu
        .createKernel(fn)
        .setConstants(consts)
        .setPipeline(pipeline)
        .setPrecision('single')
        // The default tactic picks a GLSL precision qualifier from the texture
        // size, and `lowp` would destroy magnitudes that run to six figures.
        // Asked for explicitly so every backend carries the same fp32.
        .setTactic('precision')
        .setOutput([n, n]);

    const kBlur = mk(function (a) {
      let acc = 0;
      for (let ky = 0; ky < 5; ky++) {
        let yy = this.thread.y + ky - 2;
        if (yy < 0) yy = 0;
        if (yy > this.constants.nm1) yy = this.constants.nm1;
        let wy = 6;
        if (ky === 0 || ky === 4) wy = 1;
        if (ky === 1 || ky === 3) wy = 4;
        for (let kx = 0; kx < 5; kx++) {
          let xx = this.thread.x + kx - 2;
          if (xx < 0) xx = 0;
          if (xx > this.constants.nm1) xx = this.constants.nm1;
          let wx = 6;
          if (kx === 0 || kx === 4) wx = 1;
          if (kx === 1 || kx === 3) wx = 4;
          acc += wy * wx * a[yy][xx];
        }
      }
      return acc;
    }, true);

    const kGrad = mk(function (a) {
      const x = this.thread.x;
      const y = this.thread.y;
      let xl = x - 1;
      if (xl < 0) xl = 0;
      let xr = x + 1;
      if (xr > this.constants.nm1) xr = this.constants.nm1;
      let yu = y - 1;
      if (yu < 0) yu = 0;
      let yd = y + 1;
      if (yd > this.constants.nm1) yd = this.constants.nm1;

      const p00 = a[yu][xl];
      const p01 = a[yu][x];
      const p02 = a[yu][xr];
      const p10 = a[y][xl];
      const p12 = a[y][xr];
      const p20 = a[yd][xl];
      const p21 = a[yd][x];
      const p22 = a[yd][xr];
      const gx = p02 + 2 * p12 + p22 - (p00 + 2 * p10 + p20);
      const gy = p20 + 2 * p21 + p22 - (p00 + 2 * p01 + p02);
      let ax = gx;
      if (ax < 0) ax = -ax;
      let ay = gy;
      if (ay < 0) ay = -ay;

      let sector = 3;
      if (2 * ay < ax) sector = 0;
      else if (2 * ax < ay) sector = 2;
      else if (gx * gy < 0) sector = 1;
      return (ax + ay) * 4 + sector;
    }, true);

    const kNms = mk(function (a) {
      const x = this.thread.x;
      const y = this.thread.y;
      const v = a[y][x];
      const m = Math.floor(v * 0.25);
      const sector = v - 4 * m;

      let xa = x + 1;
      let ya = y;
      let xb = x - 1;
      let yb = y;
      if (sector === 1) {
        ya = y - 1;
        yb = y + 1;
      }
      if (sector === 2) {
        xa = x;
        ya = y + 1;
        xb = x;
        yb = y - 1;
      }
      if (sector === 3) {
        ya = y + 1;
        yb = y - 1;
      }
      if (xa < 0) xa = 0;
      if (xa > this.constants.nm1) xa = this.constants.nm1;
      if (xb < 0) xb = 0;
      if (xb > this.constants.nm1) xb = this.constants.nm1;
      if (ya < 0) ya = 0;
      if (ya > this.constants.nm1) ya = this.constants.nm1;
      if (yb < 0) yb = 0;
      if (yb > this.constants.nm1) yb = this.constants.nm1;

      const ma = Math.floor(a[ya][xa] * 0.25);
      const mb = Math.floor(a[yb][xb] * 0.25);
      let out = 0;
      if (m >= ma && m >= mb) out = m;
      return out;
    }, true);

    const kThresh = mk(function (a) {
      const v = a[this.thread.y][this.thread.x];
      let out = 0;
      if (v > this.constants.lo) out = 1;
      if (v > this.constants.hi) out = 2;
      return out;
    }, true);

    const hystBody = function (a) {
      const x = this.thread.x;
      const y = this.thread.y;
      const v = a[y][x];
      let out = v;
      if (v > 0.5 && v < 1.5) {
        let strong = 0;
        for (let ky = 0; ky < 3; ky++) {
          let yy = y + ky - 1;
          if (yy < 0) yy = 0;
          if (yy > this.constants.nm1) yy = this.constants.nm1;
          for (let kx = 0; kx < 3; kx++) {
            let xx = x + kx - 1;
            if (xx < 0) xx = 0;
            if (xx > this.constants.nm1) xx = this.constants.nm1;
            if (a[yy][xx] > 1.5) strong = 1;
          }
        }
        if (strong > 0.5) out = 2;
      }
      return out;
    };
    // Two pipelined instances to ping-pong, plus one unpipelined so the last
    // pass resolves on a real array rather than a handle to queued work.
    const kHystA = mk(hystBody, true);
    const kHystB = mk(hystBody, true);
    const kHystLast = mk(hystBody, false);

    const all = [kBlur, kGrad, kNms, kThresh, kHystA, kHystB, kHystLast];

    return {
      async run() {
        // Nothing between these lines touches the host. Four 16 MB
        // intermediates stay resident; only the edge map comes back.
        let t = await kBlur(rows);
        t = await kGrad(t);
        t = await kNms(t);
        t = await kThresh(t);
        for (let p = 0; p < hyst - 1; p++) t = await (p % 2 === 0 ? kHystA : kHystB)(t);
        return await kHystLast(t);
      },
      backend: () => kBlur.kernel && kBlur.kernel.constructor.mode,
      destroy() {
        all.forEach(k => k.destroy && k.destroy());
      },
    };
  },

  /**
   * Hand-written WebGPU, borrowing nothing from gpu.js. Five entry points in one
   * module over six storage buffers, and every dispatch of the whole pipeline
   * recorded into ONE compute pass: WebGPU orders dispatches within a pass and
   * makes each one's writes visible to the next, so the entire chain is a single
   * submit and a single read-back. The bind group layout is declared explicitly
   * rather than left to `layout: 'auto'` so all five pipelines can share it and
   * the stages can be wired together by swapping bind groups.
   */
  async webgpu(device, { n, nm1, hi, lo, hyst }, { src }) {
    const bytes = n * n * 4;
    const S = GPUBufferUsage.STORAGE;

    const bufSrc = device.createBuffer({
      size: bytes,
      usage: S | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true,
    });
    new Float32Array(bufSrc.getMappedRange()).set(src);
    bufSrc.unmap();

    const mkBuf = () => device.createBuffer({ size: bytes, usage: S | GPUBufferUsage.COPY_SRC });
    const b1 = mkBuf();
    const b2 = mkBuf();
    const b3 = mkBuf();
    const bA = mkBuf();
    const bB = mkBuf();
    const read = device.createBuffer({
      size: bytes,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    const TILE = 16;
    const module = device.createShaderModule({
      code: `
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read_write> o: array<f32>;

const N: i32 = ${n};
const NM1: i32 = ${nm1};
const HI: f32 = ${hi}.0;
const LO: f32 = ${lo}.0;

fn w5(k: i32) -> f32 {
  if (k == 0 || k == 4) { return 1.0; }
  if (k == 1 || k == 3) { return 4.0; }
  return 6.0;
}

@compute @workgroup_size(${TILE}, ${TILE})
fn blur(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = i32(gid.x);
  let y = i32(gid.y);
  if (x >= N || y >= N) { return; }
  var acc = 0.0;
  for (var ky = 0; ky < 5; ky++) {
    let yy = clamp(y + ky - 2, 0, NM1) * N;
    let wy = w5(ky);
    for (var kx = 0; kx < 5; kx++) {
      let xx = clamp(x + kx - 2, 0, NM1);
      acc = acc + wy * w5(kx) * a[yy + xx];
    }
  }
  o[y * N + x] = acc;
}

@compute @workgroup_size(${TILE}, ${TILE})
fn grad(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = i32(gid.x);
  let y = i32(gid.y);
  if (x >= N || y >= N) { return; }
  let xl = max(x - 1, 0);
  let xr = min(x + 1, NM1);
  let yu = max(y - 1, 0) * N;
  let yd = min(y + 1, NM1) * N;
  let row = y * N;

  let p00 = a[yu + xl];
  let p01 = a[yu + x];
  let p02 = a[yu + xr];
  let p10 = a[row + xl];
  let p12 = a[row + xr];
  let p20 = a[yd + xl];
  let p21 = a[yd + x];
  let p22 = a[yd + xr];
  let gx = p02 + 2.0 * p12 + p22 - (p00 + 2.0 * p10 + p20);
  let gy = p20 + 2.0 * p21 + p22 - (p00 + 2.0 * p01 + p02);
  let ax = abs(gx);
  let ay = abs(gy);

  var sector = 3.0;
  if (2.0 * ay < ax) { sector = 0.0; }
  else if (2.0 * ax < ay) { sector = 2.0; }
  else if (gx * gy < 0.0) { sector = 1.0; }
  o[row + x] = (ax + ay) * 4.0 + sector;
}

@compute @workgroup_size(${TILE}, ${TILE})
fn nms(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = i32(gid.x);
  let y = i32(gid.y);
  if (x >= N || y >= N) { return; }
  let v = a[y * N + x];
  let m = floor(v * 0.25);
  let sector = v - 4.0 * m;

  var xa = x + 1;
  var ya = y;
  var xb = x - 1;
  var yb = y;
  if (sector == 1.0) { ya = y - 1; yb = y + 1; }
  else if (sector == 2.0) { xa = x; ya = y + 1; xb = x; yb = y - 1; }
  else if (sector == 3.0) { ya = y + 1; yb = y - 1; }

  let ma = floor(a[clamp(ya, 0, NM1) * N + clamp(xa, 0, NM1)] * 0.25);
  let mb = floor(a[clamp(yb, 0, NM1) * N + clamp(xb, 0, NM1)] * 0.25);
  var out = 0.0;
  if (m >= ma && m >= mb) { out = m; }
  o[y * N + x] = out;
}

@compute @workgroup_size(${TILE}, ${TILE})
fn thresh(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = i32(gid.x);
  let y = i32(gid.y);
  if (x >= N || y >= N) { return; }
  let v = a[y * N + x];
  var out = 0.0;
  if (v > LO) { out = 1.0; }
  if (v > HI) { out = 2.0; }
  o[y * N + x] = out;
}

@compute @workgroup_size(${TILE}, ${TILE})
fn hyst(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = i32(gid.x);
  let y = i32(gid.y);
  if (x >= N || y >= N) { return; }
  let v = a[y * N + x];
  var out = v;
  if (v > 0.5 && v < 1.5) {
    var strong = false;
    for (var ky = 0; ky < 3; ky++) {
      let yy = clamp(y + ky - 1, 0, NM1) * N;
      for (var kx = 0; kx < 3; kx++) {
        let xx = clamp(x + kx - 1, 0, NM1);
        if (a[yy + xx] > 1.5) { strong = true; }
      }
    }
    if (strong) { out = 2.0; }
  }
  o[y * N + x] = out;
}`,
    });

    const layout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      ],
    });
    const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [layout] });
    const pipe = entryPoint =>
      device.createComputePipeline({ layout: pipelineLayout, compute: { module, entryPoint } });
    const pBlur = pipe('blur');
    const pGrad = pipe('grad');
    const pNms = pipe('nms');
    const pThresh = pipe('thresh');
    const pHyst = pipe('hyst');

    const bind = (from, to) =>
      device.createBindGroup({
        layout,
        entries: [
          { binding: 0, resource: { buffer: from } },
          { binding: 1, resource: { buffer: to } },
        ],
      });
    const gSrc1 = bind(bufSrc, b1);
    const g12 = bind(b1, b2);
    const g23 = bind(b2, b3);
    const g3A = bind(b3, bA);
    const gAB = bind(bA, bB);
    const gBA = bind(bB, bA);

    const groups = Math.ceil(n / TILE);
    // Hysteresis starts in A and alternates, so an even pass count ends in A.
    const final = hyst % 2 === 0 ? bA : bB;

    return {
      async run() {
        const enc = device.createCommandEncoder();
        const pass = enc.beginComputePass();
        pass.setPipeline(pBlur);
        pass.setBindGroup(0, gSrc1);
        pass.dispatchWorkgroups(groups, groups);
        pass.setPipeline(pGrad);
        pass.setBindGroup(0, g12);
        pass.dispatchWorkgroups(groups, groups);
        pass.setPipeline(pNms);
        pass.setBindGroup(0, g23);
        pass.dispatchWorkgroups(groups, groups);
        pass.setPipeline(pThresh);
        pass.setBindGroup(0, g3A);
        pass.dispatchWorkgroups(groups, groups);
        pass.setPipeline(pHyst);
        for (let p = 0; p < hyst; p++) {
          pass.setBindGroup(0, p % 2 === 0 ? gAB : gBA);
          pass.dispatchWorkgroups(groups, groups);
        }
        pass.end();
        enc.copyBufferToBuffer(final, 0, read, 0, bytes);
        device.queue.submit([enc.finish()]);
        await read.mapAsync(GPUMapMode.READ);
        const out = new Float32Array(read.getMappedRange()).slice();
        read.unmap();
        return out;
      },
      destroy() {
        [bufSrc, b1, b2, b3, bA, bB, read].forEach(b => b.destroy && b.destroy());
      },
    };
  },

  /**
   * Index-weighted count of surviving strong pixels. The map holds 0 (nothing),
   * 1 (weak and never joined to anything) and 2 (edge); only 2 is an edge, so
   * the weak leftovers are read as absent rather than counted. The `>= 1.5`
   * rather than `=== 2` costs nothing and means a backend that carried the map
   * through an unsigned texture at some point is still judged on what it meant.
   *
   * Every pixel is visited and the index weight means the same edge count in
   * the wrong places does not match.
   */
  reduce(out, { n }) {
    let acc = 0;
    let i = 0;
    if (ArrayBuffer.isView(out)) {
      for (; i < out.length; i++) acc += (out[i] >= 1.5 ? 1 : 0) * (1 + (i % 17));
    } else {
      for (let y = 0; y < out.length; y++) {
        const row = out[y];
        for (let x = 0; x < row.length; x++, i++) acc += (row[x] >= 1.5 ? 1 : 0) * (1 + (i % 17));
      }
    }
    return acc / (n * n);
  },
};
