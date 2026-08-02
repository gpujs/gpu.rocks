/**
 * Seam carving — content-aware image resizing, 32 seams off a 1024 × 576 image.
 *
 * THIS ROW IS ALLOWED TO LOSE. Like `wavefront`, it is here because of the
 * shape of the dependency rather than in spite of it.
 *
 * One seam is four stages: measure the energy of every pixel, run a cumulative
 * minimum-cost dynamic program DOWN the rows, backtrack the cheapest path from
 * the bottom, and delete those pixels by shifting each row left. Stages one and
 * four are embarrassingly parallel — a million independent pixels each. Stage
 * two is not: row y needs row y-1, so a 576-row image is 576 dependent steps
 * with only ~1000 useful lanes each, and no amount of hardware collapses them.
 * Stage three is worse: the backtrack is ONE cell per row, 576 times, strictly
 * in order. And the whole thing repeats 32 times, because removing a seam
 * changes the energy field and the next seam has to be found in the new image.
 *
 * So the GPU is asked to run 18,528 dispatches where a CPU runs three tight
 * loops over a 2 MB working set that never leaves L2. It may well lose outright.
 * That is the finding, and nothing here is arranged to produce it: the kernels
 * are the ordinary formulation, every intermediate stays resident on the device,
 * and there is no read-back until the carved image comes home at the end.
 *
 * Read it next to `wavefront`, which loses for the same reason with different
 * geometry, and next to `canny`, which is the same kind of multi-stage image
 * pipeline WITHOUT a serial stage in the middle. The three together say what one
 * of them cannot: a GPU's problem is not "image work" or "many stages", it is
 * depth.
 *
 * ── WHY EVERY NUMBER IN HERE IS AN EXACT INTEGER ────────────────────────────
 *
 * Seam carving is decisions all the way down, and its decisions are unusually
 * brittle. The backtrack picks the smallest of three cumulative costs. If two of
 * them are equal, the tie is broken by whatever rule the code happens to
 * implement — and a tie broken the other way removes a DIFFERENT seam, which
 * changes the image, which changes the energy field, which changes every one of
 * the 31 seams after it. There is no "close enough" here: one flipped comparison
 * and the outputs diverge completely, and the runner would report a correct GPU
 * as WRONG.
 *
 * Ties are not rare, either. Measured on this exact image, 9.2% of the DP cells
 * have at least two of their three parents exactly equal. So the tie-break has
 * to be part of the specification, and it has to be reproducible bit for bit.
 *
 * Two things make it so, and they are the same two `canny` uses:
 *
 *   - Every value is an integer that fp32 represents exactly. The image is 8-bit
 *     grey stored as floats; the energy is the L1 gradient |dx| + |dy| of
 *     integers, so an integer <= 510; the cumulative cost is a sum of those down
 *     576 rows, measured at 2681 on this image and bounded by 293,760, both far
 *     under 2^24 = 16,777,216. Removal only copies pixels, so the image stays
 *     integral through all 32 rounds. fp32 and fp64 therefore compute the SAME
 *     numbers, not merely close ones — an fround-on-every-operation rerun of the
 *     whole carve differs from the fp64 oracle in 0 of 571,392 output pixels.
 *   - The tie-break itself is written identically three times: prefer the LEFT
 *     parent, then the middle, then the right, using strict `<` in that order,
 *     and start the seam at the LEFTMOST minimum of the bottom row. Because the
 *     values compared are bit-identical everywhere, so is the choice.
 *
 * The alternative — a real-valued energy such as sqrt(gx^2 + gy^2) — would look
 * more textbook and would make this row unusable: fp32 and fp64 would order two
 * near-equal parents differently every few thousand cells, and the row would
 * report noise. The L1 gradient is what Canny's own paper suggests for
 * cheapness; here it is load-bearing for correctness.
 *
 * ── THE BUFFER IS 1024 WIDE FOREVER; THE IMAGE INSIDE IT SHRINKS ───────────
 *
 * After s seams the image is 1024 - s columns wide, but reallocating a kernel's
 * output every round would mean recompiling it every round (gpu.js rebuilds on
 * setOutput). So all three columns keep one 1024-wide buffer and carry the live
 * width as an argument; columns at or past it are stale scratch that nothing
 * reads and reduce() ignores. The bare-WebGPU column additionally sizes each
 * dispatch to the live width, so it does exactly the work plain JS does; the
 * gpu.js columns have a fixed output and compute the dead columns too, which by
 * the last seam is 3% of wasted lanes. That 3% counts AGAINST the GPU, which is
 * the safe direction for a row that is already expected to lose.
 *
 * ── WHAT gpu.js CANNOT DO HERE, AND WHY THAT IS THE POINT OF THE COLUMN ────
 *
 * The backtrack needs the whole cumulative table: it walks up from the bottom
 * row, and the DP produced the rows top-down, so the two orders are opposite and
 * every row has to still exist when the walk starts. A hand-written WGSL kernel
 * simply keeps the table in a storage buffer and backtracks it with a single
 * thread in ONE dispatch — 576 serial steps, but no host involved.
 *
 * gpu.js cannot express either half of that. A kernel writes its entire output,
 * so there is no way to fill row y of a 1024 × 576 table without rewriting the
 * other 575 rows; and one thread cannot emit 576 values. So the gpu.js columns
 * do the only thing available: keep each DP row as its own pipeline texture (576
 * of them, live at once, deleted after the walk) and run the backtrack as 576
 * more dispatches, each advancing one row of a 576-wide state vector that every
 * other lane copies through untouched. That is 2 dispatches per row instead of
 * 1, so the gpu.js columns pay roughly twice the launch count of the bare column
 * for arithmetic that is identical. The gap between the two WebGPU columns on
 * this row is therefore larger than launch overhead alone — part of it is a
 * kernel this runtime has no way to write. That is worth knowing, and it is
 * exactly what a "via gpu.js" column is for; it is not a handicap invented here.
 *
 * ── WHAT MIGHT MISLEAD ─────────────────────────────────────────────────────
 *
 * 32 seams is a 3% resize, not the dramatic before/after picture seam carving is
 * famous for. The count was chosen to put the plain-JS baseline in the middle of
 * the sizing band (about 370 ms) at the lowest possible dispatch count; carving 300
 * seams would be the same arithmetic ten times over and would take a minute per
 * GPU column to no additional effect.
 */

const W = 1024;
const H = 576;
// Each seam costs about 11.7 ms of plain JS. 32 of them lands at ~370 ms —
// mid-band — for 32 * (576 * 2 + 2) = 36,928 gpu.js dispatches, which is already
// three times `wavefront`. More seams would only buy a bigger number.
const SEAMS = 32;

/**
 * An 8-bit grey image with structure a seam should want to avoid: a smooth
 * low-frequency background (cheap to cross) with hard-edged filled discs over it
 * (expensive), plus a little noise so the background is not exactly flat.
 *
 * Noise matters more than it looks. On a perfectly flat background every energy
 * is 0, every cumulative cost is 0, and the DP is one enormous tie — the seam
 * would be decided entirely by the tie-break and the row would be testing
 * nothing but that. +/- 5 levels of noise gives the background real texture at
 * integer resolution while leaving the discs unambiguously more expensive than
 * anything around them.
 *
 * Kept cheap as well as deterministic — the sizing script warns when make()
 * costs more than js() — hence the separable background and discs rasterised
 * over their bounding boxes rather than tested against every pixel.
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
  for (let i = 0; i < w; i++) sx[i] = Math.sin(i * 0.006);
  for (let i = 0; i < h; i++) cy[i] = Math.cos(i * 0.011);
  for (let y = 0; y < h; y++) {
    const row = y * w;
    const c = 46 * cy[y];
    for (let x = 0; x < w; x++) a[row + x] = 128 + c * sx[x];
  }

  for (let k = 0; k < 18; k++) {
    const ccx = rnd() * w;
    const ccy = rnd() * h;
    const r = 30 + rnd() * 90;
    const level = 20 + rnd() * 215;
    const r2 = r * r;
    const y0 = Math.max(0, Math.ceil(ccy - r));
    const y1 = Math.min(h - 1, Math.floor(ccy + r));
    for (let y = y0; y <= y1; y++) {
      const dy = y - ccy;
      const half = Math.sqrt(Math.max(0, r2 - dy * dy));
      const row = y * w;
      const lo = Math.max(0, Math.ceil(ccx - half));
      const hi = Math.min(w - 1, Math.floor(ccx + half));
      for (let x = lo; x <= hi; x++) a[row + x] = level;
    }
  }

  for (let i = 0; i < a.length; i++) {
    const v = a[i] + (rnd() - 0.5) * 11;
    a[i] = v < 0 ? 0 : v > 255 ? 255 : Math.round(v);
  }
  return a;
}

export default {
  id: 'seam-carving',
  name: 'Seam carving',
  params: `${W} × ${H} 8-bit · ${SEAMS} seams · ${H}-row DP each`,
  tag: 'sequential DP',
  group: 'image',
  size: { w: W, h: H, seams: SEAMS, finalW: W - SEAMS },

  make({ w, h }) {
    const src = image(w, h, 0x51a3f27b);
    const rows = [];
    for (let y = 0; y < h; y++) rows.push(src.subarray(y * w, y * w + w));
    return { src, rows };
  },

  /**
   * The oracle, and a fair baseline: four flat passes per seam in the same order
   * and with the same arithmetic as the four kernels.
   *
   * It is written the way anyone would write it — row-major everywhere, the row
   * offsets hoisted out of the inner loops, one Float32Array per stage allocated
   * once outside the seam loop, and the removal done as an in-place leftward
   * shift of the tail of each row rather than a copy into a second buffer. It
   * does NOT do the two things a serial implementation could legitimately do and
   * a GPU cannot: it does not keep the energy field between seams and patch only
   * the band the removed seam touched, and it does not stop the DP early. Both
   * would be real optimisations of the algorithm and both would make the columns
   * incomparable, so neither side gets them.
   *
   * It works on a fresh copy of the source each call, because run() is called
   * seven times and a carve that mutated the shared input would return a
   * different image every time.
   */
  js({ w, h, seams }, { src }) {
    const img = src.slice();
    const energy = new Float32Array(w * h);
    const cost = new Float32Array(w * h);
    const seam = new Int32Array(h);
    let cw = w;

    for (let s = 0; s < seams; s++) {
      const last = cw - 1;

      // ── 1. energy: L1 gradient, clamped at the live borders ───────────────
      for (let y = 0; y < h; y++) {
        const row = y * w;
        const up = (y > 0 ? y - 1 : 0) * w;
        const dn = (y < h - 1 ? y + 1 : h - 1) * w;
        for (let x = 0; x < cw; x++) {
          const xl = x > 0 ? x - 1 : 0;
          const xr = x < last ? x + 1 : last;
          const gx = img[row + xr] - img[row + xl];
          const gy = img[dn + x] - img[up + x];
          energy[row + x] = (gx < 0 ? -gx : gx) + (gy < 0 ? -gy : gy);
        }
      }

      // ── 2. cumulative cost, one row at a time. THE SERIAL STAGE ───────────
      for (let x = 0; x < cw; x++) cost[x] = energy[x];
      for (let y = 1; y < h; y++) {
        const row = y * w;
        const prev = row - w;
        for (let x = 0; x < cw; x++) {
          const xl = x > 0 ? x - 1 : 0;
          const xr = x < last ? x + 1 : last;
          let m = cost[prev + xl];
          const c = cost[prev + x];
          if (c < m) m = c;
          const r = cost[prev + xr];
          if (r < m) m = r;
          cost[row + x] = energy[row + x] + m;
        }
      }

      // ── 3. backtrack. Leftmost minimum, then prefer left / middle / right ──
      const base = (h - 1) * w;
      let bx = 0;
      let best = cost[base];
      for (let x = 1; x < cw; x++) {
        const v = cost[base + x];
        if (v < best) {
          best = v;
          bx = x;
        }
      }
      seam[h - 1] = bx;
      for (let y = h - 2; y >= 0; y--) {
        const row = y * w;
        const xl = bx > 0 ? bx - 1 : 0;
        const xr = bx < last ? bx + 1 : last;
        let bv = cost[row + xl];
        let nx = xl;
        const c = cost[row + bx];
        if (c < bv) {
          bv = c;
          nx = bx;
        }
        const r = cost[row + xr];
        if (r < bv) {
          bv = r;
          nx = xr;
        }
        bx = nx;
        seam[y] = bx;
      }

      // ── 4. remove: shift the tail of each row one column left ─────────────
      for (let y = 0; y < h; y++) {
        const row = y * w;
        for (let x = seam[y]; x < last; x++) img[row + x] = img[row + x + 1];
      }
      cw = last;
    }
    return img;
  },

  /**
   * gpu.js: ten kernels, 1,154 dispatches per seam, and nothing going back to
   * the host until the carved image at the end.
   *
   * The pipeline plumbing is doing real work here and is worth reading:
   *
   *   - A mutable pipelined kernel reuses ONE output texture, so a kernel may
   *     never be handed its own previous output as an input. The energy, remove
   *     and backtrack stages each read the stage before them, so remove and
   *     backtrack are TWO kernel objects taking turns (the `canny` hysteresis
   *     trick).
   *   - The DP row kernel is the exception: its 576 outputs must ALL still exist
   *     when the backtrack starts, so it is immutable — a fresh texture per call
   *     — and the 576 textures are deleted the moment the walk is done. Without
   *     the delete this leaks 18,432 textures per run.
   */
  gpujs(gpu, { w, h, seams }, { rows }) {
    const constants = { w, h, hm1: h - 1 };
    const mk = (fn, output, opts = {}) => {
      const k = gpu
        .createKernel(fn)
        .setConstants(constants)
        .setPipeline(opts.pipeline !== false)
        .setImmutable(Boolean(opts.immutable))
        // Cumulative costs reach the low thousands and column indices reach
        // 1023; the default precision tactic could pick a GLSL qualifier that
        // carries neither. Asked for explicitly so every backend holds the same
        // fp32 and the exact-integer argument in the header actually holds.
        .setPrecision('single')
        .setTactic('precision')
        .setOutput(output);
      // Only the backtrack has a loop, and its bottom-row scan runs to 1024 —
      // past gpu.js's default ceiling of 1000, which would silently truncate the
      // search on a GL backend and pick the wrong seam.
      if (opts.loops) k.setLoopMaxIterations(opts.loops);
      return k;
    };

    // One upload of the source per run, into a device-resident copy. The carve
    // then mutates that copy, exactly as js() mutates its own slice().
    const kCopy = mk(function (a) {
      return a[this.thread.y][this.thread.x];
    }, [w, h]);

    const kZeroRow = mk(function () {
      return 0;
    }, [w]);
    const kZeroState = mk(function () {
      return 0;
    }, [h]);

    // ── 1. energy ─────────────────────────────────────────────────────────────
    // Every clamp assigns an integer to an integer (`xr = x` rather than
    // `xr = last`), because `last` is derived from a kernel ARGUMENT and is
    // therefore a float: mixing the two in an assignment is how a gpu.js kernel
    // stops compiling.
    const kEnergy = mk(function (img, cw) {
      const x = this.thread.x;
      const y = this.thread.y;
      const last = cw - 1;
      if (x > last) return 0;
      let xl = x - 1;
      if (xl < 0) xl = 0;
      let xr = x + 1;
      if (xr > last) xr = x;
      let yu = y - 1;
      if (yu < 0) yu = 0;
      let yd = y + 1;
      if (yd > this.constants.hm1) yd = y;
      let gx = img[y][xr] - img[y][xl];
      if (gx < 0) gx = -gx;
      let gy = img[yd][x] - img[yu][x];
      if (gy < 0) gy = -gy;
      return gx + gy;
    }, [w, h]);

    // ── 2. one row of the DP. Called h times per seam, in order ──────────────
    const kRow = mk(
      function (prev, energy, yy, cw) {
        const x = this.thread.x;
        const last = cw - 1;
        if (x > last) return 0;
        const e = energy[yy][x];
        if (yy < 0.5) return e;
        let xl = x - 1;
        if (xl < 0) xl = 0;
        let xr = x + 1;
        if (xr > last) xr = x;
        let m = prev[xl];
        const c = prev[x];
        if (c < m) m = c;
        const r = prev[xr];
        if (r < m) m = r;
        return e + m;
      },
      [w],
      { immutable: true }
    );

    // ── 3. one step of the backtrack ────────────────────────────────────────
    // Output is the whole 576-entry seam; lane k rewrites only entry yy and
    // copies the rest through. 575 of the 576 lanes do nothing, which is the
    // honest cost of expressing a serial walk as a kernel.
    const traceBody = function (state, mrow, yy, cw) {
      const k = this.thread.x;
      const last = cw - 1;
      let out = state[k];
      // `k === yy` compares an integer thread index against a float argument;
      // the half-open window is the same comparison written so gpu.js does not
      // have to reconcile the two types.
      if (k > yy - 0.5 && k < yy + 0.5) {
        if (yy > this.constants.hm1 - 0.5) {
          // Bottom row: leftmost minimum over the live width.
          let bx = 0;
          let best = mrow[0];
          for (let x = 1; x < this.constants.w; x++) {
            if (x > last) break;
            const v = mrow[x];
            if (v < best) {
              best = v;
              bx = x;
            }
          }
          out = bx;
        } else {
          const c = state[k + 1];
          let xl = c - 1;
          if (xl < 0) xl = 0;
          let xr = c + 1;
          if (xr > last) xr = c;
          let bv = mrow[xl];
          out = xl;
          const mc = mrow[c];
          if (mc < bv) {
            bv = mc;
            out = c;
          }
          const mr = mrow[xr];
          if (mr < bv) {
            bv = mr;
            out = xr;
          }
        }
      }
      return out;
    };
    const kTraceA = mk(traceBody, [h], { loops: w });
    const kTraceB = mk(traceBody, [h], { loops: w });

    // ── 4. remove ───────────────────────────────────────────────────────────
    const removeBody = function (img, seam, cw) {
      const x = this.thread.x;
      const y = this.thread.y;
      const last = cw - 1;
      if (x > last - 1) return 0;
      const s = seam[y];
      let src = x;
      if (x > s - 0.5) src = x + 1;
      return img[y][src];
    };
    const kRemoveA = mk(removeBody, [w, h]);
    const kRemoveB = mk(removeBody, [w, h]);
    // The last seam resolves onto a real array rather than a handle to queued
    // work, which is what makes run() honest about having finished.
    const kRemoveLast = mk(removeBody, [w, h], { pipeline: false });

    const all = [kCopy, kZeroRow, kZeroState, kEnergy, kRow, kTraceA, kTraceB, kRemoveA, kRemoveB, kRemoveLast];

    return {
      async run() {
        const zeroRow = await kZeroRow();
        let img = await kCopy(rows);
        let cw = w;

        for (let s = 0; s < seams; s++) {
          const energy = await kEnergy(img, cw);

          // The serial stage: h dispatches, each depending on the one before.
          const table = [];
          let prev = zeroRow;
          for (let y = 0; y < h; y++) {
            prev = await kRow(prev, energy, y, cw);
            table.push(prev);
          }

          // The walk back up, h more dispatches. The state never leaves the
          // device: reading the chosen column back to feed the next step would
          // price 18,432 round trips instead of the algorithm.
          let state = await kZeroState();
          for (let y = h - 1; y >= 0; y--) {
            state = await ((h - 1 - y) % 2 === 0 ? kTraceA : kTraceB)(state, table[y], y, cw);
          }
          for (let i = 0; i < table.length; i++) if (table[i].delete) table[i].delete();

          const kRemove = s === seams - 1 ? kRemoveLast : s % 2 === 0 ? kRemoveA : kRemoveB;
          img = await kRemove(img, state, cw);
          cw = cw - 1;
        }
        return img;
      },
      backend: () => kEnergy.kernel && kEnergy.kernel.constructor.mode,
      destroy() {
        all.forEach(k => k.destroy && k.destroy());
      },
    };
  },

  /**
   * Hand-written WebGPU, borrowing nothing from gpu.js.
   *
   * Three things this column can do that the one to its left cannot:
   *
   *   - the cumulative table is an ordinary 2.4 MB storage buffer, so the DP
   *     writes row y and the backtrack reads all 576 rows afterwards.
   *   - the backtrack is ONE dispatch of ONE thread that walks the table from
   *     the bottom to the top and writes the whole seam. It is deliberately
   *     single-threaded — that is what the algorithm is — and on a GPU it costs
   *     576 dependent memory round trips with one lane awake. Making it look
   *     better would mean not doing it.
   *   - all 18,528 dispatches of all 32 seams go into ONE compute pass and one
   *     submit. Dispatches inside a pass are ordered and each one's writes are
   *     visible to the next, which is exactly the guarantee a row-by-row DP
   *     needs, so the host is not involved between seams either.
   *
   * The row index and the live width arrive as two uniforms read at dynamic
   * offsets out of pre-filled plan buffers — 576 row slots and 32 width slots,
   * reused across the pass. Rewriting a uniform between dispatches would force a
   * submit per row and the column would be measuring the queue.
   */
  async webgpu(device, { w, h, seams }, { src }) {
    const bytes = w * h * 4;
    const S = GPUBufferUsage.STORAGE;

    const source = device.createBuffer({
      size: bytes,
      usage: S | GPUBufferUsage.COPY_SRC,
      mappedAtCreation: true,
    });
    new Float32Array(source.getMappedRange()).set(src);
    source.unmap();

    const img0 = device.createBuffer({ size: bytes, usage: S | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    const img1 = device.createBuffer({ size: bytes, usage: S | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    const energy = device.createBuffer({ size: bytes, usage: S });
    const cost = device.createBuffer({ size: bytes, usage: S });
    const seam = device.createBuffer({ size: h * 4, usage: S });
    const read = device.createBuffer({ size: bytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });

    // One 256-byte slot per value — 256 is the dynamic-offset alignment.
    const STRIDE = 256;
    const mkPlan = values => {
      const data = new Uint32Array((values.length * STRIDE) / 4);
      for (let i = 0; i < values.length; i++) data[(i * STRIDE) / 4] = values[i];
      const buf = device.createBuffer({
        size: data.byteLength,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      device.queue.writeBuffer(buf, 0, data);
      return buf;
    };
    const rowPlan = mkPlan(Array.from({ length: h }, (_, y) => y));
    const widthPlan = mkPlan(Array.from({ length: seams }, (_, s) => w - s));

    const TILE = 16;
    const LANES = 64;
    const module = device.createShaderModule({
      code: `
struct U32Box { v: u32, p0: u32, p1: u32, p2: u32 };
@group(0) @binding(0) var<storage, read> imgIn: array<f32>;
@group(0) @binding(1) var<storage, read_write> imgOut: array<f32>;
@group(0) @binding(2) var<storage, read_write> energy: array<f32>;
@group(0) @binding(3) var<storage, read_write> cost: array<f32>;
@group(0) @binding(4) var<storage, read_write> seam: array<i32>;
@group(0) @binding(5) var<uniform> row: U32Box;
@group(0) @binding(6) var<uniform> live: U32Box;

const W: i32 = ${w};
const H: i32 = ${h};

@compute @workgroup_size(${TILE}, ${TILE})
fn energyPass(@builtin(global_invocation_id) gid: vec3<u32>) {
  let cw = i32(live.v);
  let x = i32(gid.x);
  let y = i32(gid.y);
  if (x >= cw || y >= H) { return; }
  let r = y * W;
  let xl = max(x - 1, 0);
  let xr = min(x + 1, cw - 1);
  let yu = max(y - 1, 0) * W;
  let yd = min(y + 1, H - 1) * W;
  energy[r + x] = abs(imgIn[r + xr] - imgIn[r + xl]) + abs(imgIn[yd + x] - imgIn[yu + x]);
}

@compute @workgroup_size(${LANES})
fn dpRow(@builtin(global_invocation_id) gid: vec3<u32>) {
  let cw = i32(live.v);
  let x = i32(gid.x);
  if (x >= cw) { return; }
  let y = i32(row.v);
  let r = y * W;
  let e = energy[r + x];
  if (y == 0) { cost[x] = e; return; }
  let p = r - W;
  let xl = max(x - 1, 0);
  let xr = min(x + 1, cw - 1);
  // min() of three exactly-representable integers: no NaNs, no rounding, and
  // the order of the reductions cannot change the answer. Only the BACKTRACK
  // needs a tie-break, and it is written out longhand below.
  cost[r + x] = e + min(min(cost[p + xl], cost[p + x]), cost[p + xr]);
}

// One thread, 576 dependent steps. This is the algorithm, not a shortcut.
@compute @workgroup_size(1)
fn tracePass() {
  let cw = i32(live.v);
  let last = cw - 1;
  let base = (H - 1) * W;
  var bx = 0;
  var best = cost[base];
  for (var x = 1; x < cw; x++) {
    let v = cost[base + x];
    if (v < best) { best = v; bx = x; }
  }
  seam[H - 1] = bx;
  for (var y = H - 2; y >= 0; y--) {
    let r = y * W;
    let xl = max(bx - 1, 0);
    let xr = min(bx + 1, last);
    // Prefer left, then middle, then right — strict '<' in that order, the same
    // three comparisons in the same order as the plain-JS oracle.
    var bv = cost[r + xl];
    var nx = xl;
    let c = cost[r + bx];
    if (c < bv) { bv = c; nx = bx; }
    let rr = cost[r + xr];
    if (rr < bv) { bv = rr; nx = xr; }
    bx = nx;
    seam[y] = bx;
  }
}

@compute @workgroup_size(${TILE}, ${TILE})
fn removePass(@builtin(global_invocation_id) gid: vec3<u32>) {
  let cw = i32(live.v);
  let x = i32(gid.x);
  let y = i32(gid.y);
  if (x >= cw - 1 || y >= H) { return; }
  let r = y * W;
  var srcX = x;
  if (x >= seam[y]) { srcX = x + 1; }
  imgOut[r + x] = imgIn[r + srcX];
}`,
    });

    const storage = { type: 'storage' };
    const readOnly = { type: 'read-only-storage' };
    const dynUniform = { type: 'uniform', hasDynamicOffset: true };
    const layout = device.createBindGroupLayout({
      entries: [readOnly, storage, storage, storage, storage, dynUniform, dynUniform].map((buffer, binding) => ({
        binding,
        visibility: GPUShaderStage.COMPUTE,
        buffer,
      })),
    });
    const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [layout] });
    const pipe = entryPoint => device.createComputePipeline({ layout: pipelineLayout, compute: { module, entryPoint } });
    const pEnergy = pipe('energyPass');
    const pDp = pipe('dpRow');
    const pTrace = pipe('tracePass');
    const pRemove = pipe('removePass');

    const bind = (from, to) =>
      device.createBindGroup({
        layout,
        entries: [from, to, energy, cost, seam, rowPlan, widthPlan].map((buf, binding) => ({
          binding,
          resource: binding >= 5 ? { buffer: buf, offset: 0, size: 16 } : { buffer: buf },
        })),
      });
    const g01 = bind(img0, img1);
    const g10 = bind(img1, img0);

    const gy = Math.ceil(h / TILE);
    // Seam s writes into the buffer it is not reading, so an even seam count
    // ends back where it started.
    const final = seams % 2 === 0 ? img0 : img1;

    return {
      async run() {
        const enc = device.createCommandEncoder();
        // Start every run from the pristine image, as js() does with slice().
        enc.copyBufferToBuffer(source, 0, img0, 0, bytes);
        const pass = enc.beginComputePass();

        for (let s = 0; s < seams; s++) {
          const cw = w - s;
          const g = s % 2 === 0 ? g01 : g10;
          const off = [0, s * STRIDE];
          // Dispatch only over the live width, so this column does exactly the
          // arithmetic the plain-JS baseline does and not a lane more.
          pass.setPipeline(pEnergy);
          pass.setBindGroup(0, g, off);
          pass.dispatchWorkgroups(Math.ceil(cw / TILE), gy);

          pass.setPipeline(pDp);
          const groups = Math.ceil(cw / LANES);
          for (let y = 0; y < h; y++) {
            pass.setBindGroup(0, g, [y * STRIDE, s * STRIDE]);
            pass.dispatchWorkgroups(groups);
          }

          pass.setPipeline(pTrace);
          pass.setBindGroup(0, g, off);
          pass.dispatchWorkgroups(1);

          pass.setPipeline(pRemove);
          pass.setBindGroup(0, g, off);
          pass.dispatchWorkgroups(Math.ceil((cw - 1) / TILE), gy);
        }

        pass.end();
        enc.copyBufferToBuffer(final, 0, read, 0, bytes);
        device.queue.submit([enc.finish()]);
        // The read-back is the only thing that proves 18,528 dispatches ran.
        await read.mapAsync(GPUMapMode.READ);
        const out = new Float32Array(read.getMappedRange()).slice();
        read.unmap();
        return out;
      },
      destroy() {
        [source, img0, img1, energy, cost, seam, read, rowPlan, widthPlan].forEach(b => b.destroy && b.destroy());
      },
    };
  },

  /**
   * Every pixel of the CARVED image — the 992 × 576 that survived — index
   * weighted so two rows carved differently cannot produce the same total.
   *
   * The columns from finalW to 1023 are deliberately skipped: they are the tail
   * of the fixed-width scratch buffer described in the header, and no backend
   * writes anything meaningful there. Requiring them to agree would be requiring
   * three implementations to leave identical litter.
   *
   * Every value in the sum is an 8-bit integer and the weights are small
   * integers, so the total is exact in fp64 and identical across backends to the
   * last bit. The runner's 1e-4 tolerance is slack this row never spends — and
   * must not, since a single differently-broken tie changes hundreds of pixels.
   */
  reduce(out, { w, h, finalW }) {
    const flat = ArrayBuffer.isView(out);
    let acc = 0;
    let i = 0;
    for (let y = 0; y < h; y++) {
      const row = flat ? null : out[y];
      const base = y * w;
      for (let x = 0; x < finalW; x++, i++) {
        acc += (flat ? out[base + x] : row[x]) * (1 + (i % 17));
      }
    }
    return acc / (h * finalW);
  },
};
