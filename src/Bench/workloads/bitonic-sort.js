/**
 * Bitonic sort — the sort a GPU can actually express.
 *
 * A bitonic network compares a FIXED set of pairs in a FIXED order: pass (k, j)
 * always pairs element i with element i xor j and always keeps the smaller one
 * on the side the k-bit of i says. Nothing about the schedule depends on the
 * data. That is why it survives on a GPU when quicksort and radix sort do not —
 * every thread knows, before any value is read, exactly which two slots it is
 * responsible for and which one it writes. It is a gather, and gather is the
 * only thing a gpu.js kernel can do.
 *
 * The price is arithmetic: 253 passes over 2^22 elements, log²n/2 compare-
 * exchanges per element, where a comparison sort needs log n. A GPU pays that
 * bill in parallelism. A CPU just pays it.
 *
 * READ THIS BEFORE READING THE ROW. The plain-JS column runs the SAME network,
 * because a row where two columns run different algorithms is not a benchmark.
 * It is not what a JS programmer would write: `Float32Array.prototype.sort`
 * sorts these same 2^22 floats in about 410 ms on the machine this was written
 * on, roughly 4.5× faster than the bitonic network below. So the speed-up
 * printed on this row is the speed-up over *a bitonic sort in JS*, and the
 * honest comparison against the sort you would really write is 410 ms against
 * whatever the GPU columns say. Both numbers are worth knowing; only one of
 * them fits in a table cell.
 *
 * The companion row is radix-sort: same keys, same count, a sort whose schedule
 * DOES depend on the data — and four N/A cells to show what that costs.
 */

const LOG = 22;
const N = 1 << LOG;
const PASSES = (LOG * (LOG + 1)) / 2; // 253 for 2^22

// Deterministic and cheap: two columns must be handed the same bytes.
function fill(a, seed) {
  let s = seed >>> 0;
  for (let i = 0; i < a.length; i++) {
    s = (s * 1664525 + 1013904223) >>> 0;
    a[i] = (s >>> 8) / 0x1000000;
  }
  return a;
}

// The (k, j) schedule, built once. Identical for every column — this list IS
// the algorithm, and sharing it is how the columns are kept honest.
function schedule(n) {
  const out = [];
  for (let k = 2; k <= n; k <<= 1) for (let j = k >> 1; j > 0; j >>= 1) out.push([j, k]);
  return out;
}

/**
 * One compare-exchange pass, as a gather: the thread that owns slot i decides
 * its OWN new value and returns it, rather than swapping a pair.
 *
 * The partner is i xor j, computed with a divide and a modulo rather than a
 * bitwise xor: WebGL 1's GLSL has no integer bitwise operators, so gpu.js
 * emulates them, and `floor(i / j) % 2` says the same thing in arithmetic every
 * backend has natively. j and k arrive as arguments, not constants, so the same
 * compiled kernel runs all 253 passes.
 */
const stepFn = function (arr, j, k) {
  const i = this.thread.x;
  const lowSide = Math.floor(i / j) % 2 < 0.5;
  const ascending = Math.floor(i / k) % 2 < 0.5;
  const mine = arr[i];
  const theirs = lowSide ? arr[i + j] : arr[i - j];
  // The low slot of the pair keeps the smaller value when the block ascends;
  // the high slot keeps the larger. Reversed when the block descends.
  if (lowSide === ascending) return Math.min(mine, theirs);
  return Math.max(mine, theirs);
};

export default {
  id: 'bitonic-sort',
  name: 'Bitonic sort',
  params: `2^${LOG} keys · ${PASSES} passes, fp32`,
  tag: 'fixed schedule',
  group: 'movement',
  size: { n: N },

  make({ n }) {
    return { keys: fill(new Float32Array(n), 0x27d4eb2f) };
  },

  /**
   * The oracle: the same network, in place, with the pair handled once by its
   * low element. In-place and swap-based because that is the fast way to run a
   * sorting network on one core — the gather form the GPU columns use would
   * double the reads for no benefit here. Same comparisons, same result.
   *
   * The copy is deliberate: make() is shared by every column and run() is
   * called a dozen times, so the input must survive being sorted.
   */
  js({ n }, { keys }) {
    const a = keys.slice();
    for (let k = 2; k <= n; k <<= 1) {
      for (let j = k >> 1; j > 0; j >>= 1) {
        for (let i = 0; i < n; i++) {
          const l = i ^ j;
          if (l > i) {
            const x = a[i];
            const y = a[l];
            if ((x > y) === ((i & k) === 0)) {
              a[i] = y;
              a[l] = x;
            }
          }
        }
      }
    }
    return a;
  },

  gpujs(gpu, { n }, { keys }) {
    const steps = schedule(n);

    // Three kernels from one body. `first` exists only so that the ping-pong
    // pair never sees its argument change type: on the WebGL backend an Array
    // argument and a pipeline Texture argument compile to different kernels, so
    // a single kernel fed an array on pass 1 and a texture on pass 2 would
    // recompile on every run and time the shader compiler.
    const first = gpu.createKernel(stepFn).setOutput([n]).setPipeline(true);
    const ping = gpu.createKernel(stepFn).setOutput([n]).setPipeline(true);
    const pong = gpu.createKernel(stepFn).setOutput([n]).setPipeline(true);

    return {
      // 253 dispatches, and the array never comes back to the host between
      // them: each pass hands the next one a texture. Only the last result is
      // read back — which is also the only thing that proves the work finished.
      async run() {
        let cur = await first(keys, steps[0][0], steps[0][1]);
        for (let s = 1; s < steps.length; s++) {
          const kernel = s % 2 === 1 ? ping : pong;
          cur = await kernel(cur, steps[s][0], steps[s][1]);
        }
        return cur.toArray ? cur.toArray() : cur;
      },
      backend: () => ping.kernel && ping.kernel.constructor.mode,
      destroy() {
        [first, ping, pong].forEach(k => k.destroy && k.destroy());
      },
    };
  },

  /**
   * Hand-written WebGPU. Same 253 passes, but in place and with half the
   * threads: only the low element of each pair is dispatched, and it writes
   * both slots. No thread touches a slot another thread in the same pass owns,
   * so no synchronisation is needed inside a pass, and WebGPU orders one
   * dispatch after the last automatically.
   *
   * The 253 (j, k) pairs live in one uniform buffer at 256-byte strides with a
   * bind group per pass. That is the WebGPU way to vary a uniform inside a
   * single submit: queue.writeBuffer would be ordered against the submit as a
   * whole, not interleaved between dispatches, and would silently give all 253
   * passes the last pass's parameters.
   */
  async webgpu(device, { n }, { keys }) {
    const steps = schedule(n);
    const bytes = n * 4;

    const src = device.createBuffer({
      size: bytes,
      usage: GPUBufferUsage.COPY_SRC,
      mappedAtCreation: true,
    });
    new Float32Array(src.getMappedRange()).set(keys);
    src.unmap();

    const work = device.createBuffer({
      size: bytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    const read = device.createBuffer({ size: bytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });

    const STRIDE = 256; // minUniformBufferOffsetAlignment
    const params = device.createBuffer({
      size: steps.length * STRIDE,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const staging = new Uint32Array((steps.length * STRIDE) / 4);
    for (let s = 0; s < steps.length; s++) {
      staging[(s * STRIDE) / 4] = steps[s][0];
      staging[(s * STRIDE) / 4 + 1] = steps[s][1];
    }
    device.queue.writeBuffer(params, 0, staging);

    const module = device.createShaderModule({
      code: `
struct P { j: u32, k: u32 };
@group(0) @binding(0) var<storage, read_write> data: array<f32>;
@group(0) @binding(1) var<uniform> p: P;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let t = gid.x;
  let j = p.j;
  // t enumerates the slots whose j-bit is clear, in order.
  let i = (t / j) * (j << 1u) + (t % j);
  let l = i + j;
  let a = data[i];
  let b = data[l];
  if ((a > b) == ((i & p.k) == 0u)) {
    data[i] = b;
    data[l] = a;
  }
}`,
    });

    const pipeline = device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: 'main' } });
    const layout = pipeline.getBindGroupLayout(0);
    const binds = steps.map((_, s) =>
      device.createBindGroup({
        layout,
        entries: [
          { binding: 0, resource: { buffer: work } },
          { binding: 1, resource: { buffer: params, offset: s * STRIDE, size: 16 } },
        ],
      })
    );
    const groups = Math.ceil(n / 2 / 256);

    return {
      async run() {
        const enc = device.createCommandEncoder();
        // The sort is in place, so every run starts from the pristine copy —
        // the same slice() the JS column pays for.
        enc.copyBufferToBuffer(src, 0, work, 0, bytes);
        const pass = enc.beginComputePass();
        pass.setPipeline(pipeline);
        for (let s = 0; s < binds.length; s++) {
          pass.setBindGroup(0, binds[s]);
          pass.dispatchWorkgroups(groups);
        }
        pass.end();
        enc.copyBufferToBuffer(work, 0, read, 0, bytes);
        device.queue.submit([enc.finish()]);
        await read.mapAsync(GPUMapMode.READ);
        const out = new Float32Array(read.getMappedRange()).slice();
        read.unmap();
        return out;
      },
      destroy() {
        [src, work, read, params].forEach(b => b.destroy && b.destroy());
      },
    };
  },

  /**
   * Two things at once, because a sort can be wrong in two ways. The weighted
   * sum uses a strictly increasing weight so that any transposition of two
   * different values changes it — an `i % 17` weight would hide a swap of
   * elements exactly 17 apart. The inversion count then catches what a weighted
   * sum cannot: it adds a whole 1 to a number that is about 0.75, so a single
   * pair out of order is four orders of magnitude outside the tolerance.
   *
   * Sorting moves values without doing arithmetic on them, so a correct backend
   * agrees with the oracle to the last bit, not to 1e-4.
   */
  reduce(out, { n }) {
    const a = ArrayBuffer.isView(out) ? out : Float32Array.from(out);
    let acc = 0;
    let inversions = 0;
    for (let i = 0; i < a.length; i++) {
      acc += a[i] * (1 + i / n);
      if (i > 0 && a[i] < a[i - 1]) inversions++;
    }
    return acc / n + inversions;
  },
};
