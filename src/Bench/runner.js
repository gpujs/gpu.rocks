/**
 * src/Bench/runner.js — the timing protocol, in one place.
 *
 * The old benchmark's numbers were not comparable to each other: different
 * columns did different amounts of work, nothing was warmed up, and a single
 * sample decided the answer. Everything here exists to stop one of those.
 *
 * THE PROTOCOL
 *
 *   build once, warm up twice, then take the MEDIAN of at least three timed
 *   runs, and check every backend produced the same answer.
 *
 * Warm-up, because the first call to a gpu.js kernel compiles it. Timing that
 * measures a shader compiler. Median rather than mean, because a GC pause or a
 * scheduler hiccup lands in one sample and a mean carries it into the result.
 * At least three so a median exists at all.
 *
 * ADAPTIVE REPETITION. A fixed seven runs is wrong at both ends: seven runs of
 * a two-second baseline is a quarter-minute for one cell, and seven runs of a
 * 40 microsecond kernel measures the clock. After the warm-up we know roughly
 * how long one run takes, so the count is chosen to spend about a second per
 * cell, clamped to [3, 25]. The count is reported next to the number — a median
 * of 3 and a median of 25 do not deserve equal trust, and the page should say
 * which one it is showing.
 *
 * CORRECTNESS IS PART OF THE MEASUREMENT. Every column returns a checksum, and
 * a column whose checksum disagrees with the plain-JS baseline is reported as
 * WRONG, not as fast. A backend that skips work is not a backend that is quick,
 * and that failure mode is exactly what a benchmark is most likely to reward.
 */

// Roughly how long to spend per cell once the cost of a run is known.
const TARGET_MS = 1000;
const MIN_REPS = 3;
const MAX_REPS = 25;
const WARMUPS = 2;

// A cell that cannot finish one run inside this is reported as too slow rather
// than being allowed to hang the page. Deliberately generous: a plain-JS
// baseline for a big workload is legitimately seconds long.
const RUN_CEILING_MS = 20000;

// Checksums are floats summed in different orders on different hardware, so
// they never match bit for bit. This is a relative tolerance.
const CHECK_EPS = 1e-4;

export const COLUMNS = [
  { id: 'webgpu', label: 'WebGPU', sub: 'via gpu.js', kind: 'gpujs', mode: 'webgpu' },
  { id: 'webgl2', label: 'WebGL2', sub: 'via gpu.js', kind: 'gpujs', mode: 'webgl2' },
  { id: 'webgl', label: 'WebGL', sub: 'via gpu.js', kind: 'gpujs', mode: 'webgl' },
  // Between WebGL and CPU because that is where it sits in gpu.js's own
  // kernelOrder: any working GL backend outranks it, and it outranks the CPU
  // fallback. The column answers "what do you get when there is no GPU at all
  // but you still want the kernel compiled rather than interpreted".
  { id: 'webasm', label: 'WebASM', sub: 'via gpu.js', kind: 'gpujs', mode: 'webasm' },
  { id: 'cpu', label: 'CPU', sub: 'via gpu.js', kind: 'gpujs', mode: 'cpu' },
  { id: 'bare-webgpu', label: 'WebGPU', sub: 'no gpu.js runtime', kind: 'webgpu', bare: true },
  { id: 'bare-js', label: 'CPU', sub: 'no gpu.js · baseline', kind: 'js', bare: true, baseline: true },
];

export const BASELINE = 'bare-js';

const now = () => (typeof performance !== 'undefined' ? performance.now() : Date.now());

function median(xs) {
  const s = [...xs].sort((a, b) => a - b);
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}

/**
 * Times an already-built runnable. `run` must return a promise that settles
 * only when the work is genuinely finished — for a GPU backend that means the
 * result has been read back, because a dispatch that has not been awaited has
 * not been measured.
 */
async function timeIt(run) {
  for (let i = 0; i < WARMUPS; i++) {
    const t0 = now();
    await run();
    if (now() - t0 > RUN_CEILING_MS) {
      return { tooSlow: true, ms: now() - t0, reps: 0 };
    }
  }

  const probe = now();
  const value = await run();
  const one = now() - probe;

  const reps = Math.max(MIN_REPS, Math.min(MAX_REPS, Math.round(TARGET_MS / Math.max(one, 0.001))));
  const samples = [one];
  for (let i = 1; i < reps; i++) {
    const t0 = now();
    await run();
    samples.push(now() - t0);
  }
  return { ms: median(samples), reps: samples.length, min: Math.min(...samples), value };
}

/**
 * Why WebGPU is or is not usable here — the reason, not just a boolean.
 *
 * There are three quite different ways to have no WebGPU, and reporting all of
 * them as "no adapter" sends people hunting for a driver problem they do not
 * have. The common one by far is the first: navigator.gpu is only exposed in a
 * SECURE CONTEXT, and `http://<lan-ip>` is not one — localhost is, by special
 * case, so a page that works on the dev machine reports no WebGPU the moment it
 * is opened from a phone on the same network.
 */
export async function webgpuStatus() {
  if (typeof window !== 'undefined' && window.isSecureContext === false) {
    return { ok: false, reason: 'WebGPU needs a secure context — this page is plain http, so navigator.gpu is not exposed. Use https or localhost.' };
  }
  if (typeof navigator === 'undefined' || !navigator.gpu) {
    return { ok: false, reason: 'this browser does not implement WebGPU' };
  }
  try {
    const adapter = await navigator.gpu.requestAdapter();
    return adapter
      ? { ok: true, reason: 'WebGPU adapter present' }
      : { ok: false, reason: 'WebGPU is implemented but no adapter is available here' };
  } catch (e) {
    return { ok: false, reason: `requestAdapter threw: ${String(e.message || e).slice(0, 80)}` };
  }
}

/** Boolean form, for the column guards. */
export async function webgpuAvailable() {
  return (await webgpuStatus()).ok;
}

async function webgpuDevice() {
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error('no WebGPU adapter');
  return adapter.requestDevice();
}

/**
 * Runs one workload across every column. Yields a result per column through
 * `onCell` as it goes, so a long row fills in rather than appearing at the end.
 */
export async function runWorkload(workload, { GPU, onCell, signal, makeCanvas, columns } = {}) {
  const size = workload.size;
  const inputs = workload.make ? workload.make(size) : null;
  const cells = {};

  // The baseline first, and always: every other column is reported relative to
  // it, and its checksum is what the others are judged against.
  let baselineCheck = null;

  // The baseline is never optional: every other cell is reported as a ratio to
  // it, so a table without it has no speed-ups in it at all.
  const wanted = columns && columns.length ? new Set([...columns, BASELINE]) : null;
  const ordered = [...COLUMNS]
    .filter(c => !wanted || wanted.has(c.id))
    .sort((a, b) => (a.id === BASELINE ? -1 : b.id === BASELINE ? 1 : 0));

  for (const col of ordered) {
    if (signal && signal.aborted) break;
    // Say which column is being measured before measuring it. Without this the
    // table shows a dash for however long the cell takes — up to 20 s for a
    // plain-JS baseline — and a reader cannot tell working from stuck.
    if (onCell) onCell(col.id, { running: true });
    let cell;
    try {
      cell = await runColumn(workload, col, { GPU, size, inputs, makeCanvas });
      if (cell.value !== undefined && workload.reduce) {
        cell.check = workload.reduce(cell.value, size);
        delete cell.value;
        // A checksum that is not a number means the workload could not read its
        // own output — and "we could not check this" is not "this is correct".
        // Skipping it silently is how matmul shipped with five of its seven
        // columns unverified: its reduce flattened with [].concat(...out),
        // which does not spread Float32Array rows, so every 2-D gpu.js result
        // reduced to NaN. NaN failed the isFinite test below, the comparison
        // was skipped, and the cell rendered a time as though it had passed.
        // JSON.stringify then wrote it out as `check: null`, which reads like
        // "this column has no checksum" rather than "this column has no idea".
        if (!Number.isFinite(cell.check)) {
          cell.error = `checksum is not a finite number (${cell.check}) — reduce() could not read this output`;
        } else if (col.id === BASELINE) {
          baselineCheck = cell.check;
        } else if (baselineCheck != null && Number.isFinite(baselineCheck)) {
          const scale = Math.max(Math.abs(baselineCheck), 1e-9);
          if (Math.abs(cell.check - baselineCheck) / scale > CHECK_EPS) cell.wrong = true;
        }
      }
    } catch (e) {
      cell = { error: String((e && e.message) || e).slice(0, 160) };
    }
    cells[col.id] = cell;
    if (onCell) onCell(col.id, cell);
  }
  return cells;
}

async function runColumn(workload, col, { GPU, size, inputs, makeCanvas }) {
  if (col.kind === 'js') {
    if (!workload.js) return { na: true, reason: 'no plain-JS reference' };
    return timeIt(() => workload.js(size, inputs));
  }

  if (col.kind === 'webgpu') {
    if (!workload.webgpu) return { na: true, reason: workload.webgpuReason || 'no bare WebGPU implementation' };
    const status = await webgpuStatus();
    if (!status.ok) return { na: true, reason: status.reason };
    const device = await webgpuDevice();
    const built = await workload.webgpu(device, size, inputs);
    try {
      return await timeIt(() => built.run());
    } finally {
      if (built.destroy) built.destroy();
      if (device.destroy) device.destroy();
    }
  }

  // gpu.js, pinned to one backend. `mode` is explicit on purpose: 'gpu' would
  // silently pick whichever of webgl2/webgl exists, and then two columns would
  // be the same measurement wearing different labels.
  if (col.mode === 'webgpu') {
    const status = await webgpuStatus();
    if (!status.ok) return { na: true, reason: status.reason };
  }
  // WebAssembly is everywhere a modern browser is, but the column should say
  // so rather than throw if it ever is not — and an old gpu.js without the
  // backend registered would otherwise fail with a bare "unknown mode".
  if (col.mode === 'webasm') {
    const supported = typeof GPU.isWebAssemblySupported === 'boolean'
      ? GPU.isWebAssemblySupported
      : typeof WebAssembly !== 'undefined';
    if (!supported) return { na: true, reason: 'this browser has no WebAssembly' };
  }
  if (workload.declines && workload.declines.includes(col.mode)) {
    return { na: true, reason: workload.declinesReason || 'this kernel cannot run on that backend' };
  }

  let gpu = null;
  let built = null;
  try {
    // a worker has no document for gpu.js to take a canvas from
    gpu = new GPU(makeCanvas ? { mode: col.mode, canvas: makeCanvas() } : { mode: col.mode });
    built = await workload.gpujs(gpu, size, inputs);
    const out = await timeIt(() => built.run());
    // gpu.js swaps in a CPU kernel rather than failing when a kernel will not
    // compile, and a silently-CPU "WebGL" column is a lie. So ask what actually
    // ran — `static get mode` survives minification, `constructor.name` does not.
    //
    // Both GL kernels report 'gpu', not 'webgl2'/'webgl', so this compares
    // FAMILIES. That is enough: an explicit `mode: 'webgl2'` throws outright
    // when WebGL2 is unsupported (gpu.js only downgrades for mode 'gpu'), so a
    // GL column cannot quietly become the other one. What it can become is CPU,
    // and that is exactly what this catches.
    const expected = { webgpu: 'webgpu', webgl2: 'gpu', webgl: 'gpu', webasm: 'webasm', cpu: 'cpu' }[col.mode];
    const actual = built.backend && built.backend();
    if (actual && expected && actual !== expected) out.fellBackTo = actual;

    // Which lowering ran. A pipelined row can reach a fused executor, or fall
    // back to the generic one, or reach fused-sync where fused-threaded was
    // possible — and those are different products with the same name. The
    // table shows what gpu.js gives you by DEFAULT rather than pinning it to
    // the slowest common denominator, so the cell has to say when the default
    // it got was not the best one available. Same reasoning as fellBackTo:
    // disclose, do not suppress.
    if (built.executor) {
      const kind = built.executor();
      if (kind && kind !== 'fused-threaded' && kind !== 'fused-encoder') out.executor = kind;
    }
    return out;
  } finally {
    if (built && built.destroy) built.destroy();
    if (gpu && gpu.destroy) await gpu.destroy();
  }
}
