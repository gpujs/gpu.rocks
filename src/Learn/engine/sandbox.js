// engine/sandbox.js — the execution core: everything that actually runs
// learner code against gpu.js.
//
// This module is loaded in TWO places and must not care which:
//   • sandbox.worker.js — the normal path. The worker owns the code, the GPU
//     instances, the kernels, the canvases and the task's tests, so a runaway
//     kernel can be killed with terminate() instead of freezing the page.
//   • runner.js's fallback — when the browser has no Worker or no
//     OffscreenCanvas, the same functions run on the main thread exactly as
//     they did before the worker existed (the pre-flight guard still applies,
//     but a `while (true)` is unrecoverable there — nothing can be done).
//
// Nothing here touches `document` directly; canvas work goes through
// ./utils.js, which branches on the environment.

import { CPUKernel, GPU, utils as gpuUtils } from 'gpu.js';
import {
  utils,
  assert,
  assertClose,
  formatValue,
  readCanvasPixels,
  snapshotCanvas,
  timeString,
  toErrorMessage,
} from './utils';

// True when this copy of the module is running inside the worker sandbox.
const IN_WORKER = typeof document === 'undefined';

// One-line disclosure of which execution path a run used, emitted as the first
// `system` log line of every run. The learner sees it because it changes what
// happens to their code: only one of these two paths can be interrupted.
const SANDBOX_NOTE = IN_WORKER
  ? '▸ sandbox: Web Worker — a runaway kernel can be stopped'
  : '▸ sandbox: main thread (no Worker sandbox in use) — a runaway kernel cannot be stopped';

const IDENT_RE = /^[A-Za-z_$][A-Za-z0-9_$]*$/;

// GPU instances created by the previous run; destroyed on the next run.
let previousInstances = [];

async function destroyPreviousRun() {
  const instances = previousInstances;
  previousInstances = [];
  for (const gpu of instances) {
    try {
      await gpu.destroy();
    } catch (e) {
      // a lost context is fine — we are throwing it away anyway
    }
  }
}

export function gpuSupported() {
  try {
    return Boolean(GPU.isGPUSupported);
  } catch (e) {
    return false;
  }
}

// Did gpu.js build a CPU kernel for this run-shortcut? Identity, not name:
// esbuild mangles `class CPUKernel` down to `class C` in the production build,
// so a constructor.name check works in dev and silently never fires in the
// built site — which is the one place the disclosure matters.
function isCpuKernel(k) {
  try {
    const built = k.kernel;
    if (!built) return false;
    if (typeof CPUKernel === 'function' && built instanceof CPUKernel) return true;
    return Boolean(built.constructor && built.constructor.name === 'CPUKernel');
  } catch (e) {
    return false;
  }
}

function makeConsoleProxy(push) {
  const real = typeof console !== 'undefined' ? console : null;
  const capture = type => (...args) => {
    if (real && real[type]) real[type](...args);
    push({
      type: type === 'warn' ? 'warn' : type === 'error' ? 'error' : 'log',
      time: timeString(),
      text: args.map(a => formatValue(a)).join(' '),
    });
  };
  return {
    log: capture('log'),
    info: capture('info'),
    debug: capture('debug'),
    warn: capture('warn'),
    error: capture('error'),
  };
}

// The course convention for images is nested arrays image[y][x] = [r,g,b,a].
// gpu.js auto-infers those as a flat 3D 'Array', which the GL backend cannot
// partially index (`const pixel = image[y][x]`). Typing them explicitly as
// 'Array2D(4)' — a 2D array of vec4 pixels — makes the same kernel source
// compile on both backends. Applied before the kernel's first build, and only
// when the author has not set argumentTypes themselves.
function isImagePixel(v) {
  return (Array.isArray(v) || ArrayBuffer.isView(v)) && v.length === 4 && typeof v[0] === 'number';
}

function isImageLike(arg) {
  return Array.isArray(arg) && arg.length > 0 &&
    Array.isArray(arg[0]) && arg[0].length > 0 &&
    isImagePixel(arg[0][0]);
}

function applyImageArgumentTypes(target, args) {
  try {
    const kernel = target.kernel;
    if (!kernel || kernel.built || kernel.argumentTypes) return;
    const names = kernel.argumentNames || [];
    if (names.length !== args.length) return;
    let hasImage = false;
    const types = args.map(arg => {
      if (isImageLike(arg)) {
        hasImage = true;
        return 'Array2D(4)';
      }
      const inferred = gpuUtils.getVariableType(arg, kernel.strictIntegers);
      return inferred === 'Integer' ? 'Number' : inferred;
    });
    if (hasImage) target.setArgumentTypes(types);
  } catch (e) {
    // typing is a compatibility shim — never let it break a run
  }
}

// Wraps a kernel-run shortcut so every invocation records .lastArgs and the
// first invocation emits a "kernel compiled" system log line.
function patchKernel(kernel, push) {
  let announced = false;
  const proxy = new Proxy(kernel, {
    apply(target, thisArg, args) {
      target.lastArgs = args;
      applyImageArgumentTypes(target, args);
      const result = Reflect.apply(target, thisArg, args);
      if (!announced) {
        announced = true;
        try {
          const built = target.kernel;
          const output = built && built.output ? Array.from(built.output) : null;
          if (output && output.length) {
            const threads = output.reduce((a, b) => a * b, 1);
            push({
              type: 'system',
              time: timeString(),
              text: `▸ kernel compiled · output ${output.join('×')} · ${threads.toLocaleString('en-US')} threads`,
            });
          }
        } catch (e) {
          // announcement is cosmetic — never let it break a run
        }
      }
      return result;
    },
  });
  return proxy;
}

// ---- pre-flight guard ------------------------------------------------------
//
// A kernel that does pathological per-thread work (classically: multiplying a
// whole row of an image instead of one pixel, which makes V8 stringify the
// array on every thread) turns a 512×512 run into minutes of work. In the
// worker that is now survivable — the watchdog kills it — but a 10 s stall
// with nothing to show beats no answer, and the diagnostic below explains the
// actual mistake instead of just reporting a stop. So: measure first — run the
// user's code once with every output axis clamped small, then refuse the
// full-size run if the measured cost per thread implies an absurd total.
//
// The guard fails OPEN in every ambiguous case (probe errored, nothing to
// clamp, small output): a false refusal of legitimate code would be worse
// than the freeze it prevents. It does not catch a plain `while (true)` in
// user code — only the worker's watchdog can.
const PROBE_AXIS_CAP = 64; // clamp each output axis to this during the probe
const PROBE_MIN_THREADS = 65536; // only guard runs larger than this
const RUN_BUDGET_MS = 5000; // refuse a run estimated to exceed this

function clampOutputSetting(output) {
  if (Array.isArray(output)) {
    const clamped = output.map(n => (typeof n === 'number' ? Math.min(n, PROBE_AXIS_CAP) : n));
    const threads = of => of.reduce((a, b) => a * (typeof b === 'number' ? b : 1), 1);
    return { clamped, requestedThreads: threads(output), clampedThreads: threads(clamped) };
  }
  if (output && typeof output === 'object') {
    const axes = ['x', 'y', 'z'].filter(k => typeof output[k] === 'number');
    if (!axes.length) return null;
    const clamped = { ...output };
    axes.forEach(k => { clamped[k] = Math.min(output[k], PROBE_AXIS_CAP); });
    const threads = o => axes.reduce((a, k) => a * o[k], 1);
    return { clamped, requestedThreads: threads(output), clampedThreads: threads(clamped) };
  }
  return null;
}

// Rewrites createKernel arguments so the probe runs a small slice of the work.
// Anything it cannot clamp is recorded, which makes the probe inconclusive.
function clampKernelArgs(args, stats) {
  const settings = args[1];
  const info = settings && typeof settings === 'object' ? clampOutputSetting(settings.output) : null;
  if (!info) {
    stats.unclamped = true;
    return args;
  }
  stats.requestedThreads = Math.max(stats.requestedThreads, info.requestedThreads);
  stats.clampedThreads = Math.max(stats.clampedThreads, info.clampedThreads);
  const rest = args.slice(2);
  return [args[0], { ...settings, output: info.clamped }, ...rest];
}

// Runs the code once at clamped size and extrapolates. Resolves to null when
// the full-size run should go ahead, or a description of the refusal.
async function preflight(code, mode, task) {
  let probe;
  try {
    probe = await executeRun(code, { mode, task, probe: true });
  } catch (e) {
    return null; // the probe itself is best-effort
  }
  const stats = probe.probeStats || {};
  if (!probe.ok || stats.unclamped) return null;
  if (!stats.requestedThreads || !stats.clampedThreads) return null;
  if (stats.clampedThreads >= stats.requestedThreads) return null; // nothing was clamped
  if (stats.requestedThreads <= PROBE_MIN_THREADS) return null;
  const scale = stats.requestedThreads / stats.clampedThreads;
  const estimateMs = probe.durationMs * scale;
  if (estimateMs <= RUN_BUDGET_MS) return null;
  return { probeMs: probe.durationMs, estimateMs, threads: stats.requestedThreads };
}

// ---- executeRun ------------------------------------------------------------

// executeRun(code, { mode, task, probe, onLog }) → Promise<InternalRunResult>,
// never rejects:
//   {
//     ok, error?: { message, stack? },
//     logs: [{ type, time, text?, canvas?, snapshot? }],
//     kernels: [kernel],   // LIVE — never leaves this thread
//     canvas,              // LIVE — never leaves this thread
//     resolvedMode: 'gpu' | 'cpu',
//     durationMs, fellBackToCPU, refusedAsTooSlow?, probeStats?
//   }
//
// `onLog` (optional) is called with every log entry as it is produced. The
// supervisor uses it to stream text lines to the main thread, so a run that is
// later killed by the watchdog can still show what it managed to do.
export async function executeRun(code, { mode = 'auto', task, probe = false, onLog } = {}) {
  await destroyPreviousRun();

  const logs = [];
  const push = entry => {
    logs.push(entry);
    if (onLog) {
      try {
        onLog(entry);
      } catch (e) {
        // streaming is best-effort — never let it break a run
      }
    }
  };
  const kernels = [];
  const instances = [];
  const probeStats = { requestedThreads: 0, clampedThreads: 0, unclamped: false };
  let renderedCanvas = null;

  if (!probe) push({ type: 'system', time: timeString(), text: SANDBOX_NOTE });

  const gpuOk = gpuSupported();
  let resolvedMode;
  if (mode === 'cpu') {
    resolvedMode = 'cpu';
    push({ type: 'system', time: timeString(), text: '▸ mode "cpu" → selected cpu' });
  } else if (mode === 'gpu') {
    if (gpuOk) {
      resolvedMode = 'gpu';
      push({ type: 'system', time: timeString(), text: '▸ mode "gpu" → selected gpu (WebGL)' });
    } else {
      resolvedMode = 'cpu';
      push({
        type: 'system',
        time: timeString(),
        text: '▸ mode "gpu" requested but WebGL is unavailable here — falling back to cpu',
      });
    }
  } else {
    resolvedMode = gpuOk ? 'gpu' : 'cpu';
    push({
      type: 'system',
      time: timeString(),
      text: gpuOk
        ? '▸ mode "auto" → selected gpu (WebGL)'
        : '▸ mode "auto" → selected cpu (WebGL unavailable)',
    });
  }

  // GPU subclass: forces the resolved mode, records instances and kernels.
  class RecordingGPU extends GPU {
    constructor(settings = {}) {
      super({ ...settings, mode: resolvedMode });
      instances.push(this);
      previousInstances.push(this);
    }

    createKernel(...args) {
      const built = super.createKernel(...(probe ? clampKernelArgs(args, probeStats) : args));
      const kernel = patchKernel(built, push);
      kernels.push(kernel);
      return kernel;
    }

    createKernelMap(...args) {
      const built = super.createKernelMap(...(probe ? clampKernelArgs(args, probeStats) : args));
      const kernel = patchKernel(built, push);
      kernels.push(kernel);
      return kernel;
    }
  }

  const render = canvas => {
    renderedCanvas = canvas || renderedCanvas;
    push({
      type: 'canvas',
      time: timeString(),
      text: `render: ${canvas && canvas.constructor ? canvas.constructor.name : 'canvas'}`,
      canvas: canvas || null,
      // capture pixels NOW — a later render() of this same canvas must not
      // retroactively change what this entry shows
      snapshot: snapshotCanvas(canvas),
    });
  };

  const consoleProxy = makeConsoleProxy(push);

  // Injected globals: GPU, console, render, utils, mode, plus task inputs.
  const globals = {
    GPU: RecordingGPU,
    console: consoleProxy,
    render,
    utils,
    mode: resolvedMode,
  };

  // Measure a small slice before committing to the full run.
  if (!probe) {
    const refusal = await preflight(code, mode, task);
    if (refusal) {
      const seconds = Math.round(refusal.estimateMs / 1000);
      const message =
        `refused to run: this would take about ${seconds}s and freeze the page. ` +
        `A ${PROBE_AXIS_CAP}×${PROBE_AXIS_CAP} slice took ${refusal.probeMs.toFixed(0)} ms, and the ` +
        `kernel asks for ${refusal.threads.toLocaleString('en-US')} threads. That much work per ` +
        `thread usually means a kernel is handling a whole row or array where it should handle one ` +
        `value — check that every array is indexed down to a number before you do arithmetic on it.`;
      push({ type: 'error', time: timeString(), text: message });
      await destroyPreviousRun();
      return {
        ok: false,
        error: { message },
        logs,
        kernels: [],
        canvas: null,
        resolvedMode,
        durationMs: 0,
        fellBackToCPU: false,
        refusedAsTooSlow: true,
      };
    }
    await destroyPreviousRun(); // release the probe's contexts before the real run
  }

  const started = performance.now();
  let error = null;
  try {
    if (task && typeof task.inputs === 'function') {
      const inputs = task.inputs(utils) || {};
      for (const [name, value] of Object.entries(inputs)) {
        if (IDENT_RE.test(name)) globals[name] = value;
      }
    }
    const names = Object.keys(globals);
    // eslint-disable-next-line no-new-func
    const fn = new Function(
      ...names,
      `"use strict";\nreturn (async () => {\n${code}\n})();`
    );
    await fn(...names.map(n => globals[n]));
  } catch (e) {
    error = {
      message: toErrorMessage(e),
      stack: e && e.stack ? String(e.stack) : undefined,
    };
    push({ type: 'error', time: timeString(), text: error.message });
  }
  const durationMs = performance.now() - started;

  // Last canvas passed to render(), else the last graphical kernel's canvas.
  let canvas = renderedCanvas;
  if (!canvas) {
    for (let i = kernels.length - 1; i >= 0; i--) {
      const k = kernels[i];
      try {
        if (k.kernel && k.kernel.graphical && k.canvas) {
          canvas = k.canvas;
          break;
        }
      } catch (e) {
        // kernel never built — keep looking
      }
    }
  }

  // What backend did gpu.js ACTUALLY use? It silently swaps in a CPU kernel
  // when a kernel can't compile for WebGL — graphical kernels are pinned to
  // 'unsigned' precision (backend/kernel.js setGraphical), which has no vec4
  // array type, so any graphical kernel taking an image[y][x] array lands on
  // the CPU. Saying so beats letting the console claim a GPU run.
  const usedCpuKernel = kernels.some(isCpuKernel);
  const fellBackToCPU = resolvedMode === 'gpu' && usedCpuKernel;
  if (fellBackToCPU && !probe) {
    push({
      type: 'warn',
      time: timeString(),
      text:
        '▸ gpu.js could not compile this kernel for WebGL and ran it on the CPU backend instead ' +
        '(graphical kernels use unsigned precision, which has no 2D pixel-array type)',
    });
  }

  if (!error) {
    push({
      type: 'ok',
      time: timeString(),
      text: `✓ run complete in ${durationMs.toFixed(1)} ms${fellBackToCPU ? ' (on the CPU backend)' : ''}`,
    });
  }

  return {
    ok: !error,
    error,
    logs,
    kernels,
    canvas,
    resolvedMode,
    durationMs,
    fellBackToCPU,
    probeStats: probe ? probeStats : undefined,
  };
}

// ---- wire form -------------------------------------------------------------

// Strips the live objects (kernels, canvases) out of an InternalRunResult so
// the rest can cross a postMessage boundary, and collects the ImageBitmaps
// that have to be transferred rather than cloned.
export function toWireResult(internal, extra = {}) {
  const transfer = [];
  const logs = internal.logs.map(log => {
    if (!log.canvas && !log.snapshot) return log;
    const { canvas, snapshot, ...rest } = log;
    if (snapshot && snapshot.bitmap) {
      transfer.push(snapshot.bitmap);
      return { ...rest, snapshot };
    }
    // main-thread fallback: the snapshot is already a data URL
    return snapshot ? { ...rest, snapshot } : rest;
  });
  const canvas = internal.canvas;
  return {
    result: {
      ok: internal.ok,
      error: internal.error || null,
      logs,
      canvasInfo: canvas && canvas.width
        ? { width: canvas.width, height: canvas.height }
        : null,
      kernelCount: (internal.kernels || []).length,
      resolvedMode: internal.resolvedMode,
      durationMs: internal.durationMs,
      fellBackToCPU: Boolean(internal.fellBackToCPU),
      refusedAsTooSlow: Boolean(internal.refusedAsTooSlow),
      ...extra,
    },
    transfer,
  };
}

// ---- test running ---------------------------------------------------------

// ctx handed to each test: the run result spread (INCLUDING the live kernels
// and canvas — this only ever happens in the thread that owns them), plus
// kernel (last created), task, utils, assert, assertClose and getPixels().
export function buildTestContext(runResult, task) {
  const kernels = runResult.kernels || [];
  return {
    ...runResult,
    task,
    kernel: kernels.length ? kernels[kernels.length - 1] : null,
    utils,
    assert,
    assertClose,
    // Uint8ClampedArray from the last graphical kernel's getPixels();
    // falls back to a 2D readback of the run's canvas.
    getPixels(flip) {
      for (let i = kernels.length - 1; i >= 0; i--) {
        const k = kernels[i];
        try {
          if (k.kernel && k.kernel.graphical && typeof k.getPixels === 'function') {
            return k.getPixels(flip);
          }
        } catch (e) {
          // fall through to the next candidate
        }
      }
      const pixels = readCanvasPixels(runResult.canvas);
      if (pixels) return pixels;
      throw new Error('no graphical kernel or canvas to read pixels from');
    },
  };
}

// Runs a task's public + private tests against an InternalRunResult.
// → { results: [{ name, private, passed, ms, error? }], passed, total, allPassed }
export async function executeTests(task, runResult) {
  const suites = [
    { tests: task.publicTests || [], isPrivate: false },
    { tests: task.privateTests || [], isPrivate: true },
  ];
  const results = [];
  for (const { tests, isPrivate } of suites) {
    for (const test of tests) {
      const ctx = buildTestContext(runResult, task);
      const t0 = performance.now();
      let passed = true;
      let errorMessage;
      try {
        await test.run(ctx);
      } catch (e) {
        passed = false;
        errorMessage = toErrorMessage(e);
      }
      results.push({
        name: test.name,
        private: isPrivate,
        passed,
        ms: performance.now() - t0,
        error: errorMessage,
      });
    }
  }
  const passed = results.filter(r => r.passed).length;
  return { results, passed, total: results.length, allPassed: passed === results.length };
}

// ---- benchmark ------------------------------------------------------------
//
// Strategy (unchanged from when this lived on the main thread): run the user
// code twice (mode 'cpu', then mode 'gpu'). For each run: one warm-up call,
// then an adaptive timed loop re-invoking every recorded kernel with its
// .lastArgs — stop at ≥5 iterations or ≥250 ms — and report the median
// iteration time. It runs here, in the sandbox, because those extra kernel
// invocations are exactly as capable of hanging as the run itself.

function invokableKernels(runResult) {
  return (runResult.kernels || []).filter(k => Array.isArray(k.lastArgs));
}

// Graphical and pipeline kernels only ENQUEUE GL commands — without a readback
// the timed call returns in ~0 ms no matter how much work the GPU is doing.
// Force the pipeline to drain inside the timed region so the numbers are real.
function syncKernel(k, result) {
  try {
    if (result && typeof result.toArray === 'function') {
      // pipeline kernels return a texture — reading it back forces completion
      result.toArray();
      return;
    }
    const built = k.kernel;
    if (built && built.graphical) {
      const gl = built.context;
      if (gl && typeof gl.readPixels === 'function') {
        // reading one pixel blocks until every queued draw has finished
        gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array(4));
      }
    }
  } catch (e) {
    // sync is best-effort — cpu kernels have nothing to drain
  }
}

function timeKernels(kernels) {
  // warm-up (compilation, first-run allocation)
  for (const k of kernels) syncKernel(k, k(...k.lastArgs));

  const times = [];
  const started = performance.now();
  while (times.length < 5 && performance.now() - started < 250) {
    const t0 = performance.now();
    for (const k of kernels) syncKernel(k, k(...k.lastArgs));
    times.push(performance.now() - t0);
  }
  times.sort((a, b) => a - b);
  return times[Math.floor(times.length / 2)];
}

export async function executeBenchmark(code, task) {
  try {
    const cpuRun = await executeRun(code, { mode: 'cpu', task });
    if (!cpuRun.ok) return { error: cpuRun.error };
    const cpuKernels = invokableKernels(cpuRun);
    if (!cpuKernels.length) {
      return { error: { message: 'nothing to benchmark — the code never invoked a kernel' } };
    }
    const cpuMs = timeKernels(cpuKernels);

    if (!gpuSupported()) return { gpuUnavailable: true, cpuMs };

    // WebGL exists here, so a failure below is the user's code failing in gpu
    // mode (the GL backend rejects some code the cpu backend tolerates) — that
    // is gpuFailed, never gpuUnavailable.
    const gpuRun = await executeRun(code, { mode: 'gpu', task });
    if (!gpuRun.ok) return { gpuFailed: true, cpuMs, error: gpuRun.error };
    const gpuKernels = invokableKernels(gpuRun);
    if (!gpuKernels.length) {
      return {
        gpuFailed: true,
        cpuMs,
        error: { message: 'the code never invoked a kernel in gpu mode' },
      };
    }
    const gpuMs = timeKernels(gpuKernels);

    return {
      cpuMs,
      gpuMs,
      ratio: gpuMs > 0 ? cpuMs / gpuMs : Infinity,
      fasterOn: gpuMs <= cpuMs ? 'gpu' : 'cpu',
      // gpu.js swaps in a CPU kernel for anything it can't compile for WebGL,
      // so "gpu mode" here may have been the CPU backend twice over — the chip
      // has to say that rather than report a meaningless 1.0× ratio.
      gpuRanOnCpu: Boolean(gpuRun.fellBackToCPU),
    };
  } catch (e) {
    // e.g. re-invoking a kernel the user's code already destroyed via
    // gpu.destroy() — surface it in the chip instead of rejecting unhandled
    return { error: { message: toErrorMessage(e) } };
  }
}
