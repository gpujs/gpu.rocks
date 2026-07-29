// engine/runner.js — sandboxed execution of learner code against gpu.js.
//
// runUserCode(code, { mode, task }) → Promise<RunResult>, never rejects:
//   {
//     ok: boolean,
//     error?: { message, stack? },        // on throw / compile failure
//     logs: [{ type: 'system'|'log'|'warn'|'error'|'canvas'|'ok',
//              time,                       // 'HH:MM:SS.mmm' wall-clock string
//              text?, canvas? }],
//     kernels: [kernel],                  // every kernel created, in order;
//                                         // each records .lastArgs on invocation
//     canvas,                             // last canvas passed to render(), else
//                                         // the last graphical kernel's canvas
//     resolvedMode: 'gpu' | 'cpu',
//     durationMs: number,
//   }
//
// Kernels (well, their owning GPU instances) created by one run are destroyed
// at the start of the next run so WebGL contexts don't leak.

import { GPU, utils as gpuUtils } from 'gpu.js';

// ---- deterministic utils (shared with task inputs and tests) --------------

// mulberry32 — small, fast, fully deterministic PRNG. Returns () => [0, 1).
export function seededRandom(seed) {
  let s = seed >>> 0;
  return function next() {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function clamp01(v) {
  return v < 0 ? 0 : v > 1 ? 1 : v;
}

// quantize to 8-bit steps so float → canvas → Uint8 readbacks compare cleanly
function q8(v) {
  return Math.round(clamp01(v) * 255) / 255;
}

// Deterministic seeded RGBA test image as nested arrays:
// image[y][x] = [r, g, b, a], all channels 0–1. Same size → same image, always.
export function makeTestImage(size) {
  const rand = seededRandom(0x6770752e ^ (size * 2654435761));
  const image = new Array(size);
  for (let y = 0; y < size; y++) {
    const row = new Array(size);
    const ny = y / size;
    for (let x = 0; x < size; x++) {
      const nx = x / size;
      row[x] = [
        q8(0.2 + 0.55 * nx + 0.25 * rand()),
        q8(0.2 + 0.55 * ny + 0.25 * rand()),
        q8(0.15 + 0.6 * Math.abs(Math.sin(3.1 * (nx + ny))) + 0.25 * rand()),
        1,
      ];
    }
    image[y] = row;
  }
  return image;
}

// Flattens arbitrarily nested arrays / typed arrays into one plain Array.
export function flatten(arr) {
  const out = [];
  const stack = [arr];
  while (stack.length) {
    const value = stack.pop();
    if (Array.isArray(value) || ArrayBuffer.isView(value)) {
      for (let i = value.length - 1; i >= 0; i--) stack.push(value[i]);
    } else {
      out.push(value);
    }
  }
  return out;
}

export function assert(cond, message) {
  if (!cond) throw new Error(message || 'assertion failed');
}

export function assertClose(a, b, eps = 1e-4, message) {
  const prefix = message ? `${message} — ` : '';
  if (typeof a !== 'number' || Number.isNaN(a)) {
    throw new Error(`${prefix}expected a number close to ${b}, got ${a}`);
  }
  if (Math.abs(a - b) > eps) {
    throw new Error(`${prefix}expected ${b} ± ${eps}, got ${a}`);
  }
}

export const utils = { seededRandom, makeTestImage, flatten, assert, assertClose };

// ---- run bookkeeping ------------------------------------------------------

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

function timeString() {
  const d = new Date();
  const pad = (n, w) => String(n).padStart(w, '0');
  return `${pad(d.getHours(), 2)}:${pad(d.getMinutes(), 2)}:${pad(d.getSeconds(), 2)}.${pad(d.getMilliseconds(), 3)}`;
}

function formatValue(value, depth = 0) {
  if (value === null) return 'null';
  if (value === undefined) return 'undefined';
  const type = typeof value;
  if (type === 'string') return depth === 0 ? value : JSON.stringify(value);
  if (type === 'number' || type === 'boolean' || type === 'bigint') return String(value);
  if (type === 'function') return `ƒ ${value.name || '(anonymous)'}`;
  if (value instanceof Error) {
    // message/stack are non-enumerable, so JSON.stringify(new Error()) === '{}'
    return `${value.name || 'Error'}: ${value.message}`;
  }
  if (ArrayBuffer.isView(value)) {
    const head = Array.from(value.slice(0, 8), v => formatValue(v, depth + 1));
    const more = value.length > 8 ? ', …' : '';
    return `${value.constructor.name}(${value.length}) [${head.join(', ')}${more}]`;
  }
  if (Array.isArray(value)) {
    if (depth >= 2) return `Array(${value.length})`;
    const head = value.slice(0, 8).map(v => formatValue(v, depth + 1));
    const more = value.length > 8 ? ', …' : '';
    return `[${head.join(', ')}${more}]`;
  }
  if (typeof HTMLCanvasElement !== 'undefined' && value instanceof HTMLCanvasElement) {
    return 'HTMLCanvasElement';
  }
  try {
    const json = JSON.stringify(value);
    return json && json.length > 200 ? `${json.slice(0, 200)}…` : json || String(value);
  } catch (e) {
    try {
      return String(value);
    } catch (e2) {
      // circular AND unstringifiable — a console.log must never abort the run
      return '[unprintable object]';
    }
  }
}

// Coerces a caught value into a plain string message. User code can throw
// anything — including objects whose .message is itself an object — and the
// result is rendered directly as a React child, so it must be a string.
export function toErrorMessage(e) {
  try {
    const raw = e && e.message;
    return raw ? String(raw) : String(e);
  } catch (e2) {
    return 'unprintable error';
  }
}

function makeConsoleProxy(logs) {
  const real = typeof console !== 'undefined' ? console : null;
  const capture = type => (...args) => {
    if (real && real[type]) real[type](...args);
    logs.push({
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
function patchKernel(kernel, logs) {
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
            logs.push({
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

// Snapshot a (possibly WebGL) canvas into a data URL. Called at render()-log
// time so that two render() calls of the same canvas in one run each capture
// their own frame, and again from the UI as a fallback for the run's canvas.
export function snapshotCanvas(canvas) {
  try {
    if (!canvas || typeof document === 'undefined') return null;
    const w = canvas.width;
    const h = canvas.height;
    if (!w || !h) return null;
    const tmp = document.createElement('canvas');
    tmp.width = w;
    tmp.height = h;
    tmp.getContext('2d').drawImage(canvas, 0, 0);
    return { url: tmp.toDataURL(), w, h };
  } catch (e) {
    return null;
  }
}

export function isGPUSupported() {
  try {
    return Boolean(GPU.isGPUSupported);
  } catch (e) {
    return false;
  }
}

// ---- runUserCode ----------------------------------------------------------

export async function runUserCode(code, { mode = 'auto', task } = {}) {
  await destroyPreviousRun();

  const logs = [];
  const kernels = [];
  const instances = [];
  let renderedCanvas = null;

  const gpuOk = isGPUSupported();
  let resolvedMode;
  if (mode === 'cpu') {
    resolvedMode = 'cpu';
    logs.push({ type: 'system', time: timeString(), text: '▸ mode "cpu" → selected cpu' });
  } else if (mode === 'gpu') {
    if (gpuOk) {
      resolvedMode = 'gpu';
      logs.push({ type: 'system', time: timeString(), text: '▸ mode "gpu" → selected gpu (WebGL)' });
    } else {
      resolvedMode = 'cpu';
      logs.push({
        type: 'system',
        time: timeString(),
        text: '▸ mode "gpu" requested but WebGL is unavailable here — falling back to cpu',
      });
    }
  } else {
    resolvedMode = gpuOk ? 'gpu' : 'cpu';
    logs.push({
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
      const kernel = patchKernel(super.createKernel(...args), logs);
      kernels.push(kernel);
      return kernel;
    }

    createKernelMap(...args) {
      const kernel = patchKernel(super.createKernelMap(...args), logs);
      kernels.push(kernel);
      return kernel;
    }
  }

  const render = canvas => {
    renderedCanvas = canvas || renderedCanvas;
    logs.push({
      type: 'canvas',
      time: timeString(),
      text: `render: ${canvas && canvas.constructor ? canvas.constructor.name : 'canvas'}`,
      canvas: canvas || null,
      // capture pixels NOW — a later render() of this same canvas must not
      // retroactively change what this entry shows
      snapshot: snapshotCanvas(canvas),
    });
  };

  const consoleProxy = makeConsoleProxy(logs);

  // Injected globals: GPU, console, render, utils, mode, plus task inputs.
  const globals = {
    GPU: RecordingGPU,
    console: consoleProxy,
    render,
    utils,
    mode: resolvedMode,
  };

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
    logs.push({ type: 'error', time: timeString(), text: error.message });
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

  if (!error) {
    logs.push({
      type: 'ok',
      time: timeString(),
      text: `✓ run complete in ${durationMs.toFixed(1)} ms`,
    });
  }

  return { ok: !error, error, logs, kernels, canvas, resolvedMode, durationMs };
}

// ---- test running ---------------------------------------------------------

// ctx handed to each test: the RunResult spread, plus kernel (last created),
// task, utils, assert, assertClose and getPixels().
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
      const source = runResult.canvas;
      if (source && source.width) {
        const tmp = document.createElement('canvas');
        tmp.width = source.width;
        tmp.height = source.height;
        const g = tmp.getContext('2d');
        g.drawImage(source, 0, 0);
        return g.getImageData(0, 0, tmp.width, tmp.height).data;
      }
      throw new Error('no graphical kernel or canvas to read pixels from');
    },
  };
}

// Runs a task's public + private tests against a RunResult.
// → { results: [{ name, private, passed, ms, error? }], passed, total, allPassed }
export async function runTests(task, runResult) {
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
