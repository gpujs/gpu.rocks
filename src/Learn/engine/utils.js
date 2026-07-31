// engine/utils.js — helpers shared by every execution path.
//
// Deliberately free of gpu.js and of any hard DOM dependency: this module is
// imported by the main thread AND by sandbox.worker.js, where `document` does
// not exist. Anything that needs a canvas branches on what the environment
// actually offers (document → HTMLCanvasElement, worker → OffscreenCanvas).
//
// The one global it does require is `ImageData` (plainToImageData) — present in
// both a window and a worker, and only ever touched when called, so the content
// modules that import this file still load in plain node (scripts/prerender.mjs).

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

// 8-bit-exact copy of one [r, g, b, a?] pixel. An ImageData can only hold 8-bit
// channels, so a pixel literal like [0.3, 0.5, 0.7, 1] does NOT survive the trip
// unchanged — quantize it here and test expectations computed from the result are
// exactly what the kernel sees on every backend.
export function quantizePixel(pixel) {
  return [q8(pixel[0]), q8(pixel[1]), q8(pixel[2]), q8(pixel[3] === undefined ? 1 : pixel[3])];
}

// Nested-array image → an ImageData every gpu.js backend can read on the GPU.
//
// WHY IMAGEDATA. `graphical: true` pins a kernel to 'unsigned' precision, whose
// kernel-value map has no entry for 'Array2D(4)' (backend/web-gl{,2}/kernel-
// value-maps.js) — so gpu.js quietly substitutes a CPUKernel for any graphical
// kernel handed an image[y][x] = [r, g, b, a] nested array. 'ImageData' IS in
// every one of those maps (both precisions, dynamic and not), and it is
// constructible in a Worker, so it is the one image shape that runs on the GPU
// on WebGL2, WebGL and CPU alike.
//
// WHY THE ROWS ARE REVERSED. Both backends read an ImageData bottom-up: the GL
// backends upload with UNPACK_FLIP_Y_WEBGL = true (web-gl/kernel-value/html-
// image.js), and CPUKernel's _mediaTo2DArray fills row y from scanline
// height-1-y. Writing the plain array's LAST row first therefore makes
// `image[this.thread.y][this.thread.x]` return exactly `plain[y][x]` as an
// [r, g, b, a] vec4 in 0–1 — the course's convention, unchanged, so no kernel
// source, prose or diagram has to change.
//
// The returned object carries two extras for host-side test code:
//   .plain      the same nested array (channels must be 8-bit exact — see
//               quantizePixel — or it will not match what the kernel reads)
//   .at(x, y)   → plain[y][x]
// Both are non-enumerable, and NEITHER may be called `type`: gpu.js's
// getVariableType checks `value.hasOwnProperty('type')` before it checks
// `value instanceof ImageData`, so a `.type` property would mis-type the
// argument.
export function plainToImageData(plain) {
  const height = plain.length;
  const width = plain[0].length;
  const data = new Uint8ClampedArray(width * height * 4);
  let i = 0;
  for (let y = height - 1; y >= 0; y--) {
    const row = plain[y];
    for (let x = 0; x < width; x++) {
      const p = row[x];
      data[i++] = Math.round(clamp01(p[0]) * 255);
      data[i++] = Math.round(clamp01(p[1]) * 255);
      data[i++] = Math.round(clamp01(p[2]) * 255);
      data[i++] = Math.round(clamp01(p[3] === undefined ? 1 : p[3]) * 255);
    }
  }
  const image = new ImageData(data, width, height);
  Object.defineProperties(image, {
    plain: { value: plain, enumerable: false },
    at: { value: (x, y) => plain[y][x], enumerable: false },
  });
  return image;
}

// Deterministic seeded RGBA test image, as an ImageData a graphical kernel can
// run on the GPU: in-kernel `image[y][x]` is [r, g, b, a] with channels 0–1, and
// `image.plain[y][x]` / `image.at(x, y)` is the same pixel host-side. Channels
// are quantized to 8-bit steps, which is what makes that round trip lossless.
// Same size → same image, always.
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
  return plainToImageData(image);
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

// A Promise is truthy, so `assert(result[0] === 42)` on an un-awaited kernel
// does not merely fail — `result[0]` is undefined and the comparison is false,
// which reads as ordinary wrongness. Catch the shape before the condition.
export function assertNotPromise(value, message) {
  if (isPromiseLike(value)) {
    throw new Error(`${message ? `${message} — ` : ''}${AWAIT_HINT}`);
  }
}

export function assert(cond, message) {
  if (!cond) throw new Error(message || 'assertion failed');
}

// The single most likely wrong answer in the whole course, and the one a
// learner can least diagnose: a kernel call without `await`.
//
// Kernels return a Promise on every backend now, so a missing `await` hands
// the tests a Promise instead of a result. Left alone that surfaces as
// `undefined`, `{}`, `0`, or "expected 4071.75, got NaN" — a value mismatch
// that says nothing about the cause, on code whose arithmetic is perfect. The
// course's own rule is that a probe must name the mistake when it can prove
// it, and here it can: nothing else in a task is a thenable.
export function isPromiseLike(v) {
  return Boolean(v) && (typeof v === 'object' || typeof v === 'function') && typeof v.then === 'function';
}

export const AWAIT_HINT =
  'this is a Promise, not a result — a kernel call is missing its `await`. ' +
  'Kernels hand back a promise on every backend, so write `const out = await myKernel(...)` ' +
  '(awaiting a plain value is harmless, so the same line works in every mode).';

function promiseGuard(value, prefix) {
  if (isPromiseLike(value)) throw new Error(`${prefix}${AWAIT_HINT}`);
}

export function assertClose(a, b, eps = 1e-4, message) {
  const prefix = message ? `${message} — ` : '';
  promiseGuard(a, prefix);
  if (typeof a !== 'number' || Number.isNaN(a)) {
    throw new Error(`${prefix}expected a number close to ${b}, got ${a}`);
  }
  if (Math.abs(a - b) > eps) {
    throw new Error(`${prefix}expected ${b} ± ${eps}, got ${a}`);
  }
}

export const utils = {
  seededRandom,
  makeTestImage,
  flatten,
  assert,
  assertClose,
  assertNotPromise,
  isPromiseLike,
};

// ---- rich console payloads -------------------------------------------------
//
// These describe a chart, a frame strip or a control, and they cross
// postMessage out of the worker — so everything here must be plain JSON. No
// DOM, no typed arrays, no functions. The renderer lives in task/ConsolePane.

// A chart is a picture, not a data dump: past a few hundred points a line plot
// cannot show more, and serialising 131,072 floats per log line would cost more
// than the run. Stride down, and keep the true length so the label can say so.
const MAX_PLOT_POINTS = 400;

export function toNumberArray(value) {
  if (ArrayBuffer.isView(value)) return Array.from(value, Number);
  if (Array.isArray(value)) return value;
  return null;
}

// Sampled rather than exhaustive: this runs on every console.log, and walking a
// 131k-element array to decide whether to offer a sparkline would be a tax on
// every line the learner prints.
export function looksNumeric(value) {
  const arr = toNumberArray(value);
  if (!arr || arr.length < 4) return false;
  const step = Math.max(1, Math.floor(arr.length / 32));
  for (let i = 0; i < arr.length; i += step) {
    const n = arr[i];
    if (typeof n !== 'number' || !Number.isFinite(n)) return false;
  }
  return true;
}

export function downsample(values, max = MAX_PLOT_POINTS) {
  const total = values.length;
  if (total <= max) return { values: Array.from(values, Number), total };
  const stride = total / max;
  const out = [];
  for (let i = 0; i < max; i++) out.push(Number(values[Math.floor(i * stride)]));
  return { values: out, total };
}

/**
 * Accepts the shapes a learner will actually reach for:
 *   plot([1, 2, 3])                       one unnamed series
 *   plot({ jacobi: [...], redBlack: [...] })   named series, one per key
 *   plot([[...], [...]])                  several unnamed series
 * Returns null when there is nothing plottable, so the caller can fall back to
 * an ordinary log line rather than render an empty chart.
 */
export function normalisePlot(data, options = {}) {
  const series = [];
  const push = (name, values) => {
    const arr = toNumberArray(values);
    if (!arr || !arr.length) return;
    const { values: v, total } = downsample(arr);
    series.push({ name: name || '', values: v, total });
  };
  const asArray = toNumberArray(data);
  if (asArray && asArray.length && toNumberArray(asArray[0])) {
    asArray.forEach((row, i) => push(options.names ? options.names[i] : '', row));
  } else if (asArray) {
    push(options.name || '', asArray);
  } else if (data && typeof data === 'object') {
    for (const [name, values] of Object.entries(data)) push(name, values);
  }
  if (!series.length) return null;
  const x = toNumberArray(options.x);
  return {
    series,
    title: options.title ? String(options.title) : '',
    xLabel: options.xLabel ? String(options.xLabel) : '',
    yLabel: options.yLabel ? String(options.yLabel) : '',
    // log scale is what the course's own convergence figures use — a residual
    // falling 36x is invisible on a linear axis
    log: Boolean(options.log),
    x: x ? downsample(x).values : null,
  };
}

// A control declaration. `value` is what the program actually ran with, so the
// renderer shows the slider already in the right position on the first run.
export function normaliseControl(name, options = {}) {
  const min = Number.isFinite(Number(options.min)) ? Number(options.min) : 0;
  const max = Number.isFinite(Number(options.max)) ? Number(options.max) : 1;
  const rawStep = Number(options.step);
  const step = Number.isFinite(rawStep) && rawStep > 0 ? rawStep : (max - min) / 100 || 0.01;
  return {
    kind: 'slider',
    name: String(name),
    label: options.label ? String(options.label) : String(name),
    min,
    max,
    step,
  };
}

// ---- console / log formatting ---------------------------------------------

export function timeString() {
  const d = new Date();
  const pad = (n, w) => String(n).padStart(w, '0');
  return `${pad(d.getHours(), 2)}:${pad(d.getMinutes(), 2)}:${pad(d.getSeconds(), 2)}.${pad(d.getMilliseconds(), 3)}`;
}

export function formatValue(value, depth = 0) {
  if (value === null) return 'null';
  if (value === undefined) return 'undefined';
  // JSON.stringify(promise) is '{}', which tells a learner nothing. Naming it
  // turns the single most confusing console line in the course into the
  // answer: `console.log(myKernel())` without `await` prints this.
  if (isPromiseLike(value)) return 'Promise { … } ← missing `await`?';
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
  // the course's images: naming one beats JSON-stringifying a megapixel of
  // channels only to truncate it at 200 characters
  if (typeof ImageData !== 'undefined' && value instanceof ImageData) {
    return `ImageData(${value.width}×${value.height})`;
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
  // in the worker sandbox the same kernel.canvas is an OffscreenCanvas — name it
  // rather than letting JSON.stringify render it as '{}'
  if (typeof OffscreenCanvas !== 'undefined' && value instanceof OffscreenCanvas) {
    return `OffscreenCanvas(${value.width}×${value.height})`;
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

// ---- canvas snapshots ------------------------------------------------------

// Snapshot a (possibly WebGL) canvas so the console can show the frame that
// existed at this moment. Called at render()-log time so that two render()
// calls of the same canvas in one run each capture their own frame.
//
// Two shapes, one per environment, because only one of them can cross a worker
// boundary and only one of them can go straight into an <img>:
//   main thread → { url, w, h }      data URL, ready for ConsolePane
//   worker      → { bitmap, w, h }   transferable ImageBitmap; runner.js turns
//                                    it into { url, w, h } on arrival
export function snapshotCanvas(canvas) {
  try {
    if (!canvas) return null;
    const w = canvas.width;
    const h = canvas.height;
    if (!w || !h) return null;
    if (typeof document !== 'undefined') {
      const tmp = document.createElement('canvas');
      tmp.width = w;
      tmp.height = h;
      tmp.getContext('2d').drawImage(canvas, 0, 0);
      return { url: tmp.toDataURL(), w, h };
    }
    if (typeof OffscreenCanvas !== 'undefined') {
      const tmp = new OffscreenCanvas(w, h);
      tmp.getContext('2d').drawImage(canvas, 0, 0);
      // transferToImageBitmap is synchronous, so the frame is captured now and
      // not whenever a promise happens to settle
      return { bitmap: tmp.transferToImageBitmap(), w, h };
    }
    return null;
  } catch (e) {
    return null;
  }
}

// Reads a canvas back as RGBA bytes (the getPixels() fallback for tests).
export function readCanvasPixels(canvas) {
  if (!canvas || !canvas.width) return null;
  let tmp;
  if (typeof document !== 'undefined') {
    tmp = document.createElement('canvas');
    tmp.width = canvas.width;
    tmp.height = canvas.height;
  } else if (typeof OffscreenCanvas !== 'undefined') {
    tmp = new OffscreenCanvas(canvas.width, canvas.height);
  } else {
    return null;
  }
  const g = tmp.getContext('2d');
  g.drawImage(canvas, 0, 0);
  return g.getImageData(0, 0, tmp.width, tmp.height).data;
}
