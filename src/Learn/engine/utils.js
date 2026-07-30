// engine/utils.js — helpers shared by every execution path.
//
// Deliberately free of gpu.js and of any hard DOM dependency: this module is
// imported by the main thread AND by sandbox.worker.js, where `document` does
// not exist. Anything that needs a canvas branches on what the environment
// actually offers (document → HTMLCanvasElement, worker → OffscreenCanvas).

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

// ---- console / log formatting ---------------------------------------------

export function timeString() {
  const d = new Date();
  const pad = (n, w) => String(n).padStart(w, '0');
  return `${pad(d.getHours(), 2)}:${pad(d.getMinutes(), 2)}:${pad(d.getSeconds(), 2)}.${pad(d.getMilliseconds(), 3)}`;
}

export function formatValue(value, depth = 0) {
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
