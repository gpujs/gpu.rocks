// engine/runner.js — the public entry point for running learner code, and the
// main-thread supervisor of the sandbox that actually runs it.
//
// WHY A WORKER: a gpu.js kernel call is ONE synchronous task with no yield
// points. A kernel doing pathological per-thread work (multiplying a whole
// image row, so V8 stringifies it per thread) froze the page for 197 s, and no
// timer, promise or AbortController on the main thread can interrupt that. The
// only real recovery is to run the code somewhere killable, so all learner code
// — plus the task's tests and the benchmark's kernel re-invocations, which
// tripled the original freeze — lives in sandbox.worker.js, and this file
// enforces a wall-clock budget and calls terminate() when it is blown.
//
// PUBLIC CONTRACT
//
// runUserCode(code, { mode, task }) → Promise<RunResult>, never rejects:
//   {
//     ok: boolean,
//     error?: { message, stack? },        // on throw / compile failure
//     logs: [{ type: 'system'|'log'|'warn'|'error'|'canvas'|'ok',
//              time,                      // 'HH:MM:SS.mmm' wall-clock string
//              text?, snapshot? }],       // snapshot: { url, w, h }
//     canvasInfo: { width, height } | null,
//     kernelCount: number,
//     resolvedMode: 'gpu' | 'cpu',
//     durationMs: number,
//     fellBackToCPU: boolean,             // gpu.js silently used a CPUKernel
//     refusedAsTooSlow?: true,            // pre-flight guard refused the run
//     stoppedByWatchdog?: true,           // budget blown, worker terminated
//     runToken, sandboxGeneration,        // opaque; identify the retained run
//   }
//
// runTests(task, runResult) → Promise<
//   { results: [{ name, private, passed, ms, error? }], passed, total, allPassed }>
//   Runs in the sandbox against the run `runToken` identifies. When that run's
//   worker is gone (the watchdog killed it) every test is reported as failed
//   with the reason, so the Tests panel explains itself instead of emptying.
//
// warmUpSandbox()        — spawn the sandbox early (TaskPage calls it on mount)
// sandboxGpuSupported()  — does the SANDBOX have WebGL (async: it must be asked)
// sandboxInfo()          — { path, gpuSupported, spawnMs, reason }
//
// CONTRACT CHANGE vs the pre-worker version: the result no longer carries
// `kernels` (live kernel functions) or `canvas` (a live canvas). They cannot
// cross postMessage, and they must not: the tests are the only thing that ever
// needed them, and the tests now run in the worker where those objects live, so
// a test still calls ctx.kernel(args), ctx.getPixels() and ctx.kernels[i]
// exactly as before. What the main thread gets instead is `kernelCount` and
// `canvasInfo` (for reporting) and, per canvas log line, a `snapshot` data URL
// produced from an ImageBitmap transferred out of the worker. The only caller
// that used live kernels was benchmark.js, which now also runs in the sandbox.
//
// FALLBACK: without Worker or OffscreenCanvas the same sandbox module runs on
// the main thread, behaving exactly as it did before (probe guard included, but
// unrecoverable on a `while (true)` — nothing can be done there). Which path is
// in use is disclosed as the first `system` log line of every run. On that path
// the result is a SUPERSET of the contract above — the live `kernels` and
// `canvas` are in-thread, so they stay on it — but nothing may depend on that.

import { modules } from '../content/index';
import {
  assert,
  assertClose,
  flatten,
  makeTestImage,
  seededRandom,
  snapshotCanvas,
  timeString,
  toErrorMessage,
  utils,
} from './utils';

export {
  assert,
  assertClose,
  flatten,
  makeTestImage,
  seededRandom,
  snapshotCanvas,
  toErrorMessage,
  utils,
};

// ---- watchdog budgets ------------------------------------------------------
//
// Wall-clock, enforced here on the main thread; blowing one terminates the
// worker. Measured headroom: the whole 75-task course verifies in ~1.2 s (cpu)
// / ~5.4 s (gpu) for 150 runs plus 150 test suites, so the slowest single task
// is orders of magnitude inside these.
//
// A run is the pre-flight probe plus the real thing.
const RUN_WATCHDOG_MS = 10000;
// Tests re-invoke the run's kernels, so they get their own budget.
const TESTS_BUDGET_MS = 15000;
// The benchmark is two full runs plus two adaptive timing loops.
const BENCHMARK_BUDGET_MS = 30000;
// Worker boot: module graph (gpu.js + the content registry) parse and evaluate.
const HELLO_BUDGET_MS = 15000;

// ---- sandbox supervisor ----------------------------------------------------

let sandboxPath = null; // 'worker' | 'main', decided once
let initPromise = null;
let unavailableReason = null;
let cachedGpuSupported = null;
let spawnMs = null;

let worker = null;
let generation = 0; // bumped on every spawn; identifies a worker's run state
let nextId = 1;
const pending = new Map(); // id → { settle, timer, logs }

// Escape hatch for diagnostics and for A/B-ing the two paths in the browser:
// set globalThis.__learnForceSandbox = 'main' | 'worker' before the first run.
function forcedPath() {
  try {
    const forced = globalThis.__learnForceSandbox;
    return forced === 'main' || forced === 'worker' ? forced : null;
  } catch (e) {
    return null;
  }
}

function workerCapable() {
  return (
    typeof Worker !== 'undefined' &&
    typeof OffscreenCanvas !== 'undefined' &&
    typeof URL !== 'undefined'
  );
}

function settleAll(reply) {
  for (const [, entry] of pending) {
    clearTimeout(entry.timer);
    entry.settle({ ...reply, logs: entry.logs });
  }
  pending.clear();
}

function killWorker() {
  const dying = worker;
  worker = null;
  generation++; // any run state that worker held is now unreachable
  if (dying) {
    dying.onmessage = null;
    dying.onerror = null;
    dying.onmessageerror = null;
    try {
      dying.terminate();
    } catch (e) {
      // already gone
    }
  }
  // nothing else will ever answer those requests
  settleAll({ failed: true, error: { message: 'the sandbox worker was stopped' } });
}

function onWorkerMessage(event) {
  const msg = event.data || {};
  const entry = pending.get(msg.id);
  if (!entry) return; // a reply to a request we already gave up on
  if (msg.kind === 'log') {
    entry.logs.push(msg.log);
    return;
  }
  clearTimeout(entry.timer);
  pending.delete(msg.id);
  if (msg.kind === 'failed') {
    entry.settle({ failed: true, error: msg.error || { message: 'sandbox failed' }, logs: entry.logs });
    return;
  }
  entry.settle({ result: msg.result, logs: entry.logs });
}

function onWorkerBroken(reason) {
  settleAll({ failed: true, error: { message: reason } });
  killWorker();
}

function spawnWorker() {
  worker = new Worker(new URL('./sandbox.worker.js', import.meta.url), {
    type: 'module',
    name: 'learn-sandbox',
  });
  generation++;
  worker.onmessage = onWorkerMessage;
  worker.onerror = event => {
    onWorkerBroken(
      (event && event.message) ? `sandbox worker error: ${event.message}` : 'sandbox worker failed to load'
    );
  };
  worker.onmessageerror = () => onWorkerBroken('sandbox worker could not decode a message');
}

// Sends one request. Resolves (never rejects) to exactly one of:
//   { result, logs } | { timedOut: true, logs } | { failed: true, error, logs }
function call(message, budgetMs) {
  try {
    if (!worker) spawnWorker();
  } catch (e) {
    // e.g. a CSP that forbids worker scripts — the caller reports it, and
    // ensureSandbox() will have moved to the main-thread path already
    return Promise.resolve({ failed: true, error: { message: toErrorMessage(e) }, logs: [] });
  }
  const id = nextId++;
  const target = worker;
  return new Promise(resolve => {
    let done = false;
    const settle = reply => {
      if (done) return;
      done = true;
      resolve(reply);
    };
    const timer = setTimeout(() => {
      pending.delete(id);
      const logs = entry.logs;
      // The worker is wedged in a synchronous task; terminate() is the only
      // way out. Spawn its replacement now so the next Run does not pay for it.
      killWorker();
      try {
        spawnWorker();
      } catch (e) {
        worker = null; // the next request tries again and reports the failure
      }
      settle({ timedOut: true, logs });
    }, budgetMs);
    const entry = { settle, timer, logs: [] };
    pending.set(id, entry);
    try {
      target.postMessage({ ...message, id });
    } catch (e) {
      clearTimeout(timer);
      pending.delete(id);
      settle({ failed: true, error: { message: toErrorMessage(e) }, logs: [] });
    }
  });
}

// Decides the execution path once, and warms the worker up (spawning it costs
// a module-graph evaluation of gpu.js plus the content registry).
export function ensureSandbox() {
  if (initPromise) return initPromise;
  // Resolves, never rejects: everything downstream promises not to throw.
  initPromise = (async () => {
    const forced = forcedPath();
    if (forced === 'main' || (!forced && !workerCapable())) {
      sandboxPath = 'main';
      unavailableReason = forced
        ? 'forced by __learnForceSandbox'
        : 'this browser has no Worker or no OffscreenCanvas';
      await loadMainThreadSandbox();
      return sandboxPath;
    }
    try {
      const t0 = performance.now();
      spawnWorker();
      const reply = await call({ kind: 'hello' }, HELLO_BUDGET_MS);
      if (!reply.result) {
        throw new Error(
          reply.failed && reply.error ? reply.error.message : 'sandbox worker did not answer'
        );
      }
      spawnMs = performance.now() - t0;
      cachedGpuSupported = Boolean(reply.result.gpuSupported);
      sandboxPath = 'worker';
    } catch (e) {
      killWorker();
      sandboxPath = 'main';
      unavailableReason = toErrorMessage(e);
      await loadMainThreadSandbox();
    }
    return sandboxPath;
  })();
  return initPromise;
}

// The fallback's copy of the execution core, imported lazily so the normal
// (worker) path does not pull gpu.js into the page's own bundle.
async function loadMainThreadSandbox() {
  try {
    const sandbox = await import('./sandbox');
    cachedGpuSupported = sandbox.gpuSupported();
    return sandbox;
  } catch (e) {
    cachedGpuSupported = false;
    return null;
  }
}

// Optional: start the sandbox before the learner presses Run, so the spawn
// happens while they are reading the brief instead of inside their first run.
export function warmUpSandbox() {
  ensureSandbox().catch(() => {});
}

// Which path is in use, whether the sandbox has WebGL, and what the spawn cost.
export async function sandboxInfo() {
  const path = await ensureSandbox();
  return {
    path,
    gpuSupported: Boolean(cachedGpuSupported),
    spawnMs,
    reason: unavailableReason,
  };
}

// Does the sandbox (the thread that actually builds kernels) have WebGL?
export async function sandboxGpuSupported() {
  await ensureSandbox();
  return Boolean(cachedGpuSupported);
}

// ---- task identity ---------------------------------------------------------

// Functions cannot cross postMessage, so the worker looks tasks up in its own
// copy of the content registry. The main thread only sends the coordinates:
// { uuid, taskSlug } — the module's permanent identity and the task's own
// slug, never a position, so the ref survives tasks being reordered.
const taskRefs = new WeakMap();
let taskRefsBuilt = false;

function buildTaskRefs() {
  taskRefsBuilt = true;
  for (const module of modules) {
    (module.tasks || []).forEach(task => {
      if (task && typeof task === 'object') {
        taskRefs.set(task, { uuid: module.uuid, taskSlug: task.slug });
      }
    });
  }
}

export function taskRefFor(task) {
  if (!task || typeof task !== 'object') return null;
  if (!taskRefsBuilt) buildTaskRefs();
  return taskRefs.get(task) || null;
}

// ---- main-thread fallback --------------------------------------------------

let mainRun = null; // { token, internal } — the fallback's retained run state
let mainTokenSeq = 0;

async function runOnMainThread(code, mode, task) {
  const sandbox = await loadMainThreadSandbox();
  if (!sandbox) {
    return failedResult(mode, { message: 'the execution engine could not be loaded' }, []);
  }
  const internal = await sandbox.executeRun(code, { mode, task });
  const token = `main-${++mainTokenSeq}`;
  mainRun = { token, internal };
  const canvas = internal.canvas;
  // The fallback keeps the live `kernels`/`canvas` on the result: they are
  // in-thread here, and anything that used to read them still can.
  return {
    ...internal,
    canvasInfo: canvas && canvas.width ? { width: canvas.width, height: canvas.height } : null,
    kernelCount: (internal.kernels || []).length,
    runToken: token,
    sandboxGeneration: generation,
    sandboxPath: 'main',
  };
}

// ---- run results -----------------------------------------------------------

// ImageBitmaps are transferable, data URLs are what <img> wants: convert on
// arrival so ConsolePane keeps rendering `log.snapshot.url` unchanged.
function bitmapToDataUrl(snapshot) {
  if (!snapshot || !snapshot.bitmap) return snapshot || null;
  const { bitmap, w, h } = snapshot;
  try {
    const tmp = document.createElement('canvas');
    tmp.width = w;
    tmp.height = h;
    tmp.getContext('2d').drawImage(bitmap, 0, 0);
    return { url: tmp.toDataURL(), w, h };
  } catch (e) {
    return null;
  } finally {
    try {
      bitmap.close();
    } catch (e) {
      // nothing to release
    }
  }
}

function hydrate(result, gen) {
  return {
    ...result,
    logs: (result.logs || []).map(log =>
      log.snapshot && log.snapshot.bitmap ? { ...log, snapshot: bitmapToDataUrl(log.snapshot) } : log
    ),
    sandboxGeneration: gen,
    sandboxPath: 'worker',
  };
}

// The mode the sandbox WOULD have resolved — used only when the worker died
// before it could tell us what it picked.
function assumedMode(mode) {
  if (mode === 'cpu') return 'cpu';
  return cachedGpuSupported ? 'gpu' : 'cpu';
}

function stoppedResult(mode, logs, budgetMs, verb) {
  const message =
    `stopped after ${Math.round(budgetMs / 1000)}s — your code was still running and the page ` +
    'would have frozen';
  return {
    ok: false,
    error: { message },
    logs: [...logs, { type: 'error', time: timeString(), text: message }],
    canvasInfo: null,
    kernelCount: 0,
    resolvedMode: assumedMode(mode),
    durationMs: budgetMs,
    fellBackToCPU: false,
    stoppedByWatchdog: true,
    // the worker that held this run's state is gone: no generation can match,
    // so runTests reports the stop instead of asking the fresh worker
    sandboxGeneration: -1,
    sandboxPath: 'worker',
    stoppedDuring: verb,
  };
}

function failedResult(mode, error, logs) {
  const message = (error && error.message) || 'the sandbox failed';
  return {
    ok: false,
    error: { message },
    logs: [...logs, { type: 'error', time: timeString(), text: message }],
    canvasInfo: null,
    kernelCount: 0,
    resolvedMode: assumedMode(mode),
    durationMs: 0,
    fellBackToCPU: false,
    sandboxFailed: true,
    sandboxGeneration: -1,
    sandboxPath: 'worker',
  };
}

// ---- runUserCode -----------------------------------------------------------

export async function runUserCode(code, { mode = 'auto', task } = {}) {
  try {
    const path = await ensureSandbox();
    const taskRef = taskRefFor(task);
    // A task that is not in the registry (hand-built in a console, say) cannot
    // be looked up by the worker — its inputs and tests are functions. Here.
    if (path !== 'worker' || (task && !taskRef)) return await runOnMainThread(code, mode, task);

    const gen = generation;
    const reply = await call({ kind: 'run', code, mode, taskRef }, RUN_WATCHDOG_MS);
    if (reply.timedOut) return stoppedResult(mode, reply.logs, RUN_WATCHDOG_MS, 'run');
    if (reply.failed || !reply.result) return failedResult(mode, reply.error, reply.logs);
    return hydrate(reply.result, gen);
  } catch (e) {
    // the contract is that this never rejects — the UI has no other channel
    return failedResult(mode, { message: toErrorMessage(e) }, []);
  }
}

// ---- runTests --------------------------------------------------------------

// Every test failing with one explanation. Used when the run's state is gone
// (the worker was terminated), so the Tests panel still lists the task's tests
// instead of showing an empty report.
function syntheticReport(task, message) {
  const results = [];
  (task.publicTests || []).forEach(t => {
    results.push({ name: t.name, private: false, passed: false, ms: 0, error: message });
  });
  (task.privateTests || []).forEach(t => {
    results.push({ name: t.name, private: true, passed: false, ms: 0, error: message });
  });
  return { results, passed: 0, total: results.length, allPassed: false };
}

export async function runTests(task, runResult) {
  try {
    return await runTestsInner(task, runResult);
  } catch (e) {
    return syntheticReport(task, toErrorMessage(e));
  }
}

async function runTestsInner(task, runResult) {
  const path = await ensureSandbox();

  if (path === 'worker') {
    const taskRef = taskRefFor(task);
    const token = runResult && runResult.runToken;
    const sameWorker = runResult && runResult.sandboxGeneration === generation;
    if (taskRef && token && sameWorker) {
      const reply = await call({ kind: 'tests', runToken: token, taskRef }, TESTS_BUDGET_MS);
      if (reply.timedOut) {
        return syntheticReport(
          task,
          `stopped after ${Math.round(TESTS_BUDGET_MS / 1000)}s — the tests were still running ` +
            'and the page would have frozen'
        );
      }
      if (reply.failed || !reply.result) {
        return syntheticReport(task, (reply.error && reply.error.message) || 'the sandbox failed');
      }
      if (reply.result.staleToken || reply.result.unknownTask) {
        return syntheticReport(task, 'the run this report belongs to is no longer available — run again');
      }
      return reply.result;
    }
    // No state to test against: say why, in the tests themselves.
    const why =
      runResult && runResult.error && runResult.error.message
        ? runResult.error.message
        : 'the run did not complete, so there is nothing to test — run again';
    return syntheticReport(task, why);
  }

  // main-thread fallback
  const sandbox = await loadMainThreadSandbox();
  if (!sandbox) return syntheticReport(task, 'the execution engine could not be loaded');
  const internal =
    mainRun && runResult && runResult.runToken === mainRun.token ? mainRun.internal : runResult;
  return sandbox.executeTests(task, internal || { logs: [], kernels: [] });
}

// ---- benchmark bridge ------------------------------------------------------

// benchmark.js drives this; it lives here so there is exactly one supervisor.
export async function runBenchmarkInSandbox(code, task) {
  try {
    const path = await ensureSandbox();
    const taskRef = taskRefFor(task);
    if (path !== 'worker' || (task && !taskRef)) {
      const sandbox = await loadMainThreadSandbox();
      if (!sandbox) return { error: { message: 'the execution engine could not be loaded' } };
      return await sandbox.executeBenchmark(code, task);
    }
    const reply = await call({ kind: 'benchmark', code, taskRef }, BENCHMARK_BUDGET_MS);
    if (reply.timedOut) {
      return {
        error: {
          message:
            `stopped after ${Math.round(BENCHMARK_BUDGET_MS / 1000)}s — your code was still ` +
            'running and the page would have frozen',
        },
      };
    }
    if (reply.failed || !reply.result) {
      return { error: { message: (reply.error && reply.error.message) || 'the sandbox failed' } };
    }
    return reply.result;
  } catch (e) {
    return { error: { message: toErrorMessage(e) } };
  }
}
