// engine/sandbox.worker.js — the terminable sandbox.
//
// Everything a learner's code can do happens here: building the code with
// `new Function`, the GPU instances and kernels, console capture, render() and
// its canvas snapshots, the pre-flight probe, the task's TESTS and the
// benchmark's repeated kernel invocations. All of it is on this thread, so a
// kernel that would have frozen the page for three minutes is now just a
// worker the supervisor can terminate().
//
// Functions cannot cross postMessage, so the worker cannot be handed a task's
// inputs()/publicTests/privateTests — it imports the content registry itself
// and looks the task up by { uuid, taskSlug }: the module's permanent identity
// and the task's own slug, never a position. The main thread only ever sends
// identifiers, code and a mode.
//
// Protocol — request/response keyed by `id`:
//   main → worker
//     { id, kind: 'hello' }
//     { id, kind: 'run',       code, mode, taskRef, cardCapture? }
//     { id, kind: 'tests',     runToken, taskRef }
//     { id, kind: 'benchmark', code, taskRef }
//   worker → main
//     { id, kind: 'log',    log }        text-only, streamed as it happens, so
//                                        a run killed by the watchdog can still
//                                        show what it managed to do
//     { id, kind: 'result', result }     ImageBitmaps in `transfer`
//     { id, kind: 'failed', error }      the worker itself broke

import { getTaskBySlug } from '../content/index';
import {
  executeBenchmark,
  executeRun,
  executeTests,
  gpuSupported,
  toWireResult,
} from './sandbox';
import { toErrorMessage } from './utils';

// The run whose live kernels and canvases the next 'tests' request will use.
// Held until the next run replaces it (or the worker is terminated, which is
// how a runaway run's state gets collected).
let lastRun = null; // { token, internal }
let tokenSeq = 0;

// A log-flooding run must not also flood postMessage.
const MAX_STREAMED_LOGS = 300;

function lookupTask(taskRef) {
  if (!taskRef) return null;
  const found = getTaskBySlug(taskRef.uuid, taskRef.taskSlug);
  return found ? found.task : null;
}

function post(message, transfer) {
  self.postMessage(message, transfer && transfer.length ? transfer : undefined);
}

// Streams text lines only: the ImageBitmap in a canvas snapshot can be
// transferred exactly once, and that belongs to the final result.
function makeLogStreamer(id) {
  let sent = 0;
  return entry => {
    if (sent >= MAX_STREAMED_LOGS) return;
    sent++;
    post({ id, kind: 'log', log: { type: entry.type, time: entry.time, text: entry.text } });
  };
}

async function handleRun(msg) {
  const task = lookupTask(msg.taskRef);
  const internal = await executeRun(msg.code, {
    mode: msg.mode,
    task,
    controls: msg.controls,
    cardCapture: msg.cardCapture,
    onLog: makeLogStreamer(msg.id),
  });
  const token = `run-${++tokenSeq}`;
  lastRun = { token, internal };
  const { result, transfer } = toWireResult(internal, { runToken: token });
  post({ id: msg.id, kind: 'result', result }, transfer);
}

async function handleTests(msg) {
  const task = lookupTask(msg.taskRef);
  if (!task) {
    post({ id: msg.id, kind: 'result', result: { unknownTask: true } });
    return;
  }
  if (!lastRun || (msg.runToken && msg.runToken !== lastRun.token)) {
    // the run this report belongs to is gone (a newer run replaced it, or the
    // worker was recycled) — the supervisor synthesizes the report instead
    post({ id: msg.id, kind: 'result', result: { staleToken: true } });
    return;
  }
  const report = await executeTests(task, lastRun.internal);
  post({ id: msg.id, kind: 'result', result: report });
}

async function handleBenchmark(msg) {
  const task = lookupTask(msg.taskRef);
  const result = await executeBenchmark(msg.code, task);
  // the benchmark's own runs invalidate whatever state a previous run left
  lastRun = null;
  post({ id: msg.id, kind: 'result', result });
}

self.onmessage = async event => {
  const msg = event.data || {};
  try {
    switch (msg.kind) {
      case 'hello':
        post({
          id: msg.id,
          kind: 'result',
          result: { gpuSupported: gpuSupported(), sandbox: 'worker' },
        });
        break;
      case 'run':
        await handleRun(msg);
        break;
      case 'tests':
        await handleTests(msg);
        break;
      case 'benchmark':
        await handleBenchmark(msg);
        break;
      default:
        post({ id: msg.id, kind: 'failed', error: { message: `unknown request "${msg.kind}"` } });
    }
  } catch (e) {
    // executeRun/executeTests/executeBenchmark all swallow user errors, so
    // reaching here means the sandbox itself broke (a DataCloneError on an
    // exotic log value, an OOM…). Report it instead of hanging the caller.
    post({ id: msg.id, kind: 'failed', error: { message: toErrorMessage(e) } });
  }
};
