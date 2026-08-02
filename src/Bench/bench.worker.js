/**
 * src/Bench/bench.worker.js — the suite runs here, not on the main thread.
 *
 * WHY. Two of the six columns are synchronous by nature: the plain-JS baseline
 * IS a JavaScript loop, and gpu.js's CPU backend transpiles to one. Measured on
 * the main thread, a single row blocked for 23.2 s in total with a worst task of
 * 14.1 s — the page rendered at about 9 fps and no button could be clicked. A
 * benchmark whose Stop button cannot be pressed while it runs is not usable, and
 * the reading is worse too: layout and paint compete with the thing being timed.
 *
 * The course's task engine already does this (engine/sandbox.worker.js), so the
 * shape is proven here — including gpu.js on a WebGL backend inside a worker,
 * which needs an OffscreenCanvas because there is no document to make one from.
 *
 * Protocol, deliberately tiny:
 *     in   { id, workloadId, columns? }
 *     out  { id, cellId, cell }     one per column, as it finishes
 *          { id, done: true, cells }
 *          { id, failed: true, error }
 *
 * Results cross as structured-cloneable plain objects. Nothing live (no kernel,
 * no GPU, no canvas) ever leaves this file.
 */
import { GPU } from 'gpu.js';
import workloads from './workloads/index.js';
import { runWorkload } from './runner.js';

const byId = new Map(workloads.map(w => [w.id, w]));

// gpu.js asks the document for a canvas when it is not given one, and a worker
// has no document. One per GPU instance rather than one shared: a GL context is
// bound to its canvas, and the runner builds a fresh GPU per column.
const makeCanvas = () => new OffscreenCanvas(1, 1);

self.onmessage = async event => {
  const { id, workloadId, columns } = event.data || {};
  const workload = byId.get(workloadId);
  if (!workload) {
    self.postMessage({ id, failed: true, error: `unknown workload: ${workloadId}` });
    return;
  }
  try {
    const cells = await runWorkload(workload, {
      GPU,
      makeCanvas,
      columns,
      // stream each column as it lands so a long row fills in rather than
      // appearing all at once at the end
      onCell: (cellId, cell) => self.postMessage({ id, cellId, cell }),
    });
    self.postMessage({ id, done: true, cells });
  } catch (e) {
    self.postMessage({ id, failed: true, error: String((e && e.message) || e).slice(0, 200) });
  }
};
