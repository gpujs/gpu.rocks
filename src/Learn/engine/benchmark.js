// engine/benchmark.js — CPU-vs-GPU comparison for the toolbar's ⏱ button.
//
// runBenchmark(code, task) → Promise:
//   { cpuMs, gpuMs, ratio, fasterOn: 'gpu' | 'cpu', gpuRanOnCpu }  // medians
//   { gpuUnavailable: true, cpuMs? }                    // no WebGL in the sandbox
//   { gpuFailed: true, cpuMs, error }                   // code failed in gpu mode
//   { error: { message } }                              // user code failed
//
// The measurement itself (two runs, warm-up, adaptive timed loop of ≥5
// iterations or ≥250 ms, medians) lives in engine/sandbox.js and executes in
// the sandbox worker: a benchmark re-invokes every recorded kernel several
// times, so it is if anything MORE capable of hanging than a plain run and has
// to be just as terminable. This module is only the bridge.

import { runBenchmarkInSandbox } from './runner';

export async function runBenchmark(code, task) {
  return runBenchmarkInSandbox(code, task);
}
