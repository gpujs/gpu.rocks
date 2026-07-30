// engine/benchmark.js — CPU-vs-GPU comparison for the toolbar's ⏱ button.
//
// runBenchmark(code, task) → Promise:
//   { cpuMs, gpuMs, ratio, fasterOn: 'gpu' | 'cpu' }   // medians
//   { gpuUnavailable: true, cpuMs? }                    // no WebGL here
//   { gpuFailed: true, cpuMs, error }                   // code failed in gpu mode
//   { error: { message } }                              // user code failed
//
// Strategy: run the user code twice (mode 'cpu', then mode 'gpu'). For each
// run: one warm-up call, then an adaptive timed loop re-invoking every
// recorded kernel with its .lastArgs — stop at ≥5 iterations or ≥250 ms —
// and report the median iteration time.

import { runUserCode, isGPUSupported, toErrorMessage } from './runner';

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

export async function runBenchmark(code, task) {
  try {
    const cpuRun = await runUserCode(code, { mode: 'cpu', task });
    if (!cpuRun.ok) return { error: cpuRun.error };
    const cpuKernels = invokableKernels(cpuRun);
    if (!cpuKernels.length) {
      return { error: { message: 'nothing to benchmark — the code never invoked a kernel' } };
    }
    const cpuMs = timeKernels(cpuKernels);

    if (!isGPUSupported()) return { gpuUnavailable: true, cpuMs };

    // WebGL exists here, so a failure below is the user's code failing in gpu
    // mode (the GL backend rejects some code the cpu backend tolerates) — that
    // is gpuFailed, never gpuUnavailable.
    const gpuRun = await runUserCode(code, { mode: 'gpu', task });
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
