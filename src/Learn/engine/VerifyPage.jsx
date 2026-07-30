import React, { useEffect } from 'react';
import { modules } from '../content/index.js';
import { runUserCode, runTests, sandboxGpuSupported, sandboxInfo } from './runner';

// Headless verification hook for scripts/verify-learn.mjs.
//
// It goes through runUserCode/runTests — the exact pipeline a learner's ▶ Run
// uses, worker sandbox and watchdog included — so what this verifies is what
// users actually run. It does NOT reach into the execution core directly.
//
// window.__verifyLearn({ moduleId?, mode = 'cpu' }) → JSON-serializable:
//   { skipped: true, reason }                        // mode 'gpu' without WebGL
//   { gpuSupported, sandbox, mode, tasks: [taskReport] }
//
// taskReport:
//   { id, slug, title, ok,
//     metadata: { ok, problems: [] },
//     solution: { ok, problems: [], tests: [{ name, private, passed, ms, error? }] },
//     starter:  { ok, problem? } }                   // ok ⇔ starter does NOT pass
//
// Checks per task (contract):
//   1. solutionCode runs clean and passes ALL public + private tests;
//   2. starterCode does NOT already pass (≥1 failing test, or the run errors);
//   3. metadata is well-formed (non-empty intro/goal/requirements, ≥1 hint,
//      ≥1 public and ≥1 private test, unique slug).

function nonEmptyString(value) {
  return typeof value === 'string' && value.trim().length > 0;
}

function checkMetadata(task, module, seenSlugs) {
  const problems = [];
  if (!nonEmptyString(task.slug)) problems.push('missing slug');
  else if (seenSlugs.has(task.slug)) problems.push(`duplicate slug "${task.slug}"`);
  else seenSlugs.add(task.slug);
  if (!nonEmptyString(task.title)) problems.push('missing title');
  if (!nonEmptyString(task.intro)) problems.push('empty intro');
  if (!nonEmptyString(task.goal)) problems.push('empty goal');
  if (!Array.isArray(task.requirements) || task.requirements.length === 0 ||
      !task.requirements.every(nonEmptyString)) {
    problems.push('requirements must be a non-empty list of non-empty strings');
  }
  if (!Array.isArray(task.hints) || task.hints.length < 1 ||
      !task.hints.every(h => h && nonEmptyString(h.title) && nonEmptyString(h.body))) {
    problems.push('need ≥1 hint with title and body');
  }
  if (!nonEmptyString(task.transfer)) problems.push('empty transfer box');
  if (!nonEmptyString(task.starterCode)) problems.push('empty starterCode');
  if (!nonEmptyString(task.solutionCode)) problems.push('empty solutionCode');
  const validTest = t => t && nonEmptyString(t.name) && typeof t.run === 'function';
  if (!Array.isArray(task.publicTests) || task.publicTests.length < 1 ||
      !task.publicTests.every(validTest)) {
    problems.push('need ≥1 public test with name and run()');
  }
  if (!Array.isArray(task.privateTests) || task.privateTests.length < 1 ||
      !task.privateTests.every(validTest)) {
    problems.push('need ≥1 private test with name and run()');
  }
  return { ok: problems.length === 0, problems };
}

async function verifyTask(task, module, index, mode, seenSlugs) {
  const id = `${module.id}-${index + 1}`;
  const metadata = checkMetadata(task, module, seenSlugs);

  // 1. the reference solution must run clean and pass everything
  const solution = { ok: false, problems: [], tests: [] };
  try {
    const run = await runUserCode(task.solutionCode, { mode, task });
    if (!run.ok) {
      solution.problems.push(`solution run failed: ${run.error.message}`);
    } else {
      const tested = await runTests(task, run);
      solution.tests = tested.results.map(r => ({
        name: r.name,
        private: r.private,
        passed: r.passed,
        ms: Math.round(r.ms * 10) / 10,
        error: r.error,
      }));
      if (!tested.allPassed) {
        tested.results
          .filter(r => !r.passed)
          .forEach(r => solution.problems.push(`test "${r.name}" failed: ${r.error}`));
      }
      solution.ok = tested.allPassed;
    }
  } catch (e) {
    solution.problems.push(`verifier error: ${e.message}`);
  }

  // 2. the starter code must NOT already pass
  const starter = { ok: false, problem: null };
  try {
    const run = await runUserCode(task.starterCode, { mode, task });
    if (!run.ok) {
      starter.ok = true; // starter erroring out counts as "does not pass"
    } else {
      const tested = await runTests(task, run);
      if (tested.allPassed) {
        starter.problem = 'starter code already passes every test — the task completes itself';
      } else {
        starter.ok = true;
      }
    }
  } catch (e) {
    starter.problem = `verifier error: ${e.message}`;
  }

  return {
    id,
    slug: task.slug,
    title: task.title,
    metadata,
    solution,
    starter,
    ok: metadata.ok && solution.ok && starter.ok,
  };
}

function VerifyPage() {
  useEffect(() => {
    // where the sandbox lives, whether IT has WebGL, and what spawning cost —
    // reported by the harness, and handy when debugging by hand
    window.__learnSandboxInfo = () => sandboxInfo();
    window.__verifyLearn = async ({ moduleId, mode = 'cpu' } = {}) => {
      // ask the thread that will actually build the kernels, not this one
      const gpuSupported = await sandboxGpuSupported();
      const sandbox = await sandboxInfo();
      if (mode === 'gpu' && !gpuSupported) {
        return {
          skipped: true,
          reason: 'WebGL unavailable in the sandbox',
          gpuSupported,
          sandbox,
          mode,
        };
      }
      const selected = moduleId ? modules.filter(m => m.id === moduleId) : modules;
      if (moduleId && selected.length === 0) {
        return { error: `no module with id "${moduleId}"`, gpuSupported, sandbox, mode };
      }
      const tasks = [];
      for (const module of selected) {
        const seenSlugs = new Set();
        for (let i = 0; i < module.tasks.length; i++) {
          // eslint-disable-next-line no-await-in-loop
          tasks.push(await verifyTask(module.tasks[i], module, i, mode, seenSlugs));
        }
      }
      return { gpuSupported, sandbox, mode, tasks };
    };
    return () => {
      delete window.__verifyLearn;
      delete window.__learnSandboxInfo;
    };
  }, []);
  return <div className="mono">verify harness ready — window.__verifyLearn(&#123;moduleId, mode&#125;)</div>;
}

export default VerifyPage;
