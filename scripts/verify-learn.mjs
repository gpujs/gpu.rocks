/**
 * Content verification for the /learn course, in a real browser.
 *
 * For every task (optionally scoped to one module) and each requested mode it
 * checks, via the window.__verifyLearn hook on /learn-verify:
 *   1. the reference solution runs clean and passes ALL public+private tests;
 *   2. the starter code does NOT already pass;
 *   3. the task metadata is well-formed.
 *
 * The hook runs the SAME pipeline a learner's ▶ Run uses — the worker sandbox
 * in engine/sandbox.worker.js, supervised by engine/runner.js — so a task that
 * passes here is a task that works in the app.
 *
 * Usage: node scripts/verify-learn.mjs [--module <ref>] [--mode cpu|gpu|both]
 *                                      [--base http://localhost:4173]
 *   --module names one module by uuid, short id ('f1399353'), slug
 *   ('hello-kernel') or legacy id ('1-2').
 *   --mode defaults to both; gpu is skipped gracefully when headless WebGL
 *   is unavailable.
 *   --base checks an already-running server instead of spawning one — use it
 *   to gate the PRODUCTION build (`yarn build && yarn preview`), where the
 *   worker bundle is a separate minified chunk and a dev-only worker URL would
 *   go unnoticed.
 *
 * Without --base it spawns `yarn vite --port 0` (ephemeral port — parallel
 * agents run this concurrently, so the port is parsed from vite's output,
 * never hardcoded).
 */
import { spawn } from 'node:child_process';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { launch } from './browser.mjs';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');

// ---- args -----------------------------------------------------------------

const args = process.argv.slice(2);
let moduleRef = null;
let modeArg = 'both';
let baseArg = null;
for (let i = 0; i < args.length; i++) {
  if (args[i] === '--module') moduleRef = args[++i];
  else if (args[i] === '--mode') modeArg = args[++i];
  else if (args[i] === '--base') baseArg = args[++i];
  else {
    console.error(`unknown argument: ${args[i]}`);
    process.exit(2);
  }
}
if (!['cpu', 'gpu', 'both'].includes(modeArg)) {
  console.error(`--mode must be cpu, gpu or both (got "${modeArg}")`);
  process.exit(2);
}
const modes = modeArg === 'both' ? ['cpu', 'gpu'] : [modeArg];

// ---- dev server on an ephemeral port --------------------------------------

const ANSI = /\[[0-9;]*m/g;

function startServer() {
  return new Promise((resolvePort, reject) => {
    const child = spawn('yarn', ['vite', '--port', '0'], {
      cwd: root,
      detached: true,
      stdio: ['ignore', 'pipe', 'pipe'],
    });
    let output = '';
    const timer = setTimeout(() => {
      reject(new Error(`vite did not report a port within 60s. Output:\n${output}`));
    }, 60000);
    const onData = chunk => {
      output += chunk.toString();
      const match = output.replace(ANSI, '').match(/Local:\s+https?:\/\/(?:localhost|127\.0\.0\.1):(\d+)/);
      if (match) {
        clearTimeout(timer);
        resolvePort({ child, port: Number(match[1]) });
      }
    };
    child.stdout.on('data', onData);
    child.stderr.on('data', onData);
    child.on('exit', code => {
      clearTimeout(timer);
      reject(new Error(`vite exited early (code ${code}). Output:\n${output}`));
    });
  });
}

function stopServer(child) {
  if (!child || child.exitCode !== null) return;
  try {
    process.kill(-child.pid, 'SIGTERM'); // whole process group: yarn + vite
  } catch (e) {
    try { child.kill('SIGTERM'); } catch (e2) { /* already gone */ }
  }
}

// ---- report printing ------------------------------------------------------

function printModeReport(mode, report) {
  const sandbox = report.sandbox
    ? `, sandbox=${report.sandbox.path}${
        report.sandbox.spawnMs != null ? ` in ${Math.round(report.sandbox.spawnMs)}ms` : ''
      }${report.sandbox.reason ? ` — ${report.sandbox.reason}` : ''}`
    : '';
  console.log(`\n== mode: ${mode} (gpuSupported=${report.gpuSupported}${sandbox}) ==`);
  const pad = (s, w) => String(s).padEnd(w);
  console.log(pad('RESULT', 7) + pad('TASK', 13) + pad('META', 6) + pad('SOLUTION', 10) + pad('STARTER', 9) + 'TITLE');
  let failures = 0;
  for (const t of report.tasks) {
    const solved = t.solution.tests.filter(r => r.passed).length;
    const total = t.solution.tests.length;
    console.log(
      pad(t.ok ? 'PASS' : 'FAIL', 7) +
      pad(t.id, 13) +
      pad(t.metadata.ok ? 'ok' : 'BAD', 6) +
      pad(total ? `${solved}/${total}` : 'ERROR', 10) +
      pad(t.starter.ok ? 'fails✓' : 'PASSES', 9) +
      t.title
    );
    if (!t.ok) {
      failures++;
      t.metadata.problems.forEach(p => console.log(`         ✗ metadata: ${p}`));
      t.solution.problems.forEach(p => console.log(`         ✗ solution: ${p}`));
      if (t.starter.problem) console.log(`         ✗ starter: ${t.starter.problem}`);
    }
  }
  return failures;
}

// ---- main -----------------------------------------------------------------

let server = null;
let browser = null;
let failures = 0;

try {
  let base = baseArg;
  if (base) {
    base = base.replace(/\/$/, '');
    console.log(`checking the server already running on ${base}`);
  } else {
    const started = await startServer();
    server = started.child;
    base = `http://localhost:${started.port}`;
    console.log(`vite dev server on ${base}`);
  }

  browser = await launch();
  const page = await browser.newPage();
  page.setDefaultTimeout(180000);
  page.on('pageerror', e => console.error(`(pageerror) ${String(e).split('\n')[0]}`));

  await page.goto(`${base}/learn-verify`, { waitUntil: 'networkidle2', timeout: 60000 });
  await page.waitForFunction('typeof window.__verifyLearn === "function"', { timeout: 30000 });

  for (const mode of modes) {
    const report = await page.evaluate(
      opts => window.__verifyLearn(opts),
      { module: moduleRef || undefined, mode }
    );
    if (report.skipped) {
      console.log(`\n== mode: ${mode} — SKIPPED (${report.reason}) ==`);
      continue;
    }
    if (report.error) {
      console.error(`\n== mode: ${mode} — ERROR: ${report.error} ==`);
      failures++;
      continue;
    }
    if (!report.tasks.length) {
      console.error(`\n== mode: ${mode} — no tasks found${moduleRef ? ` for module ${moduleRef}` : ''} ==`);
      failures++;
      continue;
    }
    failures += printModeReport(mode, report);
  }
} catch (e) {
  console.error(`verify-learn fatal: ${e.message}`);
  failures++;
} finally {
  if (browser) await browser.close().catch(() => {});
  stopServer(server);
}

console.log(failures ? `\n${failures} task check(s) failed` : '\nall task checks passed');
process.exit(failures ? 1 : 0);
