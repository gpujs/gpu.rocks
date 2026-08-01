/**
 * scripts/bench-size.mjs — is every workload sized for the measurement?
 *
 * A benchmark row is only worth reading if its plain-JS baseline runs long
 * enough to time and short enough to wait for. Too small and the clock, the
 * dispatch and the transfer dominate, so the row reports the harness. Too big
 * and nobody ever clicks Run all.
 *
 * This times ONLY the plain-JS baseline — no browser, no GPU — because that is
 * the number the sizing rule is written against and the one that runs anywhere.
 *
 *   node scripts/bench-size.mjs [id ...]
 *   node scripts/bench-size.mjs --file src/Bench/workloads/foo.js
 *
 * Exit code 1 if any workload lands outside the band, so it can gate a build.
 */
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');

// The band. Deliberately wide: a workload whose baseline is 250 ms and one at
// 2.5 s are both perfectly readable rows, and forcing every workload to the
// same duration would mean distorting its natural shape.
const MIN_MS = 200;
const MAX_MS = 3000;

const only = process.argv.slice(2).filter(a => !a.startsWith('--'));
// --file lets a workload be checked BEFORE it is in the generated registry,
// which is what makes it usable while several are being written at once.
const fileArg = process.argv.indexOf('--file');
const workloads = fileArg >= 0
  ? [(await import(`${ROOT}/${process.argv[fileArg + 1].replace(/^\.?\//, '')}`)).default]
  : (await import(`${ROOT}/src/Bench/workloads/index.js`)).default;

const median = xs => {
  const s = [...xs].sort((a, b) => a - b);
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
};

let bad = 0;
console.log(`sizing band: ${MIN_MS}–${MAX_MS} ms of plain JS\n`);
console.log('  status   baseline   build   workload');

for (const w of workloads) {
  if (only.length && !only.includes(w.id)) continue;

  const t0 = performance.now();
  const inputs = w.make ? w.make(w.size) : null;
  const build = performance.now() - t0;

  // one warm-up so the JIT has seen the loop, then three samples
  w.js(w.size, inputs);
  const samples = [];
  for (let i = 0; i < 3; i++) {
    const t = performance.now();
    w.js(w.size, inputs);
    samples.push(performance.now() - t);
  }
  const ms = median(samples);

  // A build that costs more than the work it feeds means the row would be
  // timing its own setup if anyone ever moved make() inside a column.
  const buildWarn = build > ms;
  const ok = ms >= MIN_MS && ms <= MAX_MS;
  if (!ok) bad++;
  const status = ok ? (buildWarn ? 'BUILD? ' : 'ok     ') : ms < MIN_MS ? 'TOO SML' : 'TOO BIG';
  console.log(
    `  ${status}  ${ms.toFixed(0).padStart(7)} ms  ${build.toFixed(0).padStart(5)} ms   ` +
      `${w.id.padEnd(22)} ${w.params}`
  );
  if (!ok) {
    const factor = ms < MIN_MS ? MIN_MS / ms : MAX_MS / ms;
    console.log(`           ^ scale the work by about ${factor.toFixed(1)}x`);
  }
  if (buildWarn) console.log('           ^ make() costs more than js() — check it is not doing the work');
}

console.log(bad ? `\n${bad} workload(s) outside the band` : '\nall workloads sized for measurement');
process.exit(bad ? 1 : 0);
