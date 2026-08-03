import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { GPU } from 'gpu.js';
import { ThemeProvider, useTheme } from '../Learn/ThemeContext';
import workloads, { GROUPS } from './workloads/index.js';
import savedRuns from './saved/index.js';
import { BASELINE, COLUMNS, runWorkload, webgpuStatus } from './runner.js';
import { SIGNATURE_IDS } from './signature.js';
import './scss/bench.scss';

// The benchmark page. Same visual language as the course — same tokens, same
// nav, same theme switch — because it is the same site, but none of the
// course's vocabulary: a row here is an operation and its parameters, legible
// to somebody who has never opened a lesson.

function capitalize(w) {
  return w.charAt(0).toUpperCase() + w.slice(1);
}

function Jelly() {
  return (
    <svg className="jelly" viewBox="0 0 24 24" width="26" height="26" aria-hidden="true">
      <path
        d="M12 3c-4.4 0-7 3.2-7 6.6 0 1.1.9 1.9 2 1.9h10c1.1 0 2-.8 2-1.9C19 6.2 16.4 3 12 3z"
        fill="#ff79c6"
      />
      <path
        d="M7.5 13.5c.3 2-.8 3.2-.5 5M11 14c.2 2.4-.6 3.6-.3 6M14.5 13.8c.3 2-.7 3.4-.4 5.4M17 13.5c.2 1.6-.6 2.6-.4 4.2"
        stroke="#20a4f3"
        strokeWidth="1.4"
        fill="none"
        strokeLinecap="round"
      />
    </svg>
  );
}

function BenchNav() {
  const { pref, cycleTheme } = useTheme();
  const [open, setOpen] = useState(false);
  const navRef = useRef(null);
  useEffect(() => {
    if (!open) return undefined;
    const onKey = e => e.key === 'Escape' && setOpen(false);
    const onDown = e => {
      if (navRef.current && !navRef.current.contains(e.target)) setOpen(false);
    };
    document.addEventListener('keydown', onKey);
    document.addEventListener('pointerdown', onDown);
    return () => {
      document.removeEventListener('keydown', onKey);
      document.removeEventListener('pointerdown', onDown);
    };
  }, [open]);

  return (
    <>
      {open && <div className="nav-scrim" aria-hidden="true" onClick={() => setOpen(false)} />}
      <nav className={open ? 'nav open' : 'nav'} ref={navRef}>
        <a className="brand" href="/" aria-label="GPU.js benchmark — home">
          <Jelly />
          <span className="brand-word">GPU.js</span> <span className="tag-pill">benchmark</span>
        </a>
        <button
          type="button"
          className="nav-burger"
          aria-expanded={open}
          aria-controls="bench-menu"
          aria-label={open ? 'Close menu' : 'Open menu'}
          onClick={() => setOpen(o => !o)}
        >
          <span aria-hidden="true">{open ? '✕' : '☰'}</span>
        </button>
        <div className="links" id="bench-menu" onClick={() => setOpen(false)}>
          <a href="/">Home</a>
          <a href="/learn/">Learn</a>
          <a href="/api/">API</a>
          <a href="/examples">Examples</a>
          <a href="/benchmark" className="active">Benchmark</a>
          <a href="https://github.com/gpujs/gpu.js">GitHub</a>
        </div>
        <button
          type="button"
          className="btn theme-btn"
          onClick={cycleTheme}
          aria-label={`Theme: ${capitalize(pref)} — tap to change`}
        >
          <span aria-hidden="true">{pref === 'dark' ? '☾' : pref === 'light' ? '☀' : '◐'}</span>
        </button>
      </nav>
    </>
  );
}

function fmtMs(ms) {
  if (ms >= 1000) return `${(ms / 1000).toFixed(2)} s`;
  if (ms >= 10) return `${ms.toFixed(0)} ms`;
  if (ms >= 1) return `${ms.toFixed(1)} ms`;
  if (ms >= 0.1) return `${ms.toFixed(2)} ms`;
  return `${(ms * 1000).toFixed(0)} µs`;
}

function fmtX(x) {
  return x < 1 ? `${x.toFixed(2)}×` : `${x.toFixed(1)}×`;
}

function Cell({ cell, base, isBaseline, bare }) {
  const cls = ['num', bare ? 'bare' : '', isBaseline ? 'baseline' : ''].filter(Boolean).join(' ');
  if (!cell) return <td className={cls}><span className="dash">—</span></td>;
  if (cell.running) {
    return (
      <td className={`${cls} working`}>
        <span className="spin" role="status" aria-label="measuring">
          <i />
          <i />
          <i />
        </span>
      </td>
    );
  }
  if (cell.na) return <td className={`${cls} na`} title={cell.reason}><span className="dash">N/A</span></td>;
  if (cell.error) return <td className={`${cls} err`} title={cell.error}><span className="dash">error</span></td>;
  if (cell.tooSlow) return <td className={`${cls} na`} title="one run exceeded the ceiling"><span className="dash">too slow</span></td>;

  const x = base && cell.ms ? base / cell.ms : null;
  const tone = isBaseline ? 'base' : !x ? 'flat' : x >= 1.5 ? 'win' : x < 0.95 ? 'lose' : 'flat';
  // The factor leads. It is the answer to the question the page asks; the
  // millisecond figure is the evidence for it, and reads better underneath.
  return (
    <td className={cls} title={cell.reps ? `median of ${cell.reps}` : undefined}>
      <span className={`x ${tone}`}>
        {cell.wrong ? 'WRONG' : isBaseline ? '1.0×' : x ? fmtX(x) : ''}
      </span>
      <span className="t">{fmtMs(cell.ms)}</span>
      {cell.fellBackTo && <span className="fell" title={`gpu.js actually ran this on ${cell.fellBackTo}`}>↘{cell.fellBackTo}</span>}
    </td>
  );
}

function BenchTable({ rows, results, onRun, onStop, running, readOnly }) {
  return (
    <div className="tablecard">
      <div className="scroll">
        <table>
          <thead>
            <tr>
              <th>Work</th>
              {COLUMNS.map(c => (
                <th key={c.id} className={c.bare ? 'bare' : undefined}>
                  {c.label}
                  <span className="g">{c.sub}</span>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map(w => {
              const cells = results[w.id] || {};
              const base = cells[BASELINE] && cells[BASELINE].ms;
              return (
                <tr
                  key={w.id}
                  // addressable by id: scripts/bench-record.mjs --only <id>
                  // drives a single row, and matching on the display name is a
                  // rename away from silently selecting nothing
                  data-workload={w.id}
                  className={cells.__running ? 'running' : undefined}
                >
                  {/* the grid lives on an inner div, never on the th: a th with
                      display:grid is no longer a table-cell, and position:sticky
                      stops tracking the scroll container — the header stayed put
                      while these scrolled away underneath it */}
                  <th scope="row" className="work">
                    <div className="workinner">
                      {/* disabled, not removed: pulling the button out of the
                          row would reflow every name the moment you switch
                          source, and the table would look like it changed */}
                      {/* A row that is not the one running cannot be started on
                          top of it, but the row that IS running stays clickable
                          — that button is the stop control. */}
                      <button
                        type="button"
                        className={`run-one${cells.__running ? ' isrunning' : ''}`}
                        onClick={() => (cells.__running ? onStop() : onRun([w]))}
                        disabled={readOnly || (running && !cells.__running)}
                        aria-label={
                          readOnly
                            ? `Benchmark ${w.name} — disabled while a saved run is shown`
                            : cells.__running
                              ? `Stop benchmarking ${w.name}`
                              : `Benchmark ${w.name}`
                        }
                      >
                        {cells.__running ? (
                          <>
                            <span className="runspin" aria-hidden="true" />
                            <svg className="stopicon" viewBox="0 0 24 24" width="14" height="14" aria-hidden="true">
                              <rect x="6" y="6" width="12" height="12" rx="2" fill="currentColor" />
                            </svg>
                          </>
                        ) : (
                          <svg viewBox="0 0 24 24" width="14" height="14" aria-hidden="true">
                            <polygon points="6 4 20 12 6 20" fill="currentColor" />
                          </svg>
                        )}
                      </button>
                      <span className="wname">
                        {w.name}
                        <span className="tag">{w.tag}</span>
                      </span>
                      <span className="wdesc">{w.params}</span>
                    </div>
                  </th>
                  {COLUMNS.map(c => (
                    <Cell
                      key={c.id}
                      cell={cells[c.id]}
                      base={base}
                      isBaseline={c.id === BASELINE}
                      bare={c.bare}
                    />
                  ))}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

// Reading order, not filename order. Dense matmul leads because it is the case
// every GPU claim starts from and the one a reader already has an intuition
// for; everything after it is grouped by the primitive it stresses, which is
// what the footer promises.
const ORDERED = (() => {
  const rank = new Map(GROUPS.map(([g], i) => [g, i]));
  const sig = new Map(SIGNATURE_IDS.map((id, i) => [id, i]));
  return [...workloads].sort((a, b) => {
    // The ten signature rows sit at the top in their own order, so brief mode
    // is literally the first ten rows rather than a different table. matmul is
    // first of those and therefore first overall.
    const sa = sig.has(a.id) ? sig.get(a.id) : Infinity;
    const sb = sig.has(b.id) ? sig.get(b.id) : Infinity;
    if (sa !== sb) return sa - sb;
    const byGroup = (rank.has(a.group) ? rank.get(a.group) : 99) - (rank.has(b.group) ? rank.get(b.group) : 99);
    return byGroup || a.name.localeCompare(b.name);
  });
})();

const BRIEF = ORDERED.filter(w => SIGNATURE_IDS.includes(w.id));

// Rough, and honest about being rough: measured wall time for a full pass is
// tens of minutes and depends entirely on the machine. The number exists to set
// an expectation before someone commits to it, not to be accurate.
const MINUTES_FULL = 30;

function BenchPage() {
  // The effective theme ('light' | 'dark'), never the preference: 'auto' is
  // resolved against prefers-color-scheme by the provider, and the stylesheet
  // only knows [data-theme='dark']. Without this the toggle changed its own
  // glyph and nothing else, and auto never resolved at all.
  const { theme } = useTheme();
  const [source, setSource] = useState('live');
  // Off by default. Brief is the ten most flattering rows, and a benchmark
  // that opens on its own best case has chosen what to show you before you
  // asked. The full run is the default and the warning explains the cost.
  const [brief, setBrief] = useState(false);
  const [cols, setCols] = useState(() => new Set(COLUMNS.map(c => c.id)));
  const [confirmFull, setConfirmFull] = useState(false);
  const [savedId, setSavedId] = useState(savedRuns.length ? savedRuns[0].id : '');
  const [live, setLive] = useState({});
  const [status, setStatus] = useState('idle');
  const [gpuInfo, setGpuInfo] = useState('');
  const abortRef = useRef(null);
  const workerRef = useRef(null);
  const pendingRef = useRef(null);

  // The suite runs off the main thread: the plain-JS baseline and the gpu.js
  // CPU backend are synchronous multi-second loops, and on the main thread one
  // row blocked for 23 s and Stop could not be clicked.
  const spawnWorker = useCallback(() => {
    try {
      const w = new Worker(new URL('./bench.worker.js', import.meta.url), { type: 'module' });
      w.onerror = () => {
        workerRef.current = null;
      };
      return w;
    } catch (e) {
      return null;
    }
  }, []);

  useEffect(() => {
    workerRef.current = spawnWorker();
    return () => {
      if (workerRef.current) workerRef.current.terminate();
      workerRef.current = null;
    };
  }, [spawnWorker]);

  useEffect(() => {
    let dead = false;
    webgpuStatus().then(s => {
      if (!dead) setGpuInfo(s.ok ? s.reason : `${s.reason}`);
    });
    return () => {
      dead = true;
    };
  }, []);

  const saved = useMemo(() => savedRuns.find(r => r.id === savedId), [savedId]);
  const readOnly = source === 'saved';
  const results = readOnly ? (saved ? saved.results : {}) : live;

  // One workload at a time, in the worker. Serial on purpose: two workloads
  // measured at once would be measuring each other.
  const runOne = useCallback((w, ctl) => new Promise(resolve => {
    const worker = workerRef.current;
    if (!worker) {
      // no Worker (or it failed to construct): run here and accept the freeze,
      // which is still better than the page not working at all
      runWorkload(w, {
        GPU,
        signal: ctl,
        columns: [...cols],
        onCell: (colId, cell) =>
          setLive(prev => ({ ...prev, [w.id]: { ...(prev[w.id] || {}), __running: true, [colId]: cell } })),
      }).then(cells => {
        setLive(prev => ({ ...prev, [w.id]: cells }));
        resolve();
      });
      return;
    }
    const id = `${w.id}:${Date.now()}`;
    // stop() kills the worker mid-measurement, and this promise is what the run
    // loop is awaiting — without a way to settle it the loop would hang on a
    // worker that no longer exists.
    pendingRef.current = { id, workloadId: w.id, resolve };
    const onMessage = e => {
      const m = e.data || {};
      if (m.id !== id) return;
      if (m.cellId) {
        setLive(prev => ({ ...prev, [w.id]: { ...(prev[w.id] || {}), __running: true, [m.cellId]: m.cell } }));
        return;
      }
      worker.removeEventListener('message', onMessage);
      pendingRef.current = null;
      setLive(prev => ({ ...prev, [w.id]: m.failed ? { __error: m.error } : m.cells }));
      resolve();
    };
    worker.addEventListener('message', onMessage);
    // columns travels with every request: the no-worker fallback above read
    // `cols` directly and this path did not, so unchecking a column changed
    // nothing at all on the normal path.
    worker.postMessage({ id, workloadId: w.id, columns: [...cols] });
  }), [cols]);

  const run = useCallback(async list => {
    const ctl = { aborted: false };
    abortRef.current = ctl;
    for (let i = 0; i < list.length; i++) {
      if (ctl.aborted) break;
      const w = list[i];
      setStatus(`running ${w.name}${list.length > 1 ? ` — ${i + 1} of ${list.length}` : ''}`);
      setLive(prev => ({ ...prev, [w.id]: { __running: true } }));
      // eslint-disable-next-line no-await-in-loop
      await runOne(w, ctl);
    }
    abortRef.current = null;
    setStatus(ctl.aborted ? 'stopped' : 'done');
  }, [runOne]);

  // How scripts/bench-record.mjs lifts a finished table out of the page. A
  // recorded run has to be the SAME numbers the page just showed, so it is read
  // from the page rather than re-measured somewhere else.
  useEffect(() => {
    window.__benchResults = live;
    window.__benchStatus = status;
  }, [live, status]);

  const running = Boolean(abortRef.current) || status.startsWith('running');
  // Setting the flag alone only ends the run BETWEEN workloads: a row is one
  // postMessage and the worker is busy in a compute loop, so a stop pressed
  // during Conway's Life did nothing visible for the several seconds that row
  // had left. Pressing stop should stop. So the worker is terminated outright
  // and a fresh one takes its place — that also releases the GPU contexts the
  // dead one was holding — and the promise the run loop is awaiting is settled
  // by hand, since no message is ever coming back for it.
  const stop = () => {
    if (abortRef.current) abortRef.current.aborted = true;
    const pending = pendingRef.current;
    pendingRef.current = null;
    if (workerRef.current) {
      workerRef.current.terminate();
      workerRef.current = spawnWorker();
    }
    if (pending) {
      // keep whatever columns did land, drop the running marker
      setLive(prev => {
        const row = { ...(prev[pending.workloadId] || {}) };
        delete row.__running;
        return { ...prev, [pending.workloadId]: row };
      });
      pending.resolve();
    }
  };

  return (
    <div className="bench-root" data-theme={theme}>
      <BenchNav />
      <div className="wrap">
        <p className="eyebrow">gpu.js · benchmark</p>
        <h1>What the GPU is actually worth</h1>
        <p className="sub">
          {workloads.length} GPGPU workload{workloads.length === 1 ? '' : 's'}, timed on every backend
          gpu.js can reach and against hand-written baselines. The rightmost column is plain
          JavaScript with no gpu.js in the picture — every speed-up on this page is measured
          against it. Each column runs the best implementation for its platform, not the same
          code six times; what is held identical is the problem, and every answer is checked.
        </p>

        <div className="toolbar">
          {/* Both stay put and grey out. A saved run is read-only, but hiding
              the controls moved every other control in the bar, which reads as
              the page having changed rather than the source having. */}
          {/* One control, three states: play, spinning while it works, and a
              stop square under the pointer. A separate Stop button spent a
              permanent slot in the bar on something that is only meaningful
              for the minutes a run is going, and left the run button looking
              pressable while it was doing nothing. */}
          <button
            type="button"
            className={`btn primary${running ? ' isrunning' : ''}`}
            disabled={readOnly}
            title={readOnly ? 'Showing a saved run — switch to Live to measure on this machine' : undefined}
            aria-label={running ? 'Stop the run' : undefined}
            onClick={() => {
              if (running) return stop();
              return brief ? run(BRIEF) : setConfirmFull(true);
            }}
          >
            {running ? (
              <>
                <span className="runspin" aria-hidden="true" />
                <svg className="stopicon" viewBox="0 0 24 24" width="16" height="16" aria-hidden="true">
                  <rect x="6" y="6" width="12" height="12" rx="2" fill="currentColor" />
                </svg>
                <span className="lbl-run">Running</span>
                <span className="lbl-stop">Stop</span>
              </>
            ) : (
              <>
                <svg viewBox="0 0 24 24" width="16" height="16" aria-hidden="true">
                  <polygon points="6 4 20 12 6 20" fill="currentColor" />
                </svg>
                {brief ? `Run ${BRIEF.length}` : `Run all ${workloads.length}`}
              </>
            )}
          </button>
          <div className="seg" role="group" aria-label="Result source">
            <button type="button" aria-pressed={source === 'live'} onClick={() => setSource('live')}>
              Live
            </button>
            <button
              type="button"
              aria-pressed={source === 'saved'}
              onClick={() => setSource('saved')}
              disabled={!savedRuns.length}
            >
              Saved run
            </button>
          </div>
          {readOnly && (
            <label className="pick">
              <span className="lbl">Run</span>
              <select value={savedId} onChange={e => setSavedId(e.target.value)}>
                {savedRuns.map(r => (
                  <option key={r.id} value={r.id}>
                    {r.label}
                  </option>
                ))}
              </select>
            </label>
          )}
          <span className="grow" />
          <span className="stat">
            {readOnly ? (saved ? saved.machine : 'no saved runs') : `${status} · ${gpuInfo}`}
          </span>
        </div>

        {!readOnly && (
          <div className="opts">
            <label className="opt brief">
              <input type="checkbox" checked={brief} onChange={e => setBrief(e.target.checked)} />
              <span>
                <b>Brief</b> — the {BRIEF.length} biggest wins in the saved run, about a tenth
                of the time. Not a representative sample: the rows where the GPU loses are
                in the full run.
              </span>
            </label>
            <div className="opt cols">
              <span className="lbl">Columns</span>
              {COLUMNS.map(c => {
                const isBase = c.id === BASELINE;
                return (
                  <label key={c.id} className={c.bare ? 'bare' : undefined}>
                    <input
                      type="checkbox"
                      // addressable by column id, so tooling can select a
                      // column without matching on its label
                      data-col={c.id}
                      checked={cols.has(c.id) || isBase}
                      disabled={isBase}
                      title={isBase ? 'the baseline every speed-up divides by — always measured' : undefined}
                      onChange={e =>
                        setCols(prev => {
                          const next = new Set(prev);
                          if (e.target.checked) next.add(c.id);
                          else next.delete(c.id);
                          return next;
                        })
                      }
                    />
                    <span>
                      {c.label}
                      <i>{c.sub}</i>
                    </span>
                  </label>
                );
              })}
            </div>
          </div>
        )}

        {readOnly && saved && (
          <div className="ro">
            <b>Read-only.</b>
            <span>
              Showing a saved run — <code>{saved.machine}</code>, {saved.date}. Running is disabled
              so a stored result can never be half-overwritten by this machine.
            </span>
          </div>
        )}

        {confirmFull && (
          <div className="runwarn-wrap" onMouseDown={e => e.target === e.currentTarget && setConfirmFull(false)}>
            <div className="runwarn" role="dialog" aria-modal="true" aria-labelledby="full-title">
              <p className="eyebrow">Full run</p>
              <h2 id="full-title">This will take about half an hour</h2>
              <p>
                All {workloads.length} workloads across {cols.size} column
                {cols.size === 1 ? '' : 's'}, measured properly — warm-ups, then a median of at
                least three runs each. Longer on a slower machine, and it holds the GPU the
                whole time.
              </p>
              <p className="alt">
                If you do not have half an hour: <b>Brief</b> runs the {BRIEF.length} rows
                where gpu.js gained the most, in roughly a tenth of the time — the best case
                rather than the whole picture. Or read a <b>saved run</b> — a full table
                already measured on a known machine.
              </p>
              <div className="runwarn-actions">
                <button type="button" className="btn" onClick={() => setConfirmFull(false)}>
                  Cancel
                </button>
                <button
                  type="button"
                  className="btn"
                  onClick={() => {
                    setConfirmFull(false);
                    setSource('saved');
                  }}
                  disabled={!savedRuns.length}
                >
                  Read a saved run
                </button>
                <button
                  type="button"
                  className="btn"
                  onClick={() => {
                    setConfirmFull(false);
                    setBrief(true);
                    run(BRIEF);
                  }}
                >
                  Run brief instead
                </button>
                <button
                  type="button"
                  className="btn primary"
                  onClick={() => {
                    setConfirmFull(false);
                    run(ORDERED);
                  }}
                >
                  Run all {workloads.length}
                </button>
              </div>
            </div>
          </div>
        )}

        <BenchTable
          rows={brief && !readOnly ? BRIEF : ORDERED}
          results={results}
          onRun={run}
          onStop={stop}
          running={running}
          readOnly={readOnly}
        />

        <div className="legend">
          <div>
            <b>The baseline is the last column</b>
            <p>Plain JavaScript, no gpu.js. Every × on the row is that column divided by the cell.</p>
          </div>
          <div>
            <b>Two columns have no gpu.js in them</b>
            <p>Hand-written WebGPU and plain JS, tinted amber. They separate “the GPU is fast” from
              “gpu.js is fast”.</p>
          </div>
          <div>
            <b>N/A means this machine, not this workload</b>
            <p>Every workload here runs on every backend. An N/A is about where you are reading
              from — no WebGPU adapter, or a page served over plain http, which does not expose
              <code> navigator.gpu</code> at all. Hover it for the reason.</p>
          </div>
          <div>
            <b>Each column is the best that platform can do</b>
            <p>Not one algorithm compiled six ways. Selecting the top 512 is a heap in plain JS
              and a threshold bisection on a GPU, because that is what each is good at. What is
              held identical is the <em>problem</em> and the answer — every column is checksummed
              against the same oracle.</p>
          </div>
          <div>
            <b>Median of ≥3, after 2 warm-ups</b>
            <p>The first call compiles. Every column is also checksummed against the plain-JS
              result, and a mismatch reads WRONG rather than fast.</p>
          </div>
        </div>
      </div>
    </div>
  );
}

export default function BenchApp() {
  return (
    <ThemeProvider>
      <BenchPage />
    </ThemeProvider>
  );
}
