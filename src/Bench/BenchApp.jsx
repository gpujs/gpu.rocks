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
        <a className="brand" href="/" aria-label="GPU.js Benchmark Gauntlet — home">
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

/**
 * A command block you can take with you.
 *
 * navigator.clipboard is NOT always there: it needs a secure context, and this
 * page is routinely read over plain http at a LAN address while someone tests
 * on their phone — exactly the reader most likely to want to copy a docker
 * command. So the modern API is tried first and the old execCommand path is
 * the fallback, and if both fail the button says so instead of pretending.
 */
function CmdBlock({ children }) {
  const [state, setState] = useState('idle');
  const copy = async () => {
    const text = String(children);
    try {
      if (navigator.clipboard && window.isSecureContext) {
        await navigator.clipboard.writeText(text);
      } else {
        const ta = document.createElement('textarea');
        ta.value = text;
        // off-screen rather than hidden: a display:none textarea cannot be
        // selected, and the copy silently does nothing
        ta.style.cssText = 'position:fixed;top:-1000px;opacity:0';
        document.body.appendChild(ta);
        ta.select();
        const ok = document.execCommand && document.execCommand('copy');
        document.body.removeChild(ta);
        if (!ok) throw new Error('execCommand refused');
      }
      setState('done');
    } catch (e) {
      setState('failed');
    }
    setTimeout(() => setState('idle'), 2000);
  };
  return (
    <div className="cmdwrap">
      <pre className="cmd"><code>{children}</code></pre>
      <button
        type="button"
        className={`copybtn${state !== 'idle' ? ` ${state}` : ''}`}
        onClick={copy}
        aria-label="Copy these commands"
      >
        {state === 'done' ? 'Copied' : state === 'failed' ? 'Select it' : 'Copy'}
      </button>
    </div>
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

  // Runs opened from a file live alongside the ones this repo ships. They are
  // not persisted anywhere: a reload drops them, which is the honest lifetime
  // for a file someone handed the page once.
  const [loadedRuns, setLoadedRuns] = useState([]);
  const [loadError, setLoadError] = useState('');
  const allRuns = useMemo(() => [...loadedRuns, ...savedRuns], [loadedRuns]);
  const saved = useMemo(() => allRuns.find(r => r.id === savedId), [allRuns, savedId]);
  const readOnly = source === 'saved';
  const results = readOnly ? (saved ? saved.results : {}) : live;

  // A saved run is JSON someone else produced, so it is checked before it is
  // trusted enough to render. Not for safety — React escapes what it prints —
  // but because a malformed file should say what is wrong with it rather than
  // rendering as an empty table and looking like a bug in the page.
  const openRunFile = async file => {
    setLoadError('');
    if (!file) return;
    let parsed;
    try {
      parsed = JSON.parse(await file.text());
    } catch (e) {
      setLoadError(`${file.name} is not valid JSON`);
      return;
    }
    if (!parsed || typeof parsed !== 'object' || !parsed.results || typeof parsed.results !== 'object') {
      setLoadError(`${file.name} has no results — is it a saved run?`);
      return;
    }
    const known = Object.keys(parsed.results).filter(id => workloads.some(w => w.id === id));
    if (!known.length) {
      setLoadError(`${file.name} has no workloads this page knows about`);
      return;
    }
    const run = {
      ...parsed,
      // fall back rather than render undefined: an older or hand-made file may
      // be missing the descriptive fields without being useless
      id: String(parsed.id || file.name.replace(/\.json$/, '')),
      label: String(parsed.label || file.name),
      machine: String(parsed.machine || 'unknown machine'),
      date: String(parsed.date || ''),
      gpujs: String(parsed.gpujs || '?'),
      fromFile: true,
    };
    setLoadedRuns(prev => [run, ...prev.filter(r => r.id !== run.id)]);
    setSavedId(run.id);
    setSource('saved');
    if (known.length < Object.keys(parsed.results).length) {
      setLoadError(
        `Loaded ${known.length} of ${Object.keys(parsed.results).length} rows — the rest are workloads this page no longer has.`
      );
    }
  };

  // What a reader should do about a bad cell: it is a gpu.js bug, not a page
  // bug, and the issue is worth more than the observation.
  const trouble = useMemo(() => {
    const out = [];
    for (const [id, cells] of Object.entries(results || {})) {
      if (!cells || typeof cells !== 'object') continue;
      for (const [col, cell] of Object.entries(cells)) {
        if (cell && cell.wrong) out.push(`${id} · ${col} · WRONG`);
        else if (cell && cell.error) out.push(`${id} · ${col} · error`);
      }
    }
    return out;
  }, [results]);

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
      // Keep whatever columns did land; drop the row marker AND every cell that
      // was still a {running:true} placeholder. Those placeholders are pushed
      // by onCell before a column is measured, and normally get overwritten
      // when the worker reports back — but the worker was just terminated, so
      // nothing is coming. Left alone they animate forever on a stopped run.
      // Removing the key entirely renders a dash, which is the truth: that
      // cell was never measured.
      setLive(prev => {
        const row = prev[pending.workloadId];
        if (!row) return prev;
        const kept = {};
        for (const [col, cell] of Object.entries(row)) {
          if (col === '__running' || (cell && cell.running)) continue;
          kept[col] = cell;
        }
        return { ...prev, [pending.workloadId]: kept };
      });
      pending.resolve();
    }
  };

  return (
    <div className="bench-root" data-theme={theme}>
      <BenchNav />
      <div className="wrap">
        <p className="eyebrow">gpu.js</p>
        <h1>The Benchmark Gauntlet</h1>
        <p className="sub">
          <b>{workloads.length} GPGPU workloads</b> run the gauntlet on every backend gpu.js can
          reach, and against hand-written implementations with no gpu.js in them at all. The
          rightmost column is <b>plain JavaScript</b> — every speed-up on this page is measured
          against it. Nothing here is a claim we cannot make you watch: press run, and the
          numbers are yours, taken on your machine.
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
                {/* The library version is part of a run's identity, not a
                    footnote to it: two tables from the same machine on either
                    side of a gpu.js release are exactly what a reader wants to
                    compare, and they are indistinguishable by label alone. */}
                {allRuns.map(r => (
                  <option key={r.id} value={r.id}>
                    {`${r.label}${r.gpujs ? ` · gpu.js ${r.gpujs}` : ''}${r.fromFile ? ' (opened)' : ''}`}
                  </option>
                ))}
              </select>
            </label>
          )}
          {/* Only alongside the picker it feeds. In Live mode the button had
              nothing to do with what the page was showing, and offering it
              there implied opening a file was a way to start measuring.

              A label wrapping a hidden input, not a button calling .click() on
              one: this way the keyboard and the file picker behave the way the
              platform intends without any of it being reimplemented. */}
          {readOnly && (
            <label className="btn openrun">
              <input
                type="file"
                accept="application/json,.json"
                onChange={e => {
                  openRunFile(e.target.files && e.target.files[0]);
                  e.target.value = '';
                }}
              />
              Open a run…
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
              Showing {saved.fromFile ? 'a run opened from a file' : 'a saved run'} —{' '}
              <code>{saved.machine}</code>{saved.date ? `, ${saved.date}` : ''}, gpu.js{' '}
              {saved.gpujs}. Running is disabled so a stored result can never be
              half-overwritten by this machine.
            </span>
          </div>
        )}

        {loadError && (
          <div className="loaderr" role="alert">
            <span>{loadError}</span>
            <button type="button" onClick={() => setLoadError('')} aria-label="Dismiss">✕</button>
          </div>
        )}

        {/* Only when there is something to report. A standing invitation to file
            bugs on a clean table is noise; the same words next to an actual
            WRONG are a next step. */}
        {trouble.length > 0 && (
          <div className="warnbox inline" role="note">
            <h3>{trouble.length} cell{trouble.length === 1 ? '' : 's'} did not pass its check</h3>
            <p>
              A cell reading <b>WRONG</b> or <b>error</b> is a gpu.js bug, not a property of your
              machine — the same kernel produced the right answer on the other backends. These
              are worth reporting, and a report with the workload name and your adapter in it is
              worth several without:
            </p>
            <ul className="troublelist">
              {trouble.slice(0, 8).map(t => <li key={t}><code>{t}</code></li>)}
              {trouble.length > 8 && <li>…and {trouble.length - 8} more</li>}
            </ul>
            <p>
              <a href="https://github.com/gpujs/gpu.js/issues/new" target="_blank" rel="noreferrer">
                Raise an issue on gpu.js →
              </a>
            </p>
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

        {/* One continuous read: the instrument, then the two things that would
            be misread, then what the numbers say. It was five commentary cards
            over five legend cards, and the legend restated the lede — the
            baseline column, best-implementation-per-platform, every answer
            checked — while the commentary had already introduced the amber
            columns the legend went on to explain. Prose forces the order to be
            argued rather than tiled. Every figure below is a median over the
            recorded run and moves when the table is re-recorded. */}
        <div className="essay">
          <h2>How the gauntlet is run</h2>
          <p>
            Each column runs <b>the best implementation for its platform</b>, not one algorithm
            compiled six ways. Selecting the top 512 of a million numbers is a heap in plain
            JavaScript and a threshold bisection on a GPU, because that is what each is actually
            good at; a benchmark that forced the heap onto the GPU would be measuring a mistake
            nobody would make twice. What is held identical is the <em>problem</em> and the
            answer to it.
          </p>
          <p>
            With one exception that is worth stating rather than hiding: <b>the four gpu.js
            columns necessarily share a single kernel</b>. That is what gpu.js is — you write a
            function once and a backend is chosen for you — so those columns cannot each pick
            their platform's best algorithm, and the two hand-written columns can. On most rows
            that costs nothing. On a problem that wants <em>scatter</em> — a histogram bumping a
            shared counter, a compaction writing to an address it computes — it costs a great
            deal, because a gpu.js thread may only write its own output cell on <b>every</b>
            backend, CPU included. Those rows are not measuring a slow backend. They are pricing
            a programming model against a problem shaped the wrong way for it, which is a real
            thing to know before you choose one.
          </p>
          <p>
            Two of the columns are tinted amber, and they carry <b>no gpu.js at all</b>: one is
            WebGPU written by hand in WGSL, the other is the plain-JavaScript baseline. They are
            there to keep two different claims apart. “The GPU is fast” is a fact about your
            hardware; “gpu.js is fast” is a fact about this library. Only the amber columns can
            tell you which one a number belongs to.
          </p>
          <p>
            Every cell is a <b>median of at least three runs after two warm-ups</b>, because the
            first call compiles a shader and would time the compiler instead of the kernel. Runs
            are repeated until a cell has taken about a second, so a fast backend is measured
            over enough work to mean something. And every column is checksummed against the same
            plain-JavaScript oracle before its time is shown.
          </p>

          <div className="warnbox" role="note">
            <h3>Two things that look like results and are not</h3>
            <p>
              <b>N/A is about your machine, not the workload.</b> Every workload here runs on
              every backend. An N/A means the backend is not reachable from where you are
              reading — no WebGPU adapter, or a page served over plain http, which does not
              expose<code> navigator.gpu</code> at all. Hover the cell for the reason.
            </p>
            <p>
              <b>WRONG is not a slow result. It is a wrong one.</b> When a column's checksum
              misses the oracle by more than one part in ten thousand, the cell reports WRONG
              instead of a time. A benchmark that prints a fast number for the wrong answer is
              worse than no benchmark, so this page would rather show you a hole than a lie.
            </p>
          </div>

          <h2>Running it on a machine that is not this one</h2>
          <p>
            The most useful row in this table is the one from <em>your</em> hardware, and the
            hardware worth measuring is usually not the laptop you are reading on. There is a
            container for that: it builds the page, drives real Chromium against a real GPU, and
            hands back a saved run you can drop into the picker above.
          </p>
          <CmdBlock>{`docker build -t gpu-rocks-bench .
mkdir -p out
docker run --rm --gpus all -v "$PWD/out:/out" \\
  gpu-rocks-bench --label "RTX 4090 · Linux"`}</CmdBlock>
          <p>
            <b>Whether a container can see a GPU is entirely up to the host.</b> On Linux with
            NVIDIA it needs the container toolkit and <code>--gpus all</code>; on Intel or AMD,
            pass the render node with <code>--device /dev/dri</code>. On <b>Docker Desktop for
            macOS there is no GPU passthrough at all</b> — Docker runs Linux in a VM that cannot
            reach the Apple GPU, so run the recorder natively there instead
            (<code>node scripts/bench-record.mjs</code>) and skip the container entirely.
          </p>
          <p>
            Headless Chromium also does not use a GPU on Linux unless told to, and which flags
            work depends on the driver underneath. The image defaults to ANGLE over Vulkan, which
            is the combination that works on NVIDIA; override{' '}
            <code>CHROME_FLAGS</code> for anything else.
          </p>

          <div className="warnbox" role="note">
            <h3>If it refuses to save, believe it</h3>
            <p>
              Before spending fifteen minutes the recorder reads the WebGPU adapter and the WebGL
              renderer, and <b>refuses to write a run if either is software</b>. A container with
              no GPU reports <code>google swiftshader</code> and stops there. That is not the
              image failing — it is the image declining to hand you a CPU wearing a GPU's label,
              which is the one number this page cannot afford to publish. Full notes in{' '}
              <code>docker/README.md</code>.
            </p>
          </div>

          <p>
            <b>If you run it, send the result back.</b> A saved run from hardware nobody here
            owns is the most useful thing this page can be given — a different vendor, an older
            card, a phone. Open a pull request against{' '}
            <a href="https://github.com/gpujs/gpu.rocks" target="_blank" rel="noreferrer">gpu.rocks</a>{' '}
            with the JSON dropped into <code>src/Bench/saved/</code>, and it joins the picker
            above. And if any cell comes back <b>WRONG</b> or <b>error</b>, that is a gpu.js bug
            worth an{' '}
            <a href="https://github.com/gpujs/gpu.js/issues/new" target="_blank" rel="noreferrer">issue</a>{' '}
            rather than a shrug: the same kernel got the right answer on the other backends, so
            the disagreement is real and reproducible.
          </p>

          <h2>What the shape of the table says</h2>
          <p>
            <b>The GPU's win is enormous, and enormously uneven.</b> Across the gauntlet the best
            GPU column beats plain JavaScript by a median of <b>~54×</b> — but the spread runs
            from <b>1073×</b> down to <b>0.58×</b>. Three orders of magnitude separate the best
            row from the worst. Any single number you have been quoted for “GPU speed-up” was a
            choice of workload, and this table is the argument for asking which one.
          </p>
          <p>
            <b>Convenience costs about half.</b> Where both run, gpu.js on WebGPU takes a median
            of <b>2.1× longer</b> than the hand-written WebGPU doing the same work. That is the
            price of writing a kernel as a JavaScript function instead of WGSL. Whether half the
            speed is worth not writing shader code is a real decision — the point of measuring it
            is that you get to make it deliberately.
          </p>
          <p>
            <b>Newer is not automatically faster.</b> gpu.js on WebGL2 beats gpu.js on WebGPU in{' '}
            <b>7 of {workloads.length} rows</b>, the transforms and the small image kernels among
            them. The WebGPU backend is younger, and per-dispatch overhead still shows on work
            that is many small passes rather than one large one.
          </p>
          <p>
            <b>WebAssembly is a floor, not a ceiling.</b> It answers a different question from the
            GPU columns: not “how fast can this get” but “what is left when there is{' '}
            <em>no GPU at all</em>”. The answer is <b>single-digit multiples</b> of plain
            JavaScript — up to <b>7×</b> here, against the GPU's hundreds. That is the right shape
            for it. A compiled scalar loop with SIMD is still <em>one core doing one thing at a
            time</em>, and the gap to a GPU is not a tuning gap, it is thousands of lanes.
          </p>
          <p>
            <b>The CPU backend is slower than the JavaScript it replaces.</b> gpu.js's CPU mode
            runs at a median <b>0.48×</b> of plain JS and is faster on only{' '}
            <b>5 of {workloads.length}</b> rows. It exists so that a kernel written once still{' '}
            <em>runs</em> where there is nothing to run it on — not so that it runs fast. Read it
            as the fallback keeping its promise, not as a result.
          </p>
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
