import React, {
  useCallback,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
} from 'react';
import { Navigate, useNavigate, useParams } from 'react-router';
import CodeMirror from '@uiw/react-codemirror';
import { javascript } from '@codemirror/lang-javascript';
import { EditorView, keymap } from '@codemirror/view';
import { indentUnit } from '@codemirror/language';
import { indentWithTab } from '@codemirror/commands';
import { Prec } from '@codemirror/state';
import LearnNav from '../components/LearnNav';
import TaskDots from '../components/TaskDots';
import { useTheme } from '../ThemeContext';
import { tracks, modules, getTask } from '../content/index';
import { getFigures } from '../content/figures/index';
import { runUserCode, runTests, snapshotCanvas, utils, warmUpSandbox } from '../engine/runner';
import { runBenchmark } from '../engine/benchmark';
import {
  getSavedCode,
  saveCode,
  clearCode,
  markTaskDone,
  isTaskDone,
  moduleProgress,
} from '../engine/storage';
import { lightEditorTheme, darkEditorTheme } from './editorThemes';
import { moduleTaskMeta } from '../../routeMeta';
import { setPageMeta } from '../../pageMeta';
import { learnCompletions } from './completion/completions';
import { signatureHelp } from './completion/signatureHelp';
import ConsolePane from './ConsolePane';
import CompletionModal from './CompletionModal';
import { highlightCodeBlocks } from './highlightCode';

// ---- helpers ---------------------------------------------------------------

// render() snapshots at log time (see engine/sandbox.js); this is the fallback
// for a canvas whose pixels couldn't be captured then — retried right after the
// run, in the same microtask chain, before the browser can composite + clear
// it. Only the main-thread fallback path can ever hit it: a worker run has no
// live canvas to retry, it ships an ImageBitmap that runner.js has already
// turned into a data URL.
function decorateLog(log) {
  if (log.type === 'canvas' && log.canvas && !log.snapshot) {
    const snapshot = snapshotCanvas(log.canvas);
    if (snapshot) return { ...log, snapshot };
  }
  return log;
}

function fmtMs(ms) {
  if (typeof ms !== 'number' || !Number.isFinite(ms)) return '—';
  return ms >= 100 ? ms.toFixed(0) : ms.toFixed(1);
}

function fmtRatio(ratio) {
  if (!Number.isFinite(ratio)) return '∞';
  return ratio >= 10 ? ratio.toFixed(0) : ratio.toFixed(1);
}

// ---- benchmark chip --------------------------------------------------------

// below Chrome's coarsened performance.now() resolution a median is noise
const TIMER_FLOOR_MS = 0.05;

function BenchChip({ bench }) {
  if (!bench) return null;
  if (bench.gpuUnavailable) {
    return (
      <span className="bench-chip note">
        GPU unavailable here
        {typeof bench.cpuMs === 'number' ? ` · CPU ${fmtMs(bench.cpuMs)} ms` : ''}
      </span>
    );
  }
  if (bench.gpuFailed) {
    return (
      <span className="bench-chip note">
        code failed in GPU mode
        {bench.error ? ` · ${bench.error.message}` : ''}
        {typeof bench.cpuMs === 'number' ? ` · CPU ${fmtMs(bench.cpuMs)} ms` : ''}
      </span>
    );
  }
  if (bench.error) {
    return <span className="bench-chip note">benchmark failed · {bench.error.message}</span>;
  }
  if (bench.gpuRanOnCpu) {
    return (
      <span className="bench-chip note">
        no GPU comparison — gpu.js ran this kernel on the CPU in both modes ·{' '}
        {fmtMs(bench.cpuMs)} ms vs {fmtMs(bench.gpuMs)} ms
      </span>
    );
  }
  if (bench.cpuMs < TIMER_FLOOR_MS || bench.gpuMs < TIMER_FLOOR_MS) {
    return (
      <span className="bench-chip note">
        too fast to compare · CPU {fmtMs(bench.cpuMs)} ms vs GPU {fmtMs(bench.gpuMs)} ms
      </span>
    );
  }
  if (bench.fasterOn === 'cpu') {
    const ratio = bench.cpuMs > 0 ? bench.gpuMs / bench.cpuMs : Infinity;
    return (
      <span className="bench-chip cpu">
        CPU {fmtRatio(ratio)}× faster{' '}
        <span className="vs">· {fmtMs(bench.cpuMs)} ms vs {fmtMs(bench.gpuMs)} ms</span>
      </span>
    );
  }
  return (
    <span className="bench-chip">
      GPU {fmtRatio(bench.ratio)}× faster{' '}
      <span className="vs">· {fmtMs(bench.gpuMs)} ms vs {fmtMs(bench.cpuMs)} ms</span>
    </span>
  );
}

// screen-reader summary of a benchmark result (visually the chip shows it)
function benchStatusMessage(b) {
  if (!b) return 'Benchmark finished.';
  if (b.gpuUnavailable) return 'Benchmark finished — GPU unavailable here.';
  if (b.gpuFailed) {
    return `Benchmark finished — code failed in GPU mode${b.error ? `: ${b.error.message}` : ''}`;
  }
  if (b.error) return `Benchmark failed — ${b.error.message}`;
  if (b.gpuRanOnCpu) return 'Benchmark finished — no GPU comparison, this kernel ran on the CPU backend.';
  if (b.cpuMs < TIMER_FLOOR_MS || b.gpuMs < TIMER_FLOOR_MS) {
    return 'Benchmark finished — both runs too fast to compare.';
  }
  if (b.fasterOn === 'cpu') {
    const ratio = b.cpuMs > 0 ? b.gpuMs / b.cpuMs : Infinity;
    return `Benchmark finished — CPU ${fmtRatio(ratio)} times faster.`;
  }
  return `Benchmark finished — GPU ${fmtRatio(b.ratio)} times faster.`;
}

// ---- tests panel -----------------------------------------------------------

function TestRow({ name, result }) {
  const status = result ? (result.passed ? 'PASS' : 'FAIL') : '—';
  const stClass = result ? (result.passed ? 'pass' : 'fail') : 'lock';
  return (
    <div className="test">
      <span className={`st ${stClass}`}>{status}</span>
      <span className="desc">
        {/* trusted HTML: test names are authored in-repo and may contain <code> */}
        <span dangerouslySetInnerHTML={{ __html: name }} />
        {result && !result.passed && result.error && (
          <span className="fail-msg">{result.error}</span>
        )}
      </span>
      <span className="ms">{result ? `${fmtMs(result.ms)} ms` : ''}</span>
    </div>
  );
}

function TestsPanel({ task, report, active }) {
  const publicTests = task.publicTests || [];
  const privateTests = task.privateTests || [];
  const pub = report ? report.results.filter(r => !r.private) : null;
  const priv = report ? report.results.filter(r => r.private) : null;
  return (
    <div
      className={`bpanel tests${active ? ' on' : ''}`}
      role="tabpanel"
      id="bpanel-tests"
      aria-labelledby="btab-tests"
    >
      <div className="testlist">
        {publicTests.map((t, i) => (
          <TestRow key={i} name={t.name} result={pub ? pub[i] : null} />
        ))}
        {privateTests.length > 0 && (
          <div className="privnote">
            🔒 {privateTests.length} private test{privateTests.length === 1 ? '' : 's'} — hidden
            inputs, same rules
          </div>
        )}
        {priv &&
          priv.map((r, i) => (
            <div className="test" key={`p${i}`}>
              <span className={`st ${r.passed ? 'pass' : 'fail'}`}>
                {r.passed ? 'PASS' : 'FAIL'}
              </span>
              <span className="desc">private test #{i + 1}</span>
              <span className="ms">{fmtMs(r.ms)} ms</span>
            </div>
          ))}
      </div>
      {report && report.allPassed && report.total > 0 && (
        <div className="testsum">
          ✓ {report.passed} of {report.total} passed — Next task unlocked
        </div>
      )}
    </div>
  );
}

// ---- brief pane ------------------------------------------------------------

// All HTML fragments here are trusted: course content authored in this repo.
// That includes the figure SVGs — in-repo markup, injected verbatim.
function Figures({ figures }) {
  return figures.map((fig, i) => (
    <figure className="diagram" key={i}>
      <div dangerouslySetInnerHTML={{ __html: fig.svg }} />
      <figcaption>{fig.caption}</figcaption>
    </figure>
  ));
}

function BriefPane({ moduleId, task, taskNum, total }) {
  const figures = getFigures(moduleId, taskNum);
  const briefRef = useRef(null);
  // Static syntax highlighting for authored <pre><code> blocks. Layout effect
  // so the spans exist before paint (no flash of unhighlighted code, no
  // post-paint layout thrash); idempotent, re-runs when the task changes.
  useLayoutEffect(() => {
    highlightCodeBlocks(briefRef.current);
  }, [task]);
  return (
    <aside className="brief" ref={briefRef}>
      <p className="eyebrow">Task {taskNum} of {total}</p>
      <h2>{task.title}</h2>
      <div dangerouslySetInnerHTML={{ __html: task.intro || '' }} />
      <Figures figures={figures.filter(f => f.placement === 'intro')} />
      {task.goal && (
        <div className="goal" dangerouslySetInnerHTML={{ __html: task.goal }} />
      )}
      <Figures figures={figures.filter(f => f.placement === 'goal')} />
      {Array.isArray(task.requirements) && task.requirements.length > 0 && (
        <ul className="reqs">
          {task.requirements.map((req, i) => (
            <li key={i} dangerouslySetInnerHTML={{ __html: req }} />
          ))}
        </ul>
      )}
      {(task.hints || []).map((hint, i) => (
        <details key={i}>
          <summary>{hint.title}</summary>
          <div dangerouslySetInnerHTML={{ __html: hint.body }} />
        </details>
      ))}
      {task.transfer && (
        <div className="transfer">
          <b>Same idea elsewhere</b>
          <span dangerouslySetInnerHTML={{ __html: task.transfer }} />
        </div>
      )}
    </aside>
  );
}

// ---- workspace (keyed by taskId so state resets per task) ------------------

function TaskWorkspace({ module, task, taskNum, taskIndex, taskId, total }) {
  const navigate = useNavigate();
  const { theme } = useTheme();

  // The editor is intentionally UNcontrolled after mount (stable `value` prop):
  // @uiw/react-codemirror's controlled-value sync can arm a stale forceUpdate
  // while its "typing latch" is active (its 1 ms-interval scheduler is clamped
  // /throttled by browsers), which later REPLACES the doc with an old value —
  // deleting the user's last keystrokes. TaskWorkspace remounts per task
  // (key={taskId}), so the initial doc is all the value prop ever needs to
  // carry; edits flow through onChange → codeRef, resets dispatch to the view.
  const [initialCode] = useState(() => {
    const saved = getSavedCode(taskId);
    return saved != null ? saved : task.starterCode;
  });
  const [mode, setMode] = useState('auto');
  const [logs, setLogs] = useState([]);
  const [report, setReport] = useState(null);
  const [bench, setBench] = useState(null);
  const [running, setRunning] = useState(false);
  const [benching, setBenching] = useState(false);
  const [tab, setTab] = useState('console');
  const [done, setDone] = useState(() => isTaskDone(taskId));
  const [fullscreen, setFullscreen] = useState(false);
  const [progressTick, setProgressTick] = useState(0);
  const [status, setStatus] = useState(''); // announced via the role=status region
  const [celebration, setCelebration] = useState(null); // null | { kind: 'module' | 'track' }

  const codeRef = useRef(initialCode);
  const viewRef = useRef(null);
  const modeRef = useRef(mode);
  modeRef.current = mode;
  const busyRef = useRef(false);
  const saveTimer = useRef(null);
  const nextBtnRef = useRef(null);
  const consoleTabRef = useRef(null);
  const testsTabRef = useRef(null);

  // title/description/canonical for this task — same values the prerender
  // bakes into the static HTML for this route
  useEffect(() => {
    setPageMeta(moduleTaskMeta(module, task, taskNum));
  }, [module, task, taskNum]);

  // Spawn the sandbox worker (gpu.js + the content registry evaluate in it)
  // while the learner reads the brief, so the first ▶ Run does not pay for it.
  useEffect(() => {
    warmUpSandbox();
  }, []);

  const track = tracks.find(t => t.number === module.track);
  const progress = useMemo(
    () => moduleProgress(module),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [module, progressTick, done]
  );

  // debounced (~500 ms) save of the editor content
  const handleChange = useCallback(
    value => {
      codeRef.current = value;
      clearTimeout(saveTimer.current);
      saveTimer.current = setTimeout(() => {
        saveTimer.current = null;
        saveCode(taskId, value);
      }, 500);
    },
    [taskId]
  );
  // on unmount, FLUSH a pending save (don't discard it) — navigating away
  // within 500 ms of the last keystroke must not lose the user's edits
  useEffect(
    () => () => {
      if (saveTimer.current != null) {
        clearTimeout(saveTimer.current);
        saveTimer.current = null;
        saveCode(taskId, codeRef.current);
      }
    },
    [taskId]
  );

  const handleRun = useCallback(async () => {
    if (busyRef.current) return;
    busyRef.current = true;
    setRunning(true);
    try {
      const result = await runUserCode(codeRef.current, { mode: modeRef.current, task });
      // snapshot canvases immediately, before the browser composites the frame
      setLogs(result.logs.map(decorateLog));
      const rep = await runTests(task, result);
      setReport(rep);
      if (rep.total > 0 && rep.allPassed) {
        markTaskDone(taskId);
        setDone(true);
        setProgressTick(t => t + 1);
      }
      setStatus(
        !result.ok
          ? `Run failed — ${result.error.message}`
          : rep.total > 0
            ? rep.allPassed
              ? `Run complete — all ${rep.total} tests passed. Next task unlocked.`
              : `Run complete — ${rep.passed} of ${rep.total} tests passed, ${rep.total - rep.passed} failed.`
            : 'Run complete.'
      );
    } finally {
      setRunning(false);
      busyRef.current = false;
    }
  }, [task, taskId]);

  const handleBenchmark = useCallback(async () => {
    if (busyRef.current) return;
    busyRef.current = true;
    setBenching(true);
    try {
      const result = await runBenchmark(codeRef.current, task);
      setBench(result);
      setStatus(benchStatusMessage(result));
    } finally {
      setBenching(false);
      busyRef.current = false;
    }
  }, [task]);

  const handleReset = useCallback(() => {
    if (!window.confirm('Reset code to the starter? Your changes to this task will be lost.')) {
      return;
    }
    clearTimeout(saveTimer.current);
    saveTimer.current = null; // an explicit Reset must not be flushed back on unmount
    clearCode(taskId);
    codeRef.current = task.starterCode;
    const view = viewRef.current;
    if (view) {
      view.dispatch({
        changes: { from: 0, to: view.state.doc.length, insert: task.starterCode },
      });
    }
    setReport(null);
    setBench(null);
  }, [task, taskId]);

  // next module in the global sorted order; when it belongs to another track
  // (or doesn't exist) this module is the last of its track
  const nextModule = useMemo(() => {
    const idx = modules.findIndex(m => m.id === module.id);
    return idx >= 0 ? modules[idx + 1] || null : null;
  }, [module]);
  const lastInTrack = !nextModule || nextModule.track !== module.track;

  // Prev walks back within the module only — it never crosses into the
  // previous module (Next does, but backwards that would drop you at the far
  // end of work you may not have started). At task 1 the way out is the
  // course list, which is its own button.
  const prevEnabled = taskNum > 1;

  const handlePrev = useCallback(() => {
    if (taskNum > 1) navigate(`/learn/${module.id}/${taskNum - 1}`);
  }, [navigate, module, taskNum]);

  const handleAllModules = useCallback(() => navigate('/learn'), [navigate]);

  const handleNext = useCallback(() => {
    if (taskNum < total) {
      navigate(`/learn/${module.id}/${taskNum + 1}`);
      return;
    }
    // last task of the module AND every task done (including this one):
    // celebrate instead of navigating
    if (moduleProgress(module).done === total) {
      const kind = lastInTrack ? 'track' : 'module';
      setCelebration({ kind });
      setStatus(kind === 'track' ? 'Track complete' : 'Module complete');
      return;
    }
    // earlier tasks were skipped — keep the plain navigation behavior
    navigate(nextModule ? `/learn/${nextModule.id}/1` : '/learn');
  }, [navigate, module, taskNum, total, nextModule, lastInTrack]);

  // Esc / backdrop close: stay on the task, hand focus back to the Next button
  const closeCelebration = useCallback(() => {
    setCelebration(null);
    if (nextBtnRef.current) nextBtnRef.current.focus();
  }, []);

  // ⌘/Ctrl + Enter anywhere on the page runs the code. The CodeMirror keymap
  // below preventDefaults first, so check defaultPrevented to avoid a double run.
  useEffect(() => {
    const onKey = e => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'Enter' && !e.defaultPrevented) {
        e.preventDefault();
        handleRun();
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [handleRun]);

  // completion needs the task's input names (they are injected as globals)
  const inputNames = useMemo(() => {
    try {
      if (task && typeof task.inputs === 'function') {
        return Object.keys(task.inputs(utils) || {});
      }
    } catch {
      /* inputs that throw at completion time just aren't offered */
    }
    return [];
  }, [task]);

  const extensions = useMemo(
    () => [
      javascript(),
      indentUnit.of('  '),
      keymap.of([indentWithTab]),
      Prec.highest(
        keymap.of([
          {
            key: 'Mod-Enter',
            run: () => {
              handleRun();
              return true;
            },
          },
        ])
      ),
      ...learnCompletions(inputNames),
      ...signatureHelp(),
      EditorView.contentAttributes.of({ 'aria-label': 'kernel.js code editor' }),
    ],
    [handleRun, inputNames]
  );

  // ARIA tabs pattern: arrow keys move + activate within the tablist
  const handleTablistKey = useCallback(
    e => {
      if (e.key !== 'ArrowRight' && e.key !== 'ArrowLeft') return;
      e.preventDefault();
      const next = tab === 'console' ? 'tests' : 'console';
      setTab(next);
      const ref = next === 'console' ? consoleTabRef : testsTabRef;
      if (ref.current) ref.current.focus();
    },
    [tab]
  );

  const busy = running || benching;
  const nextEnabled = done || Boolean(report && report.total > 0 && report.allPassed);
  const passedNow = Boolean(report && report.total > 0 && report.allPassed);
  const testBadge = report ? `${report.passed}/${report.total}` : null;

  return (
    <div className="taskscreen">
      <LearnNav />

      <div className="crumbbar">
        <span>Track {module.track}{track ? ` · ${track.title}` : ''}</span>
        <span className="sep">/</span>
        <b>Module {String(module.id).replace('-', '.')} — {module.title}</b>
        <span className="sep">/</span>
        <span className="tcount">Task {taskNum} of {total}</span>
        <TaskDots total={total} doneCount={progress.done} currentIndex={taskIndex} />
      </div>

      <div className="toolbar">
        <button type="button" className="tb-run" onClick={handleRun} disabled={busy}>
          {running ? '… Running' : '▶ Run'}
        </button>
        <label className="tb-select">
          <span className="lbl">Mode</span>
          <select
            value={mode}
            onChange={e => setMode(e.target.value)}
            aria-label="Execution mode"
          >
            <option value="auto">Auto</option>
            <option value="cpu">CPU</option>
            <option value="gpu">GPU</option>
          </select>
          <span aria-hidden="true">▾</span>
        </label>
        <button type="button" className="tb-btn" onClick={handleBenchmark} disabled={busy}>
          {benching ? '⏱ Benchmarking…' : '⏱ Benchmark'}
        </button>
        <BenchChip bench={bench} />
        <div className="tb-right">
          {passedNow && <span className="pass-note">✓ All tests passed</span>}
          <button type="button" className="tb-btn" onClick={handleReset}>
            Reset code
          </button>
          {/* same wording as the completion modal's way out, so leaving a
              module reads the same whether you finished it or not */}
          <button type="button" className="tb-btn" onClick={handleAllModules}>
            Exit module
          </button>
          <button
            type="button"
            className="tb-btn"
            onClick={handlePrev}
            disabled={!prevEnabled}
          >
            ← Prev task
          </button>
          <button
            type="button"
            className="tb-next"
            ref={nextBtnRef}
            onClick={handleNext}
            disabled={!nextEnabled}
          >
            Next task →
          </button>
        </div>
      </div>

      {/* visually hidden — announces run/benchmark outcomes to screen readers */}
      <div className="sr-status" role="status" aria-live="polite">
        {status}
      </div>

      <div className="workspace">
        <BriefPane moduleId={module.id} task={task} taskNum={taskNum} total={total} />

        <div className="editpane">
          <div className={fullscreen ? 'editor editor-fullscreen' : 'editor'}>
            <div className="edtabs">
              <span className="edtab on">kernel.js</span>
              {/* WCAG 2.1.2: Tab indents in the editor, so advertise the way out */}
              <span className="ed-hint">Esc then Tab exits the editor</span>
              <button
                type="button"
                className="ed-fullbtn"
                onClick={() => setFullscreen(f => !f)}
                aria-label={fullscreen ? 'Exit full screen editor' : 'Full screen editor'}
              >
                {fullscreen ? '✕ Close' : '⛶ Full screen'}
              </button>
            </div>
            <CodeMirror
              className="cm-host"
              value={initialCode}
              onChange={handleChange}
              onCreateEditor={view => {
                viewRef.current = view;
                // exposes the EditorView for the headless verification harness
                window.__learnEditorView = view;
              }}
              theme={theme === 'dark' ? darkEditorTheme : lightEditorTheme}
              extensions={extensions}
              basicSetup={{
                foldGutter: false,
                autocompletion: false,
                highlightSelectionMatches: false,
              }}
            />
          </div>

          <div className="bottompane">
            <div
              className="btabs"
              role="tablist"
              aria-label="Output panes"
              onKeyDown={handleTablistKey}
            >
              <button
                type="button"
                role="tab"
                id="btab-console"
                ref={consoleTabRef}
                className={`btab${tab === 'console' ? ' on' : ''}`}
                aria-selected={tab === 'console'}
                aria-controls="bpanel-console"
                tabIndex={tab === 'console' ? 0 : -1}
                onClick={() => setTab('console')}
              >
                Console
              </button>
              <button
                type="button"
                role="tab"
                id="btab-tests"
                ref={testsTabRef}
                className={`btab${tab === 'tests' ? ' on' : ''}`}
                aria-selected={tab === 'tests'}
                aria-controls="bpanel-tests"
                tabIndex={tab === 'tests' ? 0 : -1}
                onClick={() => setTab('tests')}
              >
                Tests {testBadge && <span className="cnt">{testBadge}</span>}
              </button>
              <button type="button" className="clearbtn" onClick={() => setLogs([])}>
                clear
              </button>
            </div>
            <ConsolePane logs={logs} active={tab === 'console'} />
            <TestsPanel task={task} report={report} active={tab === 'tests'} />
          </div>
        </div>
      </div>

      {celebration && (
        <CompletionModal
          kind={celebration.kind}
          module={module}
          track={track}
          totalTasks={total}
          courseEnd={
            celebration.kind === 'track' &&
            Boolean(track) &&
            track.number === tracks[tracks.length - 1].number
          }
          onClose={closeCelebration}
          onEnd={() => navigate('/learn')}
          onNextModule={() => navigate(`/learn/${nextModule.id}/1`)}
        />
      )}
    </div>
  );
}

// ---- route component -------------------------------------------------------

function TaskPage() {
  const { moduleId, taskNum } = useParams();
  const found = getTask(moduleId, taskNum);
  if (!found) return <Navigate to="/learn" replace />;
  // key by taskId so all workspace state (code, logs, tests…) resets per task
  return <TaskWorkspace key={found.taskId} {...found} />;
}

export default TaskPage;
