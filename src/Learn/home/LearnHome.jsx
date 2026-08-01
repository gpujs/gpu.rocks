import React, { useEffect, useRef } from 'react';
import { Link } from 'react-router';
import LearnNav from '../components/LearnNav';
import KernelGrid from '../components/KernelGrid';
import TaskDots from '../components/TaskDots';
import { moduleNumber, orphanModules, parseTaskKey, tracks } from '../content/index.js';
import { FEEDBACK_URL } from '../feedback';
import { getProgress, moduleProgress } from '../engine/storage.js';
import { getFigures } from '../content/figures/index.js';
import moduleRenders from '../content/moduleRenders.js';
import { learnHomeMeta } from '../../routeMeta';
import { setPageMeta } from '../../pageMeta';

// /learn landing page: hero + highlights + track/module catalogue.
// Layout, spacing and copy come from the approved mockup; module/track data
// comes from content/index.js and per-module progress from engine/storage.js.

// Defensive wrapper around engine/storage.js moduleProgress() — the engine
// module is written by another agent; fall back to an untouched-module shape
// if it is absent or throws (contract: { done, total, currentIndex, state }).
function progressOf(module) {
  const total = Array.isArray(module.tasks) ? module.tasks.length : 0;
  const fallback = { done: 0, total, currentIndex: 0, state: 'todo' };
  try {
    if (typeof moduleProgress === 'function') {
      const p = moduleProgress(module);
      if (p && typeof p === 'object') return { ...fallback, ...p };
    }
  } catch (e) {
    // private browsing / storage unavailable — treat as fresh
  }
  return fallback;
}

// Work the learner did at an EARLIER version of this module.
//
// Bumping a module's version starts it unsolved but destroys nothing: the old
// keys stay in localStorage under the old version (`<uuid>:v<n>:<taskSlug>`).
// Without this lookup such a card would render as never-started, which is a
// lie — the point of the note below is to say so plainly.
//
// We cannot know how many tasks the older version HAD (that content is gone),
// so "they finished it" is inferred from an earlier version's completed count
// reaching the module's CURRENT task total. A miss degrades to the softer
// "worked on" wording, never to silence.
//
// FRICTION: engine/storage.js exposes no archived-version rollup, so this
// reads the raw progress object and parses keys itself. If storage grows a
// proper accessor, this should collapse into a call to it.
function priorVersionOf(module) {
  const current = Number(module.version) || 1;
  const total = Array.isArray(module.tasks) ? module.tasks.length : 0;
  let progress;
  try {
    progress = getProgress();
  } catch (e) {
    return null; // storage unavailable — nothing to say
  }
  const doneByVersion = new Map();
  Object.keys(progress || {}).forEach(key => {
    const entry = progress[key];
    if (!entry || !entry.done) return;
    const parsed = parseTaskKey(key);
    if (!parsed || parsed.uuid !== module.uuid || !(parsed.version < current)) return;
    doneByVersion.set(parsed.version, (doneByVersion.get(parsed.version) || 0) + 1);
  });
  if (doneByVersion.size === 0) return null;

  // "Completed" wins over "touched": finishing v1 and then abandoning v2 still
  // means they finished this module once, which is the more useful thing to
  // say. The version NUMBER is deliberately not surfaced — it is storage
  // plumbing, and the learner never chose it.
  const completed = [...doneByVersion.keys()].some(
    v => total > 0 && doneByVersion.get(v) >= total
  );
  return { completed };
}

function stateLabel(progress, prior) {
  if (progress.state === 'done') return 'Completed';
  if (progress.state === 'now') return `Continue · ${progress.done}/${progress.total}`;
  return prior ? 'Start again' : 'Start';
}

function ModuleCard({ module }) {
  const progress = progressOf(module);
  const taskCount = Array.isArray(module.tasks) ? module.tasks.length : 0;
  const isCurrent = progress.state === 'now';
  // Only surfaced when the CURRENT version is untouched — that is exactly when
  // the card would otherwise read as never-started. Once they restart, the
  // dots and "Continue" already tell the true story and the note is noise.
  const prior = progress.done === 0 ? priorVersionOf(module) : null;
  return (
    <Link
      to={module.url}
      className={isCurrent ? 'module current' : 'module'}
    >
      <ModuleThumb module={module} />
      <span className="mno">
        {/* orphan modules ("Others") have no number — they are unordered */}
        {moduleNumber(module) ? `MODULE ${moduleNumber(module)} · ` : ''}
        {taskCount} TASK{taskCount === 1 ? '' : 'S'}
      </span>
      <h3>{module.title}</h3>
      {/* blurbs are trusted course copy authored in-repo; some contain <code> */}
      <p dangerouslySetInnerHTML={{ __html: module.blurb || '' }} />
      {prior && (
        <span className="mstale">
          Updated since you {prior.completed ? 'completed' : 'last worked on'} it — your
          earlier work is kept.
        </span>
      )}
      <div className="foot">
        <TaskDots
          total={progress.total}
          doneCount={progress.done}
          currentIndex={isCurrent ? progress.currentIndex : -1}
        />
        <span className={`mstate ${progress.state}${prior ? ' again' : ''}`}>
          {stateLabel(progress, prior)}
        </span>
      </div>
    </Link>
  );
}

// Card art. A module that PAINTS something gets its real output — captured by
// scripts/capture-module-renders.mjs from the actual solution running in the
// actual sandbox — so the thumbnail is the thing itself rather than a drawing
// of it. Everything else falls back to the module's own opening figure, art a
// human already drew in the course's visual language.
//
// Both go in a 16:9 box: module output is square (128x128 grids) and figures
// are about 2.5:1, so without one fixed ratio the catalogue reads as a jumble.
function ModuleThumb({ module }) {
  if (moduleRenders.has(module.slug)) {
    // ?v=<content hash>: the PNG lives at a stable path but its bytes change
    // whenever the art is re-captured, and both the CDN and the browser cache
    // it for four hours. Without this a re-capture keeps serving the old
    // picture — and no cache purge can reach a phone that already has it.
    return (
      <span className="mart render">
        <img
          src={`/img/modules/${module.slug}.png?v=${moduleRenders.get(module.slug)}`}
          alt=""
          loading="lazy"
          aria-hidden="true"
        />
      </span>
    );
  }
  // the first task that has a figure — its opening idea, not its finale
  for (const task of module.tasks || []) {
    const figures = getFigures(module, task);
    if (figures.length) {
      return (
        <span
          className="mart figure"
          aria-hidden="true"
          // trusted in-repo markup, same as the task page inlines
          dangerouslySetInnerHTML={{ __html: figures[0].svg }}
        />
      );
    }
  }
  return <span className="mart blank" aria-hidden="true" />;
}

function Track({ track }) {
  const number = track.number != null ? track.number : track.id;
  const modules = Array.isArray(track.modules) ? track.modules : [];
  return (
    <div className="track">
      <div className="track-head">
        <span className="tno">TRACK {number}</span>
        <h2>{track.title}</h2>
        <span className="tagline">{track.tagline}</span>
      </div>
      <div className="modules">
        {/* "MODULE 1.2" comes from moduleNumber(), i.e. the module's POSITION
            in this track's ordered uuid list — never from its id (a uuid). */}
        {modules.map(module => <ModuleCard key={module.uuid} module={module} />)}
      </div>
    </div>
  );
}

// Modules in no track. Deliberately the same card language as a track — they
// are the same kind of thing to work through — but a different promise in the
// head: no TRACK n, no sequence, and (see CompletionModal) no next module when
// you finish one. Order is by title, which content/index.js already applies.
//
// Renders NOTHING when every module belongs to a track: an empty "Others"
// heading would advertise a category that isn't there.
function Others({ modules }) {
  if (!Array.isArray(modules) || modules.length === 0) return null;
  return (
    <div className="track others">
      <div className="track-head">
        <span className="tno">OTHERS</span>
        <h2>Standalone modules</h2>
        <span className="tagline">Not part of a track — take them in any order</span>
      </div>
      <div className="modules">
        {modules.map(module => <ModuleCard key={module.uuid} module={module} />)}
      </div>
    </div>
  );
}

function LearnHome() {
  const tracksRef = useRef(null);
  const trackList = Array.isArray(tracks) ? tracks : [];
  // "Start GPGPU 101" opens the first module of the first track — never a
  // hardcoded url, so reordering or renaming content cannot strand it.
  const firstModule = trackList.find(t => t.modules && t.modules.length);
  const firstModuleUrl = firstModule ? firstModule.modules[0].url : '/learn';

  useEffect(() => {
    setPageMeta(learnHomeMeta());
  }, []);

  const scrollToTracks = () => {
    const el = tracksRef.current;
    if (!el) return;
    let reduced = false;
    try {
      reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    } catch (e) {
      reduced = false;
    }
    el.scrollIntoView({ behavior: reduced ? 'auto' : 'smooth', block: 'start' });
  };

  return (
    <div className="screen">
      <LearnNav />

      <section className="hero">
        <div>
          <p className="eyebrow">gpu.js · interactive course</p>
          <h1>
            Stop writing loops.<br />
            <span className="accent">Launch threads.</span>
          </h1>
          <p className="sub">
            A hands-on course in <strong>GPGPU</strong> — general-purpose computing on graphics
            hardware — built on <strong>gpu.js</strong> and powered by <strong>WebGPU</strong>.
            Write real kernels, run them on <strong>your own GPU</strong>, and learn the mental
            model behind every parallel computing platform.
          </p>
          <div className="cta-row">
            <Link to={firstModuleUrl} className="btn btn-primary">Start GPGPU 101</Link>
            <button type="button" className="btn btn-ghost" onClick={scrollToTracks}>
              Browse all modules ↓
            </button>
          </div>
        </div>
        <KernelGrid />
      </section>

      <section className="highlights">
        <div className="hl">
          <h3>Real GPGPU concepts</h3>
          <p>
            Kernels, threads, memory layout, reductions, pipelines — learned by writing and
            running them with gpu.js, not by reading slides.
          </p>
        </div>
        <div className="hl">
          <h3>Zero installation</h3>
          <p>
            Everything runs in your browser, on your actual graphics card. No toolchain,
            no drivers, no signup. Open a module and start.
          </p>
        </div>
        <div className="hl">
          <h3>Skills that transfer</h3>
          <p>The concepts here carry straight over to every serious GPU platform.</p>
          <div className="chips">
            <span className="chip">WebGPU</span>
            <span className="chip">CUDA</span>
            <span className="chip">ROCm</span>
            <span className="chip">Metal</span>
          </div>
        </div>
      </section>

      <section className="tracks" ref={tracksRef}>
        {trackList.map(track => (
          <Track key={track.number != null ? track.number : track.id} track={track} />
        ))}
        <Others modules={Array.isArray(orphanModules) ? orphanModules : []} />
      </section>

      <div className="feedback-card">
        <div className="txt">
          <b>Help shape this course</b>
          <p>
            Stuck on a task? Found a bug? Want a module that doesn't exist yet? Every task was
            tested, but real learners find what tests can't.
          </p>
        </div>
        <a
          className="btn-feedback"
          href={FEEDBACK_URL}
          target="_blank"
          rel="noopener noreferrer"
        >
          <span aria-hidden="true">💬 </span>Leave feedback on GitHub →
        </a>
      </div>

      <footer className="pagefoot">
        GPU.js is MIT licensed · learn.gpu.rocks runs entirely in your browser — nothing to
        install, nothing uploaded.
      </footer>
    </div>
  );
}

export default LearnHome;
