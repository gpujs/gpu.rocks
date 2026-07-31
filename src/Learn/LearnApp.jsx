import React, { useLayoutEffect } from 'react';
import { Routes, Route, Navigate, useLocation, useMatch, useParams } from 'react-router';
import { ThemeProvider, useTheme } from './ThemeContext';
import LearnHome from './home/LearnHome';
import TaskPage from './task/TaskPage';
import VerifyPage from './engine/VerifyPage';
import { LEARN_BASE, getTask, moduleUrl, parseModulePath, taskUrl } from './content/index';
import { moduleProgress } from './engine/storage';
import './scss/learn.scss';

// ---- url canonicalisation ---------------------------------------------------
//
// A learn URL is `/learn/<module-slug>-<shortId>[/<step>]`. Only the SHORT ID
// resolves; the slug is decoration and the step is a 1-based position. So there
// are several spellings of the same page — a stale slug left over from before a
// module was renamed, a bare short id, a task addressed by its slug instead of
// its number, a trailing slash — and every one of them must land on the page
// rather than 404, with the address bar quietly corrected.
//
// "Quietly" is the point: this is NOT a redirect. A redirect would render an
// empty frame, push the router through a second location, and (for anything
// with a body) throw away the query string. Instead we render the route tree
// for the CANONICAL location on the very first pass — `<Routes location>` is
// exactly that override — and fix the address bar with history.replaceState in
// a layout effect, before the browser paints. No navigation, no history entry,
// no re-mount, no extra document request; the learner sees the right page and a
// correct URL, and never sees the wrong one.

/**
 * The canonical spelling of the current learn URL, or null when there is
 * nothing to fix (already canonical, or not a resolvable module page — those
 * fall through to the routes below, which send them to /learn).
 *
 * Query and fragment are carried across untouched: they are not ours to drop.
 */
function useCanonicalLearnPath() {
  const { pathname, search, hash } = useLocation();
  // Matched with the router rather than by hand-parsing the pathname, so these
  // stay in step with the <Route path> patterns below.
  const taskMatch = useMatch(`${LEARN_BASE}/:moduleParam/:step`);
  const moduleMatch = useMatch(`${LEARN_BASE}/:moduleParam`);
  const params = (taskMatch || moduleMatch || {}).params;
  if (!params) return null;

  const found = parseModulePath(params.moduleParam);
  if (!found) return null; // unknown short id — let the route redirect to /learn

  let path;
  if (params.step == null) {
    path = moduleUrl(found.module);
  } else {
    // getTask() accepts a step number or a task slug; either way the record
    // carries the canonical url for the task it resolved to.
    const record = getTask(found.module, params.step);
    if (!record) return null; // no such task — let the route redirect to /learn
    path = record.url;
  }
  return path === pathname ? null : path + search + hash;
}

// /learn/<module-slug>-<shortId> → first incomplete task (task 1 when the
// module is untouched or fully complete).
function ModuleRedirect() {
  const { moduleParam } = useParams();
  const { search, hash } = useLocation();
  const found = parseModulePath(moduleParam);
  if (!found) return <Navigate to={LEARN_BASE} replace />;
  const { currentIndex } = moduleProgress(found.module);
  const step = currentIndex === -1 ? 1 : currentIndex + 1;
  return <Navigate to={taskUrl(found.module, step) + search + hash} replace />;
}

function LearnRoutes() {
  const canonical = useCanonicalLearnPath();

  // Before paint, and only when the spelling actually changed. Preserving
  // history.state keeps the router's own entry bookkeeping (its key/index)
  // intact — we are correcting this entry, not adding one.
  useLayoutEffect(() => {
    if (canonical) window.history.replaceState(window.history.state, '', canonical);
  }, [canonical]);

  return (
    // The override (when there is one) is why every route below can assume its
    // :moduleParam is canonical.
    <Routes location={canonical || undefined}>
      <Route index element={<LearnHome />} />
      <Route path=":moduleParam" element={<ModuleRedirect />} />
      <Route path=":moduleParam/:step" element={<TaskPage />} />
      {/* anything else — including the pre-uuid /learn/1-2/3 urls, which are
          dropped rather than redirected — goes back to the course list */}
      <Route path="*" element={<Navigate to={LEARN_BASE} replace />} />
    </Routes>
  );
}

// Everything inside .learn-root is scoped away from the site's global CSS.
// data-theme always carries the *effective* theme ('light' | 'dark');
// 'auto' is resolved in ThemeContext via matchMedia.
function LearnRoot({ children }) {
  const { theme } = useTheme();
  return (
    <div className="learn-root" data-theme={theme}>
      <div className="screen">{children}</div>
    </div>
  );
}

// Mounted at /learn/* (course UI) and at /learn-verify (verify prop set —
// the headless test hook used by scripts/verify-learn.mjs).
function LearnApp({ verify = false }) {
  return (
    <ThemeProvider>
      <LearnRoot>{verify ? <VerifyPage /> : <LearnRoutes />}</LearnRoot>
    </ThemeProvider>
  );
}

export default LearnApp;
