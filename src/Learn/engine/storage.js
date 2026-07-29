// engine/storage.js — the ONLY learn module that touches localStorage.
//
// Keys:
//   gpujs-learn:theme         → 'auto' | 'light' | 'dark'
//   gpujs-learn:progress      → JSON { [taskId]: { done: true, completedAt: ISO } }
//   gpujs-learn:code:<taskId> → the user's editor content
//
// Every accessor is wrapped in try/catch so private browsing / disabled
// storage degrades to "nothing persists" instead of throwing.

const THEME_KEY = 'gpujs-learn:theme';
const PROGRESS_KEY = 'gpujs-learn:progress';
const CODE_PREFIX = 'gpujs-learn:code:';

const THEME_PREFS = ['auto', 'light', 'dark'];

function read(key) {
  try {
    return window.localStorage.getItem(key);
  } catch (e) {
    return null;
  }
}

function write(key, value) {
  try {
    window.localStorage.setItem(key, value);
    return true;
  } catch (e) {
    return false;
  }
}

function remove(key) {
  try {
    window.localStorage.removeItem(key);
    return true;
  } catch (e) {
    return false;
  }
}

// ---- theme ----------------------------------------------------------------

export function getThemePref() {
  const value = read(THEME_KEY);
  return THEME_PREFS.includes(value) ? value : 'auto';
}

export function setThemePref(pref) {
  write(THEME_KEY, THEME_PREFS.includes(pref) ? pref : 'auto');
}

// ---- progress -------------------------------------------------------------

// → { [taskId]: { done: true, completedAt: ISO string } }
export function getProgress() {
  const raw = read(PROGRESS_KEY);
  if (!raw) return {};
  try {
    const parsed = JSON.parse(raw);
    return parsed && typeof parsed === 'object' ? parsed : {};
  } catch (e) {
    return {};
  }
}

export function markTaskDone(taskId) {
  const progress = getProgress();
  if (progress[taskId] && progress[taskId].done) return progress;
  progress[taskId] = { done: true, completedAt: new Date().toISOString() };
  write(PROGRESS_KEY, JSON.stringify(progress));
  return progress;
}

export function isTaskDone(taskId) {
  const entry = getProgress()[taskId];
  return Boolean(entry && entry.done);
}

// Per-module rollup for the landing page and crumb bar.
// → { done, total, currentIndex, state: 'done' | 'now' | 'todo' }
// currentIndex is the 0-based index of the first incomplete task
// (-1 when every task is done).
export function moduleProgress(module) {
  const tasks = module && Array.isArray(module.tasks) ? module.tasks : [];
  const progress = getProgress();
  const total = tasks.length;
  let done = 0;
  let currentIndex = -1;
  for (let i = 0; i < total; i++) {
    const id = `${module.id}-${i + 1}`;
    if (progress[id] && progress[id].done) {
      done += 1;
    } else if (currentIndex === -1) {
      currentIndex = i;
    }
  }
  const state = total > 0 && done === total ? 'done' : done > 0 ? 'now' : 'todo';
  return { done, total, currentIndex, state };
}

// ---- saved code -----------------------------------------------------------

export function getSavedCode(taskId) {
  return read(CODE_PREFIX + taskId);
}

export function saveCode(taskId, code) {
  write(CODE_PREFIX + taskId, String(code));
}

export function clearCode(taskId) {
  remove(CODE_PREFIX + taskId);
}
