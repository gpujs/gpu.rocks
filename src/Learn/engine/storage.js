// engine/storage.js — the ONLY learn module that touches localStorage.
//
// Keys:
//   gpujs-learn:theme          → 'auto' | 'light' | 'dark'
//   gpujs-learn:schema         → storage schema version, '2' (see MIGRATION)
//   gpujs-learn:progress       → JSON { [taskKey]: { done, completedAt, total? } }
//   gpujs-learn:code:<taskKey> → the user's editor content, verbatim
//
// taskKey is `<uuid>:v<version>:<taskSlug>` — built by content/registry.js and
// nowhere else. Module uuid + module version + the task's own SLUG:
//
//   • uuid      — renaming a module (title, slug, url) cannot orphan progress;
//   • version   — bumping a module ARCHIVES what the learner did at the old
//                 version instead of overwriting or deleting it;
//   • task slug — inserting or reordering tasks cannot scramble progress the
//                 way a task NUMBER would.
//
// ---------------------------------------------------------------------------
// ARCHIVE ON BUMP
//
// Because the version is inside the key, "archiving" needs no copying: keys
// written at v1 simply stop matching the lookups a v2 module performs, and sit
// in storage untouched. Reading is therefore version-scoped by construction
// (moduleProgress / isTaskDone / getSavedCode), and the older records are
// reachable through the deliberately separate archive accessors:
//
//   previousVersionProgress(module) → what they finished at the last version
//   archivedVersions(module)        → every older version with records
//   getArchivedCode(module, t, v)   → their code, from that older version
//
// Nothing a learner wrote is ever destroyed by a version bump.
//
// ---------------------------------------------------------------------------
// MIGRATION (schema 1 → 2)
//
// Schema 1 keyed everything by the pre-uuid module id and the task NUMBER:
// progress entries `"1-2-3"` and code keys `gpujs-learn:code:1-2-3`.
// migrateLegacyProgress() below copies those to `<uuid>:v1:<taskSlug>` once,
// then records `gpujs-learn:schema = 2` so it never runs again. The mapping is
// DERIVED from the registry (module.legacyId + task order), never hardcoded.
//
// Every accessor is wrapped in try/catch so private browsing / disabled
// storage degrades to "nothing persists" instead of throwing.
//
// NOTE: this file imports the VITE registry (content/index.js) for the module
// list the migration mapping is built from, so — like engine/runner.js and
// engine/sandbox.worker.js — it is app-only and must not be imported by a node
// script.

import { getModule, modules } from '../content/index.js';
import { moduleKeyPrefix, parseTaskKey, taskKey } from '../content/registry.js';

const THEME_KEY = 'gpujs-learn:theme';
const SCHEMA_KEY = 'gpujs-learn:schema';
const PROGRESS_KEY = 'gpujs-learn:progress';
const CODE_PREFIX = 'gpujs-learn:code:';

const THEME_PREFS = ['auto', 'light', 'dark'];

// The schema this build writes. 1 (implicit — no marker) was the pre-uuid
// `<oldModuleId>-<taskNum>` layout; 2 is `<uuid>:v<version>:<taskSlug>`.
const SCHEMA_VERSION = 2;

// Schema-1 records predate module versioning, so they all belong to v1.
const LEGACY_MODULE_VERSION = 1;

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

// Every key currently in localStorage. Needed by the archive accessors and the
// migration, which both have to find keys they cannot name in advance.
function allKeys() {
  try {
    const store = window.localStorage;
    const out = [];
    for (let i = 0; i < store.length; i++) {
      const key = store.key(i);
      if (key != null) out.push(key);
    }
    return out;
  } catch (e) {
    return [];
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

// → { [taskKey]: { done: true, completedAt: ISO string, total?: number } }
//
// ALL versions of all modules, exactly as stored — this is the raw object.
// Callers that care about "the current version of module X" want
// moduleProgress() / isTaskDone(), which are version-scoped through taskKey().
export function getProgress() {
  const raw = read(PROGRESS_KEY);
  if (!raw) return {};
  try {
    const parsed = JSON.parse(raw);
    return parsed && typeof parsed === 'object' && !Array.isArray(parsed) ? parsed : {};
  } catch (e) {
    return {};
  }
}

// How many tasks the module named by `key` has, but ONLY when the key is for
// the module's current version — that is the one case where today's task count
// is the count that key was written against.
//
// Recording it makes an archived version legible later: without it, a version's
// entries say how many tasks were finished but not how many there were, and a
// learner who did 2 of 5 would be indistinguishable from one who finished a
// 2-task module. See previousVersionProgress().
function totalFor(key) {
  const parsed = parseTaskKey(key);
  if (!parsed) return null;
  const module = getModule(parsed.uuid);
  if (!module || module.version !== parsed.version || !Array.isArray(module.tasks)) return null;
  return module.tasks.length;
}

export function markTaskDone(taskId) {
  const progress = getProgress();
  if (progress[taskId] && progress[taskId].done) return progress;
  const entry = { done: true, completedAt: new Date().toISOString() };
  const total = totalFor(taskId);
  if (total != null) entry.total = total;
  progress[taskId] = entry;
  write(PROGRESS_KEY, JSON.stringify(progress));
  return progress;
}

export function isTaskDone(taskId) {
  const entry = getProgress()[taskId];
  return Boolean(entry && entry.done);
}

// Per-module rollup for the landing page and crumb bar, for the module's
// CURRENT version only — a bumped module reads as untouched here even though
// the old version's records are still in storage (previousVersionProgress()).
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
    const id = taskKey(module, tasks[i]);
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

// ---- the archive (records from earlier versions of a module) ---------------

function slugOf(task) {
  return typeof task === 'string' ? task : task && task.slug ? task.slug : '';
}

// Every version of `module` OLDER than the current one that still has records
// (progress entries or saved code), newest first.
export function archivedVersions(module) {
  if (!module || !module.uuid) return [];
  const current = module.version;
  const prefix = moduleKeyPrefix(module);
  const found = new Set();

  const consider = key => {
    const parsed = parseTaskKey(key);
    if (parsed && parsed.uuid === module.uuid && parsed.version < current) found.add(parsed.version);
  };

  Object.keys(getProgress()).forEach(key => {
    if (key.startsWith(prefix)) consider(key);
  });
  allKeys().forEach(key => {
    if (key.startsWith(CODE_PREFIX)) consider(key.slice(CODE_PREFIX.length));
  });

  return [...found].sort((a, b) => b - a);
}

/**
 * previousVersionProgress(module) → what the learner did at the most recent
 * OLDER version of this module, or null if they did nothing there.
 *
 *   { version, done, total, complete, completedAt }
 *
 * `complete` is the flag the home page wants: true means they finished the
 * module and it has been revised since ("updated since you completed it").
 *
 * `total` is the task count that version had. It comes from the count recorded
 * on the entries themselves (see totalFor); when entries predate that (or the
 * count is unknowable) it falls back to the number of distinct task slugs seen
 * at that version, counting saved code as well as progress — a floor, never an
 * overstatement, so `complete` cannot be claimed on a module the learner only
 * partly finished.
 */
export function previousVersionProgress(module) {
  if (!module || !module.uuid) return null;
  const current = module.version;
  const prefix = moduleKeyPrefix(module);
  const versions = new Map(); // version → { done, slugs, total, completedAt }

  const bucket = version => {
    if (!versions.has(version)) {
      versions.set(version, { done: 0, slugs: new Set(), total: 0, completedAt: null });
    }
    return versions.get(version);
  };

  const progress = getProgress();
  Object.keys(progress).forEach(key => {
    if (!key.startsWith(prefix)) return;
    const parsed = parseTaskKey(key);
    if (!parsed || parsed.uuid !== module.uuid || parsed.version >= current) return;
    const entry = progress[key];
    if (!entry || entry.done !== true) return;
    const at = bucket(parsed.version);
    at.done += 1;
    at.slugs.add(parsed.taskSlug);
    if (Number.isInteger(entry.total) && entry.total > at.total) at.total = entry.total;
    if (typeof entry.completedAt === 'string' && entry.completedAt > (at.completedAt || '')) {
      at.completedAt = entry.completedAt;
    }
  });

  if (versions.size === 0) return null;

  // Saved code widens the picture: a task they opened but never finished still
  // proves the version HAD that task.
  allKeys().forEach(key => {
    if (!key.startsWith(CODE_PREFIX)) return;
    const parsed = parseTaskKey(key.slice(CODE_PREFIX.length));
    if (!parsed || parsed.uuid !== module.uuid) return;
    const at = versions.get(parsed.version);
    if (at) at.slugs.add(parsed.taskSlug);
  });

  const version = Math.max(...versions.keys());
  const at = versions.get(version);
  const total = Math.max(at.total, at.slugs.size);
  return {
    version,
    done: at.done,
    total,
    complete: total > 0 && at.done >= total,
    completedAt: at.completedAt,
  };
}

/**
 * getArchivedCode(module, task, version) → the code the learner saved for that
 * task at an OLDER version of this module, or null.
 *
 * `task` is a task object or a task slug (a slug that no longer exists in the
 * current version is fine — that is the point). `version` defaults to the most
 * recent archived version.
 */
export function getArchivedCode(module, task, version) {
  if (!module || !module.uuid) return null;
  const slug = slugOf(task);
  if (!slug) return null;
  const at = version == null ? archivedVersions(module)[0] : Number(version);
  if (!Number.isInteger(at) || at < 1 || at >= module.version) return null;
  return read(`${CODE_PREFIX}${module.uuid}:v${at}:${slug}`);
}

// ---- migration: schema 1 (`<oldModuleId>-<taskNum>`) → schema 2 ------------

export function getSchemaVersion() {
  const raw = read(SCHEMA_KEY);
  const parsed = Number.parseInt(raw, 10);
  // No marker at all = schema 1: the layout that shipped before uuids.
  return Number.isInteger(parsed) && parsed > 0 ? parsed : 1;
}

// The old→new key mapping, DERIVED from the registry: module.legacyId is the
// pre-uuid module id, and a task's position is the pre-uuid task number, so
// `"<legacyId>-<i+1>"` → `<uuid>:v1:<tasks[i].slug>`. 15 modules, 75 tasks.
//
// Built from content rather than written out by hand so it cannot drift from
// the course; a module without a legacyId simply has no schema-1 identity and
// contributes nothing.
function legacyKeyMap() {
  const map = new Map();
  modules.forEach(module => {
    if (!module.legacyId || !Array.isArray(module.tasks)) return;
    // Today's task count is the v1 count only while the module still IS v1;
    // for anything already bumped we record no total and let
    // previousVersionProgress() fall back to counting slugs.
    const total = module.version === LEGACY_MODULE_VERSION ? module.tasks.length : null;
    module.tasks.forEach((task, i) => {
      if (!task || !task.slug) return;
      map.set(`${module.legacyId}-${i + 1}`, {
        key: `${module.uuid}:v${LEGACY_MODULE_VERSION}:${task.slug}`,
        total,
      });
    });
  });
  return map;
}

// A schema-1 progress value → a schema-2 entry, or null if it does not say
// "done" (corrupt values, half-written objects, explicit false).
function migratedEntry(value, total) {
  const done = value === true || (value && typeof value === 'object' && value.done === true);
  if (!done) return null;
  const completedAt =
    value && typeof value === 'object' && typeof value.completedAt === 'string'
      ? value.completedAt
      : new Date().toISOString();
  const entry = { done: true, completedAt };
  if (Number.isInteger(total)) entry.total = total;
  return entry;
}

/**
 * migrateLegacyProgress() → a report, and never throws.
 *
 *   { ran, reason, from, progress, code, skipped }
 *
 * `from` is the schema version found in storage; `progress`/`code` count the
 * records copied and `skipped` the recognised ones deliberately not copied.
 *
 * Runs at most once per browser: it no-ops as soon as `gpujs-learn:schema` is
 * at 2 or above. Called for its side effect at import time (below); exported
 * so it can be driven directly in verification.
 *
 * Guarantees, in the order they matter:
 *   • COPIES, never moves — the schema-1 entries and code keys stay exactly
 *     where they were. Nothing a learner wrote is destroyed, and a stale build
 *     served from cache still finds its own data.
 *   • never overwrites an existing schema-2 entry (so a learner who already
 *     worked in the new layout wins over their old records);
 *   • leaves every key it does not recognise completely alone, including a
 *     `gpujs-learn:progress` that is not parseable JSON;
 *   • degrades to a no-op when localStorage is unavailable (private browsing);
 *   • idempotent: running it again (marker cleared) changes nothing, because
 *     every target key now exists.
 */
export function migrateLegacyProgress() {
  const report = { ran: false, reason: '', from: 1, progress: 0, code: 0, skipped: 0 };
  try {
    const schema = getSchemaVersion();
    report.from = schema;
    if (schema >= SCHEMA_VERSION) return { ...report, reason: 'already-migrated' };

    // A probe write, so unavailable storage is reported as such rather than as
    // a successful migration of nothing.
    if (!write(SCHEMA_KEY, String(schema))) return { ...report, reason: 'unavailable' };

    const map = legacyKeyMap();

    // ---- progress: one object, rewritten once
    const raw = read(PROGRESS_KEY);
    let parsed = null;
    if (raw) {
      try {
        const candidate = JSON.parse(raw);
        if (candidate && typeof candidate === 'object' && !Array.isArray(candidate)) {
          parsed = candidate;
        }
      } catch (e) {
        // Corrupt progress blob: unreadable, so unmigratable — but it is the
        // learner's, so leave the bytes exactly as they are.
        parsed = null;
      }
    }
    if (parsed) {
      const next = { ...parsed };
      let changed = false;
      Object.keys(parsed).forEach(oldKey => {
        const target = map.get(oldKey);
        if (!target) return; // not a schema-1 key we know — untouched
        if (Object.prototype.hasOwnProperty.call(next, target.key)) {
          report.skipped += 1; // schema-2 entry already there; it wins
          return;
        }
        const entry = migratedEntry(parsed[oldKey], target.total);
        if (!entry) {
          report.skipped += 1; // not done / corrupt value
          return;
        }
        next[target.key] = entry;
        changed = true;
        report.progress += 1;
      });
      if (changed) write(PROGRESS_KEY, JSON.stringify(next));
    }

    // ---- saved code: one key per task, copied byte for byte
    allKeys().forEach(key => {
      if (!key.startsWith(CODE_PREFIX)) return;
      const target = map.get(key.slice(CODE_PREFIX.length));
      if (!target) return;
      const newKey = CODE_PREFIX + target.key;
      if (read(newKey) != null) {
        report.skipped += 1;
        return;
      }
      const value = read(key);
      if (value == null) return;
      if (write(newKey, value)) report.code += 1;
    });

    write(SCHEMA_KEY, String(SCHEMA_VERSION));
    return { ...report, ran: true, reason: 'migrated' };
  } catch (e) {
    // Belt and braces: the accessors above already swallow storage errors, and
    // a learner's stored junk must never take the page down with it.
    return { ...report, reason: 'error' };
  }
}

// Run it the moment the storage layer is loaded — before any component can ask
// what the learner has done. Guarded so importing this file outside a browser
// is inert rather than fatal.
if (typeof window !== 'undefined') migrateLegacyProgress();
