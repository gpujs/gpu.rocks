// content/registry.js — the NODE-SAFE core of the course registry.
//
// Everything that turns raw content files into resolved course data lives
// here: identity rules, url building, storage-key building, validation, and
// buildRegistry() itself. It is deliberately free of vite syntax, DOM access
// and filesystem access, so BOTH sides can use the same code:
//
//   • the app       — content/index.js globs the module files with
//                     import.meta.glob and hands them to buildRegistry();
//   • node scripts  — scripts/contentLoader.mjs reads the same files with
//                     node's fs glob and hands them to buildRegistry().
//
// KEEP IT THAT WAY. content/index.js is vite-only and must never be imported
// by a node script; this file is the twin that node imports instead.
//
// ---------------------------------------------------------------------------
// IDENTITY
//
// A module is identified by a v4 `uuid` declared in its own file, plus an
// integer `version` starting at 1. The uuid NEVER changes. The version is
// bumped by hand when the module's tasks change materially; bumping it starts
// the module unsolved without destroying what a learner did at the old
// version (see taskKey / parseTaskKey).
//
// The SHORT ID is the first 8 hex characters of the uuid. It is derived, never
// stored, and it is what a URL actually resolves on:
//
//     /learn/<module-slug>-<shortId>/<step>
//
// The slug is decoration — renaming a module changes it, and the app
// redirects a stale slug to the canonical one rather than 404ing. `step` is
// the 1-based task position, and it is the ONLY place a task is addressed by
// position: storage keys use the task's slug so reordering tasks never
// scrambles progress.

export const UUID_RE =
  /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/;

export const SHORT_ID_LENGTH = 8;
export const LEARN_BASE = '/learn';

const SLUG_RE = /^[a-z0-9]+(?:-[a-z0-9]+)*$/;
const SHORT_ID_RE = new RegExp(`^[0-9a-f]{${SHORT_ID_LENGTH}}$`);

// `/learn/<slug>-<shortId>`; the slug part is optional so a bare short id
// still resolves (and then redirects to the canonical, slugged form).
const MODULE_PARAM_RE = new RegExp(`^(?:(.*)-)?([0-9a-f]{${SHORT_ID_LENGTH}})$`);

// ---- pure helpers (no registry needed) ------------------------------------

function uuidOf(module) {
  if (typeof module === 'string') return module;
  return module && module.uuid ? module.uuid : '';
}

// shortId('f1399353-b65c-…') → 'f1399353'. Accepts a module or a uuid string.
export function shortId(module) {
  return uuidOf(module).slice(0, SHORT_ID_LENGTH);
}

// Title → url slug. Modules STORE their slug rather than deriving it at read
// time, so that editing a title cannot silently move a page; this exists to
// propose a slug for a new module and to keep that spelling consistent.
export function slugify(text) {
  return String(text || '')
    .toLowerCase()
    .replace(/&/g, ' and ')
    .replace(/['’]/g, '')
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '');
}

// moduleUrl(module) → '/learn/hello-kernel-f1399353'
export function moduleUrl(module) {
  return `${LEARN_BASE}/${module.slug}-${shortId(module)}`;
}

// taskUrl(module, step) → '/learn/hello-kernel-f1399353/3' (step is 1-based)
export function taskUrl(module, step) {
  return `${moduleUrl(module)}/${step}`;
}

// The human-facing module number, '1.2' — track number and 1-based position
// within it. null for an orphan: modules in no track are UNORDERED by design
// ("Others" has no sequence), so there is no number to show.
//
// Derived from the track's ordering, never stored: it is a label, not an
// identity, and it changes freely when content is reordered.
export function moduleNumber(module) {
  return module && module.track != null && module.trackIndex >= 0
    ? `${module.track}.${module.trackIndex + 1}`
    : null;
}

// The prefix every progress/code key for this module shares, across versions.
// Storage uses it to find what a learner did at an EARLIER version — bumping a
// version archives, it never deletes.
export function moduleKeyPrefix(module) {
  return `${uuidOf(module)}:`;
}

// The prefix for one specific version of a module.
export function versionKeyPrefix(module) {
  return `${uuidOf(module)}:v${module.version}:`;
}

// taskKey(module, task) → '<uuid>:v<version>:<taskSlug>'
//
// THE storage identity of a task: the key in the `gpujs-learn:progress` object
// and the suffix of `gpujs-learn:code:<key>`. Task SLUG, never task number, so
// inserting or reordering tasks leaves existing progress attached to the right
// exercise. Version is part of the key, so a version bump archives the old
// keys instead of overwriting them.
export function taskKey(module, task) {
  return `${versionKeyPrefix(module)}${typeof task === 'string' ? task : task.slug}`;
}

// parseTaskKey('<uuid>:v1:first-kernel') → { uuid, version, taskSlug } | null.
// The inverse of taskKey — for the storage layer reading back keys it did not
// write in this session (older versions included).
export function parseTaskKey(key) {
  const match = /^([0-9a-f-]{36}):v(\d+):(.+)$/.exec(String(key || ''));
  if (!match) return null;
  return { uuid: match[1], version: Number(match[2]), taskSlug: match[3] };
}

// Split a `:moduleParam` route segment into its parts WITHOUT resolving it.
// → { slug, shortId } | null. Use registry.parseModulePath() to resolve.
export function parseModuleParam(param) {
  const match = MODULE_PARAM_RE.exec(String(param || '').toLowerCase());
  if (!match) return null;
  return { slug: match[1] || '', shortId: match[2] };
}

// ---- validation -----------------------------------------------------------

// Returns a list of human-readable problems; empty means the content is sound.
// Checks only what can be checked from the data itself — figure FILES are
// checked by content/figures/index.js (app) and scripts/validate-content.mjs
// (build), which are the two places that can see the filesystem.
export function validateContent(modules, trackMeta) {
  const problems = [];
  const seen = { uuid: new Map(), short: new Map(), slug: new Map(), legacy: new Map() };

  const name = (module, i) =>
    module && (module.title || module.uuid) ? `${module.title || module.uuid}` : `module #${i + 1}`;

  modules.forEach((module, i) => {
    const where = name(module, i);
    if (!module || typeof module !== 'object') {
      problems.push(`${where}: not an object`);
      return;
    }

    if (!module.uuid) problems.push(`${where}: missing uuid`);
    else if (!UUID_RE.test(module.uuid)) {
      problems.push(`${where}: uuid "${module.uuid}" is not a well-formed v4 uuid`);
    } else if (seen.uuid.has(module.uuid)) {
      problems.push(`${where}: uuid ${module.uuid} is also used by "${seen.uuid.get(module.uuid)}"`);
    } else {
      seen.uuid.set(module.uuid, where);
      const short = shortId(module);
      // Two uuids sharing their first 8 hex chars would make two URLs
      // resolve to each other. Astronomically unlikely; still fatal.
      if (seen.short.has(short)) {
        problems.push(
          `${where}: short id ${short} collides with "${seen.short.get(short)}" — ` +
            `regenerate one of the two uuids`
        );
      } else seen.short.set(short, where);
    }

    if (!Number.isInteger(module.version) || module.version < 1) {
      problems.push(`${where}: version must be a positive integer (got ${JSON.stringify(module.version)})`);
    }

    if (!module.slug) problems.push(`${where}: missing slug`);
    else if (!SLUG_RE.test(module.slug)) {
      problems.push(`${where}: slug "${module.slug}" is not kebab-case`);
    } else if (seen.slug.has(module.slug)) {
      problems.push(`${where}: slug "${module.slug}" is also used by "${seen.slug.get(module.slug)}"`);
    } else seen.slug.set(module.slug, where);

    if (module.legacyId) {
      if (seen.legacy.has(module.legacyId)) {
        problems.push(
          `${where}: legacyId "${module.legacyId}" is also used by "${seen.legacy.get(module.legacyId)}"`
        );
      } else seen.legacy.set(module.legacyId, where);
    }

    if (!module.title || !String(module.title).trim()) problems.push(`${where}: missing title`);

    if (!Array.isArray(module.tasks) || module.tasks.length === 0) {
      problems.push(`${where}: needs at least one task`);
      return;
    }
    const taskSlugs = new Map();
    module.tasks.forEach((task, t) => {
      const step = t + 1;
      if (!task || typeof task !== 'object') {
        problems.push(`${where} task ${step}: not an object`);
        return;
      }
      if (!task.slug) problems.push(`${where} task ${step}: missing slug`);
      else if (!SLUG_RE.test(task.slug)) {
        problems.push(`${where} task ${step}: slug "${task.slug}" is not kebab-case`);
      } else if (taskSlugs.has(task.slug)) {
        // Task slugs are storage keys within a module — a duplicate would
        // make two exercises share one progress entry.
        problems.push(
          `${where} task ${step}: slug "${task.slug}" duplicates task ${taskSlugs.get(task.slug)}`
        );
      } else taskSlugs.set(task.slug, step);

      // Optional: a task sized so the better algorithm actually wins may ask
      // for more than the default 5 s run budget. Capped, because the guard
      // exists to stop a learner freezing the page — a task cannot opt out of
      // it, only widen it.
      if (task.budgetMs !== undefined) {
        const asked = Number(task.budgetMs);
        if (!Number.isFinite(asked) || asked <= 0 || asked > 60000) {
          problems.push(
            `${where} task ${step}: budgetMs must be a positive number of ms up to 60000, got ${task.budgetMs}`
          );
        }
      }

      // A task may pin the GPU path to one backend. Only 'webgl' is meaningful
      // today: it stops mode "auto" using gpu.js's per-kernel WebGPU upgrade,
      // which would let a pipelined chain hand a texture across backends and
      // pay a readback the task exists to teach away.
      if (task.backend !== undefined && task.backend !== 'webgl') {
        problems.push(
          `${where} task ${step}: backend must be 'webgl' if set, got ${JSON.stringify(task.backend)}`
        );
      }

      // Hint TITLES are rendered as JSX text (<summary>{hint.title}</summary>),
      // so React escapes them and any markup shows up as literal tags to the
      // learner. Hint BODIES are trusted HTML and may contain markup.
      (Array.isArray(task.hints) ? task.hints : []).forEach((hint, h) => {
        if (hint && typeof hint.title === 'string' && /<[a-z/][^>]*>/i.test(hint.title)) {
          problems.push(
            `${where} task ${step}: hint ${h + 1} title contains HTML — titles render as ` +
              `plain text, so the tags would be shown to the learner: ${JSON.stringify(hint.title)}`
          );
        }
      });
    });
  });

  // ---- tracks
  const owner = new Map(); // uuid -> track title
  (trackMeta || []).forEach((track, i) => {
    const where = `track ${track && track.number != null ? track.number : i + 1}`;
    if (!Array.isArray(track.modules)) {
      problems.push(`${where}: needs a "modules" array of uuids`);
      return;
    }
    track.modules.forEach(uuid => {
      if (!seen.uuid.has(uuid)) {
        problems.push(`${where}: lists uuid ${uuid}, which no module file declares`);
        return;
      }
      // "A module may appear in at most one track" — including twice in one.
      if (owner.has(uuid)) {
        problems.push(
          `${where}: module "${seen.uuid.get(uuid)}" (${uuid}) is already in ${owner.get(uuid)}`
        );
      } else owner.set(uuid, where);
    });
  });

  return problems;
}

// ---- the registry ---------------------------------------------------------

function decorate(module, track, trackIndex) {
  // A shallow copy: `tasks` (and every task object) stays the SAME reference
  // the content file exports, so identity-keyed lookups elsewhere still work.
  return {
    ...module,
    shortId: shortId(module),
    // The track NUMBER, or null for an orphan ("Others"). Track membership is
    // owned by content/tracks.js; a module file does not declare it.
    track: track ? track.number : null,
    // 0-based position within its track, or -1 for an orphan. Orphans are
    // unordered by design: no "module N of M", no next-module offer.
    trackIndex: track ? trackIndex : -1,
    url: moduleUrl(module),
  };
}

/**
 * buildRegistry(moduleFiles, trackMeta) → the resolved course.
 *
 * @param {object[]} moduleFiles the default exports of content/modules/*.js
 * @param {object[]} trackMeta   content/tracks.js (each with a `modules`
 *                               array of module uuids, in teaching order)
 * @throws if validateContent() finds anything wrong — loudly, at import time,
 *         so a broken identity can never reach a build or a learner.
 */
export function buildRegistry(moduleFiles, trackMeta) {
  const raw = (moduleFiles || []).filter(Boolean);
  const problems = validateContent(raw, trackMeta);
  if (problems.length) {
    throw new Error(
      `gpu.js learn content is invalid (${problems.length} problem${problems.length === 1 ? '' : 's'}):\n` +
        problems.map(p => `  • ${p}`).join('\n')
    );
  }

  const byUuidRaw = new Map(raw.map(module => [module.uuid, module]));

  // Canonical order: every track in its declared order, then the orphans
  // ("Others") by title. Adjacency in this list is what "next module" means.
  const tracks = (trackMeta || []).map(track => ({
    ...track,
    modules: track.modules.map((uuid, i) => decorate(byUuidRaw.get(uuid), track, i)),
  }));

  const claimed = new Set(tracks.flatMap(track => track.modules.map(module => module.uuid)));
  const orphanModules = raw
    .filter(module => !claimed.has(module.uuid))
    .map(module => decorate(module, null, -1))
    .sort((a, b) => String(a.title).localeCompare(String(b.title)));

  const modules = [...tracks.flatMap(track => track.modules), ...orphanModules];

  const byUuid = new Map(modules.map(module => [module.uuid, module]));
  const byShort = new Map(modules.map(module => [module.shortId, module]));
  const bySlug = new Map(modules.map(module => [module.slug, module]));
  const byLegacy = new Map(
    modules.filter(module => module.legacyId).map(module => [module.legacyId, module])
  );

  // Resolve anything that names a module: the module object itself, a uuid, a
  // short id, a slug, or the pre-uuid legacy id ('1-2').
  //
  // The legacy-id branch is TRANSITIONAL — it exists only so call sites that
  // have not moved to uuids yet keep working. Nothing new should rely on it.
  function getModule(ref) {
    if (!ref) return null;
    if (typeof ref === 'object') return byUuid.get(ref.uuid) || null;
    const key = String(ref);
    return (
      byUuid.get(key) ||
      byShort.get(key.toLowerCase()) ||
      bySlug.get(key) ||
      byLegacy.get(key) ||
      null
    );
  }

  function moduleByShortId(short) {
    return byShort.get(String(short || '').toLowerCase()) || null;
  }

  function moduleBySlug(slug) {
    return bySlug.get(String(slug || '')) || null;
  }

  /**
   * parseModulePath('hello-kernel-f1399353') → { module, canonical: true }
   * parseModulePath('old-name-f1399353')     → { module, canonical: false }
   * parseModulePath('nope')                  → null
   *
   * The SHORT ID resolves; the slug is decoration. `canonical: false` means
   * the caller should replace the URL with moduleUrl(module) (in place, no
   * navigation) instead of 404ing — renaming a module never breaks a link.
   */
  function parseModulePath(param) {
    const parts = parseModuleParam(param);
    if (!parts) return null;
    const module = byShort.get(parts.shortId);
    if (!module) return null;
    return { module, canonical: parts.slug === module.slug };
  }

  function taskResult(module, index) {
    const task = module.tasks[index];
    const step = index + 1;
    const key = taskKey(module, task);
    return {
      module,
      task,
      step,
      taskIndex: index,
      taskSlug: task.slug,
      taskKey: key,
      total: module.tasks.length,
      url: taskUrl(module, step),
      // Transitional aliases for call sites still speaking the old shape.
      taskNum: step,
      taskId: key,
    };
  }

  /**
   * getTask(module, step) → task record | null. `step` is 1-based (it is the
   * URL's step). `module` is anything getModule() accepts; `step` may also be
   * a task SLUG, which is how identity-keyed callers (the worker) look up.
   */
  function getTask(ref, step) {
    const module = getModule(ref);
    if (!module) return null;
    if (typeof step === 'string' && !/^\d+$/.test(step)) return getTaskBySlug(module, step);
    const num = Number(step);
    if (!Number.isInteger(num) || num < 1 || num > module.tasks.length) return null;
    return taskResult(module, num - 1);
  }

  // getTaskBySlug(module, 'first-kernel') → task record | null.
  // Position-free lookup: what postMessage refs and storage keys resolve with.
  function getTaskBySlug(ref, taskSlug) {
    const module = getModule(ref);
    if (!module) return null;
    const index = module.tasks.findIndex(task => task.slug === taskSlug);
    return index === -1 ? null : taskResult(module, index);
  }

  // The module after this one in canonical order, or null. Orphans have no
  // "next": finishing one offers only the exit.
  function nextModule(ref) {
    const module = getModule(ref);
    if (!module || module.track == null) return null;
    const index = modules.indexOf(module);
    const next = index === -1 ? null : modules[index + 1] || null;
    return next && next.track === module.track ? next : null;
  }

  return {
    modules,
    tracks,
    orphanModules,
    getModule,
    moduleByShortId,
    moduleBySlug,
    parseModulePath,
    getTask,
    getTaskBySlug,
    nextModule,
  };
}
