// content/index.js — the app's course registry. VITE ONLY.
//
// This file's whole job is the two lines that need a bundler: globbing
// ./modules/*.js and handing the results to buildRegistry(). Everything else —
// identity rules, url building, storage-key building, validation — lives in
// ./registry.js, which is plain node-safe ESM.
//
// NEVER import this file from a node script (scripts/prerender.mjs and
// friends): import.meta.glob only exists inside vite. Node scripts use
// scripts/contentLoader.mjs, which feeds the same registry from the same
// files.
//
// Module files self-register simply by existing as ./modules/<uuid>.js with a
// default export carrying { uuid, version, slug, title, tasks }. The file name
// is opaque on purpose; every file carries a header comment naming its module.
// Which track a module belongs to is declared in ./tracks.js, not here.
//
// Anything wrong with the content (a malformed or duplicated uuid, a colliding
// short id, duplicate slugs, a track pointing at a module that doesn't exist,
// a module in two tracks) throws HERE, at import time, before a single pixel
// renders. scripts/validate-content.mjs runs the same check ahead of the build.

import { buildRegistry } from './registry.js';
import trackMeta from './tracks.js';

const moduleFiles = import.meta.glob('./modules/*.js', { eager: true });

const registry = buildRegistry(
  // Sorted by path so the input order is stable across platforms; the
  // registry's own canonical order comes from tracks.js, not from this.
  Object.keys(moduleFiles)
    .sort()
    .map(path => moduleFiles[path].default),
  trackMeta
);

// ---- resolved course data --------------------------------------------------

// Every module, decorated with { shortId, track, trackIndex, url }, in
// canonical order: each track's modules in teaching order, then the orphans
// (modules in no track) by title.
export const modules = registry.modules;

// Tracks with their ordered `modules` resolved from uuids.
export const tracks = registry.tracks;

// Modules belonging to no track — the "Others" category. Sorted by title,
// unordered by intent: no sequence, and no next-module offer on completion.
export const orphanModules = registry.orphanModules;

// ---- lookups ---------------------------------------------------------------

export const getModule = registry.getModule;
export const moduleByShortId = registry.moduleByShortId;
export const moduleBySlug = registry.moduleBySlug;
export const parseModulePath = registry.parseModulePath;
export const getTask = registry.getTask;
export const getTaskBySlug = registry.getTaskBySlug;
export const nextModule = registry.nextModule;

// ---- identity / url / storage-key helpers ----------------------------------
//
// Re-exported so app code has ONE import for course identity. They are pure
// functions; node scripts import them from ./registry.js directly.

export {
  LEARN_BASE,
  SHORT_ID_LENGTH,
  UUID_RE,
  moduleKeyPrefix,
  moduleNumber,
  moduleUrl,
  parseModuleParam,
  parseTaskKey,
  shortId,
  slugify,
  taskKey,
  taskUrl,
  versionKeyPrefix,
} from './registry.js';
