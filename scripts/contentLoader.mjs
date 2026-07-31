/**
 * scripts/contentLoader.mjs — the NODE side of the course registry.
 *
 * The twin of src/Learn/content/index.js: same content files, same
 * buildRegistry(), same validation — but the file list comes from node's fs
 * glob instead of vite's import.meta.glob, so plain `node` can enumerate the
 * course (prerender, sitemap, content validation) without a bundler.
 *
 * src/Learn/content/index.js must NEVER be imported from a node script; this
 * is what to import instead. Everything else about the registry lives in
 * src/Learn/content/registry.js, which both sides share verbatim.
 */
import { glob } from 'node:fs/promises';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { buildRegistry } from '../src/Learn/content/registry.js';
import trackMeta from '../src/Learn/content/tracks.js';

export const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
export const MODULES_DIR = 'src/Learn/content/modules';
export const FIGURES_DIR = 'src/Learn/content/figures';

// The module files on disk, sorted by path (i.e. by uuid) for a stable read
// order. Canonical course order comes from tracks.js, not from this.
export async function moduleFilePaths(root = ROOT) {
  const files = [];
  for await (const file of glob(`${MODULES_DIR}/*.js`, { cwd: root })) files.push(file);
  return files.sort();
}

/**
 * loadContent() → the same registry object the app gets from content/index.js:
 * { modules, tracks, orphanModules, getModule, moduleByShortId, moduleBySlug,
 *   parseModulePath, getTask, getTaskBySlug, nextModule }
 *
 * Throws (loudly, with every problem listed) if the content is invalid.
 */
export async function loadContent(root = ROOT) {
  const files = await moduleFilePaths(root);
  if (files.length === 0) {
    throw new Error(`no course content modules found (${MODULES_DIR}/*.js)`);
  }
  const loaded = [];
  for (const file of files) {
    const mod = (await import(pathToFileURL(join(root, file)).href)).default;
    if (!mod) throw new Error(`content module ${file} has no default export`);
    loaded.push(mod);
  }
  return buildRegistry(loaded, trackMeta);
}
