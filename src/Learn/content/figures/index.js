// content/figures/index.js — figure auto-discovery. VITE ONLY.
//
// Figures are keyed by module UUID and TASK SLUG, matching the identity rules
// in ../registry.js: a module's figures live in ./<uuid>/, and each one is
// attached to a task by slug, never by task number, so reordering tasks does
// not silently move pictures.
//
// Authors add:
//   ./<uuid>/index.js                    metadata (default export, below)
//   ./<uuid>/<taskSlug>-<name>.svg       the markup
// and both are picked up here. Neither this file nor the metadata ever writes
// an svg path out — it is derived from (taskSlug, name).
//
// Metadata schema (default export of ./<uuid>/index.js):
//   { '<taskSlug>': [{ name, caption, alt?, placement: 'intro' | 'goal' }] }
//
// Every mismatch is fatal AT IMPORT TIME, in both directions: metadata naming
// an svg that does not exist, and an svg no metadata claims. Task slugs are
// checked against the real module, so a renamed task cannot orphan its figure.
// scripts/validate-content.mjs repeats these checks ahead of the build.

import { getModule } from '../index.js';

const metaFiles = import.meta.glob('./*/index.js', { eager: true });
const svgFiles = import.meta.glob('./*/*.svg', {
  eager: true,
  query: '?raw',
  import: 'default',
});

const PLACEMENTS = ['intro', 'goal'];

// uuid -> taskSlug -> [{ svg, caption, alt?, placement }]
// Built eagerly so a metadata typo throws at import time, not first render.
const figuresByModule = {};
const claimedSvgs = new Set();

for (const [path, file] of Object.entries(metaFiles)) {
  const uuid = path.match(/^\.\/([^/]+)\/index\.js$/)[1];
  const module = getModule(uuid);
  if (!module) {
    throw new Error(
      `figures/${uuid}/index.js: no module file declares uuid ${uuid} — ` +
        `the figure directory name must be a module's uuid`
    );
  }
  const byTask = {};
  for (const [taskSlug, entries] of Object.entries(file.default || {})) {
    if (!module.tasks.some(task => task.slug === taskSlug)) {
      throw new Error(
        `figures/${uuid}/index.js: "${module.title}" has no task with slug "${taskSlug}" ` +
          `(known: ${module.tasks.map(t => t.slug).join(', ')})`
      );
    }
    byTask[taskSlug] = entries.map(({ name, caption, alt, placement }) => {
      const svgPath = `./${uuid}/${taskSlug}-${name}.svg`;
      const svg = svgFiles[svgPath];
      if (svg == null) {
        throw new Error(
          `figures/${uuid}/index.js: task "${taskSlug}" references figure "${name}" but ` +
            `src/Learn/content/figures/${uuid}/${taskSlug}-${name}.svg does not exist`
        );
      }
      if (!PLACEMENTS.includes(placement)) {
        throw new Error(
          `figures/${uuid}/index.js: task "${taskSlug}" figure "${name}" has ` +
            `placement "${placement}" — expected one of: ${PLACEMENTS.join(', ')}`
        );
      }
      claimedSvgs.add(svgPath);
      return { svg, caption, alt, placement };
    });
  }
  figuresByModule[uuid] = byTask;
}

const orphanSvgs = Object.keys(svgFiles).filter(path => !claimedSvgs.has(path));
if (orphanSvgs.length) {
  throw new Error(
    `figures: ${orphanSvgs.length} svg file(s) no metadata claims — either add them to the ` +
      `matching <uuid>/index.js or delete them:\n` +
      orphanSvgs.map(path => `  src/Learn/content/figures/${path.slice(2)}`).join('\n')
  );
}

/**
 * getFigures(module, task) → [{ svg, caption, alt?, placement }]; [] when none.
 *
 * `module` is anything content/index.js getModule() accepts (module object,
 * uuid, short id, slug); `task` is a task object, a task slug, or a 1-based
 * step number.
 */
export function getFigures(moduleRef, taskRef) {
  const module = getModule(moduleRef);
  if (!module) return [];
  const byTask = figuresByModule[module.uuid];
  if (!byTask) return [];
  let taskSlug = null;
  if (taskRef && typeof taskRef === 'object') taskSlug = taskRef.slug;
  else if (typeof taskRef === 'number' || /^\d+$/.test(String(taskRef))) {
    const task = module.tasks[Number(taskRef) - 1];
    taskSlug = task ? task.slug : null;
  } else taskSlug = String(taskRef || '');
  return (taskSlug && byTask[taskSlug]) || [];
}
