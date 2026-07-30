// content/figures/index.js — figure auto-discovery. THIS FILE IS NEVER EDITED
// AGAIN: authors add ./module-<moduleId>.js metadata plus
// ./<moduleId>/<taskNum>-<slug>.svg markup and both are picked up here.
// Vite-only (import.meta.glob) — never import from node scripts.
//
// Metadata schema (default export of module-<moduleId>.js):
//   { '<taskNum>': [{ slug, caption, alt?, placement: 'intro' | 'goal' }] }
// The svg path is derived, never written out: ./<moduleId>/<taskNum>-<slug>.svg

const metaFiles = import.meta.glob('./module-*.js', { eager: true });
const svgFiles = import.meta.glob('./*/*.svg', {
  eager: true,
  query: '?raw',
  import: 'default',
});

const PLACEMENTS = ['intro', 'goal'];

// moduleId -> taskNum (string) -> [{ svg, caption, alt?, placement }]
// Built eagerly so a metadata typo throws at import time, not first render.
const figuresByModule = {};

for (const [path, file] of Object.entries(metaFiles)) {
  const moduleId = path.match(/module-(.+)\.js$/)[1];
  const byTask = {};
  for (const [taskNum, entries] of Object.entries(file.default || {})) {
    byTask[taskNum] = entries.map(({ slug, caption, alt, placement }) => {
      const svgPath = `./${moduleId}/${taskNum}-${slug}.svg`;
      const svg = svgFiles[svgPath];
      if (svg == null) {
        throw new Error(
          `figures/module-${moduleId}.js: task ${taskNum} references slug "${slug}" ` +
            `but src/Learn/content/figures/${moduleId}/${taskNum}-${slug}.svg does not exist. ` +
            `Known svg files: ${Object.keys(svgFiles).join(', ') || '(none)'}`
        );
      }
      if (!PLACEMENTS.includes(placement)) {
        throw new Error(
          `figures/module-${moduleId}.js: task ${taskNum} figure "${slug}" has ` +
            `placement "${placement}" — expected one of: ${PLACEMENTS.join(', ')}`
        );
      }
      return { svg, caption, alt, placement };
    });
  }
  figuresByModule[moduleId] = byTask;
}

// getFigures('1-2', 3) → [{ svg, caption, alt?, placement }]; [] when none.
export function getFigures(moduleId, taskNum) {
  const byTask = figuresByModule[moduleId];
  return (byTask && byTask[String(taskNum)]) || [];
}
