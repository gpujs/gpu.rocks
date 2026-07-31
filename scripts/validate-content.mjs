/**
 * scripts/validate-content.mjs — the build's content gate. Wired into
 * `yarn build` AHEAD of vite, so a broken course identity fails in two
 * seconds with a readable list instead of thirty seconds into a bundle.
 *
 * Data checks (uuid present / well-formed / unique, short ids unique, version
 * a positive integer, module slugs unique, task slugs unique within a module,
 * every uuid a track lists exists, no module in two tracks) come from
 * validateContent() inside src/Learn/content/registry.js — the same code the
 * app runs at import time, so the two can never disagree.
 *
 * This script adds the checks that need a FILESYSTEM, which registry.js by
 * design cannot do:
 *   • every module file is named <uuid>.js after the uuid it declares;
 *   • every figure directory is a module uuid;
 *   • every figure the metadata references has an svg file;
 *   • every svg file is referenced by metadata (no orphans);
 *   • figure metadata only names task slugs that exist in that module.
 *
 * Run standalone any time: `node scripts/validate-content.mjs`
 */
import { existsSync, readdirSync } from 'node:fs';
import { basename, join } from 'node:path';
import { pathToFileURL } from 'node:url';
import { FIGURES_DIR, MODULES_DIR, ROOT, loadContent, moduleFilePaths } from './contentLoader.mjs';

const PLACEMENTS = ['intro', 'goal'];
const problems = [];

// ---- data checks (shared with the app) -------------------------------------

let registry;
try {
  registry = await loadContent();
} catch (e) {
  // buildRegistry() already formats every problem it found, one per line.
  console.error(`validate-content: FAILED\n${e.message}`);
  process.exit(1);
}

// ---- file layout -----------------------------------------------------------

for (const file of await moduleFilePaths()) {
  const declared = (await import(pathToFileURL(join(ROOT, file)).href)).default.uuid;
  const named = basename(file, '.js');
  if (named !== declared) {
    problems.push(
      `${file}: file name says ${named} but the module declares uuid ${declared} — ` +
        `content module files are named <uuid>.js`
    );
  }
}

// ---- figures ---------------------------------------------------------------

const figuresRoot = join(ROOT, FIGURES_DIR);
const figureDirs = readdirSync(figuresRoot, { withFileTypes: true })
  .filter(entry => entry.isDirectory())
  .map(entry => entry.name);

for (const uuid of figureDirs) {
  const dir = join(figuresRoot, uuid);
  const module = registry.getModule(uuid);
  if (!module || module.uuid !== uuid) {
    problems.push(`${FIGURES_DIR}/${uuid}/: no module declares uuid ${uuid}`);
    continue;
  }
  const metaPath = join(dir, 'index.js');
  if (!existsSync(metaPath)) {
    problems.push(`${FIGURES_DIR}/${uuid}/: no index.js (figure metadata)`);
    continue;
  }
  const meta = (await import(pathToFileURL(metaPath).href)).default || {};
  const onDisk = new Set(readdirSync(dir).filter(name => name.endsWith('.svg')));
  const claimed = new Set();

  for (const [taskSlug, entries] of Object.entries(meta)) {
    if (!module.tasks.some(task => task.slug === taskSlug)) {
      problems.push(
        `${FIGURES_DIR}/${uuid}/index.js: "${module.title}" has no task with slug "${taskSlug}"`
      );
      continue;
    }
    for (const figure of entries) {
      const file = `${taskSlug}-${figure.name}.svg`;
      if (!onDisk.has(file)) {
        problems.push(`${FIGURES_DIR}/${uuid}/${file}: referenced by index.js but missing on disk`);
      } else claimed.add(file);
      if (!PLACEMENTS.includes(figure.placement)) {
        problems.push(
          `${FIGURES_DIR}/${uuid}/index.js: figure "${figure.name}" has placement ` +
            `"${figure.placement}" — expected one of ${PLACEMENTS.join(', ')}`
        );
      }
      if (!figure.caption || !String(figure.caption).trim()) {
        problems.push(`${FIGURES_DIR}/${uuid}/index.js: figure "${figure.name}" has no caption`);
      }
    }
  }
  for (const file of onDisk) {
    if (!claimed.has(file)) {
      problems.push(`${FIGURES_DIR}/${uuid}/${file}: on disk but no metadata claims it`);
    }
  }
}

// ---- verdict ---------------------------------------------------------------

if (problems.length) {
  console.error(`validate-content: FAILED — ${problems.length} problem(s)`);
  problems.forEach(problem => console.error(`  • ${problem}`));
  process.exit(1);
}

const taskCount = registry.modules.reduce((sum, module) => sum + module.tasks.length, 0);
const figureCount = figureDirs.reduce(
  (sum, uuid) => sum + readdirSync(join(figuresRoot, uuid)).filter(n => n.endsWith('.svg')).length,
  0
);
console.log(
  `validate-content: ok — ${registry.modules.length} modules ` +
    `(${registry.tracks.length} tracks, ${registry.orphanModules.length} others), ` +
    `${taskCount} tasks, ${figureCount} figures`
);
