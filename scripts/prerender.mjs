/**
 * Prerender + SEO build step — wired into `yarn build` (CI deploys dist/ on
 * every master push, so everything crawlers need must be baked here).
 *
 * Crawlers and link unfurlers don't run JS, so every canonical URL must serve
 * real HTML with its own <head> tags at a 200 on GitHub Pages. This script:
 *
 *  1. reads dist/index.html (the vite output) as the template, strips its
 *     page-specific head tags, and for every route writes
 *     dist/<path>/index.html with route-specific <title>, meta description,
 *     canonical, Open Graph, Twitter card, and JSON-LD tags injected;
 *  2. writes dist/404.html — the app shell with a generic title and
 *     <meta name=robots content=noindex>. GH Pages serves it (404 status) for
 *     unknown paths, keeping deep SPA routes like /learn-verify functional
 *     without getting them indexed;
 *  3. writes dist/sitemap.xml listing exactly the prerendered canonical URLs
 *     (public/robots.txt points crawlers at it).
 *
 * Meta values come from src/routeMeta.js — the single source of truth shared
 * with the SPA runtime (src/pageMeta.js). Course content is enumerated with
 * scripts/contentLoader.mjs, the node-safe twin of the app's registry —
 * NEVER the vite-only src/Learn/content/index.js.
 *
 * Learn URLs are `/learn/<module-slug>-<shortId>[/<step>]`, built by the
 * registry (moduleUrl/taskUrl) so the static pages, the sitemap and the running
 * app can never disagree about what a page is called. The pre-uuid
 * `/learn/<track>-<module>/<task>` paths are DELIBERATELY not emitted and not
 * in the sitemap — the owner chose to drop them outright rather than ship
 * redirect stubs, and the SPA sends any leftover link to /learn. Do not
 * "restore" them.
 *
 * The build fails loudly on missing/empty meta or output-path collisions.
 *
 * NOTE: public/CNAME → dist/CNAME (the custom-domain mechanism) is copied by
 * vite itself; do not touch it here. Same for public/api/ and public/robots.txt.
 */
import { existsSync, mkdirSync, readFileSync, statSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import {
  ORIGIN,
  SITE_ROUTES,
  learnHomeMeta,
  moduleMeta,
  moduleTaskMeta,
} from '../src/routeMeta.js';
import { loadContent } from './contentLoader.mjs';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const dist = join(root, 'dist');

// The link-preview card: public/img/ogimage.png, 1200×630 (1.91:1) branded
// jelly art — matches twitter:card summary_large_image.
const OG_IMAGE = `${ORIGIN}/img/ogimage.png`;
const OG_IMAGE_WIDTH = 1200;
const OG_IMAGE_HEIGHT = 630;
const OG_IMAGE_ALT = 'GPU.js jellyfish logo — GPU accelerated JavaScript';
const SITE_NAME = 'GPU.js';

function fail(message) {
  console.error(`prerender: FATAL — ${message}`);
  process.exit(1);
}

// ---- load course content ---------------------------------------------------
//
// Identity, ordering and validation all come from the shared registry: modules
// arrive in canonical order (each track's, in teaching order, then the modules
// in no track by title) already carrying uuid/version/slug/shortId. Invalid
// content throws here with every problem listed.

async function loadModules() {
  try {
    return (await loadContent(root)).modules;
  } catch (e) {
    fail(e.message);
    return [];
  }
}

// ---- head construction -----------------------------------------------------

function esc(value) {
  return String(value)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function headBlock({ title, description, path }, jsonLd) {
  const url = ORIGIN + path;
  const lines = [
    `<title>${esc(title)}</title>`,
    `<meta name="description" content="${esc(description)}">`,
    `<link rel="canonical" href="${esc(url)}">`,
    `<meta property="og:type" content="website">`,
    `<meta property="og:site_name" content="${esc(SITE_NAME)}">`,
    `<meta property="og:url" content="${esc(url)}">`,
    `<meta property="og:title" content="${esc(title)}">`,
    `<meta property="og:description" content="${esc(description)}">`,
    `<meta property="og:image" content="${esc(OG_IMAGE)}">`,
    `<meta property="og:image:width" content="${OG_IMAGE_WIDTH}">`,
    `<meta property="og:image:height" content="${OG_IMAGE_HEIGHT}">`,
    `<meta property="og:image:alt" content="${esc(OG_IMAGE_ALT)}">`,
    `<meta name="twitter:card" content="summary_large_image">`,
    `<meta name="twitter:title" content="${esc(title)}">`,
    `<meta name="twitter:description" content="${esc(description)}">`,
    `<meta name="twitter:image" content="${esc(OG_IMAGE)}">`,
  ];
  if (jsonLd) {
    // JSON-LD may not contain a literal "</script" — JSON.stringify keeps "/"
    // unescaped, so guard by escaping it inside string values.
    const json = JSON.stringify(jsonLd).replace(/<\//g, '<\\/');
    lines.push(`<script type="application/ld+json">${json}</script>`);
  }
  return lines.map(line => `    ${line}`).join('\n');
}

// ---- JSON-LD builders ------------------------------------------------------

function courseRef() {
  const learn = learnHomeMeta();
  return { '@type': 'Course', name: learn.title, url: ORIGIN + learn.path };
}

function courseLd(modules) {
  const learn = learnHomeMeta();
  return {
    '@context': 'https://schema.org',
    '@type': 'Course',
    name: learn.title,
    description: learn.description,
    url: ORIGIN + learn.path,
    provider: { '@type': 'Organization', name: SITE_NAME, url: ORIGIN },
    isAccessibleForFree: true,
    offers: [{ '@type': 'Offer', category: 'Free', price: 0, priceCurrency: 'USD' }],
    hasCourseInstance: [
      {
        '@type': 'CourseInstance',
        courseMode: 'Online',
        courseWorkload: 'PT10H',
      },
    ],
    hasPart: modules.map(mod => {
      const meta = moduleMeta(mod);
      return {
        '@type': 'LearningResource',
        name: mod.title,
        url: ORIGIN + meta.path,
      };
    }),
  };
}

function moduleLd(mod) {
  const meta = moduleMeta(mod);
  return {
    '@context': 'https://schema.org',
    '@type': 'LearningResource',
    name: mod.title,
    description: meta.description,
    url: ORIGIN + meta.path,
    learningResourceType: 'Course module',
    isAccessibleForFree: true,
    isPartOf: courseRef(),
  };
}

function taskLd(mod, task, taskNum, meta) {
  const modMeta = moduleMeta(mod);
  return {
    '@context': 'https://schema.org',
    '@type': 'LearningResource',
    name: task.title,
    description: meta.description,
    url: ORIGIN + meta.path,
    position: taskNum,
    learningResourceType: 'Exercise',
    isAccessibleForFree: true,
    isPartOf: {
      '@type': 'LearningResource',
      name: mod.title,
      url: ORIGIN + modMeta.path,
      isPartOf: courseRef(),
    },
  };
}

// ---- main ------------------------------------------------------------------

const shellPath = join(dist, 'index.html');
if (!existsSync(shellPath)) fail('dist/index.html not found — run `vite build` first');
const shell = readFileSync(shellPath, 'utf8');
if (!shell.includes('</head>') || !shell.includes('id="root"')) {
  fail('dist/index.html does not look like the app shell');
}

// Strip the page-specific head tags the template ships with; each output page
// gets its own complete set injected.
const template = shell
  .replace(/[ \t]*<title>[\s\S]*?<\/title>[ \t]*\n?/g, '')
  .replace(/[ \t]*<meta name="description"[^>]*>[ \t]*\n?/g, '')
  .replace(/[ \t]*<meta property="og:[^>]*>[ \t]*\n?/g, '')
  .replace(/[ \t]*<meta name="twitter:[^>]*>[ \t]*\n?/g, '')
  .replace(/[ \t]*<link rel="canonical"[^>]*>[ \t]*\n?/g, '')
  .replace(/[ \t]*<script type="application\/ld\+json">[\s\S]*?<\/script>[ \t]*\n?/g, '');
if (/og:|twitter:|rel="canonical"|<title>/.test(template)) {
  fail('template still contains page-specific head tags after stripping');
}

function renderPage(block) {
  return template.replace('</head>', `${block}\n  </head>`);
}

const modules = await loadModules();

const pages = [];
const seenPaths = new Set();
function addPage(meta, jsonLd) {
  const { path, title, description } = meta || {};
  if (!path || !/^\/[a-z0-9\-/]*$/i.test(path)) fail(`bad route path: ${JSON.stringify(path)}`);
  if (!title || !String(title).trim()) fail(`empty title for ${path}`);
  if (!description || !String(description).trim()) fail(`empty description for ${path}`);
  if (seenPaths.has(path)) fail(`duplicate route path: ${path}`);
  seenPaths.add(path);
  pages.push({ path, title, description, jsonLd });
}

for (const route of SITE_ROUTES) addPage(route);
addPage(learnHomeMeta(), courseLd(modules));
for (const mod of modules) {
  addPage(moduleMeta(mod), moduleLd(mod));
  mod.tasks.forEach((task, i) => {
    const meta = moduleTaskMeta(mod, task, i + 1);
    addPage(meta, taskLd(mod, task, i + 1, meta));
  });
}

// Write every page: '/' enriches dist/index.html in place, everything else
// goes to BOTH dist/<path>.html and dist/<path>/index.html. The extensionless
// twin makes GitHub Pages serve the canonical slashless URL (/learn) with a
// direct 200 — without it, GH 301-redirects /learn → /learn/ with an absolute
// http:// Location when the CDN fetches the origin over plain HTTP, and that
// https→http downgrade hop loops forever in strict webviews (Telegram).
let written = 0;
for (const page of pages) {
  const html = renderPage(headBlock(page, page.jsonLd));
  const outDir = page.path === '/' ? dist : join(dist, page.path.slice(1));
  if (existsSync(outDir) && !statSync(outDir).isDirectory()) {
    fail(`output path collides with an existing file: ${outDir}`);
  }
  mkdirSync(outDir, { recursive: true });
  writeFileSync(join(outDir, 'index.html'), html);
  if (page.path !== '/') {
    writeFileSync(join(dist, `${page.path.slice(1)}.html`), html);
  }
  written++;
}

// 404.html — app shell fallback for unknown paths (GH Pages serves it with a
// 404 status). Generic title, noindex, no canonical/OG: it must never be
// treated as a real page, but it still boots the SPA for deep routes.
writeFileSync(
  join(dist, '404.html'),
  renderPage(
    [
      `<title>${esc(SITE_NAME)}</title>`,
      `<meta name="robots" content="noindex">`,
    ].map(line => `    ${line}`).join('\n')
  )
);

// sitemap.xml — exactly the prerendered canonical URLs, nothing else.
const sitemap =
  `<?xml version="1.0" encoding="UTF-8"?>\n` +
  `<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n` +
  pages.map(page => `  <url><loc>${esc(ORIGIN + page.path)}</loc></url>\n`).join('') +
  `</urlset>\n`;
writeFileSync(join(dist, 'sitemap.xml'), sitemap);

const taskCount = modules.reduce((sum, mod) => sum + mod.tasks.length, 0);
console.log(
  `prerender: wrote ${written} pages (${SITE_ROUTES.length} site + 1 learn + ` +
    `${modules.length} modules + ${taskCount} tasks), 404.html (noindex), ` +
    `sitemap.xml (${pages.length} URLs)`
);
