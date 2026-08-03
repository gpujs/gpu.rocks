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
import { createHash } from 'node:crypto';
import { existsSync, mkdirSync, readFileSync, readdirSync, statSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import {
  ORIGIN,
  SITE_ROUTES,
  learnHomeMeta,
  moduleMeta,
  moduleTaskMeta,
} from '../src/routeMeta.js';
import poster from '../src/Bench/poster.js';
import { loadContent } from './contentLoader.mjs';
import {
  STATIC_STYLE,
  buildLlmsFull,
  buildLlmsTxt,
  loadFigureMeta,
  pageMarkdown,
  proseHtml,
} from './staticContent.mjs';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const dist = join(root, 'dist');

// Two link-preview cards, because /learn is a different product from the rest
// of the site: the jelly says "this is gpu.js", the course card says "this is a
// free course you can start now", and a shared /learn link should say the
// second thing. Chosen per page by content kind, not by path matching, so a new
// learn route cannot forget to opt in.
const OG_SITE = {
  url: `${ORIGIN}/img/ogimage.png`,
  width: 1200,
  height: 630,
  alt: 'GPU.js jellyfish logo — GPU accelerated JavaScript',
};
const OG_LEARN = {
  // pngcrush -brute -rem alla -reduce: 1,592,714 -> 1,161,798 bytes (27%), and
  // pixel-identical (the alpha channel was fully opaque, so -reduce dropping it
  // costs nothing — verified over all 1,070,848 pixels).
  url: `${ORIGIN}/img/oglearn.png`,
  width: 1424,
  height: 752,
  alt:
    'Learn GPGPU in your browser — a free, hands-on course built on gpu.js. Six ' +
    'panels, one per track: GPGPU 101, Parallel Primitives, Math & Simulation, ' +
    'Computer Vision, Signal Processing and Computational Graphics.',
};

// A third card, for the benchmark. Unlike the other two it is GENERATED — by
// scripts/bench-infographic.mjs, from the saved run — so it cannot drift from
// the numbers the page reports. Its ?v= is a content hash for the same reason
// the poster's is: the URL is stable while the bytes change, and the deploy
// purges HTML rather than assets.
const OG_BENCH = {
  url: `${ORIGIN}${poster.og}`,
  width: poster.ogWidth,
  height: poster.ogHeight,
  alt:
    'The Benchmark Gauntlet — the ten fastest of thirty GPGPU workloads, each ' +
    'as a bar showing how many times faster WebGPU through gpu.js runs it than ' +
    `plain JavaScript, measured on ${poster.machine}.`,
};

function ogImageFor(page) {
  if (page.path === '/benchmark') return OG_BENCH;
  return page.content && page.content.kind !== 'site' ? OG_LEARN : OG_SITE;
}
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

async function loadRegistry() {
  try {
    return await loadContent(root);
  } catch (e) {
    fail(e.message);
    return null;
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

function headBlock(page, jsonLd) {
  const { title, description, path } = page;
  const image = ogImageFor(page);
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
    `<meta property="og:image" content="${esc(image.url)}">`,
    `<meta property="og:image:width" content="${image.width}">`,
    `<meta property="og:image:height" content="${image.height}">`,
    `<meta property="og:image:alt" content="${esc(image.alt)}">`,
    `<meta name="twitter:card" content="summary_large_image">`,
    `<meta name="twitter:title" content="${esc(title)}">`,
    `<meta name="twitter:description" content="${esc(description)}">`,
    `<meta name="twitter:image" content="${esc(image.url)}">`,
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

const ROOT_DIV = /<div id="root">\s*<\/div>/;
if (!ROOT_DIV.test(template)) fail('could not find an empty <div id="root"> to fill');

// Each page gets its head tags AND a static prose body inside #root. The
// client uses createRoot (src/index.jsx), which discards container children on
// first render, so the prose is what a crawler or a JS-less visitor reads and
// what a slow connection shows while the bundle arrives — it is never
// hydrated. See scripts/staticContent.mjs for why this is not React SSR.
function renderPage(block, prose) {
  const html = template.replace('</head>', `${block}\n  </head>`);
  return prose ? html.replace(ROOT_DIV, `<div id="root">${prose}</div>`) : html;
}

const registry = await loadRegistry();
const modules = registry.modules;
const tracks = registry.tracks;
const orphans = registry.orphanModules || [];
const figureMeta = await loadFigureMeta(root);
const staticCtx = { figureMeta, siteRoutes: SITE_ROUTES };

const pages = [];
const seenPaths = new Set();
function addPage(meta, jsonLd, content) {
  const { path, title, description } = meta || {};
  if (!path || !/^\/[a-z0-9\-/]*$/i.test(path)) fail(`bad route path: ${JSON.stringify(path)}`);
  if (!title || !String(title).trim()) fail(`empty title for ${path}`);
  if (!description || !String(description).trim()) fail(`empty description for ${path}`);
  if (seenPaths.has(path)) fail(`duplicate route path: ${path}`);
  seenPaths.add(path);
  pages.push({ path, title, description, jsonLd, content: content || { kind: 'site' } });
}

for (const route of SITE_ROUTES) addPage(route);
const learnMeta = learnHomeMeta();
addPage(learnMeta, courseLd(modules), {
  kind: 'learn-home',
  tracks,
  orphans,
  meta: learnMeta,
});
for (const mod of modules) {
  addPage(moduleMeta(mod), moduleLd(mod), { kind: 'module', module: mod });
  mod.tasks.forEach((task, i) => {
    const meta = moduleTaskMeta(mod, task, i + 1);
    addPage(meta, taskLd(mod, task, i + 1, meta), {
      kind: 'task',
      module: mod,
      task,
      step: i + 1,
    });
  });
}

// Write every page: '/' enriches dist/index.html in place, everything else
// goes to BOTH dist/<path>.html and dist/<path>/index.html. The extensionless
// twin makes GitHub Pages serve the canonical slashless URL (/learn) with a
// direct 200 — without it, GH 301-redirects /learn → /learn/ with an absolute
// http:// Location when the CDN fetches the origin over plain HTTP, and that
// https→http downgrade hop loops forever in strict webviews (Telegram).
let written = 0;
let markdownBytes = 0;
for (const page of pages) {
  // Every page advertises its Markdown twin, so a client that prefers text
  // can find it from the HTML without knowing the convention.
  const mdPath = (page.path === '/' ? '/index' : page.path) + '.md';
  const head =
    headBlock(page, page.jsonLd) +
    `\n    <link rel="alternate" type="text/markdown" href="${esc(ORIGIN + mdPath)}">` +
    `\n    ${STATIC_STYLE}`;
  const html = renderPage(head, proseHtml(page, staticCtx));
  const outDir = page.path === '/' ? dist : join(dist, page.path.slice(1));
  if (existsSync(outDir) && !statSync(outDir).isDirectory()) {
    fail(`output path collides with an existing file: ${outDir}`);
  }
  mkdirSync(outDir, { recursive: true });
  writeFileSync(join(outDir, 'index.html'), html);
  if (page.path !== '/') {
    writeFileSync(join(dist, `${page.path.slice(1)}.html`), html);
  }

  const markdown = pageMarkdown(page, staticCtx);
  const mdFile = join(dist, mdPath.slice(1));
  mkdirSync(dirname(mdFile), { recursive: true });
  writeFileSync(mdFile, markdown);
  markdownBytes += Buffer.byteLength(markdown);
  written++;
}

// llms.txt — the llmstxt.org index — and llms-full.txt, the whole site as one
// document. Both are Markdown, both point at the .md twins written above.
writeFileSync(
  join(dist, 'llms.txt'),
  buildLlmsTxt({ siteRoutes: SITE_ROUTES, tracks, orphans, learnMeta })
);
const llmsFull = buildLlmsFull(pages, staticCtx);
writeFileSync(join(dist, 'llms-full.txt'), llmsFull);

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

// version.json — what a running tab polls to notice it has gone stale.
//
// Every deploy replaces the whole gh-pages tree, so a tab left open since the
// last deploy is holding filenames that no longer exist: the moment it lazily
// loads a chunk it 404s. The id is derived from the built asset NAMES, which
// are content hashes, so it changes exactly when the code does and two
// identical builds stay byte-identical (CI checks that).
const assetNames = readdirSync(join(dist, 'assets')).sort();
const buildId = createHash('sha256').update(assetNames.join('\n')).digest('hex').slice(0, 16);
writeFileSync(join(dist, 'version.json'), `${JSON.stringify({ build: buildId }, null, 2)}\n`);

const taskCount = modules.reduce((sum, mod) => sum + mod.tasks.length, 0);
const kb = bytes => `${Math.round(bytes / 1024)} KB`;
console.log(
  `prerender: wrote ${written} pages (${SITE_ROUTES.length} site + 1 learn + ` +
    `${modules.length} modules + ${taskCount} tasks), 404.html (noindex), ` +
    `sitemap.xml (${pages.length} URLs), version.json (build ${buildId})`
);
console.log(
  `prerender: static prose in every #root, ${written} .md twins (${kb(markdownBytes)}), ` +
    `llms.txt, llms-full.txt (${kb(Buffer.byteLength(llmsFull))})`
);
