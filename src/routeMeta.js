// routeMeta.js — SINGLE SOURCE OF TRUTH for per-route page metadata.
//
// Every route's {title, description, path} lives here, consumed by:
//   1. the SPA at runtime (src/pageMeta.js applies it on history navigation);
//   2. the prerender build script, which bakes the same values into static
//      HTML (plus OG/Twitter tags) for each route.
//
// KEEP THIS FILE PLAIN ESM, IMPORTABLE FROM NODE: no JSX, no DOM access, no
// vite-only syntax (import.meta.glob lives in Learn/content/index.js — never
// import that from here). Functions that need course content take the module/
// task objects as ARGUMENTS; a node script enumerates them with
// scripts/contentLoader.mjs. Learn/content/registry.js, imported below for
// url building, is the node-safe half of the registry and safe to pull in.

import { moduleUrl, taskUrl } from './Learn/content/registry.js';

export const ORIGIN = 'https://gpu.rocks';

// ---- site pages (Components/*) --------------------------------------------

const HOME = {
  path: '/',
  title: 'GPU.js — GPU accelerated JavaScript',
  description:
    'GPGPU operations using pure JavaScript. gpu.js compiles your JavaScript ' +
    'functions into shader code and runs them on your GPU — with a CPU fallback.',
};

export const SITE_ROUTES = [
  HOME,
  {
    path: '/benchmark',
    title: 'Benchmark — GPU.js',
    description:
      'Thirty GPGPU workloads timed in your browser on every backend gpu.js can ' +
      'reach — WebGPU, WebGL2, WebGL, WebAssembly and CPU — against hand-written ' +
      'implementations with no gpu.js in them. Every answer is checked against a ' +
      'plain-JavaScript oracle before it is timed.',
  },
  {
    path: '/install',
    title: 'Installation — GPU.js',
    description:
      'Install GPU.js from npm, yarn, or a CDN script tag and write your first ' +
      'GPU-accelerated JavaScript kernel in the browser or Node.js.',
  },
  {
    path: '/examples',
    title: 'Examples — GPU.js',
    description:
      'GPU.js examples: matrix multiplication and more GPGPU snippets you can ' +
      'read, run, and benchmark on your own graphics card.',
  },
];

// Meta for a site pathname. Unknown paths render the home page (the SPA's
// catch-all route), so they canonicalize to '/'.
export function siteMeta(pathname) {
  const clean = ('/' + String(pathname || '').replace(/^\/+/, '')).replace(/\/+$/, '') || '/';
  return SITE_ROUTES.find(route => route.path === clean) || HOME;
}

// ---- learn pages (src/Learn) ----------------------------------------------

export function learnHomeMeta() {
  return {
    path: '/learn',
    title: 'Learn GPGPU in your browser — GPU.js Learn',
    description:
      'A free hands-on GPGPU course built on gpu.js: write real kernels in your ' +
      'browser, run them on your own GPU, and learn ideas that transfer to CUDA ' +
      'and WebGPU.',
  };
}

// Meta for /learn/<module-slug>-<shortId>. `module` is the object from a
// course content file (src/Learn/content/modules/<uuid>.js). In the SPA this
// path redirects to the current task; as a prerendered page it stands on its
// own with the module blurb.
export function moduleMeta(module) {
  return {
    path: moduleUrl(module),
    title: `${module.title} — GPU.js Learn`,
    description: truncate(stripHtml(module.blurb || ''), 160),
  };
}

// Meta for /learn/<module-slug>-<shortId>/<step>. `module` and `task` are the
// objects from the course content files; step is the 1-based task position.
export function moduleTaskMeta(module, task, step) {
  return {
    path: taskUrl(module, step),
    title: `${task.title} · ${module.title} — GPU.js Learn`,
    description: truncate(stripHtml(task.goal || task.intro || ''), 160),
  };
}

// ---- helpers (node-safe: no DOM) ------------------------------------------

const ENTITIES = {
  '&amp;': '&',
  '&lt;': '<',
  '&gt;': '>',
  '&quot;': '"',
  '&#39;': "'",
  '&apos;': "'",
  '&nbsp;': ' ',
};

// Turn a trusted in-repo HTML fragment into plain text for a description tag.
export function stripHtml(html) {
  return String(html || '')
    .replace(/<[^>]*>/g, ' ')
    .replace(/&[a-zA-Z#0-9]+;/g, entity => ENTITIES[entity] || ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

export function truncate(text, max) {
  if (text.length <= max) return text;
  const cut = text.slice(0, max - 1);
  const lastSpace = cut.lastIndexOf(' ');
  return (lastSpace > max * 0.6 ? cut.slice(0, lastSpace) : cut) + '…';
}
