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
// task objects as ARGUMENTS; a node script can import the content files
// directly (src/Learn/content/track*/module-*.js and tracks.js are plain ESM).

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
      'Run the GPU.js benchmark in your browser: matrix multiplication on your ' +
      'GPU vs CPU, charted against reference hardware results.',
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

// Meta for /learn/:moduleId. `module` is the object from a course content
// file (src/Learn/content/track*/module-*.js). In the SPA this path redirects
// to task 1; as a prerendered page it stands on its own with the module blurb.
export function moduleMeta(module) {
  return {
    path: `/learn/${module.id}`,
    title: `${module.title} — GPU.js Learn`,
    description: truncate(stripHtml(module.blurb || ''), 160),
  };
}

// Meta for /learn/:moduleId/:taskNum. `module` and `task` are the objects from
// the course content files; taskNum is 1-based. (Module redirect pages
// /learn/:moduleId never settle, so they have no meta.)
export function moduleTaskMeta(module, task, taskNum) {
  return {
    path: `/learn/${module.id}/${taskNum}`,
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
