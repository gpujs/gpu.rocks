// pageMeta.js — applies routeMeta values to the live document during SPA
// navigation (history routing means the server-sent <head> only matches the
// first page loaded). The prerender build bakes the same values statically;
// this keeps them correct as the user navigates.
//
// DOM-only counterpart to src/routeMeta.js — never import this from node.

import { ORIGIN } from './routeMeta';

function upsert(selector, create) {
  let el = document.head.querySelector(selector);
  if (!el) {
    el = create();
    document.head.appendChild(el);
  }
  return el;
}

function upsertMeta(attr, key) {
  return upsert(`meta[${attr}="${key}"]`, () => {
    const el = document.createElement('meta');
    el.setAttribute(attr, key);
    return el;
  });
}

// setPageMeta({ title, description, path }) — updates document.title,
// <meta name=description>, <link rel=canonical> (from ORIGIN + path), and the
// matching og: tags. Singleton tags are created on first use, reused after.
export function setPageMeta({ title, description, path, canonical } = {}) {
  if (title) {
    document.title = title;
    upsertMeta('property', 'og:title').setAttribute('content', title);
  }
  if (description != null) {
    upsertMeta('name', 'description').setAttribute('content', description);
    upsertMeta('property', 'og:description').setAttribute('content', description);
  }
  const href = canonical || (path ? ORIGIN + path : null);
  if (href) {
    upsert('link[rel="canonical"]', () => {
      const el = document.createElement('link');
      el.setAttribute('rel', 'canonical');
      return el;
    }).setAttribute('href', href);
  }
}
