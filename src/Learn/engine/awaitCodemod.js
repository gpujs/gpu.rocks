/**
 * engine/awaitCodemod.js — rewrite a learner's saved program for async mode.
 *
 * The course runs on gpu.js `async` mode, where EVERY kernel call returns a
 * Promise whichever backend wins. Code written before that still says
 * `const r = k(data)`, which now yields a Promise and fails its tests with no
 * clue as to why: the learner's code did not change, the ground under it did.
 *
 * So attempt the rewrite for them. Three rules, because silently mangling
 * someone's work is far worse than leaving it alone:
 *
 *   1. Only touch what is provably a kernel — a name bound from
 *      createKernel/createKernelMap in that same document.
 *   2. SYNTAX-CHECK the result. `await` inside a non-async callback
 *      (`arr.map(x => k(x))`) is a syntax error; rather than try to recognise
 *      every such shape, parse the output and discard it if it broke.
 *   3. The caller keeps the original, always.
 *
 * Deliberately NO dependency on content/, the DOM or localStorage: this file
 * has to be importable from node so it can be tested directly.
 */

// Names bound from a kernel factory: `const blur = gpu.createKernel(...)`,
// including let/var and a bare `x = gpu.createKernel(...)` reassignment.
export function kernelNamesIn(code) {
  const names = new Set();
  const re =
    /(?:const|let|var)?\s*([A-Za-z_$][\w$]*)\s*=\s*[A-Za-z_$][\w$]*\s*\.\s*createKernel(?:Map)?\s*\(/g;
  let m;
  while ((m = re.exec(code))) names.add(m[1]);
  return names;
}

export function syntaxOk(code) {
  try {
    // Learner code runs inside `async () => { … }` (engine/sandbox.js), so it
    // must be checked in exactly that position — `await` at the top level of
    // that body is legal, and illegal in a plain Function body.
    // eslint-disable-next-line no-new-func
    new Function(`"use strict"; return (async () => {\n${code}\n});`);
    return true;
  } catch (e) {
    return false;
  }
}

// Index just past the `)` that closes the `(` at `open`, or -1 if unbalanced.
// Skips over string and template literals so a paren inside one cannot
// unbalance the count; comments are not tracked, which is acceptable because an
// unbalanced paren inside a comment makes the scan bail rather than mis-cut.
function endOfCall(src, open) {
  let depth = 0;
  for (let i = open; i < src.length; i++) {
    const c = src[i];
    if (c === '"' || c === "'" || c === '`') {
      const quote = c;
      i++;
      while (i < src.length && src[i] !== quote) {
        if (src[i] === '\\') i++;
        i++;
      }
      continue;
    }
    if (c === '(') depth++;
    else if (c === ')') {
      depth--;
      if (depth === 0) return i + 1;
    }
  }
  return -1;
}

/**
 * Insert `await` at kernel call sites and on texture readbacks.
 * Returns the rewritten source, or null when nothing needed changing or the
 * rewrite did not parse (both meaning: leave the document exactly as it is).
 */
export function addAwaits(code) {
  const source = String(code == null ? '' : code);
  const names = kernelNamesIn(source);
  if (!names.size) return null;

  // Collect every site first, then apply from the END so earlier offsets stay
  // valid. `await` binds looser than member access, so `await k(a)[0]` means
  // `await ((k(a))[0])` — indexing the PROMISE and awaiting undefined, which
  // parses cleanly and is silently wrong. Any call followed by `[`, `.` or a
  // further call therefore has to be parenthesised: `(await k(a))[0]`.
  const sites = [];
  const add = (start, open, label) => {
    const before = source.slice(0, start);
    if (/\bawait\s+$/.test(before)) return;
    if (/\b(function|class)\s+$/.test(before)) return;
    const end = endOfCall(source, open);
    if (end < 0) return;
    const after = source.slice(end);
    const needsParens = /^\s*[[.(]/.test(after);
    sites.push({ start, end, needsParens, label });
  };

  for (const name of names) {
    const re = new RegExp(`(^|[^\\w$.])(${name})\\s*\\(`, 'g');
    let m;
    while ((m = re.exec(source))) {
      const start = m.index + m[1].length;
      add(start, m.index + m[0].length - 1, name);
    }
  }
  // The course's texture idiom, `r.toArray ? r.toArray() : r` — toArray() is a
  // Promise under the async contract too.
  const toArray = /(^|[^\w$.])([A-Za-z_$][\w$]*)\.toArray\s*\(/g;
  let t;
  while ((t = toArray.exec(source))) {
    const start = t.index + t[1].length;
    add(start, t.index + t[0].length - 1, `${t[2]}.toArray`);
  }

  if (!sites.length) return null;

  // Apply as INSERTIONS, not range replacements. Calls nest — `b(a(2))` has
  // the outer site's range containing the inner's — so replacing ranges makes
  // the two edits fight and one gets dropped. Two insertions per site never
  // conflict with nesting, whatever the depth.
  const edits = [];
  for (const site of sites) {
    edits.push({ pos: site.start, text: site.needsParens ? '(await ' : 'await ', tie: 1 });
    if (site.needsParens) edits.push({ pos: site.end, text: ')', tie: 0 });
  }
  // Right to left so earlier offsets stay valid; at equal positions the
  // closing paren goes in first so it ends up outside a prefix inserted there.
  edits.sort((a, b) => b.pos - a.pos || a.tie - b.tie);

  let out = source;
  for (const edit of edits) {
    out = out.slice(0, edit.pos) + edit.text + out.slice(edit.pos);
  }

  if (out === source) return null;
  return syntaxOk(out) ? out : null;
}
