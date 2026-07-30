// task/highlightCode.js — static syntax highlighting for brief-pane code
// blocks. No CodeMirror instance: the block's text is parsed once with the
// Lezer JS parser and rebuilt as <span class="tok-*"> runs (classHighlighter's
// stable class names), which _task.scss maps onto the --ed-* editor tokens so
// both themes just work. Inline <code> (no <pre> parent) is untouched — it
// keeps the chip style.

import { javascriptLanguage } from '@codemirror/lang-javascript';
import { highlightTree, classHighlighter } from '@lezer/highlight';

/**
 * Syntax-highlight every <pre><code> block inside `root` (a brief-pane DOM
 * element). Idempotent: blocks already processed (data-highlighted) are
 * skipped, so re-running the effect after a re-render is free.
 */
export function highlightCodeBlocks(root) {
  if (!root) return;
  const blocks = root.querySelectorAll('pre > code');
  for (const code of blocks) {
    if (code.dataset.highlighted) continue;
    const text = code.textContent;
    const tree = javascriptLanguage.parser.parse(text);
    const frag = document.createDocumentFragment();
    let pos = 0;
    highlightTree(tree, classHighlighter, (from, to, classes) => {
      if (from > pos) frag.appendChild(document.createTextNode(text.slice(pos, from)));
      const span = document.createElement('span');
      span.className = classes;
      span.textContent = text.slice(from, to);
      frag.appendChild(span);
      pos = to;
    });
    if (pos < text.length) frag.appendChild(document.createTextNode(text.slice(pos)));
    code.textContent = '';
    code.appendChild(frag);
    code.dataset.highlighted = 'true';
  }
}

export default highlightCodeBlocks;
