// starterPreamble.js — the task's injected globals, printed as a comment block
// above the starter code.
//
// WHY GENERATED, NOT AUTHORED. The same shapes are in the brief pane, the hover
// tooltip and the completion flyout, all from task/inputDoc.js descriptors. A
// 76th copy pasted into each of the 75 content files would be the one that goes
// stale the first time an input's seed changes, so this is derived at load time
// from the same descriptors and nothing in content/modules/ changes.
//
// SAFETY. Everything emitted is a `//` line comment, so the code the learner
// runs is byte-identical in behaviour to task.starterCode — the "starter must
// not pass its tests" invariant scripts/verify-learn.mjs enforces is untouched
// (and that harness reads task.starterCode directly anyway). Newlines are
// stripped from every generated fragment: a line comment can only be escaped by
// one, so that is the whole attack surface.
//
// Learners with saved code never see this — TaskPage only reaches for the
// starter when there is nothing saved, and Reset restores exactly this text.

// Wrapping width. The preamble must never be the widest line in the editor —
// it must not be the thing that makes the editor scroll sideways — so it wraps
// to the width the task's OWN starter code already occupies, clamped to a
// readable band. (Across the course the widest starter line runs 59…98
// columns, so the floor is never the binding constraint in practice.)
const MIN_WIDTH = 56;
const MAX_WIDTH = 74;

const HEADING = 'Task inputs (already defined as globals — no setup needed):';

function widthFor(code) {
  let widest = 0;
  for (const line of String(code || '').split('\n')) {
    if (line.length > widest) widest = line.length;
  }
  return Math.max(MIN_WIDTH, Math.min(MAX_WIDTH, widest));
}

function clean(text) {
  return String(text == null ? '' : text).replace(/[\r\n]+/g, ' ').trim();
}

// smallest content a wrapped entry's last line may end up with before words are
// pulled down to join it — "… each 0.5 or" / "1" reads as a mistake
const MIN_TAIL = 12;

// Word wrap. `first` prefixes the opening line, `rest` every continuation, so a
// wrapped entry stays visually hung under its own bullet. A one-word orphan on
// the final line is rebalanced by pulling words down from the line above.
function wrap(text, first, rest, out, limit) {
  const words = clean(text).split(/\s+/).filter(Boolean);
  if (words.length === 0) return;
  const lines = [[]]; // words per line
  let width = first.length;
  for (const word of words) {
    const current = lines[lines.length - 1];
    if (current.length > 0 && width + 1 + word.length > limit) {
      lines.push([word]);
      width = rest.length + word.length;
    } else {
      width += (current.length ? 1 : 0) + word.length;
      current.push(word);
    }
  }
  for (let guard = 0; guard < 8 && lines.length > 1; guard++) {
    const last = lines[lines.length - 1];
    const prev = lines[lines.length - 2];
    if (last.join(' ').length >= MIN_TAIL || prev.length < 2) break;
    last.unshift(prev.pop());
  }
  lines.forEach((line, i) => out.push((i === 0 ? first : rest) + line.join(' ')));
}

/**
 * inputPreamble(docs, width) → a `//` comment block, or '' when the task has no
 * inputs. Two to four lines per input: what it is called and what shape it is,
 * an elided sample of the real values, and the indexing note when there is one.
 * `width` defaults to the widest a preamble is ever allowed to be; callers with
 * the starter code in hand should pass widthFor(starter) instead.
 */
export function inputPreamble(docs, width = MAX_WIDTH) {
  if (!Array.isArray(docs) || docs.length === 0) return '';
  const limit = Math.max(MIN_WIDTH, Math.min(MAX_WIDTH, width));
  const lines = [];
  const notesSeen = new Set();
  wrap(HEADING, '// ', '// ', lines, limit);
  for (const doc of docs) {
    wrap(`${doc.name}: ${doc.type} — ${doc.summary}`, '//   ', '//     ', lines, limit);
    // `preview` is the wide, hard-elided list ("… 58 more"); for an ImageData
    // it is a single pixel, so a megapixel is never printed
    if (doc.preview) wrap(doc.preview, '//     ', '//     ', lines, limit);
    // Four matrices in one task means four copies of "Indexed x[y][x] — row y,
    // column x", which is the same sentence four times. Print each SHAPE of
    // note once; a task mixing an image and a grid still gets both.
    if (doc.note) {
      const shape = clean(doc.note).split(doc.name).join('·');
      if (!notesSeen.has(shape)) {
        notesSeen.add(shape);
        wrap(doc.note, '//     ', '//     ', lines, limit);
      }
    }
  }
  return lines.join('\n');
}

/**
 * starterCodeFor(task, inputDocs) → the starter code TaskPage puts in the
 * editor. The single source for both the initial document and Reset, so the two
 * can never drift apart.
 */
export function starterCodeFor(task, inputDocs) {
  const starter = (task && task.starterCode) || '';
  const preamble = inputPreamble(inputDocs, widthFor(starter));
  return preamble ? `${preamble}\n\n${starter}` : starter;
}
