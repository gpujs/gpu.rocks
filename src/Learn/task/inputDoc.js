// inputDoc.js — describes a task's injected inputs from the REAL values.
//
// Every task hands the sandbox a bag of globals via `task.inputs(utils)`
// (engine/sandbox.js injects them by name). Learners used to be told only that
// a name was "injected as a global for this task", which says nothing about the
// one thing they need: its shape. Everything needed to say more is already in
// the value, so nothing is authored per task — every task in the course
// describes itself, with no edits to any content module.
//
// A descriptor is:
//   { name, type, summary, sample, note }
//     type    monospace shape line, e.g. 'number[64]' / 'number[16][16]' /
//             'ImageData 512×512'
//     summary one prose clause: how many, what range
//     sample  monospace excerpt of the real data ('' when there is nothing
//             worth showing)
//     note    optional sentence about indexing (2D+ arrays and images), kept
//             consistent with content/layoutNote.js — the row-major gotcha
//
// COST. These run on the main thread while the learner reads the brief, and
// one input can be a 512×512 image, so:
//   * nothing is ever stringified wholesale — samples are built from a handful
//     of leading elements;
//   * a range costs one pass over at most SCAN_BUDGET values, strided so it
//     still spans the whole array rather than just its head; beyond that cap
//     the numbers are labelled "(sampled)" instead of quietly guessing;
//   * ImageData pixels are never walked at all (dimensions come from the
//     object, channels are 0–1 by definition);
//   * every descriptor is computed once per task and memoized (taskInputDocs),
//     because building the inputs themselves is the expensive part;
//   * nothing here may throw: a weird value degrades to a plain type name.

// Values visited when computing a range. A stride keeps the whole array
// represented without walking all of it; the budget is set above the largest
// input the course actually ships (65,536 numbers, ~0.3 ms) so every real
// description is exact, and the stride only ever engages for something
// pathological. When it does engage, the range is approximate and the
// exact-values phrasing ("each 0 or 1") is withheld — see rangeClause.
const SCAN_BUDGET = 65536;
const SAMPLE_COUNT = 3; // leading values shown in `sample`
const MAX_SAMPLE_CHARS = 64;
const MAX_DEPTH = 4; // deeper nesting is reported as an opaque array

// ---- primitives ------------------------------------------------------------

function isArrayLike(v) {
  return Array.isArray(v) || ArrayBuffer.isView(v);
}

function isImageData(v) {
  return typeof ImageData !== 'undefined' && v instanceof ImageData;
}

// Compact, readable numbers: integers stay integers, floats keep about three
// significant decimals, extremes go exponential. 0.32363921568627446 → 0.324.
export function fmtNum(v) {
  if (typeof v !== 'number') return String(v);
  if (Number.isNaN(v)) return 'NaN';
  if (!Number.isFinite(v)) return v > 0 ? 'Infinity' : '-Infinity';
  if (Number.isInteger(v)) return Math.abs(v) < 1e9 ? String(v) : v.toExponential(2);
  const abs = Math.abs(v);
  if (abs < 1e-4 || abs >= 1e7) return v.toExponential(2);
  const digits = abs < 1 ? 3 : abs < 100 ? 2 : 1;
  return String(Number(v.toFixed(digits)));
}

// one element, for a sample list — never recurses far enough to print a whole
// nested structure
function fmtLeaf(v, depth = 0) {
  if (v === null) return 'null';
  if (v === undefined) return 'undefined';
  const t = typeof v;
  if (t === 'number') return fmtNum(v);
  if (t === 'boolean') return String(v);
  if (t === 'string') return JSON.stringify(v.length > 12 ? `${v.slice(0, 12)}…` : v);
  if (t === 'function') return 'ƒ';
  if (isArrayLike(v)) {
    if (depth >= 1) return `[…${v.length}]`;
    const head = [];
    for (let i = 0; i < Math.min(v.length, SAMPLE_COUNT); i++) head.push(fmtLeaf(v[i], depth + 1));
    return `[${head.join(', ')}${v.length > SAMPLE_COUNT ? ', …' : ''}]`;
  }
  return '{…}';
}

function sampleList(arr) {
  const head = [];
  for (let i = 0; i < Math.min(arr.length, SAMPLE_COUNT); i++) head.push(fmtLeaf(arr[i]));
  let text = `[${head.join(', ')}${arr.length > SAMPLE_COUNT ? ', …' : ''}]`;
  if (text.length > MAX_SAMPLE_CHARS) text = `${text.slice(0, MAX_SAMPLE_CHARS - 2)}…]`;
  return text;
}

// ---- shape + range ---------------------------------------------------------

// Dimensions from the FIRST element chain (cheap), plus a bounded raggedness
// check on the outermost level so `number[16][16]` is only claimed when the
// rows really do agree.
function arrayShape(value) {
  const dims = [];
  let cur = value;
  let depth = 0;
  while (isArrayLike(cur) && depth < MAX_DEPTH) {
    dims.push(cur.length);
    if (cur.length === 0) return { dims, leaf: 'empty', ragged: false };
    cur = cur[0];
    depth++;
  }
  const leaf = isArrayLike(cur)
    ? 'array' // deeper than MAX_DEPTH
    : cur === null
      ? 'null'
      : typeof cur;
  let ragged = false;
  if (dims.length > 1) {
    const rows = value.length;
    const stride = Math.max(1, Math.ceil(rows / 256));
    for (let i = 0; i < rows; i += stride) {
      const row = value[i];
      if (!isArrayLike(row) || row.length !== dims[1]) {
        ragged = true;
        break;
      }
    }
  }
  return { dims, leaf, ragged };
}

// Stride-sampled min/max plus the first few distinct values. Bails out of the
// distinct set as soon as it holds more than two — that set only exists to
// recognise masks (0/1 grids), where "each 0 or 1" beats "0 … 1".
function scanRange(value, dims) {
  const total = dims.reduce((a, b) => a * b, 1);
  const stride = Math.max(1, Math.ceil(total / SCAN_BUDGET));
  let min = Infinity;
  let max = -Infinity;
  let seen = 0;
  let numeric = true;
  let hasNaN = false;
  const distinct = new Set();
  let distinctOverflow = false;

  const visit = v => {
    if (typeof v !== 'number') {
      numeric = false;
      return;
    }
    // a NaN still means "this is numeric data" — it just can't be ranged, and
    // saying so beats degrading the whole array to an untyped "3 values"
    if (Number.isNaN(v)) {
      hasNaN = true;
      seen++;
      return;
    }
    seen++;
    if (v < min) min = v;
    if (v > max) max = v;
    if (!distinctOverflow) {
      distinct.add(v);
      if (distinct.size > 2) distinctOverflow = true;
    }
  };

  if (dims.length === 1) {
    for (let i = 0; i < value.length && numeric; i += stride) visit(value[i]);
  } else if (dims.length === 2) {
    // spread the stride over rows first, then within a row, so a tall-thin and
    // a short-wide array are both sampled across their whole extent
    const rowStride = Math.max(1, Math.ceil(dims[0] / Math.max(1, Math.floor(SCAN_BUDGET / dims[1]))));
    const colStride = Math.max(1, Math.ceil(stride / rowStride));
    for (let y = 0; y < value.length && numeric; y += rowStride) {
      const row = value[y];
      if (!isArrayLike(row)) {
        numeric = false;
        break;
      }
      for (let x = 0; x < row.length && numeric; x += colStride) visit(row[x]);
    }
  } else {
    // 3D+: sample the leading planes only — enough to type the leaves
    let budget = SCAN_BUDGET;
    const walk = (v, depth) => {
      if (budget <= 0 || !numeric) return;
      if (depth === dims.length - 1) {
        for (let i = 0; i < v.length && budget > 0 && numeric; i++, budget--) visit(v[i]);
        return;
      }
      for (let i = 0; i < v.length && budget > 0 && numeric; i++) {
        if (!isArrayLike(v[i])) {
          numeric = false;
          return;
        }
        walk(v[i], depth + 1);
      }
    };
    walk(value, 0);
  }

  if (!numeric || seen === 0) return null;
  return {
    min,
    max,
    hasNaN,
    // only trustworthy when every element was visited
    exhaustive: seen === total,
    distinct: distinctOverflow ? null : [...distinct].sort((a, b) => a - b),
  };
}

// "64 numbers, 0.02 … 9.98" / "256 numbers, each 0 or 1" / "16 numbers, all 1"
function rangeClause(range) {
  if (!range) return null;
  const suffix = range.hasNaN ? ', some NaN' : '';
  if (range.min === Infinity) return 'all NaN'; // nothing finite to range
  // A strided scan (only for inputs bigger than SCAN_BUDGET — nothing the
  // course ships) can miss values, so it may not claim to have seen them all.
  if (!range.exhaustive) {
    return `${fmtNum(range.min)} … ${fmtNum(range.max)} (sampled)${suffix}`;
  }
  const distinct = range.hasNaN ? null : range.distinct;
  if (distinct && distinct.length === 1) return `all ${fmtNum(distinct[0])}`;
  if (distinct && distinct.length === 2) {
    return `each ${fmtNum(distinct[0])} or ${fmtNum(distinct[1])}`;
  }
  if (range.min === range.max) return `all ${fmtNum(range.min)}${suffix}`;
  return `${fmtNum(range.min)} … ${fmtNum(range.max)}${suffix}`;
}

function plural(n, word) {
  return `${n.toLocaleString('en-US')} ${word}${n === 1 ? '' : 's'}`;
}

// ---- the describers --------------------------------------------------------

function describeArray(name, value) {
  const { dims, leaf, ragged } = arrayShape(value);
  const ctor = ArrayBuffer.isView(value) ? value.constructor.name : null;

  if (dims[0] === 0) {
    return { name, type: ctor ? `${ctor}[0]` : 'array[0]', summary: 'empty', sample: '', note: '' };
  }

  // ---- flat ----
  if (dims.length === 1) {
    const range = leaf === 'number' ? scanRange(value, dims) : null;
    const type =
      ctor ? `${ctor}[${dims[0]}]` : leaf === 'number' && range ? `number[${dims[0]}]` : `array[${dims[0]}]`;
    const clause = rangeClause(range);
    const summary = range
      ? `${plural(dims[0], 'number')}, ${clause}`
      : `${plural(dims[0], 'value')}`;
    return { name, type, summary, sample: sampleList(value), note: '' };
  }

  // ---- ragged ----
  if (ragged) {
    return {
      name,
      type: `array[${dims[0]}][…]`,
      summary: `${plural(dims[0], 'row')} of varying length`,
      sample: `row 0: ${sampleList(value[0])}`,
      note: `Indexed ${name}[y][x] — row y, column x.`,
    };
  }

  // ---- rectangular 2D ----
  if (dims.length === 2) {
    const range = leaf === 'number' ? scanRange(value, dims) : null;
    const type = range ? `number[${dims[0]}][${dims[1]}]` : `array[${dims[0]}][${dims[1]}]`;
    const clause = rangeClause(range);
    const size = `${dims[0]} row${dims[0] === 1 ? '' : 's'} × ${dims[1]} column${dims[1] === 1 ? '' : 's'}`;
    const summary = range ? `${size} of numbers, ${clause}` : size;
    return {
      name,
      type,
      summary,
      sample: `row 0: ${sampleList(value[0])}`,
      note: `Indexed ${name}[y][x] — row y, column x.`,
    };
  }

  // ---- 3D+ ----
  const range = leaf === 'number' ? scanRange(value, dims) : null;
  const shape = dims.join('][');
  const clause = rangeClause(range);
  return {
    name,
    type: range ? `number[${shape}]` : `array[${shape}]`,
    summary: `${dims.join(' × ')}${range ? ` numbers, ${clause}` : ' nested values'}`,
    sample: '',
    // gpu.js only defines the [z][y][x] convention up to three dimensions —
    // deeper than that, say nothing rather than say something wrong
    note: dims.length === 3 ? `Indexed ${name}[z][y][x] — depth z, row y, column x.` : '',
  };
}

// The course's images. Deliberately says the same thing content/layoutNote.js
// says, because this is exactly the confusion that callout exists for.
function describeImageData(name, value) {
  const { width, height } = value;
  let sample = '';
  try {
    // makeTestImage attaches the non-enumerable `plain` nested array; one pixel
    // is one cheap read, and an anchor beats any amount of prose
    const px = value.plain && value.plain[0] && value.plain[0][0];
    if (Array.isArray(px)) {
      sample = `${name}[0][0] = [${px.map(fmtNum).join(', ')}]`;
    }
  } catch (e) {
    sample = '';
  }
  return {
    name,
    type: `ImageData ${width}×${height}`,
    summary: `${plural(width * height, 'RGBA pixel')}`,
    sample,
    note:
      `Inside a kernel ${name}[y][x] is the pixel in row y, column x — ` +
      'an [r, g, b, a] array with channels from 0 to 1.',
  };
}

function describeObject(name, value) {
  const ctor =
    value.constructor && value.constructor.name && value.constructor.name !== 'Object'
      ? value.constructor.name
      : null;
  if (ctor) return { name, type: ctor, summary: `a ${ctor}`, sample: '', note: '' };
  let keys = [];
  try {
    keys = Object.keys(value);
  } catch (e) {
    keys = [];
  }
  const shown = keys.slice(0, 6).join(', ');
  return {
    name,
    type: 'object',
    summary: keys.length ? `keys: ${shown}${keys.length > 6 ? ', …' : ''}` : 'no keys',
    sample: '',
    note: '',
  };
}

/**
 * describeInput(name, value) → { name, type, summary, sample, note }
 * Never throws: an unrecognisable value still gets a usable type line.
 */
export function describeInput(name, value) {
  try {
    if (value === null) return { name, type: 'null', summary: 'no value', sample: '', note: '' };
    if (value === undefined) {
      return { name, type: 'undefined', summary: 'no value', sample: '', note: '' };
    }
    const t = typeof value;
    if (t === 'number') {
      return { name, type: 'number', summary: `the number ${fmtNum(value)}`, sample: '', note: '' };
    }
    if (t === 'boolean') return { name, type: 'boolean', summary: String(value), sample: '', note: '' };
    if (t === 'bigint') return { name, type: 'bigint', summary: String(value), sample: '', note: '' };
    if (t === 'string') {
      return {
        name,
        type: 'string',
        summary: plural(value.length, 'character'),
        sample: JSON.stringify(value.length > 40 ? `${value.slice(0, 40)}…` : value),
        note: '',
      };
    }
    if (t === 'function') {
      return {
        name,
        type: 'function',
        summary: `${value.name ? `ƒ ${value.name}` : 'an anonymous function'}, ${plural(value.length, 'argument')}`,
        sample: '',
        note: '',
      };
    }
    if (isImageData(value)) return describeImageData(name, value);
    if (isArrayLike(value)) return describeArray(name, value);
    if (t === 'object') return describeObject(name, value);
    return { name, type: t, summary: `a ${t}`, sample: '', note: '' };
  } catch (e) {
    // introspection must never take the page down with it
    return {
      name,
      type: 'value',
      summary: 'an injected global whose value could not be inspected',
      sample: '',
      note: '',
    };
  }
}

// ---- per-task descriptors (memoized) ---------------------------------------

// Keyed by the task object, which the registry creates once per module load, so
// this survives remounts and re-renders — the descriptors are built once, not
// per keystroke. WeakMap so an unloaded module is still collectable.
const cache = new WeakMap();

// the exact filter engine/sandbox.js applies before injecting a global, so the
// brief can never advertise a name the sandbox does not actually define
const IDENT_RE = /^[A-Za-z_$][A-Za-z0-9_$]*$/;

/**
 * taskInputDocs(task, utils) → descriptor[] (empty when the task has no
 * inputs, or when building them throws — the run itself reports that failure).
 *
 * OPTIONAL AUTHORED OVERRIDE. A task may carry `inputNotes: { <name>: 'html-
 * free sentence' }`; when present it REPLACES the derived `note` for that
 * input. Nothing in the course needs it today — every input describes itself —
 * so it exists only so a future task with genuinely surprising semantics can
 * add one sentence without anyone hand-writing 75 descriptions.
 */
export function taskInputDocs(task, utils) {
  if (!task) return [];
  const hit = cache.get(task);
  if (hit) return hit;
  let docs = [];
  try {
    if (typeof task.inputs === 'function') {
      const inputs = task.inputs(utils) || {};
      docs = Object.entries(inputs)
        .filter(([name]) => IDENT_RE.test(name))
        .map(([name, value]) => {
          const doc = describeInput(name, value);
          const authored = task.inputNotes && task.inputNotes[name];
          return authored ? { ...doc, note: authored } : doc;
        });
    }
  } catch (e) {
    docs = [];
  }
  cache.set(task, docs);
  return docs;
}

// ---- shared rendering ------------------------------------------------------

/** One-line plain text, for aria labels and tests. */
export function inputDocText(doc) {
  return [`${doc.name}: ${doc.type}`, doc.summary, doc.sample, doc.note]
    .filter(Boolean)
    .join(' — ');
}

/**
 * The editor's rendering of a descriptor: a monospace `name: type` header over
 * the prose, the same flyout shape gpujsApi entries use (see
 * completion/completions.js infoNode). Shared by the completion info panel and
 * the hover tooltip so the editor has one visual language.
 */
export function inputDocDom(doc, extraClass = '') {
  const dom = document.createElement('div');
  dom.className = `cm-learn-info${extraClass ? ` ${extraClass}` : ''}`;
  const sig = document.createElement('div');
  sig.className = 'sig';
  sig.textContent = `${doc.name}: ${doc.type}`;
  dom.appendChild(sig);
  const body = document.createElement('div');
  body.className = 'doc';
  body.textContent = doc.summary;
  if (doc.sample) {
    body.appendChild(document.createTextNode(' '));
    const code = document.createElement('code');
    code.textContent = doc.sample;
    body.appendChild(code);
  }
  dom.appendChild(body);
  if (doc.note) {
    const note = document.createElement('div');
    note.className = 'input-note';
    note.textContent = doc.note;
    dom.appendChild(note);
  }
  return dom;
}
