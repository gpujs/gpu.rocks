// kernelDoc.js — describes what the learner's OWN code produces, by static
// analysis of the editor document.
//
// task/inputDoc.js answers "what is `data`?" from a real value. This answers
// the other half a learner needs before they ever press Run:
//   * `double`  — what kind of kernel is this, and what does calling it give me?
//   * `result`  — what did `const result = double(data)` just put in my hands?
//   * `data`    — inside `createKernel(function (data) { … })`, which argument
//                 is this and what will actually be passed to it?
//
// Everything is derived from the createKernel settings in the document, so it
// is correct the moment the learner types it and needs no run. Descriptors use
// the SAME { name, type, summary, sample, note } shape as inputDoc.js and go
// through the same renderer, so inputs, outputs and arguments read in one voice.
//
// gpu.js semantics encoded here (source of truth: the gpu.js README dimensions
// table, the same one content/layoutNote.js cites):
//   output: [w]        → Float32Array(w),                       result[x]
//   output: [w, h]     → Array(h) of Float32Array(w),            result[y][x]
//   output: [w, h, d]  → Array(d) of Array(h) of Float32Array(w), result[z][y][x]
//   pipeline: true     → a Texture on the GL backend (plain arrays on CPU)
//   graphical: true    → nothing useful; the pixels live on kernel.canvas
// Sizes are given width-first but indexing runs row-first — the inversion the
// array-layout callout exists for, restated here at the point of use.

import { syntaxTree } from '@codemirror/language';

// Kernel-shaped setters we follow, either chained onto the createKernel call or
// applied later as their own statement (`halve.setOutput([n])`).
const SETTERS = {
  setOutput: 'output',
  setGraphical: 'graphical',
  setPipeline: 'pipeline',
  setImmutable: 'immutable',
  setDynamicOutput: 'dynamicOutput',
};

// ---- tiny syntax-tree helpers ----------------------------------------------

function textOf(state, node) {
  return state.sliceDoc(node.from, node.to);
}

// `const a = 1, b = f()` is FLAT in lezer-javascript: VariableDeclaration holds
// (VariableDefinition, Equals, <value>) triples separated by ',' tokens.
function declarators(node) {
  const out = [];
  let name = null;
  let expectValue = false;
  for (let ch = node.firstChild; ch; ch = ch.nextSibling) {
    if (ch.name === 'VariableDefinition') {
      name = ch;
      expectValue = false;
    } else if (ch.name === 'Equals') {
      expectValue = true;
    } else if (expectValue) {
      if (name) out.push({ name, value: ch });
      name = null;
      expectValue = false;
    }
  }
  return out;
}

// the callee of a CallExpression / NewExpression (skipping `new` and the args)
function calleeOf(call) {
  for (let ch = call.firstChild; ch; ch = ch.nextSibling) {
    if (ch.name === 'new' || ch.name === 'ArgList' || ch.name === 'TypeArgList') continue;
    return ch;
  }
  return null;
}

// positional arguments of a CallExpression, skipping the punctuation tokens
function argsOf(call) {
  const list = [];
  for (let ch = call.firstChild; ch; ch = ch.nextSibling) {
    if (ch.name !== 'ArgList') continue;
    for (let a = ch.firstChild; a; a = a.nextSibling) {
      if (a.name === '(' || a.name === ')' || a.name === ',') continue;
      list.push(a);
    }
    break;
  }
  return list;
}

// `gpu.createKernel` → { object: 'gpu', property: 'createKernel' }; a bare
// `double` → { object: null, property: 'double' }
function calleeParts(state, call) {
  const callee = calleeOf(call);
  if (!callee) return null;
  if (callee.name === 'VariableName') {
    return { object: null, property: textOf(state, callee), objectNode: null };
  }
  if (callee.name !== 'MemberExpression') return null;
  const prop = callee.lastChild && callee.lastChild.name === 'PropertyName' ? callee.lastChild : null;
  if (!prop) return null;
  return {
    object: textOf(state, callee.firstChild),
    property: textOf(state, prop),
    objectNode: callee.firstChild,
  };
}

// ---- settings reading -------------------------------------------------------

function readObject(state, obj) {
  const out = {};
  if (!obj || obj.name !== 'ObjectExpression') return out;
  for (let prop = obj.firstChild; prop; prop = prop.nextSibling) {
    if (prop.name !== 'Property') continue;
    const key = prop.firstChild;
    if (!key) continue;
    const name = textOf(state, key).replace(/^['"]|['"]$/g, '');
    let value = key.nextSibling;
    while (value && (value.name === ':' || value.name === 'Equals')) value = value.nextSibling;
    if (value) out[name] = value;
  }
  return out;
}

function readNumber(state, node, consts) {
  if (!node) return null;
  if (node.name === 'Number') {
    const n = Number(textOf(state, node));
    return Number.isFinite(n) ? n : null;
  }
  if (node.name === 'VariableName') {
    const hit = consts.get(textOf(state, node));
    return typeof hit === 'number' ? hit : null;
  }
  return null;
}

// output: [w] | [w, h] | [w, h, d] | { x, y?, z? } → [{ n, text }, …] or null.
// `n` is null when the size is an expression we cannot evaluate (`output: [n]`
// inside a function) — the DIMENSIONALITY is still known, and that is what
// drives the indexing advice, so an unresolved size is not a failure.
function readOutput(state, node, consts) {
  if (!node) return null;
  if (node.name === 'ArrayExpression') {
    const dims = [];
    for (let ch = node.firstChild; ch; ch = ch.nextSibling) {
      if (ch.name === '[' || ch.name === ']' || ch.name === ',') continue;
      dims.push({ n: readNumber(state, ch, consts), text: textOf(state, ch) });
    }
    return dims.length ? dims : null;
  }
  if (node.name === 'ObjectExpression') {
    const keys = readObject(state, node);
    const dims = [];
    for (const axis of ['x', 'y', 'z']) {
      if (!keys[axis]) break;
      dims.push({ n: readNumber(state, keys[axis], consts), text: textOf(state, keys[axis]) });
    }
    return dims.length ? dims : null;
  }
  return null;
}

function readBoolean(state, node) {
  if (!node) return null;
  if (node.name === 'BooleanLiteral') return textOf(state, node) === 'true';
  return null;
}

// ---- the document analysis --------------------------------------------------

// Unwraps `gpu.createKernel(fn, s).setOutput([8]).setPipeline(true)` down to the
// createKernel call, collecting the setters applied on the way.
function unwrapKernelCall(state, node) {
  const chain = [];
  let cur = node;
  for (let guard = 0; guard < 12 && cur && cur.name === 'CallExpression'; guard++) {
    const parts = calleeParts(state, cur);
    if (!parts) return null;
    if (parts.property === 'createKernel' || parts.property === 'createKernelMap') {
      return { call: cur, chain, map: parts.property === 'createKernelMap' };
    }
    if (!SETTERS[parts.property] || !parts.objectNode) return null;
    chain.unshift({ setter: parts.property, args: argsOf(cur) });
    cur = parts.objectNode;
  }
  return null;
}

function paramNames(state, fnNode) {
  if (!fnNode) return [];
  if (fnNode.name !== 'FunctionExpression' && fnNode.name !== 'ArrowFunction') return [];
  const names = [];
  for (let ch = fnNode.firstChild; ch; ch = ch.nextSibling) {
    if (ch.name !== 'ParamList') continue;
    for (let p = ch.firstChild; p; p = p.nextSibling) {
      if (p.name === 'VariableDefinition') names.push({ name: textOf(state, p), node: p });
    }
    break;
  }
  return names;
}

function applySettings(info, state, settingsNode, consts) {
  const keys = readObject(state, settingsNode);
  if (keys.output) {
    const dims = readOutput(state, keys.output, consts);
    if (dims) info.output = dims;
  }
  for (const flag of ['graphical', 'pipeline', 'immutable', 'dynamicOutput']) {
    const value = readBoolean(state, keys[flag]);
    if (value !== null) info[flag] = value;
  }
}

// `later` distinguishes `k.setOutput([n])` written as its own statement — which
// really does resize the kernel between calls — from a setter chained onto
// createKernel, which is just how that kernel was configured in the first place.
function applySetter(info, state, setter, args, consts, later) {
  const key = SETTERS[setter];
  if (!key) return;
  if (key === 'output') {
    const dims = readOutput(state, args[0], consts);
    if (dims) {
      info.output = dims;
      if (later) info.resized = true;
    } else if (later) {
      info.resized = true; // resized to something we cannot evaluate
    }
    return;
  }
  const value = readBoolean(state, args[0]);
  if (value !== null) info[key] = value;
}

function analyzeDocument(state) {
  const tree = syntaxTree(state);
  const consts = new Map(); // simple `const N = 128` bindings, for output sizes
  const kernels = new Map(); // name → kernel info
  const results = new Map(); // name → { kernel, callText, argTexts }
  const calls = new Map(); // kernel name → [{ argNodes, argTexts, text }]
  const pendingResults = []; // resolved after every kernel is known
  const pendingSetters = []; // `k.setOutput([…])` statements

  tree.iterate({
    enter: ref => {
      if (ref.name === 'VariableDeclaration') {
        for (const { name, value } of declarators(ref.node)) {
          const varName = textOf(state, name);
          if (value.name === 'Number') {
            const n = Number(textOf(state, value));
            if (Number.isFinite(n)) consts.set(varName, n);
            continue;
          }
          if (value.name !== 'CallExpression') continue;
          const unwrapped = unwrapKernelCall(state, value);
          if (unwrapped) {
            const args = argsOf(unwrapped.call);
            // createKernelMap takes (subKernels, fn, settings)
            const fnIndex = unwrapped.map ? 1 : 0;
            const info = {
              name: varName,
              params: paramNames(state, args[fnIndex]),
              output: null,
              graphical: null,
              pipeline: null,
              immutable: null,
              dynamicOutput: null,
              resized: false,
              map: unwrapped.map,
              fnNode: args[fnIndex] || null,
              from: value.from,
            };
            applySettings(info, state, args[fnIndex + 1], consts);
            for (const link of unwrapped.chain) {
              applySetter(info, state, link.setter, link.args, consts, false);
            }
            kernels.set(varName, info);
            continue;
          }
          // maybe `const result = someKernel(args)` — resolved in pass two
          const parts = calleeParts(state, value);
          if (parts && !parts.object) {
            pendingResults.push({ varName, callee: parts.property, call: value });
          }
        }
        return;
      }
      if (ref.name !== 'CallExpression') return;
      const parts = calleeParts(state, ref.node);
      if (!parts) return;
      if (!parts.object) {
        // a call of a bare name — a candidate kernel invocation
        const argNodes = argsOf(ref.node);
        const list = calls.get(parts.property) || [];
        list.push({
          argTexts: argNodes.map(a => textOf(state, a)),
          // `gamma(normalize(signal))` — an argument that is itself a call; the
          // callee name is all we need to reach that kernel's output later
          argCallees: argNodes.map(a => {
            if (a.name !== 'CallExpression') return null;
            const inner = calleeParts(state, a);
            return inner && !inner.object ? inner.property : null;
          }),
          text: textOf(state, ref.node).replace(/\s+/g, ' '),
        });
        calls.set(parts.property, list);
      } else if (SETTERS[parts.property]) {
        pendingSetters.push({ target: parts.object, setter: parts.property, args: argsOf(ref.node) });
      }
    },
  });

  for (const { target, setter, args } of pendingSetters) {
    const info = kernels.get(target);
    if (info) applySetter(info, state, setter, args, consts, true);
  }
  for (const { varName, callee, call } of pendingResults) {
    const info = kernels.get(callee);
    if (!info) continue;
    results.set(varName, {
      kernel: info,
      callText: textOf(state, call).replace(/\s+/g, ' '),
      argNodes: argsOf(call),
    });
  }
  return { kernels, results, calls, consts };
}

// One analysis per document version: the syntax Tree is stable for a given
// state, and hover, completion and signature help all ask on the same one.
const analysisCache = new WeakMap();

export function documentAnalysis(state) {
  try {
    const tree = syntaxTree(state);
    const hit = analysisCache.get(tree);
    if (hit) return hit;
    const result = analyzeDocument(state);
    analysisCache.set(tree, result);
    return result;
  } catch (e) {
    // a half-typed document must never break the editor
    return { kernels: new Map(), results: new Map(), calls: new Map(), consts: new Map() };
  }
}

// ---- shape language ---------------------------------------------------------

function sizeText(dim) {
  return dim.n != null ? String(dim.n) : dim.text;
}

function outputLiteral(info) {
  return info.output ? `[${info.output.map(sizeText).join(', ')}]` : null;
}

function threadCount(info) {
  if (!info.output) return null;
  let total = 1;
  for (const d of info.output) {
    if (d.n == null) return null;
    total *= d.n;
  }
  return total;
}

// what a call of this kernel evaluates to, as a type line
export function returnType(info) {
  if (info.graphical) return 'no useful return value';
  if (info.pipeline) return 'Texture';
  const dims = info.output;
  if (!dims) return 'Float32Array (output not set yet)';
  const [w, h, d] = dims;
  if (dims.length === 1) return `Float32Array(${sizeText(w)})`;
  if (dims.length === 2) return `Array(${sizeText(h)}) of Float32Array(${sizeText(w)})`;
  if (dims.length === 3) {
    return `Array(${sizeText(d)}) of Array(${sizeText(h)}) of Float32Array(${sizeText(w)})`;
  }
  return 'Float32Array';
}

// `result[y][x]` — the index expression for this kernel's result
export function indexExpression(name, info) {
  if (info.graphical || info.pipeline || !info.output) return null;
  if (info.output.length === 1) return `${name}[x]`;
  if (info.output.length === 2) return `${name}[y][x]`;
  if (info.output.length === 3) return `${name}[z][y][x]`;
  return null;
}

// The row-first inversion, restated with this kernel's own numbers. Same
// gotcha, same wording family as content/layoutNote.js.
function indexingNote(name, info) {
  const dims = info.output;
  if (!dims) return '';
  const axis = (dim, label) =>
    dim.n != null ? `${label} = 0 … ${dim.n - 1}` : `${label} over ${dim.text}`;
  if (dims.length === 2) {
    return (
      `output: ${outputLiteral(info)} is [width, height], but the result is indexed ` +
      `row-first — ${name}[y][x], with ${axis(dims[1], 'y')} down the height and ` +
      `${axis(dims[0], 'x')} across the width. Swap them and you read the transpose.`
    );
  }
  if (dims.length === 3) {
    return (
      `output: ${outputLiteral(info)} is [width, height, depth], but the result is ` +
      `indexed ${name}[z][y][x] — depth first, then row, then column.`
    );
  }
  return '';
}

function sizeNote(info) {
  if (!info.resized) return '';
  return ' The size is changed by setOutput() before the call, so it is whatever was set last.';
}

function count(n, word) {
  return `${n.toLocaleString('en-US')} ${word}${n === 1 ? '' : 's'}`;
}

// "output: [64] — 64 threads, one per cell" (sizes we could not evaluate still
// give the dimensionality, which is what the indexing advice needs)
function outputClause(info) {
  const literal = outputLiteral(info);
  if (!literal) return 'output not set in this createKernel call';
  const threads = threadCount(info);
  return `output: ${literal}${threads != null ? ` — ${count(threads, 'thread')}, one per cell` : ''}`;
}

// ---- descriptors ------------------------------------------------------------

// Host-side advice for a pipeline result. The mode-safe guard is the exact form
// pipelines-and-textures task 1 teaches — same code, so a learner meets it once.
const PIPELINE_NOTE =
  'Pass it straight into another kernel to keep it on the GPU, or call .toArray() to ' +
  'read the numbers. The CPU backend returns a plain array with no .toArray — guard ' +
  'with result.toArray ? result.toArray() : result.';

function graphicalNote(kernelName) {
  return (
    `The kernel calls this.color(r, g, b, a) once per pixel; read the image from ` +
    `${kernelName}.canvas (e.g. render(${kernelName}.canvas)) or ${kernelName}.getPixels() — ` +
    `not from the value the call returns.`
  );
}

// What `const result = double(data)` holds. `name` is the variable being
// described; `via` is the call text when there is one.
export function outputDoc(name, info, via) {
  const from = via ? `Returned by ${via} — ` : '';
  if (info.graphical) {
    return {
      name,
      type: 'no useful return value',
      summary: `${from}graphical: true draws to a canvas, so the call's return value is not the image.`,
      sample: `${info.name}.canvas · ${info.name}.getPixels()`,
      note: graphicalNote(info.name),
    };
  }
  if (info.pipeline) {
    return {
      name,
      type: 'Texture',
      summary: `${from}pipeline: true keeps the result on the GPU, so this is a Texture, not an array.`,
      sample: `${name}.toArray ? ${name}.toArray() : ${name}`,
      note: PIPELINE_NOTE,
    };
  }
  const dims = info.output;
  if (!dims) {
    return {
      name,
      type: 'Float32Array',
      summary: `${from}${info.name} has no output yet — gpu.js needs one before the first call.`,
      sample: '',
      note: `Set it in the settings object (output: [64]) or with ${info.name}.setOutput([64]).`,
    };
  }
  const [w, h, d] = dims;
  let shape;
  if (dims.length === 1) {
    shape = `${w.n != null ? count(w.n, 'number') : `${sizeText(w)} numbers`}, one per thread`;
  } else if (dims.length === 2) {
    shape = `${sizeText(h)} rows × ${sizeText(w)} columns of numbers, one per thread`;
  } else if (dims.length === 3) {
    shape = `${sizeText(d)} planes × ${sizeText(h)} rows × ${sizeText(w)} columns of numbers`;
  } else shape = `${dims.length} dimensions of numbers`;
  return {
    name,
    type: returnType(info),
    summary: `${from}${shape}.${sizeNote(info)}`,
    sample: indexExpression(name, info) ? `${indexExpression(name, info)} → number` : '',
    note: indexingNote(name, info),
  };
}

/** The kernel variable itself: `double` in `const double = gpu.createKernel(…)`. */
export function kernelDoc(name, info) {
  const params = info.params.map(p => p.name).join(', ');
  const call = `${name}(${params})`;
  let sample = call;
  let note = '';
  if (info.graphical) {
    sample = `${name}.canvas · ${name}.getPixels()`;
    note = graphicalNote(name);
  } else if (info.pipeline) {
    sample = `const out = ${call}`;
    note = PIPELINE_NOTE;
  } else if (info.output) {
    const index = indexExpression('', info);
    sample = index ? `${call}${index} → number` : call;
    // the note must talk about the RESULT, not about the kernel variable
    note = indexingNote('result', info);
  } else {
    note = `Set it in the settings object (output: [64]) or with ${name}.setOutput([64]).`;
  }
  return {
    name,
    type: `${call} → ${returnType(info)}`,
    summary: `A kernel over ${outputClause(info)}.${sizeNote(info)}`,
    sample,
    note,
  };
}

// ---- kernel arguments -------------------------------------------------------

// How a value INSIDE a kernel is indexed. Unlike indexExpression this also
// answers for a pipeline result: a Texture handed to a kernel is unpacked by
// gpu.js and reads exactly like the array it came from — `.toArray()` is
// host-side only, and calling it in kernel code is a mistake worth pre-empting.
function insideIndex(name, info) {
  const dims = info.output;
  if (!dims) return null;
  if (dims.length === 1) return `${name}[x]`;
  if (dims.length === 2) return `${name}[y][x]`;
  if (dims.length === 3) return `${name}[z][y][x]`;
  return null;
}

// Describes an argument whose value is another kernel's output, from the point
// of view of the kernel RECEIVING it.
function kernelSideDoc(name, info, lead) {
  if (info.graphical) {
    return {
      name,
      type: 'nothing useful',
      summary: `${lead}, but that kernel is graphical: true — its call returns no data, only a canvas.`,
      sample: '',
      note: graphicalNote(info.name),
    };
  }
  const index = insideIndex(name, info);
  const dims = info.output;
  const shape = dims
    ? dims.length === 1
      ? `${sizeText(dims[0])} numbers`
      : dims.length === 2
        ? `${sizeText(dims[1])} rows × ${sizeText(dims[0])} columns`
        : `${sizeText(dims[2])} planes × ${sizeText(dims[1])} rows × ${sizeText(dims[0])} columns`
    : 'numbers';
  const rowFirst =
    dims && dims.length >= 2 ? ` Row-first: ${index} is row y, column x.` : '';
  const note = info.pipeline
    ? `On the host that value is a Texture, but inside a kernel gpu.js unpacks it — index ` +
      `it like an array (${index || `${name}[x]`}); .toArray() is host-side only.${rowFirst}`
    : dims && dims.length >= 2
      ? `Indexed ${index} — row y, column x, the same inversion as the output size.`
      : '';
  return {
    name,
    type: info.pipeline ? 'Texture' : returnType(info),
    summary: `${lead} — ${shape}.`,
    sample: index ? `${index} → number` : '',
    note,
  };
}

// The enclosing kernel when `pos` sits inside a createKernel function body or
// parameter list — matched by node position, since SyntaxNodes have no identity.
function kernelAt(analysis, pos) {
  for (const info of analysis.kernels.values()) {
    const fn = info.fnNode;
    if (fn && pos >= fn.from && pos <= fn.to) return info;
  }
  return null;
}

/**
 * kernelArgDoc(analysis, pos, name, inputByName) → descriptor | null
 *
 * Inside `gpu.createKernel(function (data) { … })`, says which argument `data`
 * is and — when a call site is in the same document — what will actually be
 * passed to it. When the call site passes a task input, the input's own
 * descriptor is reused verbatim, so the two never drift apart.
 */
export function kernelArgDoc(analysis, pos, name, inputByName = new Map()) {
  const info = kernelAt(analysis, pos);
  if (!info) return null;
  const index = info.params.findIndex(p => p.name === name);
  if (index < 0) return null;
  const ordinal = `Argument ${index + 1} of the kernel ${info.name}`;
  const sites = analysis.calls.get(info.name) || [];
  const site = sites.find(s => s.argTexts.length > index);
  if (!site) {
    return {
      name,
      type: `kernel argument ${index + 1}`,
      summary: `${ordinal}. Nothing calls ${info.name} with that many arguments in this file yet.`,
      sample: '',
      note: '',
    };
  }
  // several call sites with different arguments: the first is an example
  const varied = new Set(sites.filter(s => s.argTexts.length > index).map(s => s.argTexts[index]));
  const lead = varied.size > 1 ? 'e.g. ' : '';
  const passed = site.argTexts[index];
  const call = `${lead}${site.text}`;

  const input = inputByName.get(passed);
  if (input) {
    return {
      name,
      type: input.type,
      summary: `${ordinal} — ${call} passes the task input ${passed}. ${input.summary}`,
      // the input descriptor's sample/note are already written from inside a
      // kernel ("photo[y][x] is the pixel…"), so they carry over verbatim
      sample: input.sample,
      note: input.note,
    };
  }
  // `contrast(mapTexture)` — a variable holding another kernel's result — or
  // `gamma(normalize(signal))`, the same thing spelled inline
  const viaVar = analysis.results.get(passed);
  const viaCall = site.argCallees[index] ? analysis.kernels.get(site.argCallees[index]) : null;
  const upstream = (viaVar && viaVar.kernel) || viaCall;
  if (upstream) {
    return kernelSideDoc(name, upstream, `${ordinal} — ${call} passes ${passed}`);
  }
  const kernelArg = analysis.kernels.get(passed);
  if (kernelArg) {
    return {
      name,
      type: returnType(kernelArg),
      summary: `${ordinal} — ${call} passes the kernel ${passed} itself.`,
      sample: '',
      note: '',
    };
  }
  return {
    name,
    type: `kernel argument ${index + 1}`,
    summary: `${ordinal} — ${call} passes ${passed}.`,
    sample: '',
    note: '',
  };
}

// ---- the resolver shared by hover and completion ----------------------------

/**
 * describeIdentifier(state, pos, name, inputByName) → descriptor | null
 *
 * Most specific binding wins: a kernel parameter shadows a task input of the
 * same name (and its descriptor embeds that input anyway), then the document's
 * kernels and their results, then the task's injected globals.
 */
export function describeIdentifier(state, pos, name, inputByName = new Map()) {
  try {
    const analysis = documentAnalysis(state);
    const arg = kernelArgDoc(analysis, pos, name, inputByName);
    if (arg) return arg;
    const kernel = analysis.kernels.get(name);
    if (kernel) return kernelDoc(name, kernel);
    const result = analysis.results.get(name);
    if (result) return outputDoc(name, result.kernel, result.callText);
    return inputByName.get(name) || null;
  } catch (e) {
    return inputByName.get(name) || null;
  }
}

/**
 * Document-scoped names worth offering documentation for in the completion
 * popup: the kernels the learner has declared and the variables they assigned
 * from a kernel call. CodeMirror's local-scope source already offers these
 * names with no documentation, so completions.js merges ours over them.
 */
export function documentIdentifierDocs(state, pos, inputByName = new Map()) {
  const out = [];
  try {
    const analysis = documentAnalysis(state);
    for (const [name, info] of analysis.kernels) out.push(kernelDoc(name, info));
    for (const [name, res] of analysis.results) {
      out.push(outputDoc(name, res.kernel, res.callText));
    }
    const inFn = kernelAt(analysis, pos);
    if (inFn) {
      for (const p of inFn.params) {
        const doc = kernelArgDoc(analysis, pos, p.name, inputByName);
        if (doc) out.push(doc);
      }
    }
  } catch (e) {
    return out;
  }
  return out;
}

/**
 * A signature-help entry for calling a kernel the learner declared, shaped like
 * a gpujsApi dataset entry so completion/signatureHelp.js can render it with no
 * special case. Returns null for anything that is not a known kernel.
 */
export function kernelSignature(state, path) {
  if (!Array.isArray(path) || path.length !== 1) return null;
  try {
    const info = documentAnalysis(state).kernels.get(path[0]);
    if (!info) return null;
    const names = info.params.map(p => p.name);
    const out = outputDoc('result', info, null);
    return {
      name: info.name,
      kind: 'method',
      context: 'kernel-method',
      signature: `${info.name}(${names.join(', ')}): ${returnType(info)}`,
      params: info.params.map((p, i) => ({
        name: p.name,
        type: 'kernel argument',
        doc: `Argument ${i + 1} of this kernel — every thread sees the whole value; index it down to a number with this.thread.x before doing arithmetic.`,
      })),
      doc: out.summary + (out.note ? ` ${out.note}` : ''),
    };
  } catch (e) {
    return null;
  }
}
