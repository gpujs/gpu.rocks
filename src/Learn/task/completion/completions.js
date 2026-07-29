// completions.js — VS Code-style, context-aware completions for the learn
// editor, layered over lang-javascript's defaults (keyword snippets + locals).
//
// Contexts (resolved via the syntax tree + a cheap assignment scan):
//   gpu.<|>            → GPU instance methods (also any `const x = new GPU()` var)
//   this.<|>           → kernel-inside API, only inside a fn passed to createKernel
//   this.thread.<|>    → x / y / z (same for this.output.)
//   utils.<|>          → utils members
//   Math.<|>           → whole Math dataset; non-whitelisted entries carry a
//                        "not available inside kernels" note (never blocked)
//   console. / Date. / JSON.<|> → the dotted global entries
//   k.<|>              → kernel methods, when k is assigned from *.createKernel
//   { <|> } of createKernel(fn, { … }) → option-key completions (name position)
//   bare identifiers   → sandbox globals + task input names (+ CM's locals)
//
// Tab accepts ONLY while the completion popup is open (Prec.highest binding
// falling through to the existing indentWithTab otherwise). The Esc-then-Tab
// focus escape hatch is untouched: @codemirror/view skips ALL keymap Tab
// bindings while tabFocusMode is armed.

import {
  autocompletion,
  acceptCompletion,
  snippetCompletion,
  completionStatus,
} from '@codemirror/autocomplete';
import { syntaxTree } from '@codemirror/language';
import { javascriptLanguage } from '@codemirror/lang-javascript';
import { keymap } from '@codemirror/view';
import { Prec } from '@codemirror/state';
import { byName } from './gpujsApi';

// ---- rendering -------------------------------------------------------------

function iconType(entry) {
  switch (entry.kind) {
    case 'method':
      return 'method';
    case 'property':
    case 'option':
      return 'property';
    case 'kernel-api':
      return entry.name === 'color' ? 'method' : 'property';
    case 'global':
      if (entry.name === 'GPU') return 'class';
      return /\(/.test(entry.signature) ? 'function' : 'variable';
    default:
      return 'variable';
  }
}

// info panel: monospace signature header + prose doc (VS Code flyout style).
// entry.doc/param docs are trusted in-repo authored HTML.
function infoNode(entry) {
  const dom = document.createElement('div');
  dom.className = 'cm-learn-info';
  const sig = document.createElement('div');
  sig.className = 'sig';
  sig.textContent = entry.signature;
  dom.appendChild(sig);
  const doc = document.createElement('div');
  doc.className = 'doc';
  doc.innerHTML = entry.doc;
  dom.appendChild(doc);
  if (entry.kernelSafe === false) {
    const note = document.createElement('div');
    note.className = 'kernel-note';
    note.textContent = 'Not available inside kernel functions (host code only).';
    dom.appendChild(note);
  }
  return dom;
}

// entry → CM completion; boost lifts dataset entries above generic words
function toCompletion(entry, { boost = 1, label = entry.name, insertText } = {}) {
  const spec = {
    label,
    type: iconType(entry),
    boost,
    info: () => infoNode(entry),
  };
  const snip = insertText !== undefined ? insertText : entry.insertText;
  // dataset placeholder convention ${label} is CM snippet syntax already
  if (snip && snip !== label) return snippetCompletion(snip, spec);
  return spec;
}

function contextCompletions(ctx, boost) {
  return Object.values(byName[ctx]).map(e => toCompletion(e, { boost }));
}

// ---- prebuilt option lists -------------------------------------------------

const gpuInstanceOptions = contextCompletions('gpu-instance', 2);
const kernelMethodOptions = contextCompletions('kernel-method', 2);
const kernelInsideOptions = contextCompletions('kernel-inside', 2);
const utilsOptions = contextCompletions('utils-member', 2);
const mathOptions = contextCompletions('math-member', 2);
const settingsOptions = contextCompletions('kernel-settings', 3);

// dotted globals (console.log, Date.now, JSON.stringify) become member
// completions for their receiver, with the receiver stripped off the snippet
function dottedMembers(receiver) {
  const out = [];
  for (const [name, entry] of Object.entries(byName.global)) {
    if (!name.startsWith(`${receiver}.`)) continue;
    const label = name.slice(receiver.length + 1);
    const insertText =
      entry.insertText && entry.insertText.startsWith(`${receiver}.`)
        ? entry.insertText.slice(receiver.length + 1)
        : entry.insertText;
    out.push(toCompletion(entry, { boost: 2, label, insertText }));
  }
  return out;
}

const consoleOptions = dottedMembers('console');
const dateOptions = dottedMembers('Date');
const jsonOptions = dottedMembers('JSON');

const bareGlobalOptions = Object.values(byName.global).map(e => toCompletion(e, { boost: 1 }));

// after `new ` only constructors make sense — plain labels, no `new` doubling
const constructorOptions = ['GPU', 'Float32Array', 'Promise'].map(name =>
  toCompletion(byName.global[name], { boost: 2, insertText: name })
);

function axisDoc(text) {
  const dom = document.createElement('div');
  dom.className = 'cm-learn-info';
  const doc = document.createElement('div');
  doc.className = 'doc';
  doc.textContent = text;
  dom.appendChild(doc);
  return dom;
}

const threadAxes = [
  ['x', 'Column of the output cell this thread computes (0-based).'],
  ['y', 'Row of the output cell — 0 when the output is 1D.'],
  ['z', 'Depth of the output cell — 0 unless the output is 3D.'],
].map(([label, doc]) => ({ label, type: 'property', boost: 2, info: () => axisDoc(doc) }));

const outputAxes = [
  ['x', 'Output width (the [w] in output: [w, h]).'],
  ['y', 'Output height — 1 when the output is 1D.'],
  ['z', 'Output depth — 1 unless the output is 3D.'],
].map(([label, doc]) => ({ label, type: 'property', boost: 2, info: () => axisDoc(doc) }));

// ---- cheap doc analysis ----------------------------------------------------

// best-effort variable classification: `const g = new GPU()` → gpu instance,
// `const k = <x>.createKernel(…)` → kernel. Names in both sets are dropped.
function scanVars(state) {
  const text = state.doc.toString();
  const gpuVars = new Set(['gpu']);
  const kernelVars = new Set();
  const gpuRe = /(?:const|let|var)\s+([\w$]+)\s*=\s*new\s+GPU\b/g;
  const kernelRe = /(?:const|let|var)\s+([\w$]+)\s*=\s*[\w$]+\s*\.\s*createKernel(?:Map)?\s*\(/g;
  let m;
  while ((m = gpuRe.exec(text))) gpuVars.add(m[1]);
  while ((m = kernelRe.exec(text))) kernelVars.add(m[1]);
  for (const name of kernelVars) {
    if (gpuVars.has(name)) {
      gpuVars.delete(name);
      kernelVars.delete(name);
    }
  }
  return { gpuVars, kernelVars };
}

// is `node` inside a function that is a direct argument of *.createKernel(…)
// (or createKernelMap / addFunction — same kernel language applies)?
function insideKernelFn(state, node) {
  for (let cur = node; cur; cur = cur.parent) {
    if (
      cur.name === 'FunctionExpression' ||
      cur.name === 'ArrowFunction' ||
      cur.name === 'FunctionDeclaration'
    ) {
      const arg = cur.parent;
      if (arg && arg.name === 'ArgList' && arg.parent && arg.parent.name === 'CallExpression') {
        const callee = arg.parent.firstChild;
        const text = state.sliceDoc(callee.from, callee.to);
        if (/(?:^|\.)\s*(?:createKernel(?:Map)?|addFunction)\s*$/.test(text)) return true;
      }
    }
  }
  return false;
}

// option-key completions inside the settings object of createKernel(fn, { … })
function settingsCompletion(context) {
  const { state, pos } = context;
  let child = null;
  for (let cur = syntaxTree(state).resolveInner(pos, -1); cur; child = cur, cur = cur.parent) {
    // SyntaxNode wrappers have no stable identity — compare by position
    if (cur.name === 'Property' && child && child.from !== cur.firstChild.from) return null; // value position
    if (cur.name !== 'ObjectExpression') continue;
    const arg = cur.parent;
    if (!arg || arg.name !== 'ArgList' || !arg.parent || arg.parent.name !== 'CallExpression') {
      return null;
    }
    const callee = arg.parent.firstChild;
    if (!/createKernel(?:Map)?\s*$/.test(state.sliceDoc(callee.from, callee.to))) return null;
    const word = context.matchBefore(/[\w$]*$/);
    if (!context.explicit && word.from === pos && state.sliceDoc(pos - 1, pos) !== '{') {
      // only auto-open on typing or right after `{`
      return null;
    }
    return { from: word.from, options: settingsOptions, validFor: /^[\w$]*$/ };
  }
  return null;
}

// ---- the source ------------------------------------------------------------

function memberOptions(state, path, node) {
  const [head] = path;
  if (path.length === 1) {
    if (head === 'Math') return mathOptions;
    if (head === 'utils') return utilsOptions;
    if (head === 'console') return consoleOptions;
    if (head === 'Date') return dateOptions;
    if (head === 'JSON') return jsonOptions;
    if (head === 'this') return insideKernelFn(state, node) ? kernelInsideOptions : null;
    const { gpuVars, kernelVars } = scanVars(state);
    if (gpuVars.has(head)) return gpuInstanceOptions;
    if (kernelVars.has(head)) return kernelMethodOptions;
    return null;
  }
  if (path.length === 2 && head === 'this') {
    if (path[1] === 'thread') return threadAxes;
    if (path[1] === 'output') return outputAxes;
  }
  return null;
}

export function buildCompletionSource(inputNames) {
  const inputOptions = inputNames.map(name => ({
    label: name,
    type: 'variable',
    boost: 3,
    info: () => axisDoc('Task input — injected as a global for this task (from task.inputs).'),
  }));
  const bareOptions = [...bareGlobalOptions, ...inputOptions];

  return context => {
    const { state, pos } = context;
    const node = syntaxTree(state).resolveInner(pos, -1);
    if (/Comment|String|Regexp/.test(node.name)) return null;

    const inSettings = settingsCompletion(context);
    if (inSettings) return inSettings;

    // member access: receiver chain + partial word (no spaces — best effort)
    const member = context.matchBefore(/(?:[\w$]+\.)+[\w$]*$/);
    if (member) {
      const parts = member.text.split('.');
      const word = parts.pop();
      const options = memberOptions(state, parts, node);
      if (!options) return null;
      return { from: pos - word.length, options, validFor: /^[\w$]*$/ };
    }

    // bare identifier
    const word = context.matchBefore(/[\w$]+$/);
    if (!word && !context.explicit) return null;
    const from = word ? word.from : pos;
    if (/new\s+$/.test(state.sliceDoc(Math.max(0, from - 6), from))) {
      return { from, options: constructorOptions, validFor: /^[\w$]*$/ };
    }
    return { from, options: bareOptions, validFor: /^[\w$]*$/ };
  };
}

// ---- extension bundle ------------------------------------------------------

// Tab accepts iff the popup is open with a selected option; otherwise the
// binding reports "not handled" and the default-precedence indentWithTab runs.
const tabAcceptKeymap = Prec.highest(
  keymap.of([
    {
      key: 'Tab',
      run: view => completionStatus(view.state) === 'active' && acceptCompletion(view),
    },
  ])
);

export function learnCompletions(inputNames = []) {
  return [
    autocompletion({ icons: true }),
    javascriptLanguage.data.of({ autocomplete: buildCompletionSource(inputNames) }),
    tabAcceptKeymap,
  ];
}
