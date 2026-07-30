// signatureHelp.js — VS Code-style parameter hints for the learn editor.
//
// A StateField tracks one "session": the ArgList of a call whose callee
// resolves via the gpujsApi dataset. Sessions open when the user types "(" or
// "," (or presses Mod-Shift-Space), stay live while the cursor remains inside
// that ArgList (active parameter recomputed from top-level commas via the
// syntax tree, so nested parens/brackets/strings never miscount), and close
// when the cursor leaves the call or on Esc.
//
// Esc handling: the binding dispatches the dismiss effect but returns FALSE,
// so the same keypress still reaches @codemirror/view's base handler that arms
// the Esc-then-Tab focus escape hatch — dismissing the tooltip never costs the
// user their way out of the editor.

import { StateField, StateEffect } from '@codemirror/state';
import { showTooltip, keymap } from '@codemirror/view';
import { syntaxTree } from '@codemirror/language';
import { getSignature, byName } from './gpujsApi';

const triggerSignature = StateEffect.define(); // Mod-Shift-Space
const dismissSignature = StateEffect.define(); // Esc

// ---- syntax-tree helpers ---------------------------------------------------

// every enclosing ArgList (innermost first) whose parens contain `pos`
function callChain(state, pos) {
  const chain = [];
  for (let node = syntaxTree(state).resolveInner(pos, -1); node; node = node.parent) {
    if (node.name !== 'ArgList' || pos <= node.from) continue;
    const close = node.lastChild;
    if (close && close.name === ')' && pos > close.from) continue; // after the ')'
    const call = node.parent;
    if (call && (call.name === 'CallExpression' || call.name === 'NewExpression')) {
      chain.push({ argList: node, call });
    }
  }
  return chain;
}

// callee as a property path: gpu.createKernel → ['gpu','createKernel'];
// `new GPU(...)` → ['GPU']. Complex callees (calls, subscripts) → null.
function calleePath(state, call) {
  let callee = null;
  for (let ch = call.firstChild; ch; ch = ch.nextSibling) {
    if (ch.name === 'new' || ch.name === 'ArgList' || ch.name === 'TypeArgList') continue;
    callee = ch;
    break;
  }
  if (!callee) return null;
  const text = state.sliceDoc(callee.from, callee.to).replace(/\s+/g, '');
  if (!/^[\w$]+(\.[\w$]+)*$/.test(text)) return null;
  return text.split('.');
}

// active parameter = top-level commas of the ArgList before the cursor
function activeIndex(argList, pos) {
  let idx = 0;
  for (let ch = argList.firstChild; ch; ch = ch.nextSibling) {
    if (ch.name === ',' && ch.from < pos) idx++;
  }
  return idx;
}

// Object-literal drill-down (VS Code style): when the active parameter is an
// options object with a documented key context (param.options names it — e.g.
// createKernel's settings → 'kernel-settings') and the cursor sits on/inside a
// Property of that object literal, return that option's dataset entry. Right
// after the '{' (no enclosing Property) this returns null and the generic
// parameter line stays. Nested calls never reach here: resolveChain already
// resolved the innermost known call, so a `utils.flatten(` inside a property
// value shows flatten's own signature instead.
function optionAt(state, argList, entry, active, pos) {
  const param = entry.params[active];
  const optionCtx = param && param.options && byName[param.options];
  if (!optionCtx) return null;
  let prop = null; // deepest Property passed on the way up — becomes the
  // object literal's DIRECT child by the time the walk reaches it
  for (let node = syntaxTree(state).resolveInner(pos, -1); node; node = node.parent) {
    if (node.from <= argList.from) return null; // walked out of the ArgList
    if (node.name === 'Property') prop = node;
    if (node.name !== 'ObjectExpression') continue;
    const parent = node.parent;
    if (!parent || parent.name !== 'ArgList' || parent.from !== argList.from) continue;
    // `node` is the argument object literal itself (SyntaxNode wrappers have
    // no identity — compare by position, like completions.js does)
    if (!prop || prop.parent.from !== node.from) return null;
    const key = prop.firstChild;
    if (!key) return null;
    const name = state.sliceDoc(key.from, key.to).replace(/^['"]|['"]$/g, '');
    return optionCtx[name] || null;
  }
  return null;
}

// ---- tooltip DOM -----------------------------------------------------------

// split "name(a, b?): Ret" into prefix / top-level params / suffix
function splitSignature(sig) {
  const open = sig.indexOf('(');
  if (open < 0) return null;
  let depth = 0;
  let quote = null;
  let close = -1;
  const commas = [];
  for (let i = open; i < sig.length; i++) {
    const c = sig[i];
    if (quote) {
      if (c === quote) quote = null;
    } else if (c === "'" || c === '"') {
      quote = c;
    } else if (c === '(' || c === '[' || c === '{') {
      depth++;
    } else if (c === ')' || c === ']' || c === '}') {
      depth--;
      if (depth === 0 && c === ')') {
        close = i;
        break;
      }
    } else if (c === ',' && depth === 1) {
      commas.push(i);
    }
  }
  if (close < 0) return null;
  const params = [];
  let start = open + 1;
  for (const i of [...commas, close]) {
    params.push(sig.slice(start, i).trim());
    start = i + 1;
  }
  return {
    prefix: sig.slice(0, open + 1),
    params: params.filter(p => p.length > 0),
    suffix: sig.slice(close),
  };
}

function span(text, className) {
  const el = document.createElement('span');
  if (className) el.className = className;
  el.textContent = text;
  return el;
}

function buildDom(entry, active, option) {
  const dom = document.createElement('div');
  dom.className = 'cm-learn-signature';
  const sig = document.createElement('div');
  sig.className = 'sig';
  const parts = splitSignature(entry.signature);
  // clamp: a trailing ...rest param soaks up all later arguments
  const nParams = entry.params.length;
  const rest = nParams > 0 && entry.params[nParams - 1].name.startsWith('...');
  const clamped = Math.min(active, nParams - 1);
  const highlight = active < nParams || rest ? clamped : -1;
  if (parts && parts.params.length === nParams && nParams > 0) {
    sig.appendChild(span(parts.prefix));
    parts.params.forEach((p, i) => {
      if (i > 0) sig.appendChild(span(', '));
      sig.appendChild(span(p, i === highlight ? 'param on' : 'param'));
    });
    sig.appendChild(span(parts.suffix));
  } else {
    sig.textContent = entry.signature;
  }
  dom.appendChild(sig);
  const param = highlight >= 0 ? entry.params[highlight] : null;
  if (option) {
    // drill-down: the cursor is on a documented key of an options-object
    // argument — the detail line describes that option instead of the param
    const doc = document.createElement('div');
    doc.className = 'pdoc';
    const name = document.createElement('b');
    name.textContent = option.name;
    doc.appendChild(name);
    // type/shape = the option signature minus its leading "name:"
    const shape = option.signature.startsWith(`${option.name}:`)
      ? option.signature.slice(option.name.length + 1).trim()
      : null;
    if (shape) doc.appendChild(span(`: ${shape}`, 'otype'));
    const body = document.createElement('span');
    body.innerHTML = ` — ${option.doc}`; // trusted in-repo authored HTML
    doc.appendChild(body);
    dom.appendChild(doc);
  } else if (param && param.doc) {
    const doc = document.createElement('div');
    doc.className = 'pdoc';
    const name = document.createElement('b');
    name.textContent = param.name;
    doc.appendChild(name);
    const body = document.createElement('span');
    body.innerHTML = ` — ${param.doc}`; // trusted in-repo authored HTML
    doc.appendChild(body);
    dom.appendChild(doc);
  }
  return dom;
}

// ---- state field -----------------------------------------------------------

function makeValue(state, argList, entry, pos) {
  const active = activeIndex(argList, pos);
  return {
    from: argList.from,
    entry,
    active,
    option: optionAt(state, argList, entry, active, pos),
    tooltip: null, // filled below (needs the final active index)
  };
}

function withTooltip(val) {
  return {
    ...val,
    tooltip: {
      pos: val.from, // anchored at the open paren — stable while typing args
      above: true,
      strictSide: false,
      create: () => ({ dom: buildDom(val.entry, val.active, val.option) }),
    },
  };
}

// innermost enclosing call (of `chain`) whose callee the dataset knows
function resolveChain(state, chain, pos) {
  for (const { argList, call } of chain) {
    const path = calleePath(state, call);
    const entry = path && getSignature(path);
    if (entry) return makeValue(state, argList, entry, pos);
  }
  return null;
}

// did this user-input transaction insert a trigger character?
function typedTrigger(tr) {
  if (!tr.isUserEvent('input')) return false;
  let found = false;
  tr.changes.iterChanges((fromA, toA, fromB, toB, inserted) => {
    if (/[(,]/.test(inserted.toString())) found = true;
  });
  return found;
}

const signatureField = StateField.define({
  create: () => null,
  update(value, tr) {
    if (!tr.docChanged && !tr.selection && !tr.effects.length) return value;
    if (tr.effects.some(e => e.is(dismissSignature))) return null;
    const manual = tr.effects.some(e => e.is(triggerSignature));
    const retrigger = manual || typedTrigger(tr);
    let val = value;
    if (val && tr.docChanged) val = { ...val, from: tr.changes.mapPos(val.from, 1) };
    if (!val && !retrigger) return null;

    const state = tr.state;
    const pos = state.selection.main.head;
    const chain = callChain(state, pos);

    let next = null;
    if (retrigger) next = resolveChain(state, chain, pos);
    if (!next && val) {
      // keep the existing session while the cursor stays inside its ArgList
      const still = chain.find(c => c.argList.from === val.from);
      if (still) {
        next = makeValue(state, still.argList, val.entry, pos);
      } else if (!tr.isUserEvent('select.pointer')) {
        // the cursor left the session's ArgList without a mouse click — e.g.
        // it stepped over an inner call's `)` (a selection-only skip when the
        // paren was auto-closed) — fall back to the innermost known enclosing
        // call so the OUTER signature returns. Clicks still close the session.
        next = resolveChain(state, chain, pos);
      }
    }
    if (!next) return null;
    if (
      val &&
      val.tooltip &&
      val.from === next.from &&
      val.entry === next.entry &&
      val.active === next.active &&
      val.option === next.option
    ) {
      return val; // identical — keep tooltip identity, no DOM churn
    }
    return withTooltip(next);
  },
  provide: f => showTooltip.from(f, v => (v ? v.tooltip : null)),
});

const signatureKeymap = keymap.of([
  {
    key: 'Escape',
    run: view => {
      if (view.state.field(signatureField)) {
        view.dispatch({ effects: dismissSignature.of(null) });
      }
      // never consume: the base keydown handler must still arm Esc-then-Tab
      return false;
    },
  },
  {
    key: 'Mod-Shift-Space',
    run: view => {
      view.dispatch({ effects: triggerSignature.of(null) });
      return true;
    },
  },
]);

export function signatureHelp() {
  return [signatureField, signatureKeymap];
}
