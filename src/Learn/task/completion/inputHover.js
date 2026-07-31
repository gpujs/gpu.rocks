// inputHover.js — hover an injected input name in the editor and get exactly
// the description the brief pane's "Task inputs" section and the completion
// popup show (all three render task/inputDoc.js descriptors).
//
// This is the path that still works when the completion popup can't help:
// once `data` is a binding in the learner's own code, CodeMirror's local-scope
// source owns that name in the popup, and hovering is how you ask what the
// value actually is.
//
// It is the only hoverTooltip in this editor, which is why the tooltip chrome
// can be styled as plain `.cm-tooltip-hover` (see _task.scss) — CodeMirror
// hosts every hover tooltip inside that container and adds `cm-tooltip-section`
// to the dom we return, so our own class rides on the inner element.

import { hoverTooltip } from '@codemirror/view';
import { syntaxTree } from '@codemirror/language';
import { inputDocDom } from '../inputDoc';

const WORD = /[\w$]/;

// Token positions where an identifier-shaped word is NOT the injected global:
// comment/string text, and the property side of `obj.data` / `{ data: 1 }`.
// VariableDefinition is deliberately absent — the starter code names the kernel
// function's parameter after the global it receives, and that parameter is the
// single most likely thing a learner points at.
const NOT_A_REFERENCE = /Comment|String|Regexp|PropertyName|PropertyDefinition/;

export function inputHover(docs = []) {
  if (!docs.length) return [];
  const byName = new Map(docs.map(d => [d.name, d]));
  return [
    hoverTooltip(
      (view, pos, side) => {
        const { state } = view;
        const line = state.doc.lineAt(pos);
        let start = pos;
        let end = pos;
        while (start > line.from && WORD.test(line.text[start - line.from - 1])) start--;
        while (end < line.to && WORD.test(line.text[end - line.from])) end++;
        // pointer sits in the gap beside a word rather than on it
        if ((start === pos && side < 0) || (end === pos && side > 0)) return null;
        const doc = byName.get(line.text.slice(start - line.from, end - line.from));
        if (!doc) return null;
        if (NOT_A_REFERENCE.test(syntaxTree(state).resolveInner(start, 1).name)) return null;
        return {
          pos: start,
          end,
          above: true,
          create: () => ({ dom: inputDocDom(doc, 'cm-learn-inputdoc') }),
        };
      },
      { hoverTime: 250 }
    ),
  ];
}
