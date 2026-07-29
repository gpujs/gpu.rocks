// task/editorThemes.js — CodeMirror 6 themes for the task-page editor.
// Colors are the mockup's --ed-* tokens, verbatim: a warm-paper light theme
// and a Monokai-style dark theme, switched by the learn theme (ThemeContext).

import { EditorView } from '@codemirror/view';
import { HighlightStyle, syntaxHighlighting } from '@codemirror/language';
import { tags as t } from '@lezer/highlight';

const MONO = 'ui-monospace, "SF Mono", Menlo, Consolas, monospace';

function buildTheme(c, dark) {
  const theme = EditorView.theme(
    {
      '&': {
        backgroundColor: c.bg,
        color: c.ink,
        height: '100%',
        fontSize: '.84rem',
      },
      '.cm-scroller': {
        fontFamily: MONO,
        lineHeight: '1.65',
      },
      '.cm-content': {
        caretColor: c.caret,
        padding: '.8rem 0 1.5rem',
      },
      '.cm-line': {
        padding: '0 1.2rem 0 .4rem',
      },
      '.cm-cursor, .cm-dropCursor': {
        borderLeftColor: c.caret,
        borderLeftWidth: '2px',
      },
      '&.cm-focused': {
        outline: 'none',
      },
      '.cm-gutters': {
        backgroundColor: c.bg,
        color: c.gutter,
        border: 'none',
      },
      '.cm-lineNumbers .cm-gutterElement': {
        minWidth: '3.2rem',
        padding: '0 1.1rem 0 .5rem',
      },
      '.cm-activeLine': {
        backgroundColor: c.line,
      },
      '.cm-activeLineGutter': {
        backgroundColor: c.line,
        color: c.ink,
      },
      '&.cm-focused > .cm-scroller > .cm-selectionLayer .cm-selectionBackground, .cm-selectionBackground, .cm-content ::selection': {
        backgroundColor: c.selection,
      },
      '.cm-selectionMatch': {
        backgroundColor: c.selection,
      },
      '&.cm-focused .cm-matchingBracket': {
        backgroundColor: c.line,
        outline: `1px solid ${c.gutter}`,
      },
      '&.cm-focused .cm-nonmatchingBracket': {
        outline: `1px solid ${c.kw}`,
      },
    },
    { dark }
  );

  const highlight = HighlightStyle.define([
    { tag: [t.comment, t.lineComment, t.blockComment], color: c.cm },
    {
      tag: [
        t.keyword,
        t.self,
        t.bool,
        t.null,
        t.atom,
        t.controlKeyword,
        t.operatorKeyword,
        t.definitionKeyword,
        t.moduleKeyword,
      ],
      color: c.kw,
    },
    { tag: [t.string, t.special(t.string), t.regexp], color: c.str },
    { tag: [t.number], color: c.num },
    {
      tag: [t.function(t.variableName), t.function(t.propertyName), t.function(t.definition(t.variableName))],
      color: c.fn,
    },
    { tag: [t.propertyName, t.definition(t.propertyName)], color: c.prop, fontStyle: 'italic' },
    { tag: [t.variableName, t.definition(t.variableName)], color: c.ink },
    { tag: [t.operator, t.punctuation, t.bracket, t.separator], color: c.ink },
  ]);

  return [theme, syntaxHighlighting(highlight)];
}

// --ed-* tokens, light theme (warm paper)
export const lightEditorTheme = buildTheme(
  {
    bg: '#f7f5ef',
    gutter: '#a9a494',
    line: 'rgba(60, 50, 20, .06)',
    ink: '#2c2a24',
    kw: '#c7226e',
    str: '#9a7d0a',
    num: '#7c4dbe',
    fn: '#3e7d0e',
    prop: '#0e7ab8',
    cm: '#8a877a',
    caret: '#d6288f', // --pink (light)
    selection: 'rgba(60, 50, 20, .14)',
  },
  false
);

// --ed-* tokens, dark theme (Monokai-style)
export const darkEditorTheme = buildTheme(
  {
    bg: '#272822',
    gutter: '#75715a',
    line: 'rgba(255, 255, 255, .045)',
    ink: '#f8f8f2',
    kw: '#f92672',
    str: '#e6db74',
    num: '#ae81ff',
    fn: '#a6e22e',
    prop: '#66d9ef',
    cm: '#75715a',
    caret: '#ff79c6', // --pink (dark)
    selection: 'rgba(255, 255, 255, .14)',
  },
  true
);
