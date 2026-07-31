// Figures for "The Heat Equation & Stability" — uuid 514063bb-13a7-4aa9-b507-c449996df7ef.
//
// Keyed by TASK SLUG (never task number: tasks may be reordered).
// Schema: { '<taskSlug>': [{ name, caption, alt?, placement: 'intro' | 'goal' }] }
// Markup lives beside this file at ./<taskSlug>-<name>.svg — the path is
// derived by content/figures/index.js and never written out here.
//
// Three figures, on the three tasks whose idea has a shape: the weights that
// make a step an average, the mechanism by which that average detonates, and
// the one-word difference between an explicit step and a linear solve. Tasks 3
// and 5 get none — their picture is the table the learner prints.

export default {
  'explicit-step': [
    {
      name: 'weights',
      caption: 'five weights that add to one — while the middle one is positive',
      placement: 'intro',
    },
  ],
  'blow-it-up': [
    {
      name: 'past-the-line',
      caption: 'the fastest pattern the grid can hold, flipping and growing, every step',
      placement: 'intro',
    },
  ],
  'implicit-step': [
    {
      name: 'who-you-read',
      caption: "one word changes — 'new' — and the step becomes a system of equations",
      placement: 'intro',
    },
  ],
};
