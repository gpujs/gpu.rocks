// Figures for "Wavefronts: Aligning DNA on the Diagonal" —
// uuid a85ca6d9-49ae-48e4-94ea-cf5761be9dae.
//
// Keyed by TASK SLUG (never task number: tasks may be reordered).
// Schema: { '<taskSlug>': [{ name, caption, alt?, placement: 'intro' | 'goal' }] }
// Markup lives beside this file at ./<taskSlug>-<name>.svg — the path is
// derived by content/figures/index.js and never written out here.
//
// Three figures, and no more: this module's ideas are almost entirely SHAPE, so
// the three that got one are the three a paragraph could not carry — where the
// dependencies point, which way the independent cells run, and how a bare
// thread index turns into a matrix coordinate. Everything after that is a loop.

export default {
  'scoring-recurrence': [
    {
      name: 'three-ways',
      caption: 'four candidates, one winner — and the 0 is the escape hatch',
      placement: 'intro',
    },
  ],
  'the-anti-diagonal': [
    {
      name: 'wavefront',
      caption: 'the matrix is not serial, it is serial along the wrong axis',
      placement: 'intro',
    },
  ],
  'index-the-diagonal': [
    {
      name: 'unroll',
      caption: 'a thread arrives knowing only its number — two lines make it a cell',
      placement: 'intro',
    },
  ],
};
