// Figures for "Monte Carlo Methods" — uuid 9ea19810-b622-4611-a049-9daa49021ca2.
//
// Keyed by TASK SLUG (never task number: tasks may be reordered).
// Schema: { '<taskSlug>': [{ name, caption, alt?, placement: 'intro' | 'goal' }] }
// Markup lives beside this file at ./<taskSlug>-<name>.svg — the path is
// derived by content/figures/index.js and never written out here.

export default {
  'darts-at-a-circle': [
    {
      name: 'darts',
      caption: "throw darts, count hits — the circle’s area falls out of the ratio",
      placement: 'intro',
    },
  ],
  'integrate-the-unintegrable': [
    {
      name: 'mean-height',
      caption: "no antiderivative, no problem — average enough heights and it’s the area",
      placement: 'intro',
    },
  ],
};
