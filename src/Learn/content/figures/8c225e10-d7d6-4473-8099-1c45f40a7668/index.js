// Figures for "Filtering in the Frequency Domain" — uuid
// 8c225e10-d7d6-4473-8099-1c45f40a7668.
//
// Keyed by TASK SLUG (never task number: tasks may be reordered).
// Schema: { '<taskSlug>': [{ name, caption, alt?, placement: 'intro' | 'goal' }] }
// Markup lives beside this file at ./<taskSlug>-<name>.svg — the path is
// derived by content/figures/index.js and never written out here.
//
// Three tasks get a picture, because three of the five ideas have a shape:
// the theorem is a commuting square, the wrap-around is a fold, and Gibbs
// ringing is a curve. The Gaussian roll-off and the magnitude gate are both
// "the same picture with a different response", so they get prose instead.

export default {
  'convolution-theorem': [
    {
      name: 'two-roads',
      caption: "the long way round is the fast way — and it arrives at the same numbers",
      placement: 'intro',
    },
  ],
  'brick-wall': [
    {
      name: 'ringing',
      caption: "you cut the spectrum with a cliff; the cliff hands you back a ripple",
      placement: 'intro',
    },
  ],
  'zero-padding': [
    {
      name: 'wrap-around',
      caption: "eight samples with nowhere to go, and they do not go quietly",
      placement: 'intro',
    },
  ],
};
