// Figures for "Hello, Kernel" — uuid f1399353-b65c-463a-bbba-adcaeb779e17.
//
// Keyed by TASK SLUG (never task number: tasks may be reordered).
// Schema: { '<taskSlug>': [{ name, caption, alt?, placement: 'intro' | 'goal' }] }
// Markup lives beside this file at ./<taskSlug>-<name>.svg — the path is
// derived by content/figures/index.js and never written out here.

export default {
  'first-kernel': [
    {
      name: 'fan-out',
      caption: "one function, sixteen launches — the loop you never wrote",
      placement: 'intro',
    },
  ],
  'index-formula': [
    {
      name: 'loop-unroll',
      caption: "same body, new address — the loop unrolls into threads",
      placement: 'intro',
    },
  ],
  'checkerboard': [
    {
      name: 'thread-grid',
      caption: "two coordinates per thread — the grid is the picture",
      placement: 'intro',
    },
  ],
};
