// Figures for "The FFT Butterfly" — uuid d4375da7-7178-4bee-8442-e04e80d563d1.
//
// Keyed by TASK SLUG (never task number: tasks may be reordered).
// Schema: { '<taskSlug>': [{ name, caption, alt?, placement: 'intro' | 'goal' }] }
// Markup lives beside this file at ./<taskSlug>-<name>.svg — the path is
// derived by content/figures/index.js and never written out here.

export default {
  'the-butterfly': [
    {
      name: 'network8',
      caption: 'twelve butterflies, and which pair you belong to is a fact about your index',
      placement: 'intro',
    },
  ],
  'bit-reversal': [
    {
      name: 'table',
      caption: 'the permutation is just the index, spelled backwards',
      placement: 'intro',
    },
  ],
  'dft-versus-fft': [
    {
      name: 'work',
      caption: 'one pass over a square, or thirteen passes over a line',
      placement: 'intro',
    },
  ],
};
