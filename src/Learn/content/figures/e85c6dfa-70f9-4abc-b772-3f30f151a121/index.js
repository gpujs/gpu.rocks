// Figures for "Optical Flow" — uuid e85c6dfa-70f9-4abc-b772-3f30f151a121.
//
// Keyed by TASK SLUG (never task number: tasks may be reordered).
// Schema: { '<taskSlug>': [{ name, caption, alt?, placement: 'intro' | 'goal' }] }
// Markup lives beside this file at ./<taskSlug>-<name>.svg — the path is
// derived by content/figures/index.js and never written out here.

export default {
  'aperture-problem': [
    {
      name: 'only-normal',
      caption: "the window cannot tell these apart — only the across-the-edge part is real",
      alt:
        'An anti-diagonal edge seen through a small circular aperture. The true motion arrow ' +
        'points right; the only recoverable component points down-right, at right angles to the ' +
        'edge. A dashed line through both arrowheads marks the whole family of motions that fit ' +
        'the same evidence.',
      placement: 'intro',
    },
  ],
  'lucas-kanade': [
    {
      name: 'window',
      caption: "25 equations walk into a 2x2 system; one of them comes out with an answer",
      alt:
        'A five by five window of pixels, each contributing one brightness-constancy equation, ' +
        'gathered into five running sums that fill a two by two matrix and a right-hand side, ' +
        'solved once for the centre pixel.',
      placement: 'intro',
    },
  ],
  'good-features': [
    {
      name: 'three-windows',
      caption: "flat says nothing, an edge says half of it, only a corner says both",
      alt:
        'Three windows side by side. A flat patch with no gradients scores zero on both ' +
        'eigenvalues; a straight edge scores high on the larger and zero on the smaller; a ' +
        'corner scores high on both and is the only one marked trustworthy.',
      placement: 'intro',
    },
  ],
};
