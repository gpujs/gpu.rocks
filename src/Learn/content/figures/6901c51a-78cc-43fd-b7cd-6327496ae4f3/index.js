// Figures for "The Canny Edge Pipeline" — uuid 6901c51a-78cc-43fd-b7cd-6327496ae4f3.
//
// Keyed by TASK SLUG (never task number: tasks may be reordered).
// Schema: { '<taskSlug>': [{ name, caption, alt?, placement: 'intro' | 'goal' }] }
// Markup lives beside this file at ./<taskSlug>-<name>.svg — the path is
// derived by content/figures/index.js and never written out here.
//
// Three figures, on the three tasks whose idea has a SHAPE: the map of the
// whole chain (task 1, so a learner arriving cold knows where they are), the
// two-neighbour comparison that non-maximum suppression turns on (task 3), and
// the front that advances one pixel per hysteresis pass (task 5).

export default {
  'gaussian-blur': [
    {
      name: 'stages',
      caption: "five kernels in a row — you build them left to right, then chain them in task 6",
      alt:
        'The five stages of Canny as a strip of pictures: noisy photo, blurred photo, a thick ' +
        'gradient ridge, that ridge thinned to one pixel, the thin line broken into strong and ' +
        'weak segments, and one unbroken edge after hysteresis.',
      placement: 'intro',
    },
  ],
  'non-maximum-suppression': [
    {
      name: 'along-the-gradient',
      caption: "the two neighbours that matter are across the edge, not along it",
      alt:
        'A patch of gradient magnitudes holding a three-pixel-wide ridge. The pixel under test is ' +
        'compared with the two neighbours lying along its gradient direction, which crosses the ' +
        'ridge; the edge itself runs at right angles to that. On the right, the three-wide band ' +
        'becomes a one-pixel line.',
      placement: 'intro',
    },
  ],
  hysteresis: [
    {
      name: 'propagation',
      caption: "the front moves one pixel per pass, and nobody knows how many passes that is",
      alt:
        'A strong pixel beside a chain of weak ones. Each pass promotes exactly one more weak ' +
        'cell, so the strong front advances one pixel at a time until the chain is complete. A ' +
        'separate group of weak pixels touching nothing strong is never promoted, and is dropped.',
      placement: 'intro',
    },
  ],
};
