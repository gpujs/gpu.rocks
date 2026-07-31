// Figures for "Jump Flooding: Voronoi in log n Passes" —
// uuid a741a650-84e8-4362-a344-43a4d7018c7f.
//
// Keyed by TASK SLUG (never task number: tasks may be reordered).
// Schema: { '<taskSlug>': [{ name, caption, alt?, placement: 'intro' | 'goal' }] }
// Markup lives beside this file at ./<taskSlug>-<name>.svg — the path is
// derived by content/figures/index.js and never written out here.
//
// Three figures, on the three tasks whose idea has a SHAPE: the stencil, the
// binary decomposition that makes seven passes enough, and the subtraction that
// turns two unsigned fields into a signed one. The distance pass and the error
// audit are arithmetic on a field that is already understood — nothing to draw.

export default {
  'one-pass': [
    {
      name: 'nine-candidates',
      caption: 'nine cells, and one of them is you — which is how a cell keeps what it already had',
      alt:
        'A lattice of grid cells with nine marked: the thread\'s own cell at the centre and eight ' +
        'neighbours k cells away in each direction, forming the corners and edge midpoints of a ' +
        'square of side 2k.',
      placement: 'intro',
    },
  ],
  'halving-ladder': [
    {
      name: 'seven-jumps',
      caption: '105 = 64 + 32 + 8 + 1 — no distance on a 128-wide grid needs an eighth pass',
      alt:
        'Seven stacked rows, one per pass, with the stride halving from 64 to 1. The bar reaches ' +
        '64, then 96, stalls at 16, reaches 104 at stride 8, stalls at 4 and 2, and arrives at 105 ' +
        'on the final stride-1 pass.',
      placement: 'intro',
    },
  ],
  'signed-distance': [
    {
      name: 'two-floods',
      caption: 'one term is always zero, so the subtraction is the entire sign logic',
      alt:
        'Three panels of the same star: the field flooded from the inside pixels (zero inside), ' +
        'minus the field flooded from the outside pixels (zero outside), equals a signed field ' +
        'that is negative inside and positive outside.',
      placement: 'intro',
    },
  ],
};
