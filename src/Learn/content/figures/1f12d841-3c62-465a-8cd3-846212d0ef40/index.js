// Figures for "The Ising Model: Colour to Break the Race" —
// uuid 1f12d841-3c62-465a-8cd3-846212d0ef40.
//
// Keyed by TASK SLUG (never task number: tasks may be reordered).
// Schema: { '<taskSlug>': [{ name, caption, alt?, placement: 'intro' | 'goal' }] }
// Markup lives beside this file at ./<taskSlug>-<name>.svg — the path is
// derived by content/figures/index.js and never written out here.
//
// Three tasks get a picture, and only three: the ones whose idea has a shape.
// Task 1 is a five-entry table that happens to be drawable, and seeing all five
// neighbourhoods at once is what makes "ΔE has only five values" obvious. Task 3
// is the whole argument of the module in one before/after pair — two threads,
// one bond, both charged for a change that never happened. Task 6 is a curve
// falling off a cliff, and the honest gap between the exact answer and what a
// 128×128 lattice actually does is worth drawing rather than describing.
//
// Task 4 deliberately has NO chessboard diagram: Iterative Linear Solvers
// already ships one (figures/e73b8e1f…/red-black-halves-chessboard.svg) and the
// task's whole point is that this is the same picture, not a second one.

export default {
  'flip-energy': [
    {
      name: 'cost-table',
      caption: 'five neighbourhoods, five prices — and only two of them are ever a gamble',
      alt: 'Five spin neighbourhoods side by side, with four, three, two, one and zero neighbours agreeing with the centre spin, and the resulting flip costs +8, +4, 0, -4 and -8.',
      placement: 'intro',
    },
  ],
  'the-race': [
    {
      name: 'both-at-once',
      caption: 'nobody overwrote anybody. the arithmetic was still describing a lattice that had already moved',
      alt: 'Two neighbouring up spins sharing a satisfied bond. Each thread prices flipping itself assuming the other holds still, both accept, and after both flip the bond is satisfied again — so the cost they each paid was never incurred.',
      placement: 'intro',
    },
  ],
  'phase-transition': [
    {
      name: 'order-parameter',
      caption: 'the exact answer drops off a cliff at 2.269; 16,384 spins in 150 sweeps take the corner wide',
      alt: 'Magnetisation against temperature, with the exact Onsager curve dropping vertically to zero at 2.269 and the measured 128 by 128 curve hanging on until about 2.4 before collapsing.',
      placement: 'intro',
    },
  ],
};
