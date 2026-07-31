// Figures for "Seam Carving: Content-Aware Resizing" —
// uuid a23a0d9b-0e9b-47f1-8c49-8f0b6317ee43.
//
// Keyed by TASK SLUG (never task number: tasks may be reordered).
// Schema: { '<taskSlug>': [{ name, caption, alt?, placement: 'intro' | 'goal' }] }
// Markup lives beside this file at ./<taskSlug>-<name>.svg — the path is
// derived by content/figures/index.js and never written out here.
//
// Three shapes, three figures: the dependency that decides the launch pattern,
// the walk that reads the answer out, and the gather that closes the gap.
// The energy map and the two payoff tasks get none — an energy map is a
// picture the learner's own run renders, and a loop is not a shape.

export default {
  'cumulative-cost': [
    {
      name: 'row-by-row',
      caption: 'a whole row at once, because nothing in it looks sideways',
      alt: 'Two rows of a cumulative cost map. Three neighbouring cells in the finished row above are marked as the only cells the highlighted cell below can have come from. Every cell of the lower row is computed at the same time; the rows go one after another.',
      placement: 'intro',
    },
  ],
  'read-the-seam': [
    {
      name: 'walk-up',
      caption: 'the cost map holds the price; the path has to be walked back out of it',
      alt: 'A six by four grid of cumulative costs. The cheapest cell of the bottom row, holding eight, is the start; from it a path is traced upwards, each step taking the cheapest of the three cells above, until it reaches the top row.',
      placement: 'intro',
    },
  ],
  'carve-one-seam': [
    {
      name: 'reflow',
      caption: 'nobody is pushed aside — every output pixel reaches for its own source',
      alt: 'Eight input pixels above seven output pixels. Input three is the seam and is removed. Outputs zero to two read straight down; outputs three to six read diagonally from inputs four to seven, one column further right.',
      placement: 'intro',
    },
  ],
};
