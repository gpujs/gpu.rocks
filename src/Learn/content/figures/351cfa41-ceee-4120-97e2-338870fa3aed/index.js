// Figures for "Prefix Sums (Scan)" — uuid 351cfa41-ceee-4120-97e2-338870fa3aed.
//
// Keyed by TASK SLUG (never task number: tasks may be reordered).
// Schema: { '<taskSlug>': [{ name, caption, alt?, placement: 'intro' | 'goal' }] }
// Markup lives beside this file at ./<taskSlug>-<name>.svg — the path is
// derived by content/figures/index.js and never written out here.

export default {
  'hillis-steele': [
    {
      name: 'ladder',
      caption: "1, 2, 4 — every pass reaches twice as far, and cell 7 collects the lot",
      placement: 'intro',
    },
  ],
  'exclusive-scan': [
    {
      name: 'shift',
      caption: "a zero goes in the front, the grand total drops off the back",
      placement: 'intro',
    },
  ],
  'blelloch': [
    {
      name: 'sweeps',
      caption: "up the tree to build subtotals, down it again to hand them out",
      placement: 'intro',
    },
  ],
};
