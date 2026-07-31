// Figures for "N-Body Gravity" — uuid 5de47751-c27a-47ca-ad4a-17f875176788.
//
// Keyed by TASK SLUG (never task number: tasks may be reordered).
// Schema: { '<taskSlug>': [{ name, caption, alt?, placement: 'intro' | 'goal' }] }
// Markup lives beside this file at ./<taskSlug>-<name>.svg — the path is
// derived by content/figures/index.js and never written out here.

export default {
  'sum-the-sky': [
    {
      name: 'all-pulls',
      caption: "sixty-three pulls per body, summed in one thread — n² work, n time",
      placement: 'intro',
    },
  ],
  'softening': [
    {
      name: 'softening',
      caption: "close encounters flatten out instead of blowing up",
      placement: 'intro',
    },
  ],
};
