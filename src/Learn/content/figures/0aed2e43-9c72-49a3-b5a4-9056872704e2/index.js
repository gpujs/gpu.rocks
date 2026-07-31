// Figures for "Stream Compaction" — uuid 0aed2e43-9c72-49a3-b5a4-9056872704e2.
//
// Keyed by TASK SLUG (never task number: tasks may be reordered).
// Schema: { '<taskSlug>': [{ name, caption, alt?, placement: 'intro' | 'goal' }] }
// Markup lives beside this file at ./<taskSlug>-<name>.svg — the path is
// derived by content/figures/index.js and never written out here.

export default {
  'where-do-i-land': [
    {
      name: 'three-rows',
      caption: "flags say who survives; the scan under them says where each one lands",
      placement: 'intro',
    },
  ],
  'gather-not-scatter': [
    {
      name: 'pull-not-push',
      caption: "same arrows, opposite owner — and only one of the two is legal",
      placement: 'intro',
    },
  ],
  'binary-search-gather': [
    {
      name: 'lower-bound',
      caption: "the running count only ever goes up, so you can halve your way to it",
      placement: 'intro',
    },
  ],
};
