// Figures for "Video Filters" — uuid 4d39e404-8261-48a2-9912-176a3ea0b1bf.
//
// Keyed by TASK SLUG (never task number: tasks may be reordered).
// Schema: { '<taskSlug>': [{ name, caption, alt?, placement: 'intro' | 'goal' }] }
// Markup lives beside this file at ./<taskSlug>-<name>.svg — the path is
// derived by content/figures/index.js and never written out here.

export default {
  'frame-budget': [
    {
      name: 'budget',
      caption: "same three kernels, same arithmetic — the readbacks are what miss the frame",
      placement: 'intro',
    },
  ],
  'temporal-average': [
    {
      name: 'carry',
      caption: "each frame folds into what the last one left behind — state outlives the frame",
      placement: 'intro',
    },
  ],
  'background-model': [
    {
      name: 'panels',
      caption: "subtract the room you already know, and what is left is what arrived",
      placement: 'intro',
    },
  ],
};
