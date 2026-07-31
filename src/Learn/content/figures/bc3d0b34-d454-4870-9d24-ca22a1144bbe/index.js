// Figures for "Reaction–Diffusion" — uuid bc3d0b34-d454-4870-9d24-ca22a1144bbe.
//
// Keyed by TASK SLUG (never task number: tasks may be reordered).
// Schema: { '<taskSlug>': [{ name, caption, alt?, placement: 'intro' | 'goal' }] }
// Markup lives beside this file at ./<taskSlug>-<name>.svg — the path is
// derived by content/figures/index.js and never written out here.

export default {
  'laplacian': [
    {
      name: 'laplacian',
      caption: "five reads, one number — how far am i from my neighbors' average?",
      placement: 'intro',
    },
  ],
  'gray-scott-step': [
    {
      name: 'two-grids',
      caption: "one snapshot in, two grids out — u·v² is where the chemistry happens",
      placement: 'intro',
    },
  ],
};
