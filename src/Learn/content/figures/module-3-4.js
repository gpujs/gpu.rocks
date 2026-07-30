// Figures for module 3-4 — Reaction–Diffusion.
// Schema: { '<taskNum>': [{ slug, caption, alt?, placement: 'intro' | 'goal' }] }
// Markup lives at ./3-4/<taskNum>-<slug>.svg (path derived by figures/index.js).

export default {
  '1': [
    {
      slug: 'laplacian',
      caption: "five reads, one number — how far am i from my neighbors' average?",
      placement: 'intro',
    },
  ],
  '2': [
    {
      slug: 'two-grids',
      caption: 'one snapshot in, two grids out — u·v² is where the chemistry happens',
      placement: 'intro',
    },
  ],
};
