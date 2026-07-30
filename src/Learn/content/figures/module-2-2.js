// Figures for module 2-2 — Reductions.
// Schema: { '<taskNum>': [{ slug, caption, alt?, placement: 'intro' | 'goal' }] }
// Markup lives at ./2-2/<taskNum>-<slug>.svg (path derived by figures/index.js).

export default {
  '2': [
    {
      slug: 'strided',
      caption: 'thread x takes every 64th element — neighbours read neighbours at every step',
      placement: 'intro',
    },
  ],
  '3': [
    {
      slug: 'one-rung',
      caption: 'your partner lives one output-width away',
      placement: 'intro',
    },
  ],
  '4': [
    {
      slug: 'ladder',
      caption: 'halve, halve, halve — the ladder every platform climbs',
      placement: 'intro',
    },
  ],
};
