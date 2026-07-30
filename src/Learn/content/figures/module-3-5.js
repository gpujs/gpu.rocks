// Figures for module 3-5 — Ray-Marched Metaballs.
// Schema: { '<taskNum>': [{ slug, caption, alt?, placement: 'intro' | 'goal' }] }
// Markup lives at ./3-5/<taskNum>-<slug>.svg (path derived by figures/index.js).

export default {
  '1': [
    {
      slug: 'sdf-sign',
      caption: 'one function knows the whole sphere: + outside, 0 on the skin, − inside',
      placement: 'intro',
    },
  ],
  '3': [
    {
      slug: 'march',
      caption: "the field value is a promise — hop exactly that far and you'll never overshoot",
      placement: 'intro',
    },
  ],
  '4': [
    {
      slug: 'normals',
      caption: 'nudge ± e, subtract, normalize — six taps and the surface points at you',
      placement: 'intro',
    },
  ],
};
