// Figures for "Ray-Marched Metaballs" — uuid 8b1282bd-fb68-4620-a92f-0ed12f5200e7.
//
// Keyed by TASK SLUG (never task number: tasks may be reordered).
// Schema: { '<taskSlug>': [{ name, caption, alt?, placement: 'intro' | 'goal' }] }
// Markup lives beside this file at ./<taskSlug>-<name>.svg — the path is
// derived by content/figures/index.js and never written out here.

export default {
  'sphere-sdf': [
    {
      name: 'sdf-sign',
      caption: "one function knows the whole sphere: + outside, 0 on the skin, − inside",
      placement: 'intro',
    },
  ],
  'ray-march': [
    {
      name: 'march',
      caption: "the field value is a promise — hop exactly that far and you'll never overshoot",
      placement: 'intro',
    },
  ],
  'normals': [
    {
      name: 'normals',
      caption: "nudge ± e, subtract, normalize — six taps and the surface points at you",
      placement: 'intro',
    },
  ],
};
