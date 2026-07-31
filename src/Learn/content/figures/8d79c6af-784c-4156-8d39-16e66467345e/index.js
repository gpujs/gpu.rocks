// Figures for "Colour Spaces" — uuid 8d79c6af-784c-4156-8d39-16e66467345e.
//
// Keyed by TASK SLUG (never task number: tasks may be reordered).
// Schema: { '<taskSlug>': [{ name, caption, alt?, placement: 'intro' | 'goal' }] }
// Markup lives beside this file at ./<taskSlug>-<name>.svg — the path is
// derived by content/figures/index.js and never written out here.

export default {
  'rgb-to-hsv': [
    {
      name: 'wheel',
      caption: "grey lives in the hole in the middle, where none of the three formulas apply",
      placement: 'intro',
    },
  ],
  'hue-wraps': [
    {
      name: 'midpoint',
      caption: "far apart on a line, neighbours on a wheel — and only one of those is true",
      placement: 'intro',
    },
  ],
  'colour-mask': [
    {
      name: 'same-pixel',
      caption: "turn the light down and RGB loses the colour; HSV only loses the V",
      placement: 'intro',
    },
  ],
};
