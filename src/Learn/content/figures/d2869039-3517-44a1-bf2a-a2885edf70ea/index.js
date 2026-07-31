// Figures for "Pixels from Scratch" — uuid d2869039-3517-44a1-bf2a-a2885edf70ea.
//
// Keyed by TASK SLUG (never task number: tasks may be reordered).
// Schema: { '<taskSlug>': [{ name, caption, alt?, placement: 'intro' | 'goal' }] }
// Markup lives beside this file at ./<taskSlug>-<name>.svg — the path is
// derived by content/figures/index.js and never written out here.

export default {
  'coordinate-gradient': [
    {
      name: 'uv-ramp',
      caption: "every thread knows where it stands — divide by 128 and position becomes color",
      placement: 'intro',
    },
  ],
  'plot-a-wave': [
    {
      name: 'distance-test',
      caption: "no thread draws the line — each pixel just answers: am i near it?",
      placement: 'intro',
    },
  ],
  'radial-ripples': [
    {
      name: 'polar',
      caption: "subtract the center and (x, y) becomes r — anything you do to r is a circle",
      placement: 'intro',
    },
  ],
};
