// Figures for module 3-1 — Pixels from Scratch.
// Schema: { '<taskNum>': [{ slug, caption, alt?, placement: 'intro' | 'goal' }] }
// Markup lives at ./3-1/<taskNum>-<slug>.svg (path derived by figures/index.js).

export default {
  '1': [
    {
      slug: 'uv-ramp',
      caption: 'every thread knows where it stands — divide by 128 and position becomes color',
      placement: 'intro',
    },
  ],
  '3': [
    {
      slug: 'distance-test',
      caption: 'no thread draws the line — each pixel just answers: am i near it?',
      placement: 'intro',
    },
  ],
  '4': [
    {
      slug: 'polar',
      caption: 'subtract the center and (x, y) becomes r — anything you do to r is a circle',
      placement: 'intro',
    },
  ],
};
