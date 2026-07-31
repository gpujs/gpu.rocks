// Figures for "Escape-Time Fractals" — uuid 0de4764c-e40f-4966-9014-05e3a26c0eec.
//
// Keyed by TASK SLUG (never task number: tasks may be reordered).
// Schema: { '<taskSlug>': [{ name, caption, alt?, placement: 'intro' | 'goal' }] }
// Markup lives beside this file at ./<taskSlug>-<name>.svg — the path is
// derived by content/figures/index.js and never written out here.

export default {
  'pixel-to-plane': [
    {
      name: 'pixel-to-plane',
      caption: "xMin, yMin, step: three numbers turn a pixel count into a place",
      placement: 'intro',
    },
  ],
  'escape-time': [
    {
      name: 'escape-orbit',
      caption: "iterate z² + c and watch: settle in, or fly off — the count is the answer",
      placement: 'intro',
    },
  ],
  'julia-dial': [
    {
      name: 'julia-flip',
      caption: "same loop, roles swapped — mandelbrot varies c, julia varies z₀",
      placement: 'intro',
    },
  ],
};
