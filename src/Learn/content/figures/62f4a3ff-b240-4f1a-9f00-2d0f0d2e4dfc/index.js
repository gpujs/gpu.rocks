// Figures for "ODE Integrators" — uuid 62f4a3ff-b240-4f1a-9f00-2d0f0d2e4dfc.
//
// Keyed by TASK SLUG (never task number: tasks may be reordered).
// Schema: { '<taskSlug>': [{ name, caption, alt?, placement: 'intro' | 'goal' }] }
// Markup lives beside this file at ./<taskSlug>-<name>.svg — the path is
// derived by content/figures/index.js and never written out here.
//
// Three tasks get a picture, and only the three whose idea has a shape: the
// launch that turned inside out, the geometry of a single better step, and the
// two energy curves that are the whole point of the module. The convergence
// tables are numbers a learner prints themselves; a drawing of a table would
// add nothing.

export default {
  'euler-trajectory': [
    {
      name: 'one-thread',
      caption: 'the clock moved inside the kernel — one launch, one thread, one whole trajectory',
      placement: 'intro',
    },
  ],
  'midpoint': [
    {
      name: 'slopes',
      caption: 'the trial half-step is thrown away; only the slope it found is kept',
      placement: 'intro',
    },
  ],
  'long-orbit': [
    {
      name: 'energy-drift',
      caption: 'the accurate method leaks; the symplectic one ripples and stays',
      placement: 'intro',
    },
  ],
};
