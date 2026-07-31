// Figures for "Hydraulic Erosion: Carving Terrain by Accumulation" —
// uuid 07165ca1-0a5d-4c84-9b63-fdd3c9480661.
//
// Keyed by TASK SLUG (never task number: tasks may be reordered).
// Schema: { '<taskSlug>': [{ name, caption, alt?, placement: 'intro' | 'goal' }] }
// Markup lives beside this file at ./<taskSlug>-<name>.svg — the path is
// derived by content/figures/index.js and never written out here.
//
// Three tasks earn a picture, because three ideas here have a SHAPE: a slope
// measured across two cells and turned into a lit surface; the two-pass
// structure that a gather forces on you; and one exchange spent in opposite
// directions by two kernels. The routing arithmetic and the time loop do not —
// they are prose and code.

export default {
  hillshade: [
    {
      name: 'normal-from-neighbours',
      caption: 'two neighbours make a slope, a slope makes a normal, a normal catches the light',
      placement: 'intro',
    },
  ],
  'route-the-water': [
    {
      name: 'two-passes',
      caption: 'the total lives two rings away, so it gets a pass of its own',
      placement: 'intro',
    },
  ],
  'pick-up-and-drop': [
    {
      name: 'one-number-twice',
      caption: 'one delta, computed twice, spent in opposite directions',
      placement: 'intro',
    },
  ],
};
