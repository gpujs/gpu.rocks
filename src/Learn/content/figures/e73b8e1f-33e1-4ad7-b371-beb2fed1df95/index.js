// Figures for "Iterative Linear Solvers" — uuid e73b8e1f-33e1-4ad7-b371-beb2fed1df95.
//
// Keyed by TASK SLUG (never task number: tasks may be reordered).
// Schema: { '<taskSlug>': [{ name, caption, alt?, placement: 'intro' | 'goal' }] }
// Markup lives beside this file at ./<taskSlug>-<name>.svg — the path is
// derived by content/figures/index.js and never written out here.
//
// Three tasks get a picture, and only three: the ones whose idea has a shape.
// Task 1 is two grids and a one-way arrow between them (why Jacobi is
// embarrassingly parallel). Task 3 is literally a chessboard. Task 5 is a
// number falling, twice, past a line. Tasks 2 and 4 are arithmetic and
// plumbing — a diagram there would be decoration.

export default {
  'jacobi-sweep': [
    {
      name: 'two-grids',
      caption: 'everyone reads yesterday — which is exactly why everyone can go at once',
      placement: 'intro',
    },
  ],
  'red-black-halves': [
    {
      name: 'chessboard',
      caption: "a red cell's four neighbours are all black, so no red waits on a red",
      placement: 'intro',
    },
  ],
  'count-the-sweeps': [
    {
      name: 'race',
      caption: 'same finish line, same work per sweep — 170 sweeps against 275',
      placement: 'intro',
    },
  ],
};
