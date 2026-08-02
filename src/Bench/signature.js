/**
 * src/Bench/signature.js — the ten rows worth running if you only run ten.
 *
 * A full pass is half an hour or more, which is a long time to wait to find out
 * whether a GPU is worth it on your machine. This is the short answer: one row
 * per distinct thing a GPU can be good or bad at, chosen so that no two of them
 * tell you the same thing.
 *
 * Curated, not derived. "Signature" is a judgement about what a reader learns,
 * and no property of a workload encodes it — which is why this list lives in
 * its own file where the reasoning is visible rather than as a boolean scattered
 * across thirty-four modules.
 *
 * matmul is first and always first: it is the case every GPU claim starts from
 * and the one a reader already has an intuition for, so it anchors everything
 * below it.
 */
const SIGNATURE = [
  ['matmul', 'dense arithmetic — the best case, and where every claim starts'],
  ['reduction', 'many-to-one, the tree most other primitives are built from'],
  ['prefix-sum', 'a dependency chain that still parallelises'],
  ['blur-separable', 'bandwidth rather than FLOPs: almost no arithmetic per byte'],
  ['nbody', 'all-pairs — every thread reads every body'],
  ['heat', 'a stencil, stepped: neighbour reads over and over'],
  ['escape-time', 'branch divergence — neighbouring threads doing different work'],
  ['sobel', 'the image stencil under most of computer vision'],
  ['fft', 'a transform, where the algorithm matters more than the hardware'],
  ['undersized', 'the honest loss: too little work to pay for the dispatch'],
];

export const SIGNATURE_IDS = SIGNATURE.map(([id]) => id);
export const SIGNATURE_WHY = new Map(SIGNATURE);
export default SIGNATURE_IDS;
