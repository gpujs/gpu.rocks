// content/tracks.js — track metadata AND track membership.
//
// A track owns an ORDERED list of module uuids; that order is the teaching
// order, and adjacency in it is what "next module" means. A module file does
// NOT declare which track it belongs to — this file is the only place that
// says so, and content/registry.js validates that no module is listed twice
// (here or across tracks) and that every uuid listed actually exists.
//
// A module in NO track is an orphan: it renders in the "Others" category on
// the learn home — track-like to look at, but unordered (no "module N of M")
// and with no next-module offer when you finish it.
//
// Numbers, titles and taglines come from the approved mockup.
//
// Plain ESM data, no imports: node scripts read this file directly.

const tracks = [
  {
    number: 1,
    title: 'GPGPU 101',
    tagline: 'From zero to your first thousand threads',
    modules: [
      'f1399353-b65c-463a-bbba-adcaeb779e17', // Hello, Kernel
      '42b68d01-e46e-455d-9321-2425db8b9eb6', // Data In, Data Out
      'c3876efb-c01c-42ee-826b-d166a182bcd1', // Thinking in Parallel
      '9f4aeaa5-71d4-4b0d-afc4-b08c24b7e08e', // Pipelines & Textures
      'b9188894-0ae1-4e75-8538-4348f6fc61ae', // Measuring Speed Honestly
    ],
  },
  {
    number: 2,
    title: 'Advanced Math',
    tagline: 'Heavy math, thousands of threads at once',
    modules: [
      '972e080b-a2a9-4151-ac98-d1d9caf7b6b9', // Matrix Multiply
      '3dadc130-6cf9-4798-a0c9-be53e8f78d67', // Reductions
      '66933805-3a1d-48c0-a287-16d3d7d00016', // Convolution & Filters
      '9ea19810-b622-4611-a049-9daa49021ca2', // Monte Carlo Methods
      '5de47751-c27a-47ca-ad4a-17f875176788', // N-Body Gravity
    ],
  },
  {
    number: 3,
    title: 'Computational Graphics',
    tagline: 'Pictures computed, not drawn',
    modules: [
      'd2869039-3517-44a1-bf2a-a2885edf70ea', // Pixels from Scratch
      '0de4764c-e40f-4966-9014-05e3a26c0eec', // Escape-Time Fractals
      '407c2c34-b316-4301-8ec2-b5c829b591e6', // Cellular Automata
      'bc3d0b34-d454-4870-9d24-ca22a1144bbe', // Reaction–Diffusion
      '8b1282bd-fb68-4620-a92f-0ed12f5200e7', // Ray-Marched Metaballs
    ],
  },
];

export default tracks;
