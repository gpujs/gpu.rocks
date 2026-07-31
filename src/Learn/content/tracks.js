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
      '9f4aeaa5-71d4-4b0d-afc4-b08c24b7e08e', // Pipelines & Textures
      'b9188894-0ae1-4e75-8538-4348f6fc61ae', // Measuring Speed Honestly
    ],
  },
  {
    number: 2,
    title: 'Parallel Primitives',
    tagline: 'The handful of patterns everything else is built from',
    modules: [
      'c3876efb-c01c-42ee-826b-d166a182bcd1', // Thinking in Parallel
      '3dadc130-6cf9-4798-a0c9-be53e8f78d67', // Reductions
      '351cfa41-ceee-4120-97e2-338870fa3aed', // Prefix Sums (Scan)
      '0aed2e43-9c72-49a3-b5a4-9056872704e2', // Stream Compaction
      'dfb254f4-b68c-464e-af9e-1439efb7fcec', // Histograms & Binning
      '1ba56df3-64f4-4387-8723-958f4ad53c09', // Top-K Selection
      '84e0728e-6dbd-4f06-8c76-14b708a55b47', // Bitonic Sort
      'fd3ff796-daed-4036-9202-987b374bb4d3', // Radix Sort
    ],
  },
  {
    number: 3,
    title: 'Math & Simulation',
    tagline: 'Heavy math, thousands of threads at once',
    modules: [
      '972e080b-a2a9-4151-ac98-d1d9caf7b6b9', // Matrix Multiply
      '9ea19810-b622-4611-a049-9daa49021ca2', // Monte Carlo Methods
      '5de47751-c27a-47ca-ad4a-17f875176788', // N-Body Gravity
    ],
  },
  {
    number: 4,
    title: 'Computer Vision',
    tagline: 'Teaching a GPU to look at pictures, not just draw them',
    modules: [
      '8d79c6af-784c-4156-8d39-16e66467345e', // Colour Spaces
      '66933805-3a1d-48c0-a287-16d3d7d00016', // Convolution & Filters
      '670eaafa-4b13-4e77-88b3-f2904105c615', // Thresholding & Morphology
      '6901c51a-78cc-43fd-b7cd-6327496ae4f3', // The Canny Edge Pipeline
      'f57b4bed-0519-42f0-a9fb-739679e67957', // Template Matching
      'e85c6dfa-70f9-4abc-b772-3f30f151a121', // Optical Flow
      '4d39e404-8261-48a2-9912-176a3ea0b1bf', // Video Filters
    ],
  },
  {
    // Placed after Computer Vision on purpose: this track's credits point back
    // at it — frequency-domain filtering builds on Convolution & Filters, and
    // the spectrogram's colour ramp on Colour Spaces — so the dependencies read
    // backwards rather than forwards.
    number: 5,
    title: 'Signal Processing',
    tagline: 'Time in, frequency out — and the algorithm that made it practical',
    modules: [
      'ad14836c-62d4-4243-afd8-401694d13c75', // Sampling & Aliasing
      '7b1e3f9b-baf5-4b75-9ad1-3c05f445a3db', // The DFT, Honestly
      'd4375da7-7178-4bee-8442-e04e80d563d1', // The FFT Butterfly
      'f563138d-cbb0-4aa5-b874-ff028b277677', // Windowing & Spectral Leakage
      '8c225e10-d7d6-4473-8099-1c45f40a7668', // Filtering in the Frequency Domain
      '9ecd2295-c9d9-4023-b393-bbdc776a2d77', // Spectrograms
      'b159433f-d1bb-4ed5-8ae3-ef7a3db50f57', // Autocorrelation & Pitch
    ],
  },
  {
    number: 6,
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
