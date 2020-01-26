import React from 'react'

// Images
import shadedRelief from '../../img/examples/shaded-relief.png'
import RPS from '../../img/examples/rock-paper-scissors.png'
import videoConv from '../../img/examples/video-conv.png'
import canvas from '../../img/examples/canvas.png'
import laserDetection from '../../img/examples/laser-detection.png'
import inverse from '../../img/examples/inverse.png'
import mandelbulb from '../../img/examples/mandelbulb.png'
import loop from '../../img/examples/loop.png'
import voronoi from '../../img/examples/voronoi.png'
import djikstra from '../../img/examples/djikstra.png'
import heatmap from '../../img/examples/heatmap.png'
import loadImage from '../../img/examples/load-img.png'
import conv from '../../img/examples/conv.png'
import CAPoC from '../../img/examples/CA-PoC.png'
import rasterProjection from '../../img/examples/raster-projection.png'
import helloV2 from '../../img/fire1.gif'
import mandelbrotSet from '../../img/examples/mandelbrot-set.png'
import juliaSet from '../../img/examples/julia-set.png'
import slowFade from '../../img/examples/slow-fade.png'

const links = [
  {
    img: helloV2,
    author: {
      name: 'Fil',
      link: 'https://observablehq.com/@fil'
    },
    title: 'Hello, GPU.js v2!',
    description: 'An example with the shiny new v2 of GPU.js, "Cosmic Jellyfish"',
    footerLinks: [
      <a href="https://observablehq.com/@fil/hello-gpu-js-v2">Observable Notebook</a>,
      <a href="https://github.com/gpujs/gpu.js/releases/tag/2.0.0">What's New</a>
    ]
  },
  {
    img: loadImage,
    title: 'Image to GPU.js',
    author: {
      name: 'Fil',
      link: 'https://observablehq.com/@fil'
    },
    description: 'A simple example to load an image into GPU.js.',
    footerLinks: [
      <a href="https://observablehq.com/@fil/image-to-gpu">Observable Notebook</a>
    ]
  },
  {
    img: canvas,
    title: 'GPU.js Canvas',
    author: {
      name: 'Alf Eaton',
      link: 'https://observablehq.com/@hubgit'
    },
    description: 'A very simple canvas example with GPU.js',
    footerLinks: [
      <a href="https://observablehq.com/@hubgit/gpu-js-canvas">Observable Notebook</a>
    ]
  },
  {
    img: slowFade,
    title: 'Slow Fade',
    description: 'A simple example wherein colors slowly fade in and fade out.',
    footerLinks: [
      <a href="https://observablehq.com/@robertleeplummerjr/gpu-js-example-slow-fade">Observable Notebook</a>
    ]
  },
  {
    img: loop,
    author: {
      name: 'Fil',
      link: 'https://observablehq.com/@fil'
    },
    title: 'The GPU.js Loop',
    description: 'Different colors changing in a loop.',
    footerLinks: [
      <a href="https://observablehq.com/@fil/the-gpu-js-loop">Observable Notebook</a>
    ]
  },
  {
    img: juliaSet,
    author: {
      name: 'UKABUER',
      link: 'https://observablehq.com/@ukabuer'
    },
    title: 'Julia Set',
    description: 'The plot of a particular set of complex numbers called the Julia set (which is a fractal).',
    footerLinks: [
      <a href="https://observablehq.com/@ukabuer/julia-set-fractal-using-gpu-js">Observable Notebook</a>,
      <a href="https://en.wikipedia.org/wiki/Julia_set">Read More</a>
    ]
  },
  {
    img: mandelbrotSet,
    title: 'Mandelbrot Set',
    description: 'The plot of a particular set of complex numbers called the Mandelbrot set (which is a fractal).',
    footerLinks: [
      <a href="https://observablehq.com/@robertleeplummerjr/gpu-js-example-mandelbrot-set">Observable Notebook</a>,
      <a href="https://en.wikipedia.org/wiki/Mandelbrot_set">Read More</a>
    ]
  },
  {
    img: mandelbulb,
    title: 'Mandelbulb',
    description: 'The Mandelbulb is a three-dimensional fractal, constructed by Daniel White and Paul Nylander.',
    footerLinks: [
      <a href="https://observablehq.com/@robertleeplummerjr/gpu-js-example-mandelbulb">Observable Notebook</a>,
      <a href="https://en.wikipedia.org/wiki/Mandelbulb">Read More</a>
    ]
  },
  {
    img: rasterProjection,
    author: {
      name: 'Fil',
      link: 'https://observablehq.com/@fil'
    },
    title: 'Raster Projection',
    description: 'An equirectangular image of the world, reprojected as an azimuthal equal-area image',
    footerLinks: [
      <a href="https://observablehq.com/@fil/raster-projection-with-gpu-js">Observable Notebook</a>
    ]
  },
  {
    img: CAPoC,
    author: {
      name: 'Alex Lamb',
      link: 'https://observablehq.com/@alexlamb'
    },
    title: 'CA Proof of Concept',
    description: 'A very basic GPU.js proof of concept cellular automaton with an accidentally discovered update function.',
    footerLinks: [
      <a href="https://observablehq.com/@alexlamb/gpu-js-ca-proof-of-concept">Observable Notebook</a>
    ]
  },
  {
    img: RPS,
    author: {
      name: 'Alax Lamb',
      link: 'https://observablehq.com/@alexlamb'
    },
    title: 'Rock Paper Scissors',
    description: 'Another GPU.js proof of concept. This time with a multi-stage update function.',
    footerLinks: [
      <a href="https://observablehq.com/@alexlamb/gpu-rock-paper-scissors">Observable Notebook</a>
    ]
  },
  {
    img: conv,
    author: {
      name: 'UKABUER',
      link: 'https://observablehq.com/@ukabuer'
    },
    title: 'Image Convolution',
    description: 'Basic Parallel Image Convolution using GPU.js.',
    footerLinks: [
      <a href="https://observablehq.com/@ukabuer/image-convolution-using-gpu-js">Observable Notebook</a>
    ]
  },
  {
    img: videoConv,
    author: {
      name: 'UKABUER',
      link: 'https://observablehq.com/@ukabuer'
    },
    title: 'Video Convolution',
    description: 'Parallel Video Convolution using GPU.js.',
    footerLinks: [
      <a href="https://observablehq.com/@robertleeplummerjr/video-convolution-using-gpu-js">Observable Notebook</a>
    ]
  },
  {
    img: heatmap,
    author: {
      name: 'Wenbo Tao',
      link: 'https://observablehq.com/@tracyhenry'
    },
    title: 'Heatmap',
    description: 'GPU Accelerated Heatmap using GPU.js',
    footerLinks: [
      <a href="https://observablehq.com/@tracyhenry/gpu-accelerated-heatmap-using-gpu-js">Observable Notebook</a>
    ]
  },
  {
    img: djikstra,
    author: {
      name: 'Fil',
      link: 'https://observablehq.com/@fil'
    },
    title: 'Djikstra Search',
    description: 'Computing Dijkstra’s shortest paths on the GPU',
    footerLinks: [
      <a href="https://observablehq.com/@fil/dijkstras-algorithm-in-gpu-js">Observable Notebook</a>
    ]
  },
  {
    img: voronoi,
    author: {
      name: 'Fil',
      link: 'https://observablehq.com/@fil'
    },
    title: 'Voronoi',
    description: 'A Voronoi diagram is a partition of a plane into regions close to each of a given set of objects.',
    footerLinks: [
      <a href="https://observablehq.com/@fil/voronoi-with-gpu-js">Observable Notebook</a>,
      <a href="https://en.wikipedia.org/wiki/Voronoi_diagram">Read More</a>
    ]
  },
  {
    img: inverse,
    author: {
      name: 'Roger Veciana i Rovira',
      link: 'https://observablehq.com/@rveciana'
    },
    title: 'Inverse Distance',
    description: 'This is the calculation of the inverse of the distance with GPU.js.',
    footerLinks: [
      <a href="https://observablehq.com/@rveciana/inverse-of-the-distance-with-gpu-js">Observable Notebook</a>
    ]
  },
  {
    img: laserDetection,
    author: {
      name: 'Brian Hann',
      link: 'https://observablehq.com/@c0bra'
    },
    title: 'Laser Detection',
    description: 'GPU.js laser detection v2.',
    footerLinks: [
      <a href="https://observablehq.com/@robertleeplummerjr/gpu-js-laser-detection-v2">Observable Notebook</a>
    ]
  },
  {
    img: shadedRelief,
    author: {
      name: 'Roger Veciana i Rovira',
      link: 'https://observablehq.com/@rveciana'
    },
    title: 'Shaded Relief',
    description: 'The hillshade data is calculated using GPU and drawn directly into a canvas element.',
    footerLinks: [
      <a href="https://observablehq.com/@rveciana/shaded-relief-with-gpujs-and-d3js/2">Observable Notebook</a>
    ]
  }
]

export default links