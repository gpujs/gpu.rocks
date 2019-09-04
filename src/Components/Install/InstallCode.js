export default {
  linux: `
  sudo apt install mesa-common-dev libxi-dev
  `,

  npm: `
  npm install gpu.js
  `,

  yarn: `
  yarn add gpu.js
  `,

  node: `
  const { GPU } = require('gpu.js');
  const gpu = new GPU();
  `,

  type: `
  import { GPU } from 'gpu.js';
  const gpu = new GPU();
  `,

  browser: `
  <script src="path/to/gpu-browser.min.js"></script>
  <script>
      const gpu = new GPU();
  </script>
  `
}