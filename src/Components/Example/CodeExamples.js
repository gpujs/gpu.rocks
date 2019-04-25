export const generateMatrices = 
`
  const generateMatrices = () => {
    const matrices = [[], []]
    for (let y = 0; y < 512; y++){
      matrices[0].push([])
      matrices[1].push([])
      for (let x = 0; x < 512; x++){
        matrices[0][y].push(Math.random())
        matrices[1][y].push(Math.random())
      }
    }
    return matrices;
  }

`

export const createKernel = 
`
  const gpu = new GPU();
  const multiplyMatrix = gpu.createKernel(function(a, b) {
    var sum = 0;
    for (var i = 0; i < 512; i++) {
      sum += a[this.thread.y][i] * b[i][this.thread.x];
    }
    return sum;
  }).setOutput([512, 512]);

`

const exampleCode = {
  generateMatrices,
  createKernel
}

export default exampleCode