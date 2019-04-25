const generateMatrices = 
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
const exampleCode = {
  generateMatrices
}

export default exampleCode