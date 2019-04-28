import $ from 'jquery'
import GPU from 'gpu.js'

export const initBenches = () => {
  $('#benchmark').ready(() => {
    $('#size-val').text('512')
    $('#size').val('512').on('input', () => {
      $('#size-val').text($('#size').val())
    })
  })

}
const benchmarkIt = () => {
  let size = $('#size').val() || 512

  let checkboxes = {
    cpu: $('#cpu').prop('checked'),
    gpujsCpu: $('#gpujscpu').prop('checked'),
    gpu: $('#gpu').prop('checked'),
    gpuTex: $('#gputex').prop('checked'),
    matMult: $('#matmult').prop('checked'),
    conv: $('#conv').prop('checked')
  }

  const kernel = [
    [1, 2, 1],
    [2, 0, 2],
    [1, 2, 1]
  ]

  size = parseInt(size);

  const gpu = new GPU({ mode: 'gpu' }),
    cpu = new GPU({ mode: 'cpu' })

  const matMultFunc = `function (a, b){
    var sum = 0;
    for (var i = 0; i < ${size}; i++){
      sum += a[this.thread.y][i] * b[i][this.thread.x];
    }
    return sum;
  }`

  const matConvFunc = `function (array, kernel) {
    var sum = 0;
    for (var i = 0; i < 3; i++){
      for (var j = 0; j < 3; j++){
        sum += kernel[j][i] * array[this.thread.y + j][this.thread.x + i]
      }
    }
    return sum;
  }`

  const paddificateFunc = `function(array, paddingSize) {
    const positionX = Math.min(Math.max(this.thread.x - paddingSize, 0), ${size} - 1)
    const positionY = Math.min(Math.max(this.thread.y - paddingSize, 0), ${size} - 1)

    return array[positionY][positionX]
  }`

  const cpuMatMult = (arr1, arr2) => {
    let out = []
    for (var i = 0; i < size; i++) {
      out.push([])
      for (var j = 0; j < size; j++) {
        let sum = 0
        for (var k = 0; k < size; k++) {
          sum += arr1[i][k] * arr2[k][j]
        }
        out[i][j] = sum
      }
    }
    return out
  }

  const cpuConv = (array, ker) => {
    let out = []
    for (var i = 0; i < size; i++) {
      out.push([])
      for (var j = 0; j < size; j++) {
        let sum = 0
        for (var k = 0; k < 3; k++) {
          for (var l = 0; l < 3; l++) {
            sum += array[j + l][l + k] * ker[l][k]
          }
        }
        out[i][j] = sum
      }
    }
    return out
  }

  const matGenFunc = `function() {
    return (this.thread.x + this.thread.y + this.thread.z)/11
  }`

  const funcs = {
    generateMatrices: gpu.createKernel(matGenFunc, {
      output: [2, size, size],
      outputToTexture: true
    }),
    
    gpuTexMatMult: gpu.createKernel(matMultFunc, {
      output: [size, size],
      outputToTexture: true
    }),
    gpuMatMult: gpu.createKernel(matMultFunc, {
      output: [size, size],
      outputToTexture: false
    }),
    gpujsCpuMatMult: cpu.createKernel(matMultFunc, { output: [size, size] }),
    cpuMatMult,

    gpuConv: gpu.createKernel(matConvFunc, {
      output: [size, size],
      outputToTexture: false
    }),
    gpuTexConv: gpu.createKernel(matConvFunc, {
      output: [size, size],
      outputToTexture: true
    }),
    gpujsCpuConv: cpu.createKernel(matConvFunc, {
      output: [size, size]
    }),
    cpuConv,
    // PADDING
    padificate: gpu.createKernel(paddificateFunc, {
      output: [size + 2, size + 2],
      outputToTexture: true
    })
  }

  function benchIt(func) {
    let time = -1 * performance.now()
    const ret = func()
    time += performance.now()
    time = Math.floor(time)
    return { time, ret }
  }

  const matMultBenches = {},
    matConvBenches = {}

  funcs.generateMatrices.build()

  const mat = benchIt(function(){
    return funcs.generateMatrices().toArray(gpu)
  })
  const matGen = mat.time,
    matrices = mat.ret

  const pad = {}
  if (checkboxes.conv)
    pad.build = benchIt(function() {
      funcs.padificate.build(matrices[0], 1)
    })
    pad.mat = benchIt(function() {
      return funcs.padificate(matrices[0], 1).toArray(gpu)
    })

  if (checkboxes.matMult) {
    if (checkboxes.gpuTex)
      matMultBenches.gpuTexCompilePerf = benchIt(function() {
        funcs.gpuTexMatMult.build(matrices[0], matrices[1])
      }).time

    if (checkboxes.gpu)
      matMultBenches.gpuCompilePerf = benchIt(function() {
        funcs.gpuMatMult.build(matrices[0], matrices[1])
      }).time

    if (checkboxes.gpujsCpu)
      matMultBenches.cpuCompilePerf = benchIt(function() {
        funcs.gpujsCpuMatMult.build(matrices[0], matrices[1])
      }).time

    if (checkboxes.gpu)
      matMultBenches.gpuPerform = benchIt(function() {
        funcs.gpuMatMult(matrices[0], matrices[1])
      }).time

    if (checkboxes.cpu)
      matMultBenches.cpuPerform = benchIt(function() {
        funcs.cpuMatMult(matrices[0], matrices[1])
      }).time

    if (checkboxes.gpujsCpu)
      matMultBenches.gpujsCpuPerform = benchIt(function() {
        funcs.gpujsCpuMatMult(matrices[0], matrices[1])
      }).time

    if (checkboxes.gpuTex)
      matMultBenches.gpuTexPerform = benchIt(function() {
        funcs.gpuTexMatMult(matrices[0], matrices[1])
      }).time
  }

  // Convolution

  if (checkboxes.conv) {
    if (checkboxes.gpuTex)
      matConvBenches.gpuTexCompilePerf = benchIt(function() {
        funcs.gpuTexConv.build(pad.mat.ret, kernel)
      }).time

    if (checkboxes.gpu)
      matConvBenches.gpuCompilePerf = benchIt(function() {
        funcs.gpuConv.build(pad.mat.ret, kernel)
      }).time

    if (checkboxes.gpujsCpu)
      matConvBenches.cpuCompilePerf = benchIt(function() {
        funcs.gpujsCpuConv.build(pad.mat.ret, kernel)
      }).time

      if (checkboxes.gpu)
      matConvBenches.gpuPerform = benchIt(function() {
        funcs.gpuConv(pad.mat.ret, kernel)
      }).time

    if (checkboxes.cpu)
      matConvBenches.cpuPerform = benchIt(function() {
        funcs.cpuConv(pad.mat.ret, kernel)
      }).time

    if (checkboxes.gpujsCpu)
      matConvBenches.gpujsCpuPerform = benchIt(function() {
        funcs.gpujsCpuConv(pad.mat.ret, kernel)
      }).time

      if (checkboxes.gpuTex)
      matConvBenches.gpuTexPerform = benchIt(function() {
        funcs.gpuTexConv(pad.mat.ret, kernel)
      }).time
  }

  $('#out').html(`
    <p><b>Matrix Generation Time</b>: ${matGen}ms</p>
    ${checkboxes.conv ? `<p><b>Matrix Padding Time</b>: ${pad.mat.time}ms</p>` : ''}
    ${
      checkboxes.matMult
        ? `<h5>Matrix Multiplication</h5>
    <table class="centered">
      <thead>
        <tr>
          <th>Benchmark</th>
          <th>Time Taken</th>
          <th>Compile Time</th>
          <th>Total</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>CPU</td>
          <td>${
            checkboxes.cpu ? matMultBenches.cpuPerform + 'ms' : 'Not Benchmarked'
          }</td>
          <td>No Compilation Required</td>
          <td>${
            checkboxes.cpu ? matMultBenches.cpuPerform + 'ms' : 'Not Benchmarked'
          }
        </tr>
        <tr>
          <td>CPU(GPUjs)</td>
          <td>${
            checkboxes.gpujsCpu
              ? matMultBenches.gpujsCpuPerform + 'ms'
              : 'Not Benchmarked'
          }</td>
          <td>${
            checkboxes.gpujsCpu
              ? matMultBenches.cpuCompilePerf + 'ms'
              : 'Not Benchmarked'
          }</td>
          <td>${
            checkboxes.gpujsCpu
              ? matMultBenches.cpuCompilePerf +
                matMultBenches.gpujsCpuPerform +
                'ms'
              : 'Not Benchmarked'
          }</td>
        </tr>
        <tr>
          <td>GPU</td>
          <td>${
            checkboxes.gpu ? matMultBenches.gpuPerform + 'ms' : 'Not Benchmarked'
          }</td>
          <td>${
            checkboxes.gpu
              ? matMultBenches.gpuCompilePerf + 'ms'
              : 'Not Benchmarked'
          }</td>
          <td>${
            checkboxes.gpu
              ? matMultBenches.gpuCompilePerf + matMultBenches.gpuPerform + 'ms'
              : 'Not Benchmarked'
          }</td>
        </tr>
        <tr>
          <td>GPU(Texture Mode)</td>
          <td>${
            checkboxes.gpuTex
              ? matMultBenches.gpuTexPerform + 'ms'
              : 'Not Benchmarked'
          }</td>
          <td>${
            checkboxes.gpuTex
              ? matMultBenches.gpuTexCompilePerf + 'ms'
              : 'Not Benchmarked'
          }</td>
          <td>${
            checkboxes.gpuTex
              ? matMultBenches.gpuTexCompilePerf +
                matMultBenches.gpuTexPerform +
                'ms'
              : 'Not Benchmarked'
          }</td>
        </tr>
      </tbody>
    </table>`
        : ''
    }
  ${
      checkboxes.conv
        ? `<h5>Kernel Convolution</h5>
    <table class='striped centered'>
      <thead>
        <tr>
          <th>Benchmark</th>
          <th>Time Taken</th>
          <th>Compile Time</th>
          <th>Total</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>CPU</td>
          <td>${
            checkboxes.cpu ? matConvBenches.cpuPerform + 'ms' : 'Not Benchmarked'
          }</td>
          <td>Not Applicable</td>
          <td>${
            checkboxes.cpu ? matConvBenches.cpuPerform + 'ms' : 'Not Benchmarked'
          }</td>
        </tr>
        <tr>
          <td>CPU(GPUjs)</td>
          <td>${
            checkboxes.gpujsCpu
              ? matConvBenches.gpujsCpuPerform + 'ms'
              : 'Not Benchmarked'
          }</td>
          <td>${
            checkboxes.gpujsCpu
              ? matConvBenches.cpuCompilePerf + 'ms'
              : 'Not Benchmarked'
          }</td>
          <td>${
            checkboxes.gpujsCpu
              ? matConvBenches.cpuCompilePerf +
                matConvBenches.gpujsCpuPerform +
                'ms'
              : 'Not Benchmarked'
          }</td>
        </tr>
        <tr>
          <td>GPU</td>
          <td>${
            checkboxes.gpu ? matConvBenches.gpuPerform + 'ms' : 'Not Benchmarked'
          }</td>
          <td>${
            checkboxes.gpu
              ? matConvBenches.gpuCompilePerf + 'ms'
              : 'Not Benchmarked'
          }</td>
          <td>${
            checkboxes.gpu
              ? matConvBenches.gpuCompilePerf + matConvBenches.gpuPerform + 'ms'
              : 'Not Benchmarked'
          }</td>
        </tr>
        <tr>
          <td>GPU(Texture Mode)</td>
          <td>${
            checkboxes.gpuTex
              ? matConvBenches.gpuTexPerform + 'ms'
              : 'Not Benchmarked'
          }</td>
          <td>${
            checkboxes.gpuTex
              ? matConvBenches.gpuTexCompilePerf + 'ms'
              : 'Not Benchmarked'
          }</td>
          <td>${
            checkboxes.gpuTex
              ? matConvBenches.gpuTexCompilePerf +
                matConvBenches.gpuTexPerform +
                'ms'
              : 'Not Benchmarked'
          }</td>
        </tr>
      </tbody>
    </table>`
        : ''
    }
  `)
}
export const benchIt = () => {
  benchmarkIt()
}

export default benchIt