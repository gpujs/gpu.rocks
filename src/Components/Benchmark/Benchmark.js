import React, { Component } from 'react'
import Button from 'react-materialize/lib/Button'
import Range from 'react-materialize/lib/Range'
import Checkbox from 'react-materialize/lib/Checkbox'
import Row from 'react-materialize/lib/Row'
import Col from 'react-materialize/lib/Col'

import Graph from '../Util/Graph/Graph'
import sizes from '../../Data/gt1030/gt1030-node'
import { benchmark } from '@gpujs/benchmark'
import ScrollButton from '../ScrollButton/ScrollButton'

import './Benchmark.scss'

class Benchmark extends Component {
  benchmarkFormHandler = (e) => {
    e.preventDefault();

    const size = document.querySelector('#size').value || 256,
      num_benchmarks = parseInt(document.querySelector('#num-bench').value),
      cpu = document.querySelector('#cpu').checked

    const bench = benchmark({
      num_benchmarks,
      matrix_size: size,
      logs: false,
      cpu_benchmark: cpu,
    }).getData()

    const performerMap = {
      cpu: 'CPU',
      gpu: 'GPU',
      pipe: 'GPU(Pipeline Mode)'
    }

    const tidyNumber = (num) => num > 0 ? num : 'less than 1';

    document.querySelector('#out').innerHTML = `
      <h4>Score</h4>
      <table class="striped responsive-table highlight centered">
        <thead>
          <tr>
            <th>
              Mode
            </th>
            <th>
              Score
            </th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>
              GPU
            </td>
            <td>
              ${bench.score['gpu']}
            </td>
          </tr>
          <tr>
            <td>
              CPU
            </td>
            <td>
              ${cpu ? bench.score['cpu'] : 'Not Benchmarked'}
            </td>
          </tr>
        </tbody>
      </table>
      <hr />

      <h4>Build Times</h4>
      <table class="striped responsive-table highlight centered">
        <thead>
          <tr>
            <th>
              Benchmark
            </th>
            <th>
              Time Taken(GPU)
            </th>
            <th>
              Time Taken(PIPELINE)
            </th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>
              Matrix Multiplication
            </td>
            <td>
              ${tidyNumber(bench.build_time.mat_mult['gpu'])} ms
            </td>
            <td>
              ${tidyNumber(bench.build_time.mat_mult['pipe'])} ms
            </td>
          </tr>
          <tr>
            <td>
              Matrix Convolution
            </td>
            <td>
              ${tidyNumber(bench.build_time.mat_conv['gpu'])} ms
            </td>
            <td>
              ${tidyNumber(bench.build_time.mat_conv['pipe'])} ms
            </td>
          </tr>
        </tbody>
      </table>
      <hr />

      <h4>Run Times</h4>
      <table class="striped responsive-table highlight centered">
        <thead>
          <tr>
            <th>
              Benchmark
            </th>
            <th>
              Time Taken(GPU)
            </th>
            <th>
              Time Taken(CPU)
            </th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>
              Matrix Multiplication
            </td>
            <td>
              <b>Min</b>: ${tidyNumber(bench.run_time.mat_mult['gpu'].min)} ms<br />
              <b>Max</b>: ${tidyNumber(bench.run_time.mat_mult['gpu'].max)} ms<br />
              <b>Avg</b>: ${tidyNumber(bench.run_time.mat_mult['gpu'].avg)} ms<br />
            </td>
            <td>
              ${cpu ? `
                <b>Min</b>: ${tidyNumber(bench.run_time.mat_mult['cpu'].min)} ms<br />
                <b>Max</b>: ${tidyNumber(bench.run_time.mat_mult['cpu'].max)} ms<br />
                <b>Avg</b>: ${tidyNumber(bench.run_time.mat_mult['cpu'].avg)} ms<br />
              ` : `Not Benchmarked`}
            </td>
          </tr>
          <tr>
            <td>
              Matrix Convolution
            </td>
            <td>
              <b>Min</b>: ${tidyNumber(bench.run_time.mat_conv['gpu'].min)} ms<br />
              <b>Max</b>: ${tidyNumber(bench.run_time.mat_conv['gpu'].max)} ms<br />
              <b>Avg</b>: ${tidyNumber(bench.run_time.mat_conv['gpu'].avg)} ms<br />
            </td>
            <td>
              ${cpu ? `
                <b>Min</b>: ${tidyNumber(bench.run_time.mat_conv['cpu'].min)} ms<br />
                <b>Max</b>: ${tidyNumber(bench.run_time.mat_conv['cpu'].max)} ms<br />
                <b>Avg</b>: ${tidyNumber(bench.run_time.mat_conv['cpu'].avg)} ms<br />
              ` : `Not Benchmarked`}
            </td>
          </tr>
        </tbody>
      </table>
      <hr />

      <h4>Pipelining Benchmark</h4>
      <table class="striped responsive-table highlight centered">
        <thead>
          <tr>
            <th>
              Benchmark
            </th>
            <th>
              Time Taken(GPU)
            </th>
            <th>
              Time Taken(CPU)
            </th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>
              Matrix Multiplication
            </td>
            <td>
              <b>Min</b>: ${tidyNumber(bench.run_time.pipe['gpu'].min)} ms<br />
              <b>Max</b>: ${tidyNumber(bench.run_time.pipe['gpu'].max)} ms<br />
              <b>Avg</b>: ${tidyNumber(bench.run_time.pipe['gpu'].avg)} ms<br />
            </td>
            <td>
              ${cpu ? `
                <b>Min</b>: ${tidyNumber(bench.run_time.pipe['cpu'].min)} ms<br />
                <b>Max</b>: ${tidyNumber(bench.run_time.pipe['cpu'].max)} ms<br />
                <b>Avg</b>: ${tidyNumber(bench.run_time.pipe['cpu'].avg)} ms<br />
              ` : `Not Benchmarked`}
            </td>
          </tr>
        </tbody>
      </table>
      <hr />

      <h4>Statistics</h4>
      <h5>Build Times</h5>

      <table class="centered hightlight striped responsive-table">
        <thead>
          <tr>
            <th>Benchmark</th>
            <th>GPU v/s PIPELINE</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>
              Matrix Multiplication
            </td>
            <td>
              <b>${performerMap[bench.stats.build_time.mat_mult.diff.gpu_pipe.winner]}</b> took ${bench.stats.build_time.mat_mult.diff.gpu_pipe.percentage}% less time to compile
            </td>
          </tr>
          <tr>
            <td>
              Matrix Convolution
            </td>
            <td>
              <b>${performerMap[bench.stats.build_time.mat_conv.diff.gpu_pipe.winner]}</b> took ${bench.stats.build_time.mat_conv.diff.gpu_pipe.percentage}% less time to compile
            </td>
          </tr>
        </tbody>
      </table>
      <hr />

      <h5>Run Times</h5>

      <table class="centered hightlight striped responsive-table">
        <thead>
          <tr>
            <th>Benchmark</th>
            <th>GPU v/s CPU</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>
              Matrix Multiplication
            </td>
            <td>
              <b>${ cpu ?
                `${performerMap[bench.stats.run_time.mat_mult.diff.cpu_gpu.avg.winner]}</b> took ${bench.stats.run_time.mat_mult.diff.cpu_gpu.avg.percentage}% less time` : 
                `Not Benchmarked`
                }
            </td>
          </tr>
          <tr>
            <td>
              Matrix Convolution
            </td>
            <td>
              <b>${ cpu ?
                `${performerMap[bench.stats.run_time.mat_conv.diff.cpu_gpu.avg.winner]}</b> took ${bench.stats.run_time.mat_conv.diff.cpu_gpu.avg.percentage}% less time` : 
                `Not Benchmarked`
                }
            </td>
          </tr>
        </tbody>
      </table>
    <hr />
    <br /><br />`
  }

  sizeChangeHandler = () => {
    const size = document.querySelector('#size')
    const cpu = document.querySelector('#cpu')
    document.querySelector('#size-val').textContent = size.value
    if (size.value >= 2500) {
      cpu.checked = false
      cpu.disabled = true
    } else {
      cpu.checked = false
      cpu.disabled = false
    }
  }

  render(){
    return (
      <div id="benchmark">
        <ScrollButton />
        <h2 className="center">Benchmark</h2>
        <hr />
        <div className="benchmark-container">
          <p className="center"><b>GPU.js version:</b> &nbsp;v2.1.0</p>

          <form id="benchmark-form" onSubmit={this.benchmarkFormHandler}>
            <div className="input-field">
              <label htmlFor="size">Size of Matrix(uniform) -> <b><span id="size-val">256</span></b></label><br />
              <Range id="size" defaultValue="256" min="1" max="4096" onInput={this.sizeChangeHandler}/>
            </div>
            <div className="input-field">
              <label htmlFor="num-bench">Number of Benchmarks</label><br />
              <input id="num-bench" type="number" defaultValue={2}/>
            </div>
            <h6> Select Modes</h6>
            <p>
              <Checkbox label="GPU" value="GPU" id="gpu" filledIn checked disabled />
            </p>
            <p>
              <Checkbox label="GPU(Pipeline Mode)" value="GPU(pipe)" id="pipe" filledIn checked disabled />
            </p>
            <p>
              <Checkbox label="CPU" value="CPU" id="cpu" filledIn />
            </p>
            <Button waves="light" id="bench" className="purple darken-3" >
              benchmark!
            </Button>
          </form>
          <br /><br />
          <div id="out"></div>
        </div>

        <h3 className="center">Common Benchmarks</h3>

        <p style={{paddingLeft: '0.75rem'}}>Here is a chart representing the time taken (GPU v/s CPU) of matrix multiplication of different arrays and different modes.</p>
        <div>
          <Row>
            <Col s={12}>
              <ul>
                <li><b>Hardware:</b> &nbsp;<a href="https://ark.intel.com/content/www/us/en/ark/products/97123/intel-core-i5-7500-processor-6m-cache-up-to-3-80-ghz.html">i5-7500</a> + <a href="https://www.geforce.com/hardware/desktop-gpus/geforce-gt-1030/specifications">GT 1030</a></li>
                <li><b>Operating System:</b> &nbsp;<a href="https://ubuntu.com">Ubuntu</a> 18.04.3 LTS (64-bit)</li>
                <li><b>Environment:</b> &nbsp;NodeJS v10.15.3 + <b>GPU.js</b> v2.0.1</li>
                <li>
                  <b>Benchmarks:</b>
                  <ul>
                    <li><span className="bench-color" style={{backgroundColor: sizes.cpu_run_time_mat_mult.colors.lineColor}}></span> CPU</li>
                    <li><span className="bench-color" style={{backgroundColor: sizes.gpu_run_time_mat_mult.colors.lineColor}}></span> GPU</li>
                  </ul>
                </li>
                <li><b>Last Updated:</b> &nbsp;1 November 2019</li>
                <li>The Benchmarks were run using a tool called <a href="https://github.com/gpujs/benchmark"><b>@gpujs/benchmark</b></a> which is created by the <b>GPU.js</b> org to specifically benchmark <b>GPU.js</b>.</li>
              </ul>
            </Col>
          </Row>
        </div>

        <div className="center">
          <Graph 
            info={
              {
                data: [
                  {
                    id: 'GPU Run Time',
                    color: sizes.gpu_run_time_mat_mult.colors.lineColor,
                    data: sizes.gpu_run_time_mat_mult.series
                  },
                  {
                    id: 'CPU Run Time',
                    color: sizes.cpu_run_time_mat_mult.colors.lineColor,
                    data: sizes.cpu_run_time_mat_mult.series
                  }
                ]
              }
            }
            title={{
              x: 'Matrix Size',
              y: 'Time Taken (in ms)'
            }}
          />
        </div>
      </div>
    )
  }
}

export default Benchmark
