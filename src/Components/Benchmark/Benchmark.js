import React, { Component } from 'react'
import { Button, Range, Checkbox, Row, Col } from 'react-materialize'
import getActiveElems from '../../utils/getActiveElems'
import Graph from '../Util/Graph/Graph'
import sizes, {obj} from '../../Data/different-sizes/gt1030-firefox'
import benchmark from '@gpujs/benchmark'
import $ from 'jquery'
import ScrollButton from '../ScrollButton/ScrollButton'
// import getDb from '../../db/firebase'

import './Benchmark.scss'

class Benchmark extends Component {

  state = {
    active: {}
  }

  componentDidMount() {
    $(document).on('DOMContentLoaded scroll', () => {
      const ids = {
        benchmark: {
          id: 'benchmark',
          thresh: 600
        },
        sizes: {
          id: 'sizes',
          thresh: 500
        }
      }

      this.setState({
        active: getActiveElems(ids)
      })
    })
  }

  handleDB() {
    // const db = getDb();
  }

  componentWillUnmount() {
    $(document).off('DOMContentLoaded scroll')
  }

  benchmarkFormHandler = (e) => {
    e.preventDefault();

    const size = $('#size').val() || 256,
      num_benchmarks = parseInt($('#num-bench').val()),
      cpu = $('#cpu').prop('checked');

    const bench = benchmark({
      num_benchmarks,
      matrix_size: size,
      logs: false,
      cpu_benchmark: cpu,
    })

    const performerMap = {
      cpu: 'CPU',
      gpu: 'GPU',
      pipe: 'GPU(Pipeline Mode)'
    }

    const tidyNumber = (num) => num > 0 ? num : 'less than 1';

    $('#out').html(`
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

      <h5>Overall</h5>

      <table class="centered hightlight striped responsive-table">
        <thead>
          <tr>
            <th>Benchmark</th>
            <th>Best Performer</th>
            <th>Worst Performer</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>
              Matrix Multiplication
            </td>
            <td>
              <b>${performerMap[bench.stats.overall.mat_mult.best_performer]}</b>
            </td>
            <td>
              <b>${performerMap[bench.stats.overall.mat_mult.worst_performer]}</b>
            </td>
          </tr>
          <tr>
            <td>
              Matrix Convolution
            </td>
            <td>
              <b>${performerMap[bench.stats.overall.mat_conv.best_performer]}</b>
            </td>
            <td>
              <b>${performerMap[bench.stats.overall.mat_conv.worst_performer]}</b>
            </td>
          </tr>
        </tbody>
      </table>

    <br /><br />`)
  }

  sizeChangeHandler = () => {
    $('#size-val').text($('#size').val())
    if ($('#size').val() >= 2500) $('#cpu').prop('checked', false).prop('disabled', true)
    else $('#cpu').prop('checked', false).prop('disabled', false)
  }

  render(){
    return (
      <div id="benchmark">
        <ScrollButton />
        <h2 className="center">Benchmark</h2>
        <hr />
        <div className="benchmark-container">
          <p className="center"><b>GPU.js version:</b> &nbsp;v2.0.0</p>

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
            <Button waves="light" id="bench" className="blue lighten-1" >
              benchmark!
            </Button>
          </form>
          <br /><br />
          <div id="out"></div>
        </div>

        <h3 className="center">Common Benchmarks</h3>
        <div id="sizes" className="center">
          <p>Here is a chart representing the performance of matrix multiplication of different arrays and different modes (lower is better)</p>
        </div>

        <div>
          <Row>
            <Col offset="s1" s={10}>
              <ul>
                <li><b>Hardware:</b> &nbsp;i5-7500 + GT1030</li>
                <li><b>Operating System:</b> &nbsp;Ubuntu 18.04.2 LTS (64-bit)</li>
                <li><b>Environment:</b> &nbsp;NodeJS v10.15.3 (64-bit) + gpu.js v2.0.0-rc.13</li>
                <li>
                  <b>Benchmarks:</b>
                  <ul>
                    <li><span className="browser-color" style={{backgroundColor: obj.cpu.lineColor}}></span> CPU</li>
                    <li><span className="browser-color" style={{backgroundColor: obj.gpu.lineColor}}></span> GPU</li>
                    <li><span className="browser-color" style={{backgroundColor: obj.pipe.lineColor}}></span> GPU(pipeline mode)</li>
                  </ul>
                </li>
                <li><b>Last Updated:</b> &nbsp;12 May 2019</li>
                <li>The Benchmarks were run using a tool called <a href="https://github.com/gpujs/benchmark"><b>@gpujs/benchmark</b></a> which is created by the gpu.js org to specifically benchmark gpu.js.</li>
              </ul>
            </Col>
          </Row>
        </div>

        <div className="center">
          <Graph info={sizes} title={{
            x: 'matrix size',
            y: 'time taken (in ms)'
          }}
          interpolation={true}
          />
        </div>
      </div>
    )
  }
}

export default Benchmark