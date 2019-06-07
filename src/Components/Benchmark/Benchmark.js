import React, { Component } from 'react'
import { Container, Button, Range, Checkbox, Row, Col } from 'react-materialize'
import Heading from '../Heading/Heading'
import getActiveElems from '../../utils/getActiveElems'
import Graph from '../Graph/Graph'
import sizes, {obj} from '../../Data/different-sizes/gt1030-firefox'
import benchmark from '@gpujs/benchmark'
import $ from 'jquery'

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

  benchmarkFormHandler = (e) => {
    e.preventDefault();

    const size = $('#size').val() || 128,
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
    console.log(bench.stats)

    $('#out').html(`
    <h5>Initialization</h5>
    <Table>
      <thead>
        <tr>
          <th>
            Task
          </th>
          <th>
            Time
          </th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>Matrix Generation</td>
          <td>${tidyNumber(bench.mat_gen)}ms</td>
        </tr>
        <tr>
          <td>Matrix Padding</td>
          <td>${tidyNumber(bench.mat_pad)}ms</td>
        </tr>
      </tbody>
    </Table>

    <h5>Compile Time Performance</h5>
    <Table>
      <thead>
        <tr>
          <th>
            Benchmark
          </th>
          <th>
            GPU
          </th>
          <th>
            GPU(Pipeline Mode)
          </th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>Matrix Multiplication</td>
          <td>${tidyNumber(bench.build_time.mat_mult.gpu)}ms</td>
          <td>${tidyNumber(bench.build_time.mat_mult.pipe)}ms</td>
        </tr>
        <tr>
          <td>Matrix Convolution</td>
          <td>${tidyNumber(bench.build_time.mat_conv.gpu)}ms</td>
          <td>${tidyNumber(bench.build_time.mat_conv.pipe)}ms</td>
        </tr>
      </tbody>
    </Table>

    <h5>Run Time Performance</h5>
    <Table>
      <thead>
        <tr>
          <th>
            Benchmark
          </th>
          <th>
            CPU
          </th>
          <th>
            GPU
          </th>
          <th>
            GPU(Pipeline Mode)
          </th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>Matrix Multiplication</td>
          <td>
            ${bench.run_time.mat_mult.cpu.min === -1 ? 'Not Benchmarked' : `
              min: ${tidyNumber(bench.run_time.mat_mult.cpu.min)}ms<br>
              max: ${tidyNumber(bench.run_time.mat_mult.cpu.max)}ms<br>
              avg: ${tidyNumber(bench.run_time.mat_mult.cpu.avg)}ms<br>
            `}
          </td>
          <td>
            min: ${tidyNumber(bench.run_time.mat_mult.gpu.min)}ms<br>
            max: ${tidyNumber(bench.run_time.mat_mult.gpu.max)}ms<br>
            avg: ${tidyNumber(bench.run_time.mat_mult.gpu.avg)}ms<br>
          </td>
          <td>
            min: ${tidyNumber(bench.run_time.mat_mult.pipe.min)}ms<br>
            max: ${tidyNumber(bench.run_time.mat_mult.pipe.max)}ms<br>
            avg: ${tidyNumber(bench.run_time.mat_mult.pipe.avg)}ms<br>
          </td>
        </tr>
        <tr>
          <td>Matrix Convolution</td>
          <td>
            ${bench.run_time.mat_conv.cpu.min === -1 ? 'Not Benchmarked' : `
              min: ${tidyNumber(bench.run_time.mat_conv.cpu.min)}ms<br>
              max: ${tidyNumber(bench.run_time.mat_conv.cpu.max)}ms<br>
              avg: ${tidyNumber(bench.run_time.mat_conv.cpu.avg)}ms<br>
            `}
          </td>
          <td>
            min: ${tidyNumber(bench.run_time.mat_conv.gpu.min)}ms<br>
            max: ${tidyNumber(bench.run_time.mat_conv.gpu.max)}ms<br>
            avg: ${tidyNumber(bench.run_time.mat_conv.gpu.avg)}ms<br>
          </td>
          <td>
            min: ${tidyNumber(bench.run_time.mat_conv.pipe.min)}ms<br>
            max: ${tidyNumber(bench.run_time.mat_conv.pipe.max)}ms<br>
            avg: ${tidyNumber(bench.run_time.mat_conv.pipe.avg)}ms<br>
          </td>
        </tr>
      </tbody>
    </Table>

    <h5>Run Time Statistics</h5>
    <Table>
      <thead>
        <tr>
          <th>
            Benchmark
          </th>
          <th>
            Best Performer
          </th>
          <th>
            Worst Performer
          </th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>Matrix Multiplication</td>
          <td>
            ${performerMap[bench.stats.run_time.mat_mult.best_performer]}
          </td>
          <td>
            ${performerMap[bench.stats.run_time.mat_mult.worst_performer]}
          </td>
        </tr>
        <tr>
          <td>Matrix Convolution</td>
          <td>
            ${performerMap[bench.stats.run_time.mat_conv.best_performer]}
          </td>
          <td>
            ${performerMap[bench.stats.run_time.mat_conv.worst_performer]}
          </td>
        </tr>
      </tbody>
    </Table>
    `)
  }

  sizeChangeHandler = () => {
    $('#size-val').text($('#size').val())
  }
  
  render(){
    return (
      <div id="benchmark">
        <Heading active={this.state.active.benchmark} >Benchmarks</Heading>
        <Container>
          <p className="center"><b>GPU.js version:</b> &nbsp;v2.0.0</p>

          <form id="benchmark-form" onSubmit={this.benchmarkFormHandler}>
            <div className="input-field">
              <label htmlFor="size">Size of Matrix(uniform) -> <b><span id="size-val">128</span></b></label><br />
              <Range id="size" default={512} min="1" max="1024" onInput={this.sizeChangeHandler}/>
            </div>
            <div className="input-field">
              <label htmlFor="num-bench">Number of Benchmarks</label><br />
              <input id="num-bench" type="number" defaultValue={5}/>
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
        </Container>

        <Container id="sizes" className="center">
          <Heading active={this.state.active.sizes}>Common Benchmarks</Heading>
          <h6>Here is a chart representing the performance of matrix multiplication of different arrays and different modes (lower is better)(some values are interpolated)</h6>
        </Container>

        <Container>
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
                <li>The Benchmarks were run using a tool called <a href="https://github.com/gpujs/benchmark"><b>@gpujs/benchmark</b></a> which is created by the gpu.js team to specifically benchmark gpu.js.</li>
              </ul>
            </Col>
          </Row>
        </Container>

        <Container className="center">
          <Graph info={sizes} title={{
            x: {
              text: 'matrix size',
              font: {
                family: 'Courier New, monospace',
                size: 14,
                color: '#7f7f7f'
              }
            },
            y: {
              text: 'time taken (in ms)',
              font: {
                family: 'Courier New, monospace',
                size: 14,
                color: '#7f7f7f'
              }
            }
          }} interpolation={true} />
        </Container>
      </div>
    )
  }
}

export default Benchmark