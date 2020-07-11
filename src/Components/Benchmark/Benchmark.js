import React, { Component } from 'react'
import Button from 'react-materialize/lib/Button'
import Range from 'react-materialize/lib/Range'
import Checkbox from 'react-materialize/lib/Checkbox'
import Row from 'react-materialize/lib/Row'
import Col from 'react-materialize/lib/Col'
import { GPU } from 'gpu.js'
import { dependencies } from '../../../package.json'

import Graph from '../Util/Graph/Graph'
import sizes from '../../Data/gt1030/gt1030-node'
import { benchmark } from '@gpujs/benchmark'
import ScrollButton from '../ScrollButton/ScrollButton'
import BenchmarkTables from './BenchmarkTables'

import './Benchmark.scss'

const performerMap = {
  cpu: 'CPU',
  gpu: 'GPU',
  pipe: 'GPU(Pipeline Mode)'
}

class Benchmark extends Component {
  constructor(props) {
    super(props);
    this.state = {
      benchmarked: false
    }
  }

  benchmarkFormHandler = (e) => {
    e.preventDefault();

    const size = document.querySelector('input[type="range"][min="1"][max="4096"]').value || 256,
      num_iterations = parseInt(document.querySelector('#num-iter').value),
      cpu = document.querySelector('#cpu').checked

    this.setState({cpu});
    this.setState({
      bench: benchmark({
        cpu: new GPU({mode: 'cpu'}),
        gpu: new GPU(),
        num_iterations,
        matrix_size: size,
        logs: false,
        cpu_benchmark: cpu,
      }).getData()
    })

    this.setState({benchmarked: true})
  }

  sizeChangeHandler = () => {
    const size = document.querySelector('input[type="range"][min="1"][max="4096"]')
    const cpu = document.querySelector('#cpu')
    document.querySelector('#size-val').innerHTML = size.value;
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
          <p className="center"><b>GPU.js version:</b> &nbsp;v{dependencies['gpu.js'].replace('^', '')}</p>

          <form id="benchmark-form" onSubmit={this.benchmarkFormHandler}>
            <div className="input-field">
              <label htmlFor="size">Size of Matrix(uniform) -&gt; <b><span id="size-val">256</span></b></label><br />
              <Range id="size" defaultValue="256" min="1" max="4096" onChange={this.sizeChangeHandler}/>
            </div>
            <div className="input-field">
              <label htmlFor="num-iter">Number of Iterations</label><br />
              <input id="num-iter" type="number" defaultValue={5}/>
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
          {
            this.state.benchmarked && <BenchmarkTables cpu={this.state.cpu} bench={this.state.bench} performerMap={performerMap} />
          }
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
