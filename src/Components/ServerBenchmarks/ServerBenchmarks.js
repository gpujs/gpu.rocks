import React from 'react'
import Row from 'react-materialize/lib/Row'
import Col from 'react-materialize/lib/Col'
import Graph from '../Util/Graph/Graph'
import sizes from '../../Data/beast-benchmark/beast-bench'

import './ServerBenchmarks.scss'

const ServerBenchmarks = () => {
  return (
    <div id="server-benchmarks">
      <h2 className="center">Server Benchmarks</h2>
      <hr />
      <div className="center">
        <p className="center">Here is a chart representing the performance of <b>GPU.js</b> in matrix multiplication of matrices of different sizes</p>
      </div>
      <div>
        <div>
          <Row>
            <Col offset="s1" s={8} className="specs">
              <ul>
                <li><b>Hardware:</b> &nbsp;<a href="https://ark.intel.com/content/www/us/en/ark/products/193396/intel-xeon-gold-5217-processor-11m-cache-3-00-ghz.html">Xeon Gold 5217</a> + 8 x <a href="https://www.nvidia.com/en-in/geforce/graphics-cards/rtx-2080-ti/">RTX 2080ti</a></li>
                <li><b>Operating System:</b> &nbsp;<a href="https://ubuntu.com/">Ubuntu</a> 18.04</li>
                <li><b>Environment:</b> &nbsp;NodeJS v8.10.0 + <b>GPU.js</b> v2.2.0</li>
                <li>
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
    </div>
  )
}

export default ServerBenchmarks
