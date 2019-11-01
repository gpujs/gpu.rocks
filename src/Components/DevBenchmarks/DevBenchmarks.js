import React from 'react'
import Row from 'react-materialize/lib/Row'
import Col from 'react-materialize/lib/Col'
// import Graph from '../Util/Graph/Graph'

import './DevBenchmarks.scss'

const DevBenchmarks = () => {
  return (
    <div id="dev-benchmarks">
      <h2 className="center">Development Benchmarks</h2>
      <hr />
      <div className="center">
        <h6>Here is a chart representing the performance of gpu.js in matrix multiplication of matrices of different sizes</h6>
      </div>
      <div className="grid-test">
        {/* <Graph info={gtx1080} /> */}
        <div>
          <Row>
            <Col offset="s3" s={8} className="specs">
              <ul>
                <li><b>Hardware:</b> &nbsp;Xeon Gold 5217 + 8 x RTX 2080ti</li>
                <li><b>Operating System:</b> &nbsp;Ubuntu 18.04</li>
                <li><b>Environment:</b> &nbsp;NodeJS v8.10.0</li>
                <li>
                  <b>Browser:</b>
                  <ul>
                    {/* <li><span className="bench-color" style={{backgroundColor: gtx1080Obj.firefox.lineColor}}></span> Firefox 54.0.1 (32-Bit)</li>
                    <li><span className="bench-color" style={{backgroundColor: gtx1080Obj.chrome.lineColor}}></span> Chrome 59.0.3071.115 (64-Bit)</li>
                    <li><span className="bench-color" style={{backgroundColor: gtx1080Obj.edge.lineColor}}></span> Edge 40.15063.0.0 (64-Bit)</li> */}
                  </ul>
                </li>
                <li><b>Last Updated:</b> &nbsp;1 November 2019</li>
                <li>The Benchmarks were run using a tool called <a href="https://github.com/gpujs/benchmark"><b>@gpujs/benchmark</b></a> which is created by the gpu.js org to specifically benchmark gpu.js.</li>
              </ul>
            </Col>
          </Row>
        </div>
      </div>
    </div>
  )
}

export default DevBenchmarks
