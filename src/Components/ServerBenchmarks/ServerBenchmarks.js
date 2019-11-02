import React from 'react'
import Row from 'react-materialize/lib/Row'
import Col from 'react-materialize/lib/Col'
// import Graph from '../Util/Graph/Graph'

import './ServerBenchmarks.scss'

const ServerBenchmarks = () => {
  return (
    <div id="dev-benchmarks">
      <h2 className="center">Server Benchmarks</h2>
      <hr />
      <div className="center">
        <p className="center">Here is a chart representing the performance of <b>GPU.js</b> in matrix multiplication of matrices of different sizes</p>
      </div>
      <div className="grid-test">
        {/* <Graph info={gtx1080} /> */}
        <div>
          <Row>
            <Col offset="s3" s={8} className="specs">
              <ul>
                <li><b>Hardware:</b> &nbsp;<a href="https://ark.intel.com/content/www/us/en/ark/products/193396/intel-xeon-gold-5217-processor-11m-cache-3-00-ghz.html">Xeon Gold 5217</a> + 8 x <a href="https://www.nvidia.com/en-in/geforce/graphics-cards/rtx-2080-ti/">RTX 2080ti</a></li>
                <li><b>Operating System:</b> &nbsp;<a href="https://ubuntu.com/">Ubuntu</a> 18.04</li>
                <li><b>Environment:</b> &nbsp;NodeJS v8.10.0 + <b>GPU.js</b> v2.2.0</li>
                <li>
                  <ul>
                    {/* <li><span className="bench-color" style={{backgroundColor: gtx1080Obj.firefox.lineColor}}></span> Firefox 54.0.1 (32-Bit)</li>
                    <li><span className="bench-color" style={{backgroundColor: gtx1080Obj.chrome.lineColor}}></span> Chrome 59.0.3071.115 (64-Bit)</li>
                    <li><span className="bench-color" style={{backgroundColor: gtx1080Obj.edge.lineColor}}></span> Edge 40.15063.0.0 (64-Bit)</li> */}
                  </ul>
                </li>
                <li><b>Last Updated:</b> &nbsp;1 November 2019</li>
                <li>The Benchmarks were run using a tool called <a href="https://github.com/gpujs/benchmark"><b>@gpujs/benchmark</b></a> which is created by the <b>GPU.js</b> org to specifically benchmark <b>GPU.js</b>.</li>
              </ul>
            </Col>
          </Row>
        </div>
      </div>
    </div>
  )
}

export default ServerBenchmarks
