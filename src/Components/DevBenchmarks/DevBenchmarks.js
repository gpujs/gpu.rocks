import React from 'react'
import Heading from '../Heading/Heading'
import { Container, Row, Col } from 'react-materialize'
import { gtx1080, gtx1080Obj, mbpFirefox } from '../../Data/DevelopmentBenchmarks'
import Graph from '../Graph/Graph'

import './DevBenchmarks.scss'

const DevBenchmarks = ({active}) => {
  return (
    <div id="dev-benchmarks">
      <Heading active={active.devBenchmarks}>Development Benchmarks</Heading>
      <Container className="center">
        <h6>Here is a chart representing the performance of a 512x512 matrix multiplication throughout our development (lower is better)</h6>
      </Container>

      <Container>
        <Row>
          <Col offset="s1" s={10}>
            <ul>
              <li><b>Hardware:</b> &nbsp;i7-7700K + GTX1080</li>
              <li><b>Operating System:</b> &nbsp;Windows 10 (Build 15063.483)</li>
              <li>
                <b>Browser:</b>
                <ul>
                  <li><span className="browser-color" style={{backgroundColor: gtx1080Obj.firefox.lineColor}}></span> Firefox 54.0.1 (32-Bit)</li>
                  <li><span className="browser-color" style={{backgroundColor: gtx1080Obj.chrome.lineColor}}></span> Chrome 59.0.3071.115 (64-Bit)</li>
                  <li><span className="browser-color" style={{backgroundColor: gtx1080Obj.edge.lineColor}}></span> Edge 40.15063.0.0 (64-Bit)</li>
                </ul>
              </li>
              <li><b>Last Updated:</b> &nbsp;23 July 2017</li>
            </ul>
          </Col>
        </Row>
      </Container>

      <Container className="center">
        <Graph info={gtx1080} />
      <h4>Non GPU benchmarks</h4>
      <h6>Here is the chart for our benchmarks on a device without a GPU</h6>
      </Container>

      <Container>
        <Row>
          <Col offset="s1" s={10}>
            <ul>
              <li><b>Hardware:</b> &nbsp;Macbook Pro Retina 2012</li>
              <li><b>Operating System:</b> &nbsp;MacOS X 10.12.5</li>
              <li>
                <b>Browser:</b>
                <ul>
                  <li><span className="browser-color" style={{backgroundColor: gtx1080Obj.firefox.lineColor}}></span> Firefox 54.0.1 (64-Bit)</li>
                </ul>
              </li>
              <li><b>Last Updated:</b> &nbsp;23 July 2017</li>
            </ul>
          </Col>
        </Row>
      </Container>

      <Container className="center">
        <Graph info={mbpFirefox} />
      </Container>
    </div>
  )
}

export default DevBenchmarks