import React from 'react'
import Heading from '../Heading/Heading'
import { Container, Row, Col } from 'react-materialize'
import { gtx1080, gtx1080Obj, mbpFirefox } from '../../Data/DevelopmentBenchmarks'
import Plot from 'react-plotly.js'

import './DevBenchmarks.scss'

const DevBenchmarks = ({active}) => {
  const plotData = {
    gtx1080: [],
    mbp: []
  }

  const pushPlotData = (brow, pushTo) => {
    const displayName = brow.displayName,
      x_series = brow.data.x_series,
      y_series_upper = brow.data.y_series_upper,
      y_series = brow.data.y_series,
      y_series_lower = brow.data.y_series_lower,
      lineColor = brow.lineColor,
      shadowColor = brow.shadowColor

    pushTo.push({
      name: `${displayName} (Upper Bound)`,
      x: x_series,
      y: y_series_upper,
      type: 'scatter',
      mode: 'lines',
      marker: {color:'#444', },
      line: {width:0},
    })
    pushTo.push({
      name: `${displayName} (Mean)`,
      x: x_series,
      y: y_series,
      type: 'scatter',
      mode: 'lines',
      line: {
        color: lineColor
      },
      fillcolor: shadowColor,
      fill: 'tonexty'
    })
    pushTo.push({
      name: `${displayName} (Lower Bound)`,
      x: x_series,
      y: y_series_lower,
      type: 'scatter',
      mode: 'lines',
      marker: {color:'#444'},
      line: {width:0},
      fillcolor: shadowColor,
      fill: 'tonexty'
    })
  }

  gtx1080.forEach(brow => pushPlotData(brow, plotData.gtx1080))
  mbpFirefox.forEach(brow => pushPlotData(brow, plotData.mbp))

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
        <Plot 
          data={plotData.gtx1080}
          layout={{
            yxais: {
              type: 'log',
              autorange: true
            },
            showlegend: false,
            hovermode: 'closest'
          }}
        />
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
        <Plot 
          data={plotData.mbp}
          layout={{
            yxais: {
              type: 'log',
              autorange: true
            },
            showlegend: false,
            hovermode: 'closest'
          }}
        />
      </Container>
    </div>
  )
}

export default DevBenchmarks