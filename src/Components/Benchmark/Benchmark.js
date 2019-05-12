import React, { Component } from 'react'
import { Container, Button, Range, Checkbox, Row, Col } from 'react-materialize'
import Heading from '../Heading/Heading'
import getActiveElems from '../../utils/getActiveElems'
import Graph from '../Graph/Graph'
import sizes, {obj} from '../../Data/different-sizes/gt1030-firefox'
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
  
  render(){
    return (
      <div id="benchmark">
        <Heading active={this.state.active.benchmark} >Benchmarks</Heading>
        <Container>
          <div className="input-field">
            <label htmlFor="size">Size of Matrix(uniform) -> <span id="size-val"></span></label><br />
            <Range id="size" default={512} min="1" max="1024" />
          </div>
          <h6> Select Benchmarks</h6>
          <p>
            <Checkbox label="Matrix Multiplication" value="matmult" id="matmult" filledIn/>
          </p>
          <p>
            <Checkbox label="Kernel Convolution" value="conv" id="conv" filledIn />
            <a href="https://en.wikipedia.org/wiki/Kernel_(image_processing)"> (Read More)</a>
          </p>
          <h6> Select Modes</h6>
          <p>
            <Checkbox label="CPU" value="CPU" id="cpu" filledIn />
          </p>
          <p>
            <Checkbox label="CPU(GPUjs)" value="gpujscpu" id="gpujscpu" filledIn />
          </p>
          <p>
            <Checkbox label="GPU" value="gpu" id="gpu" filledIn />
          </p>
          <p>
            <Checkbox label="GPU(Texture Mode)" value="gputex" id="gputex" filledIn />
          </p>
          <Button waves="light" id="bench" className="blue lighten-1" >
            benchmark!
          </Button>
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
                <li><b>Environment:</b> &nbsp;NodeJS v10.15.3 (64-bit) + gpu.js v2.0.0rc-13</li>
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
          <Graph info={sizes} interpolation={true} />
        </Container>
      </div>
    )
  }
}

export default Benchmark