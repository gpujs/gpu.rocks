import React, { Component } from 'react'
import { Container, Button, Range, Checkbox } from 'react-materialize'
import Heading from '../Heading/Heading'
import getActiveElems from '../../utils/getActiveElems'
import Graph from '../Graph/Graph'
import sizes from '../../Data/different-sizes/gt1030-firefox'
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
          <Graph info={sizes} />
        </Container>
      </div>
    )
  }
}

export default Benchmark