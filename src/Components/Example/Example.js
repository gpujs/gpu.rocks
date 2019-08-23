import React from 'react'
import Code from '../Util/Code/Code'
import { animated, useSpring } from 'react-spring'
import Heading from '../Util/Heading/Heading'
import { generateMatrices, createKernel, callKernel, getOutput } from './CodeExamples'
import { NavLink } from 'react-router-dom'

const Example = (props) => {
  const config = {
    duration: 400
  }

  const {opacity1, opacity2, opacity3, opacity4} = useSpring({
    config,
    opacity1: props.active.egCode1 ? 1 : 0,
    opacity2: props.active.egCode2 ? 1 : 0,
    opacity3: props.active.egCode3 ? 1 : 0,
    opacity4: props.active.egCode4 ? 1 : 0,
    from: {
      opacity: 0
    }
  })

  return (
    <div>
      <Heading active={props.active.example} id="example">Example</Heading>
      <section style={{overflow: 'hidden'}}>
        <h4 className="center" style={{color: '#666'}}><b>Matrix Multiplication</b></h4>
        <h5 className="center" style={{marginBottom: '4rem'}}>In this example, two <b>512x512</b> <b>matrices</b> (2d arrays) are  <i><a href="https://mathsisfun.com/algebra/matrix-multiplying.html">multiplied</a></i>. The computation is done in parallel on the GPU.</h5>

        <h6><b>1)</b> Generate The Matrices</h6>
        <animated.div id="example-code1" style={{opacity: opacity1}}>
          <Code code={generateMatrices} />
        </animated.div>

        <h6><b>2)</b> Create The "<b>Kernel</b>"<i>(A fancy word for a function that runs on a GPU)</i></h6>
        <animated.div id="example-code2" style={{opacity: opacity2}}>
          <Code code={createKernel} />
        </animated.div>

        <h6><b>3)</b> Call The Kernel With The Matrices as Parameters</h6>
        <animated.div id="example-code3" style={{opacity: opacity3}}>
          <Code code={callKernel} />
        </animated.div>

        <h6><b>4)</b> The Output Matrix</h6>
        <animated.div id="example-code4" style={{opacity: opacity4}}>
          <Code code={getOutput} />
        </animated.div>

        <h6 className="center"><NavLink to="/benchmark">Click</NavLink> to benchmark this code <b>GPU</b> v/s <b>CPU</b> on your device.</h6>
      </section>
    </div>
  )
}

export default Example