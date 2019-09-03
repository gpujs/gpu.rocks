import React from 'react'
import Code from '../Code/Code'
import { generateMatrices, createKernel, callKernel, getOutput } from './CodeExamples'
import { NavLink } from 'react-router-dom'

import './Example.scss'

const Example = () => {

  return (
    <div id="example">
      <h2 className="center">Example</h2>
      <hr />
      <h5 className="center"><strong>Matrix Multiplication</strong></h5>
      <p className="center">In this example, two <strong>512x512</strong> <strong>matrices</strong> (2d arrays) are <em><a href="https://mathsisfun.com/algebra/matrix-multiplying.html">multiplied</a></em>. The computation is done in parallel on the GPU.</p>
      <section>
        <h6><strong>1. </strong> Generate The Matrices</h6>
          <Code code={generateMatrices} />

        <h6><strong>2. </strong> Create The "<strong>Kernel</strong>"<em>(A fancy word for a function that runs on a GPU)</em></h6>
          <Code code={createKernel} />

        <h6><strong>3. </strong> Call The Kernel With The Matrices as Parameters</h6>
          <Code code={callKernel} />

        <h6><strong>4. </strong> The Output Matrix</h6>
          <Code code={getOutput} />
      </section>
      <h6 className="center"><NavLink to="/benchmark">Click</NavLink> to benchmark this code <strong>GPU</strong> v/s <strong>CPU</strong> on your device.</h6>
    </div>
  )
}

export default Example