import React from 'react'
import Code from '../Code/Code'
import { animated, useSpring } from 'react-spring'

const Benchmark = (props) => {
  const config = {
    duration: 400
  }

  const {opacity} = useSpring({
    config,
    opacity: props.active ? 1 : 0,
    from: {
      opacity: 0
    }
  })

  return (
    <animated.section id="benchmark" style={{overflow: 'hidden', opacity}}>
      <h5 class="center"><b>Matrix Multiplication</b></h5>
      <Code>
      {` 
    const gpu = new GPU();
    const multiplyMatrix = gpu.createKernel(function(a, b) {
      var sum = 0;
      for (var i = 0; i < 512; i++) {
        sum += a[this.thread.y][i] * b[i][this.thread.x];
      }
      return sum;
    }).setOutput([512, 512]);
 `}
      </Code>
    </animated.section>
  )
}

export default Benchmark