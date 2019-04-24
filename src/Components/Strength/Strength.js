import { Row, Col } from 'react-materialize'
import React from 'react'
import { animated, useSpring } from 'react-spring'

import threads from '../../img/threads.png'
import thread from '../../img/thread.png'

const Strength = (props) => {
  const config = {
    mass: 5,
    tension: 500,
    friction: 50
  }

  const {x, opacity} = useSpring({
    config,
    x: props.active ? 0 : 100,
    opacity: 1,
    from: {
      x: 100,
      opacity: 0
    }
  })

  return (
    <Row id="strength" style={{overflow: 'hidden'}}>
      <Col m={12} l={6} className="center" style={{marginBottom: '3rem'}}>
        <animated.div style={{transform: x.interpolate(x => `translateX(${-x}vw)`), opacity}} >
          <img src={threads} className="threads" alt="threads" />
          <h6 className="blue-grey-text text-darken-2"><b>Perform massively parallel GPGPU computations using GPU.</b></h6>
        </animated.div>
      </Col>
      <Col m={12} l={6} className="center" >
        <animated.div style={{transform: x.interpolate(x => `translateX(${x}vw)`), opacity}} >
          <img src={thread} className="thread" alt="thread" />
          <h6 className="blue-grey-text text-darken-2"><b>Graceful pure JavaScript fallback when GPU is not available. </b></h6>
        </animated.div>
      </Col>
    </Row>
  )
}

export default Strength