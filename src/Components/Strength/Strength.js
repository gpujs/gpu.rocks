import { Row, Col } from 'react-materialize'
import React from 'react'
import { animated, useSpring } from 'react-spring'

import threads from '../../img/threads.png'
import thread from '../../img/thread.png'
import nodejs from '../../img/nodejs.png'

const Strength = (props) => {
  const config = {
    mass: 5,
    tension: 500,
    friction: 50
  }

  const {x, opacity} = useSpring({
    config,
    x: props.active ? 0 : 100,
    opacity: props.active ? 1 : 0,
    from: {
      x: 100,
      opacity: 0
    }
  })

  return (
    <div id="strength">
      <Row style={{overflow: 'hidden'}}>
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
      <Row>
        <Col s={12} className="center" style={{marginBottom: '3rem'}}>
          <animated.div style={{opacity}} >
            <img src={nodejs} className="nodejs" alt="nodejs" />
            <h5 className="blue-grey-text text-darken-2"><b>Works in <a href="https://nodejs.org">NodeJS</a>!</b></h5>
          </animated.div>
        </Col>
      </Row>
    </div>
  )
}

export default Strength