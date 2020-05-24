import Row from 'react-materialize/lib/Row'
import Col from 'react-materialize/lib/Col'
import React from 'react'

import threads from '../../img/threads.svg'
import thread from '../../img/thread.svg'
import nodejs from '../../img/nodejs.png'

import './Strength.scss'

const Strength = () => {

  return (
    <div id="strength">
      <Row className="strength-container">
        <Col m={12} l={6} className="center">
          <img src={threads} className="threads" alt="threads" />
          <h6>Perform massively parallel GPGPU computations using GPU.</h6>
        </Col>
        <Col m={12} l={6} className="center">
          <img src={thread} className="thread" alt="thread" />
          <h6>Graceful pure JavaScript fallback when GPU is not available.</h6>
        </Col>
      </Row>
      <Row className="node-container">
        <Col s={12} className="center">
          <img src={nodejs} className="nodejs" alt="nodejs" />
          <h6>Works in <a href="https://nodejs.org" target="_blank" rel="noopener noreferrer">NodeJS</a>!</h6>
        </Col>
      </Row>
    </div>
  )
}

export default Strength
