import React from 'react'
import { Container, Row, Col } from 'react-materialize'

import threads from '../../img/threads.png'
import thread from '../../img/thread.png'
import './Content.scss'

const Content = () => {
 return (
  <Container id="content">
    <Row>
      <Col m={12} l={6} className="center" style={{marginBottom: '3rem'}}>
        <img src={threads} className="threads" alt="threads" />
        <h6 className="blue-grey-text text-darken-2"><b>Perform massively parallel GPGPU computations using GPU.</b></h6>
      </Col>
      <Col m={12} l={6} className="center">
        <img src={thread} className="thread" alt="thread" />
        <h6 className="blue-grey-text text-darken-2"><b>Graceful pure JavaScript fallback when GPU is not available. </b></h6>
      </Col>
    </Row>
  </Container>
 )
}

export default Content