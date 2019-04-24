import React, { Component } from 'react'
import { Row, Col } from 'react-materialize'
import FadeIn from './FadeIn'

class Header extends Component {
  render() {
    return (
      <header>
        <Row>
          <Col s={12} m={6} className="teal center white-text">
            Yay
          </Col>
          <Col s={12} m={6} className="teal center white-text">
            Yay
            <FadeIn><h1>Hello!</h1></FadeIn>
          </Col>
        </Row>
      </header>
    )
  }
}

export default Header