import React from 'react'
import { Row } from 'react-materialize'
import { animated, useSpring } from 'react-spring'

import logo from '../../img/jelly.png'

import './Header.scss'

const Header = () => {
  const titleComponents = [
    <h1 className="name">GPU.js</h1>,
    <h5 className="desc"><b>GPU accelerated JavaScript</b></h5>
  ]

  const {y} = useSpring({
    config: {
      mass: 3,
      tension: 100,
      friction: 20
    },
    y: 0,
    from: {
      y: 100
    }
  })

  return (
    <header>
      <div className="header-container">
        <Row className="center responsive-img">
          <animated.img src={logo} style={{transform: y.interpolate(y => `translateY(${y}%)`)}} alt="logo" />
        </Row>
        <Row className="center">
          {titleComponents}
        </Row>
      </div>
    </header>
  )
}

export default Header