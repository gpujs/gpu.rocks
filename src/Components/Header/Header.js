import React from 'react'
import { Row, Container } from 'react-materialize'
import { animated, useTrail, useSpring } from 'react-spring'
import Nav from '../Nav/Nav'

import logo from '../../img/jelly.png'

import './Header.scss'

const Header = () => {
  const titleComponents = [
    <h1 className="name" >GPU.js</h1>,
    <h5 className="desc" ><b>GPU accelerated JavaScript</b></h5>
  ]
  const config = {
    mass: 5,
    tension: 1000,
    friction: 50
  }
  const titleTrail = useTrail(titleComponents.length, {
    config,
    opacity: 1,
    y: 0,
    height: 80,
    from: { opacity: 0, y: -1000, height: 0 },
  })

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
    <header id="header">
      <Nav />
      <Container>
        <Row className="center responsive-img"><animated.img src={logo} style={{transform: y.interpolate(y => `translateY(${y}%)`)}} alt="logo" /></Row>
        <Row className="center" >
          {titleTrail.map(({y, height, ...styles}, i) => {
            return <animated.div
              key={i}
              style={{...styles, transform: y.interpolate(y => `translateY(${y}%)`)}}>
              {titleComponents[i]}
            </animated.div>
          })}
        </Row>
      </Container>
    </header>
  )
}

export default Header