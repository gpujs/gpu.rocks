import React from 'react'
import MaterialIcon from '../MaterialIcon/MaterialIcon'
import { animated, useSpring } from 'react-spring'

import './Heading.scss'

const Heading = (props) => {

  const config = {
    mass: 3,
    tension: 30,
    friction: 15
  }
  const { x } = useSpring({
    config,
    x: props.active ? 0 : 50,
    from: {
      x: 50
    }
  })

  return (
    <h2 className={`center heading ${props.className || ''}`} style={{overflow: 'hidden', maxHeight: '110px'}} id={props.id || ''}>
      <animated.p style={{display: 'inline-block', transform: x.interpolate(x => `translateX(${-x}vw)`)}}><MaterialIcon icon="chevron_right" /></animated.p>
        {props.children}
      <animated.p style={{display: 'inline-block', transform: x.interpolate(x => `translateX(${x}vw)`)}}><MaterialIcon icon="chevron_left" /></animated.p>
    </h2>
  )
}

export default Heading