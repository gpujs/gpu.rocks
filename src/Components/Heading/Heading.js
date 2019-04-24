import React from 'react'
import MaterialIcon from 'material-icons-react'
import { animated, useSpring } from 'react-spring'

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
    <h2 className={`center blue-text text-lighten-2 ${props.className || ''}`} style={{overflow: 'hidden', maxHeight: '110px'}} id={props.id || ''}>
      <animated.p style={{display: 'inline-block', transform: x.interpolate(x => `translateX(${-x}vw)`)}}><MaterialIcon icon="chevron_right" size="medium" color="red" /></animated.p>
        {props.children}
      <animated.p style={{display: 'inline-block', transform: x.interpolate(x => `translateX(${x}vw)`)}}><MaterialIcon icon="chevron_left" size="medium" color="red" /></animated.p>
    </h2>
  )
}

export default Heading