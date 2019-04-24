import React from 'react'
import {useSpring, animated} from 'react-spring'

const FadeIn = (props) => {
  const durationMap = {
    fast: 200,
    default: 200,
    slow: 1000
  }
  const ref = props.ref;
  const dur = props.duration || 'default'
  const style = useSpring({to: {opacity: 1}, from: {opacity: 0}, config: {duration: durationMap[dur] || props.duration}, ref})
  return <animated.div style={style}>{props.children}</animated.div>
}

export default FadeIn