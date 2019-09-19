import { animated, useSpring } from 'react-spring'

import React from 'react'
import fayyah from './fayyah.webp'
import logo from '../../img/jelly.png'
import mask from './jelly-transparent.png'

export default function JellyOnFayyah() {
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

  const ignite = () => {
    const mainLogo = document.querySelector('#main-logo'),
      fayyah = document.querySelector('#fayyah')

    mainLogo.style.display = mainLogo.style.display == 'none' ? 'initial': 'none';
    fayyah.style.display = fayyah.style.display == 'initial' ? 'none' : 'initial';
  }

  const setIgnition = () => {
    setInterval(() => {
      if (Math.random() > Math.random() * 10) ignite();
    }, 200)
  }
  
  return (
    <div>
      <animated.img src={logo} style={{transform: y.interpolate(y => `translateY(${y}%)`)}} alt="logo" id="main-logo" />
      <img src={mask} style={{
        backgroundImage: `url(${fayyah})`,
        backgroundPosition: 'center',
        backgroundSize: 'cover',
        display: 'none'
      }} alt="fayyah" id="fayyah" onLoad={setIgnition} />
    </div>
  )
}
