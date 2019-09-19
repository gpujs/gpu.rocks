import './Header.scss'

import JellyOnFayyah from '../JellyOnFayyah/JellyOnFayyah'
import React from 'react'
import Row from 'react-materialize/lib/Row'
import logo from '../../img/jelly.png'

const Header = () => {
  const titleComponents = [
    <h1 className="name">GPU.js</h1>,
    <h5 className="desc"><b>GPU accelerated JavaScript</b></h5>
  ]

  return (
    <header>
      <div className="header-container">
        <Row className="center responsive-img">
          <JellyOnFayyah />
        </Row>
        <Row className="center">
          {titleComponents}
        </Row>
      </div>
    </header>
  )
}

export default Header
