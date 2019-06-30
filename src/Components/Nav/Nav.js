import React from 'react'
import HeaderLinks from '../HeaderLinks/HeaderLinks'
import { Navbar } from 'react-materialize'

import './Nav.scss'

const Nav = () => {
  return (
    <div id="nav">
      <Navbar brand={<a href="/#" style={{marginLeft: '1rem'}}>GPU.js</a>} alignLinks="right" >
        <HeaderLinks />
      </Navbar>
    </div>
  )
}

export default Nav