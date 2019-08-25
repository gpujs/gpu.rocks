import React from 'react'
import HeaderLinks from '../HeaderLinks/HeaderLinks'
import { Navbar } from 'react-materialize'
import { NavLink } from 'react-router-dom'

import './Nav.scss'

const Nav = () => {
  return (
    <div id="nav">
      <Navbar brand={<NavLink to="/" style={{marginLeft: '1rem'}}>GPU.js</NavLink>} alignLinks="right" >
        <HeaderLinks />
      </Navbar>
    </div>
  )
}

export default Nav