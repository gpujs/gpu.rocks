import React from 'react'
import HeaderLinks from '../HeaderLinks/HeaderLinks'
import { Navbar } from 'react-materialize'
import { NavLink } from 'react-router-dom'

import jellyLogo from '../../img/jelly-nav.png'

import './Nav.scss'

function Nav() {
  return (
    <div id="nav" className="navbar navbar-fixed">
      <Navbar
        brand={
          <NavLink to="/" className="brand-logo"><img src={jellyLogo} className="jelly-nav-logo" alt="Jelly Logo" />GPU.js</NavLink>
        }
        alignLinks="right"
      >
        <HeaderLinks />
      </Navbar>
    </div>
  )
}

export default Nav