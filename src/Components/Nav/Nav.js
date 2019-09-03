import React from 'react'
import HeaderLinks from '../HeaderLinks/HeaderLinks'
import { Navbar } from 'react-materialize'

import jellyLogo from '../../img/jelly-nav.png'

import './Nav.scss'

const Nav = () => {
  return (
    <div id="nav" className="navbar navbar-fixed">
      <Navbar
        brand={
          <a href="/" className="brand-logo"><img src={jellyLogo} className="jelly-nav-logo" alt="Jelly Logo" />GPU.js</a>
        } alignLinks="right">
        <HeaderLinks />
      </Navbar>
    </div>
  )
}

export default Nav