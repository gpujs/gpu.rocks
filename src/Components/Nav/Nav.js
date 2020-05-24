import React from 'react'
import HeaderLinks from './HeaderLinks/HeaderLinks'
import Navbar from 'react-materialize/lib/Navbar'
import * as Link from 'react-router-dom/Link'

import jellyLogo from '../../img/jelly-nav.png'

import './Nav.scss'

function Nav() {
  return (
    <div id="nav" className="navbar navbar-fixed">
      <Navbar
        brand={
          <Link to="/" className="brand-logo"><img src={jellyLogo} className="jelly-nav-logo" alt="Jelly Logo" />GPU.js</Link>
        }
        alignLinks="right"
      >
        <HeaderLinks />
      </Navbar>
    </div>
  )
}

export default Nav
