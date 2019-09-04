import React, { Component } from 'react'
import HeaderLinks from '../HeaderLinks/HeaderLinks'
import { Navbar } from 'react-materialize'
import { NavLink } from 'react-router-dom'
import $ from 'jquery'

import jellyLogo from '../../img/jelly-nav.png'

import './Nav.scss'

class Nav extends Component {
  ComponentDidMount {
    
  }

  render() {
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
}

export default Nav