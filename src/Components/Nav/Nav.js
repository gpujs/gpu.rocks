import React from 'react'
import { Navbar, NavItem } from 'react-materialize'
// import HeaderLinks from '../HeaderLinks/HeaderLinks'

const Nav = () => {
  return (
    <nav>
      <div class="nav-wrapper">
        <a class="brand-logo center">GPU.js</a>
        <ul id="nav-mobile" class="right hide-on-med-and-down">
          <li><a href="sass.html">Sass</a></li>
          <li><a href="badges.html">Components</a></li>
          <li><a href="collapsible.html">JavaScript</a></li>
        </ul>
      </div>
    </nav>
  )
}

export default Nav