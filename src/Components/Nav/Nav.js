import React from 'react'
import HeaderLinks from '../HeaderLinks/HeaderLinks'
import { Navbar } from 'react-materialize'

const Nav = () => {
  return (
    <Navbar brand={<a href="/#">GPU.js</a>} centerLogo={true} className="blue" >
      <HeaderLinks />
    </Navbar>
  )
}

export default Nav