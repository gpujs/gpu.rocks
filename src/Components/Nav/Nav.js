import React from 'react'
import HeaderLinks from '../HeaderLinks/HeaderLinks'
import { Navbar } from 'react-materialize'

const Nav = () => {
  return (
    <Navbar brand={<a href="/#" style={{marginLeft: '1rem'}}>GPU.js</a>} alignLinks="right" className="blue" >
      <HeaderLinks />
    </Navbar>
  )
}

export default Nav