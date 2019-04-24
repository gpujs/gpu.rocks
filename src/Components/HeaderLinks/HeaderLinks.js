import React from 'react'
import { NavLink } from 'react-router-dom'
import LinkBtn from './LinkBtn'
import { Container, Row, Column, NavItem } from 'react-materialize'

const HeaderLinks = () => {
  return (
    <div>
      <NavItem>
        <NavLink to="/docs" >
          {/* <LinkBtn tooltipContent="Documentation" btnContent="Docs" /> */}
          Yaay
        </NavLink>
      </NavItem>

      <NavItem>
        <a href="https://raw.githubusercontent.com/gpujs/gpu.js/master/bin/gpu-browser.min.js">
          {/* <LinkBtn tooltipContent="Browserified" btnContent="Download" /> */}
          Yaay
        </a>
      </NavItem>

      <NavItem>
        <a href="https://github.com/gpujs/gpu.js">
          {/* <LinkBtn tooltipContent="OpenSource" btnContent="Github" /> */}
        </a>
      </NavItem>

      <NavItem>
        <NavLink to="/install">
          {/* <LinkBtn tooltipContent="Installation" btnContent="Install" /> */}
        </NavLink>
      </NavItem>
    </div>
  )
}

export default HeaderLinks