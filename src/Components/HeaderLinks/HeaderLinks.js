import React from 'react'
import { NavLink } from 'react-router-dom'

const HeaderLinks = () => {
  return (
    <div>
      <li>
        <NavLink to="/docs" >
          Documentation
        </NavLink>
      </li>

      <li>
        <a href="https://raw.githubusercontent.com/gpujs/gpu.js/master/bin/gpu-browser.min.js">
          Download
        </a>
      </li>

      <li>
        <a href="https://github.com/gpujs/gpu.js">
          Github
        </a>
      </li>

      <li>
        <NavLink to="/install">
          Installation
        </NavLink>
      </li>
    </div>
  )
}

export default HeaderLinks