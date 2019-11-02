import React from 'react'
import { NavLink } from 'react-router-dom'

const HeaderLinks = () => {
  return (
    <ul>
    <li>
        <NavLink to="/" exact>
          Home
        </NavLink>
      </li>
      <li>
        <a href="https://github.com/gpujs/gpu.js/#readme">
          Documentation
        </a>
      </li>

      <li>
        <a href="https://raw.githubusercontent.com/gpujs/gpu.js/master/dist/gpu-browser.min.js">
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

      <li>
        <NavLink to="/benchmark">
          Benchmark
        </NavLink>
      </li>

      <li>
        <NavLink to="/examples">
          Examples
        </NavLink>
      </li>
    </ul>
  )
}

export default HeaderLinks
