import React from 'react'
import { NavLink } from 'react-router'

const HeaderLinks = () => {
  return (
    <ul>
    <li>
        <NavLink to="/" end>
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

      <li>
        {/* the interactive course; its own pages carry the standalone learn nav */}
        <NavLink to="/learn">
          Learn
        </NavLink>
      </li>
    </ul>
  )
}

export default HeaderLinks
