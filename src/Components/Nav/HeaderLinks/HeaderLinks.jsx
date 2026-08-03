import React from 'react'
import { NavLink } from 'react-router'

/**
 * The site's main navigation.
 *
 * ORDER IS SHARED. Three navs exist — this one, src/Learn/components/LearnNav.jsx
 * and the one inside src/Bench/BenchApp.jsx — and each is rendered by a
 * different app, so nothing enforces agreement between them but this note.
 * They had drifted into three different orders, which makes the site feel like
 * three sites when you move between them.
 *
 *   Home · Learn · API · Examples · Benchmark · GitHub
 *
 * A nav may ADD items and may omit ones that do not apply to it, but the items
 * it shares must appear in that relative order. This one adds Documentation,
 * Installation and Download; the learn nav adds Feedback.
 */
const HeaderLinks = () => {
  return (
    <ul>
      <li>
        <NavLink to="/" end>
          Home
        </NavLink>
      </li>

      <li>
        {/* the interactive course; its own pages carry the standalone learn nav */}
        <NavLink to="/learn">
          Learn
        </NavLink>
      </li>

      <li>
        <a href="https://github.com/gpujs/gpu.js/#readme">
          Documentation
        </a>
      </li>

      <li>
        <NavLink to="/examples">
          Examples
        </NavLink>
      </li>

      <li>
        <NavLink to="/benchmark">
          Benchmark
        </NavLink>
      </li>

      {/* acquisition, grouped: this site's own additions to the shared order */}
      <li>
        <NavLink to="/install">
          Installation
        </NavLink>
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
    </ul>
  )
}

export default HeaderLinks
