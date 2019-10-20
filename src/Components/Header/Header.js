import JellyOnFayyah from '../JellyOnFayyah/JellyOnFayyah'
import React from 'react'
import Row from 'react-materialize/lib/Row'
import GitHubButton from 'react-github-btn'

import './Header.scss'

const Header = () => {
  const titleComponents = [
    <h1 className="name" key={1}>GPU.js</h1>,
    <h5 className="desc" key={2}><b>GPU accelerated JavaScript</b></h5>,
    <GitHubButton 
      href="https://github.com/gpujs/gpu.js"
      data-color-scheme="no-preference: dark; light: dark; dark: dark;" 
      data-size="large" 
      data-icon="octicon-star"
      data-show-count="true" 
      style={{marginRight: '3rem'}}
      aria-label="Star gpujs/gpu.js on GitHub"
      key={3}
    >Star</GitHubButton>,
    <GitHubButton
      href="https://github.com/gpujs/gpu.js/fork"
      data-color-scheme="no-preference: dark; light: dark; dark: dark;"
      data-icon="octicon-repo-forked"
      data-size="large"
      data-show-count="true"
      aria-label="Fork gpujs/gpu.js on GitHub"
      key={4}
    >Fork</GitHubButton>
  ]

  return (
    <header>
      <div className="header-container">
        <Row className="center responsive-img">
          <JellyOnFayyah />
        </Row>
        
        <Row className="center">
          {titleComponents}
        </Row>
      </div>
    </header>
  )
}

export default Header
