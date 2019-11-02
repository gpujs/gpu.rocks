import React, { Component } from 'react'
import Container from 'react-materialize/lib/Container'
import Code from '../Util/Code/Code'
import code from './InstallCode'
import ScrollButton from '../ScrollButton/ScrollButton'

import './Install.scss'

class Install extends Component {
  render() {
    return (
      <div id="install">
        <ScrollButton />
        <h2 className="center">Installation</h2>
        <hr />
        <div className="install-container">
          <h5 className="center">On Linux, ensure you have the correct header files installed</h5>
          <Container>
            <Code code={code.linux} language="bash"></Code>
          </Container>

          <h5 className="center">npm</h5>
          <Container>
            <Code code={code.npm} language="bash"></Code>
          </Container>

          <h5 className="center">yarn</h5>
          <Container>
            <Code code={code.yarn} language="bash"></Code>
          </Container>

          <h5 className="center">Node</h5>
          <Container>
            <Code code={code.node}/>
          </Container>

          <h5 className="center">Node Typescript</h5>
          <Container>
            <Code code={code.type}/>
          </Container>

          <h5 className="center">Browser</h5>
          <p className="center">Download the <a href="https://raw.githubusercontent.com/gpujs/gpu.js/master/dist/gpu-browser.min.js">latest version of <b>GPU.js</b></a> and include the files in your HTML page using the following tags</p>
          <Container>
            <Code code={code.browser} language="html"/>
          </Container>
        </div>
      </div>
    )
  }
}

export default Install
