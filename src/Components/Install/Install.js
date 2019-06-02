import React, { Component } from 'react'
import Heading from '../Heading/Heading'
import getActiveElems from '../../utils/getActiveElems'
import { Container } from 'react-materialize'
import Code from '../Code/Code'
import code from './InstallCode'
import $ from 'jquery'

import './Install.scss'

class Install extends Component {
  state = {
    active: {}
  }

  componentDidMount() {
    $(document).on('DOMContentLoaded scroll', () => {
      const ids = {
        'install': {
          id: 'install',
          thresh: 1000
        }
      }

      this.setState({
        active: getActiveElems(ids)
      })
    })
  }

  render() {
    return (
      <div id="install">
        <Heading active={this.state.active.install}>Installation</Heading>
        <Container>
          <h5 className="center">On Linux, ensure you have the correct header files installed</h5>
          <Code code={code.linux} language="bash"></Code>

          <h4 className="center">npm</h4>
          <Code code={code.npm}/>

          <h4 className="center">yarn</h4>
          <Code code={code.yarn}/>

          <h4 className="center">Node</h4>
          <Code code={code.node}/>

          <h4 className="center">Node Typescript</h4>
          <Code code={code.type}/>

          <h4 className="center">Browser</h4>
          <p className="center">Download the <a href="https://raw.githubusercontent.com/gpujs/gpu.js/master/bin/gpu-browser.min.js">latest version of GPU.js</a> and include the files in your HTML page using the following tags</p>
          <Code code={code.browser} language="html"/>
        </Container>
      </div>
    )
  }
}

export default Install
