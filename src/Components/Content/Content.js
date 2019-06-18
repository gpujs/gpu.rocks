import React, { Component } from 'react'
import { Container } from 'react-materialize'
import Example from '../Example/Example'
import Strength from '../Strength/Strength'
import getActiveElems from '../../utils/getActiveElems'
import $ from 'jquery'
import Syntax from '../Syntax/Syntax'
import DevBenchmarks from '../DevBenchmarks/DevBenchmarks'

import './Content.scss'

class Content extends Component {
  state = {
    active: {}
  }

  componentDidMount() {
    $(document).on('DOMContentLoaded scroll', () => {
      const ids = {
        'strength': {
          id: 'strength',
          thresh: 700
        },
        'example': {
          id: 'example',
          thresh: 600
        },
        egCode1: {
          id: 'example-code1',
          thresh: 800
        },
        egCode2: {
          id: 'example-code2',
          thresh: 800
        },
        egCode3: {
          id: 'example-code3',
          thresh: 800
        },
        egCode4: {
          id: 'example-code4',
          thresh: 800
        },
        syntax: {
          id: 'syntax',
          thresh: 800
        },
        devBenchmarks: {
          id: 'dev-benchmarks',
          thresh: 800
        }
      }
      
      this.setState({
        active: getActiveElems(ids)
      })
    })
  }

  componentWillUnmount() {
    $(document).off('DOMContentLoaded scroll')
  }

  render() {
    return (
      <Container id="content">
        <Strength active={this.state.active.strength} />
        <hr />
        <Example active={this.state.active} />
        <Syntax active={this.state.active} />
        <DevBenchmarks active={this.state.active} />
      </Container>
    )
  }
}

export default Content