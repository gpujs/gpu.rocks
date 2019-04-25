import React, { Component } from 'react'
import { Container } from 'react-materialize'
import Example from '../Example/Example'
import Strength from '../Strength/Strength'
import getActiveElems from './getActiveElems'

import './Content.scss'

class Content extends Component {
  state = {
    active: {
      strength: false
    }
  }

  componentDidMount() {
    document.addEventListener('scroll', () => {
      const ids = {
        'strength': {
          id: 'strength',
          thresh: 500
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
          thresh: 900
        }
      }
      
      this.setState({
        active: getActiveElems(ids)
      })
    })
  }

  render() {
    return (
      <Container id="content">
        <Strength active={this.state.active.strength} />
        <hr />
        <Example active={this.state.active} />
      </Container>
    )
  }
}

export default Content