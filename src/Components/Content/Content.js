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
        egCode: {
          id: 'example-code',
          thresh: 800
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