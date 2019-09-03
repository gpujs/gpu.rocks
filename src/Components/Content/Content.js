import React, { Component } from 'react'
import Example from '../Example/Example'
import Strength from '../Strength/Strength'
import getActiveElems from '../../utils/getActiveElems'
import $ from 'jquery'
import Syntax from '../Syntax/Syntax'
import DevBenchmarks from '../DevBenchmarks/DevBenchmarks'
import OldVersions from '../OldVersions/OldVersions'

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
      <div id="content">
        <Strength active={this.state.active.strength} />
        <Example />
        <Syntax />
        <DevBenchmarks />
        <OldVersions />
      </div>
    )
  }
}

export default Content