import React, { Component } from 'react'
import { Container } from 'react-materialize'
import Benchmark from '../Benchmark/Benchmark'
import Strength from '../Strength/Strength'
import $ from 'jquery'
import MaterialIcon from 'material-icons-react';

import './Content.scss'

class Content extends Component {
  state = {
    active: {
      strength: false
    }
  }

  componentDidMount() {
    document.addEventListener('scroll', () => {
      this.setState({
        active: {
          strength: ($('#strength')[0].getBoundingClientRect().top - $(':root').prop('scrollTop')) < 500,
          benchmark: ($('#benchmark')[0].getBoundingClientRect().top - $(':root').prop('scrollTop')) < 200
        }
      })
    })
  }

  render() {
    return (
      <Container id="content">
        <Strength active={this.state.active.strength} />
        <h2 className="center blue-text text-lighten-1"><MaterialIcon color="red" icon="chevron_right" size="medium" />Example<MaterialIcon icon="chevron_left" color="red" size="medium" /></h2>
        <hr />
        <Benchmark id="benchmark" active={this.state.active.benchmark} />
      </Container>
    )
  }
}

export default Content