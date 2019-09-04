import React, { Component } from 'react'
import { Button } from 'react-materialize'
import $ from 'jquery'

import './ScrollButton.scss'

export default class ScrollButton extends Component {
  componentDidMount() {
    $(document).on('scroll', () => {
      console.log('scroll')
      if ($(':root').prop('scrollTop') > 50) $('#scroll-up-btn').fadeIn()
      else $('#scroll-up-btn').fadeOut()
    })
  }

  handleClick() {
    $(':root').animate({scrollTop: 0})
  }

  render() {
    return (
      <Button
        floating
        id="scroll-up-btn"
        fab={{direction: 'left', hoverEnabled: false}}
        icon="arrow_upward"
        style={{display: 'none'}}
        large
        onClick={this.handleClick}
      />
    )
  }
}
