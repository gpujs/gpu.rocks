import React from 'react'
import { Button } from 'react-materialize'
import Tooltip from 'react-simple-tooltip'

const LinkBtn = (props) => {
  return (
    <Tooltip border="rgba(0, 0, 0, 0)" radius="8" background="rgba(60, 60, 80, 0.7)" content={props.tooltipContent} padding="4" fadeDuration="400">
      <Button waves="dark" className="blue darken-3 header-btn">
        {props.btnContent}
      </Button>
    </Tooltip>
  )
}

export default LinkBtn