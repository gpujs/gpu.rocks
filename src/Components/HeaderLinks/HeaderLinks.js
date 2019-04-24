import React from 'react'
import { NavLink } from 'react-router-dom'
import { Button } from 'react-materialize'
import Tooltip from "react-simple-tooltip"

const HeaderLinks = () => {
  return (
    <div className="container center">
      <NavLink to="/docs"><Tooltip content="Documentation" padding="4" fadeDuration="400"><Button waves="dark" className="blue darken-3" style={{marginRight: '5px'}}>Docs</Button></Tooltip></NavLink>
      <Button waves="dark" className="blue darken-3" style={{marginRight: '5px'}}>Download</Button>
      <Button waves="dark" className="blue darken-3" style={{marginRight: '5px'}}>Repo</Button>
      <Button waves="dark" className="blue darken-3" style={{marginRight: '5px'}}>Install</Button>
    </div>
  )
}

export default HeaderLinks