import React from 'react'

export default function MaterialIcon(props) {
  return (
    <i class={`material-icons ${props.size || 'small'} ${props.classes || ''}`} style={{color: props.color || ''}}>{props.icon}</i>
  )
}
