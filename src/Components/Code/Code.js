import React from 'react'
import SyntaxHighlighter from 'react-syntax-highlighter';
import { monokai } from 'react-syntax-highlighter/dist/esm/styles/hljs';
import { Button } from 'react-materialize'
import MaterialIcon from 'material-icons-react'
import $ from 'jquery'

import './Code.scss'

const Code = (props) => {

  const copy = (text) => {
    const $temp = $('<input style="display:none />')
    $('body').append($temp)
    $($temp).val(text).select()
    document.execCommand('copy')
    $($temp).remove()
  }

  return (
    <div className="code">
      <Button floating waves="light" className="right grey darken-3" style={{marginTop: '1rem', marginRight: '1rem'}} onClick={() => copy(props.code)}><MaterialIcon icon="content_copy" size="small" color="white" /></Button>
      <SyntaxHighlighter language='javascript' style={monokai}>{props.code}</SyntaxHighlighter>
    </div>
  )
}

export default Code