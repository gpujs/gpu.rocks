import React from 'react'
import SyntaxHighlighter from 'react-syntax-highlighter';
import { monokai } from 'react-syntax-highlighter/dist/esm/styles/hljs';
import { Button } from 'react-materialize'
import MaterialIcon from 'material-icons-react'

import './Code.scss'

const Code = (props) => {
  return (
    <div className="code">
      <Button floating waves="light" className="right grey darken-3" style={{marginTop: '1rem', marginRight: '1rem'}}><MaterialIcon icon="content_copy" size="small" color="white" /></Button>
      <SyntaxHighlighter language='javascript' style={monokai}>{props.children}</SyntaxHighlighter>
    </div>
  )
}

export default Code