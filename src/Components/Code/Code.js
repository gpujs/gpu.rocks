import React from 'react'
import SyntaxHighlighter from 'react-syntax-highlighter';
import { monokai } from 'react-syntax-highlighter/dist/esm/styles/hljs';

import './Code.scss'

const Code = (props) => {
  return (
    <code className="code">
      <SyntaxHighlighter language='javascript' style={monokai}>{props.children}</SyntaxHighlighter>
    </code>
  )
}

export default Code