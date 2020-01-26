import React from 'react'
import { Light as SyntaxHighlighter } from 'react-syntax-highlighter'
import { monokai } from 'react-syntax-highlighter/dist/esm/styles/hljs'
import js from 'react-syntax-highlighter/dist/esm/languages/hljs/javascript'
import bash from 'react-syntax-highlighter/dist/esm/languages/hljs/bash'
import html from 'react-syntax-highlighter/dist/esm/languages/hljs/vbscript-html'
import { Button } from 'react-materialize'
import MaterialIcon from '../MaterialIcon/MaterialIcon'
import M from 'materialize-css'
import Tooltip from 'react-simple-tooltip'

import './Code.scss'

SyntaxHighlighter.registerLanguage('javascript', js)
SyntaxHighlighter.registerLanguage('bash', bash)
SyntaxHighlighter.registerLanguage('html', html)

const Code = (props) => {

  const copy = (text, cb) => {
    const el = document.createElement('textarea');
    el.value = text;
    document.body.appendChild(el);
    el.select();
    document.execCommand('copy');
    document.body.removeChild(el);
    if (cb) cb()
  }

  return (
    <div className="code">
      <Tooltip content="Copy" placement="left" className="right" radius={10} padding={8} fadeDuration={400} >
        <Button floating waves="light" className="grey darken-3" style={{marginTop: '1rem', marginRight: '1rem'}} onClick={() => copy(props.code, () => M.toast({html: 'Code Copied', classes: 'rounded'}))}>
          <MaterialIcon icon="content_copy" size="small" color="white" />
        </Button>
      </Tooltip>

      <SyntaxHighlighter language={props.language || 'javascript'} style={monokai}>{props.code}</SyntaxHighlighter>
    </div>
  )
}

export default Code

