import React from 'react'
import SyntaxHighlighter from 'react-syntax-highlighter/dist/cjs/light';
import monokai from 'react-syntax-highlighter/dist/cjs/styles/hljs/monokai'
import js from 'react-syntax-highlighter/dist/cjs/languages/hljs/javascript'
import bash from 'react-syntax-highlighter/dist/cjs/languages/hljs/bash'
import html from 'react-syntax-highlighter/dist/cjs/languages/hljs/vbscript-html'
import Button from 'react-materialize/lib/Button'
import MaterialIcon from '../MaterialIcon/MaterialIcon'
import M from 'materialize-css'

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
      {/* materialize ships its own tooltip; react-simple-tooltip predates
          React 18 and broke on 19, and this was its only use */}
      <span className="right copy-tooltip" data-tooltip="Copy">
        <Button floating waves="light" className="grey darken-3" style={{marginTop: '1rem', marginRight: '1rem'}} onClick={() => copy(props.code, () => M.toast({html: 'Code Copied', classes: 'rounded'}))}>
          <MaterialIcon icon="content_copy" size="small" color="white" />
        </Button>
      </span>

      <SyntaxHighlighter language={props.language || 'javascript'} style={monokai}>{props.code}</SyntaxHighlighter>
    </div>
  )
}

export default Code

