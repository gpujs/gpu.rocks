import React from 'react'
import SyntaxHighlighter from 'react-syntax-highlighter';
import { monokai } from 'react-syntax-highlighter/dist/esm/styles/hljs';
import { Button } from 'react-materialize'
import MaterialIcon from 'material-icons-react'
import './Code.scss'

const Code = (props) => {

  const copy = (text) => {
    const el = document.createElement('textarea');
    el.value = text;
    document.body.appendChild(el);
    el.select();
    document.execCommand('copy');
    document.body.removeChild(el);
  }

  return (
    <div className="code">
      <Button floating waves="light" className="right grey darken-3" style={{marginTop: '1rem', marginRight: '1rem'}} onClick={() => copy(props.code)}><MaterialIcon icon="content_copy" size="small" color="white" /></Button>
      <SyntaxHighlighter language='javascript' style={monokai}>{props.code}</SyntaxHighlighter>
    </div>
  )
}

export default Code