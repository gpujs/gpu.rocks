import React, { Component } from 'react'
import Row from 'react-materialize/lib/Row'
import Col from 'react-materialize/lib/Col'
import M from 'materialize-css'

import links from './examples-links'

import './Examples.scss'

export class Examples extends Component {
  componentDidMount() {
    const elems = document.querySelectorAll('.materialboxed');
    M.Materialbox.init(elems);
  }

  render() {
    const linksDOM = links.map(link => {
      return (
        <Col s={12} l={6}>
          <div className="card examples-card">
            <div className="card-image">
              <img src={link.img} alt={link.title + ' Image'} className="materialboxed" />
              <span className="card-title">{link.title}</span>
            </div>
            <div className="card-content">
              { link.author !== undefined ? <b>By <a href={link.author.link}>{link.author.name}</a></b> : ''}
              <p>{link.description}</p>
            </div>
            <div className="card-action">
              {link.footerLinks}
            </div>
          </div>
        </Col>
      )
    })

    let rows = [], i = 0;

    while (i < linksDOM.length) {
      rows.push((
        <Row>
          {
            [
            linksDOM[i],
            linksDOM[Math.min(i + 1, linksDOM.length)]
            ]
          }
        </Row>
      ))

      i += 2
    }

    return (
      <div id="examples">
        <h2 className="center">Examples</h2>
        <h6 className="center">All the below examples are by the community!</h6>

        <hr />
        <div className="examples-component">
          {
            rows
          }
        </div>
      </div>
    )
  }
}

export default Examples
