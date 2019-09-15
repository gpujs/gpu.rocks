import React, { Component } from 'react'
import Row from 'react-materialize/lib/Row'
import Col from 'react-materialize/lib/Col'
import M from 'materialize-css'

import mandelbrotSet from '../../img/examples/mandelbrot-set.png'
import slowFade from '../../img/examples/slow-fade.png'

import './Examples.scss'

export class Examples extends Component {
  componentDidMount() {
    const elems = document.querySelectorAll('.materialboxed');
    M.Materialbox.init(elems);
  }

  render() {
    return (
      <div id="examples">
        <h2 className="center">Examples</h2>
        <hr />
        <div className="examples-component">
          <Row>
            <Col s={6}>
              <div className="card">
                <div className="card-image">
                  <img src={slowFade} alt="slow-fade" className="materialboxed" />
                  <span className="card-title">Slow Fade</span>
                </div>
                <div className="card-content">
                  <p>A simple example wherein colors slowly fade in and fade out.</p>
                </div>
                <div className="card-action">
                  <a href="https://observablehq.com/@robertleeplummerjr/gpu-js-example-slow-fade">Observable Notebook</a>
                </div>
              </div>
            </Col>

            <Col s={6}>
              <div className="card">
                <div className="card-image">
                  <img src={mandelbrotSet} alt="mandelbrot-set" className="materialboxed" />
                  <span className="card-title">Mandelbrot Set</span>
                </div>
                <div className="card-content">
                  <p>A plot of a particular set of complex numbers called the Mandelbrot set.</p>
                </div>
                <div className="card-action">
                  <a href="https://observablehq.com/@robertleeplummerjr/gpu-js-example-slow-fade">Observable Notebook</a>
                  <a href="https://en.wikipedia.org/wiki/Mandelbrot_set">Read More</a>
                </div>
              </div>
            </Col>
          </Row>
        </div>
      </div>
    )
  }
}

export default Examples
