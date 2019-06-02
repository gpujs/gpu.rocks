import React from 'react'
import Heading from '../Heading/Heading'
import Red from './Red'
import { NavLink } from 'react-router-dom'
import { Container, Row, Col, Collection, CollectionItem } from 'react-materialize'

const Syntax = (props) => {
  return (
    <div>
      <Heading active={props.active.syntax} id="syntax">Syntax Support</Heading>
      <h6 className="center"><b>GPU.js</b> relies on the assumption that the kernel function is using only a subset of legal JavaScript syntax</h6>

      <Container>
        <Row>
          <Col offset="s2" s={8}>
            <Collection>
              <CollectionItem>
                <b>1D</b>, <b>2D</b>, <b>3D</b> <Red>array</Red> of numbers or just numbers as kernel input or output
              </CollectionItem>

              <CollectionItem>
                <Red>Number</Red> Variables
              </CollectionItem>

              <CollectionItem>
                Custom and custom native <Red>function</Red>s
              </CollectionItem>

              <CollectionItem>
                Arithmetic operators (<Red>+</Red>, <Red>+=</Red>, <Red>-</Red>, <Red>*</Red>, <Red>/</Red>, <Red>%</Red>)
              </CollectionItem>

              <CollectionItem>
                <i>Some</i> Javascript Math functions like <Red>Math.floor()</Red> (See the <NavLink to="/docs">Docs</NavLink>)
              </CollectionItem>

              <CollectionItem>
                <Red>Math.random()</Red> is supported but it isn't perfect (See the <NavLink to="/docs">Docs</NavLink>)
              </CollectionItem>

              <CollectionItem>
                <Red>Pipelining</Red> (See the <NavLink to="/docs">Docs</NavLink>)
              </CollectionItem>

              <CollectionItem>
                <Red>for</Red>, <Red>while</Red> and other loops
              </CollectionItem>

              <CollectionItem>
                <Red>if</Red> and <Red>else</Red> statements but they are not recommended (See the <NavLink to="/docs">Docs</NavLink>)
              </CollectionItem>

              <CollectionItem>
                <Red>const</Red>, <Red>let</Red> and <Red>var</Red> variables
              </CollectionItem>

              <CollectionItem>
                <b>NO</b> variables captured by a closure
              </CollectionItem>
            </Collection>
          </Col>
        </Row>
      </Container>
    </div>
  )
}

export default Syntax