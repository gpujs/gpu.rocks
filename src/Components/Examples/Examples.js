import React, { Component } from 'react'
import Heading from '../Heading/Heading'
import { Card, CardTitle, Row, Col, Container } from 'react-materialize'

export class Examples extends Component {
  render() {
    return (
      <div>
        <Heading active={true}>Examples</Heading>
        <Container>
          <Row>
            <Col m={4} s={6}>
              <Card header={<CardTitle />} actions={[<a href="https://github.com/something">Link</a>]} key={1}>
                Example 1
              </Card>
              <Card header={<CardTitle />} actions={[<a href="https://github.com/something">Link</a>]} key={2}>
                Example 2
              </Card>
              <Card header={<CardTitle />} actions={[<a href="https://github.com/something">Link</a>]} key={3}>
                Example 3
              </Card>
              <Card header={<CardTitle />} actions={[<a href="https://github.com/something">Link</a>]} key={4}>
                Example 4
              </Card>
            </Col>
            <Col m={4} s={6}>
              <Card header={<CardTitle />} actions={[<a href="https://github.com/something">Link</a>]} key={1}>
                Example 1
              </Card>
              <Card header={<CardTitle />} actions={[<a href="https://github.com/something">Link</a>]} key={2}>
                Example 2
              </Card>
              <Card header={<CardTitle />} actions={[<a href="https://github.com/something">Link</a>]} key={3}>
                Example 3
              </Card>
              <Card header={<CardTitle />} actions={[<a href="https://github.com/something">Link</a>]} key={4}>
                Example 4
              </Card>
            </Col>
            <Col m={4} s={6}>
              <Card header={<CardTitle />} actions={[<a href="https://github.com/something">Link</a>]} key={1}>
                Example 1
              </Card>
              <Card header={<CardTitle />} actions={[<a href="https://github.com/something">Link</a>]} key={2}>
                Example 2
              </Card>
              <Card header={<CardTitle />} actions={[<a href="https://github.com/something">Link</a>]} key={3}>
                Example 3
              </Card>
              <Card header={<CardTitle />} actions={[<a href="https://github.com/something">Link</a>]} key={4}>
                Example 4
              </Card>
            </Col>
          </Row>
        </Container>
      </div>
    )
  }
}

export default Examples
