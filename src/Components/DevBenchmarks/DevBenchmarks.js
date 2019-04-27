import React from 'react'
import Heading from '../Heading/Heading'
import { ResponsiveLine } from '@nivo/line'
import { Container } from 'react-materialize'
import { gtx1080 } from '../../Data/DevelopmentBenchmarks'

const DevBenchmarks = () => {

  const data = [
    {...gtx1080.firefox},
    {...gtx1080.edge},
    {...gtx1080.chrome}
  ]

  return (
    <div id="dev-benchmarks">
      <Heading active={true}>Development Benchmarks</Heading>
      <Container class="chart-container">
        <ResponsiveLine data={data} />
      </Container>
    </div>
  )
}

export default DevBenchmarks