import React from 'react'
import Heading from '../Heading/Heading'

const DevBenchmarks = ({active}) => {
  return (
    <div className="dev-benchmarks">
      <Heading  active={active.devBenchmarks}>Development Benchmarks</Heading>
    </div>
  )
}

export default DevBenchmarks