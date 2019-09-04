import React from 'react'
import Example from '../Example/Example'
import Strength from '../Strength/Strength'
import Syntax from '../Syntax/Syntax'
import DevBenchmarks from '../DevBenchmarks/DevBenchmarks'
import ScrollButton from '../ScrollButton/ScrollButton'
import OldVersions from '../OldVersions/OldVersions'

import './Content.scss'

function Content() {
  return (
    <div id="content">
      <ScrollButton />
      <Strength />
      <Example />
      <Syntax />
      <DevBenchmarks />
      <OldVersions />
    </div>
  )
}

export default Content