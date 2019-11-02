import React from 'react'
import Example from '../Example/Example'
import Strength from '../Strength/Strength'
import Syntax from '../Syntax/Syntax'
import ServerBenchmarks from '../ServerBenchmarks/ServerBenchmarks'
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
      <ServerBenchmarks />
      <OldVersions />
    </div>
  )
}

export default Content

