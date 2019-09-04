import React from 'react'
import { Container, Card } from 'react-materialize'
import MaterialIcon from '../Util/MaterialIcon/MaterialIcon'

import './OldVersions.scss'

const OldVersions = () => {
  return (
    <Container className="container-fluid">
      <h2 className="center">Older Versions</h2>
      <hr />
      <Card
        className="center"
        title="v0.0 Alpha Hackathon"
        actions={[<a className="btn btn-floating waves-effect waves-light grey darken-3" href="https://gpu.rocks/v0.0/gpu.min.js" ><MaterialIcon icon="file_download" color="white" /></a>]}
        key="v0.0"
      >
        <h5 className="grey-text">TUE FEB 28 2017</h5>
        <p>This is the legacy version for backward compatiblity with the demos made using the "hackathon" API.</p>
      </Card>
    </Container>
  )
}

export default OldVersions