import React from 'react'
import { Container, Card, Parallax } from 'react-materialize'
import MaterialIcon from 'material-icons-react'

import HackNRoll from '../../img/hacknroll.jpg'

const OldVersions = () => {
  return (
    <Parallax image={<img src={HackNRoll} alt="HackNRoll Hackathon" />}>
      <Container className="container-fluid">
        <Card
          className="white darken-1 center"
          textClassName="grey-text text-darken-2"
          title={<b>v0.0 Alpha "Hackathon"</b>}
          actions={[<a className="btn btn-floating waves-effect waves-light grey darken-3" href="https://gpu.rocks/v0.0/gpu.min.js" ><MaterialIcon icon="file_download" color="white" /></a>]}
        >
          <h5 className="grey-text">TUE FEB 28 2017</h5>
          This is the legacy version for backward compatiblity with the demos made using the "hackathon" API. 
        </Card>
      </Container>
    </Parallax>
  )
}

export default OldVersions