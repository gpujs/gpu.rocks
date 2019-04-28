import React from 'react'
import Header from '../Header/Header'
import Content from '../Content/Content'
import OldVersions from '../OldVersions/OldVersions'

const Main = () => {
  return (
    <div className="wrapper">
      <Header />
      <Content />
      <OldVersions />
    </div>
  )
}

export default Main