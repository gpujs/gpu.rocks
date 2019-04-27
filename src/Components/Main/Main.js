import React from 'react'
import Header from '../Header/Header'
import Content from '../Content/Content'
import OldVersions from '../OldVersions/OldVersions'
import PageFooter from '../PageFooter/PageFooter'

const Main = () => {
  return (
    <div>
      <Header />
      <Content />
      <OldVersions />
      <PageFooter />
    </div>
  )
}

export default Main