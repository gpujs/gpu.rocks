import React from 'react'
import { HashRouter, Routes, Route } from 'react-router';
import Main from './Components/Main/Main'
import Benchmark from './Components/Benchmark/Benchmark'
import Install from './Components/Install/Install'
import Nav from './Components/Nav/Nav'
import PageFooter from './Components/PageFooter/PageFooter'
import Examples from './Components/Examples/Examples'

import 'materialize-css'
import './scss/index.scss'
import 'materialize-css/dist/css/materialize.css'
import 'material-icons'

function App() {
  return (
    <HashRouter>
      <Nav />
      <Routes>
        <Route path="/benchmark" element={<Benchmark />} />
        <Route path="/install" element={<Install />} />
        <Route path="/examples" element={<Examples />} />
        <Route path="*" element={<Main />} />
      </Routes>
      <PageFooter />
    </HashRouter>
  )
}

export default App;
