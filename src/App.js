import React from 'react'
import * as HashRouter from 'react-router-dom/HashRouter';
import * as Switch from 'react-router-dom/Switch'
import * as Route from 'react-router-dom/Route'
import Main from './Components/Main/Main'
import Benchmark from './Components/Benchmark/Benchmark'
import Install from './Components/Install/Install'
import Nav from './Components/Nav/Nav'
import PageFooter from './Components/PageFooter/PageFooter'
import Examples from './Components/Examples/Examples'

import 'materialize-css'
import './scss/index.scss'
import '../node_modules/materialize-css/dist/css/materialize.css'
import 'material-icons'

function App() {
  return (
    <HashRouter hashType="slash">
      <Nav />
      <Switch>
        <Route path="/benchmark" component={Benchmark} />
        <Route path="/install" component={Install} />
        <Route path="/examples" component={Examples} />
        <Route component={Main} />
      </Switch>
      <PageFooter />
    </HashRouter>
  )
}

export default App;
