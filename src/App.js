import React from 'react'
import { BrowserRouter, Switch, Route } from 'react-router-dom'
import Main from './Components/Main/Main'
import Benchmark from './Components/Benchmark/Benchmark'
import Install from './Components/Install/Install'
import Nav from './Components/Nav/Nav'
import PageFooter from './Components/PageFooter/PageFooter'
import Examples from './Components/Examples/Examples'
import Playground from './Components/Playground/Playground'

import 'materialize-css'
import './scss/index.scss'
import '../node_modules/materialize-css/dist/css/materialize.css'
import 'material-icons'

function App() {
  return (
    <BrowserRouter>
      <Nav />
      <Switch>
        <Route path="/benchmark" component={Benchmark} />
        <Route path="/install" component={Install} />
        <Route path="/examples" component={Examples} />
        <Route path="/playground" component={Playground} />
        <Route component={Main} />
      </Switch>
      <PageFooter />
    </BrowserRouter>
  )
}

export default App;
