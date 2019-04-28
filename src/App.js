import React from 'react'
import { BrowserRouter, Switch, Route } from 'react-router-dom'
import Main from './Components/Main/Main'
import Benchmark from './Components/Benchmark/Benchmark'

import 'materialize-css'
import './scss/index.scss'
import '../node_modules/materialize-css/dist/css/materialize.css'
import 'material-icons'

function App() {
  return (
    <BrowserRouter>
      <Switch>
        <Route exact path="/" component={Main} />
        <Route path="/benchmark" component={Benchmark} />
      </Switch>
    </BrowserRouter>
  )
}

export default App;
