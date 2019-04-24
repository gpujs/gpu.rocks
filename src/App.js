import React from 'react'
import { BrowserRouter, Switch, Route } from 'react-router-dom'
import Main from './Components/Main/Main'

import './scss/index.scss'
import 'materialize-css'
import '../node_modules/materialize-css/dist/css/materialize.css'

function App() {
  return (
    <BrowserRouter>
      <Switch>
        <Route exact path="/" component={Main} />
      </Switch>
    </BrowserRouter>
  )
}

export default App;
