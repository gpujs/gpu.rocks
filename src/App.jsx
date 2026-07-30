import React, { Suspense, lazy, useEffect } from 'react'
import { BrowserRouter, Routes, Route, Outlet, useLocation } from 'react-router';
import { siteMeta } from './routeMeta'
import { setPageMeta } from './pageMeta'
import Main from './Components/Main/Main'
import Benchmark from './Components/Benchmark/Benchmark'
import Install from './Components/Install/Install'
import Nav from './Components/Nav/Nav'
import PageFooter from './Components/PageFooter/PageFooter'
import Examples from './Components/Examples/Examples'

// The learn platform (course content, CodeMirror, engine) is by far the
// heaviest part of the site — load it only when a learn route is visited.
const LearnApp = lazy(() => import('./Learn/LearnApp'))

import 'materialize-css'
import './scss/index.scss'
import 'materialize-css/dist/css/materialize.css'
import 'material-icons'

// Existing site pages keep the shared Nav/PageFooter chrome; the learn
// routes below render standalone (they bring their own nav).
function SiteLayout() {
  // history navigation doesn't reload the document, so retitle it here;
  // routeMeta is the same source of truth the prerender bakes into static HTML
  const { pathname } = useLocation()
  useEffect(() => {
    setPageMeta(siteMeta(pathname))
  }, [pathname])
  return (
    <>
      <Nav />
      <Outlet />
      <PageFooter />
    </>
  )
}

// Theme-neutral placeholder shown for the moment the learn chunk downloads;
// it follows the OS scheme so there is no flash on either theme.
function LearnFallback({ children }) {
  const dark = window.matchMedia('(prefers-color-scheme: dark)').matches
  return (
    <Suspense fallback={
      <div style={{
        minHeight: '100dvh',
        background: dark ? '#050218' : '#f6f6fb',
        color: dark ? '#9b94c0' : '#5f5a80',
        display: 'grid',
        placeItems: 'center',
        fontSize: '.85rem',
      }}>Loading the course…</div>
    }>
      {children}
    </Suspense>
  )
}

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/learn/*" element={<LearnFallback><LearnApp /></LearnFallback>} />
        <Route path="/learn-verify" element={<LearnFallback><LearnApp verify /></LearnFallback>} />
        <Route element={<SiteLayout />}>
          <Route path="/benchmark" element={<Benchmark />} />
          <Route path="/install" element={<Install />} />
          <Route path="/examples" element={<Examples />} />
          <Route path="*" element={<Main />} />
        </Route>
      </Routes>
    </BrowserRouter>
  )
}

export default App;
