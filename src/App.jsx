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
import UpdateBanner from './UpdateBanner'
import { isStale, looksLikeChunkFailure, markStale } from './updateState'

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

// A lazily-loaded chunk that 404s takes the whole tree down with it, and the
// cause is almost always that this tab outlived its deploy. Catch it, tell the
// banner, and offer the fix rather than showing a blank page.
class ChunkErrorBoundary extends React.Component {
  constructor(props) {
    super(props)
    this.state = { failed: false }
  }

  static getDerivedStateFromError(error) {
    // React does not always hand the boundary the underlying fetch failure, so
    // also trust the shared flag: the resource-error and unhandled-rejection
    // listeners see the 404 directly and set it before this renders.
    return { failed: true, stale: looksLikeChunkFailure(error) || isStale() }
  }

  componentDidCatch(error) {
    if (looksLikeChunkFailure(error)) markStale()
  }

  render() {
    if (!this.state.failed) return this.props.children
    const dark = window.matchMedia('(prefers-color-scheme: dark)').matches
    return (
      <div style={{
        minHeight: '100dvh',
        background: dark ? '#050218' : '#f6f6fb',
        color: dark ? '#f6f7f8' : '#191333',
        display: 'grid',
        placeItems: 'center',
        textAlign: 'center',
        padding: '2rem',
        gap: '1rem',
      }}>
        <div>
          <p style={{ margin: '0 0 1rem' }}>
            {this.state.stale
              ? 'This page was open while gpu.rocks was updated, so part of it is no longer available.'
              : 'Something went wrong loading this page.'}
          </p>
          <button
            type="button"
            onClick={() => window.location.reload()}
            style={{
              background: '#20a4f3', color: '#fff', border: 'none', borderRadius: 7,
              padding: '.5rem 1.2rem', font: 'inherit', fontWeight: 700, cursor: 'pointer',
            }}
          >
            Reload
          </button>
        </div>
      </div>
    )
  }
}

// Theme-neutral placeholder shown for the moment the learn chunk downloads;
// it follows the OS scheme so there is no flash on either theme.
function LearnFallback({ children }) {
  const dark = window.matchMedia('(prefers-color-scheme: dark)').matches
  return (
    <ChunkErrorBoundary>
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
    </ChunkErrorBoundary>
  )
}

function App() {
  return (
    <BrowserRouter>
      {/* site-wide: a stale tab breaks on any route, not just the course */}
      <UpdateBanner />
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
