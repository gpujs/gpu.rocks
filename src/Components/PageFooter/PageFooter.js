import React from 'react'
import Footer from 'react-materialize/lib/Footer'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { faGithub } from '@fortawesome/free-brands-svg-icons'
import { faExclamationCircle, faHistory } from '@fortawesome/free-solid-svg-icons'

import './PageFooter.scss'

const PageFooter = () => {
  return (
    <Footer
      links={
        <ul>
          <li>
            <a href="https://github.com/gpujs/gpu.js" className="white-text btn btn-flat"><FontAwesomeIcon icon={faGithub} size="lg" /> Github</a>
          </li>
          <li>
            <a href="https://github.com/gpujs/gpu.js/releases/latest" className="white-text btn btn-flat"><FontAwesomeIcon icon={faExclamationCircle} size="lg" /> Latest Release</a>
          </li>
          <li>
            <a href="https://github.com/gpujs/gpu.js/releases" className="white-text btn btn-flat"><FontAwesomeIcon icon={faHistory} size="lg" /> Older Releases</a>
          </li>
        </ul>
        }
      moreLinks={<div className="center"><span>&copy; {new Date().getFullYear()} GPU.js Org</span> | Images(except logo) by <a href="https://pixabay.com/users/CopyrightFreePictures-203/" className="white-text">CopyrightFreePictures</a> from <a href="https://pixabay.com/" className="white-text">Pixabay</a></div>}
      className="sticky"
    >
      <h6>
        <strong>GPU.js - GPU accelerated JavaScript</strong>
      </h6>
    </Footer>
  )
}

export default PageFooter
