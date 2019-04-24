import React from 'react'
import thread from '../../img/thread.png'
import './Content.scss'

const Content = () => {
 return (
   <div id="content">
    <img className="thread" src={thread} alt="thread" />
    <img className="thread" src={thread} alt="thread" />
    <img className="thread" src={thread} alt="thread" />
    <img className="thread" src={thread} alt="thread" />
    <img className="thread" src={thread} alt="thread" />
   </div>
 )
}

export default Content