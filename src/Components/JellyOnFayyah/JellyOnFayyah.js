import React, { Component } from 'react'
import logo from '../../img/jelly.png'
import fire1 from '../../img/fire1.gif'
import fire2 from '../../img/fire2.gif'

const alt = 'GPU.js logo';
function getImage(url, height, maxWidth) {
  return (<img src={url} alt={alt} style={{
    height,
    maxWidth,
  }} />);
}

export default class JellyOnFayyah extends Component {
  constructor(props) {
    super(props);
    this.state = {
      logoIndex: -1
    };
    this.logos = [getImage(fire1, '370px', '370px'), getImage(fire2, '350px', '350px')];
    this.activeTimeout = null;
    this.img = null;
    this.setImg = this.setImg.bind(this);
    this.ignite = this.ignite.bind(this);
    this.setIgnition = this.setIgnition.bind(this);
    this.tick = this.tick.bind(this);
  }

  ignite(logoIndex) {
    if (this.state.logoIndex > -1) return;
    this.setState({logoIndex});
    this.cool();
  }

  cool() {
    if (this.activeTimeout || this.state.logoIndex === -1) return;
    this.activeTimeout = setTimeout(() => {
      this.setState({
        logoIndex: -1
      });
      this.activeTimeout = null;
    }, Math.random() * 3000);
  }

  setIgnition() {
    setInterval(this.tick, 200);
  }

  tick() {
    if (this.activeTimeout) return;
    if (Math.random() > 0.965) {
      const logoIndex = Math.floor(Math.random() * this.logos.length);
      this.ignite(logoIndex);
    }
  }

  setImg(img) {
    if (!img) return;
    this.img = img;
  }

  render() {
    return (
      <div style={{
        width: '100%',
        height: '300px',
      }}>
        {
          this.state.logoIndex === -1
            ? (
                <img
                  style={{ maxWidth: '280px', paddingTop: '40px', paddingBottom: '-40px' }}
                  ref={this.setImg}
                  src={logo}
                  alt={alt}
                  onLoad={this.setIgnition}
                />
              )
            : this.logos[this.state.logoIndex]
        }
      </div>
    )
  }
}
