import FirefoxData from './gtx-1080/firefox'
import ChromeData from './gtx-1080/chrome'
import EdgeData from './gtx-1080/edge'
import mbpData from './mbp/firefox'

export const gtx1080 = [
  FirefoxData,
  ChromeData,
  EdgeData
]

export const gtx1080Obj = {
  firefox: FirefoxData,
  chrome: ChromeData,
  edge: EdgeData
}

export const mbpFirefox = [mbpData]

const DevelopmentBenchmarks = {
  gtx1080,
  mbpFirefox
}

export default DevelopmentBenchmarks