import CpuData from './different-sizes-cpu.json'
import GpuData from './different-sizes-gpu.json'
import PipeData from './different-sizes-pipe.json'

export default [
  {
    data: CpuData,
    displayName: 'CPU',
    lineColor: 'rgb(34, 106, 193)',
    shadowColor: 'rgba(10, 181, 59, 0.1)',
    lineOptions: {
      shape: 'spline'
    }
  },
  {
    data: GpuData,
    displayName: 'GPU',
    lineColor: 'rgb(20, 188, 62)',
    shadowColor: 'rgba(10, 181, 59, 0.1)',
    lineOptions: {
      shape: 'spline'
    }
  },
  {
    data: PipeData,
    displayName: 'GPU(pipeline mode)',
    lineColor: 'rgb(183, 64, 12)',
    shadowColor: 'rgba(10, 181, 59, 0.1)',
    lineOptions: {
      shape: 'spline'
    }
  }
]