import gpu_score from  './bench_gpu_score.json'
import cpu_score from './bench_cpu_score.json'

export default {
  gpu_score: {
    series: gpu_score,
    colors: {
      lineColor: '#00ff00',
      ptColor: '#00cc00'
    }
  },
  cpu_score: {
    series: cpu_score,
    colors: {
      lineColor: '#4db8ff',
      ptColor: '#0099ff'
    }
  }
}
