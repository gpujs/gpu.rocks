import gpu_run_time_mat_mult from './gpu_run_time_mat_mult.json'
import cpu_run_time_mat_mult from './cpu_run_time_mat_mult.json'

export default {
  gpu_run_time_mat_mult: {
    series: gpu_run_time_mat_mult,
    colors: {
      lineColor: '#00ff00',
      ptColor: '#00cc00'
    }
  },
  cpu_run_time_mat_mult: {
    series: cpu_run_time_mat_mult,
    colors: {
      lineColor: '#4db8ff',
      ptColor: '#0099ff'
    }
  }
}