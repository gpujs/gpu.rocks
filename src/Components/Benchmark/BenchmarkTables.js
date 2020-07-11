import React from 'react'
import Table from '../Util/Table/Table'

export default function BenchmarkTables({bench, cpu, performerMap}) {
  window.performerMap = performerMap;
  return (
    <div id="out">
      <h4>Score</h4>

      <Table
        headings={['Mode', 'Score']}
        rows={[
          ['GPU', bench.score['gpu']],
          ['CPU', cpu ? bench.score['cpu'] : 'Not Benchmarked']
        ]}
      />
      <hr />

      <h4>Build Times</h4>
      <Table
        headings={['Benchmark', 'Time Taken(GPU)', 'Time Taken(PIPELINE)']}
        rows={[
          ['Matrix Multiplication', `${bench.build_time.mat_mult['gpu']} ms`, `${bench.build_time.mat_mult['pipe']} ms`],
          ['Matrix Convolution', `${bench.build_time.mat_conv['gpu']} ms`, `${bench.build_time.mat_conv['pipe']} ms`]
        ]}
      />
      <hr />
      
      <h4>Run Times</h4>
      <Table
        headings={['Benchmark', `Time Taken(GPU)`, `Time Taken(CPU)`]}
        rows={[
          [
            `Matrix Multiplication`,
            <span><b>Avg</b>: {bench.run_time.mat_mult['gpu'].avg} ms &plusmn; {bench.run_time.mat_mult['gpu'].deviation}%<br /></span>,
            cpu ? <span><b>Avg</b>: {bench.run_time.mat_mult['cpu'].avg} ms &plusmn; {bench.run_time.mat_mult['cpu'].deviation}%<br /></span>
            : `Not Benchmarked`
          ],
          [
            `Matrix Convolution`,
            <span><b>Avg</b>: {bench.run_time.mat_conv['gpu'].avg} ms &plusmn; {bench.run_time.mat_conv['gpu'].deviation}%<br /></span>,
            cpu ? <span><b>Avg</b>: {bench.run_time.mat_conv['cpu'].avg} ms &plusmn; {bench.run_time.mat_conv['cpu'].deviation}%<br /></span>
            : `Not Benchmarked`
          ]
        ]}
      />
      <hr />

      <h4><a href="https://github.com/gpujs/gpu.js#pipelining">Pipelining</a> Benchmark</h4>
      <Table
        headings={['Benchmark', 'Time Taken(GPU)', 'Time Taken(CPU)']}
        rows={[
          [
            `Matrix Multiplication`,
            <span><b>Avg</b>: {bench.run_time.pipe['gpu'].avg} ms &plusmn; {bench.run_time.pipe['gpu'].deviation}%<br /></span>,
            cpu ? <span><b>Avg</b>: {bench.run_time.pipe['cpu'].avg} ms &plusmn; {bench.run_time.pipe['cpu'].deviation}%<br /></span>
            : `Not Benchmarked`
          ]
        ]}
      />
      <hr />

      <h4>Statistics</h4>
      <h5>Build Times</h5>

      <Table
        headings={[`Benchmark`, `GPU v/s PIPELINE`]}
        rows={[
          [
            `Matrix Multiplication`,
            <span><b>{performerMap[bench.stats.build_time.mat_mult.diff.gpu_pipe.winner]}</b> took {bench.stats.build_time.mat_mult.diff.gpu_pipe.percentage}% less time to compile</span>
          ],
          [
            `Matrix Convolution`,
            <span><b>{performerMap[bench.stats.build_time.mat_conv.diff.gpu_pipe.winner]}</b> took {bench.stats.build_time.mat_conv.diff.gpu_pipe.percentage}% less time to compile</span>
          ]
        ]}
      />
      <hr />

      <h5>Run Times</h5>
      <Table
        headings={[`Benchmark`, `GPU v/s CPU`]}
        rows={[
          [
            `Matrix Multiplication`,
            `
              ${ cpu ?
                `${performerMap[bench.stats.run_time.mat_mult.diff.cpu_gpu.avg.winner]} took ${bench.stats.run_time.mat_mult.diff.cpu_gpu.avg.percentage}% less time`
                : `Not Benchmarked`
              }
            `
          ],
          [
            `Matrix Convolution`,
            `
              ${ cpu ?
                `${performerMap[bench.stats.run_time.mat_conv.diff.cpu_gpu.avg.winner]} took ${bench.stats.run_time.mat_conv.diff.cpu_gpu.avg.percentage}% less time`
                : `Not Benchmarked`
              }
            `
          ]
        ]}
      />
    </div>
  )
}
