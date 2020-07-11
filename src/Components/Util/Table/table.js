import React from 'react'

export default function Table(options = {
  striped: false,
  headings: ['col1'],
  rows: [
    [
      'c1r1' // col1 row1
    ],
    [
      'c1r2'
    ]
  ]
}) {
  return (
    <table class="{options.striped ? 'striped ' : ''}responsive-table highlight centered">
      <thead>
        {options.headings.map(heading => <th>{heading}</th>)}
      </thead>
      <tbody>
        {
          options.rows.map(row => (
            <tr>
              {row.cols.map(col => <td>{col}</td>)}
            </tr>
          ))
        }
      </tbody>
    </table>
  )
}
