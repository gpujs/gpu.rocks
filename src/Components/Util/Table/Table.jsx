import React from 'react'

export default function Table(props = {
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
    <table className={`${props.striped ? 'striped ' : ''}responsive-table highlight centered`}>
      <thead>
        <tr>
          {props.headings.map((heading, i) => <th key={i}>{heading}</th>)}
        </tr>
      </thead>
      <tbody>
        {
          props.rows.map(row => (
            <tr>
              {row.map((col, i) => <td key={i}>{col}</td>)}
            </tr>
          ))
        }
      </tbody>
    </table>
  )
}
