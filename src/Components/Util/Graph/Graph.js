import React from 'react'
import { ResponsiveLine } from '@nivo/line'

import './Graph.scss'

const Graph = ({info, interpolation, title}) => {
  return (
    <div className="graph">
      <ResponsiveLine
        data={info.data}
        margin={{ top: 20, right: 100, bottom: 80, left: 100 }}
        xScale={{ type: 'point' }}
        yScale={{ type: 'linear', min: 'auto', max: 'auto' }}
        curve="linear"
        axisTop={null}
        axisRight={null}
        axisBottom={{
            orient: 'bottom',
            tickSize: 5,
            tickPadding: 8,
            tickRotation: 25,
            legend: title.x,
            legendOffset: 35,
            legendPosition: 'middle'
        }}
        axisLeft={{
            orient: 'left',
            tickSize: 8,
            tickPadding: 5,
            tickRotation: 0,
            legend: title.y,
            legendOffset: -70,
            legendPosition: 'middle'
        }}
        enableGridX={false}
        pointSize={8}
        pointColor={{ theme: 'background' }}
        pointBorderWidth={2}
        pointBorderColor={{ from: 'serieColor' }}
        enablePointLabel={false}
        pointLabelYOffset={-12}
        enableSlices="x"
        enableCrosshair={false}
        useMesh={true}
        legends={[
            {
                anchor: 'bottom-right',
                direction: 'column',
                justify: true,
                translateX: 100,
                translateY: -20,
                itemsSpacing: 0,
                itemDirection: 'right-to-left',
                itemWidth: 90,
                itemHeight: 20,
                itemOpacity: 0.75,
                symbolSize: 12,
                symbolShape: 'circle',
                symbolBorderColor: 'rgba(0, 0, 0, .5)',
                effects: [
                    {
                        on: 'hover',
                        style: {
                            itemBackground: 'rgba(0, 0, 0, .03)',
                            itemOpacity: 1
                        }
                    }
                ]
            }
        ]}
        colors={d => d.color} // sets custom colors
        animate={false}
        />
      </div>
  )
}

export default Graph
