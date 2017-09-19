(function() {
  'use strict';

  //
  // Elements
  //
  var $runBenchmark = $('#run-benchmark');
  var $textures = $('[name="texture"]');
  var $outputTo = $('.output-to');
  var $benchmarkStats = $('#benchmark-stats');
  var $textureModeDisabled = $('#texture-mode-disabled');
  var $textureModeEnabled = $('#texture-mode-enabled');

  //
  // Startup code
  //
  var matrixSize = 512;
  var allowChangeOutput = true;
  var a = new Array(matrixSize * matrixSize);
  var b = new Array(matrixSize * matrixSize);
  a = splitArray(fillArrayRandom(a), matrixSize);
  b = splitArray(fillArrayRandom(b), matrixSize);

  //
  // Important stuff
  //
  var cpu = new GPU({ mode: 'cpu' });
  var gpu = new GPU({ mode: 'gpu' });
  var multiplyMatrix = {
    cpu: createMultiplyKernel(cpu),
    gpu: createMultiplyKernel(gpu)
  };

  //
  // These are the droids you are looking for
  //
  function createMultiplyKernel(gpu) {
    var options = {
      output: [matrixSize, matrixSize],
      outputToTexture: $outputTo.filter(':checked').val() === 'texture'
    };

    return gpu.createKernel(function(a, b) {
      var sum = 0;
      for (var i = 0; i < 512; i++) {
        sum += a[this.thread.y][i] * b[i][this.thread.x];
      }
      return sum;
    }, options);
  }

  //
  // Events
  //
  $runBenchmark.click(function(e) {
    e.preventDefault();
    runBenchmark();
  });
  $textures.change(function() {
    if (!allowChangeOutput) return;
    multiplyMatrix.cpu = createMultiplyKernel(cpu);
    multiplyMatrix.gpu = createMultiplyKernel(gpu);
  });

  var suite = new Benchmark.Suite()
    .add('multiplyMatrix.cpu', function() {
      return multiplyMatrix.cpu(a, b);
    })
    .add('multiplyMatrix.gpu', function() {
      return multiplyMatrix.gpu(a, b);
    })
    .on('complete', function() {
      enableTextureChange();
      var stats = {
        cpu: this.filter(function (benchmark) {
          return benchmark.name === 'multiplyMatrix.cpu';
        }).map('stats')[0],
        gpu: this.filter(function (benchmark) {
          return benchmark.name === 'multiplyMatrix.gpu';
        }).map('stats')[0]
      };

      var faster = '';
      if (stats.cpu.mean > stats.gpu.mean) {
        var times = stats.cpu.mean / stats.gpu.mean;
        faster = ' <em>(' + times.toFixed(2) + ' times faster!)</em>';

        if (times > 10) {
          faster += '<img style="width: 100%;" src="https://media.giphy.com/media/aD7fneoMfS6Yw/giphy.gif" />';
        }
      }
      var html = '\
      <p>CPU: ' + stats.cpu.mean.toFixed(3) +'s \xb1' + stats.cpu.rme.toFixed(1) + '%</p>\
      <p>GPU: ' + stats.gpu.mean.toFixed(3) + 's \xb1' + stats.gpu.rme.toFixed(1) + '%' + faster + '</p>\
      <small><em>Benchmarks provided by <a href="https://github.com/bestiejs/benchmark.js">benchmark.js</a></em></small>';
      $benchmarkStats.removeClass('text-center');
      if (stats.gpu.rme !== 0) {
        $benchmarkStats.html(html);
      }
    })
    .on('error', function(e) {
      console.log(e);
      enableTextureChange();
      $benchmarkStats
        .removeClass('text-center')
        .removeClass('alert-info')
        .addClass('alert-danger')
        .html('There was an error running on the GPU.<br><br>If you see this, its a bug. Please file it with your hardware, OS and browser information. It will greatly help us in our efforts.<br><a href="https://github.com/gpujs/gpu.js/issues" class="btn btn-outline">Report</a>');
    });

  //
  // And finally, the culmination
  //
  function runBenchmark() {
    disableTextureChange();
    $benchmarkStats
      .removeClass('hide')
      .addClass('text-center')
      .html('<i class="fa fa-cog fa-spin" style="font-size: 60px;"></i>');

    suite.run({ async: true });
  }

  $outputTo
    .change(function() {
      if (this.value === 'number') {
        $textureModeDisabled.show();
        $textureModeEnabled.hide();
      } else {
        $textureModeDisabled.hide();
        $textureModeEnabled.show();
      }
    });
  $outputTo.filter(':checked').change();

	
	//
	// Benchmark Chart
	//
	var chartData = {
		mbp2012: [],
		gtx1080: []
	};
	
	var chartBrowsers = {
		firefox: {
			displayName: 'Firefox',
			lineColor: 'rgb(237, 111, 33)',
			shadowColor: 'rgba(237, 111, 33, 0.1)'
		},
		chrome: {
			displayName: 'Chrome',
			lineColor: 'rgb(10, 181, 59)',
			shadowColor: 'rgba(10, 181, 59, 0.1)'
		},
		edge: {
			displayName: 'Edge',
			lineColor: 'rgb(31, 119, 180)',
			shadowColor: 'rgba(31, 119, 180, 0.1)'
		}
	};
	
	loadJSON('mbp2012', 'firefox', 1);
	loadJSON('gtx1080', 'firefox', 3);
	loadJSON('gtx1080', 'chrome', 3);
	loadJSON('gtx1080', 'edge', 3);

  //
  // Utilities
  //
  function fillArrayRandom(array) {
    for(var i = 0; i < array.length; i++) {
      array[i] = Math.random();
    }
    return array;
  }

  function splitArray(array, part) {
    var result = [];
    for(var i = 0; i < array.length; i += part) {
      result.push(array.slice(i, i + part));
    }
    return result;
  }

  function disableTextureChange() {
    allowChangeOutput = false;
    $textures.prop('disabled', true);
  }

  function enableTextureChange() {
    allowChangeOutput = true;
    $textures.prop('disabled', false);
  }
	
	function loadJSON(benchmarkName, browserName, numBenchmarks) {
		var div = 'benchmark-plot-' + benchmarkName;
		var filename = 'js/benchmark-' + benchmarkName + '-' + browserName + '.json';
		var lineColor = chartBrowsers[browserName].lineColor;
		var shadowColor = chartBrowsers[browserName].shadowColor;
		var browserDisplayName = chartBrowsers[browserName].displayName;
		
		$.getJSON(filename, function(benchmark) {
			var data = [
				{
					name: browserDisplayName + ' (Upper Bound)',
					x: benchmark.x_series,
					y: benchmark.y_series_upper,
					type: 'scatter',
					mode: 'lines',
					marker: {color:'#444', },
					line: {width:0},
				},
				{
					name: browserDisplayName + ' (Mean)',
					x: benchmark.x_series,
					y: benchmark.y_series,
					type: 'scatter',
					mode: 'lines',
					line: {
						color: lineColor
					},
					fillcolor: shadowColor,
					fill: 'tonexty'
				},
				{
					name: browserDisplayName + ' (Lower Bound)',
					x: benchmark.x_series,
					y: benchmark.y_series_lower,
					type: 'scatter',
					mode: 'lines',
					marker: {color:'#444'},
					line: {width:0},
					fillcolor: shadowColor,
					fill: 'tonexty'
				}
			];
			
			chartData[benchmarkName] = chartData[benchmarkName].concat(data);
			
			if (chartData[benchmarkName].length === numBenchmarks * 3) {
				var layout = {
					yxais: {
						type: 'log',
						autorange: true
					},
					showlegend: false,
					hovermode: 'closest'
				};
				
				Plotly.newPlot(div, chartData[benchmarkName], layout);
			}
		});
	}
})();
