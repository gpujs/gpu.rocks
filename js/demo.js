(function() {
  //
  // Elements
  //
  var $runBenchmark = $('#run-benchmark');
  var $textures = $('[name="texture"]');
  var $outputToTexture = $('#output-to-texture');
  var $outputToNumber = $('#output-to-number');
  var $benchmarkStats = $('#benchmark-stats');
  var $jsCode = $('.js');

  //
  // Startup code
  //
  var matrixSize = 1;
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
      dimensions: [matrixSize, matrixSize],
      outputToTexture: $outputToTexture[0].checked
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
      console.dir(stats);

      var faster = '';
      if (stats.cpu.mean > stats.gpu.mean) {
        var times = stats.cpu.mean / stats.gpu.mean;
        faster = ' <em>(' + times.toFixed(2) + ' times faster!)</em>';

        if (times > 10) {
          faster += '<img style="width: 100%;" src="https://media.giphy.com/media/mYZFipl0aV0Tm/giphy.gif" />';
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

  //
  // Code Shenanigans
  //

  $outputToTexture.change(toggleCode);
  $outputToNumber.change(toggleCode);
  
  function toggleCode() {

    function searchSetDimensions(line) {
      return line.match(/setDimensions/g);
    }

    var code = $jsCode.text().split("\n");
    var codeDim = code.find(searchSetDimensions);
    var index = code.findIndex(searchSetDimensions);

    if ($outputToTexture.is(':checked')) {
      codeDim = codeDim.split(/;/g)[0] + '.setOutputToTexture(true);';
    } else {
      codeDim = codeDim.split(/.setOutput/)[0] + ';';
    }
    
    code[index] = codeDim;
    $jsCode.text(code.join("\n"));
    hljs.highlightBlock($jsCode[0]);
    
  }


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
    allowChangeOutput = false;
    $textures.prop('disabled', false);
  }
})();