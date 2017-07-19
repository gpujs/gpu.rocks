var cpu = new GPU({
	'mode': 'cpu'
});
var gpu = new GPU({
	'mode': 'gpu'
});

function splitArray(array, part) {
	var tmp = [];
	for(var i = 0; i < array.length; i += part) {
		tmp.push(array.slice(i, i + part));
	}
	return tmp;
}

//
// Startup code
//
var mat_size = 512;
var A = [];
var B = [];
for(var n = 0; n < mat_size*mat_size; n++) {
	var randA = Math.random();
	var randB = Math.random();
	A.push(randA);
	B.push(randB);
}
A = splitArray(A, mat_size);
B = splitArray(B, mat_size);

function createMultFromGPU(gpu) {
	var opt = {
		dimensions: [mat_size, mat_size]
	};

	return gpu.createKernel(function(A, B) {
		var sum = 0;
		for (var i=0; i<512; i++) {
			sum += A[this.thread.y][i] * B[i][this.thread.x];
		}
		return sum;
	}, opt);
}

var mult = {
	cpu: createMultFromGPU(cpu),
	gpu: createMultFromGPU(gpu)
};

var benchmarkOpt = {};

var suite = new Benchmark.Suite;

suite.add('mat_mult_cpu', function() {
	var mode = 'cpu';
	var mat_mult = mult[mode];

	var C = mat_mult(A, B);

	return C;
}, benchmarkOpt);

suite.add('mat_mult_gpu', function() {
	var mode = 'gpu';
	var mat_mult = mult[mode];

	var C = mat_mult(A, B);

	return C;
}, benchmarkOpt);

suite.on('complete', function(event) {

	var stats = {};

	stats.cpu = this.filter(function(benchmark) {
		return benchmark.name == 'mat_mult_cpu';
	}).map('stats')[0];

	stats.gpu = this.filter(function(benchmark) {
		return benchmark.name == 'mat_mult_gpu';
	}).map('stats')[0];

	console.dir(stats);

	var faster = '';
	if (stats.cpu.mean > stats.gpu.mean) {
		var times = stats.cpu.mean / stats.gpu.mean;
		faster = ' <em>(' + times.toFixed(2) + ' times faster!)</em>';
	}
	var html = '';
	html += '<p>CPU: ' + stats.cpu.mean.toFixed(3) +'s \xb1' + stats.cpu.rme.toFixed(1) + '%</p>'
	html += '<p>GPU: ' + stats.gpu.mean.toFixed(3) + 's \xb1' + stats.gpu.rme.toFixed(1) + '%' + faster + '</p>';
	html += '<small><em>Benchmarks provided by <a href="https://github.com/bestiejs/benchmark.js">benchmark.js</a></em></small>';
	$('.demo-mult').removeClass('text-center');
	if (stats.gpu.rme != 0) {
		$('.demo-mult').html(html);
	}
});

suite.on('error', function(event) {
	$('.demo-mult').removeClass('text-center');
	$('.demo-mult').removeClass('alert-info');
	$('.demo-mult').addClass('alert-danger');
	$('.demo-mult').html('There was an error running on the GPU.<br><br>If you see this, its a bug. Please file it with your hardware, OS and browser information. It will greatly help us in our efforts.<br><a href="https://github.com/gpujs/gpu.js/issues" class="btn btn-outline">Report</a>');
});

function demoMult() {
	$('.demo-mult').removeClass('hide');
	$('.demo-mult').addClass('text-center');
	$('.demo-mult').html('<i class="fa fa-cog fa-spin" style="font-size: 60px;"></i>');

	suite.run({ 'async': true });
}
