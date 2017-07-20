const cpu = new GPU({
	'mode': 'cpu'
});

const gpu = new GPU({
	'mode': 'gpu'
});

function splitArray(array, part) {
	const tmp = [];
	for (let i = 0; i < array.length; i += part) {
		tmp.push(array.slice(i, i + part));
	}
	return tmp;
}

//
// Startup code
//
const matSize = 256;
let outputAsTexture = false;
let A = [];
let B = [];

for (let n = 0; n < matSize * matSize; n++) {
	const randA = Math.random();
	const randB = Math.random();
	A.push(randA);
	B.push(randB);
}

A = splitArray(A, matSize);
B = splitArray(B, matSize);

function createMultFromGPU(gpu) {
	const opt = {
		dimensions: [matSize, matSize]
	};

	return gpu.createKernel(function (A, B) {
		var sum = 0;
		for (var i = 0; i < 256; i++) {
			sum += A[this.thread.y][i] * B[i][this.thread.x];
		}
		return sum;
	}, opt).setOutputToTexture(outputAsTexture);
}

var mult = {
	cpu: createMultFromGPU(cpu),
	gpu: createMultFromGPU(gpu)
};

const benchmarkOpt = {};

const suite = new Benchmark.Suite;

suite.add('matMultcpu', function () {
	const mode = 'cpu';
	const matMult = mult[mode];
	const C = matMult(A, B);

	return C;
}, benchmarkOpt);

suite.add('matMultgpu', function () {
	const mode = 'gpu';
	const matMult = mult[mode];
	const C = matMult(A, B);

	return C;
}, benchmarkOpt);

suite.on('complete', function (event) {

	const stats = {};

	stats.cpu = this.filter(function (benchmark) {
		return benchmark.name == 'matMultcpu';
	}).map('stats')[0];

	stats.gpu = this.filter(function (benchmark) {
		return benchmark.name == 'matMultgpu';
	}).map('stats')[0];

	console.dir(stats);

	let faster = '';
	if (stats.cpu.mean > stats.gpu.mean) {
		const times = stats.cpu.mean / stats.gpu.mean;
		console.log(times)
		faster = `<em>(${times.toFixed(2)} times faster!)</em>`;
	}
	
	let html = '';
	html += `<p>CPU: ${stats.cpu.mean.toFixed(3)}s \xb1 ${stats.cpu.rme.toFixed(1)} %</p>`;
	html += `<p>GPU: ${stats.gpu.mean.toFixed(3)}s \xb1 ${stats.gpu.rme.toFixed(1)} % ${faster}</p>`;
	html += '<small><em>Benchmarks provided by <a href="https://github.com/bestiejs/benchmark.js">benchmark.js</a></em></small>';
	
	$('.demo-mult').removeClass('text-center');
	if (stats.gpu.rme != 0) {
		$('.demo-mult').html(html);
	}
});

suite.on('error', function (event) {
	$('.demo-mult').removeClass('text-center');
	$('.demo-mult').removeClass('alert-info');
	$('.demo-mult').addClass('alert-danger');
	$('.demo-mult').html('There was an error running on the GPU.<br><br>If you see this, its a bug. Please file it with your hardware, OS and browser information. It will greatly help us in our efforts.<br><a href="https://github.com/gpujs/gpu.js/issues" class="btn btn-outline">Report</a>');
});

function demoMult() {
	$('.demo-mult').removeClass('hide');
	$('.demo-mult').addClass('text-center');
	$('.demo-mult').html('<i class="fa fa-cog fa-spin" style="font-size: 60px;"></i>');

	suite.run({
		'async': true
	});
}

const outputMode = document.getElementById('outputMode');

outputMode.onchange = function(){
	if (outputMode.checked) {
		outputAsTexture = true;
	} else {
		outputAsTexture = false;	
	}
}