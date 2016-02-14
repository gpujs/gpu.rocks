var gpu = new GPU();

function clone(obj) {
	if(obj === null || typeof(obj) !== 'object' || 'isActiveClone' in obj)
		return obj;

	var temp = obj.constructor(); // changed

	for(var key in obj) {
		if(Object.prototype.hasOwnProperty.call(obj, key)) {
			obj.isActiveClone = null;
			temp[key] = clone(obj[key]);
			delete obj.isActiveClone;
		}
	}

	return temp;
}

/**
 * Figure out how long it takes for a method to execute.
 *
 * @param {Function} method to test
 * @param {number} iterations number of executions.
 * @param {Array} args to pass in.
 * @param {T} context the context to call the method in.
 * @return {number} the time it took, in milliseconds to execute.
 */
var bench = function (method, iterations, args, context) {

	var time = 0;
	var timer = function (action) {
		var d = Date.now();
		if (time < 1 || action === 'start') {
			time = d;
			return 0;
		} else if (action === 'stop') {
			var t = d - time;
			time = 0;
			return t;
		} else {
			return d - time;
		}
	};

	var result = [];
	var i = 0;
	timer('start');
	while (i < iterations) {
		result.push(method.apply(context, args));
		i++;
	}

	var execTime = timer('stop');

	if ( typeof console === "object") {
		console.log("Mean execution time was: ", execTime / iterations);
		console.log("Sum execution time was: ", execTime);
		console.log("Result of the method call was:", result[0]);
	}

	return execTime / iterations;
};

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

var vec_size = 1048576 * 2 * 2;
var a = [];
for(var n = 0; n < vec_size; n++) {
	var randA = Math.round(Math.random()*65534);
	a.push(randA);
}

function benchmarkMult(mode) {
    var opt = {
        dimensions: [mat_size, mat_size],
        mode: mode
    };

    var mat_mult = gpu.createKernel(function(A, B) {
        var sum = 0;
        for (var i=0; i<512; i++) {
            sum += A[this.thread.y][i] * B[i][this.thread.x];
        }
        return sum;
    }, opt);

    var C = mat_mult(A, B);
    
    return C;
}

var reduce_maxes = {}
reduce_maxes.cpu = gpu.createKernel(function(arr) {
	var i = this.thread.x * 2;
	return Math.max(arr[i], arr[i+1]);
}, {
	mode: 'cpu',
});

reduce_maxes.gpu = gpu.createKernel(function(arr) {
	var i = this.thread.x * 2;
	return Math.max(arr[i], arr[i+1]);
}, {
	mode: 'gpu',
	hardcodeConstants: true,
	outputToTexture: true
});

function benchmarkMax(mode) {
	var reduce_max = reduce_maxes[mode];
	
	var arr = clone(a);
	
	for (var size=(vec_size); size!=1; size/=2) {
		reduce_max.dimensions([size/2]);
		arr = reduce_max(arr);
		//console.log(arr.toString());
	}
	
	return arr;
}

function demoMult() {
    $('.demo-mult').removeClass('hide');
    $('.demo-mult').addClass('text-center');
    $('.demo-mult').html('<i class="fa fa-cog fa-spin" style="font-size: 60px;"></i>');
    setTimeout(function() {
        try {
            var gpuTime = bench(benchmarkMult, 1, ['gpu'], window);
            var cpuTime = bench(benchmarkMult, 1, ['cpu'], window);
            var faster = '';
            if (cpuTime > gpuTime) {
                var times = cpuTime / gpuTime;
                faster = ' <em>(' + times.toFixed(2) + ' times faster!)</em>';
            }
            var html = '<p>CPU: ' + cpuTime +'ms</p><p>GPU: ' + gpuTime + 'ms' + faster + '</p>';
            $('.demo-mult').removeClass('text-center');
            $('.demo-mult').html(html);
        } catch (e) {
            $('.demo-mult').removeClass('text-center');
            $('.demo-mult').removeClass('alert-info');
            $('.demo-mult').addClass('alert-danger');
            $('.demo-mult').html('There was an error running on the GPU.');
        }
    }, 0);
}

function demoMax() {
    $('.demo-max').removeClass('hide');
    $('.demo-max').addClass('text-center');
    $('.demo-max').html('<i class="fa fa-cog fa-spin" style="font-size: 60px;"></i>');
    setTimeout(function() {
        var gpuTime = bench(benchmarkMax, 1, ['gpu'], window);
        var cpuTime = bench(benchmarkMax, 1, ['cpu'], window);
        var faster = '';
        if (cpuTime > gpuTime) {
            var times = cpuTime / gpuTime;
            faster = ' <em>(' + times.toFixed(2) + ' times faster!)</em>';
        }
        var html = '<p>CPU: ' + cpuTime +'ms</p><p>GPU: ' + gpuTime + 'ms' + faster + '</p>';
        $('.demo-max').removeClass('text-center');
        $('.demo-max').html(html);
    }, 0);
}
