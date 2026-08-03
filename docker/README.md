# Running the Benchmark Gauntlet in a container

The point of this image is to get a saved run off a machine you do not want to
install a toolchain on — a rented GPU box, a CI runner with a card in it, the
workstation under someone else's desk.

```sh
docker build -t gpu-rocks-bench .
mkdir -p out
docker run --rm --gpus all -v "$PWD/out:/out" gpu-rocks-bench --label "RTX 4090 · Linux"
```

There is a `docker-compose.yml` beside the Dockerfile that does the same thing
with the working flags already filled in:

```sh
mkdir -p out
docker compose run --rm bench --label "RTX 4090 · Linux"
```

`run`, not `up`: the recorder is one-shot and its exit code is how it tells you
the container never reached the GPU.

The run takes roughly fifteen minutes and lands a JSON file in `./out`. Drop it
into `src/Bench/saved/`, re-run `node scripts/bench-record.mjs` once (or just
regenerate the index), and the page's saved-run picker will offer it.

## The part that actually matters

**A container reaches a GPU only if the host lets it.** This is the whole
difficulty, and everything below is about it.

The image does not trust that it worked. Before spending fifteen minutes,
`scripts/bench-record.mjs` reads the WebGPU adapter and the WebGL renderer and
refuses to write anything if either is software:

```
bench-record: WebGPU google swiftshader
bench-record: WebGL  none
bench-record: software renderer (adapter) — refusing to save
```

That message means the container is running on a CPU. It is not a bug in the
image; it is the image declining to hand you a number that would be a lie. A
SwiftShader "GPU" column next to a real one on the same page would make the
whole table meaningless.

`--allow-software` overrides it, and is only ever the right call if you are
deliberately measuring software rendering and will label it as such.

## The WebASM column needs cores, and the recorder needs headers

`scripts/bench-record.mjs` serves the built page with the same
`Cross-Origin-Opener-Policy` / `Cross-Origin-Embedder-Policy` headers that
gpu.rocks sets in production, on `/benchmark` and on `/assets/*`. That is what
makes the document cross-origin isolated, which is what makes
`SharedArrayBuffer` — and therefore gpu.js's threaded WebAssembly path —
available at all. Without it the WebASM column silently runs single-core: on an
M1 Max the `heat` row measures 3995 ms unisolated and 1074 ms isolated.

Both headers matter. A dedicated worker's *script* response must carry a
compatible COEP or the browser blocks it, and the benchmark then falls back to
running on the main thread — a different measurement, not merely a slower page.

Two consequences for the container:

- **Do not cap CPUs unless you mean to.** `--cpus=2` is a real setting for the
  WebASM column; the recorder reports `navigator.hardwareConcurrency` and stores
  it in the run, so a threaded number can always be read against the core count
  that produced it.
- Every saved run carries an `isolation` block — `crossOriginIsolated`,
  `sharedArrayBuffer`, `cores`. An unisolated run is a legitimate configuration
  (it is what Safari and any unconfigured host get), so the recorder records it
  rather than refusing. Read the WebASM column against it.

## Host support

| host | GPU reachable | what you need |
|---|---|---|
| Linux + NVIDIA | yes | [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html), then `--gpus all` — **and `--disable-vulkan-surface`**, or WebGPU stays on SwiftShader; see Chromium flags below |
| Linux + NVIDIA, podman | yes | CDI instead of `--gpus`: `--device nvidia.com/gpu=all`. Works rootless. Same flag caveat as above |
| Linux + AMD / Intel | yes | pass the render node: `--device /dev/dri` |
| Docker Desktop, macOS | **no** | Docker runs Linux in a VM with no GPU passthrough. Run the recorder natively instead: `node scripts/bench-record.mjs` |
| Docker Desktop, Windows | partial | WSL2 exposes NVIDIA GPUs to containers; other vendors generally not |
| most cloud CI runners | **no** | runners without an attached GPU will hit the software refusal |

On macOS the honest answer is to skip Docker. The host toolchain is a `yarn
install` away and the recorder already runs headless, so there is nothing the
container would buy you except a CPU pretending otherwise.

## Chromium flags

Headless Chromium does not use the GPU on Linux by default, and which flags
turn it on depends on the driver stack underneath. The image ships this
default:

```
CHROME_FLAGS="--use-angle=vulkan --enable-features=Vulkan --enable-unsafe-webgpu --no-sandbox"
```

**That default is not enough on NVIDIA, and it fails in the way that costs you
the most time: it reaches the card for WebGL and quietly does not for WebGPU.**
Measured on an RTX 5090, driver 580.159.03, Chromium 151:

```
bench-record: WebGPU google swiftshader
bench-record: WebGL  ANGLE (NVIDIA, Vulkan 1.4.312 (NVIDIA GeForce RTX 5090), NVIDIA)
bench-record: software renderer (adapter) — refusing to save
```

The recorder refuses on either one, so with the default alone a full run on
NVIDIA cannot be saved at all. Add `--disable-vulkan-surface`:

```
CHROME_FLAGS="--use-angle=vulkan --enable-features=Vulkan --disable-vulkan-surface --enable-unsafe-webgpu --no-sandbox"
```

```
bench-record: WebGPU nvidia blackwell
bench-record: WebGL  ANGLE (NVIDIA, Vulkan 1.4.312 (NVIDIA GeForce RTX 5090), NVIDIA)
```

The reason, from `--enable-logging=stderr --v=1`, is worth knowing because
nothing in the failure points at it. `--enable-features=Vulkan` asks the GPU
process for its own Vulkan backend, which WebGPU needs — Dawn alone is not
enough. That instance is created with presentation-surface extensions, which
do not exist in a container with no display, so:

```
ERROR:gpu/vulkan/vulkan_instance.cc:200] vkCreateInstance() failed: -7
ERROR:gpu/ipc/service/gpu_init.cc:1424] Failed to create and initialize Vulkan implementation.
```

`-7` is `VK_ERROR_EXTENSION_NOT_PRESENT`. The GPU process gives up on Vulkan,
WebGPU has no hardware path left, and Dawn falls back to SwiftShader without
saying why. Dawn itself was never the problem — it enumerates the card
perfectly well, which you can see in its own log line:

```
Warning: maxDynamicUniformBuffersPerPipelineLayout artificially reduced from 72 to 16
```

72 is what the NVIDIA device reports; SwiftShader would not say that.
`--disable-vulkan-surface` drops a requirement nothing in this image needs,
because nothing here ever presents to a screen.

One red herring lives in the same logs. The image installs
`mesa-vulkan-drivers`, so Dawn also enumerates llvmpipe and rejects it:

```
Warning: Vulkan shaderUniform*ArrayDynamicIndexing required.
 - While initializing adapter (backend=BackendType::Vulkan)
```

That is llvmpipe being turned away for reporting
`shaderSampledImageArrayDynamicIndexing = false`, which is correct behaviour
and not why WebGPU fell back. Restricting the loader to the NVIDIA ICD with
`VK_DRIVER_FILES` silences it and changes nothing else.

Override the whole variable for a different stack:

```sh
# Intel/AMD via Mesa, where ANGLE-over-GL is often the working combination
docker run --rm --device /dev/dri -v "$PWD/out:/out" \
  -e CHROME_FLAGS='--use-angle=gl --enable-unsafe-webgpu --no-sandbox' \
  gpu-rocks-bench --label "Arc A770 · Linux"
```

## Checking before you spend the half hour

The cheapest probe is the recorder itself. It reads the adapter and the
renderer before it measures anything, and the row check comes after, so asking
for a row that does not exist prints both strings and exits in about forty
seconds:

```sh
docker compose run --rm bench --only __probe__
```

```
bench-record: WebGPU nvidia blackwell
bench-record: WebGL  ANGLE (NVIDIA, Vulkan 1.4.312 (NVIDIA GeForce RTX 5090), NVIDIA)
bench-record: no such row(s): __probe__
```

Two hardware strings there mean a full run will save. Do this after any change
to flags, driver, or how the container is started.

`vulkan-tools` is also installed, so you can look at the layer underneath:

```sh
docker run --rm --gpus all --entrypoint vulkaninfo gpu-rocks-bench --summary
```

No physical device in that output means the GPU never reached the container,
and no Chromium flag will fix it — the problem is upstream, in how the
container was started.

**A physical device in that output does not mean the run will work**, which is
the trap. `vulkaninfo` listed the RTX 5090 correctly, with the right driver
version, in exactly the configuration where WebGPU was still SwiftShader.
`vulkaninfo` uses the Vulkan loader directly; Chromium's GPU process has its
own requirements on top, and it is those that fail. Trust the probe above over
this one — `vulkaninfo` tells you the card reached the container, not that
Chromium can use it.

## Passing recorder flags

Everything after the image name goes to `scripts/bench-record.mjs`:

```sh
# name the machine — this becomes the label in the saved-run picker
docker run ... gpu-rocks-bench --label "RTX 4090 · Linux · driver 560"

# just one column, merged into an existing run rather than a fresh table
docker run ... gpu-rocks-bench --columns webgpu

# watch it happen (needs an X server on the host and DISPLAY forwarded)
docker run ... gpu-rocks-bench --headed
```

## Why the image is bigger than it looks

`gpu.js` declares [`gl`](https://www.npmjs.com/package/gl) (headless-gl) as a
hard dependency rather than an optional one, so it compiles from source at
install time and drags in a toolchain plus X11 and GL headers. Nothing in this
image uses it — the recorder drives real Chromium — but the install fails
without it.
