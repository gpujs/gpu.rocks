# Running the Benchmark Gauntlet in a container

The point of this image is to get a saved run off a machine you do not want to
install a toolchain on — a rented GPU box, a CI runner with a card in it, the
workstation under someone else's desk.

```sh
docker build -t gpu-rocks-bench .
mkdir -p out
docker run --rm --gpus all -v "$PWD/out:/out" gpu-rocks-bench --label "RTX 4090 · Linux"
```

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

## Host support

| host | GPU reachable | what you need |
|---|---|---|
| Linux + NVIDIA | yes | [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html), then `--gpus all` |
| Linux + AMD / Intel | yes | pass the render node: `--device /dev/dri` |
| Docker Desktop, macOS | **no** | Docker runs Linux in a VM with no GPU passthrough. Run the recorder natively instead: `node scripts/bench-record.mjs` |
| Docker Desktop, Windows | partial | WSL2 exposes NVIDIA GPUs to containers; other vendors generally not |
| most cloud CI runners | **no** | runners without an attached GPU will hit the software refusal |

On macOS the honest answer is to skip Docker. The host toolchain is a `yarn
install` away and the recorder already runs headless, so there is nothing the
container would buy you except a CPU pretending otherwise.

## Chromium flags

Headless Chromium does not use the GPU on Linux by default, and which flags
turn it on depends on the driver stack underneath. The image ships a default
that suits NVIDIA through the container toolkit:

```
CHROME_FLAGS="--use-angle=vulkan --enable-features=Vulkan --enable-unsafe-webgpu --no-sandbox"
```

Override the whole variable for a different stack:

```sh
# Intel/AMD via Mesa, where ANGLE-over-GL is often the working combination
docker run --rm --device /dev/dri -v "$PWD/out:/out" \
  -e CHROME_FLAGS='--use-angle=gl --enable-unsafe-webgpu --no-sandbox' \
  gpu-rocks-bench --label "Arc A770 · Linux"
```

If the run refuses with a software adapter, `vulkan-tools` is installed so you
can look before you guess:

```sh
docker run --rm --gpus all --entrypoint vulkaninfo gpu-rocks-bench --summary
```

No physical device in that output means the GPU never reached the container,
and no Chromium flag will fix it — the problem is upstream, in how the
container was started.

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
