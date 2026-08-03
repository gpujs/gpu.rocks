# Run the Benchmark Gauntlet in a container and hand back a saved run.
#
#   docker build -t gpu-rocks-bench .
#   docker run --rm --gpus all -v "$PWD/out:/out" gpu-rocks-bench
#
# The result lands in ./out as a JSON file that the page can read.
#
# ── READ THIS BEFORE YOU TRUST A NUMBER ─────────────────────────────────────
#
# A container reaches a GPU only if the host lets it. On Linux with the NVIDIA
# container toolkit, or with /dev/dri passed through for Intel and AMD, this
# works. On Docker Desktop for macOS or Windows it does NOT: those run Linux in
# a VM with no GPU passthrough, so Chromium falls back to SwiftShader, and
# SwiftShader is a CPU pretending to be a GPU.
#
# That failure is silent unless something checks, so scripts/bench-record.mjs
# checks: it reads the WebGPU adapter and the WebGL renderer before it spends
# the time, and refuses to write a run if either is software. A refusal here is
# the tool working. Passing --allow-software overrides it, and the numbers you
# get are then a measurement of your CPU wearing a GPU's label.
#
# Chromium, not google-chrome-stable: Google ships that deb for amd64 only, and
# half the machines anyone would want to run this on are arm64.
FROM node:22-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
      chromium \
      # Vulkan loader + Mesa ICDs. NVIDIA supplies its own ICD through the
      # container toolkit; these cover Intel and AMD, and vulkaninfo is here so
      # a failing run can be diagnosed without rebuilding the image.
      libvulkan1 \
      mesa-vulkan-drivers \
      vulkan-tools \
      # Chromium wants a writable font cache and these two are what its headless
      # compositor links against even with no display attached
      fontconfig \
      libgbm1 \
      libgl1 \
      ca-certificates \
      git \
      # headless-gl builds from source here. Nothing in this image uses it —
      # the recorder drives a real browser — but gpu.js declares `gl` as a hard
      # dependency rather than an optional one, so yarn has to compile it or
      # the install fails outright.
      build-essential \
      python3 \
      # node-gyp shells out to `python`, and Debian ships only `python3`
      python-is-python3 \
      pkg-config \
      libgl1-mesa-dev \
      libxi-dev \
      libx11-dev \
      libxext-dev \
    && rm -rf /var/lib/apt/lists/*

ENV CHROME_PATH=/usr/bin/chromium

# Headless Chromium on Linux does not use the GPU by default, and which flags
# turn it on depends on the driver stack. This default targets NVIDIA through
# the container toolkit. Override the whole variable for other stacks:
#   docker run -e CHROME_FLAGS='--use-angle=gl --enable-unsafe-webgpu' ...
#
# --disable-vulkan-surface is NOT optional here, and its absence fails in the
# most expensive way available: WebGL reaches the card and WebGPU silently does
# not. --enable-features=Vulkan asks the GPU process for its own Vulkan
# backend, which WebGPU needs; that instance is created with presentation
# -surface extensions, which do not exist in a container with no display, so
# vkCreateInstance fails with -7 (VK_ERROR_EXTENSION_NOT_PRESENT), the GPU
# process abandons Vulkan, and Dawn falls back to SwiftShader without saying
# why. Nothing in this image ever presents to a screen, so the requirement is
# pure cost. Measured on an RTX 5090 / driver 580.159.03 / Chromium 151:
# without it the recorder refuses every run on NVIDIA. See docker/README.md.
ENV CHROME_FLAGS="--use-angle=vulkan --enable-features=Vulkan --disable-vulkan-surface --enable-unsafe-webgpu --no-sandbox"

WORKDIR /app

# Dependencies first so a source edit does not re-resolve the tree. gpu.js is
# currently a git dependency (the WebAssembly branch), which is why git is
# installed above. --ignore-engines because react-router declares a node floor
# above what some base images ship; nothing in this project needs it.
COPY package.json yarn.lock ./
RUN yarn install --frozen-lockfile --ignore-engines

COPY . .
RUN yarn build

# /out is where the run lands. Declared so `docker run` without -v still has
# somewhere to write rather than failing at the last step of a long benchmark.
VOLUME /out

# Arguments pass straight through, so any recorder flag works:
#   docker run ... gpu-rocks-bench --label "RTX 4090 · Linux"
#   docker run ... gpu-rocks-bench --columns webgpu
ENTRYPOINT ["node", "scripts/bench-record.mjs", "--out", "/out"]
CMD ["--label", "container run"]
