FROM nvidia/cuda:12.8.0-base-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-lc"]

# ---- basic system deps ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates bzip2 \
    build-essential cmake ninja-build \
    python3 python3-pip python3-dev python3-setuptools \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# --- vulkan setup ---

RUN apt-get update && \
    apt-get install -y --no-install-recommends vulkan-tools libvulkan1 && \
    rm -rf /var/lib/apt/lists/*

# ---- install Miniconda ----
ENV CONDA_DIR=/opt/conda
RUN curl -sSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p ${CONDA_DIR} && \
    rm /tmp/miniconda.sh
ENV PATH=${CONDA_DIR}/bin:$PATH

# ---- accept Anaconda ToS (fix CondaToSNonInteractiveError) ----
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# ---- create env ----
ARG PY_VER=3.9
RUN conda create -n seq_multi_grasp python=${PY_VER} -y && conda clean -afy

# Always run inside env
ENV CONDA_DEFAULT_ENV=seq_multi_grasp
ENV PATH=${CONDA_DIR}/envs/seq_multi_grasp/bin:$PATH

# Upgrade pip tooling
RUN python -m pip install --upgrade pip setuptools wheel

WORKDIR /workspace
CMD ["bash"]
