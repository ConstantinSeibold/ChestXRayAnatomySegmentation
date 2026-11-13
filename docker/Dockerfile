# CXAS: chest X-ray anatomy segmentation + named landmarks, DICOM in / DICOM SEG + SR out.
#
# Build with docker/build.sh (stages the weights as a build context). All model code is part of the cxas package.
# The same Dockerfile builds a CPU image (default) and a CUDA image (--build-arg TORCH_FLAVOR=cu128: CUDA 12.8
# wheels, GPUs from Turing (T4) to Blackwell (RTX 50xx, RTX PRO, B200); cu124 wheels lack Blackwell kernels).
# Always built for linux/amd64 (IDC, MHub and cloud GPUs); the 2024 images were arm64-only.

ARG PYTHON_IMAGE=python:3.11-slim-bookworm

FROM ${PYTHON_IMAGE}
ARG TORCH_FLAVOR=cpu
ARG TORCH_VERSION=2.7.1
ARG TORCHVISION_VERSION=0.22.1
LABEL org.opencontainers.image.title="CXAS" \
      org.opencontainers.image.description="Chest X-ray anatomy segmentation (159+ structures) and named landmarks; DICOM SEG / SR output" \
      org.opencontainers.image.source="https://github.com/ConstantinSeibold/ChestXRayAnatomySegmentation" \
      org.opencontainers.image.licenses="MIT"

RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir "numpy==1.26.4" "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" \
        --index-url https://download.pytorch.org/whl/${TORCH_FLAVOR} --extra-index-url https://pypi.org/simple

# pinned runtime stack (models, highdicom output)
RUN pip install --no-cache-dir "numpy==1.26.4" \
        "transformers==4.57.1" "safetensors==0.6.2" "huggingface_hub==0.36.0" "PyYAML==6.0.2" \
        "pycocotools==2.0.10" "opencv-python-headless==4.10.0.84" "scikit-image==0.24.0" "scipy==1.14.1" \
        "pydicom==3.0.1" "highdicom==0.26.1" "pylibjpeg==2.1.0" "pylibjpeg-openjpeg==2.3.0" \
        "SimpleITK==2.4.0" "pandas==2.2.3" "matplotlib==3.9.2" "colorcet==3.1.0" "scikit-learn==1.5.2" \
        "Pillow==11.0.0" "tqdm==4.67.1" "gdown==5.2.0"

# weights baked in: no Google Drive / Hub access at run time
COPY --chmod=0755 --from=weights . /opt/cxas/.cxas/weights

# CXAS itself (after the weights, so a code change does not re-copy them)
COPY setup.py setup.cfg README.md /src/cxas/
COPY docs /src/cxas/docs
COPY cxas /src/cxas/cxas
COPY bin /src/cxas/bin
RUN pip install --no-cache-dir --no-deps /src/cxas && rm -rf /src/cxas

ENV CXAS_PATH=/opt/cxas \
    HF_HOME=/opt/hf \
    CXAS_NO_WELCOME=1 \
    MPLCONFIGDIR=/tmp/matplotlib \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

RUN useradd -m -u 1000 cxas && mkdir -p /data && chown cxas /data
USER cxas
WORKDIR /data
ENTRYPOINT ["cxas_idc"]
CMD ["--help"]
