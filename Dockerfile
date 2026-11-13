# Streamlit demo (UNet model). Build for amd64: docker build --platform linux/amd64 -t cxas:interactive .
# The full DICOM/IDC image (segmentation + landmarks, DICOM SEG/SR) is docker/Dockerfile.
FROM python:3.11-slim-bookworm

# Set the working directory inside the container
WORKDIR /app

# Copy the Streamlit app and all necessary files to the container
COPY ./weights/ /app/.cxas/weights
COPY interactive_cxas_app.py /app/

# Set environment variables
ENV CXAS_PATH=/app/

# Install system-level dependencies (libGL for OpenCV and other dependencies)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    libjpeg-dev \
    zlib1g-dev \
    libpng-dev \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    libsm6 \
    libxext6 && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install setuptools wheel && \
    pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu && \
    pip install cxas==0.0.20 streamlit==1.40.1 opencv-python-headless==4.10.0.84 pillow==11.0.0 "numpy<2"

# Expose the default port for Streamlit (8501)
EXPOSE 8501

# Command to run the Streamlit app on container startup
CMD ["streamlit", "run", "interactive_cxas_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
