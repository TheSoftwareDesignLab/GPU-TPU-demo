FROM ubuntu:20.04

WORKDIR /app
ENV DEBIAN_FRONTEND=noninteractive

# 1. System dependencies from Ubuntu LTS repositories
RUN apt-get update && apt-get install -y wget unzip curl sudo libsm6 libxext6 libxrender-dev libusb-1.0-0 udev python3-pip

# 2. Install static Edge TPU runtime with maximum operating frequency (libedgetpu1-max)
RUN wget https://dl.google.com/coral/edgetpu_api/edgetpu_runtime_20210119.zip && \
    unzip edgetpu_runtime_20210119.zip && \
    cd edgetpu_runtime && \
    echo "Y" | bash install.sh || true

# 3. Install PyCoral API, ZeroMQ, and OpenCV dependencies
RUN pip3 install --upgrade pip wheel && \
    pip3 install --extra-index-url https://google-coral.github.io/py-repo/ pycoral~=2.0 pyzmq opencv-python-headless numpy

# 4. Copy inference script and local INT8 Edge TPU model
COPY . .

CMD ["python3", "tpu_inference.py"]
