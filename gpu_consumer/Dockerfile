# Optimized base image with TensorRT plugins for NVIDIA Jetson Nano
FROM dustynv/jetson-inference:r32.7.1

WORKDIR /app
ENV DEBIAN_FRONTEND=noninteractive

# Install ZeroMQ orchestration dependency
RUN pip3 install pyzmq

COPY gpu_inference.py .

CMD ["python3", "gpu_inference.py"]
