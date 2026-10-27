FROM python:3.9-slim-bullseye

WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y libsm6 libxext6 libxrender-dev

# Install TensorFlow Lite Runtime, ZeroMQ, OpenCV and pin NumPy to <2
RUN pip install --upgrade pip && \
    pip install tflite-runtime pyzmq opencv-python-headless "numpy<2"

# Copy inference script, TFLite model (detect.tflite), and COCO labels
COPY . .

CMD ["python", "cpu_inference.py"]
