FROM python:3.9-slim

WORKDIR /app

# Install minimal system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Use opencv-python-headless to avoid GUI library dependencies
RUN pip install opencv-python-headless pyzmq flask

COPY producer.py .

CMD ["python", "producer.py"]
