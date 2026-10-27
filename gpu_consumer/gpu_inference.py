import os
import time

import cv2
import jetson_inference
import jetson_utils
import numpy as np
import zmq

MODEL_NAME = "ssd-mobilenet-v2"
CONFIDENCE_THRESHOLD = 0.5
MODELS_DATA_DIR = "/app/models/jetson-inference/data"
APP_WORKDIR = "/app"


def main():
    # ZeroMQ connection (Subscriber with CONFLATE to prevent memory queue bottlenecks)
    context = zmq.Context()
    frame_subscriber = context.socket(zmq.SUB)
    frame_subscriber.setsockopt(zmq.CONFLATE, 1)
    frame_subscriber.connect("tcp://producer:5555")
    frame_subscriber.setsockopt_string(zmq.SUBSCRIBE, "")

    # ZeroMQ connection (Telemetry publisher to FastAPI Server)
    telemetry_publisher = context.socket(zmq.PUB)
    telemetry_publisher.connect("tcp://server:5556")

    print("[GPU] Starting GPU Consumer...", flush=True)
    print(
        "[GPU] Loading SSD-MobileNetV2 model and compiling TensorRT engine (FP16)...",
        flush=True,
    )

    # TensorRT engine initialization and compilation (FP16) on Maxwell GPU
    os.chdir(MODELS_DATA_DIR)
    net = jetson_inference.detectNet(
        MODEL_NAME, threshold=CONFIDENCE_THRESHOLD
    )
    os.chdir(APP_WORKDIR)

    print(
        "[GPU] TensorRT engine compiled. GPU ready and processing frames...",
        flush=True,
    )

    while True:
        try:
            # Frame reception from ZeroMQ queue
            frame_bytes = frame_subscriber.recv()
            np_array = np.frombuffer(frame_bytes, np.uint8)
            frame = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

            if frame is None:
                continue

            # Frame preprocessing and CUDA memory transfer
            rgba_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            cuda_img = jetson_utils.cudaFromNumpy(rgba_img)

            # Parallel FP16 inference on GPU via TensorRT
            start_time = time.time()
            raw_detections = net.Detect(cuda_img)
            latency_ms = (time.time() - start_time) * 1000.0
            fps = 1000.0 / latency_ms if latency_ms > 0 else 0.0

            # Bounding box coordinate and label postprocessing
            detections = []
            for det in raw_detections:
                detections.append(
                    {
                        "class_id": det.ClassID,
                        "label": net.GetClassDesc(det.ClassID),
                        "left": int(det.Left),
                        "top": int(det.Top),
                        "right": int(det.Right),
                        "bottom": int(det.Bottom),
                    }
                )

            # Telemetry dispatch (Graphical Decoupling)
            telemetry = {
                "node": "GPU",
                "fps": round(fps, 1),
                "latency_ms": round(latency_ms, 1),
                "detections": detections,
            }
            telemetry_publisher.send_json(telemetry)

            print(
                f"[GPU] Frame processed | FPS: {fps:.1f} | Latency: {latency_ms:.1f} ms",
                flush=True,
            )

        except Exception as exc:
            print(
                f"[GPU] Error in reception/inference loop: {exc}",
                flush=True,
            )


if __name__ == "__main__":
    main()
