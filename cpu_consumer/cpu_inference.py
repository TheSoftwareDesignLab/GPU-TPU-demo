import time

import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter
import zmq

MODEL_PATH = "detect.tflite"
LABELS_PATH = "coco_labels.txt"
CONFIDENCE_THRESHOLD = 0.5


def load_labels(path):
    """Loads the COCO label file into a dictionary."""
    with open(path, "r", encoding="utf-8") as file:
        return {i: line.strip() for i, line in enumerate(file.readlines())}


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

    print("[CPU] Starting CPU Consumer...", flush=True)
    print(
        "[CPU] Loading standard model (FP32) for sequential execution...",
        flush=True,
    )

    # Sequential TFLite inference engine initialization (FP32/CPU)
    labels = load_labels(LABELS_PATH)
    interpreter = Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    _, height, width, _ = input_details[0]["shape"]

    print("[CPU] TFLite engine ready. Processing frames...", flush=True)

    prev_time = time.time()

    while True:
        try:
            # Frame reception from ZeroMQ queue
            frame_bytes = frame_subscriber.recv()
            np_array = np.frombuffer(frame_bytes, np.uint8)
            frame = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

            if frame is None:
                continue

            start_time = time.time()

            # Frame preprocessing (RGB conversion and input tensor shaping)
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, (width, height))
            input_data = np.expand_dims(img_resized, axis=0)

            # Sequential FP32 inference on ARM Cortex-A57 CPU
            interpreter.set_tensor(input_details[0]["index"], input_data)
            interpreter.invoke()

            boxes = interpreter.get_tensor(output_details[0]["index"])[0]
            classes = interpreter.get_tensor(output_details[1]["index"])[0]
            scores = interpreter.get_tensor(output_details[2]["index"])[0]

            latency_ms = (time.time() - start_time) * 1000.0
            current_time = time.time()
            fps = (
                1.0 / (current_time - prev_time)
                if (current_time - prev_time) > 0
                else 0.0
            )
            prev_time = current_time

            # Bounding box coordinate and label postprocessing
            detections = []
            for i, score in enumerate(scores):
                if score > CONFIDENCE_THRESHOLD:
                    ymin, xmin, ymax, xmax = boxes[i]
                    class_id = int(classes[i])
                    detections.append(
                        {
                            "class_id": class_id,
                            "label": labels.get(
                                class_id, f"Object {class_id}"
                            ),
                            "left": int(xmin * 300),
                            "top": int(ymin * 300),
                            "right": int(xmax * 300),
                            "bottom": int(ymax * 300),
                        }
                    )

            # Telemetry dispatch (Graphical Decoupling)
            telemetry = {
                "node": "CPU",
                "fps": round(fps, 1),
                "latency_ms": round(latency_ms, 1),
                "detections": detections,
            }
            telemetry_publisher.send_json(telemetry)

            print(
                f"[CPU] Frame processed | FPS: {fps:.1f} | Latency: {latency_ms:.1f} ms",
                flush=True,
            )

        except Exception as exc:
            print(
                f"[CPU] Error in reception/inference loop: {exc}",
                flush=True,
            )


if __name__ == "__main__":
    main()
