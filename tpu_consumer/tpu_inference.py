import time

import cv2
import numpy as np
from pycoral.adapters import common
from pycoral.adapters import detect
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter
import zmq

MODEL_PATH = "ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite"
LABELS_PATH = "coco_labels.txt"
CONFIDENCE_THRESHOLD = 0.5


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

    print("[TPU] Starting TPU Consumer...", flush=True)

    # Engine initialization on the Google Coral Edge TPU ASIC (INT8)
    interpreter = make_interpreter(MODEL_PATH)
    interpreter.allocate_tensors()
    labels = read_label_file(LABELS_PATH)

    print(
        "[TPU] INT8 model loaded on Edge TPU. Processing frames...",
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

            # Frame preprocessing (RGB conversion and 300x300 resizing)
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, (300, 300))

            # INT8 inference on the Edge TPU systolic array
            start_time = time.time()
            common.set_input(interpreter, img_resized)
            interpreter.invoke()
            raw_detections = detect.get_objects(
                interpreter, score_threshold=CONFIDENCE_THRESHOLD
            )
            latency_ms = (time.time() - start_time) * 1000.0
            fps = 1000.0 / latency_ms if latency_ms > 0 else 0.0

            # Bounding box coordinate and label postprocessing
            detections = []
            for det in raw_detections:
                bbox = det.bbox
                detections.append(
                    {
                        "class_id": det.id,
                        "label": labels.get(det.id, f"Object {det.id}"),
                        "left": int(bbox.xmin),
                        "top": int(bbox.ymin),
                        "right": int(bbox.xmax),
                        "bottom": int(bbox.ymax),
                    }
                )

            # Telemetry dispatch (Graphical Decoupling)
            telemetry = {
                "node": "TPU",
                "fps": round(fps, 1),
                "latency_ms": round(latency_ms, 1),
                "detections": detections,
            }
            telemetry_publisher.send_json(telemetry)

            print(
                f"[TPU] Frame processed | FPS: {fps:.1f} | Latency: {latency_ms:.1f} ms",
                flush=True,
            )

        except Exception as exc:
            print(
                f"[TPU] Error in reception/inference loop: {exc}",
                flush=True,
            )


if __name__ == "__main__":
    main()
