import threading
import time

import cv2
from flask import Flask, Response
import zmq

# ZeroMQ connection (In-memory frame publisher)
context = zmq.Context()
frame_publisher = context.socket(zmq.PUB)
frame_publisher.bind("tcp://*:5555")

# MJPEG streaming server for the Visualization Host
app = Flask(__name__)
current_frame = None


def capture_and_publish_frames():
    """Camera capture, frame preprocessing, and ZeroMQ in-memory publishing."""
    global current_frame
    print("[PRODUCER] Starting video capture on /dev/video0...", flush=True)
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while True:
        ret, frame = cap.read()
        if not ret:
            print(
                "[PRODUCER] Warning: Failed to read frame from /dev/video0",
                flush=True,
            )
            time.sleep(0.1)
            continue

        # Frame preprocessing (Resize to 300x300 and JPEG encoding)
        frame_resized = cv2.resize(frame, (300, 300))
        _, buffer = cv2.imencode(".jpg", frame_resized)
        frame_bytes = buffer.tobytes()

        # ZeroMQ in-memory broadcast to consumer nodes
        frame_publisher.send(frame_bytes)
        current_frame = frame_bytes


def generate_mjpeg_stream():
    """Generates a continuous MJPEG stream for the Visualization Host."""
    global current_frame
    while True:
        if current_frame is not None:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + current_frame + b"\r\n"
            )
        time.sleep(0.03)


@app.route("/video")
def video_feed():
    """Raw MJPEG streaming endpoint."""
    return Response(
        generate_mjpeg_stream(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


if __name__ == "__main__":
    capture_thread = threading.Thread(
        target=capture_and_publish_frames, daemon=True
    )
    capture_thread.start()
    print(
        "[PRODUCER] MJPEG Server and ZeroMQ Publisher active (Ports 5000 / 5555)...",
        flush=True,
    )
    app.run(host="0.0.0.0", port=5000, threaded=True)
