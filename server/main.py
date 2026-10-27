import asyncio

from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
import uvicorn
import zmq

app = FastAPI()

# ZeroMQ connection (Telemetry receiver from CPU, GPU, and TPU nodes)
context = zmq.Context()
telemetry_subscriber = context.socket(zmq.SUB)
telemetry_subscriber.bind("tcp://0.0.0.0:5556")
telemetry_subscriber.setsockopt_string(zmq.SUBSCRIBE, "")

# Remote visualization interface (Graphical Decoupling via HTML5 Canvas)
HTML_DASHBOARD = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Edge AI - CPU vs GPU vs TPU</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background: #121212;
            color: white;
            text-align: center;
            margin: 0;
            padding: 20px;
            min-width: 1050px;
        }
        .dashboard {
            display: flex;
            justify-content: center;
            gap: 30px;
            margin-top: 20px;
            flex-wrap: nowrap;
        }
        .node-container {
            display: flex;
            flex-direction: column;
            align-items: center;
        }
        .video-wrapper {
            position: relative;
            width: 300px;
            height: 300px;
        }
        img, canvas {
            position: absolute;
            top: 0;
            left: 0;
            width: 300px;
            height: 300px;
            border: 2px solid #333;
            border-radius: 8px;
        }
        .telemetry-panel {
            width: 274px;
            margin-top: 15px;
            background: rgba(0, 0, 0, 0.8);
            padding: 10px 13px;
            border-radius: 5px;
            text-align: left;
            font-size: 14px;
        }
        .panel-cpu { border-left: 3px solid #FF3333; }
        .panel-gpu { border-left: 3px solid #00FF00; }
        .panel-tpu { border-left: 3px solid #00FFFF; }
        .metric-title { font-weight: bold; margin-bottom: 5px; }
        .metric-value { font-family: monospace; color: white; }
    </style>
</head>
<body>
    <h2>Edge AI Benchmark: Hardware Architecture Comparison</h2>

    <div class="dashboard">
        <div class="node-container">
            <div class="video-wrapper">
                <img src="http://192.168.55.1:5000/video" alt="Video">
                <canvas id="canvas_cpu" width="300" height="300"></canvas>
            </div>
            <div class="telemetry-panel panel-cpu">
                <div class="metric-title" style="color: #FF3333;">ARM CPU (FP32)</div>
                <div>Throughput: <span id="val_fps_cpu" class="metric-value">0.0</span> FPS</div>
                <div>Latency: <span id="val_lat_cpu" class="metric-value">0.0</span> ms</div>
            </div>
        </div>

        <div class="node-container">
            <div class="video-wrapper">
                <img src="http://192.168.55.1:5000/video" alt="Video">
                <canvas id="canvas_gpu" width="300" height="300"></canvas>
            </div>
            <div class="telemetry-panel panel-gpu">
                <div class="metric-title" style="color: #00FF00;">NVIDIA GPU (FP16)</div>
                <div>Throughput: <span id="val_fps_gpu" class="metric-value">0.0</span> FPS</div>
                <div>Latency: <span id="val_lat_gpu" class="metric-value">0.0</span> ms</div>
            </div>
        </div>

        <div class="node-container">
            <div class="video-wrapper">
                <img src="http://192.168.55.1:5000/video" alt="Video">
                <canvas id="canvas_tpu" width="300" height="300"></canvas>
            </div>
            <div class="telemetry-panel panel-tpu">
                <div class="metric-title" style="color: #00FFFF;">Google TPU (INT8)</div>
                <div>Throughput: <span id="val_fps_tpu" class="metric-value">0.0</span> FPS</div>
                <div>Latency: <span id="val_lat_tpu" class="metric-value">0.0</span> ms</div>
            </div>
        </div>
    </div>

    <script>
        const ctxCpu = document.getElementById('canvas_cpu').getContext('2d');
        const ctxGpu = document.getElementById('canvas_gpu').getContext('2d');
        const ctxTpu = document.getElementById('canvas_tpu').getContext('2d');
        const ws = new WebSocket("ws://" + window.location.host + "/ws");

        function drawBoundingBoxes(ctx, detections, color, nodeName) {
            ctx.clearRect(0, 0, 300, 300);
            detections.forEach(box => {
                ctx.strokeStyle = color;
                ctx.lineWidth = 2;
                ctx.strokeRect(box.left, box.top, box.right - box.left, box.bottom - box.top);
                ctx.fillStyle = color;
                ctx.font = "14px Arial";
                ctx.fillText(nodeName + ": " + box.label, box.left, box.top > 20 ? box.top - 5 : 20);
            });
        }

        ws.onmessage = function(event) {
            const data = JSON.parse(event.data);

            if (data.node === "CPU") {
                document.getElementById("val_fps_cpu").innerText = data.fps;
                document.getElementById("val_lat_cpu").innerText = data.latency_ms;
                drawBoundingBoxes(ctxCpu, data.detections, "#FF3333", "CPU");
            } else if (data.node === "GPU") {
                document.getElementById("val_fps_gpu").innerText = data.fps;
                document.getElementById("val_lat_gpu").innerText = data.latency_ms;
                drawBoundingBoxes(ctxGpu, data.detections, "#00FF00", "GPU");
            } else if (data.node === "TPU") {
                document.getElementById("val_fps_tpu").innerText = data.fps;
                document.getElementById("val_lat_tpu").innerText = data.latency_ms;
                drawBoundingBoxes(ctxTpu, data.detections, "#00FFFF", "TPU");
            }
        };
    </script>
</body>
</html>
"""


@app.get("/")
async def get_dashboard():
    """Serves the remote visualization dashboard to the client."""
    return HTMLResponse(HTML_DASHBOARD)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Forwards ZeroMQ telemetry to the browser over WebSockets."""
    await websocket.accept()
    while True:
        try:
            telemetry_data = telemetry_subscriber.recv_json(flags=zmq.NOBLOCK)
            await websocket.send_json(telemetry_data)
        except zmq.Again:
            await asyncio.sleep(0.01)


if __name__ == "__main__":
    print("[SERVER] FastAPI & WebSocket Telemetry Server listening on port 8000...", flush=True)
    uvicorn.run(app, host="0.0.0.0", port=8000)
