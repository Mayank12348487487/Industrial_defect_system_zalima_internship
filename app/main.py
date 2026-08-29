import os
import sys
import time
import json
import cv2
import threading
import asyncio
from typing import List, Dict, Any
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Response, UploadFile, File, Form
from starlette.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

# Workaround for OpenMP issue
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from onnx_inference import ONNXDetector
from video_stream import VideoStreamProcessor
import prometheus_client
from prometheus_client import Counter, Gauge, Summary, generate_latest, CONTENT_TYPE_LATEST

# Initialize FastAPI
app = FastAPI(title="Industrial Defect Detection Service", version="1.0.0")

# Setup Prometheus Metrics
PROMETHEUS_PREFIX = "defect_system_"
METRIC_FRAMES = Counter(f"{PROMETHEUS_PREFIX}processed_frames_total", "Total number of processed frames")
METRIC_DEFECTS = Counter(f"{PROMETHEUS_PREFIX}defects_detected_total", "Total number of defects detected by class", ["defect_type"])
METRIC_LATENCY = Summary(f"{PROMETHEUS_PREFIX}inference_latency_seconds", "Inference latency in seconds")
METRIC_FPS = Gauge(f"{PROMETHEUS_PREFIX}inference_fps", "Current inference frames per second")
METRIC_CAMERA_UPTIME = Gauge(f"{PROMETHEUS_PREFIX}camera_uptime_seconds", "Uptime of the camera stream in seconds")

# Global System State
class SystemState:
    def __init__(self):
        self.latest_frame = None
        self.latest_detections = []
        self.metrics = {"preprocess_ms": 0.0, "inference_ms": 0.0, "postprocess_ms": 0.0, "total_ms": 0.0}
        self.camera_online = False
        self.start_time = time.time()
        self.active_websockets: List[WebSocket] = []
        self.lock = threading.Lock()
        
        # Dynamic source configuration
        source_env = os.getenv("VIDEO_SOURCE", "0")
        self.source_path = int(source_env) if source_env.isdigit() else source_env
        self.source_changed = False
        
state = SystemState()

# Load detector
try:
    detector = ONNXDetector(model_path="best_industrial_defect.onnx", conf_threshold=0.25)
    state.camera_online = True
except Exception as e:
    print(f"Error initializing detector: {e}")
    sys.exit(1)

# Color mapping (BGR)
COLORS = [
    (0, 0, 255),      # Red (crazing)
    (0, 255, 0),      # Green (inclusion)
    (255, 0, 0),      # Blue (patches)
    (255, 0, 255),    # Magenta (pitted_surface)
    (255, 255, 0),    # Cyan (rolled-in_scale)
    (0, 165, 255)     # Orange (scratches)
]

def broadcast_ws_message(message: Dict[str, Any]):
    """
    Broadcasts message to all active WebSocket connections.
    Uses saved asyncio event loop since WebSocket.send_json is async.
    """
    if not state.active_websockets:
        return
        
    loop = getattr(state, "loop", None)
    if loop and loop.is_running():
        for ws in list(state.active_websockets):
            asyncio.run_coroutine_threadsafe(ws.send_json(message), loop)

def simulate_plc_broadcast(detections: List[Dict[str, Any]]):
    """
    Simulates broadcasting defect coordinates to external PLCs (Programmable Logic Controllers)
    via REST / log outputs.
    """
    if not detections:
        return
        
    plc_payload = {
        "timestamp": time.time(),
        "defect_count": len(detections),
        "defects": [
            {
                "class_name": det["class_name"],
                "confidence": round(det["score"], 4),
                "bbox_xmin_ymin_xmax_ymax": det["box"]
            } for det in detections
        ]
    }
    
    # Broadcast to websocket clients so it logs on the frontend dashboard
    broadcast_ws_message({
        "type": "plc_broadcast",
        "payload": plc_payload
    })
    
    # Print simulated PLC trigger
    print(f"[PLC SIMULATOR] Triggering PLC sorting mechanism! Sent {len(detections)} defect coordinates.")

def stream_processing_thread():
    """
    Background thread to capture camera frames, run ONNX detection,
    and update Prometheus metrics.
    """
    processor = VideoStreamProcessor(source=state.source_path, model_path="best_industrial_defect.onnx", conf_threshold=0.25)
    
    print("Background stream processing thread started.")
    state.start_time = time.time()
    
    try:
        while True:
            # Check if source was changed dynamically
            if state.source_changed:
                print(f"Dynamic source change requested. Re-initializing stream to: {state.source_path}...")
                processor.release()
                processor = VideoStreamProcessor(source=state.source_path, model_path="best_industrial_defect.onnx", conf_threshold=0.25)
                state.source_changed = False
                state.start_time = time.time()
                
            t_frame_start = time.time()
            frame = processor.get_frame()
            if frame is None:
                state.camera_online = False
                METRIC_CAMERA_UPTIME.set(0.0)
                time.sleep(0.5)
                continue
                
            state.camera_online = True
            uptime = time.time() - state.start_time
            METRIC_CAMERA_UPTIME.set(uptime)
            
            # Predict
            detections, metrics = detector.predict(frame)
            
            # Update Prometheus
            METRIC_FRAMES.inc()
            METRIC_LATENCY.observe(metrics['inference_ms'] / 1000.0) # convert to seconds
            for det in detections:
                METRIC_DEFECTS.labels(defect_type=det['class_name']).inc()
                
            fps = 1000.0 / metrics['total_ms'] if metrics['total_ms'] > 0 else 0.0
            METRIC_FPS.set(fps)
            
            # Draw detections
            annotated_frame = processor.draw_detections(frame, detections, metrics)
            
            # Encode frame
            ret, jpeg = cv2.imencode('.jpg', annotated_frame)
            if ret:
                with state.lock:
                    state.latest_frame = jpeg.tobytes()
                    state.latest_detections = detections
                    state.metrics = metrics
            
            # Broadcast PLC signals
            if len(detections) > 0:
                simulate_plc_broadcast(detections)
                
            # Broadcast telemetry update to Web UI
            broadcast_ws_message({
                "type": "telemetry",
                "payload": {
                    "fps": round(fps, 1),
                    "latency_ms": round(metrics['inference_ms'], 1),
                    "defect_count": len(detections),
                    "uptime_seconds": int(uptime),
                    "detections": detections,
                    "device": detector.device_name
                }
            })
            
            # Sleep if in directory mode to throttle to ~10 FPS
            if processor.mode == "directory":
                # Ensure we match ~10 FPS (100ms cycle)
                elapsed = time.time() - t_frame_start
                sleep_time = max(0.01, 0.1 - elapsed)
                time.sleep(sleep_time)
            else:
                # Add a tiny sleep to yield execution thread
                time.sleep(0.001)
                
    except Exception as e:
        print(f"Error in background streaming thread: {e}")
    finally:
        processor.release()

# Start thread on startup
@app.on_event("startup")
async def startup_event():
    # Save the running event loop in the state
    state.loop = asyncio.get_running_loop()
    thread = threading.Thread(target=stream_processing_thread, daemon=True)
    thread.start()

# REST Endpoints
@app.get("/metrics")
def get_metrics():
    """
    Exposes metrics for Prometheus scraping.
    """
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/api/status")
def get_status():
    """
    Returns current system health status.
    """
    uptime = time.time() - state.start_time if state.camera_online else 0.0
    return {
        "status": "ONLINE" if state.camera_online else "OFFLINE",
        "camera_online": state.camera_online,
        "uptime_seconds": int(uptime),
        "latest_metrics": state.metrics,
        "latest_defect_count": len(state.latest_detections),
        "device": detector.device_name
    }

@app.get("/api/detections")
def get_latest_detections():
    """
    Returns latest frame detections.
    """
    with state.lock:
        return {
            "defect_count": len(state.latest_detections),
            "detections": state.latest_detections,
            "metrics": state.metrics
        }

# Setup upload directory
UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

@app.post("/api/set_source")
def set_source(source: str = Form(...)):
    """
    Sets the active stream source: 'webcam', 'directory', or an absolute file path.
    """
    with state.lock:
        if source == "webcam":
            state.source_path = 0
            state.source_changed = True
        elif source == "directory":
            state.source_path = "data/images/val"
            state.source_changed = True
        else:
            if os.path.exists(source):
                state.source_path = source
                state.source_changed = True
            else:
                return {"status": "ERROR", "message": f"Source path {source} does not exist."}
                
    broadcast_ws_message({
        "type": "source_changed",
        "payload": {"source_path": str(state.source_path)}
    })
    return {"status": "SUCCESS", "source": str(state.source_path)}

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Uploads a file (video or image) and sets it as the active stream source.
    """
    if not file:
        return {"status": "ERROR", "message": "No file uploaded."}
        
    file_path = UPLOAD_DIR / file.filename
    try:
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
            
        with state.lock:
            state.source_path = str(file_path)
            state.source_changed = True
            
        print(f"File uploaded and set as source: {file_path}")
        
        broadcast_ws_message({
            "type": "source_changed",
            "payload": {"source_path": str(state.source_path)}
        })
        
        return {"status": "SUCCESS", "filename": file.filename, "source": str(file_path)}
    except Exception as e:
        return {"status": "ERROR", "message": str(e)}

def video_feed_generator():
    """
    MJPEG stream generator.
    """
    while True:
        with state.lock:
            frame_bytes = state.latest_frame
            
        if frame_bytes is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.04) # Limit generator loop to ~25 FPS

@app.get("/video_feed")
def get_video_feed():
    """
    Serves the real-time annotated camera stream.
    """
    return StreamingResponse(
        video_feed_generator(), 
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

# WebSocket connection
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    state.active_websockets.append(websocket)
    print(f"WebSocket client connected. Total clients: {len(state.active_websockets)}")
    
    try:
        # Keep connection open
        while True:
            # We can receive commands from UI if needed
            data = await websocket.receive_text()
            # Respond to ping
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        state.active_websockets.remove(websocket)
        print(f"WebSocket client disconnected. Total clients: {len(state.active_websockets)}")
    except Exception as e:
        print(f"WebSocket error: {e}")
        if websocket in state.active_websockets:
            state.active_websockets.remove(websocket)

# HTML templates
templates_dir = Path("app/templates")
templates_dir.mkdir(parents=True, exist_ok=True)

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    """
    Renders the monitoring dashboard.
    """
    # Fallback HTML render if template file fails, otherwise Jinja2
    template_path = templates_dir / "index.html"
    if template_path.exists():
        with open(template_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read(), status_code=200)
    return HTMLResponse(content="<h1>Dashboard HTML template not found.</h1>", status_code=404)
