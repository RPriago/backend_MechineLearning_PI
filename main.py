from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel
import base64
import numpy as np
import mediapipe as mp
import pickle
from typing import List, Optional, Dict
import tempfile
from gtts import gTTS
import time
import asyncio
from PIL import Image
import os
import io
import json
from collections import deque
import hashlib

app = FastAPI()

# Load the model
try:
    model_path = "model.p"
    model_dict = pickle.load(open(model_path, 'rb'))
    model = model_dict['model']
except Exception as e:
    raise RuntimeError(f"Failed to load model: {e}")

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.3
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "https://*.vercel.app",
        "https://*.netlify.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables - OPTIMIZED
LAST_ENTER_TIME = 1
ENTER_COOLDOWN = 3.0
SPEECH_IN_PROGRESS = False
CAMERA_ENABLED = False
PREDICTION_CACHE = {}
MOTION_BUFFER = deque(maxlen=5)
LAST_PROCESSED_TIME = 0
# REDUCED processing interval untuk mengurangi beban
MIN_PROCESSING_INTERVAL = 0.3  # Increased from 0.15 to 0.3 seconds
GESTURE_BUFFER = deque(maxlen=3)
LAST_FRAME_HASH = None
ACTIVE_AUDIO_FILES = set()

# NEW: Adaptive processing variables
CONSECUTIVE_EMPTY_FRAMES = 0
MAX_EMPTY_FRAMES = 10  # Skip processing after 10 consecutive empty frames
LAST_GESTURE_TIME = 0
GESTURE_TIMEOUT = 2.0  # If no gesture for 2s, reduce processing frequency
ADAPTIVE_INTERVAL = MIN_PROCESSING_INTERVAL

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.client_states: Dict[WebSocket, Dict] = {}

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        self.client_states[websocket] = {
            "sentence": [],
            "last_char_time": 0,
            "last_frame_time": 0,
            "processing_active": True
        }

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        if websocket in self.client_states:
            del self.client_states[websocket]

    async def send_personal_message(self, message: dict, websocket: WebSocket):
        try:
            await websocket.send_text(json.dumps(message))
        except:
            # Connection might be closed
            self.disconnect(websocket)

manager = ConnectionManager()

class PredictionResult(BaseModel):
    prediction: str
    sentence: List[str]
    last_char_time: float
    camera_state: bool
    processing_time: float = 0
    should_continue: bool = True
    should_speak: bool = False
    audio_text: str = ""
    audio_url: Optional[str] = None
    # NEW: Adaptive processing info
    processing_interval: float = MIN_PROCESSING_INTERVAL
    frames_skipped: int = 0

class ImageData(BaseModel):
    image: Optional[str] = None
    sentence: List[str] = []
    last_char_time: float = 0
    char_delay: float = 1.5
    initial_delay: float = 2
    camera_state: Optional[bool] = None
    # NEW: Client-side timing info
    client_timestamp: Optional[float] = None
    frame_count: Optional[int] = None

# Helper functions - OPTIMIZED
def calculate_frame_hash(frame_data: str) -> str:
    # Use shorter hash for better performance
    return hashlib.md5(frame_data[:1000].encode()).hexdigest()[:6]

def detect_motion(current_landmarks: List[List[float]]) -> bool:
    if not MOTION_BUFFER or not current_landmarks:
        return True
    
    last_landmarks = MOTION_BUFFER[-1]
    if not last_landmarks:
        return True
    
    # OPTIMIZED: Check only key landmarks for motion
    total_movement = 0
    sample_points = [0, 4, 8, 12, 16, 20]  # Only check finger tips and thumb
    
    for i, hand in enumerate(current_landmarks):
        if i < len(last_landmarks):
            for j in sample_points:
                if j*2+1 < len(hand) and j*2+1 < len(last_landmarks[i]):
                    dx = hand[j*2] - last_landmarks[i][j*2]
                    dy = hand[j*2+1] - last_landmarks[i][j*2+1]
                    total_movement += (dx * dx + dy * dy)
    
    return total_movement > 0.002  # Slightly higher threshold

def get_cached_prediction(landmarks_key: str) -> Optional[str]:
    return PREDICTION_CACHE.get(landmarks_key)

def cache_prediction(landmarks_key: str, prediction: str):
    # OPTIMIZED: Limit cache size more aggressively
    if len(PREDICTION_CACHE) > 50:  # Reduced from 100
        # Remove oldest entries
        keys_to_remove = list(PREDICTION_CACHE.keys())[:10]
        for key in keys_to_remove:
            del PREDICTION_CACHE[key]
    PREDICTION_CACHE[landmarks_key] = prediction

def create_landmarks_key(data_aux: List[float]) -> str:
    # OPTIMIZED: Use fewer points for key generation
    quantized = [round(x, 1) for x in data_aux[:12]]  # Reduced precision and points
    return str(hash(tuple(quantized)))

def is_gesture_stable(current_char: str) -> bool:
    GESTURE_BUFFER.append(current_char)
    if len(GESTURE_BUFFER) < 2:
        return False
    
    recent_gestures = list(GESTURE_BUFFER)[-2:]
    return len(set(recent_gestures)) == 1 and recent_gestures[0] != ""

def should_process_frame(current_time: float, client_timestamp: Optional[float] = None) -> tuple[bool, float]:
    """
    Determine if frame should be processed based on adaptive intervals
    Returns: (should_process, next_interval)
    """
    global ADAPTIVE_INTERVAL, LAST_GESTURE_TIME, CONSECUTIVE_EMPTY_FRAMES
    
    # Check basic timing
    if current_time - LAST_PROCESSED_TIME < ADAPTIVE_INTERVAL:
        return False, ADAPTIVE_INTERVAL
    
    # Adaptive processing based on activity
    time_since_gesture = current_time - LAST_GESTURE_TIME
    
    if time_since_gesture > GESTURE_TIMEOUT:
        # No recent gestures, reduce processing frequency
        ADAPTIVE_INTERVAL = min(0.5, MIN_PROCESSING_INTERVAL * 2)
    else:
        # Recent gesture activity, maintain normal frequency
        ADAPTIVE_INTERVAL = MIN_PROCESSING_INTERVAL
    
    return True, ADAPTIVE_INTERVAL

def decode_base64_image(base64_string):
    try:
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        
        image_data = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(image_data))
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        frame = np.array(image)
        frame = np.fliplr(frame)
        
        return frame
        
    except Exception as e:
        raise ValueError(f"Failed to decode image: {e}")

def extract_landmarks_from_frame(frame):
    results = hands.process(frame)
    landmarks_data = []
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            hand_data = []
            for lm in hand_landmarks.landmark:
                hand_data.extend([lm.x, lm.y])
            landmarks_data.append(hand_data)
    
    return landmarks_data, results

def count_fingers(hand_landmarks, hand_label):
    tips_ids = [4, 8, 12, 16, 20]
    fingers = []

    if hand_label == "Right":
        fingers.append(hand_landmarks.landmark[tips_ids[0]].x < hand_landmarks.landmark[tips_ids[0] - 1].x)
    else:
        fingers.append(hand_landmarks.landmark[tips_ids[0]].x > hand_landmarks.landmark[tips_ids[0] - 1].x)

    for i in range(1, 5):
        fingers.append(hand_landmarks.landmark[tips_ids[i]].y < hand_landmarks.landmark[tips_ids[i] - 2].y)

    return sum(fingers)

def detect_space_gesture(hand_landmarks, hand_label):
    tips_ids = [4, 8, 12, 16, 20]
    
    thumb_extended = False
    if hand_label == "Right":
        thumb_extended = (hand_landmarks.landmark[tips_ids[0]].y < hand_landmarks.landmark[2].y)
    else:
        thumb_extended = (hand_landmarks.landmark[tips_ids[0]].y < hand_landmarks.landmark[2].y)
    
    other_fingers_folded = True
    for i in range(1, 5):
        if hand_landmarks.landmark[tips_ids[i]].y < hand_landmarks.landmark[tips_ids[i] - 2].y:
            other_fingers_folded = False
            break
    
    return thumb_extended and other_fingers_folded

# Audio functions (unchanged)
async def generate_speech_response(text: str) -> Optional[str]:
    global SPEECH_IN_PROGRESS
    
    if not text.strip():
        return None
        
    SPEECH_IN_PROGRESS = True
    
    try:
        tts = gTTS(text=text, lang='id')
        timestamp = int(time.time() * 1000)
        filename = f"speech_{timestamp}.mp3"
        filepath = f"temp_audio/{filename}"
        os.makedirs("temp_audio", exist_ok=True)
        tts.save(filepath)
        ACTIVE_AUDIO_FILES.add(filename)
        audio_url = f"/audio/{filename}"
        return audio_url
        
    except Exception as e:
        print(f"Error generating speech: {str(e)}")
        return None
    finally:
        SPEECH_IN_PROGRESS = False

def cleanup_audio_file(filename: str):
    try:
        filepath = f"temp_audio/{filename}"
        if os.path.exists(filepath):
            os.remove(filepath)
        ACTIVE_AUDIO_FILES.discard(filename)
    except Exception as e:
        print(f"Error cleaning up audio file {filename}: {e}")

def cleanup_all_audio_files():
    try:
        if os.path.exists("temp_audio"):
            for filename in os.listdir("temp_audio"):
                if filename.startswith("speech_"):
                    filepath = os.path.join("temp_audio", filename)
                    if os.path.exists(filepath):
                        os.remove(filepath)
        ACTIVE_AUDIO_FILES.clear()
    except Exception as e:
        print(f"Error cleaning up all audio files: {e}")

@app.get("/audio/{filename}")
async def serve_audio(filename: str):
    filepath = f"temp_audio/{filename}"
    
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="Audio file not found")
    
    try:
        with open(filepath, "rb") as audio_file:
            audio_data = audio_file.read()
        
        asyncio.create_task(delayed_cleanup(filename))
        
        return Response(
            content=audio_data,
            media_type="audio/mpeg",
            headers={
                "Content-Disposition": f"attachment; filename={filename}",
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0"
            }
        )
    except Exception as e:
        cleanup_audio_file(filename)
        raise HTTPException(status_code=500, detail=f"Error serving audio: {e}")

async def delayed_cleanup(filename: str):
    await asyncio.sleep(10)
    cleanup_audio_file(filename)

# OPTIMIZED: Simplified predict function for better performance
async def process_gesture_prediction(data: ImageData) -> PredictionResult:
    global LAST_ENTER_TIME, SPEECH_IN_PROGRESS, CAMERA_ENABLED, LAST_PROCESSED_TIME
    global LAST_FRAME_HASH, CONSECUTIVE_EMPTY_FRAMES, LAST_GESTURE_TIME
    
    start_time = time.time()
    frames_skipped = 0

    try:
        # Update camera state
        if data.camera_state is not None:
            CAMERA_ENABLED = data.camera_state

        # Quick exits for non-processing states
        if SPEECH_IN_PROGRESS or not CAMERA_ENABLED:
            return PredictionResult(
                prediction="",
                sentence=data.sentence,
                last_char_time=data.last_char_time,
                camera_state=CAMERA_ENABLED,
                processing_time=time.time() - start_time,
                should_continue=True,
                processing_interval=ADAPTIVE_INTERVAL
            )

        # Adaptive processing check
        current_time = time.time()
        should_process, next_interval = should_process_frame(current_time, data.client_timestamp)
        
        if not should_process:
            frames_skipped = 1
            return PredictionResult(
                prediction="",
                sentence=data.sentence,
                last_char_time=data.last_char_time,
                camera_state=CAMERA_ENABLED,
                processing_time=time.time() - start_time,
                should_continue=False,
                processing_interval=next_interval,
                frames_skipped=frames_skipped
            )

        if not data.image:
            CONSECUTIVE_EMPTY_FRAMES += 1
            return PredictionResult(
                prediction="",
                sentence=data.sentence,
                last_char_time=data.last_char_time,
                camera_state=CAMERA_ENABLED,
                processing_time=time.time() - start_time,
                should_continue=False,
                processing_interval=next_interval
            )

        # Reset empty frame counter
        CONSECUTIVE_EMPTY_FRAMES = 0

        # Duplicate frame detection (optimized)
        frame_hash = calculate_frame_hash(data.image)
        if frame_hash == LAST_FRAME_HASH:
            return PredictionResult(
                prediction="",
                sentence=data.sentence,
                last_char_time=data.last_char_time,
                camera_state=CAMERA_ENABLED,
                processing_time=time.time() - start_time,
                should_continue=False,
                processing_interval=next_interval
            )
        
        LAST_FRAME_HASH = frame_hash
        LAST_PROCESSED_TIME = current_time

        # Process frame
        frame = decode_base64_image(data.image)
        landmarks_data, results = extract_landmarks_from_frame(frame)

        # Motion detection (optimized)
        if not detect_motion(landmarks_data):
            MOTION_BUFFER.append(landmarks_data)
            return PredictionResult(
                prediction="",
                sentence=data.sentence,
                last_char_time=data.last_char_time,
                camera_state=CAMERA_ENABLED,
                processing_time=time.time() - start_time,
                should_continue=False,
                processing_interval=next_interval
            )
        
        MOTION_BUFFER.append(landmarks_data)

        # Process gestures
        current_char = ""
        new_sentence = data.sentence.copy()
        last_char_time = data.last_char_time
        should_speak = False
        audio_text = ""
        audio_url = None

        if results.multi_hand_landmarks:
            LAST_GESTURE_TIME = current_time  # Update gesture activity
            hands_count = len(results.multi_hand_landmarks)
            data_aux = []
            x_, y_ = [], []

            for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                for lm in hand_landmarks.landmark:
                    x_.append(lm.x)
                    y_.append(lm.y)

                for lm in hand_landmarks.landmark:
                    data_aux.append(lm.x - min(x_))
                    data_aux.append(lm.y - min(y_))

                # Detect special gestures
                if i == 0 and results.multi_handedness:
                    label = results.multi_handedness[i].classification[0].label

                    if detect_space_gesture(hand_landmarks, label):
                        current_char = "space"
                    elif count_fingers(hand_landmarks, label) == 5:
                        current_char = "enter"

            # Predict letters (optimized)
            if current_char not in ["enter", "space"]:
                if hands_count == 1:
                    data_aux += [0] * 42
                if len(data_aux) == 84:
                    landmarks_key = create_landmarks_key(data_aux)
                    cached_prediction = get_cached_prediction(landmarks_key)
                    if cached_prediction:
                        current_char = cached_prediction
                    else:
                        prediction = model.predict([data_aux])[0]
                        cache_prediction(landmarks_key, prediction)
                        current_char = prediction

        # Process actions (unchanged logic)
        if current_char == "space":
            if is_gesture_stable(current_char) and (current_time - last_char_time) > data.char_delay:
                new_sentence.append(" ")
                last_char_time = current_time

        elif current_char == "enter":
            if is_gesture_stable(current_char) and (current_time - LAST_ENTER_TIME) > ENTER_COOLDOWN:
                LAST_ENTER_TIME = current_time
                spoken_text = "".join(new_sentence).strip()
                if spoken_text:
                    audio_url = await generate_speech_response(spoken_text)
                    should_speak = True
                    audio_text = spoken_text
                new_sentence = []
                last_char_time = 0

        elif current_char and current_char not in ["enter", "space"]:
            if is_gesture_stable(current_char) and (current_time - last_char_time) > data.char_delay:
                new_sentence.append(current_char)
                last_char_time = current_time

        return PredictionResult(
            prediction=current_char,
            sentence=new_sentence,
            last_char_time=last_char_time,
            camera_state=CAMERA_ENABLED,
            processing_time=time.time() - start_time,
            should_continue=True,
            should_speak=should_speak,
            audio_text=audio_text,
            audio_url=audio_url,
            processing_interval=next_interval,
            frames_skipped=frames_skipped
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in prediction: {str(e)}")

# OPTIMIZED: Keep POST endpoint but discourage frequent use
@app.post("/predict")
async def predict(data: ImageData):
    """
    Legacy POST endpoint - Use WebSocket for better performance
    """
    return await process_gesture_prediction(data)

# MAIN OPTIMIZATION: Enhanced WebSocket with intelligent processing
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    
    try:
        while True:
            data = await websocket.receive_text()
            
            try:
                request_data = json.loads(data)
                
                # Handle different message types
                if request_data.get("type") == "heartbeat":
                    await manager.send_personal_message(
                        {"type": "heartbeat_response", "timestamp": time.time()}, 
                        websocket
                    )
                    continue
                
                if request_data.get("type") == "config":
                    # Handle configuration updates
                    global MIN_PROCESSING_INTERVAL, GESTURE_TIMEOUT
                    config = request_data.get("config", {})
                    if "processing_interval" in config:
                        MIN_PROCESSING_INTERVAL = max(0.1, config["processing_interval"])
                    if "gesture_timeout" in config:
                        GESTURE_TIMEOUT = max(1.0, config["gesture_timeout"])
                    
                    await manager.send_personal_message(
                        {"type": "config_updated", "config": config}, 
                        websocket
                    )
                    continue
                
                # Process gesture prediction
                image_data = ImageData(**request_data)
                result = await process_gesture_prediction(image_data)
                
                # Update client state
                if websocket in manager.client_states:
                    manager.client_states[websocket].update({
                        "sentence": result.sentence,
                        "last_char_time": result.last_char_time,
                        "last_frame_time": time.time()
                    })
                
                await manager.send_personal_message(result.dict(), websocket)
                
            except json.JSONDecodeError:
                await manager.send_personal_message(
                    {"error": "Invalid JSON format"}, websocket
                )
            except Exception as e:
                await manager.send_personal_message(
                    {"error": str(e)}, websocket
                )
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# REST endpoints (unchanged)
@app.post("/camera")
async def control_camera(control: dict):
    global CAMERA_ENABLED, CONSECUTIVE_EMPTY_FRAMES, LAST_GESTURE_TIME
    CAMERA_ENABLED = control.get("enable", False)
    
    if not CAMERA_ENABLED:
        cleanup_all_audio_files()
        # Reset counters
        CONSECUTIVE_EMPTY_FRAMES = 0
        LAST_GESTURE_TIME = 0
    
    return {"success": True, "camera_state": CAMERA_ENABLED}

@app.post("/clear")
async def clear_sentence():
    global LAST_ENTER_TIME
    LAST_ENTER_TIME = 0
    cleanup_all_audio_files()
    return {"sentence": [], "last_char_time": 0, "camera_state": CAMERA_ENABLED}

@app.get("/stats")
async def get_stats():
    return {
        "cache_size": len(PREDICTION_CACHE),
        "motion_buffer_size": len(MOTION_BUFFER),
        "gesture_buffer_size": len(GESTURE_BUFFER),
        "last_processed_time": LAST_PROCESSED_TIME,
        "min_processing_interval": MIN_PROCESSING_INTERVAL,
        "adaptive_interval": ADAPTIVE_INTERVAL,
        "consecutive_empty_frames": CONSECUTIVE_EMPTY_FRAMES,
        "last_gesture_time": LAST_GESTURE_TIME,
        "active_connections": len(manager.active_connections),
        "active_audio_files": len(ACTIVE_AUDIO_FILES),
        "performance": {
            "cache_hit_ratio": len(PREDICTION_CACHE) / max(1, len(PREDICTION_CACHE) + 10),
            "motion_detection_efficiency": len(MOTION_BUFFER) / 5,
            "processing_load": "adaptive" if ADAPTIVE_INTERVAL > MIN_PROCESSING_INTERVAL else "normal"
        }
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "camera_enabled": CAMERA_ENABLED,
        "speech_in_progress": SPEECH_IN_PROGRESS,
        "active_connections": len(manager.active_connections),
        "processing_interval": ADAPTIVE_INTERVAL
    }

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    cleanup_all_audio_files()
    print("Application started - cleaned up existing audio files")
    print(f"Initial processing interval: {MIN_PROCESSING_INTERVAL}s")

@app.on_event("shutdown")
async def shutdown_event():
    cleanup_all_audio_files()
    print("Application shutting down - cleaned up all audio files")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=port,
        log_level="info"
    )