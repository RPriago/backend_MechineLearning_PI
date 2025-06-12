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

# Global variables
LAST_ENTER_TIME = 1
ENTER_COOLDOWN = 3.0
SPEECH_IN_PROGRESS = False
CAMERA_ENABLED = False
PREDICTION_CACHE = {}
MOTION_BUFFER = deque(maxlen=5)
LAST_PROCESSED_TIME = 0
MIN_PROCESSING_INTERVAL = 0.15
GESTURE_BUFFER = deque(maxlen=3)
LAST_FRAME_HASH = None

# Track active audio files for immediate cleanup
ACTIVE_AUDIO_FILES = set()

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: dict, websocket: WebSocket):
        await websocket.send_text(json.dumps(message))

manager = ConnectionManager()

class PredictionResult(BaseModel):
    prediction: str
    sentence: List[str]
    last_char_time: float
    camera_state: bool
    processing_time: float = 0
    should_continue: bool = True
    # Tambahan untuk audio
    should_speak: bool = False
    audio_text: str = ""
    audio_url: Optional[str] = None

class ImageData(BaseModel):
    image: Optional[str] = None
    sentence: List[str] = []
    last_char_time: float = 0
    char_delay: float = 1.5
    initial_delay: float = 2
    camera_state: Optional[bool] = None

# Fungsi helper yang sudah ada (copy dari code asli)
def calculate_frame_hash(frame_data: str) -> str:
    return hashlib.md5(frame_data.encode()).hexdigest()[:8]

def detect_motion(current_landmarks: List[List[float]]) -> bool:
    if not MOTION_BUFFER or not current_landmarks:
        return True
    
    last_landmarks = MOTION_BUFFER[-1]
    if not last_landmarks:
        return True
    
    total_movement = 0
    for i, hand in enumerate(current_landmarks):
        if i < len(last_landmarks):
            for j in range(0, len(hand), 2):
                if j + 1 < len(hand) and j + 1 < len(last_landmarks[i]):
                    dx = hand[j] - last_landmarks[i][j]
                    dy = hand[j + 1] - last_landmarks[i][j + 1]
                    total_movement += (dx * dx + dy * dy)
    
    return total_movement > 0.001

def get_cached_prediction(landmarks_key: str) -> Optional[str]:
    return PREDICTION_CACHE.get(landmarks_key)

def cache_prediction(landmarks_key: str, prediction: str):
    if len(PREDICTION_CACHE) > 100:
        oldest_key = next(iter(PREDICTION_CACHE))
        del PREDICTION_CACHE[oldest_key]
    PREDICTION_CACHE[landmarks_key] = prediction

def create_landmarks_key(data_aux: List[float]) -> str:
    quantized = [round(x, 2) for x in data_aux[:20]]
    return str(hash(tuple(quantized)))

def is_gesture_stable(current_char: str) -> bool:
    GESTURE_BUFFER.append(current_char)
    if len(GESTURE_BUFFER) < 2:
        return False
    
    recent_gestures = list(GESTURE_BUFFER)[-2:]
    return len(set(recent_gestures)) == 1 and recent_gestures[0] != ""

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

# SOLUSI BARU: Generate audio langsung ke memory dan cleanup otomatis
async def generate_speech_response(text: str) -> Optional[str]:
    """Generate audio file dan return URL untuk frontend"""
    global SPEECH_IN_PROGRESS
    
    if not text.strip():
        return None
        
    SPEECH_IN_PROGRESS = True
    
    try:
        # Generate TTS
        tts = gTTS(text=text, lang='id')
        
        # Simpan ke file dengan nama unik
        timestamp = int(time.time() * 1000)
        filename = f"speech_{timestamp}.mp3"
        filepath = f"temp_audio/{filename}"
        
        # Pastikan direktori ada
        os.makedirs("temp_audio", exist_ok=True)
        
        # Simpan file audio
        tts.save(filepath)
        
        # Track file untuk cleanup
        ACTIVE_AUDIO_FILES.add(filename)
        
        # Return URL yang bisa diakses frontend
        audio_url = f"/audio/{filename}"
        
        return audio_url
        
    except Exception as e:
        print(f"Error generating speech: {str(e)}")
        return None
    finally:
        SPEECH_IN_PROGRESS = False

def cleanup_audio_file(filename: str):
    """Hapus file audio tertentu"""
    try:
        filepath = f"temp_audio/{filename}"
        if os.path.exists(filepath):
            os.remove(filepath)
            print(f"Cleaned up audio file: {filename}")
        
        # Remove from tracking set
        ACTIVE_AUDIO_FILES.discard(filename)
        
    except Exception as e:
        print(f"Error cleaning up audio file {filename}: {e}")

def cleanup_all_audio_files():
    """Hapus semua file audio yang ada"""
    try:
        if os.path.exists("temp_audio"):
            for filename in os.listdir("temp_audio"):
                if filename.startswith("speech_"):
                    filepath = os.path.join("temp_audio", filename)
                    if os.path.exists(filepath):
                        os.remove(filepath)
            print("Cleaned up all audio files")
        ACTIVE_AUDIO_FILES.clear()
    except Exception as e:
        print(f"Error cleaning up all audio files: {e}")

# Endpoint untuk serve audio files dengan auto-cleanup
@app.get("/audio/{filename}")
async def serve_audio(filename: str):
    """Serve audio files ke frontend dan schedule cleanup"""
    filepath = f"temp_audio/{filename}"
    
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="Audio file not found")
    
    try:
        with open(filepath, "rb") as audio_file:
            audio_data = audio_file.read()
        
        # Schedule cleanup setelah file dikirim
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
        # Cleanup jika terjadi error
        cleanup_audio_file(filename)
        raise HTTPException(status_code=500, detail=f"Error serving audio: {e}")

async def delayed_cleanup(filename: str):
    """Cleanup file audio setelah delay singkat (untuk memastikan sudah di-download)"""
    await asyncio.sleep(10)  # Delay 10 detik untuk memastikan audio sudah diputar
    cleanup_audio_file(filename)

@app.post("/predict")
async def predict(data: ImageData):
    global LAST_ENTER_TIME, SPEECH_IN_PROGRESS, CAMERA_ENABLED, LAST_PROCESSED_TIME, LAST_FRAME_HASH

    start_time = time.time()

    try:
        # Update camera state
        if data.camera_state is not None:
            CAMERA_ENABLED = data.camera_state

        # Skip processing if speech is in progress or camera is disabled
        if SPEECH_IN_PROGRESS or not CAMERA_ENABLED:
            return PredictionResult(
                prediction="",
                sentence=data.sentence,
                last_char_time=data.last_char_time,
                camera_state=CAMERA_ENABLED,
                processing_time=time.time() - start_time,
                should_continue=True
            )

        # Rate limiting
        current_time = time.time()
        if current_time - LAST_PROCESSED_TIME < MIN_PROCESSING_INTERVAL:
            return PredictionResult(
                prediction="",
                sentence=data.sentence,
                last_char_time=data.last_char_time,
                camera_state=CAMERA_ENABLED,
                processing_time=time.time() - start_time,
                should_continue=False
            )

        # Check for image
        if not data.image:
            return PredictionResult(
                prediction="",
                sentence=data.sentence,
                last_char_time=data.last_char_time,
                camera_state=CAMERA_ENABLED,
                processing_time=time.time() - start_time,
                should_continue=False
            )

        # Duplicate frame detection
        frame_hash = calculate_frame_hash(data.image)
        if frame_hash == LAST_FRAME_HASH:
            return PredictionResult(
                prediction="",
                sentence=data.sentence,
                last_char_time=data.last_char_time,
                camera_state=CAMERA_ENABLED,
                processing_time=time.time() - start_time,
                should_continue=False
            )
        LAST_FRAME_HASH = frame_hash
        LAST_PROCESSED_TIME = current_time

        # Process frame
        frame = decode_base64_image(data.image)
        landmarks_data, results = extract_landmarks_from_frame(frame)

        # Motion detection
        if not detect_motion(landmarks_data):
            MOTION_BUFFER.append(landmarks_data)
            return PredictionResult(
                prediction="",
                sentence=data.sentence,
                last_char_time=data.last_char_time,
                camera_state=CAMERA_ENABLED,
                processing_time=time.time() - start_time,
                should_continue=False
            )
        MOTION_BUFFER.append(landmarks_data)

        current_char = ""
        new_sentence = data.sentence.copy()
        last_char_time = data.last_char_time
        should_speak = False
        audio_text = ""
        audio_url = None

        # Process gestures
        if results.multi_hand_landmarks:
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

            # Predict letters
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

        # Process actions
        if current_char == "space":
            if is_gesture_stable(current_char) and (current_time - last_char_time) > data.char_delay:
                new_sentence.append(" ")
                last_char_time = current_time

        elif current_char == "enter":
            if is_gesture_stable(current_char) and (current_time - LAST_ENTER_TIME) > ENTER_COOLDOWN:
                LAST_ENTER_TIME = current_time
                spoken_text = "".join(new_sentence).strip()
                if spoken_text:
                    # Generate audio URL
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
            audio_url=audio_url
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in prediction: {str(e)}")

# WebSocket endpoint
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            
            try:
                request_data = json.loads(data)
                image_data = ImageData(**request_data)
                result = await predict(image_data)
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

@app.post("/camera")
async def control_camera(control: dict):
    global CAMERA_ENABLED
    CAMERA_ENABLED = control.get("enable", False)
    
    # Cleanup semua audio files ketika kamera dimatikan
    if not CAMERA_ENABLED:
        cleanup_all_audio_files()
    
    return {"success": True, "camera_state": CAMERA_ENABLED}

@app.post("/clear")
async def clear_sentence():
    global LAST_ENTER_TIME
    LAST_ENTER_TIME = 0
    
    # Cleanup audio files ketika sentence di-clear
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
        "active_audio_files": len(ACTIVE_AUDIO_FILES),
        "audio_files_list": list(ACTIVE_AUDIO_FILES)
    }

# Cleanup semua audio files saat startup
@app.on_event("startup")
async def startup_event():
    cleanup_all_audio_files()
    print("Application started - cleaned up existing audio files")

# Cleanup semua audio files saat shutdown
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