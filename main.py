from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import base64
import numpy as np
import mediapipe as mp
import pickle
from typing import List, Optional, Dict
import tempfile
from gtts import gTTS
import time
import pygame
import asyncio
from PIL import Image
import os
import io
import json
from collections import deque
import hashlib

app = FastAPI()

# Initialize pygame mixer for audio
pygame.mixer.init()

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
        "http://localhost:5173",  # Vite dev server
        "https://*.vercel.app",   # Vercel domains
        "https://*.netlify.app",  # Netlify domains
        # Tambahkan domain frontend Anda nanti
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for state management
LAST_ENTER_TIME = 1
ENTER_COOLDOWN = 3.0
SPEECH_IN_PROGRESS = False
CAMERA_ENABLED = False

# Optimization variables
PREDICTION_CACHE = {}  # Cache for predictions
MOTION_BUFFER = deque(maxlen=5)  # Buffer for motion detection
LAST_PROCESSED_TIME = 0
MIN_PROCESSING_INTERVAL = 0.15  # Minimum 150ms between processing (6.7 FPS)
GESTURE_BUFFER = deque(maxlen=3)  # Buffer for gesture stability
LAST_FRAME_HASH = None

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

class OptimizedImageData(BaseModel):
    # Only send landmarks instead of full image when possible
    landmarks: Optional[List[List[float]]] = None  # Hand landmarks
    image: Optional[str] = None  # Fallback for full image
    sentence: List[str] = []
    last_char_time: float = 0
    char_delay: float = 1.5
    initial_delay: float = 2
    camera_state: Optional[bool] = None
    motion_detected: Optional[bool] = True  # Client-side motion detection
    frame_hash: Optional[str] = None  # Frame hash for duplicate detection

class ImageData(BaseModel):
    image: Optional[str] = None
    sentence: List[str] = []
    last_char_time: float = 0
    char_delay: float = 1.5
    initial_delay: float = 2
    camera_state: Optional[bool] = None

class PredictionResult(BaseModel):
    prediction: str
    sentence: List[str]
    last_char_time: float
    camera_state: bool
    processing_time: float = 0
    should_continue: bool = True  # Signal to client about processing frequency

class CameraControl(BaseModel):
    enable: bool

def calculate_frame_hash(frame_data: str) -> str:
    """Calculate hash of frame for duplicate detection"""
    return hashlib.md5(frame_data.encode()).hexdigest()[:8]

def detect_motion(current_landmarks: List[List[float]]) -> bool:
    """Detect significant motion from landmarks"""
    if not MOTION_BUFFER or not current_landmarks:
        return True
    
    # Compare with last landmarks in buffer
    last_landmarks = MOTION_BUFFER[-1]
    if not last_landmarks:
        return True
    
    # Calculate movement threshold
    total_movement = 0
    for i, hand in enumerate(current_landmarks):
        if i < len(last_landmarks):
            for j in range(0, len(hand), 2):  # x, y pairs
                if j + 1 < len(hand) and j + 1 < len(last_landmarks[i]):
                    dx = hand[j] - last_landmarks[i][j]
                    dy = hand[j + 1] - last_landmarks[i][j + 1]
                    total_movement += (dx * dx + dy * dy)
    
    return total_movement > 0.001  # Threshold for significant movement

def get_cached_prediction(landmarks_key: str) -> Optional[str]:
    """Get cached prediction for similar landmarks"""
    return PREDICTION_CACHE.get(landmarks_key)

def cache_prediction(landmarks_key: str, prediction: str):
    """Cache prediction with LRU-like behavior"""
    if len(PREDICTION_CACHE) > 100:  # Limit cache size
        # Remove oldest entry
        oldest_key = next(iter(PREDICTION_CACHE))
        del PREDICTION_CACHE[oldest_key]
    PREDICTION_CACHE[landmarks_key] = prediction

def create_landmarks_key(data_aux: List[float]) -> str:
    """Create a key for landmarks caching (quantized for better matching)"""
    # Quantize landmarks to reduce precision and improve cache hits
    quantized = [round(x, 2) for x in data_aux[:20]]  # Use first 20 features
    return str(hash(tuple(quantized)))

def is_gesture_stable(current_char: str) -> bool:
    """Check if gesture is stable across multiple frames"""
    GESTURE_BUFFER.append(current_char)
    if len(GESTURE_BUFFER) < 2:
        return False
    
    # Check if last 2-3 predictions are the same
    recent_gestures = list(GESTURE_BUFFER)[-2:]
    return len(set(recent_gestures)) == 1 and recent_gestures[0] != ""

@app.post("/camera")
async def control_camera(control: CameraControl):
    """Endpoint to enable/disable camera"""
    global CAMERA_ENABLED
    CAMERA_ENABLED = control.enable
    return {"success": True, "camera_state": CAMERA_ENABLED}

def decode_base64_image(base64_string):
    """Decode base64 image to numpy array without OpenCV"""
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
    """Extract landmarks from frame using MediaPipe"""
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
    """Detect open palm (5 fingers open)"""
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
    """Detect space gesture (fist with thumb up)"""
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

async def speak_and_reset(text: str):
    """Handle text-to-speech and reset the sentence"""
    global SPEECH_IN_PROGRESS
    SPEECH_IN_PROGRESS = True
    
    if not text.strip():
        return []
        
    try:
        tts = gTTS(text=text, lang='id')
        with tempfile.NamedTemporaryFile(delete=True, suffix=".mp3") as fp:
            fp_name = fp.name
            tts.save(fp_name)
            pygame.mixer.music.load(fp_name)
            pygame.mixer.music.play()
            
            while pygame.mixer.music.get_busy():
                await asyncio.sleep(0.1)
                
        return []
        
    except Exception as e:
        print(f"Error in text-to-speech: {e}")
        return []
    finally:
        SPEECH_IN_PROGRESS = False

@app.post("/predict")
async def predict(data: ImageData):
    global LAST_ENTER_TIME, SPEECH_IN_PROGRESS, CAMERA_ENABLED, LAST_PROCESSED_TIME, LAST_FRAME_HASH
    
    start_time = time.time()
    
    try:
        # Update camera state if provided
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
        
        # Rate limiting: Skip if too frequent
        current_time = time.time()
        if current_time - LAST_PROCESSED_TIME < MIN_PROCESSING_INTERVAL:
            return PredictionResult(
                prediction="",
                sentence=data.sentence,
                last_char_time=data.last_char_time,
                camera_state=CAMERA_ENABLED,
                processing_time=time.time() - start_time,
                should_continue=False  # Tell client to slow down
            )
        
        # Skip if no image provided when camera is enabled
        if not data.image:
            return PredictionResult(
                prediction="",
                sentence=data.sentence,
                last_char_time=data.last_char_time,
                camera_state=CAMERA_ENABLED,
                processing_time=time.time() - start_time
            )
        
        # Check for duplicate frames
        frame_hash = calculate_frame_hash(data.image)
        if LAST_FRAME_HASH == frame_hash:
            return PredictionResult(
                prediction="",
                sentence=data.sentence,
                last_char_time=data.last_char_time,
                camera_state=CAMERA_ENABLED,
                processing_time=time.time() - start_time,
                should_continue=False
            )
        LAST_FRAME_HASH = frame_hash
        
        # Update last processed time
        LAST_PROCESSED_TIME = current_time
        
        # Decode base64 image to NumPy array
        frame = decode_base64_image(data.image)
        
        # Extract landmarks
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
        
        # Update motion buffer
        MOTION_BUFFER.append(landmarks_data)
        
        current_char = ""
        new_sentence = data.sentence.copy()
        last_char_time = data.last_char_time

        if results.multi_hand_landmarks:
            hands_count = len(results.multi_hand_landmarks)
            data_aux = []
            x_ = []
            y_ = []

            for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                for lm in hand_landmarks.landmark:
                    x_.append(lm.x)
                    y_.append(lm.y)

                for lm in hand_landmarks.landmark:
                    data_aux.append(lm.x - min(x_))
                    data_aux.append(lm.y - min(y_))

                # Check gestures on first hand
                if i == 0 and results.multi_handedness:
                    label = results.multi_handedness[i].classification[0].label
                    
                    if detect_space_gesture(hand_landmarks, label):
                        current_char = "space"
                    elif count_fingers(hand_landmarks, label) == 5:
                        current_char = "enter"

            # If not a special gesture, predict letter with caching
            if current_char != "enter" and current_char != "space":
                if hands_count == 1:
                    data_aux += [0] * 42
                if len(data_aux) == 84:
                    # Try cache first
                    landmarks_key = create_landmarks_key(data_aux)
                    cached_prediction = get_cached_prediction(landmarks_key)
                    
                    if cached_prediction:
                        current_char = cached_prediction
                    else:
                        prediction = model.predict([np.asarray(data_aux)])
                        current_char = str(prediction[0]).lower()
                        cache_prediction(landmarks_key, current_char)

            # Only proceed if gesture is stable
            if not is_gesture_stable(current_char):
                return PredictionResult(
                    prediction="",
                    sentence=data.sentence,
                    last_char_time=data.last_char_time,
                    camera_state=CAMERA_ENABLED,
                    processing_time=time.time() - start_time
                )

            # Calculate time delay between characters
            time_since_last = current_time - last_char_time

            if time_since_last > data.initial_delay:
                if time_since_last > data.char_delay:
                    if current_char == "enter":
                        if current_time - LAST_ENTER_TIME > ENTER_COOLDOWN:
                            spoken_text = "".join(new_sentence).strip()
                            if spoken_text:
                                print(f"Mengucapkan: {spoken_text}")
                                new_sentence = await speak_and_reset(spoken_text)
                            LAST_ENTER_TIME = current_time
                            last_char_time = current_time
                    elif current_char == "space":
                        new_sentence.append(" ")
                        print("Spasi ditambahkan")
                        last_char_time = current_time
                    elif current_char:
                        new_sentence.append(current_char.upper())
                        last_char_time = current_time

        return PredictionResult(
            prediction=current_char.upper() if current_char else "",
            sentence=new_sentence,
            last_char_time=last_char_time,
            camera_state=CAMERA_ENABLED,
            processing_time=time.time() - start_time
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# WebSocket endpoint for real-time communication
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Receive data from client
            data = await websocket.receive_text()
            
            # Parse the received data
            try:
                request_data = json.loads(data)
                
                # Convert to ImageData model
                image_data = ImageData(**request_data)
                
                # Process the prediction
                result = await predict(image_data)
                
                # Send result back to client
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

@app.post("/predict")
async def predict(data: ImageData):
    global LAST_ENTER_TIME, SPEECH_IN_PROGRESS, CAMERA_ENABLED, LAST_PROCESSED_TIME, LAST_FRAME_HASH

    start_time = time.time()

    try:
        # 1. Perbarui status kamera
        if data.camera_state is not None:
            CAMERA_ENABLED = data.camera_state

        # 2. Jangan proses jika sedang berbicara atau kamera mati
        if SPEECH_IN_PROGRESS or not CAMERA_ENABLED:
            return PredictionResult(
                prediction="",
                sentence=data.sentence,
                last_char_time=data.last_char_time,
                camera_state=CAMERA_ENABLED,
                processing_time=time.time() - start_time,
                should_continue=True
            )

        # 3. Batasi frekuensi pemrosesan (Rate limiting)
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

        # 4. Pastikan ada gambar
        if not data.image:
            return PredictionResult(
                prediction="",
                sentence=data.sentence,
                last_char_time=data.last_char_time,
                camera_state=CAMERA_ENABLED,
                processing_time=time.time() - start_time,
                should_continue=False
            )

        # 5. Deteksi frame duplikat
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

        # 6. Decode gambar dan ekstrak landmark
        frame = decode_base64_image(data.image)
        landmarks_data, results = extract_landmarks_from_frame(frame)

        # 7. Deteksi gerakan
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

        # 8. Prediksi gesture atau huruf
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

                # Deteksi gestur khusus (enter/space)
                if i == 0 and results.multi_handedness:
                    label = results.multi_handedness[i].classification[0].label

                    if detect_space_gesture(hand_landmarks, label):
                        current_char = "space"
                    elif count_fingers(hand_landmarks, label) == 5:
                        current_char = "enter"

            # Prediksi huruf dari landmark (jika bukan enter/space)
            if current_char not in ["enter", "space"]:
                if hands_count == 1:
                    data_aux += [0] * 42  # Padding tangan kedua
                if len(data_aux) == 84:
                    landmarks_key = create_landmarks_key(data_aux)
                    cached_prediction = get_cached_prediction(landmarks_key)
                    if cached_prediction:
                        current_char = cached_prediction
                    else:
                        prediction = model.predict([data_aux])[0]
                        cache_prediction(landmarks_key, prediction)
                        current_char = prediction

        # 9. Update kalimat atau lakukan aksi
        if current_char == "space":
            if is_gesture_stable(current_char) and (current_time - last_char_time) > data.char_delay:
                new_sentence.append(" ")
                last_char_time = current_time

        elif current_char == "enter":
            if is_gesture_stable(current_char) and (current_time - LAST_ENTER_TIME) > ENTER_COOLDOWN:
                LAST_ENTER_TIME = current_time
                new_sentence = await speak_and_reset("".join(new_sentence))
                last_char_time = 0

        elif current_char and current_char != "enter" and current_char != "space":
            if is_gesture_stable(current_char) and (current_time - last_char_time) > data.char_delay:
                new_sentence.append(current_char)
                last_char_time = current_time

        return PredictionResult(
            prediction=current_char,
            sentence=new_sentence,
            last_char_time=last_char_time,
            camera_state=CAMERA_ENABLED,
            processing_time=time.time() - start_time,
            should_continue=True
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in prediction: {str(e)}")

async def process_landmarks_directly(landmarks_data: List[List[float]]):
    """Process landmarks directly without MediaPipe processing"""
    # This would need implementation based on your model's input format
    # For now, return empty to maintain compatibility
    return ""

async def process_mediapipe_results(results):
    """Process MediaPipe results (extracted from original predict function)"""
    # Implementation would be similar to the original processing logic
    return ""

@app.post("/clear")
async def clear_sentence():
    """Endpoint to clear the sentence"""
    global LAST_ENTER_TIME
    LAST_ENTER_TIME = 0
    return {"sentence": [], "last_char_time": 0, "camera_state": CAMERA_ENABLED}

@app.get("/stats")
async def get_stats():
    """Get performance statistics"""
    return {
        "cache_size": len(PREDICTION_CACHE),
        "motion_buffer_size": len(MOTION_BUFFER),
        "gesture_buffer_size": len(GESTURE_BUFFER),
        "last_processed_time": LAST_PROCESSED_TIME,
        "min_processing_interval": MIN_PROCESSING_INTERVAL
    }

# Cerebrium requires this handler
def handler(data: dict, context: dict):
    return {"message": "Use specific endpoints (/predict, /predict-optimized, /ws, /camera, /clear)"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=port,
        log_level="info"
    )