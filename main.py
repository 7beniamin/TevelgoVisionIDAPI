from fastapi import FastAPI, UploadFile, File, HTTPException
from datetime import UTC
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from datetime import datetime
from PIL import Image
import io
from slowapi import Limiter
from slowapi.util import get_remote_address
from fastapi import Request
from io import BytesIO

# Initialize request rate limiter
limiter = Limiter(key_func=get_remote_address)

# Import local modules
from face_recognition import get_face_embedding
from car_license_make_model import extract_car_info
from yolo_detect import crop_and_compress_car

# Import Azure Table Storage utility functions
from utils.azure_table import (
    get_all_authorized_pedestrians,
    get_all_authorized_vehicles,
    log_pedestrian_verification_attempt,
    log_vehicle_verification_attempt,
    fetch_pedestrian_logs,
    fetch_vehicle_logs,
)

# Initialize FastAPI app and add CORS middleware
app = FastAPI()
app.state.limiter = limiter
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Compute cosine similarity between two vectors
def cosine_similarity(vec1, vec2):
    v1 = np.array(vec1)
    v2 = np.array(vec2)
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

# Matching threshold for face recognition
MATCH_THRESHOLD = 0.4

# Convert image bytes to NumPy array
def bytes_to_ndarray(image_bytes: bytes) -> np.ndarray:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return np.array(image)

# Pedestrian verification endpoint (rate limited)
@app.post("/detect_faces/")
@limiter.limit("5/minute")
async def verify_pedestrian(request: Request, image: UploadFile = File(...)):
    image_bytes = await image.read()
    embedding = get_face_embedding(image_bytes)
    face_detected = embedding is not None
    face_match = False
    timestamp = datetime.now(UTC).isoformat()

    # Compare against authorized pedestrian embeddings
    if face_detected:
        authorized_faces = get_all_authorized_pedestrians()
        for person in authorized_faces:
            stored_embedding = person["face_embedding"]
            sim = cosine_similarity(embedding, stored_embedding)
            if sim >= MATCH_THRESHOLD:
                face_match = True
                break

    # Set verification status
    if not face_detected:
        status = "Face not found"
    elif face_match:
        status = "Authorized"
    else:
        status = "Unauthorized"

    # Log pedestrian attempt
    log_pedestrian_verification_attempt({
        "Type": "pedestrian",
        "Status": status,
        "FaceMatch": str(face_match),
        "EventTimestamp": timestamp
    })

    return {
        "status": "Authorized Person" if face_match else status,
        "face_detected": face_detected,
        "timestamp": timestamp
    }

# Vehicle verification endpoint
@app.post("/verify-vehicle/")
async def verify_vehicle(image: UploadFile = File(...)):
    image_bytes = await image.read()
    image_bytes_for_face = BytesIO(image_bytes).getvalue()
    image_bytes_for_car = BytesIO(image_bytes).getvalue()

    # Detect face and extract embedding
    face_embedding = get_face_embedding(image_bytes_for_face)
    face_detected = face_embedding is not None

    # Crop and compress car image using YOLO
    car_img_bytes = crop_and_compress_car(image_bytes_for_car)
    if car_img_bytes is None:
        raise HTTPException(status_code=400, detail="No car detected.")

    # Extract license plate, make, and model
    license_plate, car_make, car_model = extract_car_info(car_img_bytes)

    # Retrieve authorized vehicle records
    vehicles = get_all_authorized_vehicles()

    face_match = False
    plate_match = False
    matched_entity = None
    matched_entity_id = None

    # Match against authorized vehicles
    for entity in vehicles:
        if face_detected:
            stored_embedding = entity["face_embedding"]
            sim = cosine_similarity(face_embedding, stored_embedding)
            if sim >= MATCH_THRESHOLD:
                face_match = True
                matched_entity = entity
                matched_entity_id = entity.get("UserID", "Unknown")
        if entity["license_plate"].strip().lower() == license_plate.strip().lower():
            plate_match = True
            if not matched_entity:
                matched_entity = entity
                matched_entity_id = entity.get("UserID", "Unknown")

    # Determine verification status
    if not face_detected:
        status = "No face detected"
    elif face_match and plate_match:
        status = "Authorized Vehicle"
    elif face_match and not plate_match:
        status = "Face match, License mismatch"
    elif not face_match and plate_match:
        status = "License match, Face mismatch"
    else:
        status = "Unauthorized Vehicle"

    # Check for car make/model mismatch
    car_mismatch = False
    if matched_entity:
        stored_make = matched_entity.get("car_make", "").strip().lower()
        stored_model = matched_entity.get("car_model", "").strip().lower()
        if stored_make != car_make.strip().lower() or stored_model != car_model.strip().lower():
            car_mismatch = True
            status += ", Car Mismatch"

    # Log vehicle verification attempt
    log_vehicle_verification_attempt({
        "Type": "vehicle",
        "Status": status,
        "FaceDetected": str(face_detected),
        "FaceMatch": str(face_match),
        "PlateMatch": str(plate_match),
        "LicensePlate": license_plate,
        "CarMake": car_make,
        "CarModel": car_model,
        "MatchedUserID": matched_entity_id,
        "EventTimestamp": datetime.utcnow().isoformat()
    })

    return {
        "status": status,
        "face_detected": face_detected,
        "face_match": face_match,
        "plate_match": plate_match,
        "car_mismatch": car_mismatch,
        "license_plate": license_plate,
        "car_make": car_make,
        "car_model": car_model,
        "matched_user_id": matched_entity_id,
        "timestamp": datetime.utcnow().isoformat()
    }

# Fetch all pedestrian verification logs
@app.get("/logs/pedestrians")
def get_pedestrian_logs():
    try:
        logs = fetch_pedestrian_logs()
        result = []
        for log in logs:
            result.append({
                "timestamp": log.get("Timestamp"),
                "status": log.get("Status"),
                "type": log.get("Type"),
                "row_key": log.get("RowKey")
            })
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Fetch all vehicle verification logs
@app.get("/logs/vehicles")
def get_vehicle_logs():
    try:
        logs = fetch_vehicle_logs()
        result = []
        for log in logs:
            result.append({
                "timestamp": log.get("Timestamp"),
                "status": log.get("Status"),
                "type": log.get("Type"),
                "license_plate": log.get("LicensePlate"),
                "car_make": log.get("CarMake"),
                "car_model": log.get("CarModel"),
                "face_match": log.get("FaceMatch"),
                "plate_match": log.get("PlateMatch"),
                "matched_user_id": log.get("MatchedUserID"),
                "row_key": log.get("RowKey")
            })
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
