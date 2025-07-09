from fastapi import FastAPI, UploadFile, File, HTTPException
from datetime import UTC
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from datetime import datetime, timezone
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

def cosine_similarity(vec1, vec2):
    v1 = np.array(vec1)
    v2 = np.array(vec2)
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

MATCH_THRESHOLD = 0.4

def bytes_to_ndarray(image_bytes: bytes) -> np.ndarray:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return np.array(image)

@app.post("/detect_faces/")
async def verify_pedestrian(request: Request, image: UploadFile = File(...)):
    image_bytes = await image.read()
    embedding = get_face_embedding(image_bytes)
    face_detected = embedding is not None
    face_match = False
    timestamp = datetime.now(UTC).isoformat()

    if face_detected:
        authorized_faces = get_all_authorized_pedestrians()
        for person in authorized_faces:
            stored_embedding = person["face_embedding"]
            sim = cosine_similarity(embedding, stored_embedding)
            if sim >= MATCH_THRESHOLD:
                face_match = True
                break

    # Consistent terminology and logging
    if not face_detected:
        status = "Unauthorized"
        face_status = "Not Found"
    elif face_match:
        status = "Authorized"
        face_status = "Match"
    else:
        status = "Unauthorized"
        face_status = "Mismatch"

    log_pedestrian_verification_attempt({
        "Type": "pedestrian",
        "Status": status,          # Only "Authorized" or "Unauthorized"
        "Face": face_status,       # "Match", "Mismatch", "Not Found"
        "EventTimestamp": timestamp
    })

    return {
        "status": status,          # Only "Authorized" or "Unauthorized"
        "face_status": face_status, # "Match", "Mismatch", "Not Found"
        "timestamp": timestamp
    }

@app.post("/verify-vehicle/")
async def verify_vehicle(image: UploadFile = File(...)):
    image_bytes = await image.read()
    image_bytes_for_face = BytesIO(image_bytes).getvalue()
    image_bytes_for_car = BytesIO(image_bytes).getvalue()

    # 1. Detect face & extract embedding
    face_embedding = get_face_embedding(image_bytes_for_face)
    face_found = face_embedding is not None

    # 2. Crop car, extract info
    car_img_bytes = crop_and_compress_car(image_bytes_for_car)
    if car_img_bytes is None:
        car_found = False
        license_plate = None
        car_make = None
        car_model = None
    else:
        car_found = True
        license_plate, car_make, car_model = extract_car_info(car_img_bytes)

    # 3. Check extraction for not found
    license_plate_found = bool(license_plate and str(license_plate).strip())
    car_make_found = bool(car_make and str(car_make).strip())
    car_model_found = bool(car_model and str(car_model).strip())

    vehicles = get_all_authorized_vehicles()
    face_match, plate_match, make_match, model_match = False, False, False, False
    matched_entity = None

    for entity in vehicles:
        # License plate
        if license_plate_found and "license_plate" in entity and entity["license_plate"]:
            if entity["license_plate"].strip().lower() == license_plate.strip().lower():
                plate_match = True
                matched_entity = entity
        # Face embedding
        if face_found and "face_embedding" in entity and entity["face_embedding"]:
            stored_embedding = entity["face_embedding"]
            sim = cosine_similarity(face_embedding, stored_embedding)
            if sim >= MATCH_THRESHOLD:
                face_match = True
                matched_entity = entity
    if matched_entity:
        # Car Make Match
        if car_make_found and "car_make" in matched_entity and matched_entity["car_make"]:
            make_match = matched_entity["car_make"].strip().lower() == car_make.strip().lower()
        elif car_make_found:
            make_match = False
        else:
            make_match = False
        # Car Model Match
        if car_model_found and "car_model" in matched_entity and matched_entity["car_model"]:
            model_match = matched_entity["car_model"].strip().lower() == car_model.strip().lower()
        elif car_model_found:
            model_match = False
        else:
            model_match = False

    # 5. Label statuses for API and log
    face_status = "Not Found" if not face_found else ("Match" if face_match else "Mismatch")
    plate_status = "Not Found" if not license_plate_found else ("Match" if plate_match else "Mismatch")
    make_status = "Not Found" if not car_make_found else ("Match" if make_match else "Mismatch")
    model_status = "Not Found" if not car_model_found else ("Match" if model_match else "Mismatch")

    # 6. Output message logic: Only Authorized/Unauthorized, never extra message
    if face_status == "Match" and plate_status == "Match":
        output_message = "Authorized"
    else:
        output_message = "Unauthorized"

    matched_user_id = matched_entity.get("UserID", "Unknown") if matched_entity else "Unknown"

    log_vehicle_verification_attempt({
        "Type": "vehicle",
        "Status": output_message,
        "Face": face_status,
        "LicensePlate": plate_status,
        "CarMake": make_status,
        "CarModel": model_status,
        "DetectedLicensePlate": license_plate if license_plate else "Unknown",
        "DetectedCarMake": car_make if car_make else "Unknown",
        "DetectedCarModel": car_model if car_model else "Unknown",
        "MatchedUserID": matched_user_id,
        "EventTimestamp": datetime.now(timezone.utc).isoformat()
    })

    return JSONResponse(content={
        "output": output_message,
        "face_status": face_status,
        "license_plate_status": plate_status,
        "car_make_status": make_status,
        "car_model_status": model_status,
        "detected_license_plate": license_plate if license_plate else "Unknown",
        "detected_car_make": car_make if car_make else "Unknown",
        "detected_car_model": car_model if car_model else "Unknown",
        "matched_user_id": matched_user_id,
        "timestamp": datetime.now(timezone.utc).isoformat()
    })

@app.get("/logs/pedestrians")
def get_pedestrian_logs():
    try:
        logs = fetch_pedestrian_logs()
        result = []
        for log in logs:
            timestamp = (
                log.get("EventTimestamp")
                or log.get("Timestamp")
                or log.get("LoggedAtUTC")
            )
            if timestamp and hasattr(timestamp, "isoformat"):
                timestamp = timestamp.isoformat()
            result.append({
                "timestamp": timestamp,
                "status": log.get("Status", "Unauthorized"),
                "face_status": log.get("Face", "Not Found"),
                "type": log.get("Type"),
                "row_key": log.get("RowKey")
            })
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/logs/vehicles")
def get_vehicle_logs():
    try:
        logs = fetch_vehicle_logs()
        result = []
        for log in logs:
            timestamp = (
                log.get("EventTimestamp")
                or log.get("Timestamp")
                or log.get("LoggedAtUTC")
            )
            if timestamp and hasattr(timestamp, "isoformat"):
                timestamp = timestamp.isoformat()
            result.append({
                "timestamp": timestamp,
                "face_status": log.get("Face", "Not Found"),
                "license_plate_status": log.get("LicensePlate", "Not Found"),
                "car_make_status": log.get("CarMake", "Not Found"),
                "car_model_status": log.get("CarModel", "Not Found"),
                "detected_license_plate": log.get("DetectedLicensePlate", "Unknown"),
                "detected_car_make": log.get("DetectedCarMake", "Unknown"),
                "detected_car_model": log.get("DetectedCarModel", "Unknown"),
                "output": log.get("Status", "Unauthorized"),
                "matched_user_id": log.get("MatchedUserID", "Unknown"),
                "row_key": log.get("RowKey")
            })
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(
#         "main:app",
#         host="0.0.0.0",
#         port=4200
#     )
