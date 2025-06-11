from ultralytics import YOLO
import cv2
import numpy as np
from typing import Optional

CAR_CLASS_IDX = 2  # COCO class index for 'car'
yolo_model = YOLO('yolov8n.pt')  # Load YOLOv8n model

# Crop car from image and compress it as JPEG
def crop_and_compress_car(image_bytes: bytes, min_side: int = 400, jpeg_quality: int = 85) -> Optional[bytes]:
    nparr = np.frombuffer(image_bytes, np.uint8)  # Convert bytes to numpy array
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # Decode image using OpenCV

    if img is None:
        print("[ERROR] Failed to decode image.")  # Log decoding failure
        return None

    results = yolo_model(img)  # Perform object detection using YOLO

    boxes = results[0].boxes  # Get detected bounding boxes
    if boxes is None or boxes.cls is None:
        print("[INFO] No boxes found in YOLO results.")  # Log absence of detections
        return None

    for i, cls in enumerate(boxes.cls.cpu().numpy()):  # Iterate through detected classes
        if int(cls) == CAR_CLASS_IDX:  # Check if detection is a car
            box = boxes.xyxy.cpu().numpy()[i]  # Get bounding box coordinates
            xmin, ymin, xmax, ymax = map(int, box)

            pad = 10  # Add padding around the bounding box
            xmin = max(0, xmin - pad)
            ymin = max(0, ymin - pad)
            xmax = min(img.shape[1], xmax + pad)
            ymax = min(img.shape[0], ymax + pad)

            if xmax <= xmin or ymax <= ymin:
                continue  # Skip invalid crops

            cropped = img[ymin:ymax, xmin:xmax]  # Crop car region from image

            h, w = cropped.shape[:2]
            scale = min_side / min(h, w)  # Calculate scale to resize smallest side
            if scale < 1:
                new_w, new_h = int(w * scale), int(h * scale)  # Calculate new dimensions
                cropped = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_AREA)  # Resize cropped image

            cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

            success, buffer = cv2.imencode('.jpg', cropped_rgb, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])  # Encode to JPEG
            if success:
                return buffer.tobytes()  # Return compressed JPEG bytes
            else:
                print("[ERROR] JPEG compression failed.")  # Log compression failure
                return None

    print("[INFO] No car detected in the image.")  # Log if no car was found
    return None
