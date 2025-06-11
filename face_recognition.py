from insightface.app import FaceAnalysis
import numpy as np
import cv2
from PIL import Image
import io

# Initialize the FaceAnalysis model once globally using the 'buffalo_l' model
face_app = FaceAnalysis(name='buffalo_l')
face_app.prepare(ctx_id=-1, det_size=(1024, 1024))  # Use CPU (ctx_id=-1) and set detection size

# Decode image bytes into an OpenCV-compatible BGR image
def decode_image_bytes(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")  # Load and convert image to RGB
    img = np.array(img)  # Convert PIL image to NumPy array
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR for OpenCV
    return img

# Crop face region from image using bounding box
def crop_face_from_bbox(img: np.ndarray, bbox: list) -> np.ndarray:
    x1, y1, x2, y2 = map(int, bbox)  # Extract and convert bounding box coordinates
    return img[y1:y2, x1:x2]  # Return cropped image region

# Extract normalized face embedding from image bytes
def get_face_embedding(image_bytes: bytes) -> list | None:
    img = decode_image_bytes(image_bytes)  # Convert image bytes to BGR image

    # Apply sharpening filter to enhance facial features
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    img = cv2.filter2D(img, -1, kernel)

    # Resize image if dimensions are smaller than 800px
    h, w = img.shape[:2]
    if max(h, w) < 800:
        img = cv2.resize(img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)

    # Detect faces in the image
    faces = face_app.get(img)
    if not faces:
        print("[DEBUG] No face detected.")  # Log if no face is found
        return None

    # Select the largest detected face
    face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))

    _ = crop_face_from_bbox(img, face.bbox)  # Optionally crop face for inspection/debug

    return face.normed_embedding.tolist()  # Return normalized face embedding as a list
