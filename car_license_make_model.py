import os
import base64
from openai import AzureOpenAI
from dotenv import load_dotenv
from PIL import Image
import io

# Load environment variables from .env file
load_dotenv()

# Retrieve Azure OpenAI credentials and config from environment
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
api_version = os.getenv("AZURE_OPENAI_API_VERSION")
subscription_key = os.getenv("AZURE_OPENAI_API_KEY")

# Initialize Azure OpenAI client with deployment details
client = AzureOpenAI(
    api_key=subscription_key,
    api_version=api_version,
    azure_endpoint=endpoint,
    azure_deployment=deployment,
)

# Convert input image bytes to JPEG format with RGB and quality 85
def ensure_jpeg_format(image_bytes: bytes) -> bytes:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=85)
    return buffer.getvalue()

# Extract license plate number, car make, and model from car image using GPT-4 Vision
def extract_car_info(image_bytes: bytes) -> tuple[str, str, str]:
    try:
        image_bytes = ensure_jpeg_format(image_bytes)  # Ensure consistent JPEG format
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")  # Encode image to base64

        # Define structured prompt for GPT image processing
        prompt = (
            "You are an AI assistant that extracts information from a car image. "
            "Please return ONLY the following, in this exact format:\n\n"
            "License Plate Number(only read letters and number no special characters or any dot(.)etc): <plate>\n"
            "Make: <make>\n"
            "Model: <model>\n\n"
            "Do not include any extra text or explanation."
        )

        # Send prompt and image to Azure OpenAI's vision model
        response = client.chat.completions.create(
            model=deployment,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            },
                        },
                    ],
                },
            ],
            max_tokens=500,
            temperature=0.2,
        )

        result_text = response.choices[0].message.content.strip()  # Extract response text
        print("GPT Response:", result_text)  # Log raw GPT output for debugging

        # Default values
        plate = make = model = "Unknown"

        # Parse values from GPT response lines
        for line in result_text.splitlines():
            if "license plate" in line.lower():
                plate = line.split(":", 1)[1].strip()
            elif "make" in line.lower():
                make = line.split(":", 1)[1].strip()
            elif "model" in line.lower():
                model = line.split(":", 1)[1].strip()

        return plate, make, model  # Return extracted information

    except Exception as e:
        print("Error in extract_car_info:", e)  # Log exceptions during processing
        return "Unknown", "Unknown", "Unknown"  # Return default fallback values
