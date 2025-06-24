import easyocr
from PIL import ImageGrab, Image
import numpy as np
import os
import urllib.request
import torch


# CONS
OCR_MODEL_DIR = "models/ocr"
RECOGNITION_MODEL = "japanese_g2.pth"
DETECTION_MODEL = "craft_mlt_25k.pth"
RECOGNITION_URL = "https://huggingface.co/xiaoyao9184/easyocr/resolve/master/japanese_g2.pth?download=true"
DETECTION_URL = "https://huggingface.co/xiaoyao9184/easyocr/resolve/master/craft_mlt_25k.pth?download=true"

# Ensuring the folder exists
os.makedirs(OCR_MODEL_DIR, exist_ok=True)

def download_model(file_name, url):
    path = os.path.join(OCR_MODEL_DIR, file_name)
    if not os.path.exists(path):
        print(f"Downloading {file_name} ...")
        urllib.request.urlretrieve(url, path)
        print(f"Saved to {path}")

# Downloading the models
download_model(RECOGNITION_MODEL, RECOGNITION_URL)
download_model(DETECTION_MODEL, DETECTION_URL)


# Safe GPU fallback
use_gpu = torch.cuda.is_available()

# Set reader
reader = easyocr.Reader(['ja'], gpu=True, model_storage_directory='models/ocr')

def get_clipboard_image():
    """Returns an image from clipboard, or None if not an image."""
    img = ImageGrab.grabclipboard()
    if isinstance(img, Image.Image):
        image = Image.open(img)
        return image
    return None

def extract_text(image):
    extracted = []

    img_rgb = image.convert("RGB")
    img_np = np.array(img_rgb)

    result = reader.readtext(img_np, detail=0)

    for txt in result:
        if any('\u3040' <= c <= '\u30ff' or '\u4e00' <= c <= '\u9faf' for c in txt): # Check for japanese ranges
            extracted.append(txt.strip())

    
    print ("\n".join(extracted))
    return "\n".join(extracted) if extracted else None

def OCR(image):
    """Performs OCR using the methods inside ocr.py"""
    boxes, image_bgr = detect_text_boxes(image)
    extracted_text = extract_text(boxes, image_bgr)
    return extracted_text
