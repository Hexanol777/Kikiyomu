import easyocr
from PIL import ImageGrab, Image
import numpy as np
import os
import urllib.request
import torch
import re


import logging
logging.getLogger("PIL").setLevel(logging.WARNING)



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
        logging.info(f"Downloading {file_name} ...")
        urllib.request.urlretrieve(url, path)
        logging.info(f"Saved to {path}")


def ensure_models(): # Ensure single execution in case of future uses of multiprocessing
    download_model(RECOGNITION_MODEL, RECOGNITION_URL)
    download_model(DETECTION_MODEL, DETECTION_URL)


ensure_models()


# Safe GPU fallback
use_gpu = torch.cuda.is_available()

# Set reader
reader = easyocr.Reader(['ja'], gpu=use_gpu, model_storage_directory='models/ocr')

def get_clipboard_image(image):
    """Returns an image from clipboard, or None if not an image."""
    img = ImageGrab.grabclipboard()
    if isinstance(img, Image.Image):
        return img
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



def join_separate_lines(text):
    """
    Joins the seperated lines from OCR into one cohesive line
    """

    if not text:
        return text

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    merged = "".join(lines)
    merged = re.sub(r"。+", "。", merged)

    return merged


def OCR(image):

    extracted_text = extract_text(image)

    if extracted_text:
        extracted_text = join_separate_lines(extracted_text)

    return extracted_text
