import easyocr
from PIL import ImageGrab, Image
import numpy as np
import cv2
import os


# Initialize once globally
reader = easyocr.Reader(['ja'], gpu=False)

def get_clipboard_image():
    """Returns an image from clipboard, or None if not an image."""
    img = ImageGrab.grabclipboard()
    if isinstance(img, Image.Image):
        return img
    return None

def detect_text_boxes(img):
        
        img_rgb = img
        img_rgb = img_rgb.convert("RGB")
        img_np = np.array(img_rgb)

        # Cropped image bg
        image_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        # Text region detection
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        sobel = cv2.Sobel(gray, cv2.CV_8U, 1, 0)
        _, thresh = cv2.threshold(sobel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 2))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = [cv2.boundingRect(cnt) for cnt in contours if cv2.contourArea(cnt) > 100]
        
        return boxes, image_bgr

def extract_japanese_from_image(boxes, image_bgr):

        # OCR
        reader = easyocr.Reader(['ja'], gpu=False)
        extracted = []
        for x, y, w, h in boxes:
            cropped = image_bgr[y:y+h, x:x+w]
            result = reader.readtext(cropped, detail=0)
            for txt in result:
                if any('\u3040' <= c <= '\u30ff' or '\u4e00' <= c <= '\u9faf' for c in txt): # Check for japanese ranges
                    extracted.append(txt.strip())

        return "\n".join(extracted) if extracted else None



def OCR(image):
    """Performs OCR using the methods inside ocr.py"""
    boxes, image_bgr = detect_text_boxes(image)
    extracted_text = extract_japanese_from_image(boxes, image_bgr)

    return extracted_text