import cv2
import base64
from ultralytics import YOLO
import fitz  # PyMuPDF
import numpy as np
from PIL import Image
import uuid
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load a model (only once)
model = YOLO("yolo_model/train9/weights/best.pt")

def detect_signatures_and_stamps(image_path):
    """Detect signatures and stamps in the image."""
    try:
        # Load the image from the given path
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image from {image_path}")

        # Perform detection
        results = model(image)
        detections = []

        for result in results:
            boxes = result.boxes.xyxy  # xyxy format
            confs = result.boxes.conf
            classes = result.boxes.cls

            for box, conf, cls in zip(boxes, confs, classes):
                x1, y1, x2, y2 = map(int, box)
                bbox = [x1, y1, x2 - x1, y2 - y1]  # Convert to [x, y, width, height]
                class_idx = int(cls)
                label = result.names[class_idx]
                confidence = float(conf)
                detections.append({
                    "bbox": bbox,
                    "label": label,
                    "confidence": confidence
                })

        # Optionally save the annotated image and encode to base64
        _, buffer = cv2.imencode('.jpg', image)
        img_str = base64.b64encode(buffer).decode('utf-8')

        return {"detections": detections, "image_with_annotations": f"data:image/jpeg;base64,{img_str}"}

    except Exception as e:
        logger.error(f"Error in detect_signatures_and_stamps: {e}")
        raise ValueError(f"Error processing image: {e}")

def get_sign_stamps(image_path):
    """Get the count of signatures and stamps detected."""
    try:
        # Initialize counts for signatures and stamps
        sign_count = 0
        stamp_count = 0

        # Perform prediction using the preloaded model
        results = model.predict([image_path])
        result = results[0].boxes.cls.tolist()

        # Count the occurrences of signatures and stamps
        for res in result:
            class_p = results[0].names[int(res)]
            if class_p == "mix":
                # Both stamp and signature detected
                sign_count += 1
                stamp_count += 1
            elif class_p == "signature":
                sign_count += 1
            elif class_p == "stamp":
                stamp_count += 1

        return {"signatures": sign_count, "stamps": stamp_count}

    except Exception as e:
        logger.error(f"Error in get_sign_stamps: {e}")
        raise ValueError(f"Error processing image: {e}")

def convert_to_png(input_image_path):
    """Convert the input image to PNG format."""
    try:
        with Image.open(input_image_path) as img:
            filename_without_extension = input_image_path.rsplit('.', 1)[0]
            rgba_img = img.convert('RGBA')
            output_image_path = f"{filename_without_extension}.png"
            rgba_img.save(output_image_path, 'PNG')
        
        return output_image_path

    except Exception as e:
        logger.error(f"Error in convert_to_png: {e}")
        return f"Error occurred: {str(e)}"

def pdf_pages_to_images(pdf_path, output_name):
    """Convert each page of a PDF to an image."""
    try:
        pdf_document = fitz.open(pdf_path)
        image_paths = []

        for page_number in range(pdf_document.page_count):
            page = pdf_document.load_page(page_number)
            pix = page.get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            output_path = f"data/{output_name}_{page_number}.png"
            img.save(output_path)
            image_paths.append(output_path)

        pdf_document.close()
        return image_paths

    except Exception as e:
        logger.error(f"Error in pdf_pages_to_images: {e}")
        raise ValueError(f"Error processing PDF: {e}")
