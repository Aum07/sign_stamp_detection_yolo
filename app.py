from fastapi import FastAPI, File, UploadFile, HTTPException
from pathlib import Path
from typing import Dict, Any
import re
import uuid
import sys
import warnings
from functions import detect_signatures_and_stamps, pdf_pages_to_images, convert_to_png

if not sys.warnoptions:
    warnings.simplefilter("ignore")

app = FastAPI()

def generate_uuid() -> str:
    """Generate and return a UUID as a string."""
    return str(uuid.uuid4())

def secure_filename(filename: str) -> str:
    """
    Pass it a filename and it will return a secure version of it. This filename can
    then safely be stored on a regular file system and passed to os.path.join. The
    filename returned is an ASCII only string for maximum portability.
    """
    _filename_ascii_strip_re = re.compile(r'[^A-Za-z0-9_.-]')
    filename = filename.strip().replace(' ', '_')
    filename = _filename_ascii_strip_re.sub('', filename)
    return filename

@app.get("/test")
async def test():
    return {"Message": "Hello User, API is Working."}

@app.post("/detect")
async def detect_certificates(
    file: UploadFile = File(...),
) -> Dict[str, Any]:
    
    try:
        # Check if file is provided and not empty
        if not file:
            raise HTTPException(status_code=400, detail="No file part")
        if file.filename == '':
            raise HTTPException(status_code=400, detail="No selected file")

        # Generate a UUID for the file
        file_uuid = generate_uuid()

        # Create a secure filename using UUID
        filename = secure_filename(f"{file_uuid}_" + Path(file.filename).suffix)
        file_path = f'data/{filename}'

        # Write uploaded file to disk
        with open(file_path, "wb") as buffer:
            buffer.write(file.file.read())

        details = {}

        if file.content_type == 'application/pdf':
            # Convert file_path to a Path object
            file_path_obj = Path(file_path)
            if not file_path_obj.is_file():
                raise HTTPException(status_code=400, detail="File does not exist")

            yolo_output = {"signatures": 0, "stamps": 0}
            image_paths = pdf_pages_to_images(file_path_obj, file_uuid)  # Pass file_uuid here
            i = 1
            for image_path in image_paths:
                try:
                    yolo_result = detect_signatures_and_stamps(image_path)
                    details[f"page{i}"] = yolo_result
                    i += 1
                except Exception as e:
                    # Handle OCR and YOLO processing errors
                    raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")
            
            # Remove the original PDF file after processing
            file_path_obj.unlink()

        else:
            # Convert uploaded image to PNG
            img = convert_to_png(file_path)

            # Stamp and signature detection
            try:
                yolo_output = detect_signatures_and_stamps(img)
            except Exception as e:
                # Handle stamp and signature detection errors
                raise HTTPException(status_code=500, detail=f"Error performing sign and stamp detection: {str(e)}")

            details = {
                "page1": yolo_output,
            }

            # Remove the uploaded image file after processing
            file_path_obj = Path(file_path)
            file_path_obj.unlink()

        return details
        
    except HTTPException as http_err:
        # Handle HTTP exceptions
        raise http_err
    except Exception as e:
        # Handle other unexpected errors
        raise HTTPException(status_code=500, detail=str(e))
