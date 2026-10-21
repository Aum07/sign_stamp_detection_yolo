from fastapi import FastAPI, File, UploadFile, HTTPException
from pathlib import Path
from typing import Dict, Any
import re
import uuid
import sys
import warnings
import logging
from functions import detect_signatures_and_stamps, pdf_pages_to_images, convert_to_png

if not sys.warnoptions:
    warnings.simplefilter("ignore")

app = FastAPI()
logger = logging.getLogger(__name__)

UPLOAD_DIR = Path("data")
UPLOAD_DIR.mkdir(exist_ok=True)

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
    include_image: bool = True,
) -> Dict[str, Any]:
    """Detect signatures and stamps. Set include_image=false to skip the annotated image (much smaller response)."""
    if not file or file.filename == '':
        raise HTTPException(status_code=400, detail="No selected file")

    # Every temporary file for this request starts with this UUID,
    # so all of them can be deleted at the end, even if processing fails.
    file_uuid = generate_uuid()
    filename = secure_filename(f"{file_uuid}_" + Path(file.filename).suffix)
    file_path = UPLOAD_DIR / filename

    try:
        # Write uploaded file to disk
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())

        details = {}

        if file.content_type == 'application/pdf':
            image_paths = pdf_pages_to_images(file_path, file_uuid)
            for i, image_path in enumerate(image_paths, start=1):
                try:
                    details[f"page{i}"] = detect_signatures_and_stamps(image_path, include_image)
                except Exception as e:
                    raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")
        else:
            try:
                img = convert_to_png(str(file_path))
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Could not read image: {str(e)}")
            try:
                details["page1"] = detect_signatures_and_stamps(img, include_image)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error performing sign and stamp detection: {str(e)}")

        return details

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Delete the upload, converted PNG and PDF page images (they contain personal data)
        for temp_file in UPLOAD_DIR.glob(f"{file_uuid}_*"):
            try:
                temp_file.unlink()
            except OSError as e:
                logger.warning(f"Could not delete temporary file {temp_file}: {e}")
