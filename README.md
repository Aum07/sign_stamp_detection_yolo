# sign_stamp_detection_yolo

# Sign and Stamp Detection API

This project provides an API for detecting signatures and stamps in images using the YOLO model. The API can be used to process images and identify the presence of these marks, with the flexibility to customize detection models as required.

## Prerequisites

Before running the application, ensure you have the following installed:

- Python 3.x
- pip (Python package installer)
- Git

### Setup Instructions

#### Step 1: Clone the Repository

Clone the repository to your local machine:

bash
git clone https://github.com/Aum07/sign_stamp_detection_yolo.git
cd sign_stamp_detection_yolo

Step 2: Install Dependencies
pip install -r requirements.txt
This will install all necessary libraries for the project.

Step 3: Download the YOLO Model
You need to download the YOLO model from Google Drive. Please follow the link below to download the model folder:
https://drive.google.com/drive/folders/1GJnSoqkhbQjMwpJFaLGJCPi7OIIgA-Vi?usp=sharing

Download YOLO Model Folder: YOLO Model Download
After downloading, extract the folder and place it in the project directory under the yolo_model/ folder.

Step 4: Run the Application
uvicorn app:app --reload


Example Request for Detection
To send an image for signature and stamp detection, use the following curl command:

bash
curl -X 'POST' \
  'http://127.0.0.1:8000/detect' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@/path/to/your/image.jpg'

