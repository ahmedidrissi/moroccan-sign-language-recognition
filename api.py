from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import cv2 
from msl import MSLModel
import os
import shutil

model = MSLModel('./model/model.pickle')

UPLOAD_FOLDER = "./uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = FastAPI()

origins = [
    "http://localhost:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class FrameRequest(BaseModel):
    frame: str

class TextRequest(BaseModel):
    text: str

async def save_upload_file(upload_file: UploadFile, destination: str) -> None:
    """
    Save an uploaded file to the specified destination path.
    Ensures the file is fully written before proceeding.
    """
    with open(destination, "wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)
        buffer.close()

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/process-frame/")
async def process_frame(frame: UploadFile = File(...)):
    # Ensure the uploaded file is an image
    if not frame.filename.lower().endswith((".png", ".jpg", ".jpeg")):
        raise HTTPException(status_code=400, detail="Invalid file type.")
    
    # Save the uploaded file temporarily
    temp_file_path = os.path.join(UPLOAD_FOLDER, frame.filename)
    try:
        await save_upload_file(frame, temp_file_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving file: {str(e)}")

    # Process the frame
    try:
        frame = cv2.imread(temp_file_path)
        detected_sign = await model.processFrame(frame)
    except Exception as e:
        os.remove(temp_file_path)
        raise HTTPException(status_code=500, detail=f"Error processing frame: {str(e)}")

    # Clean up temporary file
    os.remove(temp_file_path)

    return {"result": detected_sign}

@app.post("/process-video/")
async def process_video(video: UploadFile = File(...)):
    # Ensure the uploaded file is a video
    if not video.filename.lower().endswith((".mp4", ".mov", ".avi", ".mkv")):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a video.")

    # Save the uploaded file temporarily
    temp_file_path = os.path.join(UPLOAD_FOLDER, video.filename)
    try:
        await save_upload_file(video, temp_file_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving file: {str(e)}")

    # Process the video
    try:
        detected_signs = await model.processVideo(temp_file_path)
    except Exception as e:
        os.remove(temp_file_path)
        raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")

    # Clean up temporary file
    os.remove(temp_file_path)
    
    return {"result": detected_signs}


@app.post("/convert-text-to-sign/")
async def convert_text_to_sign(request: TextRequest):
    sign_animation = model.convertTextToSign(request.text)
    return {"result": sign_animation}