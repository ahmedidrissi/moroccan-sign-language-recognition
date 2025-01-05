from fastapi import FastAPI, WebSocket, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

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

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/process-frame/")
async def process_frame(request: FrameRequest):
    # TODO: Add the frame processing logic here
    detected_sign = "Hello"
    return {"result": f"Sign detected: {detected_sign}"}

@app.post("/process-video/")
async def process_video(video: UploadFile = File(...)):
    # TODO: Add the video processing logic here
    detected_signs = "Multiple signs detected" 
    return {"result": f"Video processed: {detected_signs}"}


@app.post("/convert-text-to-sign/")
async def convert_text_to_sign(request: TextRequest):
    # TODO: Add the text-to-sign convertion logic here
    sign_animation = "Sign language convertion"
    return {"result": f"Text converted to {sign_animation}"}