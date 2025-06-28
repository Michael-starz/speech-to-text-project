from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse
from app.schema import TranscriptionResponse
from app.services.whisper_service import WhisperService

router = APIRouter()
whisper = WhisperService()

@router.post("", response_model=TranscriptionResponse)
async def transcribe_audio(file: UploadFile = File(...)):
    """
    Transcribe an uploaded audio file into English text using Whisper API.
    """
    transcript = await whisper.transcribe(file)
    return {"transcript": transcript}
