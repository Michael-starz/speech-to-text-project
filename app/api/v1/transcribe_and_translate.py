from fastapi import APIRouter, UploadFile, File, Form
from app.schema import ApiResponse, TranscribeAndTranslate
from app.services.whisper_service import WhisperService

router = APIRouter()
whisper = WhisperService()

@router.post("", response_model=ApiResponse[TranscribeAndTranslate])
async def transcribe_and_translate(
    file: UploadFile = File(...),
    target_language: str = Form(...)
):
    """Transcribe audio and translate the result"""
    result = await whisper.transcribe_and_translate(file, target_language)
    return ApiResponse(data=result)
