from pydantic import BaseModel
from typing import Generic, TypeVar

T = TypeVar("T")

class TranslationRequest(BaseModel):
    text: str
    language: str

class TranslationResponse(BaseModel):
    translated_text: str

class TranscriptionResponse(BaseModel):
    transcript: str
    
class ApiResponse(BaseModel, Generic[T]):
    data: T

class TranscribeAndTranslate(BaseModel):
    transcription: str
    translation: str
    target_language: str
