from pydantic import BaseModel

class TranslationRequest(BaseModel):
    text: str
    language: str

class TranslationResponse(BaseModel):
    translated_text: str

class TranscriptionResponse(BaseModel):
    transcript: str
    