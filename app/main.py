from fastapi import FastAPI
from app.api.v1.translation import router as translation_router
from app.api.v1.transcription import router as transcription_router
from app.api.v1.transcribe_and_translate import router as transcribe_and_translate_router
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Multilingual Transcription & Translation API",
    description="Transcribe audio and translate text using OpenAI's Whisper and GPT models.",
    version="1.0.0"
)


# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # I need to adjust this when in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(transcription_router, prefix="/transcribe", tags=["Transcription"])
app.include_router(translation_router, prefix="/translate", tags=["Translation"])
app.include_router(transcribe_and_translate_router, prefix="/v1/transcribe-and-translate", tags=["Translate-and-Transcribe"])


