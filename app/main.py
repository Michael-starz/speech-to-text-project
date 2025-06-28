from fastapi import FastAPI
from app.api.v1.translation import router as translation_router
from app.api.v1.transcription import router as transcription_router
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
