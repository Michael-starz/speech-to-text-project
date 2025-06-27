from openai import OpenAI, OpenAIError
import os
from fastapi import UploadFile, HTTPException
import tempfile
from app.utils.logger import logger


class WhisperService:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.allowed_extensions = {".mp3", ".wav", ".m4a", ".mp4"}
        self.max_file_size_mb = 25

    async def transcribe(self, file: UploadFile) -> str:
        """
        Transcribes an audio file using OpenAI's Whisper API.

        Args:
            file (UploadFile): The uploaded audio file.

        Returns:
            str: Transcribed text from the audio.
        """

        # Validating file type
        if not any(file.filename.endswith(ext) for ext in self.allowed_extensions):
            raise HTTPException(
                status_code=400,
                detail="Unsupported file type. Please upload MP3, WAV, MP4, or M4A."
            )
        
        # Checking file size
        file_content = await file.read()
        file_mb_size = len(file_content) / (1024 * 1024)
        if file_mb_size > self.max_file_size_mb:
            raise HTTPException(
                status_code=400, 
                detail=f"File too large. Max size is {self.max_file_size_mb}MB."
                )
        
        # Save to tempfile and process
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[-1]) as tmp:
                tmp.write(await file_content)
                tmp_path = tmp.name

            # Transcribe with OpenAI Whisper
            with open(tmp_path, "rb") as audio_file:
                response = self.client.audio.transcriptions.create(
                    model="gpt-4o-transcribe",
                    file=audio_file
                )
            
            return response.text
        except OpenAIError as e:
            logger.error(f"OpenAI API error: {e}")
            raise HTTPException(
                status_code=502,
                detail="An error occurred while contacting the transcription service. Please try again later."
            )
        except Exception as e:
            logger.exception(f"Unexpected error during transcription: {e}")
            raise HTTPException(
                status_code=500,
                detail="Something went wrong during transcription. Please try again."
            )
        finally:
            # Clear temp file
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
