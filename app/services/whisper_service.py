from openai import AsyncOpenAI, OpenAIError
import os
from fastapi import UploadFile, HTTPException
import tempfile
from app.utils.logger import logger
from dotenv import load_dotenv
from typing import Dict
from ..schema import TranscribeAndTranslate
# import time
# from pydub import AudioSegment
# from pydub.utils import which

class WhisperService:
    def __init__(self):
        load_dotenv(override=True)
        api_key = os.getenv("OPENAI_API_KEY")
        self.client = AsyncOpenAI(api_key=api_key)
        self.allowed_extensions = {".mp3", ".wav", ".m4a", ".mp4", ".webm", ".mpga", ".mpeg"}
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
            logger.error("Unsupported file type. Bad request.")
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
                tmp.write(file_content)
                tmp_path = tmp.name

            # Transcribe with OpenAI Whisper
            with open(tmp_path, "rb") as audio_file:
                response = await self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    language="en"
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


    async def translate(self, text: str, language: str) -> str:
        try:
            prompt = (
                f"Translate the following English text into {language}: \n\n{text}"
            )

            response = await self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a professional translator."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
            )

            return response.choices[0].message.content.strip()
        except OpenAIError as e:
            logger.error(f"OpenAI API error during translation: {e}")
            raise HTTPException(
                status_code=502,
                detail="An error occurred while contacting the translation service. Please try again later."
            )
        except Exception as e:
            logger.exception(f"Unexpected error during translation: {e}")
            raise HTTPException(
                status_code=500,
                detail="Something went wrong during translation. Please try again."
            )
        
    
    async def transcribe_and_translate(self, file: UploadFile, target_language: str) -> Dict[str, str]:
        """
        Transcribes an audio file and translates the result.

        Args:
            file (UploadFile): The uploaded audio file
            target_language (str): Target language for translation

        Returns:
            dict: Dictionary containing transcription, translation and target language
        """
        # start_time = time.perf_counter()

        # file_size_mb = round(os.fstat(file.file.fileno()).st_size / (1024 * 1024), 2)
        
        # file.file.seek(0)  # Reset pointer
        # audio = AudioSegment.from_file(file.file)
        # audio_duration_sec = round(len(audio) / 1000, 2)  # Convert ms to seconds
        # file.file.seek(0)  # Reset again for transcription


        # First transcribe the audio
        transcribed_text = await self.transcribe(file)

        # # Then translate the transcribed text
        translated_text = await self.translate(text=transcribed_text, language=target_language)

        # end_time = time.perf_counter()
        # processing_time = round(end_time - start_time, 2)
        # logger.info(
        #     f"Processing Time: {processing_time}s | File Size: {file_size_mb}MB | Duration: {audio_duration_sec}s"
        # )
        print(TranscribeAndTranslate(
            transcription=transcribed_text,
            translation=translated_text,
            target_language=target_language
        ))
        return TranscribeAndTranslate(
            transcription=transcribed_text,
            translation=translated_text,
            target_language=target_language
        )
    