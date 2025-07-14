from openai import AsyncOpenAI, OpenAIError
import os
from fastapi import UploadFile, HTTPException
import tempfile
import aiohttp
import asyncio
from app.utils.logger import logger


class WhisperService:
    def __init__(self):
        # OpenAI setup
        self.openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # HuggingFace setup
        self.hf_api_key = os.getenv("HUGGINGFACE_API_KEY")
        self.hf_base_url = "https://api-inference.huggingface.co/models"
        
        # File settings
        self.allowed_extensions = {".mp3", ".wav", ".m4a", ".mp4"}
        self.max_file_size_mb = 25
        
        # Translation models
        self.translation_models = {
            "spanish": "Helsinki-NLP/opus-mt-en-es",
            "french": "Helsinki-NLP/opus-mt-en-fr", 
            "german": "Helsinki-NLP/opus-mt-en-de",
            "italian": "Helsinki-NLP/opus-mt-en-it",
            "portuguese": "Helsinki-NLP/opus-mt-en-pt",
        }

    async def transcribe(self, file: UploadFile) -> str:
        """Transcribe audio using OpenAI Whisper (unchanged)"""
        # Validating file type
        if not any(file.filename.endswith(ext) for ext in self.allowed_extensions):
            raise HTTPException(
                status_code=400,
                detail="Unsupported file type. Please upload MP3, WAV, MP4, or M4A."
            )
        
        # Read file content
        file_content = await file.read()
        
        # Check file size
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
                response = await self.openai_client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file
                )
            
            return response.text
        except OpenAIError as e:
            logger.error(f"OpenAI API error: {e}")
            raise HTTPException(
                status_code=502,
                detail="An error occurred while contacting the transcription service."
            )
        except Exception as e:
            logger.exception(f"Unexpected error during transcription: {e}")
            raise HTTPException(
                status_code=500,
                detail="Something went wrong during transcription."
            )
        finally:
            # Clean up temp file
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

    async def translate(self, text: str, target_language: str, use_huggingface: bool = False) -> str:
        """
        Translate text using OpenAI or HuggingFace
        
        Args:
            text: Text to translate
            target_language: Target language
            use_huggingface: If True, use HuggingFace instead of OpenAI
        """
        if use_huggingface or not os.getenv("OPENAI_API_KEY"):
            return await self._translate_with_huggingface(text, target_language)
        else:
            return await self._translate_with_openai(text, target_language)

    async def _translate_with_openai(self, text: str, target_language: str) -> str:
        """Translate using OpenAI"""
        try:
            prompt = f"Translate the following text into {target_language}: \n\n{text}"

            response = await self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a professional translator."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )

            return response.choices[0].message.content.strip()
        except OpenAIError as e:
            logger.error(f"OpenAI API error during translation: {e}")
            # Fallback to HuggingFace
            logger.info("Falling back to HuggingFace translation")
            return await self._translate_with_huggingface(text, target_language)
        except Exception as e:
            logger.exception(f"Unexpected error during OpenAI translation: {e}")
            raise HTTPException(
                status_code=500,
                detail="Translation service error"
            )

    async def _translate_with_huggingface(self, text: str, target_language: str) -> str:
        """Translate using HuggingFace"""
        try:
            model_name = self.translation_models.get(target_language.lower(), "Helsinki-NLP/opus-mt-en-es")
            
            headers = {"Content-Type": "application/json"}
            if self.hf_api_key:
                headers["Authorization"] = f"Bearer {self.hf_api_key}"
            
            payload = {"inputs": text}
            url = f"{self.hf_base_url}/{model_name}"
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, headers=headers) as response:
                    if response.status == 200:
                        result = await response.json()
                        if isinstance(result, list) and len(result) > 0:
                            return result[0].get("translation_text", "Translation failed").strip()
                    elif response.status == 503:
                        # Model loading, wait and retry
                        await asyncio.sleep(10)
                        async with session.post(url, json=payload, headers=headers) as retry_response:
                            if retry_response.status == 200:
                                result = await retry_response.json()
                                if isinstance(result, list) and len(result) > 0:
                                    return result[0].get("translation_text", "Translation failed").strip()
                    
                    error_text = await response.text()
                    logger.error(f"HuggingFace API error {response.status}: {error_text}")
                    raise HTTPException(
                        status_code=502,
                        detail="Translation service temporarily unavailable"
                    )
                    
        except Exception as e:
            logger.exception(f"HuggingFace translation error: {e}")
            raise HTTPException(
                status_code=500,
                detail="Translation service error"
            )

    async def transcribe_and_translate(self, file: UploadFile, target_language: str, use_huggingface: bool = False) -> dict:
        """Transcribe and translate with option to choose translation service"""
        transcribed_text = await self.transcribe(file)
        translated_text = await self.translate(transcribed_text, target_language, use_huggingface)
        
        return {
            "transcription": transcribed_text,
            "translation": translated_text,
            "target_language": target_language,
            "translation_service": "huggingface" if use_huggingface else "openai"
        }