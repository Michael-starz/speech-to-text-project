import os
from app.utils.logger import logger
from fastapi import HTTPException
from openai import AsyncOpenAI, OpenAIError


class GptService:
    def __init__(self):
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    async def translate(self, text: str, language: str) -> str:
        try:
            prompt = (
                f"Translate the following English text into {language}: \n\n{text}"
            )

            response = await self.client.chat.completions.create(
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
