from fastapi import APIRouter
from app.schema import TranslationRequest, TranslationResponse
from app.services.gpt_service import GptService

router = APIRouter()
translator = GptService()

@router.post("", response_model=TranslationResponse)
async def translate_text(req: TranslationRequest):
    """
    Translate transcript English text into the specified target language using GPT.
    """
    translation = await translator.translate(req.text, req.language)
    return {"translated_text": translation}
