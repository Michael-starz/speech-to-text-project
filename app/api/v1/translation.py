from fastapi import APIRouter
from app.schema import TranslationRequest, TranslationResponse
from app.services.gpt_service import GptService
from app.services.hugginface_tr import WhisperService

router = APIRouter()
translator = GptService()
hugging_face = WhisperService()

@router.post("", response_model=TranslationResponse)
async def translate_text(req: TranslationRequest):
    """
    Translate transcript English text into the specified target language using GPT.
    """
    # translation = await translator.translate(req.text, req.language)
    hugf_translate = await hugging_face.translate(text=req.text, target_language=req.language, use_huggingface=True)
    return {"translated_text": hugf_translate}
