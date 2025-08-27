from fastapi import APIRouter, UploadFile, File
import easyocr

router = APIRouter()
ocr_reader = easyocr.Reader(['ru', 'en'], gpu=True)

@router.post("/ocr")
async def ocr_image(file: UploadFile = File(...)):
    contents = await file.read()
    result = ocr_reader.readtext(contents, detail=0)
    extracted_text = " ".join(result).strip()
    if not extracted_text:
        return {"text": "", "message": "❗️Текст на изображении не найден."}
    return {"text": extracted_text, "message": "✅ Распознавание завершено успешно."}

