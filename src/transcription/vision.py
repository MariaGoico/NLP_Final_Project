# src/transcription/vision.py
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch

_processor = None
_model = None

def get_model():
    global _processor, _model
    if _model is None:
        print("Loading BLIP model...")
        _processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        _model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base",
            torch_dtype=torch.float32,
        )
        _model.eval()
        print("BLIP model loaded.")
    return _processor, _model

def describe_frame(image_path: str) -> str:
    """
    Recibe la ruta local de un frame .jpg y devuelve
    una descripción en texto de lo que se ve.
    """
    processor, model = get_model()
    image = Image.open(image_path).convert("RGB")
    inputs = processor(image, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=50)
    return processor.decode(output[0], skip_special_tokens=True)