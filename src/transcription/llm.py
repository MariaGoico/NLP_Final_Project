# src/transcription/llm.py
import os
from llama_cpp import Llama

_model = None

def get_model():
    global _model
    if _model is None:
        print("Loading LLaMA model...")
        _model = Llama(
            model_path="/models/llama-3.2-1b-instruct-q4_k_m.gguf",
            n_ctx=2048,
            n_threads=4,
            verbose=False,
        )
        print("LLaMA model loaded.")
    return _model

def answer(question: str, context_segments: list[dict]) -> str:
    """
    question: pregunta del usuario
    context_segments: lista de segmentos relevantes de Qdrant
    """
    context = ""
    for seg in context_segments:
        context += f"[{seg['start']}s - {seg['end']}s] {seg['text']}"
        if seg.get("scene_desc"):
            context += f" (visual: {seg['scene_desc']})"
        context += "\n"

    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a helpful assistant that answers questions about video content.
You are given relevant segments from a video with their timestamps and visual descriptions.
Always reference the timestamps in your answer.<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Video segments:
{context}
Question: {question}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""

    model = get_model()
    response = model(
        prompt,
        max_tokens=256,
        temperature=0.3,
        stop=["<|eot_id|>", "<|end_of_text|>"],
    )
    return response["choices"][0]["text"].strip()