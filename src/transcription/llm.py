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
    context = ""
    for i, seg in enumerate(context_segments, 1):
        context += f"[Segment {i} | {seg['start']}s - {seg['end']}s]\n"
        context += f"  Spoken: {seg['text']}\n"
        if seg.get("scene_desc"):
            context += f"  Visual: {seg['scene_desc']}\n"
        context += "\n"

    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a precise assistant that answers questions about video content.
You are given the most relevant segments from a video, each with a timestamp, spoken text, and visual description.
Rules:
- Always cite the timestamp (e.g. "at 10.8s") when referencing a segment.
- If the answer is not found in the segments, say "I could not find that in the video."
- Be concise and direct.
- Do not invent information not present in the segments.<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Relevant video segments:
{context}
Question: {question}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""

    model = get_model()
    response = model(
        prompt,
        max_tokens=300,
        temperature=0.1,
        top_p=0.9,
        repeat_penalty=1.1,
        stop=["<|eot_id|>", "<|end_of_text|>"],
    )
    return response["choices"][0]["text"].strip()