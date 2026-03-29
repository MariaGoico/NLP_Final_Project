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
            n_gpu_layers=35,   # ← todo en GPU
            verbose=False,
        )
        print("LLaMA model loaded.")
    return _model

def answer(question: str, context_segments: list[dict]) -> str:
    global_ctx = [s for s in context_segments if s.get("level") == "video_summary"]
    specific_ctx = [s for s in context_segments if s.get("level") != "video_summary"]

#     prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
# You are a precise assistant that answers questions about video content.
# You have access to both a full video summary and specific relevant segments.
# Rules:
# - For general questions (theme, topic, summary), use the full video context.
# - For specific questions, cite the timestamp (e.g. "at 10.8s").
# - If the answer is not found in the context, say "I could not find that in the video."
# - Be concise and direct.
# - Do not invent information not present in the context.<|eot_id|>
# <|start_header_id|>user<|end_header_id|>
# """
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    You are an intelligent assistant analyzing movie trailers.
    You have access to a video summary and specific segments containing character dialogue and visual descriptions.
    Rules:
    - You CAN and SHOULD use character dialogue and quotes to deduce the answer. If a character says something (e.g., "hanging like a bat"), treat it as what is happening in the scene.
    - For general questions (theme, topic, summary), use the full video context.
    - For specific questions, cite the timestamp (e.g. "at 10.8s").
    - Only if the answer cannot be deduced from the dialogue or visual descriptions at all, say "I could not find that in the video."
    - Be concise and direct.
    - Do not hallucinate information outside of the provided text and visual contexts.<|eot_id|>
    <|start_header_id|>user<|end_header_id|>
    """


    if global_ctx:
        prompt += f"Full video context:\n{global_ctx[0]['text']}\n\n"

    if specific_ctx:
        prompt += "Relevant segments:\n"
        for seg in specific_ctx:
            prompt += f"[{seg['start']}s - {seg['end']}s] {seg['text']}"
            if seg.get("scene_desc"):
                prompt += f" (visual: {seg['scene_desc']})"
            prompt += "\n"

    prompt += f"\nQuestion: {question}<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>\n"

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