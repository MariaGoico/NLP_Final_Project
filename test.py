import whisper
from moviepy import VideoFileClip

def procesar_video(ruta_video):
    # 1. Extraer audio temporalmente
    video = VideoFileClip(ruta_video)
    video.audio.write_audiofile("temp_audio.mp3")

    # 2. Cargar Whisper (modelo base es rápido y bueno para inglés)
    model = whisper.load_model("base")
    
    print("Transcribiendo... esto puede tardar dependiendo de tu hardware.")
    result = model.transcribe("temp_audio.mp3")

    # 3. Mostrar fragmentos con tiempo
    for segment in result['segments']:
        start = segment['start']
        text = segment['text']
        print(f"[{start:.2f}s]: {text}")

if __name__ == "__main__":
    # Prueba con un video que tengas en la carpeta
    procesar_video("videos/videoplayback.mp4")