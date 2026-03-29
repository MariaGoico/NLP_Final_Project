# CineRAG Benchmark

Sistema de evaluación para comparar configuraciones de CineRAG (umbrales de retrieval, modelos LLM, modelos de visión).

---

## Estructura

```
benchmark/
├── questions.json        # Set de preguntas (6 trailers × ~8 preguntas)
├── run_benchmark.py      # Lanza preguntas contra la API, guarda resultados
├── evaluate.py           # Calcula métricas y genera reporte HTML
└── README.md
```

---

## Flujo de uso

### 1. Instalar dependencias

```bash
pip install requests sentence-transformers numpy
```

### 2. Procesar trailers (si no están ya)

Abre la UI en http://localhost:8501 y procesa los 6 trailers del catálogo.

### 3. Ejecutar benchmark

```bash
python run_benchmark.py --api-url http://localhost:8001 --questions questions.json
```

Esto genera: `results_YYYYMMDD_HHMMSS.json`

### 4. Evaluar y generar reporte

```bash
python evaluate.py --results results_YYYYMMDD_HHMMSS.json --label "baseline"
```

Genera:
- `results_YYYYMMDD_HHMMSS_evaluated.json` — métricas por pregunta
- `results_YYYYMMDD_HHMMSS_baseline_report.html` — reporte visual

Abre el HTML en el navegador para ver la tabla comparativa.

---

## Cómo comparar configuraciones

El flujo para cada experimento es siempre:

```
cambiar config → reiniciar servicios → run_benchmark.py → evaluate.py --label "nombre_experimento"
```

### Ejemplos de experimentos

#### Cambiar umbrales de retrieval

En `src/transcription/main.py`, modifica los `score_threshold`:

```python
# Nivel 3 shortcut (línea ~query)
# Actualmente no hay umbral explícito aquí, se usa el top-1 siempre.
# Para añadirlo, filtra video_hits por score:
video_hits = [h for h in video_hits if h.score > 0.30]  # prueba 0.20, 0.30, 0.40

# Nivel 2 — ventanas
score_threshold=0.15   # prueba 0.10, 0.15, 0.20, 0.25

# Nivel 1 — segmentos
score_threshold=0.15   # prueba 0.10, 0.15, 0.20, 0.25
```

Luego:
```bash
python run_benchmark.py --label "threshold_020"
python evaluate.py --results results_*.json --label "threshold_020"
```

#### Cambiar modelo de visión (BLIP → alternativas)

Opciones recomendadas (local, sin API, CUDA disponible):

| Modelo | Calidad | Velocidad | VRAM | Código |
|--------|---------|-----------|------|--------|
| `Salesforce/blip-image-captioning-base` | ⭐⭐ | ⚡⚡⚡ | ~1GB | actual |
| `Salesforce/blip-image-captioning-large` | ⭐⭐⭐ | ⚡⚡ | ~2GB | cambio mínimo |
| `Salesforce/blip2-opt-2.7b` | ⭐⭐⭐⭐ | ⚡ | ~6GB | requiere `Blip2Processor` |
| `microsoft/git-base-coco` | ⭐⭐⭐ | ⚡⚡⚡ | ~1GB | cambio mínimo |

**Opción más rápida y mejor calidad sin cambiar mucho (`blip-large`):**

```python
# vision.py — solo cambiar el nombre del modelo
_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
_model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-large",
    torch_dtype=torch.float16,  # float16 para ahorrar VRAM con GPU
).to("cuda")
```

**Si quieres OCR también (leer texto en pantalla):**

```python
# Añadir en vision.py junto a BLIP
import pytesseract
from PIL import Image

def extract_text_from_frame(image_path: str) -> str:
    img = Image.open(image_path)
    return pytesseract.image_to_string(img).strip()

# En describe_frame, concatenar:
def describe_frame(image_path: str) -> str:
    caption = blip_caption(image_path)
    ocr_text = extract_text_from_frame(image_path)
    if ocr_text:
        return f"{caption}. Text visible: {ocr_text}"
    return caption
```

#### Cambiar LLM (LLaMA 1B → alternativas locales)

Opciones recomendadas para CPU+GPU local:

| Modelo | Calidad | VRAM | Formato | Notas |
|--------|---------|------|---------|-------|
| `llama-3.2-1b-instruct-q4` | ⭐⭐ | ~1GB | GGUF | actual |
| `llama-3.2-3b-instruct-q4` | ⭐⭐⭐ | ~2GB | GGUF | mejora notable |
| `mistral-7b-instruct-q4` | ⭐⭐⭐⭐ | ~5GB | GGUF | mejor opción CPU |
| `phi-3-mini-4k-instruct-q4` | ⭐⭐⭐ | ~2.5GB | GGUF | muy rápido |
| `qwen2.5-3b-instruct-q4` | ⭐⭐⭐ | ~2GB | GGUF | multilingüe |

**Para cambiar el LLM** solo necesitas:
1. Descargar el nuevo `.gguf` a `/models/`
2. Cambiar `model_path` en `src/transcription/llm.py`
3. Ajustar `n_ctx` si el modelo lo requiere

```python
# llm.py
_model = Llama(
    model_path="/models/mistral-7b-instruct-v0.2-q4_k_m.gguf",
    n_ctx=4096,        # Mistral soporta más contexto
    n_threads=8,
    n_gpu_layers=35,   # Offload capas a GPU (ajusta según tu VRAM)
    verbose=False,
)
```

---

## Métricas explicadas

| Métrica | Rango | Descripción |
|---------|-------|-------------|
| `keyword_hit_rate` | 0–1 | Fracción de conceptos clave presentes en la respuesta |
| `semantic_similarity` | 0–1 | Similitud coseno entre respuesta generada y expected_answer |
| `level_correct` | 0/1 | El nivel de retrieval activado coincide con el esperado |
| `trap_correct` | 0/1 | El sistema admite no saber (para preguntas trampa) |
| `composite_score` | 0–1 | `0.4×khr + 0.4×sem + 0.2×level_or_trap` |
| `latency_s` | segundos | Tiempo total de respuesta de la API |

### Preguntas por nivel

| Nivel | Descripción | Retrieval esperado |
|-------|-------------|-------------------|
| L3 | Generales (tema, género, protagonista) | `video_summary_shortcut` |
| L2 | Escenas o momentos (ventanas de 4 segmentos) | `hierarchical` |
| L1 | Específicas con timestamp | `hierarchical_full` |
| T  | Trampa — info no presente en el trailer | Cualquiera, pero admitir que no sabe |

---

## Tabla de resultados comparativa (rellena manualmente)

| Experimento | Composite | Semantic | KHR | Level Acc | Trap Acc | Latency |
|-------------|-----------|----------|-----|-----------|----------|---------|
| baseline    | —         | —        | —   | —         | —        | —       |
| blip-large  | —         | —        | —   | —         | —        | —       |
| mistral-7b  | —         | —        | —   | —         | —        | —       |
| thresh-020  | —         | —        | —   | —         | —        | —       |

Copia aquí los valores de `summary` de cada `_evaluated.json`.