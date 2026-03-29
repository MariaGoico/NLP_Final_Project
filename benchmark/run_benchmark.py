#!/usr/bin/env python3
"""
run_benchmark.py
----------------
Lanza todas las preguntas del benchmark contra la API de CineRAG
y guarda los resultados en results_<timestamp>.json.

Uso:
    python run_benchmark.py [--api-url URL] [--questions PATH] [--video-ids-map PATH]

Ejemplo:
    python run_benchmark.py --api-url http://localhost:8001
"""

import argparse
import json
import time
import sys
import os
from datetime import datetime
from pathlib import Path

try:
    import requests
except ImportError:
    print("ERROR: Instala requests  →  pip install requests")
    sys.exit(1)


# ── Argumentos CLI ────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="CineRAG Benchmark Runner")
parser.add_argument("--api-url",       default="http://localhost:8001", help="URL base de la API de transcription")
parser.add_argument("--questions",     default="questions.json",        help="Path al fichero de preguntas")
parser.add_argument("--video-ids-map", default=None,                    help="JSON con {title: video_id} (opcional, se auto-detecta)")
parser.add_argument("--timeout",       default=120, type=int,           help="Timeout por pregunta en segundos")
parser.add_argument("--delay",         default=1.0, type=float,         help="Segundos de espera entre preguntas")
args = parser.parse_args()

API_URL   = args.api_url.rstrip("/")
QUESTIONS = Path(args.questions)
TIMEOUT   = args.timeout
DELAY     = args.delay


# ── Cargar preguntas ──────────────────────────────────────────────────────────
if not QUESTIONS.exists():
    print(f"ERROR: No se encuentra {QUESTIONS}")
    sys.exit(1)

with open(QUESTIONS) as f:
    benchmark = json.load(f)

print(f"✅ Preguntas cargadas: {QUESTIONS}")
print(f"🌐 API: {API_URL}")
print()


# ── Obtener mapa title → video_id desde la API ────────────────────────────────
def fetch_video_map() -> dict:
    """Consulta /videos y devuelve {title: video_id}."""
    try:
        r = requests.get(f"{API_URL}/videos", timeout=10)
        r.raise_for_status()
        videos = r.json()
        return {v["title"]: v["id"] for v in videos}
    except Exception as e:
        print(f"⚠️  No se pudo obtener /videos: {e}")
        return {}

if args.video_ids_map:
    with open(args.video_ids_map) as f:
        video_map = json.load(f)
    print(f"📂 Video IDs cargados desde {args.video_ids_map}")
else:
    video_map = fetch_video_map()
    print(f"📡 Video IDs obtenidos de la API: {video_map}")

print()


# ── Ejecutar benchmark ────────────────────────────────────────────────────────
results = {
    "benchmark_version": benchmark.get("benchmark_version", "unknown"),
    "run_timestamp": datetime.now().isoformat(),
    "api_url": API_URL,
    "video_map": video_map,
    "results": []
}

total_questions = sum(len(t["questions"]) for t in benchmark["trailers"])
processed = 0

for trailer in benchmark["trailers"]:
    title    = trailer["title"]
    video_id = video_map.get(title)

    print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"🎬 {title}  (video_id={video_id})")
    print()

    if video_id is None:
        print(f"  ⚠️  Trailer no procesado — saltando {len(trailer['questions'])} preguntas\n")
        for q in trailer["questions"]:
            results["results"].append({
                "id": q["id"],
                "trailer": title,
                "video_id": None,
                "level": q["level"],
                "question": q["question"],
                "expected_answer": q["expected_answer"],
                "key_concepts": q["key_concepts"],
                "expected_level": q["expected_level"],
                "is_trap": q.get("is_trap", False),
                "status": "skipped",
                "answer": None,
                "sources": [],
                "retrieval_levels_used": {},
                "latency_s": None,
                "error": "Trailer not processed"
            })
        continue

    for q in trailer["questions"]:
        processed += 1
        print(f"  [{processed}/{total_questions}] [{q['level']}] {q['question'][:80]}...")

        payload = {
            "question": q["question"],
            "video_id": video_id,
        }

        t0 = time.time()
        try:
            r = requests.post(f"{API_URL}/query", json=payload, timeout=TIMEOUT)
            latency = round(time.time() - t0, 3)

            if r.status_code == 200:
                data = r.json()
                answer  = data.get("answer", "")
                sources = data.get("sources", [])
                levels  = data.get("retrieval_levels_used", {})
                status  = "ok"
                error   = None
                print(f"  ✓ {latency}s | level_used={levels.get('level_used','?')} | answer={answer[:80]}...")
            else:
                answer  = None
                sources = []
                levels  = {}
                status  = "http_error"
                error   = f"HTTP {r.status_code}: {r.text[:200]}"
                print(f"  ✗ HTTP {r.status_code}")

        except requests.exceptions.Timeout:
            latency = round(time.time() - t0, 3)
            answer  = None
            sources = []
            levels  = {}
            status  = "timeout"
            error   = f"Timeout after {TIMEOUT}s"
            print(f"  ✗ TIMEOUT after {TIMEOUT}s")

        except Exception as e:
            latency = round(time.time() - t0, 3)
            answer  = None
            sources = []
            levels  = {}
            status  = "exception"
            error   = str(e)
            print(f"  ✗ ERROR: {e}")

        results["results"].append({
            "id":                   q["id"],
            "trailer":              title,
            "video_id":             video_id,
            "level":                q["level"],
            "question":             q["question"],
            "expected_answer":      q["expected_answer"],
            "key_concepts":         q["key_concepts"],
            "expected_level":       q["expected_level"],
            "is_trap":              q.get("is_trap", False),
            "trap_reason":          q.get("trap_reason"),
            "status":               status,
            "answer":               answer,
            "sources":              sources,
            "retrieval_levels_used": levels,
            "latency_s":            latency,
            "error":                error,
        })

        time.sleep(DELAY)

    print()


# ── Guardar resultados ────────────────────────────────────────────────────────
timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
out_path   = Path(f"results_{timestamp}.json")

with open(out_path, "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\n{'━'*50}")
print(f"✅ Benchmark completado.")
print(f"   Preguntas lanzadas: {processed} / {total_questions}")
print(f"   Resultados guardados en: {out_path}")
print(f"{'━'*50}\n")
print(f"Siguiente paso → ejecuta evaluate.py con este fichero:")
print(f"  python evaluate.py --results {out_path}")