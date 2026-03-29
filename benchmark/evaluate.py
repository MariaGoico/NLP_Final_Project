#!/usr/bin/env python3
"""
evaluate.py
-----------
Evalúa los resultados de run_benchmark.py y genera:
  - Un JSON con métricas por pregunta
  - Un report HTML interactivo con tabla comparativa

Uso:
    python evaluate.py --results results_YYYYMMDD_HHMMSS.json [--label "baseline"]

Métricas calculadas:
  - keyword_hit_rate    : fracción de key_concepts que aparecen en la respuesta
  - semantic_similarity : similitud coseno entre respuesta generada y expected_answer
  - level_correct       : 1 si el nivel de retrieval coincide con expected_level, 0 si no
  - trap_correct        : para preguntas trampa, 1 si el sistema admite no saber
  - latency_s           : tiempo de respuesta en segundos
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

try:
    from sentence_transformers import SentenceTransformer, util
    import numpy as np
except ImportError:
    print("ERROR: Instala dependencias →  pip install sentence-transformers numpy")
    sys.exit(1)


# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="CineRAG Benchmark Evaluator")
parser.add_argument("--results", required=True, help="JSON de resultados generado por run_benchmark.py")
parser.add_argument("--label",   default="run",  help="Nombre descriptivo de esta ejecución (p.ej. 'baseline', 'blip2', 'mistral')")
parser.add_argument("--model",   default="all-MiniLM-L6-v2", help="Modelo sentence-transformers para similitud semántica")
args = parser.parse_args()

RESULTS_PATH = Path(args.results)
LABEL        = args.label

if not RESULTS_PATH.exists():
    print(f"ERROR: No se encuentra {RESULTS_PATH}")
    sys.exit(1)

with open(RESULTS_PATH) as f:
    data = json.load(f)

print(f"📂 Resultados: {RESULTS_PATH}")
print(f"🏷️  Label: {LABEL}")
print(f"🤖 Cargando modelo de similitud: {args.model}...")
model = SentenceTransformer(args.model)
print("✅ Modelo cargado\n")


# ── Funciones de métrica ──────────────────────────────────────────────────────
TRAP_PHRASES = [
    "could not find", "not found", "not mentioned", "cannot find",
    "don't know", "do not know", "not in the video", "no information",
    "unable to find", "not available", "not stated", "not specified",
    "not clear", "not provided"
]

def keyword_hit_rate(answer: str, key_concepts: list[str]) -> float:
    if not answer or not key_concepts:
        return 0.0
    answer_lower = answer.lower()
    hits = sum(1 for kw in key_concepts if kw.lower() in answer_lower)
    return round(hits / len(key_concepts), 3)

def semantic_similarity(answer: str, expected: str) -> float:
    if not answer or not expected:
        return 0.0
    emb_a = model.encode(answer,   convert_to_tensor=True)
    emb_e = model.encode(expected, convert_to_tensor=True)
    return round(float(util.cos_sim(emb_a, emb_e)), 3)

def level_correct(result: dict) -> int | None:
    """Compara el nivel activado con el expected_level."""
    levels_used = result.get("retrieval_levels_used", {})
    level_used  = levels_used.get("level_used", "")
    expected    = result.get("expected_level", "")
    if not level_used or not expected:
        return None

    # Mapeo flexible entre los valores del sistema y L1/L2/L3
    mapping = {
        "hierarchical_full":     "L1",
        "hierarchical":          "L2",
        "video_summary_shortcut": "L3",
        "L1": "L1", "L2": "L2", "L3": "L3",
    }
    actual_level = mapping.get(level_used, level_used)
    return 1 if actual_level == expected else 0

def trap_correct(result: dict) -> int | None:
    """Para preguntas trampa: 1 si la respuesta admite no saber."""
    if not result.get("is_trap"):
        return None
    answer = (result.get("answer") or "").lower()
    return 1 if any(phrase in answer for phrase in TRAP_PHRASES) else 0

def composite_score(khr: float, sem: float, lc, tc) -> float:
    """
    Score compuesto 0-1:
      - 40% keyword hit rate
      - 40% semantic similarity
      - 20% level correct (si aplica)
    Para trampas: también suma 20% si tc=1.
    """
    score = 0.4 * khr + 0.4 * sem
    if lc is not None:
        score += 0.2 * lc
    elif tc is not None:
        score += 0.2 * tc
    return round(score, 3)


# ── Evaluar ───────────────────────────────────────────────────────────────────
evaluated = []
skipped   = 0

for i, r in enumerate(data["results"]):
    print(f"[{i+1}/{len(data['results'])}] {r['id']} ...", end=" ", flush=True)

    if r["status"] != "ok" or r["answer"] is None:
        skipped += 1
        evaluated.append({**r, "metrics": None})
        print("SKIP")
        continue

    khr  = keyword_hit_rate(r["answer"], r["key_concepts"])
    sem  = semantic_similarity(r["answer"], r["expected_answer"])
    lc   = level_correct(r)
    tc   = trap_correct(r)
    comp = composite_score(khr, sem, lc, tc)

    metrics = {
        "keyword_hit_rate":    khr,
        "semantic_similarity": sem,
        "level_correct":       lc,
        "trap_correct":        tc,
        "composite_score":     comp,
    }
    evaluated.append({**r, "metrics": metrics})
    print(f"khr={khr} sem={sem} comp={comp}")

print(f"\n✅ Evaluación completada. Skipped: {skipped}")


# ── Guardar métricas JSON ─────────────────────────────────────────────────────
out_json = RESULTS_PATH.with_name(RESULTS_PATH.stem + "_evaluated.json")
output   = {
    "label":             LABEL,
    "source_results":    str(RESULTS_PATH),
    "eval_timestamp":    datetime.now().isoformat(),
    "summary": {},
    "by_level": {},
    "results":           evaluated,
}

# Resumen global
valid = [e for e in evaluated if e["metrics"]]
if valid:
    def mean(lst): return round(sum(lst) / len(lst), 3)
    output["summary"] = {
        "n_questions":         len(data["results"]),
        "n_evaluated":         len(valid),
        "n_skipped":           skipped,
        "avg_keyword_hit_rate":    mean([e["metrics"]["keyword_hit_rate"]    for e in valid]),
        "avg_semantic_similarity": mean([e["metrics"]["semantic_similarity"] for e in valid]),
        "avg_composite_score":     mean([e["metrics"]["composite_score"]     for e in valid]),
        "avg_latency_s":           mean([e["latency_s"] for e in valid if e["latency_s"]]),
        "level_accuracy":          mean([e["metrics"]["level_correct"] for e in valid if e["metrics"]["level_correct"] is not None]),
        "trap_accuracy":           mean([e["metrics"]["trap_correct"]  for e in valid if e["metrics"]["trap_correct"]  is not None]),
    }

    # Por nivel
    for lvl in ["L1", "L2", "L3", "T"]:
        lvl_results = [e for e in valid if e["level"] == lvl]
        if lvl_results:
            output["by_level"][lvl] = {
                "n": len(lvl_results),
                "avg_keyword_hit_rate":    mean([e["metrics"]["keyword_hit_rate"]    for e in lvl_results]),
                "avg_semantic_similarity": mean([e["metrics"]["semantic_similarity"] for e in lvl_results]),
                "avg_composite_score":     mean([e["metrics"]["composite_score"]     for e in lvl_results]),
            }

with open(out_json, "w") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)
print(f"💾 Métricas guardadas en: {out_json}")


# ── Generar HTML ──────────────────────────────────────────────────────────────
def score_color(v):
    """Verde si alto, rojo si bajo."""
    if v is None: return "#666"
    if v >= 0.7:  return "#1db954"
    if v >= 0.4:  return "#f5a623"
    return "#e50914"

def render_bar(v, max_v=1.0):
    if v is None: return "—"
    pct = min(100, int(v / max_v * 100))
    color = score_color(v)
    return f'<div style="background:#222;border-radius:4px;height:14px;width:100px;display:inline-block;vertical-align:middle"><div style="background:{color};width:{pct}%;height:100%;border-radius:4px"></div></div> <span style="color:{color};font-weight:600">{v}</span>'

rows_html = ""
for e in evaluated:
    m = e.get("metrics") or {}
    level_badge_colors = {"L1": "#3a86ff", "L2": "#8338ec", "L3": "#ff006e", "T": "#fb5607"}
    lc = level_badge_colors.get(e["level"], "#555")
    trap_tag = ' <span style="background:#fb5607;color:#fff;border-radius:4px;padding:1px 6px;font-size:0.7rem">TRAP</span>' if e.get("is_trap") else ""

    answer_short = (e.get("answer") or e.get("error") or "—")[:140]
    level_used = e.get("retrieval_levels_used", {}).get("level_used", "—")

    rows_html += f"""
    <tr>
      <td><span style="background:{lc};color:#fff;border-radius:4px;padding:2px 8px;font-size:0.75rem;font-weight:700">{e['level']}</span>{trap_tag}</td>
      <td style="color:#aaa;font-size:0.8rem">{e['trailer'][:22]}</td>
      <td style="max-width:260px;font-size:0.82rem">{e['question']}</td>
      <td style="max-width:260px;font-size:0.78rem;color:#ccc;font-style:italic">{answer_short}…</td>
      <td style="font-size:0.78rem;color:#aaa">{level_used}</td>
      <td>{render_bar(m.get('keyword_hit_rate'))}</td>
      <td>{render_bar(m.get('semantic_similarity'))}</td>
      <td>{render_bar(m.get('composite_score'))}</td>
      <td style="color:#aaa;font-size:0.8rem">{e.get('latency_s','—')}s</td>
    </tr>"""

summary = output.get("summary", {})
by_level = output.get("by_level", {})

level_rows = ""
for lvl, stats in by_level.items():
    level_rows += f"""
    <tr>
      <td><b>{lvl}</b></td>
      <td>{stats['n']}</td>
      <td>{render_bar(stats['avg_keyword_hit_rate'])}</td>
      <td>{render_bar(stats['avg_semantic_similarity'])}</td>
      <td>{render_bar(stats['avg_composite_score'])}</td>
    </tr>"""

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>CineRAG Benchmark — {LABEL}</title>
<style>
  body {{ background:#0d0d1a; color:#e0e0e0; font-family:'Segoe UI',sans-serif; margin:0; padding:2rem; }}
  h1 {{ background:linear-gradient(135deg,#e50914,#ff6b35); -webkit-background-clip:text; -webkit-text-fill-color:transparent; font-size:2.2rem; margin-bottom:0.2rem; }}
  h2 {{ color:#e50914; margin-top:2rem; }}
  .meta {{ color:#888; font-size:0.85rem; margin-bottom:2rem; }}
  .cards {{ display:flex; gap:1.5rem; flex-wrap:wrap; margin-bottom:2rem; }}
  .card {{ background:#1a1a2e; border-radius:12px; padding:1.2rem 1.8rem; min-width:140px; }}
  .card .val {{ font-size:2rem; font-weight:800; }}
  .card .lbl {{ color:#888; font-size:0.8rem; margin-top:0.2rem; }}
  table {{ border-collapse:collapse; width:100%; font-size:0.85rem; }}
  th {{ background:#1a1a2e; color:#e50914; padding:10px 12px; text-align:left; position:sticky; top:0; }}
  td {{ padding:8px 12px; border-bottom:1px solid #1a1a2e; vertical-align:top; }}
  tr:hover td {{ background:#1a1a2e88; }}
  .section {{ background:#111122; border-radius:12px; padding:1.5rem; margin-bottom:2rem; overflow-x:auto; }}
  .badge {{ background:#e50914; color:#fff; border-radius:20px; padding:2px 12px; font-size:0.75rem; font-weight:600; }}
</style>
</head>
<body>
<h1>🎬 CineRAG Benchmark</h1>
<div class="meta">
  Label: <span class="badge">{LABEL}</span> &nbsp;|&nbsp;
  Source: {RESULTS_PATH.name} &nbsp;|&nbsp;
  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
</div>

<div class="cards">
  <div class="card"><div class="val" style="color:#1db954">{summary.get('avg_composite_score','—')}</div><div class="lbl">Avg Composite Score</div></div>
  <div class="card"><div class="val" style="color:#3a86ff">{summary.get('avg_semantic_similarity','—')}</div><div class="lbl">Avg Semantic Similarity</div></div>
  <div class="card"><div class="val" style="color:#8338ec">{summary.get('avg_keyword_hit_rate','—')}</div><div class="lbl">Avg Keyword Hit Rate</div></div>
  <div class="card"><div class="val" style="color:#f5a623">{summary.get('level_accuracy','—')}</div><div class="lbl">Level Accuracy</div></div>
  <div class="card"><div class="val" style="color:#fb5607">{summary.get('trap_accuracy','—')}</div><div class="lbl">Trap Accuracy</div></div>
  <div class="card"><div class="val" style="color:#aaa">{summary.get('avg_latency_s','—')}s</div><div class="lbl">Avg Latency</div></div>
  <div class="card"><div class="val">{summary.get('n_evaluated','—')}</div><div class="lbl">Questions Evaluated</div></div>
  <div class="card"><div class="val" style="color:#e50914">{summary.get('n_skipped','—')}</div><div class="lbl">Skipped (not processed)</div></div>
</div>

<h2>📊 By Retrieval Level</h2>
<div class="section">
<table>
  <thead><tr><th>Level</th><th>N</th><th>Keyword Hit Rate</th><th>Semantic Similarity</th><th>Composite Score</th></tr></thead>
  <tbody>{level_rows}</tbody>
</table>
</div>

<h2>📋 All Questions</h2>
<div class="section">
<table>
  <thead>
    <tr>
      <th>Level</th><th>Trailer</th><th>Question</th><th>Answer (preview)</th>
      <th>Level Used</th><th>Keyword Hit</th><th>Semantic Sim</th><th>Composite</th><th>Latency</th>
    </tr>
  </thead>
  <tbody>{rows_html}</tbody>
</table>
</div>

</body>
</html>"""

out_html = RESULTS_PATH.with_name(RESULTS_PATH.stem + f"_{LABEL}_report.html")
with open(out_html, "w", encoding="utf-8") as f:
    f.write(html)

print(f"🌐 Report HTML guardado en: {out_html}")
print(f"\n{'━'*50}")
print(f"RESUMEN — {LABEL}")
print(f"{'━'*50}")
for k, v in summary.items():
    print(f"  {k:35s}: {v}")