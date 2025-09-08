# Symptom & OTC Assistant

Crawls MedlinePlus, builds a symptom→disease knowledge base, optionally maps symptoms to OTC ingredients (openFDA / FDA monographs), and serves a small Flask API. For research/education only — not medical advice.

---

## Why this exists

I wanted a lightweight, reproducible pipeline for:

* pulling symptom lists from reputable public docs (MedlinePlus),
* turning them into a consolidated KB with simple weights,
* doing quick semantic lookup with sentence embeddings,
* and, if needed, suggesting OTC ingredient families with links to DailyMed labels.


---

## What it does

* Crawl MedlinePlus A–Z encyclopedia pages and extract *symptom* sections per article.
* Merge per‑article items into per‑disease dictionaries with rough cue weights (e.g. “most common”, “may”, etc.).
* Build a KB (`kb_symptoms.json`) and sentence embeddings (`kb_symptom_embeddings.npz`).
* Given a free‑text input, rank likely diseases and show which KB symptoms matched.
* (Optional) Build a symptom→OTC‑ingredient map from openFDA and FDA monographs; surface a few DailyMed cards.
* Serve a Flask API with two endpoints: `/diagnose` and `/assist`.

---

## Repository layout

```
services/
  access_data.py                   # MedlinePlus crawler
  build_kb.py                      # Build KB + embeddings
  diagnose_disease.py              # SymptomDiagnoser (semantic matching)
  disease_symptom_expander.py      # Utility for disease→symptom lists
  openfda_client.py                # Minimal openFDA label client
  build_symptom_map_from_openfda.py# Symptom→OTC ingredient map
  otc_service_dynamic.py           # Runtime OTC recommender
app.py                              # Flask app (serves /diagnose, /assist)
```

---

## Setup

Python 3.10+ recommended.

```bash
python -m venv .venv && source .venv/bin/activate  # or use conda
pip install aiohttp async-timeout beautifulsoup4 lxml requests flask numpy torch sentence-transformers scikit-image
```

---

## Quick start

This is the minimal end‑to‑end path. Some steps can take a while.

1. Crawl MedlinePlus (writes `ency_symptoms.json` and `disease_symptoms_merged.json`):

```bash
python services/access_data.py
```

2. Build the KB and embeddings:

```bash
python services/build_kb.py
# → kb_symptoms.json, kb_symptom_embeddings.npz
```

3. Run the API server:

```bash
python app.py
# visit http://127.0.0.1:8000/health
```

4. Try a request:

```bash
curl -X POST http://127.0.0.1:8000/diagnose \
  -H 'Content-Type: application/json' \
  -d '{"text":"fever and dry cough with body aches"}'
```

(Optional) Build the OTC map and hit `/assist`:

```bash
python services/build_symptom_map_from_openfda.py --auto --max-ings 200 --max-labels 25
curl -X POST http://127.0.0.1:8000/assist \
  -H 'Content-Type: application/json' \
  -d '{"text":"sore throat and nasal congestion"}'
```

---

## Configuration notes

**Crawler (`services/access_data.py`)**

* `CONCURRENCY` (default 4) — drop if you see throttling.
* `limit_letters`, `limit_articles_per_letter` — useful for dev slices.
* Cue weights live in `CUE_WEIGHTS`.

**KB / embeddings (`services/build_kb.py`)**

* `EMB_MODEL` defaults to `all-MiniLM-L6-v2`. Swap to a domain model if you like.
* Normalization is intentionally simple; tighten if you see duplicates.

**Diagnosis (`services/diagnose_disease.py`)**

* Scoring knobs: `TOPK_SYMPTOMS`, `MIN_SIM`, `ALPHA`, `BETA`.
* Model/device defaults to CPU; set CUDA if available.

**openFDA**

* Without an API key you’ll hit rate limits quickly. Keep page sizes small.

---

## API

### POST /diagnose

Body:

```json
{ "text": "free‑text symptoms" }
```

Response (example):

```json
[
  { "disease": "influenza", "score": 2.73, "matched_symptoms": ["fever", "dry cough"] },
  { "disease": "common cold", "score": 2.11, "matched_symptoms": ["runny nose"] }
]
```

### POST /assist

Combines diagnosis with OTC ingredient candidates and a few DailyMed links.

Body:

```json
{ "text": "sore throat and nasal congestion" }
```

Abridged response:

```json
{
  "diagnosis": [ { "disease": "common cold", "score": 2.31 } ],
  "top_disease": "common cold",
  "selected_symptoms": ["sore throat", "nasal congestion"],
  "otc_recommendations": { "results": [ { "symptom": "sore throat", "ingredients": [ {"ingredient":"benzocaine","score":3.4} ], "candidates": [ {"title":"...","label_url":"https://dailymed..."} ] } ] }
}
```

---

## Known gaps / TODO

* Keep parsing within section boundaries (avoid `find_all_next` over‑reach).
* Add TF‑IDF‑like downweighting for very generic symptoms.
* Better negation handling (medspaCy/pyConTextNLP).
* Batch encode embeddings and add simple caching.
* A few unit tests for link extraction and section parsing wouldn’t hurt.

---

## License & attribution

MIT. MedlinePlus®, DailyMed and openFDA content is subject to their own terms. This repo is not a medical device and does not provide medical advice.
