from flask import Flask, request, jsonify, render_template
from services.diagnose_disease import SymptomDiagnoser
from services.otc_service_dynamic import recommend_otc_dynamic  # 이름 통일
import time

import os, json, re

BASE_DIR = os.path.dirname(__file__)
app = Flask(__name__, template_folder=os.path.join(BASE_DIR, "design"))

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())

def _dedup_order(xs, k):
    out, seen = [], set()
    for s in xs:
        s = _norm(s)
        if len(s) >= 3 and s not in seen:
            out.append(s); seen.add(s)
            if len(out) >= k: break
    return out

def symptoms_for_disease(disease: str, topk: int = 5):
    """services/disease_symptoms_merged.json에서 질병명으로 증상 Top-k 추출."""
    ds_path = os.path.join(os.path.dirname(__file__), "services", "disease_symptoms_merged.json")
    if not os.path.exists(ds_path):
        return []

    with open(ds_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    target = _norm(disease)

    # dict 형태: {"influenza": [...]} 또는 {"influenza": {"fever": 0.9, ...}}
    if isinstance(data, dict):
        chosen = data.get(target)
        if chosen is None:
            for k in list(data.keys()):
                if _norm(k) == target:
                    chosen = data[k]; break
        if chosen is None:
            return []
        if isinstance(chosen, dict):
            pairs = sorted(((k, float(v)) for k, v in chosen.items()), key=lambda x: -x[1])
            return _dedup_order([k for k, _ in pairs], topk)
        if isinstance(chosen, list):
            buf = []
            for item in chosen:
                if isinstance(item, str):
                    buf.append(item)
                elif isinstance(item, dict):
                    # {"symptom":"...", "score":...} 같은 경우
                    for kk in item.keys():
                        buf.append(kk)
            return _dedup_order(buf, topk)
        return []

    # list 형태: [{"disease":"influenza","symptoms":[...]}] 등
    if isinstance(data, list):
        for rec in data:
            name = _norm(rec.get("disease") or rec.get("name") or rec.get("condition"))
            if name == target:
                syms = rec.get("symptoms") or rec.get("symptom_list") or []
                if isinstance(syms, dict):
                    pairs = sorted(((k, float(v)) for k, v in syms.items()), key=lambda x: -x[1])
                    return _dedup_order([k for k, _ in pairs], topk)
                if isinstance(syms, list):
                    buf = []
                    for item in syms:
                        buf.append(item if isinstance(item, str) else item.get("symptom", ""))
                    return _dedup_order(buf, topk)
                return []
    return []

DIAGNOSER = SymptomDiagnoser()

@app.route("/")
def home():
    return render_template("design.html")  # templates/design.html을 렌더

@app.route("/health")
def health():
    return {"ok": True}

@app.route("/diagnose", methods=["POST"])
def diagnose_api():
    data = request.get_json(silent=True) or {}
    text = data.get("text") or data.get("input")
    if not text:
        return jsonify({"error": "No input text"}), 400
    results = DIAGNOSER.diagnose_from_input(text, top_k_diseases=5)
    return jsonify(results)

@app.route("/assist", methods=["POST"])
def assist():
    body = request.get_json(silent=True) or {}
    text = (body.get("text") or "").strip()
    if not text:
        return jsonify({"error": "Provide 'text'"}), 400

    k_ing = int(body.get("ingredients_per_symptom", 3))
    n_cards = int(body.get("cards_per_ingredient", 3))

    start = time.time()

    # 1) 질병 진단
    diag_results = DIAGNOSER.diagnose_from_input(text, top_k_diseases=5)

    # 2) 진단 결과에서 top disease 하나 뽑고 → 증상 Top-5 만들기
    top_disease = None
    if isinstance(diag_results, dict):
        cands = diag_results.get("candidates") or []
        if cands:
            top_disease = (cands[0].get("name") or cands[0].get("disease") or "").strip() or None

    selected_symptoms = symptoms_for_disease(top_disease, topk=5) if top_disease else []

    # 3) 증상 문자열 만들기(없으면 사용자의 입력을 그대로 사용)
    symptom_text = " and ".join(selected_symptoms) if selected_symptoms else text

    # 4) OTC 추천
    otc_payload = recommend_otc_dynamic(symptom_text,
                                        ingredients_per_symptom=k_ing,
                                        cards_per_ingredient=n_cards)

    elapsed = time.time() - start
    return jsonify({
        "diagnosis": diag_results,
        "top_disease": top_disease,
        "selected_symptoms": selected_symptoms,  # 프론트에서 바로 보여주기 좋음
        "otc_recommendations": otc_payload,
        "query": text,
        "meta": {
            "latency_sec": round(elapsed, 3),
            "disclaimer": "Not medical advice. Check labels and consult professionals."
        }
    })

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=True)
