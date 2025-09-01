from flask import Flask, request, jsonify, render_template
from app.services.diagnose_disease import SymptomDiagnoser
from app.services.otc_service_dynamic import recommend_otc_dynamic
import time

app = Flask(__name__)
DIAGNOSER = SymptomDiagnoser()

@app.route("/")
def home():
    return render_template("design.html")

@app.route("/health")
def health():
    return {"ok": True}

@app.route("/diagnose", methods=["POST"])
def diagnose_api():
    data = request.get_json(force=True) or {}
    text = data.get("text") or data.get("input")  # inputText에서 넘어올 수 있음
    if not text:
        return jsonify({"error": "No input text"}), 400

    results = DIAGNOSER.diagnose_from_input(text, top_k_diseases=5)
    return jsonify(results)

@app.route("/assist", methods=["POST"])
def assist():
    body = request.get_json(force=True) or {}
    text = (body.get("text") or "").strip()
    if not text:
        return jsonify({"error": "Provide 'text'"}), 400

    k_ing = int(body.get("ingredients_per_symptom", 3))
    n_cards = int(body.get("cards_per_ingredient", 3))

    # 질병 진단
    diag_results = DIAGNOSER.diagnose_from_input(text, top_k_diseases=5)

    # OTC 추천
    otc_payload = recommend_otc_dynamic(text,
                                        ingredients_per_symptom=k_ing,
                                        cards_per_ingredient=n_cards)

    return jsonify({
        "diagnosis": diag_results,
        "otc_recommendations": otc_payload,
        "query": text,
        "meta": {"latency_sec": round(time.time(), 3),
                 "disclaimer": "Not medical advice. Check labels and consult professionals."}
    })

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=True)
