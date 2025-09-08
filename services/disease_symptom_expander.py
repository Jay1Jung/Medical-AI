# services/disease_symptom_expander.py
from __future__ import annotations
import os, json, re
from typing import List, Any, Dict

BASE_DIR = os.path.dirname(__file__)
DS_PATH = os.path.join(BASE_DIR, "disease_symptoms_merged.json")

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())

def _dedup_order(xs: List[str], k: int) -> List[str]:
    out, seen = [], set()
    for s in xs:
        s = _norm(s)
        if len(s) >= 3 and s not in seen:
            out.append(s); seen.add(s)
            if len(out) >= k: break
    return out

# 다양한 스키마를 유연하게 처리
def symptoms_for_disease(disease: str, topk: int = 5) -> List[str]:
    if not os.path.exists(DS_PATH):
        return []
    with open(DS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    target = _norm(disease)

    # dict 케이스: { "influenza": {...} or [...] }
    if isinstance(data, dict):
        # 키 정규화 매칭
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
            # 문자열 리스트 또는 [{"symptom": "...", "score": ...}] 혼재
            buf: List[str] = []
            for item in chosen:
                if isinstance(item, str):
                    buf.append(item)
                elif isinstance(item, dict):
                    # 첫 키를 증상으로 가정
                    for kk, vv in item.items():
                        buf.append(kk)
            return _dedup_order(buf, topk)
        return []

    # 리스트 케이스: [ {"disease": "...", "symptoms":[...]} ]
    if isinstance(data, list):
        for rec in data:
            name = _norm(rec.get("disease") or rec.get("name") or rec.get("condition"))
            if name == target:
                syms = rec.get("symptoms") or rec.get("symptom_list") or []
                if isinstance(syms, dict):
                    pairs = sorted(((k, float(v)) for k, v in syms.items()), key=lambda x: -x[1])
                    return _dedup_order([k for k, _ in pairs], topk)
                if isinstance(syms, list):
                    buf: List[str] = []
                    for item in syms:
                        buf.append(item if isinstance(item, str) else item.get("symptom", ""))
                    return _dedup_order(buf, topk)
                return []
    return []
