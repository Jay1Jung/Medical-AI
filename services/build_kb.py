#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
build_kb.py
- access_data.py의 결과(disease_symptoms_merged.json)를 받아
  1) 증상 텍스트 전처리/중복 제거/가중치 합성
  2) 증상-질병 매핑 테이블(symptom -> [(disease, weight), ...]) 생성
  3) 증상 리스트 임베딩 생성(Hugging Face sentence-transformers)
  4) KB 파일 저장: kb_symptoms.json, kb_symptom_embeddings.npz
"""

import json, re, unicodedata, numpy as np
from collections import defaultdict
from sentence_transformers import SentenceTransformer

SRC = "disease_symptoms_merged.json"   # access_data.py가 생성
KB_SYMPTOMS = "kb_symptoms.json"
KB_EMB = "kb_symptom_embeddings.npz"

# 필요시 바꿔서 실험 가능(임상 특화 모델은 나중에 교체)
EMB_MODEL = "all-MiniLM-L6-v2"

# 간단한 정규화 함수: 공백/기호/유니코드 정리
def normalize_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = s.strip()
    # 괄호 안 설명 자주 중복 → 유지/삭제는 취향(지금은 유지)
    # 줄바꿈/연속 공백 정리
    s = re.sub(r"\s+", " ", s)
    # 끝의 마침표 제거(문장형 리스트를 문구로 통일)
    s = re.sub(r"\.+$", "", s)
    return s

def main():
    with open(SRC, "r", encoding="utf-8") as f:
        merged = json.load(f)  
    symptom_to_diseases = defaultdict(list)

    for disease, sym_dict in merged.items():
        for sym_text, w in sym_dict.items():
            t = normalize_text(sym_text)
            if len(t) < 2:
                continue
           
            symptom_to_diseases[t].append((disease, float(w)))

    # 동일 증상 텍스트가 여러 질병에 걸쳐 있을 수 있음 → 그대로 유지
    # 다만 완전히 동일한 (disease, symptom) 중복이 있다면 weight는 최대값 선택
    for sym, lst in symptom_to_diseases.items():
        by_dis = {}
        for d, w in lst:
            by_dis[d] = max(by_dis.get(d, 0.0), w)
        symptom_to_diseases[sym] = sorted(by_dis.items(), key=lambda x: -x[1])

    # 증상 리스트
    symptom_list = list(symptom_to_diseases.keys())

    # 임베딩
    model = SentenceTransformer(EMB_MODEL)
    emb = model.encode(symptom_list, convert_to_numpy=True)  # (N, dim)

    # 저장
    with open(KB_SYMPTOMS, "w", encoding="utf-8") as f:
        json.dump({
            "embedding_model": EMB_MODEL,
            "symptom_list": symptom_list,
            "symptom_to_diseases": symptom_to_diseases
        }, f, ensure_ascii=False, indent=2)

    np.savez_compressed(KB_EMB, embeddings=emb)
    print(f"Saved:\n- {KB_SYMPTOMS}\n- {KB_EMB}\nTotal symptoms: {len(symptom_list)}")

if __name__ == "__main__":
    main()
