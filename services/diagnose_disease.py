#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import json, re, numpy as np
from sentence_transformers import SentenceTransformer, util
import torch
import torch.nn.functional as F
import argparse

# KB 파일
KB_SYMPTOMS = "kb_symptoms.json"
KB_EMB = "kb_symptom_embeddings.npz"

# 사용자 입력 분리용(쉼표/and/with 등)
SEP_RE = re.compile(r"[.,;]|(?:\band\b)|(?:\bwith\b)", re.I)

# 아주 단순한 NegEx 패턴(정밀 NegEx/medspaCy는 다음 단계에서 붙이자)
NEG_TRIGGERS = re.compile(
    r"\b(no|not|without|denies?|negative for|free of)\b", re.I
)
ENABLE_IRRELEVANT_FILTER = False
IRRELEVANT_PARTS = {"back","neck","arm","leg","foot","toe","finger","elbow","shoulder","wrist"}  # 필요시 조정

# 점수 하이퍼파라미터
TOPK_SYMPTOMS = 3         # 사용자 문구당 매칭할 KB 증상 개수
ALPHA = 1.0               # 유사도 가중치
BETA = 1.0                # KB weight 가중치(‘most common’ 등의 cue가 반영됨)
MIN_SIM = 0.30           # 너무 낮은 유사도 컷

def _clean_phrase(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s

def _is_negated(s: str) -> bool:
  
    return NEG_TRIGGERS.search(s) is not None

def _contains_irrelevant(s: str) -> bool:
    s_low = s.lower()
    return any(bp in s_low for bp in IRRELEVANT_PARTS)

def _load_kb():
    with open(KB_SYMPTOMS, "r", encoding="utf-8") as f:
        jb = json.load(f)   

    symptom_list = jb['symptom_list']

    symptom_to_diseases = {
        k: [(d, float(w)) for d, w in v]
        for k, v in jb["symptom_to_diseases"].items()
    }
    emb_model_name = jb.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")

    emb_np = np.load(KB_EMB)["embeddings"]    
    emb = torch.from_numpy(emb_np).float() 

    emb = F.normalize(emb, p=2, dim=1)

    return jb, symptom_list, symptom_to_diseases, emb_model_name, emb


class SymptomDiagnoser:
    def __init__(self):
        (
            self.kb,
            self.symptom_list,
            self.symptom_to_diseases,
            self.emb_model_name,
            self.symptom_emb,   
        ) = _load_kb()

        self.model = SentenceTransformer(self.emb_model_name, device="cpu")
        self.symptom_emb = self.symptom_emb.to("cpu")

    def diagnose_from_input(self,user_input: str, top_k_diseases=5):

        raw = [p.strip() for p in SEP_RE.split(user_input)]
        raw = [p for p in raw if p]
        
        phrases = [
            p for p in raw if not _is_negated(p) and not _contains_irrelevant(p)
        ]

        if not phrases:
            return ["no matching symptom has detected"]


        dis_scores = {}
        matched = {}

        for ph in phrases:
            q_text = _clean_phrase(ph)
            q = self.model.encode([q_text], convert_to_tensor=True, normalize_embeddings = True)
            sims = torch.matmul(q, self.symptom_emb.T).squeeze(0)
            k = min(TOPK_SYMPTOMS, sims.shape[0])
            topv, topi = torch.topk(sims,k=k)

            for sim, i in zip(topv.tolist(), topi.tolist()):

                if sim < MIN_SIM:
                    continue
                sym = self.symptom_list[i]
                for disease, w in self.symptom_to_diseases[sym]:
                    score = ALPHA*float(sim) + BETA*float(w)
                    dis_scores[disease] = dis_scores.get(disease, 0.0) + score
                    matched.setdefault(disease, set()).add(sym)

        ranked = sorted(dis_scores.items(), key=lambda x: -x[1])[:top_k_diseases]
        out = [
            {
                "disease": dis,
                "score": round(float(sc), 4),
                "matched_symptoms": sorted(list(matched.get(dis, [])))  
            }
            for dis, sc in ranked
        ]
        if not out:
            return ["no confident match has found"]
    
        return out
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, required=True, help="사용자 증상 서술 문장")
    parser.add_argument("--topk", type=int, default=5, help="출력할 질병 개수")
    args = parser.parse_args()

    diagnoser = SymptomDiagnoser()
    results = diagnoser.diagnose_from_input(args.query, top_k_diseases=args.topk)

    try:
        import pprint
        pprint.pp(results, width=100)
    except Exception:
        print(results)

if __name__ == "__main__":
    main()
