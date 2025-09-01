#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
동적 OTC 서비스:
- symptom_to_ingredients_openfda.json (빌더 산출물)을 로드
- 사용자 입력 증상 텍스트를 정규화
- 증상 키와 일치(또는 유사어 처리 추가 가능)하는 성분 후보를 뽑고
- openfda_client로 해당 성분의 라벨을 몇 개만 붙여 카드 형태로 반환
"""

from __future__ import annotations
import json, re
from typing import Dict, List, Any
from app.services.openfda_client import search_labels_all_pages  # 제품 카드용
SYMPTOM_MAP_PATH = "symptom_to_ingredients_openfda.json"

# 맵 로드
with open(SYMPTOM_MAP_PATH, "r", encoding="utf-8") as f:
    SYM2ING = json.load(f)   # { "sore throat": [{"ingredient": "...", "score": ...}, ...], ... }

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())

def recommend_otc_dynamic(user_text: str, ingredients_per_symptom: int = 3, cards_per_ingredient: int = 3) -> Dict[str, Any]:
    parts = [p.strip() for p in re.split(r"[;,]| and | with ", user_text, flags=re.I) if p.strip()]
    payload = {"query": user_text, "results": []}

    for raw in parts:
        key = _norm(raw)
        # 완전 일치가 없으면 (선택) 임베딩 유사도 매칭 추가 가능
        ing_rows = SYM2ING.get(key, [])[:ingredients_per_symptom]
        ingredients = [r["ingredient"] for r in ing_rows]

        cards = []
        for ing in ingredients:
            labels = search_labels_all_pages(ingredient=ing, max_items=cards_per_ingredient, page_size=cards_per_ingredient)
            for it in labels:
                cards.append({
                    "ingredient": ing,
                    "title": it.get("title"),
                    "brand_name": it.get("brand_name"),
                    "generic_name": it.get("generic_name"),
                    "label_url": it.get("label_url"),
                    "purpose": it.get("purpose"),
                    "indications": it.get("indications"),
                })

        payload["results"].append({
            "symptom": key,
            "ingredients": ing_rows,
            "candidates": cards
        })

    payload["disclaimer"] = (
        "OTC 정보 제공용입니다. 진단/처방이 아니며, 제품 라벨(용법/용량/경고/상호작용)을 확인하세요. "
        "기저질환/임신·수유/복용약이 있으면 전문가와 상의하세요."
    )
    return payload
