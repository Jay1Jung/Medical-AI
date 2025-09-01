#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
openfda_client.py
- openFDA drug/label API에서 약물 라벨을 검색하는 경량 클라이언트.
- 쿼리 빌더, 안전한 요청, 결과 요약(normalize) 포함.
"""

from __future__ import annotations
import os
import time
import json
import requests
from urllib.parse import quote
from typing import Any, Dict, List, Optional


OPENFDA_LABEL_URL = "https://api.fda.gov/drug/label.json"
DAILYMED_LABEL_FMT = "https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid={setid}"

def _safe_get(url: str, timeout: int = 15) -> Optional[Dict[str, Any]]:
    """단순 GET + 에러/타임아웃 방어."""
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None

def _first(row: Dict[str, Any], path: str) -> Optional[str]:
    """
    중첩 키 접근 유틸.
    예: path="openfda.brand_name" → row["openfda"]["brand_name"][0] (list면 첫 값)
    """
    cur: Any = row
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    if isinstance(cur, list):
        return cur[0] if cur else None
    if isinstance(cur, str):
        return cur
    return None

def build_search_q(
    ingredient: Optional[str] = None,
    brand: Optional[str] = None,
    generic: Optional[str] = None,
    fulltext_contains: Optional[str] = None,
) -> str:
    """
    openFDA label 검색 쿼리 문자열 구성.
    - active_ingredient / brand_name / generic_name / full-text 검색 필드 혼합 지원.
    """
    terms: List[str] = []
    if ingredient:
        terms.append(f'active_ingredient:"{ingredient}"')
    if brand:
        terms.append(f'openfda.brand_name:"{brand}"')
    if generic:
        terms.append(f'openfda.generic_name:"{generic}"')
    if fulltext_contains:
        # indications_and_usage, purpose 같은 필드 전체 텍스트에 대한 부분검색
        terms.append(f'"{fulltext_contains}"')
    if not terms:
        # 최소 하나는 있어야 함
        terms.append('effective_time:[* TO *]')  # 모든 라벨(샘플)
    return quote(" AND ".join(terms))

def search_labels(
    ingredient: Optional[str] = None,
    brand: Optional[str] = None,
    generic: Optional[str] = None,
    fulltext_contains: Optional[str] = None,
    limit: int = 5,
    skip: int = 0,
) -> List[Dict[str, Any]]:
    """
    openFDA label 검색(페이지네이션 지원: limit/skip).
    반환은 요약 필드로 정규화.
    """
    q = build_search_q(ingredient, brand, generic, fulltext_contains)
    url = f"{OPENFDA_LABEL_URL}?search={q}&limit={limit}&skip={skip}"
    data = _safe_get(url)
    if not data:
        return []
    results = data.get("results", [])
    out: List[Dict[str, Any]] = []
    for row in results:
        # 핵심 필드 요약
        brand_name   = _first(row, "openfda.brand_name") or row.get("brand_name")
        generic_name = _first(row, "openfda.generic_name") or row.get("generic_name")
        purpose      = row.get("purpose", [None])
        purpose      = purpose[0] if isinstance(purpose, list) and purpose else purpose
        indications  = row.get("indications_and_usage", [None])
        indications  = indications[0] if isinstance(indications, list) and indications else indications
        warnings     = row.get("warnings", [None])
        warnings     = warnings[0] if isinstance(warnings, list) and warnings else warnings
        spl_set_id   = row.get("spl_set_id", [None])
        spl_set_id   = spl_set_id[0] if isinstance(spl_set_id, list) and spl_set_id else spl_set_id

        out.append({
            "source": "openFDA",
            "brand_name": brand_name,
            "generic_name": generic_name,
            "title": brand_name or generic_name,
            "label_id": row.get("id"),
            "spl_set_id": spl_set_id,
            "label_url": DAILYMED_LABEL_FMT.format(setid=spl_set_id) if spl_set_id else None,
            "purpose": purpose,
            "indications": indications,
            "warnings": warnings,
            "active_ingredients": row.get("active_ingredient"),
        })
    return out

def search_labels_all_pages(
    ingredient: Optional[str],
    max_items: int = 15,
    page_size: int = 5
) -> List[Dict[str, Any]]:
    """
    여러 페이지에 걸쳐 모으는 헬퍼(간단 rate-limit 포함).
    """
    collected: List[Dict[str, Any]] = []
    skip = 0
    while len(collected) < max_items:
        batch = search_labels(ingredient=ingredient, limit=page_size, skip=skip)
        if not batch:
            break
        collected.extend(batch)
        skip += page_size
        time.sleep(0.3)  # 과한 호출 방지
    return collected[:max_items]
