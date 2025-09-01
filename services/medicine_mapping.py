#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
openFDA 라벨에서 증상/징후를 추출해
"증상 -> [{'ingredient': ..., 'score': ...}, ...]" 맵을 생성/저장.

- 교육/개발용 샘플: rate-limit, 캐싱, 예외처리 포함(기본형)
- 실행:
    python build_symptom_map_from_openfda.py
- 산출물:
    symptom_to_ingredients_openfda.json
"""

from __future__ import annotations
import json, time, re, math, os, hashlib
from collections import defaultdict, Counter
from typing import List, Dict, Any, Optional
import requests
from urllib.parse import quote
import argparse
from bs4 import BeautifulSoup

OPENFDA_LABEL_URL = "https://api.fda.gov/drug/label.json"
OUT_OPENFDA = "symptom_to_ingredients_openfda.json"
OUT_FINAL   = "symptom_to_ingredients_final.json"
CACHE_DIR   = ".cache_openfd"
os.makedirs(CACHE_DIR, exist_ok=True)

# ====== (1) 대상 성분 목록: 여기에 성분을 계속 추가하면 맵이 커짐 ======
OTC_INGREDIENTS = [
    # ===== 진통/해열 =====
    "acetaminophen", "ibuprofen", "naproxen", "aspirin", "ketoprofen",
    "choline salicylate", "diflunisal", "indomethacin", "meloxicam", "piroxicam",
    "etodolac", "nabumetone", "diclofenac", "sulindac", "mefenamic acid",
    "celecoxib", "valdecoxib", "lumiracoxib", "phenazopyridine", "flurbiprofen",

    # ===== 기침/가래/호흡기 =====
    "dextromethorphan", "guaifenesin", "codeine", "benzonatate", "carbocisteine",
    "ambroxol", "bromhexine", "acetylcysteine", "erdosteine", "levodropropizine",
    "noscapine", "prenoxdiazine", "butamirate", "cloperastine", "tiotropium",
    "ipratropium", "salbutamol", "terbutaline", "formoterol", "montelukast",

    # ===== 코막힘/비충혈/비염 =====
    "phenylephrine", "pseudoephedrine", "oxymetazoline", "xylometazoline", "naphazoline",
    "ephedrine", "tetrahydrozoline", "levocabastine", "azelastine", "olopatadine",
    "fluticasone", "budesonide", "triamcinolone", "mometasone", "beclomethasone",
    "ciclesonide", "dexchlorpheniramine", "clemastine", "acrivastine", "ebastine",

    # ===== 항히스타민 (알레르기) =====
    "loratadine", "cetirizine", "fexofenadine", "diphenhydramine", "doxylamine",
    "chlorpheniramine", "hydroxyzine", "levocetirizine", "desloratadine", "rupatadine",
    "bilastine", "mizolastine", "cyproheptadine", "mequitazine", "brompheniramine",
    "olopatadine", "ketotifen", "epinastine", "acrivastine", "clemastine",

    # ===== 위장/소화제 =====
    "bismuth subsalicylate", "loperamide", "omeprazole", "famotidine", "calcium carbonate",
    "magnesium hydroxide", "aluminum hydroxide", "simethicone", "ranitidine", "sucralfate",
    "cimetidine", "nizatidine", "pantoprazole", "rabeprazole", "esomeprazole",
    "domperidone", "metoclopramide", "cisapride", "mosapride", "itopride",

    # ===== 변비/장운동 =====
    "polyethylene glycol", "senna", "bisacodyl", "docusate sodium", "lactulose",
    "sodium picosulfate", "castor oil", "magnesium citrate", "glycerin suppository", "sorbitol",
    "methylcellulose", "psyllium", "guar gum", "bran fiber", "calcium polycarbophil",

    # ===== 멀미/구역/오심 =====
    "meclizine", "dimenhydrinate", "ondansetron", "scopolamine", "metoclopramide",
    "prochlorperazine", "promethazine", "chlorpromazine", "haloperidol", "domperidone",
    "trimethobenzamide", "palonosetron", "granisetron", "aprepitant", "fosaprepitant",

    # ===== 피부/가려움/국소 =====
    "hydrocortisone", "benzocaine", "lidocaine", "pramoxine", "camphor",
    "calamine", "menthol", "neomycin", "polymyxin b", "bacitracin",
    "mupirocin", "fusidic acid", "clotrimazole", "miconazole", "ketoconazole",
    "terbinafine", "tolnaftate", "ciclopirox", "nystatin", "amorolfine",

    # ===== 항진균/항바이러스 (국소) =====
    "acyclovir", "penciclovir", "docosanol", "zovirax", "valacyclovir",
    "econazole", "isoconazole", "sertaconazole", "naftifine", "butenafine",

    # ===== 비타민/미네랄/보충제 =====
    "vitamin a", "vitamin b1", "vitamin b2", "vitamin b6", "vitamin b12",
    "vitamin c", "vitamin d", "vitamin e", "vitamin k", "folic acid",
    "niacin", "pantothenic acid", "biotin", "iron sulfate", "calcium citrate",
    "zinc gluconate", "magnesium oxide", "potassium chloride", "selenium", "iodine",

    # ===== 기타 자주 쓰이는 OTC 성분 =====
    "chlorhexidine", "hydrogen peroxide", "povidone-iodine", "salicylic acid", "sulfur",
    "resorcinol", "phenol", "witch hazel", "benzalkonium chloride", "cetylpyridinium chloride",
    "carboxymethylcellulose", "polyvinyl alcohol", "hyaluronic acid", "propylene glycol", "glycerin",
    "white petrolatum", "lanolin", "mineral oil", "dimethicone", "allantoin",
    "urea", "titanium dioxide", "zinc oxide", "aloe vera", "tea tree oil",

    # ===== 추가 보충 =====
    "melatonin", "valerian root", "ginkgo biloba", "ginseng", "echinacea",
    "glucosamine", "chondroitin", "coenzyme q10", "omega-3 fatty acids", "lutein",
    "lysine", "inositol", "taurine", "caffeine", "theobromine",
    "nicotine polacrilex", "varenicline", "bupropion", "ashwagandha", "turmeric"
]

def _cache_path(url):
    h = hashlib.sha256(url.encode()).hexdigest()[:24]
    return os.path.join(CACHE_DIR, f"{h}.json")

def safe_get(url: str, timeout: int = 20) -> Optional[Dict[str, Any]]:
    """간단 파일 캐시 + GET."""
    p = _cache_path(url)
    if os.path.exists(p):
        try:
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f)
        return data
    except Exception:
        return None
def harvest_otc_ingredients(max_items: int = 300, timeout: int = 15) -> List[str]:
    """
    OTC(HUMAN) 라벨에서 활성성분 빈도 집계(파셋 카운트) 후 성분 리스트 추출.
    콤보 성분은 분해(+;/,/' with ').
    """
    q = quote('openfda.product_type:"HUMAN OTC DRUG"')
    url = f"{OPENFDA_LABEL_URL}?search={q}&count=active_ingredient.exact"
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        data = r.json()
    except Exception:
        return []

    bucket: List[str] = []
    for row in data.get("results", []):
        term = row.get("term") or ""
        parts = re.split(r"(?:\s+with\s+)|[+;/,]", term, flags=re.I)
        for p in parts:
            t = re.sub(r"\s+", " ", p.strip().lower())
            if t:
                bucket.append(t)
    # 중복 제거 + 상위 max_items
    return list(dict.fromkeys(bucket))[:max_items]

def _fetch_once(q: str, page_size: int, skip: int) -> Optional[Dict[str, Any]]:
    url = f"{OPENFDA_LABEL_URL}?search={quote(q)}&limit={page_size}&skip={skip}"
    return safe_get(url)

def fetch_labels_for_ingredient(
    ingredient: str,
    max_items: int = 30,
    page_size: int = 10,
) -> List[Dict[str, Any]]:
    """
    active_ingredient로 수집(OTC 필터 포함) → 너무 적으면 substance_name.exact 폴백.
    spl_set_id 기준으로 라벨 중복 제거.
    """
    def _collect(q: str) -> List[Dict[str, Any]]:
        collected = []
        skip = 0
        seen_setids = set()
        while len(collected) < max_items:
            data = _fetch_once(q, page_size, skip)
            if not data or not data.get("results"):
                break
            for row in data["results"]:
                setid_list = row.get("spl_set_id") or []
                setid = setid_list[0] if isinstance(setid_list, list) and setid_list else None
                if setid and setid in seen_setids:
                    continue
                if setid:
                    seen_setids.add(setid)
                collected.append(row)
                if len(collected) >= max_items:
                    break
            skip += page_size
            time.sleep(0.25)  # rate limit 배려
        return collected

    base_q = f'active_ingredient:"{ingredient}" AND openfda.product_type:"HUMAN OTC DRUG"'
    rows = _collect(base_q)

    # 결과가 너무 적으면 substance_name으로도 시도
    if len(rows) < max_items // 3:
        fb_q = f'openfda.substance_name.exact:"{ingredient}" AND openfda.product_type:"HUMAN OTC DRUG"'
        rows += _collect(fb_q)

    # 최종 cap
    return rows[:max_items]
# ====== (2) NER로 증상/징후 span 추출 ======
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
NER_MODEL = "d4data/biomedical-ner-all"  # 범용 바이오 NER 체크포인트
_tok = AutoTokenizer.from_pretrained(NER_MODEL)
_ner = pipeline("token-classification",
                model=AutoModelForTokenClassification.from_pretrained(NER_MODEL),
                tokenizer=_tok,
                aggregation_strategy="simple",
                device=-1)

STOP = {"to","it","dr","high","red","ex","ur","di"}
def _ok_term(s: str) -> bool:
    s = s.strip().lower()
    if len(s) < 3: return False
    if s in STOP: return False
    if s.startswith("##"): return False
    if not re.search(r"[a-z]", s): return False
    return True

def _is_symptom_label(lbl: str) -> bool:
    lbl = (lbl or "").lower()
    # 모델별 라벨명에 맞춰 조정(예: symptom/sign/finding/clinical_finding 등)
    return ("symptom" in lbl) or ("sign" in lbl) or ("finding" in lbl)

def extract_symptoms(text: str) -> List[str]:
    """텍스트에서 증상/징후 엔티티만 추출 후 정규화."""
    if not text or not text.strip():
        return []
    ents = _ner(text)
    spans = []
    for e in ents:
        if _is_symptom_label(e.get("entity_group","")):
            span = text[e["start"]:e["end"]]
            if span:
                spans.append(span)
    # 간단 정규화
    out = []
    for s in spans:
        s2 = s.lower().strip()
        s2 = re.sub(r"\s+", " ", s2)
        s2 = s2.strip(" .;:,•·–-")
        if len(s2) >= 2:
            out.append(s2)
    out = [s2 for s2 in out if _ok_term(s2)]
    return out

def build_symptom_to_ingredient_map(
    ingredients: List[str],
    max_labels_per_ing: 50,
    page_size: int = 10,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    (증상, 성분) 빈도를 라벨 단위로 집계 → 점수화(score=freq*idf) → 상위 정렬.
    """
    sym_to_ing_counts = defaultdict(Counter)
    ing_doc_counts = defaultdict(int)

    fields_of_interest = ("indications_and_usage", "purpose", "warnings", "contraindications")

        

    for idx, ing in enumerate(ingredients, 1):
        print(f"[{idx}/{len(ingredients)}] {ing} ...", flush=True)
        labels = fetch_labels_for_ingredient(ing, max_items=30, page_size=10)
        ing_doc_counts[ing] = len(labels)

        for row in labels:
            # 텍스트 필드 모으기
            texts = []
            for key in fields_of_interest:
                v = row.get(key)
                if isinstance(v, list) and v:
                    texts.append(v[0])
                elif isinstance(v, str):
                    texts.append(v)
            # 한 라벨에서 유니크 증상 집합
            seen_sym = set()
            for t in texts:
                for s in extract_symptoms(t):
                    seen_sym.add(s)
            for s in seen_sym:
                sym_to_ing_counts[s][ing] += 1

        # 점수 계산(간단 idf 보정)
    symptom_map = {}
    for sym, cnts in sym_to_ing_counts.items():
        items = []
        for ing, f in cnts.items():
            df = max(1, ing_doc_counts.get(ing, 1))
            idf = math.log(1 + (max_labels_per_ing / df))
            score = f * idf
            items.append({"ingredient": ing, "freq": int(f), "score": round(score, 4)})
        items.sort(key=lambda x: -x["score"])
        symptom_map[sym] = items

    return symptom_map
    
MONO_INDEX_URL_DEFAULT = "https://www.fda.gov/drugs/over-counter-otc-drug-monograph-process/otc-monographs-fda"
HDRS = {"User-Agent":"Mozilla/5.0 (research; educational use)"}
def http_get(url: str) -> str:
    r = requests.get(url, headers=HDRS, timeout=20)
    r.raise_for_status()
    return r.text

def list_monograph_links(index_url: str) -> List[str]:
    html = http_get(index_url)
    soup = BeautifulSoup(html, "html.parser")
    links = []
    for a in soup.select("a[href]"):
        href = a.get("href") or ""
        text = (a.get_text(" ", strip=True) or "").lower()
        # 대략적으로 모노그래프 상세로 이어지는 링크만 수집(필요시 조정)
        if ("monograph" in href.lower() or "otc-" in href.lower()
            or "final" in text or "nonprescription" in text):
            if href.startswith("/"):
                href = "https://www.fda.gov" + href
            if href.startswith("http"):
                links.append(href)
    return list(dict.fromkeys(links))

def _normalize_text(s: str) -> str:
    s = re.sub(r"\s+", " ", (s or "")).strip()
    return s

def extract_monograph_sections(soup: BeautifulSoup) -> Dict[str, List[str]]:
    out = {"indications": [], "ingredients": []}
    # 헤더 탐색
    for h in soup.find_all(["h2","h3","h4"]):
        ht = (h.get_text(" ", strip=True) or "").lower()
        bucket = None
        if any(k in ht for k in ["indications", "indication", "uses", "use"]):
            bucket = "indications"
        elif "active" in ht and "ingredient" in ht:
            bucket = "ingredients"
        if not bucket: 
            continue
        # 섹션 다음 형제들 긁기
        for sib in h.find_all_next():
            if sib.name in ["h2","h3","h4"]:
                break
            if sib.name in ["p","li","td","th"]:
                t = _normalize_text(sib.get_text(" ", strip=True))
                if t:
                    out[bucket].append(t)
    return out

def tokenize_indications(text: str) -> List[str]:
    # 규정 문구를 억지로 증상 키워드로 쪼개는 러프 규칙(원하면 사전/동의어 추가)
    txt = text.lower()
    parts = re.split(r"[.;]| for | of | including | such as | due to | from ", txt)
    toks = []
    for p in parts:
        p = p.strip()
        p = re.sub(r"\b(temporary|minor|the|relief|symptoms|associated|and|or|with|from)\b", " ", p)
        p = re.sub(r"[^a-z ]", " ", p)
        p = re.sub(r"\s+", " ", p).strip()
        if len(p) >= 3:
            toks.append(p)
    keys = set()
    for t in toks:
        ws = t.split()
        for n in (1,2,3):
            for i in range(len(ws)-n+1):
                cand = " ".join(ws[i:i+n]).strip()
                if len(cand) >= 3:
                    keys.add(cand)
    return sorted(keys)

def normalize_ingredient_name(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\([^)]*\)", "", s)  # 괄호 내용 제거
    s = re.sub(r"[^a-z0-9 \-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def build_map_monograph(index_url: str, sleep: float = 0.5, limit: Optional[int] = None) -> Dict[str, List[Dict[str, Any]]]:
    links = list_monograph_links(index_url)
    if limit: links = links[:limit]
    sym2ing: Dict[str, Counter] = defaultdict(Counter)

    for url in links:
        try:
            html = http_get(url); time.sleep(sleep)
            soup = BeautifulSoup(html, "html.parser")
            sections = extract_monograph_sections(soup)
            ings_raw = sections.get("ingredients") or []
            inds_raw = sections.get("indications") or []
            if not ings_raw or not inds_raw:
                continue

            # ingredient 후보 만들기(리스트/표에서 토큰화)
            ing_set: set[str] = set()
            for ln in ings_raw:
                # 쉼표/구분자 분리
                parts = re.split(r"[,\u2022;]| and ", ln, flags=re.I)
                for p in parts:
                    v = normalize_ingredient_name(p)
                    if len(v) >= 3 and not v.endswith("percent"):
                        ing_set.add(v)
            if not ing_set:
                continue

            sym_keys: set[str] = set()
            for t in inds_raw:
                for k in tokenize_indications(t):
                    sym_keys.add(k)

            if ing_set and sym_keys:
                for s in sym_keys:
                    for ing in ing_set:
                        sym2ing[s][ing] += 1
                print(f"[MONO] {url} ings={len(ing_set)} syms={len(sym_keys)}")
        except Exception as e:
            print(f"[MONO:SKIP] {url}: {e}")

    out: Dict[str, List[Dict[str, Any]]] = {}
    for sym, cnts in sym2ing.items():
        items = [{"ingredient": ing, "freq": int(f), "score": int(f), "source": "monograph"}
                 for ing, f in sorted(cnts.items(), key=lambda x: -x[1])]
        out[sym] = items
    return out

# ===== Merge maps =====
def merge_maps(
    m_openfda: Dict[str, List[Dict[str, Any]]],
    m_mono: Dict[str, List[Dict[str, Any]]],
    alpha: float = 1.0,
    beta: float = 1.2,
) -> Dict[str, List[Dict[str, Any]]]:
    keys = set(m_openfda.keys()) | set(m_mono.keys())
    final: Dict[str, List[Dict[str, Any]]] = {}
    for k in keys:
        agg: Dict[str, Dict[str, Any]] = {}
        for src_list, w, src in ((m_openfda.get(k, []), alpha, "openfda"),
                                 (m_mono.get(k, []),    beta,  "monograph")):
            for row in src_list:
                ing = normalize_ingredient_name(row["ingredient"])
                add = w * float(row.get("score", 0))
                if ing not in agg:
                    agg[ing] = {"ingredient": ing, "score": 0.0, "sources": set()}
                agg[ing]["score"] += add
                agg[ing]["sources"].add(src)
        items = [{"ingredient": ing, "score": round(v["score"], 4),
                  "source": "+".join(sorted(v["sources"]))}
                 for ing, v in agg.items()]
        items.sort(key=lambda x: -x["score"])
        final[k] = items
    return final


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--auto", action="store_true", help="openFDA에서 OTC 성분 자동 수집 사용")
    p.add_argument("--max-ings", type=int, default=300, help="자동 수집 시 최대 성분 수")
    p.add_argument("--max-labels", type=int, default=30, help="성분당 최대 라벨 수")
    p.add_argument("--page-size", type=int, default=10, help="openFDA 페이지 크기")
    args = p.parse_args()
    try:
        auto_list = harvest_otc_ingredients(args.max_ings) if args.auto else []
    except Exception:
        auto_list = []

    INGREDIENTS = auto_list or OTC_INGREDIENTS
    print(f"[SETUP] ingredients: {len(INGREDIENTS)} (auto={bool(auto_list)})")

    symptom_map = build_symptom_to_ingredient_map(
        ingredients=INGREDIENTS,
        max_labels_per_ing=args.max_labels,
        page_size=args.page_size,
    )

    with open(OUT_OPENFDA, "w", encoding="utf-8") as f:
        json.dump(symptom_map, f, ensure_ascii=False, indent=2)
    print(f"[DONE] symptoms: {len(symptom_map)} → saved to {OUT_OPENFDA}")


if __name__ == "__main__":
    main()

