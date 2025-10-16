# -*- coding: utf-8 -*-
import re
from collections import Counter, defaultdict
from .config import (
    LONG_TAIL_MAX_COUNT, LONG_TAIL_BOTTOM_QUANTILE,
    CN_COLOR, EN_COLOR, LATERALITY_CN, LATERALITY_EN, SEVERITY_CN, SEVERITY_EN,
    ANATOMY_CN, ANATOMY_EN, TEMPORAL_CN, TEMPORAL_EN, PUNCTUATION_HINTS,
    NON_ENTITY_TYPE_PREFIXES, PARENT_RULES
)

ROMAN_PATTERN = re.compile(r"\b[ivx]{1,4}\b", re.I)
DEGREE_PATTERN = re.compile(r"(一度|二度|三度|四度|I{1,3}|IV|V|ⅰ|ⅱ|ⅲ|ⅳ|ⅴ)", re.I)
LOCATION_PATTERN = re.compile(r"\b(at|in|on|of)\b", re.I)
WITH_PATTERN = re.compile(r"\bwith\b", re.I)

def smart_title(s: str) -> str:
    parts = s.split(" ")
    out = []
    for p in parts:
        if re.search(r"[A-Za-z]", p):
            out.append(p[:1].upper() + p[1:].lower())
        else:
            out.append(p)
    return " ".join(out)

def normalize_type_label(t: str) -> str:
    t = t.strip()
    t = re.sub(r"[_\-]+", " ", t)
    t = re.sub(r"\s+", " ", t)
    return smart_title(t)

def has_any_token(s, tokens):
    s_lower = s.lower()
    return any(tok.lower() in s_lower for tok in tokens)

def is_fine_grained(type_label: str) -> bool:
    l = type_label.strip()
    tokens = re.split(r"\s+", l)
    long_phrase = len(tokens) >= 4
    has_color = has_any_token(l, CN_COLOR + EN_COLOR)
    has_laterality = has_any_token(l, LATERALITY_CN + LATERALITY_EN)
    has_severity = has_any_token(l, SEVERITY_CN + SEVERITY_EN) or bool(ROMAN_PATTERN.search(l)) or bool(DEGREE_PATTERN.search(l))
    has_anatomy = has_any_token(l, ANATOMY_CN + ANATOMY_EN)
    has_temporal = has_any_token(l, TEMPORAL_CN + TEMPORAL_EN)
    has_punct = any(p in l for p in PUNCTUATION_HINTS)
    has_with = bool(WITH_PATTERN.search(l)) or bool(LOCATION_PATTERN.search(l))
    score = sum([long_phrase, has_color, has_laterality, has_severity, has_anatomy, has_temporal, has_punct, has_with])
    return score >= 2

def suggest_parent(type_label: str) -> str:
    # priority: exact CT word boundary to avoid 'structure' false matches
    if re.search(r"\bCT\b", type_label, re.I):
        return "Medical Imaging Technique"
    tl = type_label.lower()
    for key, parent in PARENT_RULES:
        if key.lower() in tl:
            return parent
    if re.search(r"(wound|injur|burn)", type_label, re.I):
        return "Injury or Poisoning"
    if re.search(r"(procedure|术|操作)", type_label, re.I):
        return "Medical Procedure"
    return "Thing"

def demote_non_entity_types(entity):
    t = entity.get("type","") or ""
    if not t: return
    for pre in NON_ENTITY_TYPE_PREFIXES:
        if t.startswith(pre):
            entity["demoted_reason"] = "non_entity_like_type"
            entity["subTypeLabel"] = t
            entity["type"] = "Thing"
            return

def apply_long_tail_demote(entities):
    # compute counts after normalization
    type_counts = Counter([e.get("type","") for e in entities if e.get("type")])
    counts_sorted = sorted(type_counts.values())
    if counts_sorted:
        quantile_idx = max(0, min(len(counts_sorted)-1, int(len(counts_sorted)*LONG_TAIL_BOTTOM_QUANTILE)))
        quantile_cut = counts_sorted[quantile_idx]
        long_tail_cut = max(LONG_TAIL_MAX_COUNT, quantile_cut)
    else:
        long_tail_cut = LONG_TAIL_MAX_COUNT

    for e in entities:
        t = e.get("type","") or ""
        if not t or t == "Thing": 
            continue
        is_lt = type_counts[t] <= long_tail_cut
        is_fg = is_fine_grained(t)
        if is_lt and is_fg:
            e["demoted_reason"] = "long_tail_fine_grained"
            e["subTypeLabel"] = t
            e["type"] = suggest_parent(t)
    return long_tail_cut

def normalize_entities(entities):
    # normalize type labels
    for e in entities:
        t = e.get("type")
        if t:
            e["type_original"] = t
            e["type"] = normalize_type_label(t)
    # demote non-entity-like
    for e in entities:
        demote_non_entity_types(e)
    # light name trim
    for e in entities:
        if e.get("name"):
            e["name"] = str(e["name"]).strip()