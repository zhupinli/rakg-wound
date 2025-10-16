# -*- coding: utf-8 -*-
import os, re
import pandas as pd
from difflib import SequenceMatcher
from .config import CANON_PARENT_TO_UMLS, UMLS_ST_PATH, UMLS_CUI_PATH, SNOMED_PATH

def load_lexicon(path, key_col):
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path)
    d = {}
    for _, r in df.iterrows():
        k = str(r.get(key_col, "")).strip().lower()
        if k:
            d.setdefault(k, []).append({c: r[c] for c in df.columns if c != key_col})
    return d

def best_lexicon_match(name, lex, threshold=0.95):
    if not name: return (0.0, None)
    key = name.strip().lower()
    if key in lex:
        return (1.0, lex[key][0])
    best_score, best_rec = 0.0, None
    for k, recs in lex.items():
        s = SequenceMatcher(None, key, k).ratio()
        if s > best_score:
            best_score = s
            best_rec = recs[0]
    if best_score >= threshold:
        return (best_score, best_rec)
    return (0.0, None)

def assign_umls_semantic_type(entity):
    parent = entity.get("type","")
    if parent in CANON_PARENT_TO_UMLS:
        entity["umls_semantic_type"] = CANON_PARENT_TO_UMLS[parent]

def apply_lexicon_mapping(entities):
    umls_st_lex = load_lexicon(UMLS_ST_PATH, "name")
    umls_cui_lex = load_lexicon(UMLS_CUI_PATH, "name")
    snomed_lex   = load_lexicon(SNOMED_PATH,  "name")
    for e in entities:
        name = e.get("name","")
        # semantic type override
        sc, rec = best_lexicon_match(name, umls_st_lex, threshold=0.95)
        if rec:
            e["umls_semantic_type"] = rec.get("semantic_type", rec.get("tui", e.get("umls_semantic_type")))
            e["umls_semantic_type_source"] = "lexicon-umls-st"
        # umls cui
        sc2, rec2 = best_lexicon_match(name, umls_cui_lex, threshold=0.95)
        if rec2:
            e["umls_cui"] = rec2.get("cui")
            e["umls_pref_label"] = rec2.get("pref_label")
            e["umls_cui_source"] = "lexicon-umls-cui"
        # snomed
        sc3, rec3 = best_lexicon_match(name, snomed_lex, threshold=0.95)
        if rec3:
            e["snomed_id"] = rec3.get("snomed_id")
            e["snomed_fsn"] = rec3.get("fsn")
            e["snomed_source"] = "lexicon-snomed"