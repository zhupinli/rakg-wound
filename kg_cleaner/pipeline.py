# -*- coding: utf-8 -*-
import os
import pandas as pd
from collections import defaultdict
from .config import INPUT_JSONS, OUT_DIR
from .io_utils import safe_load_json, save_parquet, save_csv
from .normalization import normalize_entities, apply_long_tail_demote
from .mapping import assign_umls_semantic_type, apply_lexicon_mapping
from .stats import summarize_types

def verbalize_entity(e, rels_by_head, rels_by_tail, max_rels=3):
    pieces = []
    nm = e.get("name","").strip()
    tp = e.get("type","").strip()
    desc = e.get("description","")
    if nm: pieces.append(f"{nm}")
    if tp: pieces.append(f"（类型：{tp}）")
    if desc: pieces.append(desc)
    heads = rels_by_head.get(nm, [])[:max_rels]
    tails = rels_by_tail.get(nm, [])[:max_rels]
    if heads:
        triples = [f"{nm} --{r.get('type','')}--> {r.get('to','')}" for r in heads]
        pieces.append("关联: " + "; ".join(triples))
    if tails:
        triples = [f"{r.get('from','')} --{r.get('type','')}--> {nm}" for r in tails]
        pieces.append("被关联: " + "; ".join(triples))
    attrs = e.get("attributes", {})
    if isinstance(attrs, dict) and attrs:
        kv = []
        for k,v in list(attrs.items())[:5]:
            try:
                kv.append(f"{k}: {v}")
            except Exception:
                continue
        if kv:
            pieces.append("属性: " + "; ".join(kv))
    return "；".join(pieces)

def run_pipeline():
    entities_all, relations_all = [], []

    # load all
    for p in INPUT_JSONS:
        obj = safe_load_json(p)
        ents = obj.get("entities", [])
        rels = obj.get("relations", [])
        if isinstance(ents, list):
            for e in ents:
                if isinstance(e, dict):
                    entities_all.append(e)
        if isinstance(rels, list):
            for r in rels:
                if isinstance(r, dict):
                    relations_all.append({
                        "type": r.get("type") or r.get("relation") or r.get("predicate"),
                        "from": r.get("from") or r.get("head") or r.get("subject"),
                        "to":   r.get("to")   or r.get("tail") or r.get("object"),
                    })

    # idx for verbalization
    from collections import defaultdict
    rels_by_head, rels_by_tail = defaultdict(list), defaultdict(list)
    for r in relations_all:
        if r.get("from"): rels_by_head[r["from"]].append(r)
        if r.get("to"):   rels_by_tail[r["to"]].append(r)

    # step 1: normalization & demotion
    normalize_entities(entities_all)
    long_tail_cut = apply_long_tail_demote(entities_all)

    # step 2: verbalization & semantic type (coarse)
    for e in entities_all:
        e["verbalization"] = verbalize_entity(e, rels_by_head, rels_by_tail)
        assign_umls_semantic_type(e)

    # step 3: concept mapping via lexicons (optional)
    apply_lexicon_mapping(entities_all)

    # outputs
    os.makedirs(OUT_DIR, exist_ok=True)
    ents_df = pd.DataFrame(entities_all)
    rels_df = pd.DataFrame(relations_all)
    save_parquet(ents_df, f"{OUT_DIR}/out_entities.parquet")
    save_parquet(rels_df, f"{OUT_DIR}/out_relations.parquet")

    # summary
    type_df = summarize_types(entities_all)
    summary = pd.DataFrame({
        "metric": ["entities","relations","unique_types_after","long_tail_cut"],
        "value": [len(ents_df), len(rels_df), type_df.shape[0], long_tail_cut]
    })
    save_csv(summary, f"{OUT_DIR}/summary.csv")
    save_csv(type_df, f"{OUT_DIR}/type_counts.csv")

    # unmapped list
    unmapped = []
    for e in entities_all:
        if not e.get("umls_semantic_type") or (not e.get("umls_cui") and not e.get("snomed_id")):
            unmapped.append({
                "name": e.get("name",""),
                "type": e.get("type",""),
                "umls_semantic_type": e.get("umls_semantic_type",""),
                "desc": e.get("description","") or "",
                "verbalization": e.get("verbalization","") or ""
            })
    save_csv(pd.DataFrame(unmapped), f"{OUT_DIR}/unmapped_entities.csv")

    return {
        "entities_path": f"{OUT_DIR}/out_entities.parquet",
        "relations_path": f"{OUT_DIR}/out_relations.parquet",
        "summary_path": f"{OUT_DIR}/summary.csv",
        "type_counts_path": f"{OUT_DIR}/type_counts.csv",
        "unmapped_path": f"{OUT_DIR}/unmapped_entities.csv",
    }