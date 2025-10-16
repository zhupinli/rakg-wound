# Pipeline for steps 1–3: cleaning/normalization, verbalization, and coarse UMLS/SNOMED mapping hooks.
# It reads your KG JSONs, normalizes types, demotes long-tail fine-grained types to attributes,
# generates per-entity verbalizations, and attempts semantic-type assignment using rules + (optional) lexicons.
#
# Outputs:
#   /mnt/data/kg_pipeline/out_entities.parquet         (normalized entities with new fields)
#   /mnt/data/kg_pipeline/out_relations.parquet        (passed-through relations)
#   /mnt/data/kg_pipeline/summary.csv                  (counts before/after)
#   /mnt/data/kg_pipeline/unmapped_entities.csv        (entities lacking semantic type or concept mapping)
#
# Optional inputs (if provided, improve mapping):
#   /mnt/data/lexicons/umls_semantic_types.csv         (name -> UMLS semantic type label or TUI)
#   /mnt/data/lexicons/umls_concepts.csv               (name/alias -> CUI, preferred label, TUI)
#   /mnt/data/lexicons/snomed_concepts.csv             (name/alias -> SNOMED CT ID, FSN, semantic tag)
#
import os, re, json, math
from collections import Counter, defaultdict
import pandas as pd
from difflib import SequenceMatcher

# ---------------------------
# CONFIG
# ---------------------------
INPUT_JSONS = [
    "/mnt/data/kg_result_modified_normalize_debug_v2_chapter_4.json",
    "/mnt/data/kg_result_modified_normalize_debug_v2_chapter_6.json",
    "/mnt/data/kg_result_modified_normalize_debug_v2_chapter_7.json",
]
OUT_DIR = "/mnt/data/kg_pipeline"
os.makedirs(OUT_DIR, exist_ok=True)

# Long-tail / fine-grained detection parameters (same spirit as previous tool)
LONG_TAIL_MAX_COUNT = 5
LONG_TAIL_BOTTOM_QUANTILE = 0.2

# Regex/patterns for cleaning non-entity "types"
NON_ENTITY_TYPE_HINTS = re.compile(r"^(图|表|Section|Chapter|Graph|Table|Figure|Fig\.|Image|Diagram|Illustration)\b", re.I)

# Canonical parent suggestion rules (also used for UMLS semantic type coarse mapping)
PARENT_RULES = [
    (re.compile(r"烧伤|burn", re.I), "Burn Injury"),
    (re.compile(r"伤口|创(面|伤)|wound", re.I), "WoundFinding"),
    (re.compile(r"清创|debrid", re.I), "Medical Procedure"),
    (re.compile(r"包扎|敷料|dressing", re.I), "Medical Procedure"),
    (re.compile(r"感染|sepsis|cellulitis", re.I), "Infection"),
    (re.compile(r"疼痛|pain|symptom", re.I), "Sign or Symptom"),
    (re.compile(r"药|drug|medicat|药物", re.I), "Drug"),
    (re.compile(r"影像|MRI|\bCT\b|磁共振|成像|X[- ]?ray", re.I), "Medical Imaging Technique"),
    (re.compile(r"组织|tissue|解剖|anatom", re.I), "Anatomical Structure"),
    (re.compile(r"评估|assessment|评分|分级|scale|method", re.I), "Assessment Method"),
    (re.compile(r"手术|surgery|术|procedure|操作|技术", re.I), "Medical Procedure"),
    (re.compile(r"病|disease|综合征|症候群", re.I), "Disease or Syndrome"),
    (re.compile(r"蛋白|protein", re.I), "Amino Acid, Peptide, or Protein"),
    (re.compile(r"器械|器具|instrument|device|引流|管", re.I), "Medical Device"),
    (re.compile(r"wound|injur|burn", re.I), "Injury or Poisoning"),
    (re.compile(r"procedure|术|操作", re.I), "Medical Procedure"),
]

# Coarse mapping: canonical parent -> UMLS semantic type label (TUI label; you can swap to TUI codes if you have them)
CANON_PARENT_TO_UMLS = {
    "Burn Injury": "Injury or Poisoning",
    "WoundFinding": "Finding",
    "Medical Procedure": "Therapeutic or Preventive Procedure",
    "Infection": "Disease or Syndrome",
    "Sign or Symptom": "Sign or Symptom",
    "Drug": "Pharmacologic Substance",
    "Medical Imaging Technique": "Diagnostic Procedure",
    "Anatomical Structure": "Body Part, Organ, or Organ Component",
    "Assessment Method": "Intellectual Product",
    "Disease or Syndrome": "Disease or Syndrome",
    "Amino Acid, Peptide, or Protein": "Amino Acid, Peptide, or Protein",
    "Medical Device": "Medical Device",
}

# Optional external lexicons (if present)
LEX_DIR = "/mnt/data/lexicons"
UMLS_ST_PATH = os.path.join(LEX_DIR, "umls_semantic_types.csv")     # columns: name, semantic_type or tui
UMLS_CUI_PATH = os.path.join(LEX_DIR, "umls_concepts.csv")          # columns: name, cui, pref_label, semantic_type(or tui)
SNOMED_PATH  = os.path.join(LEX_DIR, "snomed_concepts.csv")         # columns: name, snomed_id, fsn, semantic_tag

# Heuristics reused from earlier for fine-grained detection
CN_COLOR = ["红", "黄", "黑", "白", "绿", "褐", "青", "紫", "棕"]
EN_COLOR = ["red", "yellow", "black", "white", "green", "brown", "blue", "purple"]
LATERALITY_CN = ["左", "右", "双侧"]
LATERALITY_EN = ["left", "right", "bilateral"]
SEVERITY_CN = ["浅", "深", "轻度", "中度", "重度", "ⅰ", "ⅱ", "ⅲ", "ⅳ", "ⅴ", "I", "II", "III", "IV", "V"]
SEVERITY_EN = ["mild", "moderate", "severe", "i", "ii", "iii", "iv", "v"]
ANATOMY_CN = ["头", "颈", "胸", "腹", "背", "臂", "前臂", "手", "指", "腕", "股", "大腿", "小腿", "踝", "足", "趾", "耳", "鼻", "口", "面", "颧", "臀", "腹股沟"]
ANATOMY_EN = ["head","neck","chest","thorax","abdomen","back","arm","forearm","hand","finger","wrist","thigh","leg","ankle","foot","toe","ear","nose","mouth","face","groin"]
TEMPORAL_EN = ["acute","chronic","subacute"]
TEMPORAL_CN = ["急性","慢性","亚急性"]
PUNCTUATION_HINTS = [":", "：", "/", "\\", "(", ")", "（", "）", ","]

ROMAN_PATTERN = re.compile(r"\b[ivx]{1,4}\b", re.I)
DEGREE_PATTERN = re.compile(r"(一度|二度|三度|四度|I{1,3}|IV|V|ⅰ|ⅱ|ⅲ|ⅳ|ⅴ)", re.I)
LOCATION_PATTERN = re.compile(r"\b(at|in|on|of)\b", re.I)
WITH_PATTERN = re.compile(r"\bwith\b", re.I)

# ---------------------------
# Helpers
# ---------------------------
def safe_load_json(path):
    if not os.path.exists(path):
        return {"entities": [], "relations": []}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"entities": [], "relations": []}

def normalize_type_label(t: str) -> str:
    t = t.strip()
    t = re.sub(r"[_\-]+", " ", t)           # underscores/hyphens -> space
    t = re.sub(r"\s+", " ", t)              # collapse spaces
    # Title case for Latin words; keep CJK as-is
    def smart_title(s):
        parts = s.split(" ")
        out = []
        for p in parts:
            if re.search(r"[A-Za-z]", p):
                out.append(p[:1].upper() + p[1:].lower())
            else:
                out.append(p)
        return " ".join(out)
    t = smart_title(t)
    return t

def has_any_token(s, tokens):
    s_lower = s.lower()
    for tok in tokens:
        if tok.lower() in s_lower:
            return True
    return False

def is_fine_grained(type_label):
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

def suggest_parent(type_label):
    for patt, parent in PARENT_RULES:
        if patt.search(type_label):
            return parent
    return "Thing"

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

def best_lexicon_match(name, lex, threshold=0.90):
    """Try exact/casefold match then fuzzy; return (score, record) or (0, None)."""
    if not name: return (0.0, None)
    key = name.strip().lower()
    if key in lex:
        # exact key match: choose first
        return (1.0, lex[key][0])
    # fuzzy search among keys
    best_score, best_rec = 0.0, None
    for k, recs in lex.items():
        s = SequenceMatcher(None, key, k).ratio()
        if s > best_score:
            best_score = s
            best_rec = recs[0]
    if best_score >= threshold:
        return (best_score, best_rec)
    return (0.0, None)

def verbalize_entity(e, rels_by_head, rels_by_tail, max_rels=3):
    pieces = []
    nm = e.get("name","").strip()
    tp = e.get("type","").strip()
    desc = e.get("description","")
    if nm: pieces.append(f"{nm}")
    if tp: pieces.append(f"（类型：{tp}）")
    if desc: pieces.append(desc)
    # add up to max_rels outgoing & incoming triples
    heads = rels_by_head.get(nm, [])[:max_rels]
    tails = rels_by_tail.get(nm, [])[:max_rels]
    if heads:
        triples = [f"{nm} --{r.get('type','')}--> {r.get('to','')}" for r in heads]
        pieces.append("关联: " + "; ".join(triples))
    if tails:
        triples = [f"{r.get('from','')} --{r.get('type','')}--> {nm}" for r in tails]
        pieces.append("被关联: " + "; ".join(triples))
    # attributes
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

# ---------------------------
# Load and merge data
# ---------------------------
entities_all = []
relations_all = []
for p in INPUT_JSONS:
    obj = safe_load_json(p)
    ents = obj.get("entities", [])
    rels = obj.get("relations", [])
    # normalize format for relations (from/to or head/tail)
    norm_rels = []
    for r in rels if isinstance(rels, list) else []:
        # if not isinstance(r, dict): 
        #     continue
        # rr = {
        #     "type": r.get("type") or r.get("relation") or r.get("predicate"),
        #     "from": r.get("from") or r.get("head") or r.get("subject"),
        #     "to":   r.get("to")   or r.get("tail") or r.get("object"),
        # }
        rr = {
            "type": r[1],
            "from": r[0],
            "to": r[2],
            "desc": r[3]
        }
        norm_rels.append(rr)
    relations_all.extend(norm_rels)
    for e in ents if isinstance(ents, list) else []:
        if isinstance(e, dict):
            entities_all.append(e)

# indices for quick neighbor lookup
rels_by_head = defaultdict(list)
rels_by_tail = defaultdict(list)
for r in relations_all:
    if r.get("from"):
        rels_by_head[r["from"]].append(r)
    if r.get("to"):
        rels_by_tail[r["to"]].append(r)

# ---------------------------
# Step 1: Cleaning & normalization
# ---------------------------
# 1) normalize type labels and name
for e in entities_all:
    t = e.get("type")
    if t:
        e["type_original"] = t
        t_norm = normalize_type_label(t)
        e["type"] = t_norm

# 2) remove non-entity types (reassign to Thing + mark)
for e in entities_all:
    t = e.get("type","")
    if t and NON_ENTITY_TYPE_HINTS.search(t):
        e["demoted_reason"] = "non_entity_like_type"
        e["subTypeLabel"] = t
        e["type"] = "Thing"

# 3) long-tail & fine-grained demotion
#    compute counts on normalized types
type_counts = Counter([e.get("type","") for e in entities_all if e.get("type")])
counts_sorted = sorted(type_counts.values())
if counts_sorted:
    quantile_idx = max(0, min(len(counts_sorted)-1, int(len(counts_sorted)*LONG_TAIL_BOTTOM_QUANTILE)))
    quantile_cut = counts_sorted[quantile_idx]
    long_tail_cut = max(LONG_TAIL_MAX_COUNT, quantile_cut)
else:
    long_tail_cut = LONG_TAIL_MAX_COUNT

for e in entities_all:
    t = e.get("type","")
    if not t: 
        continue
    is_lt = type_counts[t] <= long_tail_cut
    is_fg = is_fine_grained(t)
    if is_lt and is_fg and e.get("type") not in ("Thing",):
        e["demoted_reason"] = "long_tail_fine_grained"
        e["subTypeLabel"] = t
        # suggest parent
        parent = suggest_parent(t)
        e["type"] = parent

# 4) normalize entity names lightly (trim spaces)
for e in entities_all:
    if e.get("name"):
        e["name"] = str(e["name"]).strip()

# ---------------------------
# Step 2: Verbalization & UMLS semantic type (coarse)
# ---------------------------
# Add verbalization
for e in entities_all:
    e["verbalization"] = verbalize_entity(e, rels_by_head, rels_by_tail)

# Coarse semantic type via parent mapping
for e in entities_all:
    parent = e.get("type","")
    umls_sem_type = CANON_PARENT_TO_UMLS.get(parent)
    if umls_sem_type:
        e["umls_semantic_type"] = umls_sem_type

# Lexicon-based overrides (optional)
umls_st_lex = load_lexicon(UMLS_ST_PATH, "name")
umls_cui_lex = load_lexicon(UMLS_CUI_PATH, "name")
snomed_lex   = load_lexicon(SNOMED_PATH,  "name")

def try_lexicons(e):
    name = e.get("name","")
    # exact/fuzzy semantic type
    sc, rec = best_lexicon_match(name, umls_st_lex, threshold=0.95)
    if rec:
        e["umls_semantic_type"] = rec.get("semantic_type", rec.get("tui", e.get("umls_semantic_type")))
        e["umls_semantic_type_source"] = "lexicon-umls-st"
    # concepts
    sc2, rec2 = best_lexicon_match(name, umls_cui_lex, threshold=0.95)
    if rec2:
        e["umls_cui"] = rec2.get("cui")
        e["umls_pref_label"] = rec2.get("pref_label")
        e["umls_cui_source"] = "lexicon-umls-cui"
    sc3, rec3 = best_lexicon_match(name, snomed_lex, threshold=0.95)
    if rec3:
        e["snomed_id"] = rec3.get("snomed_id")
        e["snomed_fsn"] = rec3.get("fsn")
        e["snomed_source"] = "lexicon-snomed"

for e in entities_all:
    try_lexicons(e)

# ---------------------------
# Step 3: Concept mapping (hooks already tried above)
# ---------------------------
# We already attempted CUI/SNOMED mapping using lexicons. Anything unmapped is exported for manual/active learning.
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

# ---------------------------
# Save outputs
# ---------------------------
ents_df = pd.DataFrame(entities_all)
rels_df = pd.DataFrame(relations_all)

ents_path = os.path.join(OUT_DIR, "out_entities.parquet")
rels_path = os.path.join(OUT_DIR, "out_relations.parquet")
summary_path = os.path.join(OUT_DIR, "summary.csv")
unmapped_path = os.path.join(OUT_DIR, "unmapped_entities.csv")

ents_df.to_parquet(ents_path, index=False)
rels_df.to_parquet(rels_path, index=False)

summary_df = pd.DataFrame({
    "metric": ["entities", "relations", "unique_types_after", "long_tail_cut"],
    "value": [len(ents_df), len(rels_df), ents_df["type"].nunique(), long_tail_cut]
})
summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

pd.DataFrame(unmapped).to_csv(unmapped_path, index=False, encoding="utf-8-sig")

# Show to user
display_dataframe_to_user("KG Pipeline Summary", summary_df.head(50))
display_dataframe_to_user("Sample of normalized entities", ents_df.head(50))

print("Artifacts saved:")
print(" - Entities:", ents_path)
print(" - Relations:", rels_path)
print(" - Summary:", summary_path)
print(" - Unmapped:", unmapped_path)
