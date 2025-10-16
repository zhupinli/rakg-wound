# Code to identify "long-tail fine-grained types" and propose parent types
# It scans your uploaded KG JSON files, counts type frequencies, flags fine-grained types by heuristics,
# and suggests coarse parent types via simple rules. Results are saved to CSVs.
#
# You can re-run this cell after tweaking the CONFIG section.
import json, re, os, math
from collections import Counter, defaultdict
import pandas as pd
from difflib import SequenceMatcher
# from caas_jupyter_tools import display_dataframe_to_user

# ---------------------------
# CONFIG (feel free to edit)
# ---------------------------
INPUT_JSONS = [
    # "result/import_data/statistics/merged_entities_relations.json",
    "result/import_data/kg_result_modified_normalize_debug_v2_chapter_4.json",
    "result/import_data/kg_result_modified_normalize_debug_v2_chapter_6.json",
    "result/import_data/kg_result_modified_normalize_debug_v2_chapter_7.json",
]

# Long-tail threshold: mark types with frequency <= MAX_COUNT or in the bottom QUANTILE
LONG_TAIL_MAX_COUNT = 5           # mark any type with count <= 5 as long-tail
LONG_TAIL_BOTTOM_QUANTILE = 0.2   # additionally, bottom 20% by frequency are long-tail


# Heuristics for "fine-grained" detection
# These terms capture attributes that shouldn't live in the type label itself (color, side, degree, location words, etc.)
CN_COLOR = ["红", "黄", "黑", "白", "绿", "褐", "青", "紫", "棕"]
EN_COLOR = ["red", "yellow", "black", "white", "green", "brown", "blue", "purple"]
LATERALITY_CN = ["左", "右", "双侧", "两侧"]
LATERALITY_EN = ["left", "right", "bilateral"]
SEVERITY_CN = ["浅", "深", "轻度", "中度", "重度", "ⅰ", "ⅱ", "ⅲ", "ⅳ", "ⅴ", "I", "II", "III", "IV", "V"]
SEVERITY_EN = ["mild", "moderate", "severe", "i", "ii", "iii", "iv", "v"]
ANATOMY_CN = ["头", "颈", "胸", "腹", "背", "臂", "前臂", "手", "指", "腕", "股", "大腿", "小腿", "踝", "足", "趾", "耳", "鼻", "口", "面", "颧", "臀", "腹股沟"]
ANATOMY_EN = ["head","neck","chest","thorax","abdomen","back","arm","forearm","hand","finger","wrist","thigh","leg","ankle","foot","toe","ear","nose","mouth","face","groin"]
TEMPORAL_EN = ["acute","chronic","subacute"]
TEMPORAL_CN = ["急性","慢性","亚急性"]
NOISE_HINTS = ["图", "表", "Figure", "Fig.", "Table", "Diagram", "Illustration"]  # likely non-entity "types" or meta
PUNCTUATION_HINTS = [":", "：", "/", "\\", "(", ")", "（", "）", ","]

# Regex/patterns for cleaning non-entity "types"
NON_ENTITY_TYPE_HINTS = re.compile(r"^(图|表|Figure|Fig\.|Image|Diagram|Illustration)\b", re.I)
ROMAN_PATTERN = re.compile(r"\b[ivx]{1,4}\b", re.I)
DEGREE_PATTERN = re.compile(r"(一度|二度|三度|四度|I{1,3}|IV|V|ⅰ|ⅱ|ⅲ|ⅳ|ⅴ)", re.I)
LOCATION_PATTERN = re.compile(r"\b(at|in|on|of)\b", re.I)  # English prepositions
WITH_PATTERN = re.compile(r"\bwith\b", re.I)

# Keyword-to-parent rules (very lightweight; tune for your domain)
# PARENT_RULES = [
#     (re.compile(r"烧伤|burn", re.I), "Burn Injury"),
#     (re.compile(r"伤口|创(面|伤)|wound", re.I), "Wound"),
#     (re.compile(r"清创|debrid", re.I), "Medical Procedure"),
#     (re.compile(r"包扎|敷料|dressing", re.I), "Medical Procedure"),
#     (re.compile(r"感染|sepsis|cellulitis", re.I), "Infection"),
#     (re.compile(r"疼痛|pain", re.I), "Symptom"),
#     (re.compile(r"药|drug|medicat|药物", re.I), "Drug"),
#     (re.compile(r"影像|MRI|\bCT\b|磁共振|成像|X[- ]?ray", re.I), "Medical Imaging Technique"),
#     (re.compile(r"组织|tissue|解剖|anatom", re.I), "Anatomical Structure"),
#     (re.compile(r"评估|assessment|评分|分级|scale|method", re.I), "Assessment Method"),
#     (re.compile(r"手术|surgery|术|procedure|操作|技术", re.I), "Medical Procedure"),
#     (re.compile(r"病|disease|综合征|综合症|综合徵|症", re.I), "Disease"),
#     (re.compile(r"蛋白|protein", re.I), "Protein"),
#     (re.compile(r"指标|indicator|index|metric", re.I), "Medical Indicator"),
#     (re.compile(r"器械|器具|instrument|device|引流|管", re.I), "Medical Device"),
# ]

PARENT_RULES = [
    # 1. Burn Injury
    (re.compile(r"(烧伤|烫伤|电击伤|化学烧伤|放射性?烧伤|吸入性?损伤|inhalation injury|burn(?:s)?|\bTBSA\b|\b%TBSA\b)", re.I), "Burn Injury"),

    # 2. Wound（更聚焦术语）
    (re.compile(r"(伤口|创(?:面|口|伤)|裂伤|切(口|开伤)|撕脱伤|擦伤|刺伤|挫裂伤|枪伤|咬伤|wound|laceration|incision|avulsion|abrasion|puncture|contusion|gunshot)", re.I), "Wound"),

    # 3. Medical Procedure（细化常见操作，避免泛匹配“术”）
    (re.compile(
        r"(清创(?:术)?|冲洗(?:术)?|消毒处置|止血(?:术)?|缝(?:合|线)|减张缝合|负压(?:封闭)?引流|VSD|NPWT|植皮|取皮|皮瓣|烧痂切开|筋膜切开|换药|引流(?:管)?|异体移植|皮移植|debrid(?:e|ement)|irrigation|hemostasis|sutur(?:e|ing)|skin graft|flap|fasciotomy|escharotomy|NPWT|VAC)",
        re.I),
     "Medical Procedure"),

    # 4. Infection
    (re.compile(r"(感染|脓肿|蜂窝织炎|坏死性筋膜炎|败血症|sepsis|cellulitis|abscess|necrotizing fasciitis|bacteremi(?:a)|SIRS)", re.I), "Infection"),

    # 5. Symptom
    (re.compile(r"(疼痛|痛|红肿热痛|异味|臭味|渗出|发热|寒战|剧痛|疼痛评分|pain|tenderness|erythema|warmth|purulen|exudate|fever|chill)", re.I), "Symptom"),

    # 6. Drug（覆盖抗生素/镇痛/局麻/外用）
    (re.compile(
        r"(药(物|品)|抗生素|止痛药|外用药|麻醉药|消毒剂|碘伏|氯己定|青霉素|头孢|万古霉素|左氧氟沙星|甲硝唑|阿莫西林|布洛芬|对乙酰氨基酚|利多卡因|药膏|ointment|antibiotic|analgesic|NSAID|opioid|acetaminophen|paracetamol|lidocaine|povidone[- ]?iodine|chlorhexidine)",
        re.I),
     "Drug"),

    # 7. Medical Imaging Technique（缩写更稳健）
    (re.compile(r"(影像|成像|X[- ]?ray|X线|放射|磁共振|MRI|\bCT\b|超声|B超|US(?:G)?|ultrasound|radiograph(?:y)?|CTA|MRA)", re.I), "Medical Imaging Technique"),

    # 8. Anatomical Structure（常见部位+左右侧/近远端）
    (re.compile(
        r"(组织|解剖|皮肤|皮下|筋膜|肌(肉|腱)|神经|血管|骨(质|皮质)?|关节|腔隙|会阴|头皮|面部|颈部|胸部|腹部|背部|腰部|上肢|下肢|手(部)?|足(部)?|指|趾|创缘|近端|远端|解剖平面|tissue|derm(?:is|al)|subcut(?:is|aneous)|fascia|muscle|tendon|nerve|vessel|artery|vein|bone|periosteum|joint|perineum|scalp|face|hand|foot|digit|proximal|distal)",
        re.I),
     "Anatomical Structure"),

    # 9. Assessment Method（加入具体评分/面积）
    (re.compile(r"(评估|评分|分级|量表|方法|scale|score|assessment|rule of nines|九分法|Lund[- ]?Browder|TBSA|ISS|GCS|SOFA|qSOFA|NEWS)", re.I), "Assessment Method"),

    # 10. Disease（避免“术/症”泛匹配，列常见合并症）
    (re.compile(
        r"(病(史)?|综合征|综合症|症候群|糖尿病|外周血管病|免疫抑制|骨髓炎|气性坏疽|溃疡|瘢痕|挛缩|disease|syndrome|diabetes|peripheral arterial disease|immunosuppression|osteomyelitis|gangrene|ulcer|scar|contracture)",
        re.I),
     "Disease"),

    # 11. Protein / Biomarker（补充常见实验室蛋白/指标词干）
    (re.compile(r"(蛋白|白蛋白|C[- ]?反应蛋白|CRP|降钙素原|PCT|肌酐|乳酸|protei?n)", re.I), "Protein"),

    # 12. Medical Indicator（更明确到“数值/测量/生命体征/面积”）
    (re.compile(r"(指标|指征|数值|测量|体温|脉搏|呼吸|血压|血氧|体表面积|长度|宽度|深度|周径|温度|HR|RR|BP|SpO2|metric|index|indicator)", re.I), "Medical Indicator"),

    # 13. Medical Device（强化常见器械/耗材）
    (re.compile(
        r"(器械|器具|器材|耗材|引流|引流管|负压装置|夹板|缝线|缝合针|止血带|探子|镊子|剪刀|敷料|贴|绷带|夹板|夹闭|夹子|device|instrument|drain|catheter|tube|staple|suture|needle|tourniquet|dressing|bandage|splint|NPWT|VAC)",
        re.I),
     "Medical Device"),

    ### new
    # A) Severity / Burn Depth
    (re.compile(r"(I度|一度|II度|二度|深二度|III度|三度|(?:first|second|third)[- ]degree burn)", re.I), "Severity"),

    # B) Mechanism / Cause
    (re.compile(r"(火焰|蒸汽|热液|热金属|化学(灼|烧)伤|酸|碱|氢氟酸|电(击|烧)|高压|低压|爆炸|辐射|锐器|钝器|夹砸|挤压|动物?咬伤|烫|thermal|chemical burn|acid|alkali|HF|electric(al)?|blast|radiation|knife|blunt)", re.I), "Injury Mechanism"),

    # C) Prophylaxis / Immunization
    (re.compile(r"(破伤风|类毒素|Tdap?|TIG|狂犬(疫苗|免疫球蛋白)|rabies|tetanus(?: toxoid)?|immune globulin)", re.I), "Prophylaxis"),
]

# ---------------------------
# Helpers
# ---------------------------
def safe_load_json(path: str) -> dict:
    if not os.path.exists(path):
        return {"entities": [], "relations": []}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
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

def collect_types(json_objs):
    types = []
    examples = defaultdict(list)  # save example entity names for each type
    examples_all = defaultdict(list)
    for obj in json_objs:
        ents = obj.get("entities", [])
        for e in ents:
            if isinstance(e, dict):
                t = e.get("type")
                n = e.get("name")
                if t:
                    types.append(t)
                    examples_all[t].append(n)
                    if n and len(examples[t]) < 5:
                        examples[t].append(n)
    return types, examples, examples_all

def has_any_token(s, tokens):
    s_lower = s.lower()
    for tok in tokens:
        if tok.lower() in s_lower:
            return True
    return False

def is_fine_grained(type_label):
    """Heuristics to flag over-specific (fine-grained) type labels"""
    # signals: long phrases, color/laterality/severity/anatomy/time, punctuation, explicit locations, 'with' clauses
    l = type_label.strip()
    tokens = re.split(r"\s+", l)
    long_phrase = len(tokens) >= 4  # many words usually encode attributes
    has_color = has_any_token(l, CN_COLOR + EN_COLOR)
    has_laterality = has_any_token(l, LATERALITY_CN + LATERALITY_EN)
    has_severity = has_any_token(l, SEVERITY_CN + SEVERITY_EN) or bool(ROMAN_PATTERN.search(l)) or bool(DEGREE_PATTERN.search(l))
    has_anatomy = has_any_token(l, ANATOMY_CN + ANATOMY_EN)
    has_temporal = has_any_token(l, TEMPORAL_CN + TEMPORAL_EN)
    has_punct = any(p in l for p in PUNCTUATION_HINTS)
    has_with = bool(WITH_PATTERN.search(l)) or bool(LOCATION_PATTERN.search(l))
    # has_noise = has_any_token(l, NOISE_HINTS)
    # A type is "fine-grained" if multiple signals fire OR very long OR contains clear attribute-like tokens
    score = sum([long_phrase, has_color, has_laterality, has_severity, has_anatomy, has_temporal, has_punct, has_with])
    return score >= 2

def suggest_parent(type_label):
    for patt, parent in PARENT_RULES:
        if patt.search(type_label):
            return parent
    # fallback by suffix/prefix hints
    if re.search(r"^Medical|^Clinical|^Treatment", type_label, re.I):
        return "Medical Concept"
    if re.search(r"(wound|injur|burn)", type_label, re.I):
        return "Injury"
    if re.search(r"(procedure|术|操作)", type_label, re.I):
        return "Medical Procedure"
    return "Thing"

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

# ---------------------------
# Load and aggregate
# ---------------------------
json_objs = [safe_load_json(p) for p in INPUT_JSONS]
types, examples, examples_all = collect_types(json_objs)
type_counter = Counter(types)

if not type_counter:
    raise SystemExit("No types found. Please check INPUT_JSONS paths.")

# Compute long-tail flags
counts = list(type_counter.values())
counts_sorted = sorted(counts)
quantile_cut = counts_sorted[max(0, min(len(counts_sorted)-1, int(len(counts_sorted)*LONG_TAIL_BOTTOM_QUANTILE)))]
long_tail_cut = max(LONG_TAIL_MAX_COUNT, quantile_cut)

rows = []
for t, c in type_counter.items():
    fine = is_fine_grained(t)
    long_tail = c <= long_tail_cut
    parent = suggest_parent(t)
    # example_entities = examples_all.get(t, [])
    rows.append({
        "type": t,
        "count": c,
        "is_long_tail": long_tail,
        "is_fine_grained": fine,
        "suggested_parent": parent,
        "example_entities": "; ".join(examples_all.get(t, []) if len(examples_all.get(t, [])) <= 35 else examples_all.get(t, [])[:35]),
    })

df = pd.DataFrame(rows).sort_values(["is_long_tail","is_fine_grained","count"], ascending=[False, False, True])

# Summary
summary = pd.DataFrame({
    "metric": ["total_types", "long_tail_cut_value", "num_long_tail", "num_fine_grained", "num_both"],
    "value": [
        len(type_counter),
        int(long_tail_cut),
        int(df["is_long_tail"].sum()),
        int(df["is_fine_grained"].sum()),
        int(((df["is_long_tail"]) & (df["is_fine_grained"])).sum())
    ]
})

# Export CSVs
out_dir = "result/analysis_result/ltfg_analysis"
os.makedirs(out_dir, exist_ok=True)
all_path = os.path.join(out_dir, "all_types_analysis_examples_35.csv")
both_path = os.path.join(out_dir, "long_tail_and_fine_grained.csv")
df.to_csv(all_path, index=False, encoding="utf-8-sig")
df[(df["is_long_tail"]) & (df["is_fine_grained"])].to_csv(both_path, index=False, encoding="utf-8-sig")

# Show to user
print("Long-tail Fine-grained Type Analysis (All Types)", df)
print("Summary (Counts & Cutoff)", summary)

print("Saved:")
print(" - All types:", all_path)
print(" - Long-tail & fine-grained only:", both_path)
