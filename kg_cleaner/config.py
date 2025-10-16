# -*- coding: utf-8 -*-

# I/O
INPUT_JSONS = [
    "/mnt/data/merged_entities_relations.json",
    "/mnt/data/kg_result_modified_normalize_debug_v2_chapter_4.json",
    "/mnt/data/kg_result_modified_normalize_debug_v2_chapter_6.json",
    "/mnt/data/kg_result_modified_normalize_debug_v2_chapter_7.json",
]
OUT_DIR = "/mnt/data/kg_pipeline"

# Long-tail / fine-grained
LONG_TAIL_MAX_COUNT = 5
LONG_TAIL_BOTTOM_QUANTILE = 0.2

# Regex tokens / hints
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
PUNCTUATION_HINTS = [":", "：", "/", "\\\\", "(", ")", "（", "）", ","]

NON_ENTITY_TYPE_PREFIXES = ("图","表","Figure","Fig.","Image","Diagram","Illustration")

# Canonical parent suggestion rules (strings or regex handled in normalization)
PARENT_RULES = [
    ("烧伤", "Burn Injury"),
    ("burn", "Burn Injury"),
    ("伤口", "WoundFinding"),
    ("创面", "WoundFinding"),
    ("wound", "WoundFinding"),
    ("清创", "Medical Procedure"),
    ("debrid", "Medical Procedure"),
    ("包扎", "Medical Procedure"),
    ("敷料", "Medical Procedure"),
    ("dressing", "Medical Procedure"),
    ("感染", "Infection"),
    ("sepsis", "Infection"),
    ("cellulitis", "Infection"),
    ("疼痛", "Sign or Symptom"),
    ("pain", "Sign or Symptom"),
    ("symptom", "Sign or Symptom"),
    ("药物", "Drug"),
    ("drug", "Drug"),
    ("medicat", "Drug"),
    ("影像", "Medical Imaging Technique"),
    ("MRI", "Medical Imaging Technique"),
    ("CT", "Medical Imaging Technique"),  # exact word handled in normalization via \\bCT\\b
    ("磁共振", "Medical Imaging Technique"),
    ("成像", "Medical Imaging Technique"),
    ("X-ray", "Medical Imaging Technique"),
    ("组织", "Anatomical Structure"),
    ("tissue", "Anatomical Structure"),
    ("解剖", "Anatomical Structure"),
    ("anatom", "Anatomical Structure"),
    ("评估", "Assessment Method"),
    ("assessment", "Assessment Method"),
    ("评分", "Assessment Method"),
    ("分级", "Assessment Method"),
    ("scale", "Assessment Method"),
    ("method", "Assessment Method"),
    ("手术", "Medical Procedure"),
    ("surgery", "Medical Procedure"),
    ("术", "Medical Procedure"),
    ("procedure", "Medical Procedure"),
    ("操作", "Medical Procedure"),
    ("技术", "Medical Procedure"),
    ("病", "Disease or Syndrome"),
    ("disease", "Disease or Syndrome"),
    ("综合征", "Disease or Syndrome"),
    ("症候群", "Disease or Syndrome"),
    ("蛋白", "Amino Acid, Peptide, or Protein"),
    ("protein", "Amino Acid, Peptide, or Protein"),
    ("器械", "Medical Device"),
    ("器具", "Medical Device"),
    ("instrument", "Medical Device"),
    ("device", "Medical Device"),
    ("引流", "Medical Device"),
    ("管", "Medical Device"),
]

# Canonical parent -> UMLS semantic type (label)
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

# Optional lexicon paths
LEX_DIR = "/mnt/data/lexicons"
UMLS_ST_PATH = f"{LEX_DIR}/umls_semantic_types.csv"
UMLS_CUI_PATH = f"{LEX_DIR}/umls_concepts.csv"
SNOMED_PATH  = f"{LEX_DIR}/snomed_concepts.csv"