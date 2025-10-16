# -*- coding: utf-8 -*-
import json
import re
from collections import defaultdict
import pandas as pd
from pathlib import Path
import os

def safe_load_json(path):
    if not os.path.exists(path):
        return {"entities": [], "relations": []}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"entities": [], "relations": []}

def load_all_jsons(path):
    """Load all JSON files from a given directory."""
    json_files = os.listdir(path)
    
    entities_all = []
    relations_all = []
    for p in json_files:
        file_path = os.path.join(path, p)
        if os.path.isfile(file_path) and file_path.endswith(".json"):
            obj = safe_load_json(file_path)
            ents = obj.get("entities", [])
            rels = obj.get("relations", [])

            entities_all.extend(ents)
            relations_all.extend(rels)

    return {"entities": entities_all, "relations": relations_all}



# ---- 1) Define the target label space (from user's lists) ----
UMLS_TYPES = {
    "Therapeutic or Preventive Procedure",
    "Finding",
    "Clinical Drug",
    "Organic Chemical",
    "Bacterium",
    "Pharmacologic Substance",
    "Eukaryote",
    "Amino Acid, Peptide, or Protein",
    "Plant",
    "Injury or Poisoning",
}

SNOMED_TYPES = {
    "Disease or Syndrome",
    "Body Part, Organ, or Organ Component",
    "Laboratory Procedure",
    "Neoplastic Process",
}

TARGET_TYPES = sorted(UMLS_TYPES.union(SNOMED_TYPES))

# ---- 2) Build a rules-based mapper (baseline heuristic) ----
# Each rule: pattern(s) -> mapped type
rules = [
    # SNOMED "Disease or Syndrome" / UMLS "Finding"
    (["Disease", "疾病", "Medical Condition", "Condition", "综合征", "综合症", "征", "病理", "感染", "中毒", "休克", "衰竭"], "Disease or Syndrome", 0.85),
    (["Symptom", "症状", "体征", "Finding", "征象"], "Finding", 0.8),

    # Injuries/poisoning
    (["Injury", "伤", "损伤", "烧伤", "咬伤", "刺伤", "中毒", "Poisoning"], "Injury or Poisoning", 0.9),

    # Procedures (therapeutic/preventive)
    (["Medical Procedure", "Procedure", "治疗方法", "治疗手段", "外科", "手术", "植皮", "清创", "心肺复苏", "气管切开", "减压", "监护"], "Therapeutic or Preventive Procedure", 0.85),

    # Lab/diagnostic
    (["Laboratory Procedure", "实验室", "检验", "检查", "影像", "影像学", "内镜", "镜检", "磁共振", "MRI", "CT"], "Laboratory Procedure", 0.85),

    # Body structure
    (["Body Part", "Body_Part", "Anatomical Structure", "器官", "组织", "神经", "肌腱", "毛囊", "肋", "部位"], "Body Part, Organ, or Organ Component", 0.9),

    # Drugs & pharmacologic substances
    (["药物", "Drug", "临床药物", "Clinical Drug", "用药", "Medicine", "药膏", "阿托品", "吗啡", "多巴胺"], "Clinical Drug", 0.8),
    (["Pharmacologic Substance", "药物成分", "活性成分", "活性物质"], "Pharmacologic Substance", 0.85),

    # Chemicals
    (["Chemical", "化学", "化学物质", "化学品", "化学化合物", "Chemical Substance", "Chemical Compound"], "Organic Chemical", 0.7),

    # Bacterium / Eukaryote
    (["细菌", "Bacterium", "杆菌", "链球菌", "葡萄球菌"], "Bacterium", 0.9),
    (["Eukaryote", "真核生物"], "Eukaryote", 0.8),

    # Proteins / AAs / peptides
    (["蛋白", "肽", "氨基酸", "Protein", "Peptide", "Amino Acid"], "Amino Acid, Peptide, or Protein", 0.85),

    # Plants
    (["Plant", "植物", "草药", "中草药"], "Plant", 0.9),

    # Oncology
    (["肿瘤", "肿瘤过程", "Neoplasm", "Neoplastic"], "Neoplastic Process", 0.85),
]

# Some item-level overrides (by name) for better precision in this dataset
name_overrides = {
    "磺胺嘧啶银": ("Clinical Drug", 0.95),
    "氟化物": ("Organic Chemical", 0.85),
    "乙醇": ("Organic Chemical", 0.9),
    "苯扎氯铵": ("Organic Chemical", 0.85),
    "肌红蛋白": ("Amino Acid, Peptide, or Protein", 0.9),
    "头颈部": ("Body Part, Organ, or Organ Component", 0.95),
    "小腿": ("Body Part, Organ, or Organ Component", 0.95),
    "毛囊": ("Body Part, Organ, or Organ Component", 0.95),
    "筋膜": ("Body Part, Organ, or Organ Component", 0.95),
    "抗菌纱布": ("Therapeutic or Preventive Procedure", 0.55),  # device not in target list; map to its use context with low confidence
    "吸水敷料": ("Therapeutic or Preventive Procedure", 0.55),
    "磁共振成像": ("Laboratory Procedure", 0.9),
    "心肺复苏": ("Therapeutic or Preventive Procedure", 0.95),
    "气管切开术": ("Therapeutic or Preventive Procedure", 0.95),
    "创面清创术": ("Therapeutic or Preventive Procedure", 0.95),
    "机械辅助通气": ("Therapeutic or Preventive Procedure", 0.9),
    "氧气": ("Clinical Drug", 0.6),     # not ideal; falls into therapy/substance bucket
    "葡萄糖酸钙": ("Clinical Drug", 0.9),
    "阿托品": ("Clinical Drug", 0.9),
    "麻黄碱": ("Clinical Drug", 0.85),
    "依米丁": ("Clinical Drug", 0.85),
    "肌皮瓣": ("Therapeutic or Preventive Procedure", 0.6),
    "微粒自体皮播散植皮法": ("Therapeutic or Preventive Procedure", 0.95),
    "自、异体皮混植": ("Therapeutic or Preventive Procedure", 0.9),
    "包扎疗法": ("Therapeutic or Preventive Procedure", 0.9),
    "拔火罐": ("Therapeutic or Preventive Procedure", 0.65),
}

def map_type(name: str, typ: str, desc: str):
    # Name-level overrides
    if name in name_overrides:
        return name_overrides[name]

    text = " ".join([name or "", typ or "", desc or ""])
    score_best = 0.0
    label_best = None

    for patterns, mapped, score in rules:
        if any(re.search(pat, text, re.IGNORECASE) for pat in patterns):
            if score > score_best:
                score_best = score
                label_best = mapped

    # Fallbacks
    if label_best is None:
        # Handle some Chinese categories seen in file
        if typ in ["Person", "Professional Role", "Research Institution", "Medical Institution", "Hospital", "City",
                   "Country", "Location", "Event", "Historical Event", "Section", "Time Period", "Year",
                   "Diagram", "Tool", "Medicine", "Material", "Medical Product", "Toxin Type", "Animal", "Herb",
                   "Biological Process", "Physiological Process", "Physiological Function", "Medical Concept",
                   "Biological Structure", "Traditional Medicine"]:
            return ("UNMAPPED", 0.0)

        # If the original type explicitly equals one of the target labels, use it
        if typ in TARGET_TYPES:
            return (typ, 0.7)

        # Otherwise unknown
        return ("UNMAPPED", 0.0)

    return (label_best, score_best)

import_data_dir = 'result/import_data'
import_data = load_all_jsons(import_data_dir)
entities = import_data.get("entities", [])

mapped_rows = []
for e in entities:
    name = e.get("name", "")
    typ = e.get("type", "")
    desc = e.get("description", "")
    mapped, conf = map_type(name, typ, desc)
    mapped_rows.append({
        "name": name,
        "original_type": typ,
        "mapped_type": mapped,
        "confidence": round(conf, 2)
    })

df = pd.DataFrame(mapped_rows)
# Show a quick sample

# Save full results
out_path = Path("result/analysis_result/entity_mapping_results_v3.json")
with out_path.open("w", encoding="utf-8") as f:
    json.dump(mapped_rows, f, ensure_ascii=False, indent=2)

out_path.as_posix()
