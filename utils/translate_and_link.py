import argparse
import json
import sys
import time
import os
import requests
import re
from tqdm import tqdm
from typing import Any, Dict, Iterable, List, Set, Tuple
from collections import Counter
from .AuthV3Util import returnAuthMap
from .snowstorm_api import getDescriptionByString, find_top_level_category

from .util_prompt import type_classification_prompt

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# langchain config
OPENAI_MODEL = "qwen-plus"
OPENAI_API_KEY = "sk-16406a2b418e4d79b040091bb02e3717"
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

llm = ChatOpenAI(
    model=OPENAI_MODEL,
    api_key=OPENAI_API_KEY,
    base_url=BASE_URL,
    temperature=0,
    request_timeout=60,
)

snomed_ct_top_level_key_to_id = {
    'body structure': 123037004,
    'finding': 404684003,
    'environment / location': 308916002,
    'event': 272379006,
    'observable entity': 363787002,
    'organism': 410607006,
    'product': 373873005,
    'physical force': 78621006,
    'physical object': 260787004,
    'procedure': 71388002,
    'qualifier value': 362981000,
    'record artifact': 419891008,
    'situation': 243796009,
    'metadata': 900000000000441003,
    'social context': 48176007,
    'special concept': 370115009,
    'specimen': 123038009,
    'staging scale': 254291000,
    'substance': 105590001
}

# translate config
YOUDAO_ENDPOINT = "https://openapi.youdao.com/api"

INPUT_JSONS = [
    "result/import_data/kg_result_modified_normalize_debug_v2_chapter_4.json",
    "result/import_data/kg_result_modified_normalize_debug_v2_chapter_6.json",
    "result/import_data/kg_result_modified_normalize_debug_v2_chapter_7.json",
]

OUTPUT_DIR = "result/analysis_result"

APP_KEY = "1c63e5a0c27aa2b0"

APP_SECRET = "LydvpbhUvZJEkHWKR5pbFIFywSkHY3TN"

def iter_entity_dicts(obj: Any) -> Iterable[Dict[str, Any]]:
    """Yield all dicts that look like entities (have a 'name' key)."""
    if isinstance(obj, dict):
        if "name" in obj and isinstance(obj["name"], str):
            yield obj
    elif isinstance(obj, list):
        for it in obj:
            yield from iter_entity_dicts(it)

def translate_one(text: str, app_key: str, app_secret: str,
                  lang_from: str = "auto", lang_to: str = "en",
                  timeout: int = 15, vocab_id: str = None,
                  max_retries: int = 4, retry_base_sleep: float = 1.0) -> str:
    """
    Translate a single string using Youdao API (v3 auth).
    Returns translated text; raises RuntimeError on persistent failure.
    """
    assert isinstance(text, str) and text.strip() != ""
    params = {
        "q": text,
        "from": lang_from,
        "to": lang_to,
    }
    if vocab_id:
        params["vocabId"] = vocab_id

    # add auth params using v3 (with input hashing strategy)
    auth = returnAuthMap(app_key, app_secret, text)
    params.update(auth)

    headers = {"Content-Type": "application/x-www-form-urlencoded"}

    attempt = 0
    while True:
        try:
            resp = requests.post(YOUDAO_ENDPOINT, data=params, headers=headers, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            attempt += 1
            if attempt > max_retries:
                raise RuntimeError(f"Youdao API request failed after {max_retries} retries: {e}")
            sleep_s = min(30.0, retry_base_sleep * (2 ** (attempt - 1)))
            time.sleep(sleep_s)
            continue

        # Youdao success code is "0"
        if str(data.get("errorCode")) == "0":
            tr_list = data.get("translation") or []
            # 'translation' is a list of strings; pick the first
            if tr_list:
                return tr_list[0]
            else:
                return ""  # empty translation (unlikely)
        else:
            # Retry some transient errors; otherwise raise
            err = str(data.get("errorCode"))
            attempt += 1
            if attempt > max_retries:
                raise RuntimeError(f"Youdao API errorCode={err} for text='{text[:50]}...'")
            sleep_s = min(30.0, retry_base_sleep * (2 ** (attempt - 1)))
            time.sleep(sleep_s)

def align_to_snomed():

    import_data_dir = 'result/analysis_result/entity_align_result_snowstorm'
    output_file_dir = 'result/analysis_result/entity_align_result_all'
    files = [file for file in os.listdir(import_data_dir) if file.endswith(".json")]
    for file in files:
        entities = []
        file_path = os.path.join(import_data_dir, file)
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            entities = data.get("entities", [])
        if not entities:
            print(f"[warn] No entities found in {file_path}, skipping.", file=sys.stderr)
            continue
        for entity in tqdm(entities):
            entity_name_standard = entity.get("entity_name_standard", "").strip()
            if entity_name_standard:
                attempt = 0
                max_retries = 4
                retry_base_sleep = 1.0
                while attempt <= max_retries:
                    try:
                        align_entity_list = getDescriptionByString(entity_name_standard)
                        # 无法对齐
                        if not align_entity_list:
                            entity["align_entity"] = {}
                            entity["align_entity_type"] = ""
                            entity["top_level_category"] = ""
                            break
                        align_entity = align_entity_list[0] # 只取第一个，但并不是最相关的，还需要优化（LLM？）
                        top_level_category = find_top_level_category(align_entity.get("concept").get("conceptId", "").strip())
                        if not top_level_category:
                            entity["top_level_category"] = ""
                        else:
                            entity["top_level_category"] = top_level_category    
                        entity["align_entity"] = align_entity

                        if "concept" in align_entity and "fsn" in align_entity["concept"]:
                            term = align_entity["concept"]["fsn"]["term"]
                            # match = re.search(r"\((.*?)\)", term)
                            m = re.findall(r"\(([^()]*)\)", term)
                            if m:
                                entity["align_entity_type"] = m[-1].strip()  # 提取括号中的内容
                            else:
                                entity["align_entity_type"] = ""
                        else:
                            entity["align_entity_type"] = ""
                        break
                    except Exception as e:
                        attempt += 1
                        if attempt > max_retries:
                            print(f"[warn] Failed to align entity '{entity_name_standard}' after {max_retries} retries: {e}", file=sys.stderr)
                            entity["align_entity"] = {}
                            entity["align_entity_type"] = ""
                            entity["top_level_category"] = ""
                            break
                        sleep_s = min(30.0, retry_base_sleep * (2 ** (attempt - 1)))
                        print(f"[warn] Retrying alignment for '{entity_name_standard}' in {sleep_s:.1f} seconds due to error: {e}", file=sys.stderr)
                        time.sleep(sleep_s)

        chapter_num = file.split("_")[-3]
        output_file = os.path.join(output_file_dir, f"kg_result_chapter_{chapter_num}.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"[done] Processed and wrote: {output_file}")

def count_type():
    input_file_dir = "result/analysis_result/entity_align_result_llm"
    file_list = [os.path.join(input_file_dir, file) for file in os.listdir(input_file_dir) if file.endswith(".json")]

    counts = Counter()
    for file in file_list:
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)
            entities = data.get("entities", [])
            for entity in entities:
                entity_type = entity.get("align_entity_type", "").strip()
                if entity_type:
                    counts[entity_type] += 1
    sorted_types = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    for entity_type, count in sorted_types:
        print(f"{entity_type}: {count}")

def type_classification():
    # 打开 JSON 模式，强制模型以 JSON 返回
    llm_json = llm.bind(response_format={"type": "json_object"})
    prompt = ChatPromptTemplate.from_template(type_classification_prompt)
    parser = JsonOutputParser()
    chain = prompt | llm_json | parser  # 直接拿到 dict
    
    input_file_dir = "result/analysis_result/entity_align_result_snowstorm_with_top_level_category"
    output_file_dir = "result/analysis_result/entity_align_result_llm_all"
    file_list = [os.path.join(input_file_dir, file) for file in os.listdir(input_file_dir) if file.endswith(".json")]
    snomed_ct_types = list(snomed_ct_top_level_key_to_id.keys())
    for file in file_list:
        print(f"[info] Processing file: {file}")
        entities = []
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)
            entities = data.get("entities", [])
        if not entities:
            print(f"[warn] No entities found in {file}, skipping.", file=sys.stderr)
            continue
        for entity in tqdm(entities):
            # if not entity.get("top_level_category", "").strip():
            inputs = {"entity": entity, "allowed_types": snomed_ct_types}
            result = json.dumps(chain.invoke(inputs), ensure_ascii=False, indent=2)
            entity["top_level_category"] = json.loads(result).get("top_level_category", "")
        with open(os.path.join(output_file_dir, f"{os.path.basename(file)}"), "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from_lang", default="auto", help="Source language code (default: auto)")
    ap.add_argument("--to_lang", default="en", help="Target language code (default: en)")
    ap.add_argument("--vocab_id", default=None, help="Optional Youdao user vocabulary ID")
    ap.add_argument("--overwrite_existing", action="store_true",
                    help="If set, always re-translate even if entity_name_standard exists")
    ap.add_argument("--rate_limit_sleep", type=float, default=0.05,
                    help="Sleep seconds between requests to avoid throttling (default: 0.05)")
    args = ap.parse_args()

    for input_json in INPUT_JSONS:

        # Load JSON
        with open(input_json, "r", encoding="utf-8") as f:
            data = json.load(f)

        entities: List[Dict[str, Any]] = list(iter_entity_dicts(data["entities"]))
        print(f"[info] found {len(entities)} entity dicts with 'name'")

        # Worklist (dedup only for API call cou2nt visibility; we still translate per-entity to keep mapping simple)
        names_to_process: List[str] = []
        for ent in entities:
            name = ent.get("name", "").strip() if isinstance(ent.get("name"), str) else ""
            if not name:
                continue
            if (not args.overwrite_existing) and isinstance(ent.get("entity_name_standard"), str) and ent["entity_name_standard"].strip():
                continue
            names_to_process.append(name)

        print(f"[info] items to translate: {len(names_to_process)} (per-entity requests)")

        # Translate and fill back
        success, fail = 0, 0
        cache: Dict[Tuple[str, str, str], str] = {}  # (text, from, to) -> translation

        for ent in entities:
            name = ent.get("name", "").strip() if isinstance(ent.get("name"), str) else ""
            if not name:
                continue
            if (not args.overwrite_existing) and isinstance(ent.get("entity_name_standard"), str) and ent["entity_name_standard"].strip():
                continue
            key = (name, args.from_lang, args.to_lang)
            if key in cache:
                ent["entity_name_standard"] = cache[key]
                success += 1
                continue
            try:
                t = translate_one(text=name, app_key=APP_KEY,
                                  app_secret=APP_SECRET,
                                  lang_from=args.from_lang, lang_to=args.to_lang, vocab_id=args.vocab_id)
                ent["entity_name_standard"] = t
                cache[key] = t
                success += 1
            except Exception as e:
                # keep original and move on
                ent["entity_name_standard"] = ent.get("entity_name_standard", "")
                fail += 1
                print(f"[warn] translate fail for '{name}': {e}", file=sys.stderr)
            time.sleep(args.rate_limit_sleep)

        print(f"[info] translation done. success={success}, fail={fail}")

        # output
        output_file = os.path.join(OUTPUT_DIR, os.path.basename(input_json).replace(".json", "_translated.json"))
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"[done] wrote: {output_file}")


if __name__ == "__main__":
    # main()
    # align_to_snomed()
    # count_type()
    type_classification()