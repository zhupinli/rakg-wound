import json 
import re
import ast
from typing import List, Dict, Any

_ALLOWED_KEYS = {"relation", "target_name", "target_type", "target_description", "relation_description"}

_KV_PATTERN = re.compile(
    r'(relation|target_name|target_type|target_description|relation_description)\s*:\s*'
    r'(?:["“”]?)(.*?)(?:["“”]?)\s*(?=(?:,\s*(?:relation|target_name|target_type|target_description|relation_description)\s*:)|$)'
)

def normalize_attributes_dict(raw_text):
    if isinstance(raw_text, dict) or isinstance(raw_text, list):
        return raw_text
    else:
        try:
            return json.loads(raw_text)
        except json.JSONDecodeError:
            return []
        
def normalize_attributes_dict_origin(raw_text):
    if isinstance(raw_text, dict) or isinstance(raw_text, list):
        return raw_text
    else:
        return []

# normalize attributes from raw text
def normalize_attributes(raw_attrs):
    """
    将各种异常形态的 attributes 统一转成 List[Dict[key,value]]。
    - 规范形态 (已是 dict)          -> 原样返回
    - "key: xxx, value: xxx" 字符串 -> 解析后转 dict
    - ['key', 'value'] 等占位符     -> 返回 []
    """
    if not raw_attrs:
        return []

    # 已经是 [{key:..., value:...}, ...]
    if isinstance(raw_attrs[0], dict):
        # 这里可再做一次健壮性检查：确保都有 key/value 字段
        cleaned = [
            {"key": a.get("key", "").strip(),
             "value": a.get("value", "").strip()}
            for a in raw_attrs
            if isinstance(a, dict) and a  # 防 None
        ]
        return cleaned

    # 列表里是字符串
    cleaned = []
    for item in raw_attrs:
        if not isinstance(item, str):
            continue

        # - 普通 “key: xxx, value: xxx” 或 “key：xxx，value：xxx”（中英文混排）
        m = re.match(r"\s*key\s*[:：]\s*(.+?)\s*,\s*value\s*[:：]\s*(.+)\s*$", item)
        if m:
            cleaned.append({"key": m.group(1).strip(),
                            "value": m.group(2).strip()})
            continue

        # - 万一是 “key=xxx value=xxx” “key -> xxx / value -> xxx” 之类
        m2 = re.match(r"\s*key\s*[:=>\-]+\s*(.+?)\s+[|/,&，]\s*value\s*[:=->]+\s*(.+)", item)
        if m2:
            cleaned.append({"key": m2.group(1).strip(),
                            "value": m2.group(2).strip()})
            continue

    return cleaned  # if failed, return []


# normalize relationships from raw text

def _strip_quotes(s: str) -> str:
    '''
    match ' " ‘“ ’ ” and strip them from the start and end of the string.
    '''
    s = s.strip()
    if (s.startswith("'") and s.endswith("'")) or (s.startswith('"') and s.endswith('"')) or \
       (s.startswith('“') and s.endswith('”')):
        return s[1:-1].strip()
    return s

def _parse_dict_like_string(s: str) -> Dict[str, Any]:
    """
    尝试把形如 "{'relation': '导致', ...}" 的字符串安全转成 dict；
    如果失败，则再按 "key: value, key2: value2" 的模式用正则解析。
    """
    s = s.strip()
    # 1) 尝试 Python 字面量（单引号）或 JSON（双引号）
    if s.startswith("{") and s.endswith("}"):
        try:
            return ast.literal_eval(s)
        except Exception:
            try:
                import json
                return json.loads(s)
            except Exception:
                pass
    # 2) 按 "key: value" 正则解析一条记录（字符串里包含多个键值）
    out = {}
    for m in _KV_PATTERN.finditer(s):
        k, v = m.group(1), _strip_quotes(m.group(2))
        out[k] = v
    return out

def _parse_flat_kv_items(items: List[str]) -> List[Dict[str, Any]]:
    """
    处理“字典的 key:value 被拆成列表项”的情况：
    ["relation: \"Part Of\"", "target_name: \"人体\"", ... , "relation: \"Involved In\"", ...]
    根据出现新的 relation 视为开启一条新记录。
    """
    records = []
    curr = {}

    def flush():
        nonlocal curr
        if curr:
            records.append(curr)
            curr = {}

    for it in items:
        pairs = list(_KV_PATTERN.finditer(it))
        if not pairs:
            # 粗糙兜底：若是 "key: value" 但没被上面的正则捕到，尽量拆一次
            if ":" in it:
                key, val = it.split(":", 1)
                key = key.strip()
                val = _strip_quotes(val)
                if key in _ALLOWED_KEYS:
                    # 若遇到新的 relation 且 curr 里已有部分键，则刷新
                    if key == "relation" and any(k in curr for k in _ALLOWED_KEYS if k != "relation"):
                        flush()
                    curr[key] = val
            continue

        for m in pairs:
            k, v = m.group(1), _strip_quotes(m.group(2))
            if k == "relation" and any(k2 in curr for k2 in _ALLOWED_KEYS if k2 != "relation"):
                # 新的一条
                flush()
            curr[k] = v
    flush()
    return records

def _parse_triple_string(s: str) -> Dict[str, Any]:
    """
    解析三元组："源 - Relation - 目标"；仅有 relation 与 target_name。
    """
    parts = [p.strip() for p in s.split(" - ")]
    if len(parts) >= 3:
        source, relation, target = parts[0], parts[1], parts[2]
        return {
            "relation": relation,
            "target_name": target,
            # 无法可靠获取类型与描述
        }
    return {}

def _normalize_record(rec: Dict[str, Any]) -> Dict[str, Any]:
    """
    {
      "name": <target_name或空>,
      "type": <target_type或空>,
      "description": <target_description或空>,
      "relation": <relation或空>,
      "relation_description": <relation_description或空>
    }
    """
    name = rec.get("target_name") or rec.get("name") or ""
    type_ = rec.get("target_type") or rec.get("type") or ""
    target_desc = rec.get("target_description") or rec.get("description") or ""
    relation = rec.get("relation") or ""
    relation_desc = rec.get("relation_description") or ""

    return {
        "target_name": name,
        "target_type": type_,
        "target_description": target_desc,
        "relation": relation,
        "relation_description": relation_desc,
    }

def normalize_relationships(relationships: List[Any]) -> List[Dict[str, Any]]:
    """
    适配以下输入情况：
      1) 每个元素是 dict 或 dict 的字符串表示（单引号/双引号）
      2) 字典的 key:value 被拆成列表项（平铺）
      3) 单条字符串里是 "relation: x, target_name: y, ..." 的键值串
      4) 三元组："A - Located In - B"
      5) 只有键名、为空、写了“无”等=> 视为无有效记录

    返回：标准列表，每个元素为
      { "name": ..., "type": ..., "description": ..., "attributes": {...} }
    """
    if not relationships:
        return []

    # 明确无数据的几种写法
    flat_lower = [str(x).strip().lower() for x in relationships]
    if flat_lower in (["无"], ["none"], ["null"], ["relationships"]):
        return []
    if all(isinstance(x, str) and x.strip() in _ALLOWED_KEYS for x in relationships):
        return []

    raw_records: List[Dict[str, Any]] = []

    # 情况 A：存在标准 dict
    if any(isinstance(x, dict) for x in relationships):
        for x in relationships:
            if isinstance(x, dict):
                raw_records.append(x)
            elif isinstance(x, str):
                d = _parse_dict_like_string(x)
                if d:
                    raw_records.append(d)

    else:
        # 情况 B：全部是字符串
        strs = [str(x) for x in relationships]

        # B1：三元组（含 “ - ”）
        if all((" - " in s and s.count(" - ") >= 2) or (":" in s) for s in strs) and \
           any(" - " in s and s.count(" - ") >= 2 for s in strs):
            for s in strs:
                if " - " in s and s.count(" - ") >= 2:
                    d = _parse_triple_string(s)
                    if d:
                        raw_records.append(d)
                else:
                    d = _parse_dict_like_string(s)
                    if d:
                        raw_records.append(d)

        # B2：像 "{'k': 'v'}" 或 "k: v, k2: v2" 的整条字符串
        elif any((":" in s) for s in strs):
            # 先尝试逐条按“整条记录”解析
            tmp = []
            for s in strs:
                d = _parse_dict_like_string(s)
                if d:
                    tmp.append(d)

            whole_like = [d for d in tmp if len(d) >= 2]   # 记录中至少出现 2 个键
            # 如果每条都解析出一些键值，就当作逐条记录
            if whole_like and len(whole_like) == len(tmp):
                # 每条都是“像完整记录”的
                raw_records.extend(tmp)
            else:
                # 只要有一条是单键，就认为整体是“平铺”
                raw_records.extend(_parse_flat_kv_items(strs))

        # B3：其它字符串，无法识别 => 忽略
        else:
            pass

    # 归一化到标准结构
    normalized = [_normalize_record(r) for r in raw_records if r]

    # 简单去重（按 name/type/description）
    seen = set()
    uniq = []
    for rec in normalized:
        key = (rec["target_name"], rec["target_type"], rec["target_description"], rec["relation"], rec["relation_description"])
        if key not in seen:
            seen.add(key)
            uniq.append(rec)
    return uniq