import pandas as pd
from collections import defaultdict

# 必填：RRF 文件路径
MRCONSO = "umls_data/MRCONSO.RRF"
MRSTY   = "umls_data/MRSTY.RRF"

OUT_DIR = "data/lexicons"  # 输出目录
LANGS = ["ENG", "CHI"]          # 需要的语言子集，按需改

# 读取 MRSTY（CUI -> {TUI, STY} 多值）
sty = pd.read_csv(MRSTY, sep="|", header=None, dtype=str, usecols=[0,1,3],
                  names=["CUI","TUI","STY"])
sty_grp = sty.groupby("CUI").agg({"TUI": lambda x: list(set(x)), "STY": lambda x: list(set(x))})

def process_lang(lang):
    print(f"Processing language: {lang}")
    iter_csv = pd.read_csv(
        MRCONSO, sep="|", header=None, dtype=str, chunksize=500000,
        usecols=[0,1,11,12,13,14,15,16],
        names=["CUI","LAT","SAB","TTY","CODE","STR","SRL","SUPPRESS"]
    )
    rows = []
    for chunk in iter_csv:
        df = chunk[chunk["LAT"] == lang].copy()
        df["name"] = df["STR"].astype(str).str.strip().str.lower()
        df["cui"] = df["CUI"]
        df["sab"] = df["SAB"]
        df["tty"] = df["TTY"]
        df["code"] = df["CODE"]

        # 合并语义类型
        df = df.merge(sty_grp, how="left", left_on="cui", right_on="CUI")
        df.drop(columns=["CUI"], inplace=True)
        # 取一个“pref_label”：优先该 CUI 的 ENG/PT；如果当前 lang 是 CHI，没有 ENG/PT 就用当前 name
        # 先为所有 CUI 找 ENG/PT（一次性建立索引，避免重复扫描）
        rows.append(df[["name","cui","sab","tty","code","TUI","STY"]])

    df_all = pd.concat(rows, ignore_index=True)
    # 生成 concepts 词表（name -> 多 CUI）
    df_all["tui"] = df_all["TUI"].apply(lambda x: ";".join(x) if isinstance(x, list) else "")
    df_all["semantic_type"] = df_all["STY"].apply(lambda x: ";".join(x) if isinstance(x, list) else "")
    df_all["pref_label"] = ""  # 简化起见，这里留空；如需 ENG/PT，可额外跑一遍 ENG/PT 提取
    out_path = f"{OUT_DIR}/umls_concepts_{lang.lower()}.csv"
    df_all[["name","cui","pref_label","semantic_type","tui","sab","tty","code"]].to_csv(out_path, index=False, encoding="utf-8-sig")
    print("saved:", out_path)

    # 生成 semantic_types（name -> STY/TUI），按 name 聚合
    agg = df_all.groupby("name").agg({
        "semantic_type": lambda x: ";".join(sorted(set(";".join(x).split(";")) - {""})),
        "tui": lambda x: ";".join(sorted(set(";".join(x).split(";")) - {""}))
    }).reset_index()
    out2 = f"{OUT_DIR}/umls_semantic_types_{lang.lower()}.csv"
    agg.rename(columns={"semantic_type":"semantic_type","tui":"tui"}, inplace=True)
    agg.to_csv(out2, index=False, encoding="utf-8-sig")
    print("saved:", out2)

    # 生成 SNOMED 子集（可选）
    snomed = df_all[df_all["sab"].str.startswith("SNOMEDCT", na=False)].copy()
    snomed["snomed_id"] = snomed["code"]
    snomed["fsn"] = ""             # 若要 FSN，可在 MRCONSO 中筛 TTY=FN 再 merge
    snomed["semantic_tag"] = ""    # 若从 FSN 拆 tag，可在后处理里补
    out3 = f"{OUT_DIR}/snomed_concepts_{lang.lower()}.csv"
    snomed[["name","snomed_id","fsn","semantic_tag","cui","tui","semantic_type"]].to_csv(out3, index=False, encoding="utf-8-sig")
    print("saved:", out3)

for L in LANGS:
    process_lang(L)
