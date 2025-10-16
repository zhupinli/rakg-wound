# -*- coding: utf-8 -*-
from collections import Counter
import pandas as pd

def summarize_types(entities):
    counts = Counter([e.get("type","") for e in entities if e.get("type")])
    df = pd.DataFrame(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])), columns=["type","count"])
    return df