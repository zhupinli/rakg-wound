import jieba
from rouge_score import rouge_scorer

# 1. 准备你的文本
# 参考摘要 (标准答案，通常是人类写的)
reference_summary = "阳光明媚的下午，一只小猫在草地上懒洋洋地打滚。"

# 生成摘要 (模型生成的文本)
generated_summary = "一个阳光明媚的下午，那只猫在草地上打滚。"

# 注意：对于中文，rouge-score默认按字切分。
# 如果想按词计算，需要先用分词工具（如jieba）处理成空格隔开的字符串。
# 下面会详细说明分词的重要性。


tokenized_ref = " ".join(jieba.cut(reference_summary))
tokenized_gen = " ".join(jieba.cut(generated_summary))

print("按词分词后的参考摘要:", tokenized_ref)
print("按词分词后的生成摘要:", tokenized_gen)
# 输出:
# 按词分词后的参考摘要: 阳光明媚 的 下午 ， 一只 小猫 在 草地 上 懒洋洋 地 打滚 。
# 按词分词后的生成摘要: 一个 阳光明媚 的 下午 ， 那只 猫 在 草地 上 打滚 。

# 2. 初始化Scorer
# 你可以指定你想计算的ROUGE类型
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

# 3. 计算分数
# 传入处理好的、以空格分词的字符串
scores = scorer.score(tokenized_ref, tokenized_gen)

# 4. 查看结果
# 结果是一个字典，每个ROUGE类型都包含Precision, Recall, F-measure
for key, value in scores.items():
    print(f"--- {key} ---")
    print(f"Precision: {value.precision:.4f}")
    print(f"Recall:    {value.recall:.4f}")
    print(f"F1-score:  {value.fmeasure:.4f}")
    print("\n")
