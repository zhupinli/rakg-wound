type_classification_prompt = """
现在我需要你根据我给出的实体信息，将实体分类到Allowed_types中：
Entity: {entity}
Allowed_tyepes:{allowed_types}

要求：
1. 如果判断出实体不属于上面所给的Allowed_types, 实体类型归类为"Other"。
2. 请严格用JSON格式输出, 格式为{{"top_level_category": Type}}，Type必须是Allowed_types中的某一项或者"Other"。
"""

type_classfication_llm_check_prompt = """

Entity: {entity}
Allowed_tyepes:{allowed_types}
"""