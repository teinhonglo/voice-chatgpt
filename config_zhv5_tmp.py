OPENAI_KEY=""
PPLX_KEY=""
LANG="zh"
USER_ID="user"
INIT_PROMPT = """
你是中文考官，請考我中文口說能力
"""

GRADE_PROMPT = """
"""

TS_PROMPT = """
請以繁體中文將文本簡化至CEFR等級的Pre，並僅以JSON格式回覆我。
{""Pre"": ...}
"""

WARMUP_REPLY="我是學生，請開始問我"

RUBRIC_PROMPT="""請完結對話，以學生(使用者)能理解的簡單詞彙，簡短回覆對話結果，每次回覆不超過25字。
"""

SAVED_FILENAME="""Examplar"""

MODEL_LIST=["gpt-4o", "gpt-4", "gpt-4-32k", "gpt-3.5-turbo-1106", "gpt-3.5-turbo", "gpt-3.5-turbo-16k"]
