from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from db import repository

TOKEN_BUDGET = 2800
RECENT_MESSAGE_LIMIT = 16
MAX_SUMMARY_CHARS = 800
SUMMARY_PROMPT = PromptTemplate.from_template(
    "你负责把森林康养咨询的历史对话压缩成“要点清单”。只保留对后续咨询有用的要点："
    "用户目标与关注主题、明确偏好、已确认的健康安全限制、已选择的路线或方案、未解决问题。"
    "删除寒暄、致谢、重复表述、推理过程和一般性科普内容；不确定的信息标注“待确认”，"
    "不要把猜测写成事实，不要保存敏感身份信息。\n"
    "输出要求：每行一条以“- ”开头的短句，每条不超过 40 字，最多 10 条；"
    "没有可保留的要点时输出“- 暂无”。只输出要点清单，不要标题、解释或结尾。\n\n"
    "旧摘要：\n{previous}\n\n新对话：\n{transcript}"
)


def estimate_tokens(text: str) -> int:
    cjk = sum("\u3400" <= char <= "\u9fff" for char in text)
    return cjk + (len(text) - cjk + 3) // 4


def prepare_turn(user_id: int, model) -> tuple[list[dict], str]:
    """Load durable history, summarize older turns, and return a bounded context."""
    messages = repository.load_recent_messages(user_id, limit=200)
    summary = repository.load_summary(user_id)
    kept_reversed = []
    tokens = 0
    for message in reversed(messages):
        cost = estimate_tokens(message["content"])
        if kept_reversed and (len(kept_reversed) >= RECENT_MESSAGE_LIMIT or tokens + cost > TOKEN_BUDGET):
            break
        kept_reversed.append(message)
        tokens += cost
    recent = list(reversed(kept_reversed))
    older_count = len(messages) - len(recent)
    older = messages[:older_count]

    if older and model is not None:
        transcript = "\n".join(
            f"{'用户' if item['role'] == 'user' else '助手'}：{item['content']}"
            for item in older
        )
        try:
            updated = (SUMMARY_PROMPT | model | StrOutputParser()).invoke(
                {"previous": summary or "（无）", "transcript": transcript}
            ).strip()
            if updated:
                summary = updated[:MAX_SUMMARY_CHARS]
                repository.save_summary_and_prune(user_id, summary, recent[0]["id"])
                recent = [{"role": item["role"], "content": item["content"]} for item in recent]
        except Exception:
            recent = [{"role": item["role"], "content": item["content"]} for item in recent]
    else:
        recent = [{"role": item["role"], "content": item["content"]} for item in recent]

    return recent, summary


def update_summary(user_id: int, previous: str, messages: list[dict], answer: str, model) -> str:
    transcript = "\n".join(
        f"{'用户' if item['role'] == 'user' else '助手'}：{item['content'][:3000]}"
        for item in messages[-16:]
        if item.get("role") in {"user", "assistant"} and item.get("content")
    )
    transcript += f"\n助手：{answer[:5000]}"
    try:
        updated = (SUMMARY_PROMPT | model | StrOutputParser()).invoke(
            {"previous": previous or "（无）", "transcript": transcript}
        ).strip()
        if updated:
            summary = updated[:MAX_SUMMARY_CHARS]
            repository.save_summary(user_id, summary)
            return summary
    except Exception:
        pass
    return previous
