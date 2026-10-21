from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from db import repository

TOKEN_BUDGET = 2800
RECENT_MESSAGE_LIMIT = 16
MAX_SUMMARY_CHARS = 400
SUMMARY_COMPRESSION_THRESHOLD = 320
SUMMARY_COMPRESSION_TARGET = 280
SUMMARY_PROMPT = PromptTemplate.from_template(
    "你负责根据用户与助手的历史对话，生成供后续对话理解上下文的简洁记忆摘要。"
    "只依据用户历史中明确表达的内容，归纳用户持续关注的主题、明确提出的需求与偏好、主动提供且有助于后续回答的重要背景；"
    "仅在用户明确采纳、追问或据此作出决定时，简要保留助手提出的相关建议；不要把助手推测或建议写成用户事实。"
    "只保留未来交流可能用到的信息，合并重复内容，忽略寒暄、致谢、无关细节和助手的推理过程。"
    "旧摘要仅作为事实素材，不得沿用旧摘要的列表、标题、分栏或待确认格式；每次都要重新组织并改写为自然、连贯的中文短段落。"
    "不要按预设类别拆分，不要添加标题、栏目、待办或待确认清单。"
    "不得记录用户尚未提及、尚未决定、待确认、未解决或需要后续追问的事项；不要从信息缺失推测用户需求或偏好。"
    "严格区分用户明确说过的事实与助手提出的建议，不把推测写成事实；不要保留敏感身份信息。"
    "总长度不超过 400 字。没有值得保留的内容时输出“暂无”。只输出摘要正文。\n\n"
    "旧摘要：\n{previous}\n\n新对话：\n{transcript}"
)
SUMMARY_COMPRESSION_PROMPT = PromptTemplate.from_template(
    "请压缩下面这段对话记忆摘要，保留用户明确表达且对后续回答有用的信息，"
    "优先保留持续关注主题、明确需求和偏好、重要背景及已采纳的决定；合并重复内容，删去次要细节。"
    "不得补充推测，不得写待确认事项，不要标题或列表，输出自然连贯的中文短段落。"
    "压缩到不超过 {target} 字，只输出压缩后的摘要。\n\n摘要：\n{summary}"
)


def _fit_summary(summary: str) -> str:
    text = summary.strip()
    if len(text) <= MAX_SUMMARY_CHARS:
        return text
    clipped = text[:MAX_SUMMARY_CHARS]
    sentence_end = max(clipped.rfind(mark) for mark in "。！？；")
    if sentence_end >= MAX_SUMMARY_CHARS // 2:
        return clipped[:sentence_end + 1].rstrip()
    return clipped.rstrip()


def _generate_summary(previous: str, transcript: str, model) -> str:
    summary = (SUMMARY_PROMPT | model | StrOutputParser()).invoke(
        {"previous": previous or "（无）", "transcript": transcript}
    ).strip()
    if len(summary) > SUMMARY_COMPRESSION_THRESHOLD:
        summary = (SUMMARY_COMPRESSION_PROMPT | model | StrOutputParser()).invoke(
            {"target": SUMMARY_COMPRESSION_TARGET, "summary": summary}
        ).strip()
    return _fit_summary(summary)


def estimate_tokens(text: str) -> int:
    cjk = sum("\u3400" <= char <= "\u9fff" for char in text)
    return cjk + (len(text) - cjk + 3) // 4


def prepare_turn(user_id: int, model) -> tuple[list[dict], str]:
    """加载持久化历史、摘要较早轮次，并返回长度受限的上下文。"""
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
            updated = _generate_summary(summary, transcript, model)
            if updated:
                summary = updated
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
        updated = _generate_summary(previous, transcript, model)
        if updated:
            summary = updated
            repository.save_summary(user_id, summary)
            return summary
    except Exception:
        pass
    return previous


def rewrite_summary(user_id: int, previous: str, model) -> str:
    """把已保存的旧格式摘要改写成自然段，不引入新的用户事实。"""
    if not previous.strip():
        return previous
    try:
        updated = _generate_summary(
            previous,
            "本次没有新增对话。请仅依据旧摘要中已有的信息重新组织内容。",
            model,
        ).strip()
        if updated:
            summary = updated
            repository.save_summary(user_id, summary)
            return summary
    except Exception:
        pass
    return previous
