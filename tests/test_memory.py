from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from memory import conversation


def test_summary_prompt_prefers_user_history_and_natural_prose():
    prompt = conversation.SUMMARY_PROMPT.template

    assert "只依据用户历史中明确表达的内容" in prompt
    assert "旧摘要仅作为事实素材" in prompt
    assert "每次都要重新组织并改写" in prompt
    assert "不要按预设类别拆分" in prompt
    assert "不要添加标题、栏目、待办或待确认清单" in prompt
    assert "不得记录用户尚未提及、尚未决定、待确认、未解决或需要后续追问的事项" in prompt


def test_prepare_turn_summarizes_and_prunes_older_history(monkeypatch):
    messages = [
        {"id": index, "role": role, "content": f"消息内容 {index}", "model_id": "qwen-plus"}
        for index, role in enumerate(["user", "assistant", "user", "assistant"], 1)
    ]
    pruned = []
    monkeypatch.setattr(conversation, "RECENT_MESSAGE_LIMIT", 2)
    monkeypatch.setattr(conversation.repository, "load_recent_messages", lambda *_args, **_kwargs: messages)
    monkeypatch.setattr(conversation.repository, "load_summary", lambda _user_id: "旧摘要")
    monkeypatch.setattr(
        conversation.repository,
        "save_summary_and_prune",
        lambda user_id, summary, before_id: pruned.append((user_id, summary, before_id)),
    )
    model = RunnableLambda(lambda _prompt: AIMessage(content="合并后的摘要"))

    recent, summary = conversation.prepare_turn(7, model)

    assert [item["content"] for item in recent] == ["消息内容 3", "消息内容 4"]
    assert summary == "合并后的摘要"
    assert pruned == [(7, "合并后的摘要", 3)]


def test_token_estimate_counts_cjk_conservatively():
    assert conversation.estimate_tokens("森林 walking") == 4


def test_update_summary_persists_only_the_compacted_memory(monkeypatch):
    saved = []
    captured = {}
    monkeypatch.setattr(
        conversation.repository,
        "save_summary",
        lambda user_id, summary: saved.append((user_id, summary)),
    )

    def summarize(prompt):
        captured["prompt"] = prompt.to_string()
        return AIMessage(content="用户关注轻松的森林步行，并希望活动安排循序渐进。")

    model = RunnableLambda(summarize)

    summary = conversation.update_summary(
        12,
        "用户偏好：轻松路线\n待确认：每周活动次数",
        [{"role": "user", "content": "想走轻松路线"}],
        "建议从入口区慢行。",
        model,
    )

    assert "旧摘要仅作为事实素材" in captured["prompt"]
    assert "用户偏好：轻松路线" in captured["prompt"]
    assert "不得记录用户尚未提及、尚未决定、待确认、未解决或需要后续追问的事项" in captured["prompt"]
    assert summary == "用户关注轻松的森林步行，并希望活动安排循序渐进。"
    assert saved == [(12, summary)]


def test_rewrite_summary_reformats_saved_legacy_content(monkeypatch):
    saved = []
    monkeypatch.setattr(
        conversation.repository,
        "save_summary",
        lambda user_id, summary: saved.append((user_id, summary)),
    )
    model = RunnableLambda(lambda _prompt: AIMessage(content="用户偏好轻松的森林步行。"))

    summary = conversation.rewrite_summary(
        14,
        "- 用户偏好：轻松路线\n- 待确认：活动频率",
        model,
    )

    assert summary == "用户偏好轻松的森林步行。"
    assert saved == [(14, summary)]


def test_update_summary_compresses_when_summary_exceeds_threshold(monkeypatch):
    saved = []
    prompts = []
    long_summary = "用户明确表达了森林活动和健康生活方面的持续关注。" * 45
    compressed_summary = "用户关注森林活动及健康生活。" * 20

    def summarize(prompt):
        prompts.append(prompt.to_string())
        if len(prompts) == 1:
            return AIMessage(content=long_summary)
        return AIMessage(content=compressed_summary)

    monkeypatch.setattr(
        conversation.repository,
        "save_summary",
        lambda user_id, summary: saved.append((user_id, summary)),
    )
    model = RunnableLambda(summarize)

    summary = conversation.update_summary(
        15,
        "旧摘要",
        [{"role": "user", "content": "我关注森林活动和健康生活"}],
        "可以从轻松步行开始。",
        model,
    )

    assert len(prompts) == 2
    assert "不超过 280 字" in prompts[1]
    assert long_summary in prompts[1]
    assert len(summary) <= conversation.MAX_SUMMARY_CHARS
    assert summary == compressed_summary
    assert saved == [(15, summary)]


def test_summary_hard_limit_does_not_exceed_maximum():
    summary = conversation._fit_summary("用户关注森林活动。" * 100)

    assert len(summary) <= conversation.MAX_SUMMARY_CHARS
    assert summary.endswith("。")
