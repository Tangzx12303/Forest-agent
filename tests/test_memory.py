from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from memory import conversation


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
    monkeypatch.setattr(
        conversation.repository,
        "save_summary",
        lambda user_id, summary: saved.append((user_id, summary)),
    )
    model = RunnableLambda(lambda _prompt: AIMessage(content="新的滚动摘要"))

    summary = conversation.update_summary(
        12,
        "旧摘要",
        [{"role": "user", "content": "想走轻松路线"}],
        "建议从入口区慢行。",
        model,
    )

    assert summary == "新的滚动摘要"
    assert saved == [(12, "新的滚动摘要")]
