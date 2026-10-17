from pathlib import Path

from langchain_core.documents import Document

from config.settings import RAG_SETTINGS
from rag.vector_store import VectorStoreService


def knowledge_files() -> list[Path]:
    directory = Path(RAG_SETTINGS["knowledge_dir"])
    return sorted(path for path in directory.glob("*.txt") if path.is_file())


def test_knowledge_base_is_structured_and_cited():
    files = knowledge_files()

    assert len(files) >= 8
    for path in files:
        content = path.read_text(encoding="utf-8")
        assert len(content) >= 800, f"{path.name} 内容过短，未达到结构化知识质量要求"
        assert "主题：" in content, f"{path.name} 缺少主题行"
        assert "参考" in content, f"{path.name} 缺少参考资料"
        assert "http" in content, f"{path.name} 缺少可追溯来源链接"


def test_title_metadata_extraction_supports_topic_and_markdown_heading():
    documents = [
        Document(page_content="\n主题：森林环境与健康证据\n正文", metadata={}),
        Document(page_content="# 户外安全\n正文", metadata={}),
        Document(page_content="没有标题的文本", metadata={}),
    ]

    assert VectorStoreService._title([documents[0]], "fallback") == "森林环境与健康证据"
    assert VectorStoreService._title([documents[1]], "fallback") == "户外安全"
    assert VectorStoreService._title([documents[2]], "fallback") == "没有标题的文本"
    assert VectorStoreService._title([], "fallback") == "fallback"
