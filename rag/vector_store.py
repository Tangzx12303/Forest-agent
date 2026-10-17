import hashlib
import json
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.settings import DASHSCOPE_API_KEY, EMBEDDING_MODEL_NAME, RAG_SETTINGS


class VectorStoreService:
    def __init__(self) -> None:
        if not DASHSCOPE_API_KEY:
            raise ValueError("尚未配置 DASHSCOPE_API_KEY；请复制 .env.example 为 .env 并填写百炼 API Key。")
        self.embedding = DashScopeEmbeddings(model=EMBEDDING_MODEL_NAME, dashscope_api_key=DASHSCOPE_API_KEY)
        self.store = Chroma(
            collection_name=RAG_SETTINGS["collection_name"],
            embedding_function=self.embedding,
            persist_directory=RAG_SETTINGS["persist_directory"],
        )
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=RAG_SETTINGS["chunk_size"],
            chunk_overlap=RAG_SETTINGS["chunk_overlap"],
            separators=RAG_SETTINGS["separators"],
            length_function=len,
        )
        self.manifest_path = Path(RAG_SETTINGS["persist_directory"]) / "source_manifest.json"

    def _manifest(self) -> dict[str, str]:
        if not self.manifest_path.exists():
            return {}
        return json.loads(self.manifest_path.read_text(encoding="utf-8"))

    @staticmethod
    def _digest(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    @staticmethod
    def _title(documents: list[Document], fallback: str) -> str:
        for document in documents:
            for line in document.page_content.splitlines():
                line = line.strip()
                if not line:
                    continue
                if line.startswith("主题："):
                    return line.removeprefix("主题：").strip()[:120]
                if line.startswith("#"):
                    return line.lstrip("#").strip()[:120]
                return line[:120]
        return fallback

    def index_documents(self) -> int:
        """Incrementally index UTF-8 TXT/Markdown and PDF knowledge files."""
        knowledge_dir = Path(RAG_SETTINGS["knowledge_dir"])
        knowledge_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest = self._manifest()
        indexed_chunks = 0

        for path in sorted(knowledge_dir.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in {".txt", ".md", ".pdf"}:
                continue
            source = str(path.relative_to(knowledge_dir)).replace("\\", "/")
            digest = self._digest(path)
            if manifest.get(source) == digest:
                continue

            if source in manifest:
                self.store.delete(where={"source": source})

            if path.suffix.lower() == ".pdf":
                documents = PyPDFLoader(str(path)).load()
            else:
                documents = TextLoader(str(path), encoding="utf-8").load()
            title = self._title(documents, path.stem)
            for document in documents:
                document.metadata["source"] = source
                document.metadata["title"] = title
                document.metadata["topic"] = path.stem
            chunks: list[Document] = self.splitter.split_documents(documents)
            # Lightweight contextual retrieval: every chunk carries its document
            # topic, so later chunks still match topic-level questions.
            for chunk in chunks:
                chunk.metadata.setdefault("title", title)
                if not chunk.page_content.lstrip().startswith("主题："):
                    chunk.page_content = f"主题：{title}\n{chunk.page_content}"
            if chunks:
                ids = [hashlib.sha256(f"{source}:{index}:{chunk.page_content}".encode("utf-8")).hexdigest()
                       for index, chunk in enumerate(chunks)]
                self.store.add_documents(chunks, ids=ids)
            manifest[source] = digest
            self.manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
            indexed_chunks += len(chunks)

        return indexed_chunks

    def search(self, query: str, k: int | None = None) -> list[Document]:
        """Retrieve diverse chunks; fall back to similarity search for older stores."""
        limit = max(1, int(k or RAG_SETTINGS["retrieval_k"]))
        try:
            return self.store.max_marginal_relevance_search(
                query, k=limit, fetch_k=max(limit * 4, 12)
            )
        except Exception:
            return self.store.similarity_search(query, k=limit)
