import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env", override=False)


def _load_yaml(filename: str) -> dict:
    with (PROJECT_ROOT / "config" / filename).open(encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def load_rag_settings() -> dict:
    settings = _load_yaml("rag.yml")
    settings["persist_directory"] = str(PROJECT_ROOT / settings["persist_directory"])
    settings["knowledge_dir"] = str(PROJECT_ROOT / settings["knowledge_dir"])
    return settings


RAG_SETTINGS = load_rag_settings()
CHAT_MODEL_NAME = os.getenv("CHAT_MODEL_NAME", "qwen-plus")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-v4")
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "").strip()
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
BASE_ZONES_PATH = PROJECT_ROOT / "data" / "base" / "base_zones.json"
SYSTEM_PROMPT_PATH = PROJECT_ROOT / "prompts" / "system_prompt.txt"
TRAINING_CAPTURE_ENABLED = os.getenv("TRAINING_CAPTURE", "false").strip().lower() in {
    "1", "true", "yes", "on"
}
TRAINING_LOG_PATH = PROJECT_ROOT / os.getenv("TRAINING_LOG_PATH", "data/training/turns.jsonl")
TRAINING_SAMPLES_PATH = PROJECT_ROOT / "data" / "training" / "router_samples.jsonl"
TOOL_ROUTER_MODEL_PATH = PROJECT_ROOT / os.getenv("TOOL_ROUTER_MODEL", "data/training/tool_router.pt")
TOOL_ROUTER_HINT_THRESHOLD = float(os.getenv("TOOL_ROUTER_HINT_THRESHOLD", "0.35"))
MYSQL_HOST = os.getenv("MYSQL_HOST", "127.0.0.1")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", "3306"))
MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "forest_wellness")


def list_model_presets() -> list[dict[str, str]]:
    configured = _load_yaml("models.yml")
    return [
        item for item in configured.get("models", [])
        if os.getenv(item["api_key_env"], "").strip()
    ]


def default_model_id() -> str:
    available = list_model_presets()
    configured_default = os.getenv("CHAT_MODEL_NAME", "") or _load_yaml("models.yml").get("default_model", "")
    if any(item["id"] == configured_default for item in available):
        return configured_default
    return available[0]["id"] if available else ""
