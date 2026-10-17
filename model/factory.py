import os

from langchain_openai import ChatOpenAI

from config.settings import list_model_presets


def build_chat_model(model_id: str, *, streaming: bool = True) -> ChatOpenAI:
    preset = next((item for item in list_model_presets() if item["id"] == model_id), None)
    if preset is None:
        raise ValueError("所选模型的供应商密钥未配置，或模型不在 config/models.yml 中。")
    api_key = os.getenv(preset["api_key_env"], "").strip()
    if not api_key:
        raise ValueError(f"缺少模型密钥环境变量：{preset['api_key_env']}")
    return ChatOpenAI(
        model=preset["model"],
        api_key=api_key,
        base_url=preset["base_url"],
        streaming=streaming,
    )
