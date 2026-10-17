from pathlib import Path

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from agent.tools import build_forest_tools
from config.settings import SYSTEM_PROMPT_PATH
from rag.vector_store import VectorStoreService


def build_agent(model: ChatOpenAI, vector_store: VectorStoreService):
    """Build a ReAct agent with the local knowledge, map, and route tools."""
    prompt = Path(SYSTEM_PROMPT_PATH).read_text(encoding="utf-8")
    return create_agent(
        model=model,
        tools=build_forest_tools(vector_store),
        system_prompt=prompt,
    )
