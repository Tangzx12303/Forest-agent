import json
import os
import secrets
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import SystemMessage
from pydantic import BaseModel, Field
from starlette.middleware.sessions import SessionMiddleware

from agent.agent import build_agent
from agent.route_planner import select_routes
from agent.tools import TOOL_LABELS, build_map_payload
from config.settings import (
    PROJECT_ROOT,
    RAG_SETTINGS,
    TOOL_ROUTER_HINT_THRESHOLD,
    TRAINING_CAPTURE_ENABLED,
    list_model_presets,
)
from db import repository
from db.init_db import admin_count, initialize_database
from memory.conversation import update_summary
from model.factory import build_chat_model
from rag.vector_store import VectorStoreService
from services.virtual_map import calculate_virtual_route
from training.capture import append_turn
from training.inference import get_tool_router
from training.schema import TurnRecord, evaluate_turn, format_checks

FRONTEND_DIR = PROJECT_ROOT / "frontend"


@asynccontextmanager
async def lifespan(application: FastAPI):
    try:
        initialize_database()
        application.state.database_error = ""
    except Exception as exc:
        application.state.database_error = str(exc)
    yield


app = FastAPI(title="森语森林康养咨询 API", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    SessionMiddleware,
    secret_key=os.getenv("APP_SESSION_SECRET") or secrets.token_urlsafe(32),
    same_site="lax",
    https_only=os.getenv("APP_COOKIE_SECURE", "false").lower() == "true",
)

def current_user(request: Request) -> dict:
    user = request.session.get("user")
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="请先登录。")
    return user


def administrator(user: dict = Depends(current_user)) -> dict:
    if user.get("role") != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="需要管理员权限。")
    return user


@lru_cache(maxsize=1)
def get_vector_store() -> VectorStoreService:
    return VectorStoreService()


@lru_cache(maxsize=8)
def get_model(model_id: str):
    return build_chat_model(model_id)


@lru_cache(maxsize=8)
def get_agent(model_id: str):
    return build_agent(get_model(model_id), get_vector_store())


def build_eval_compact(checks: dict, used_tools: list[str], quality_score: float | None) -> dict:
    """构造可下发给前端 / 落库的测试评价精简结构。"""
    return {
        "overall_pass": bool(checks.get("overall_pass")),
        "tools": [{"name": name, "label": TOOL_LABELS.get(name, name)} for name in used_tools],
        "quality_score": quality_score,
        "checks": format_checks(checks),
    }


def decorate_eval_record(record: dict) -> dict:
    """把落库的质检记录解析成前端可读结构。"""
    item = dict(record)
    item["created_at"] = str(item.get("created_at") or "")
    try:
        item["detail"] = json.loads(item.get("detail_json") or "{}")
    except (TypeError, ValueError):
        item["detail"] = {}
    item.pop("detail_json", None)
    return item


class Credentials(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=256)


class Registration(Credentials):
    display_name: str = Field(default="", max_length=100)


class FirstAdmin(Registration):
    pass


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=12000)
    model_id: str
    history: list[dict[str, str]] = Field(default_factory=list, max_length=16)
    category: str = "forest"


class RoutePreferences(BaseModel):
    goals: list[str] = Field(default_factory=list, max_length=9)
    duration_minutes: int = Field(ge=15, le=600)
    mobility: str = "常规步行"


class WalkingRequest(BaseModel):
    start_zone: str
    end_zone: str


class NewAccount(BaseModel):
    username: str
    password: str
    display_name: str = ""
    role: str = "reader"


class AccountUpdate(BaseModel):
    status: str | None = None
    role: str | None = None
    password: str | None = None


@app.get("/api/status")
def app_status() -> dict:
    try:
        count = admin_count()
        database_error = ""
    except Exception as exc:
        count = 0
        database_error = str(exc)
    return {
        "ready": not database_error,
        "database_error": database_error,
        "needs_first_admin": count == 0 and not database_error,
    }


@app.post("/api/auth/login")
def login(payload: Credentials, request: Request) -> dict:
    user = repository.authenticate(payload.username, payload.password)
    if not user:
        raise HTTPException(status_code=401, detail="用户名、密码错误或账号已停用。")
    repository.clear_messages(user["id"])
    request.session["user"] = user
    return {"user": user}


@app.post("/api/auth/register", status_code=201)
def register(payload: Registration) -> dict:
    try:
        repository.register_user(payload.username, payload.password, payload.display_name)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"message": "注册成功，请登录。"}


@app.post("/api/auth/first-admin", status_code=201)
def create_first_admin(payload: FirstAdmin) -> dict:
    try:
        if admin_count() > 0:
            raise HTTPException(status_code=409, detail="管理员账号已初始化。")
        repository.create_first_admin(payload.username, payload.password, payload.display_name)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"message": "管理员已创建，请登录。"}


@app.post("/api/auth/logout")
def logout(request: Request) -> dict:
    request.session.clear()
    return {"message": "已退出登录。"}


@app.get("/api/auth/me")
def me(user: dict = Depends(current_user)) -> dict:
    repository.clear_messages(user["id"])
    try:
        summary = repository.load_summary(user["id"])
    except Exception:
        summary = ""
    return {
        "user": user,
        "messages": [],
        "summary": summary,
    }


@app.get("/api/models")
def models(_: dict = Depends(current_user)) -> dict:
    return {
        "models": [
            {key: item.get(key, "") for key in ("id", "label", "provider")}
            for item in list_model_presets()
        ]
    }


@app.post("/api/chat")
def chat(payload: ChatRequest, user: dict = Depends(current_user)) -> StreamingResponse:
    if user.get("role") == "admin":
        raise HTTPException(status_code=403, detail="管理员账号不提供对话功能，请使用读者账号咨询。")
    category_prompts = {
        "forest": "本轮属于森林知识问答。回答森林生态、植物、森林环境、基地分区与地图知识，优先依据检索资料；需要地图或路线时，可调用地图和路线工具，不要编造资料外的事实。",
        "health": "本轮属于身体健康问答。提供有依据的一般健康科普和安全建议，不作诊断或治疗方案；涉及步行能力时可调用路线工具，但不要给出个体化康复处方。",
        "lifestyle": "本轮属于健康生活建议。围绕睡眠、活动、饮食、压力管理等给出可执行的一般建议；涉及基地活动时，可调用地图和路线工具。",
    }
    if payload.category not in category_prompts:
        raise HTTPException(status_code=400, detail="咨询分类无效。")
    allowed_models = {item["id"] for item in list_model_presets()}
    if payload.model_id not in allowed_models:
        raise HTTPException(status_code=400, detail="所选模型不可用。")
    try:
        summary = repository.load_summary(user["id"])
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"咨询服务暂不可用：{exc}") from exc

    history = []
    history_size = 0
    for item in reversed(payload.history[-16:]):
        if item.get("role") not in {"user", "assistant"} or not item.get("content"):
            continue
        remaining = 24000 - history_size
        if remaining <= 0:
            break
        content = item["content"][:min(12000, remaining)]
        history.append({"role": item["role"], "content": content})
        history_size += len(content)
    history.reverse()
    if not history or history[-1]["role"] != "user" or history[-1]["content"] != payload.message:
        history.append({"role": "user", "content": payload.message})
    context_messages = [SystemMessage(content=category_prompts[payload.category])]
    if summary:
        context_messages.append(SystemMessage(content=f"之前对话摘要（仅供理解上下文）：\n{summary}"))
    try:
        tool_hints = get_tool_router().predict_tools(
            payload.message, threshold=TOOL_ROUTER_HINT_THRESHOLD
        )
    except Exception:
        tool_hints = []
    if tool_hints:
        hint_names = "、".join(item["tool"] for item in tool_hints)
        context_messages.append(SystemMessage(
            content=(
                f"本地工具路由建议（仅供参考，不要向用户提及）：{hint_names}。"
                "请根据问题实际需要决定是否调用，不要为了使用工具而偏离问题。"
            )
        ))
    context_messages.extend(history)

    def stream_response():
        answer_parts = []
        used_tools: list[str] = []
        seen_tool_calls: set[str] = set()
        map_emitted = False
        try:
            yield f"data: {json.dumps({'type': 'status', 'text': '正在连接模型…'}, ensure_ascii=False)}\n\n"
            model = get_model(payload.model_id)
            agent = get_agent(payload.model_id)
            for chunk, metadata in agent.stream({"messages": context_messages}, stream_mode="messages"):
                tool_chunks = getattr(chunk, "tool_call_chunks", None)
                if tool_chunks:
                    for call in tool_chunks:
                        name = call.get("name") or ""
                        key = str(call.get("id") or f"{name}:{call.get('index', 0)}")
                        if not name or key in seen_tool_calls:
                            continue
                        seen_tool_calls.add(key)
                        used_tools.append(name)
                        label = TOOL_LABELS.get(name, name)
                        yield f"data: {json.dumps({'type': 'tool', 'name': name, 'label': label}, ensure_ascii=False)}\n\n"
                        if name == "show_virtual_map" and not map_emitted:
                            map_emitted = True
                            yield f"data: {json.dumps({'type': 'map', 'map': build_map_payload()}, ensure_ascii=False)}\n\n"
                    continue
                if metadata.get("langgraph_node") != "model":
                    continue
                content = chunk.content
                if isinstance(content, list):
                    content = "".join(
                        block.get("text", "") for block in content
                        if isinstance(block, dict) and block.get("type") == "text"
                    )
                if isinstance(content, str) and content:
                    answer_parts.append(content)
                    yield f"data: {json.dumps({'type': 'token', 'text': content}, ensure_ascii=False)}\n\n"
            answer = "".join(answer_parts)
            if not answer:
                answer = "暂时没有生成可展示的文字，请换个问法后重试。"
                yield f"data: {json.dumps({'type': 'token', 'text': answer}, ensure_ascii=False)}\n\n"
            # 每轮对话始终计算规则化测试评价并下发 + 落库；质检是旁路，任何一步失败都不影响对话。
            checks = evaluate_turn(payload.message, payload.category, used_tools, answer)
            quality_score = None
            try:
                quality_score = get_tool_router().score_quality(payload.message, answer)
            except Exception:
                quality_score = None
            if quality_score is not None:
                checks["model_quality_score"] = quality_score
            try:
                yield f"data: {json.dumps({'type': 'eval', 'eval': build_eval_compact(checks, used_tools, quality_score)}, ensure_ascii=False)}\n\n"
            except Exception:
                pass
            try:
                repository.insert_eval_record(
                    user["id"],
                    payload.model_id,
                    payload.message[:512],
                    payload.category,
                    1 if checks.get("overall_pass") else 0,
                    ",".join(used_tools),
                    quality_score,
                    checks.get("answer_length", 0),
                    json.dumps(build_eval_compact(checks, used_tools, quality_score), ensure_ascii=False),
                )
            except Exception:
                pass
            if TRAINING_CAPTURE_ENABLED:
                try:
                    append_turn(TurnRecord(
                        query=payload.message,
                        category=payload.category,
                        tools_used=used_tools,
                        answer=answer,
                        checks=checks,
                        model_id=payload.model_id,
                    ))
                except Exception:
                    pass
            updated_summary = update_summary(user["id"], summary, history, answer, model)
            yield f"data: {json.dumps({'type': 'summary', 'summary': updated_summary}, ensure_ascii=False)}\n\n"
            yield 'data: {"type":"done"}\n\n'
        except Exception as exc:
            yield f"data: {json.dumps({'type': 'error', 'message': f'咨询服务暂不可用：{exc}'}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        stream_response(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@app.delete("/api/chat")
def clear_chat(user: dict = Depends(current_user)) -> dict:
    repository.clear_conversation(user["id"])
    return {"message": "对话和摘要已清除。"}


@app.get("/api/chat/evaluations")
def my_evaluations(user: dict = Depends(current_user)) -> dict:
    try:
        records = repository.list_eval_records_for_user(user["id"], limit=50)
    except Exception:
        records = []
    return {"evaluations": [decorate_eval_record(record) for record in records]}


@app.get("/api/routes/map")
def route_map(_: dict = Depends(current_user)) -> dict:
    return build_map_payload()


@app.post("/api/routes/recommend")
def recommend_route(payload: RoutePreferences, _: dict = Depends(current_user)) -> dict:
    _, routes = select_routes(",".join(payload.goals), payload.duration_minutes, payload.mobility)
    return {"routes": routes}


@app.post("/api/routes/walking")
def walking_route(payload: WalkingRequest, _: dict = Depends(current_user)) -> dict:
    try:
        route = calculate_virtual_route(payload.start_zone, payload.end_zone)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return route


@app.get("/api/knowledge")
def knowledge_files(_: dict = Depends(current_user)) -> dict:
    directory = Path(RAG_SETTINGS["knowledge_dir"])
    files = [
        {"name": path.name, "path": path.relative_to(directory).as_posix(), "type": path.suffix.lower()}
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path.suffix.lower() in {".txt", ".md", ".pdf"}
    ]
    return {"files": files}


@app.post("/api/knowledge/index")
def index_knowledge(_: dict = Depends(administrator)) -> dict:
    try:
        count = get_vector_store().index_documents()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"索引失败：{exc}") from exc
    return {"indexed_chunks": count}


@app.get("/api/admin/users")
def admin_users(_: dict = Depends(administrator)) -> dict:
    users = repository.list_users()
    for user in users:
        user["created_at"] = str(user.get("created_at") or "")
    return {"users": users}


@app.get("/api/admin/summaries")
def admin_reader_summaries(_: dict = Depends(administrator)) -> dict:
    summaries = repository.list_reader_summaries()
    for item in summaries:
        item["updated_at"] = str(item.get("updated_at") or "")
    return {"summaries": summaries}


@app.get("/api/admin/reports")
def admin_test_reports(_: dict = Depends(administrator)) -> dict:
    try:
        report = repository.eval_report()
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"测试报告暂不可用：{exc}") from exc
    try:
        records = repository.list_eval_records(limit=200)
    except Exception:
        records = []
    readers = []
    for item in report.get("readers") or []:
        reader = dict(item)
        reader["last_at"] = str(reader.get("last_at") or "")
        readers.append(reader)
    return {
        "overall": report.get("overall") or {},
        "readers": readers,
        "records": [decorate_eval_record(record) for record in records],
    }


@app.post("/api/admin/users", status_code=201)
def admin_create_user(payload: NewAccount, _: dict = Depends(administrator)) -> dict:
    try:
        repository.create_user(payload.username, payload.password, payload.display_name, payload.role)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"message": "账号已创建。"}


@app.patch("/api/admin/users/{user_id}")
def admin_update_user(user_id: int, payload: AccountUpdate, _: dict = Depends(administrator)) -> dict:
    try:
        if payload.status is not None:
            repository.set_user_status(user_id, payload.status)
        if payload.role is not None:
            repository.set_user_role(user_id, payload.role)
        if payload.password is not None:
            repository.reset_user_password(user_id, payload.password)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"message": "账号已更新。"}


@app.delete("/api/admin/users/{user_id}")
def admin_delete_user(user_id: int, current: dict = Depends(administrator)) -> dict:
    if user_id == current["id"]:
        raise HTTPException(status_code=400, detail="不能删除当前登录的管理员账号。")
    repository.delete_user(user_id)
    return {"message": "账号已删除。"}


app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/", include_in_schema=False)
def frontend_index() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/{path:path}", include_in_schema=False)
def frontend_fallback(path: str) -> FileResponse:
    del path
    return FileResponse(FRONTEND_DIR / "index.html")
