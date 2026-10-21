""""用法：
    python run.py                 # 127.0.0.1:8000
    python run.py --open          # 服务就绪后自动打开默认浏览器
    python run.py --host 0.0.0.0  # 允许局域网访问
    python run.py --reload        # 开发热重载
"""
from __future__ import annotations

import argparse
import os
import socket
import sys
import threading
import time
import webbrowser
from pathlib import Path

import uvicorn

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

STARTUP_TIMEOUT_SECONDS = 20.0


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name, "").strip().lower()
    if not value:
        return default
    return value in {"1", "true", "yes", "on"}


def _port_is_open(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.5):
            return True
    except OSError:
        return False


def _open_browser_when_ready(url: str, port: int) -> None:
    deadline = time.monotonic() + STARTUP_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if _port_is_open(port):
            webbrowser.open(url)
            return
        time.sleep(0.25)


def main() -> None:
    parser = argparse.ArgumentParser(description="启动森语森林康养咨询服务。")
    parser.add_argument("--host", default=os.getenv("APP_HOST", "127.0.0.1"), help="监听地址，默认 127.0.0.1")
    parser.add_argument("--port", type=int, default=int(os.getenv("APP_PORT", "8000")), help="监听端口，默认 8000")
    parser.add_argument("--reload", action="store_true", help="开发模式：代码变更后自动重启")
    parser.add_argument(
        "--open",
        dest="open_browser",
        action="store_true",
        default=_env_flag("APP_OPEN_BROWSER"),
        help="服务就绪后自动打开默认浏览器",
    )
    parser.add_argument("--no-browser", dest="open_browser", action="store_false", help="不自动打开浏览器")
    args = parser.parse_args()

    url = f"http://127.0.0.1:{args.port}"
    print()
    print("=" * 62)
    print("森语 · 森林康养咨询")
    print(f"本机网址（Ctrl+Click / 单击直接打开）：{url}")
    if args.host in {"0.0.0.0", "::"}:
        print(f"局域网地址：http://<本机局域网 IP>:{args.port}")
    print(f"API 文档：{url}/docs")
    print("按 Ctrl+C 停止服务")
    print("=" * 62)
    print(flush=True)

    if args.open_browser:
        threading.Thread(
            target=_open_browser_when_ready,
            args=(url, args.port),
            daemon=True,
        ).start()

    uvicorn.run("api:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
