from db.connection import execute, query_all, query_one
from utils.security import hash_password, verify_password


def create_user(username: str, password: str, display_name: str = "", role: str = "reader") -> int:
    if role not in {"reader", "admin"}:
        raise ValueError("Invalid role")
    if len(username.strip()) < 3 or len(username.strip()) > 64:
        raise ValueError("用户名需为 3 至 64 个字符。")
    if len(password) < 10:
        raise ValueError("密码至少需要 10 个字符。")
    return execute(
        "INSERT INTO users (username, display_name, password_hash, role) VALUES (%s,%s,%s,%s)",
        (username.strip(), display_name.strip(), hash_password(password), role),
    )


def seed_demo_account(
    username: str,
    display_name: str,
    role: str,
    legacy_usernames: tuple[str, ...] = (),
    force: bool = False,
) -> str:
    """使用专供本地演示的弱密码创建或迁移已知演示账号。

    同名账号已被非演示账号占用时默认跳过。传入 ``force=True`` 时强制把该账号改为演示账号，
    并停用同时存在的旧演示账号，适合在本机演示库中修复无法登录的演示管理员。
    """
    if not username.isascii() or not username.isalpha() or role not in {"reader", "admin"}:
        raise ValueError("演示账号名只能使用英文字母，角色无效。")

    from db.connection import get_connection

    connection = get_connection()
    try:
        connection.begin()
        with connection.cursor() as cursor:
            names = (username, *legacy_usernames)
            placeholders = ",".join(["%s"] * len(names))
            cursor.execute(
                f"SELECT id,username,display_name FROM users WHERE username IN ({placeholders}) FOR UPDATE",
                names,
            )
            matches = cursor.fetchall()
            target = next((item for item in matches if item["username"] == username), None)
            legacy = next((item for item in matches if item["username"] in legacy_usernames), None)

            if target and target["id"] != (legacy["id"] if legacy else None):
                if target["display_name"] != display_name and not force:
                    connection.rollback()
                    return "conflict"
                account_id = target["id"]
                result = "reset" if force else "updated"
            elif legacy:
                account_id = legacy["id"]
                result = "migrated"
            else:
                cursor.execute(
                    "INSERT INTO users (username,display_name,password_hash,role,status) "
                    "VALUES (%s,%s,%s,%s,'active')",
                    (username, display_name, hash_password("123456"), role),
                )
                connection.commit()
                return "created"

            cursor.execute(
                "UPDATE users SET username=%s,display_name=%s,password_hash=%s,role=%s,status='active' WHERE id=%s",
                (username, display_name, hash_password("123456"), role, account_id),
            )
            if force and legacy and legacy["id"] != account_id:
                # 演示账号已并入目标用户名，停用旧的重复演示账号，保留其历史数据。
                cursor.execute("UPDATE users SET status='disabled' WHERE id=%s", (legacy["id"],))
        connection.commit()
        return result
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def create_first_admin(username: str, password: str, display_name: str = "") -> None:
    if len(username.strip()) < 3 or len(password) < 10:
        raise ValueError("管理员用户名至少 3 个字符，密码至少 10 个字符。")
    from db.connection import get_connection

    connection = get_connection()
    lock_acquired = False
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT GET_LOCK('forest_agent_first_admin', 10) AS acquired")
            lock_acquired = bool(cursor.fetchone()["acquired"])
            if not lock_acquired:
                raise RuntimeError("无法获取管理员初始化锁，请稍后重试。")
            cursor.execute("SELECT COUNT(*) AS count FROM users WHERE role='admin'")
            if cursor.fetchone()["count"]:
                raise ValueError("管理员账号已初始化，请通过管理员登录。")
            cursor.execute(
                "INSERT INTO users (username,display_name,password_hash,role) VALUES (%s,%s,%s,'admin')",
                (username.strip(), display_name.strip(), hash_password(password)),
            )
    finally:
        if lock_acquired:
            with connection.cursor() as cursor:
                cursor.execute("SELECT RELEASE_LOCK('forest_agent_first_admin')")
        connection.close()


def register_user(username: str, password: str, display_name: str = "") -> int:
    if len(username.strip()) < 3 or len(username.strip()) > 64:
        raise ValueError("用户名需为 3 至 64 个字符。")
    if len(password) < 10:
        raise ValueError("密码至少需要 10 个字符。")
    return create_user(username, password, display_name)


def authenticate(username: str, password: str) -> dict | None:
    user = query_one(
        "SELECT id, username, display_name, password_hash, role, status FROM users WHERE username=%s",
        (username.strip(),),
    )
    if not user or user["status"] != "active" or not verify_password(password, user["password_hash"]):
        return None
    return {key: user[key] for key in ("id", "username", "display_name", "role")}


def list_users() -> list[dict]:
    return query_all("SELECT id,username,display_name,role,status,created_at FROM users ORDER BY id")


def list_reader_summaries() -> list[dict]:
    """列出读者账号及其滚动摘要，不返回完整对话记录。"""
    return query_all(
        "SELECT u.id,u.username,u.display_name,u.status,COALESCE(m.summary,'') AS summary,m.updated_at "
        "FROM users u LEFT JOIN conversation_memory m ON m.user_id=u.id "
        "WHERE u.role='reader' ORDER BY m.updated_at DESC,u.id"
    )


def set_user_status(user_id: int, status: str) -> None:
    if status not in {"active", "disabled"}:
        raise ValueError("Invalid account status")
    execute("UPDATE users SET status=%s WHERE id=%s", (status, user_id))


def set_user_role(user_id: int, role: str) -> None:
    if role not in {"reader", "admin"}:
        raise ValueError("Invalid role")
    execute("UPDATE users SET role=%s WHERE id=%s", (role, user_id))


def reset_user_password(user_id: int, password: str) -> None:
    if len(password) < 10:
        raise ValueError("密码至少需要 10 个字符。")
    execute("UPDATE users SET password_hash=%s WHERE id=%s", (hash_password(password), user_id))


def delete_user(user_id: int) -> None:
    execute("DELETE FROM users WHERE id=%s", (user_id,))


def load_recent_messages(user_id: int, limit: int = 60) -> list[dict]:
    return query_all(
        "SELECT id,role,content,model_id FROM (SELECT id,role,content,model_id,created_at "
        "FROM chat_messages WHERE user_id=%s ORDER BY id DESC LIMIT %s) recent ORDER BY id",
        (user_id, limit),
    )


def append_message(user_id: int, role: str, content: str, model_id: str) -> None:
    if role not in {"user", "assistant"}:
        raise ValueError("Invalid chat role")
    execute("INSERT INTO chat_messages (user_id,role,content,model_id) VALUES (%s,%s,%s,%s)",
            (user_id, role, content, model_id))


def clear_messages(user_id: int) -> None:
    execute("DELETE FROM chat_messages WHERE user_id=%s", (user_id,))


def delete_messages_before(user_id: int, message_id: int) -> None:
    execute("DELETE FROM chat_messages WHERE user_id=%s AND id<%s", (user_id, message_id))


def clear_conversation(user_id: int) -> None:
    clear_messages(user_id)
    delete_summary(user_id)


def load_summary(user_id: int) -> str:
    row = query_one("SELECT summary FROM conversation_memory WHERE user_id=%s", (user_id,))
    return row["summary"] if row else ""


def save_summary(user_id: int, summary: str) -> None:
    execute("INSERT INTO conversation_memory (user_id,summary) VALUES (%s,%s) "
            "ON DUPLICATE KEY UPDATE summary=VALUES(summary)", (user_id, summary))


def delete_summary(user_id: int) -> None:
    execute("DELETE FROM conversation_memory WHERE user_id=%s", (user_id,))


def save_summary_and_prune(user_id: int, summary: str, before_message_id: int) -> None:
    from db.connection import get_connection

    connection = get_connection()
    try:
        connection.begin()
        with connection.cursor() as cursor:
            cursor.execute(
                "INSERT INTO conversation_memory (user_id,summary) VALUES (%s,%s) "
                "ON DUPLICATE KEY UPDATE summary=VALUES(summary)",
                (user_id, summary),
            )
            cursor.execute("DELETE FROM chat_messages WHERE user_id=%s AND id<%s",
                           (user_id, before_message_id))
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


# ======================================================================
# 对话测试评价（质检记录）
# ======================================================================
# 每轮对话写一行，detail_json 保存可直接展示的检查项明细与工具调用标签。
# 写入失败绝不向上抛给对话流程——质检是旁路功能。

def insert_eval_record(
    user_id: int,
    model_id: str,
    query: str,
    category: str,
    passed: int,
    tools_used: str,
    model_quality_score: float | None,
    answer_length: int | None,
    detail_json: str,
) -> int:
    return execute(
        "INSERT INTO eval_turn_records "
        "(user_id, model_id, query, category, passed, tools_used, "
        " model_quality_score, answer_length, detail_json) "
        "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)",
        (user_id, model_id, query, category, passed, tools_used,
         model_quality_score, answer_length, detail_json),
    )


def list_eval_records(limit: int = 100) -> list[dict]:
    return query_all(
        "SELECT id, user_id, model_id, query, category, passed, tools_used, "
        "model_quality_score, answer_length, detail_json, created_at "
        "FROM eval_turn_records ORDER BY id DESC LIMIT %s",
        (int(limit),),
    )


def list_eval_records_for_user(user_id: int, limit: int = 50) -> list[dict]:
    return query_all(
        "SELECT id, user_id, model_id, query, category, passed, tools_used, "
        "model_quality_score, answer_length, detail_json, created_at "
        "FROM eval_turn_records WHERE user_id=%s ORDER BY id DESC LIMIT %s",
        (user_id, int(limit)),
    )


def eval_report() -> dict:
    """按读者聚合质检记录，供管理员测试报告看板使用。"""
    readers = query_all(
        "SELECT e.user_id, u.username, u.display_name, COUNT(*) AS turns, "
        "COALESCE(SUM(e.passed), 0) AS passed_turns, "
        "ROUND(AVG(e.model_quality_score), 3) AS avg_quality, "
        "MAX(e.created_at) AS last_at "
        "FROM eval_turn_records e JOIN users u ON u.id=e.user_id "
        "WHERE u.role='reader' "
        "GROUP BY e.user_id, u.username, u.display_name "
        "ORDER BY last_at DESC"
    )
    overall = query_one(
        "SELECT COUNT(*) AS total, COALESCE(SUM(passed), 0) AS passed, "
        "ROUND(AVG(model_quality_score), 3) AS avg_quality "
        "FROM eval_turn_records"
    )
    return {"overall": overall or {}, "readers": readers}
