import re

from config.settings import MYSQL_DATABASE, MYSQL_HOST, MYSQL_PASSWORD, MYSQL_PORT, MYSQL_USER
from db.connection import get_connection

DDL = [
    """CREATE TABLE IF NOT EXISTS users (
        id BIGINT AUTO_INCREMENT PRIMARY KEY,
        username VARCHAR(64) NOT NULL UNIQUE,
        display_name VARCHAR(100) NOT NULL DEFAULT '',
        password_hash VARCHAR(255) NOT NULL,
        role ENUM('reader','admin') NOT NULL DEFAULT 'reader',
        status ENUM('active','disabled') NOT NULL DEFAULT 'active',
        created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""",
    """CREATE TABLE IF NOT EXISTS chat_messages (
        id BIGINT AUTO_INCREMENT PRIMARY KEY,
        user_id BIGINT NOT NULL,
        role ENUM('user','assistant') NOT NULL,
        content MEDIUMTEXT NOT NULL,
        model_id VARCHAR(100) NOT NULL DEFAULT '',
        created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
        KEY idx_user_created (user_id, created_at),
        CONSTRAINT fk_chat_user FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""",
    """CREATE TABLE IF NOT EXISTS conversation_memory (
        user_id BIGINT PRIMARY KEY,
        summary MEDIUMTEXT NOT NULL,
        updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        CONSTRAINT fk_memory_user FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""",
    """CREATE TABLE IF NOT EXISTS eval_turn_records (
        id BIGINT AUTO_INCREMENT PRIMARY KEY,
        user_id BIGINT NOT NULL,
        model_id VARCHAR(100) NOT NULL DEFAULT '',
        query VARCHAR(512) NOT NULL,
        category VARCHAR(16) NOT NULL DEFAULT 'forest',
        passed TINYINT NOT NULL DEFAULT 0,
        tools_used VARCHAR(255) NOT NULL DEFAULT '',
        model_quality_score DOUBLE DEFAULT NULL,
        answer_length INT DEFAULT NULL,
        detail_json MEDIUMTEXT,
        created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
        KEY idx_eval_user (user_id),
        KEY idx_eval_created (created_at),
        CONSTRAINT fk_eval_user FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""",
]


def initialize_database() -> None:
    if not MYSQL_PASSWORD:
        raise RuntimeError("请在 .env 配置 MySQL 密码 MYSQL_PASSWORD。")
    if not re.fullmatch(r"[A-Za-z0-9_]+", MYSQL_DATABASE):
        raise ValueError("MYSQL_DATABASE 只能包含英文字母、数字和下划线。")
    bootstrap = get_connection(database=None)
    try:
        with bootstrap.cursor() as cursor:
            cursor.execute(
                f"CREATE DATABASE IF NOT EXISTS `{MYSQL_DATABASE}` "
                "CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
            )
    finally:
        bootstrap.close()

    connection = get_connection()
    try:
        with connection.cursor() as cursor:
            for statement in DDL:
                cursor.execute(statement)
    finally:
        connection.close()


def admin_count() -> int:
    from db.connection import query_one

    row = query_one("SELECT COUNT(*) AS count FROM users WHERE role='admin'")
    return int(row["count"])


if __name__ == "__main__":
    initialize_database()
    print(f"MySQL schema ready: {MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DATABASE}")
