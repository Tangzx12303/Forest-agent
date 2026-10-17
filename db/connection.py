import pymysql
from pymysql.cursors import DictCursor

from config.settings import MYSQL_DATABASE, MYSQL_HOST, MYSQL_PASSWORD, MYSQL_PORT, MYSQL_USER


def get_connection(*, database: str | None = MYSQL_DATABASE):
    if not MYSQL_PASSWORD:
        raise RuntimeError("尚未在 .env 配置 MYSQL_PASSWORD。")
    return pymysql.connect(
        host=MYSQL_HOST,
        port=MYSQL_PORT,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        database=database,
        charset="utf8mb4",
        cursorclass=DictCursor,
        autocommit=True,
        connect_timeout=5,
        read_timeout=10,
        write_timeout=10,
    )


def query_one(sql: str, params: tuple = (), *, database: str | None = MYSQL_DATABASE):
    connection = get_connection(database=database)
    try:
        with connection.cursor() as cursor:
            cursor.execute(sql, params)
            return cursor.fetchone()
    finally:
        connection.close()


def query_all(sql: str, params: tuple = (), *, database: str | None = MYSQL_DATABASE):
    connection = get_connection(database=database)
    try:
        with connection.cursor() as cursor:
            cursor.execute(sql, params)
            return cursor.fetchall()
    finally:
        connection.close()


def execute(sql: str, params: tuple = (), *, database: str | None = MYSQL_DATABASE) -> int:
    connection = get_connection(database=database)
    try:
        with connection.cursor() as cursor:
            return cursor.execute(sql, params)
    finally:
        connection.close()
