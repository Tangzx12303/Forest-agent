"""Opt-in seed data for local demonstrations; never run automatically."""

import argparse

from db import repository

DEMO_ACCOUNTS = (
    {
        "username": "admin",
        "legacy_usernames": ("owner", "demo_admin"),
        "display_name": "演示管理员",
        "role": "admin",
    },
    {
        "username": "reader",
        "legacy_usernames": ("demo_reader",),
        "display_name": "森林读者",
        "role": "reader",
    },
    {
        "username": "guest",
        "legacy_usernames": ("demo_guest",),
        "display_name": "访客体验",
        "role": "reader",
    },
)


def seed_demo_accounts(force: bool = False) -> tuple[list[str], list[str]]:
    """Create/reset local demo users and migrate known legacy demo usernames."""
    configured = []
    conflicts = []

    for account in DEMO_ACCOUNTS:
        username = account["username"]
        result = repository.seed_demo_account(
            username,
            account["display_name"],
            account["role"],
            account["legacy_usernames"],
            force=force,
        )
        if result == "conflict":
            conflicts.append(username)
        else:
            configured.append(f"{username} ({result})")

    return configured, conflicts


if __name__ == "__main__":
    from db.init_db import initialize_database

    parser = argparse.ArgumentParser(description="创建/迁移本地演示账号。")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="同名账号被占用时也强制重置为演示账号（密码 123456），并停用旧的重复演示账号。",
    )
    arguments = parser.parse_args()

    initialize_database()
    configured_accounts, conflicting_accounts = seed_demo_accounts(force=arguments.reset)
    print(f"Configured demo accounts: {', '.join(configured_accounts) or 'none'}")
    print(f"Skipped name conflicts: {', '.join(conflicting_accounts) or 'none'}")
    if conflicting_accounts:
        print("提示：可运行 python -m db.demo_accounts --reset 强制重置这些演示账号。")