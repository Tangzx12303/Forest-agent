from db import demo_accounts


def test_seed_demo_accounts_migrates_and_resets_known_demos(monkeypatch):
    users = [{"username": "demo_reader", "display_name": "森林读者"}]

    def seed_demo_account(username, display_name, _role, legacy_usernames, force=False):
        target = next((user for user in users if user["username"] == username), None)
        legacy = next((user for user in users if user["username"] in legacy_usernames), None)
        if target and target["display_name"] != display_name:
            return "conflict"
        account = target or legacy
        result = "updated" if target else "migrated" if legacy else "created"
        if account is None:
            account = {}
            users.append(account)
        account.update(username=username, display_name=display_name, password="123456")
        return result

    monkeypatch.setattr(demo_accounts.repository, "seed_demo_account", seed_demo_account)

    first_configured, first_conflicts = demo_accounts.seed_demo_accounts()
    second_configured, second_conflicts = demo_accounts.seed_demo_accounts()

    assert first_configured == ["admin (created)", "reader (migrated)", "guest (created)"]
    assert first_conflicts == []
    assert second_configured == ["admin (updated)", "reader (updated)", "guest (updated)"]
    assert second_conflicts == []
    assert [(user["username"], user["password"]) for user in users] == [
        ("reader", "123456"),
        ("admin", "123456"),
        ("guest", "123456"),
    ]


def test_seed_demo_accounts_forwards_force_flag(monkeypatch):
    calls = []

    def seed_demo_account(username, display_name, role, legacy_usernames, force=False):
        calls.append((username, force))
        return "reset"

    monkeypatch.setattr(demo_accounts.repository, "seed_demo_account", seed_demo_account)

    configured, conflicts = demo_accounts.seed_demo_accounts(force=True)

    assert configured == ["admin (reset)", "reader (reset)", "guest (reset)"]
    assert conflicts == []
    assert calls == [("admin", True), ("reader", True), ("guest", True)]