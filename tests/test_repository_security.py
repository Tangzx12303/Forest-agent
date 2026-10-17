from db.repository import authenticate


def test_authenticate_rejects_user_input_without_database_access(monkeypatch):
    monkeypatch.setattr("db.repository.query_one", lambda *args: None)
    assert authenticate("unknown", "any-password") is None
