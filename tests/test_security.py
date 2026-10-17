from db.repository import create_user
from utils.security import hash_password, verify_password


def test_password_hash_uses_salt_and_verifies():
    first = hash_password("a-long-test-password")
    second = hash_password("a-long-test-password")
    assert first != second
    assert verify_password("a-long-test-password", first)
    assert not verify_password("wrong-password", first)


def test_malformed_hash_is_rejected():
    assert not verify_password("secret", "not-a-password-hash")


def test_create_user_rejects_short_password_before_database_access():
    try:
        create_user("reader1", "short")
    except ValueError as error:
        assert "10 个字符" in str(error)
    else:
        raise AssertionError("short passwords must be rejected")
