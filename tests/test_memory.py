import tempfile
from pathlib import Path

from oracle.memory.store import ConversationStore


def test_store_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "test.db"
        store = ConversationStore(db_path)

        session_id = store.new_session()
        store.add_message(session_id, "user", "Hello")
        store.add_message(session_id, "assistant", "Hi there")

        messages = store.get_messages(session_id)
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Hello"
        assert messages[1]["role"] == "assistant"

        store.close()


def test_store_limited_messages():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "test.db"
        store = ConversationStore(db_path)

        session_id = store.new_session()
        for i in range(20):
            store.add_message(session_id, "user", f"msg {i}")

        messages = store.get_messages(session_id, limit=5)
        assert len(messages) == 5
        # Should be the most recent 5
        assert messages[-1]["content"] == "msg 19"

        store.close()


def test_store_sessions():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "test.db"
        store = ConversationStore(db_path)

        s1 = store.new_session()
        store.new_session()  # second session
        store.update_summary(s1, "Test summary")

        sessions = store.get_recent_sessions()
        assert len(sessions) == 2
        assert sessions[1]["summary"] == "Test summary"

        store.close()
