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


def test_profile_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        store = ConversationStore(Path(tmp) / "test.db")
        assert store.get_profile() is None
        store.update_profile("Erik likes vintage radios.")
        store.update_profile("Erik likes vintage radios and Jetsons.")
        assert store.get_profile() == "Erik likes vintage radios and Jetsons."
        store.close()


def test_unsummarized_and_latest_summarized():
    with tempfile.TemporaryDirectory() as tmp:
        store = ConversationStore(Path(tmp) / "test.db")
        s1 = store.new_session()
        store.add_message(s1, "user", "hello")
        s2 = store.new_session()
        store.add_message(s2, "user", "hi again")
        store.update_summary(s2, "Talked about greetings.")
        s3 = store.new_session()  # current, no messages

        # s1 has messages but no summary; s3 (current) and empty are excluded.
        pending = store.unsummarized_sessions(exclude=s3)
        assert [p["session_id"] for p in pending] == [s1]

        latest = store.latest_summarized_session(exclude=s3)
        assert latest["session_id"] == s2
        store.close()


async def test_context_builder_injects_long_term_memory():
    from oracle.memory.context import ContextBuilder

    with tempfile.TemporaryDirectory() as tmp:
        store = ConversationStore(Path(tmp) / "test.db")
        old = store.new_session()
        store.update_summary(old, "Erik asked about beekeeping.")
        store.update_profile("Erik is restoring a vintage radio.")

        current = store.new_session()
        ctx = ContextBuilder(store, current)
        messages = await ctx.build("You are the Librarian.")

        system_texts = [m["content"] for m in messages if m["role"] == "system"]
        joined = "\n".join(system_texts)
        assert "vintage radio" in joined
        assert "beekeeping" in joined
        store.close()


async def test_finalize_session_summarizes_and_folds(monkeypatch):
    from oracle.memory import context as ctx_mod

    async def fake_summarize(messages):
        return "Summary of the chat."

    async def fake_fold(existing, new_summary):
        return f"{existing or ''} + {new_summary}".strip(" +")

    monkeypatch.setattr(ctx_mod, "summarize_conversation", fake_summarize)
    monkeypatch.setattr(ctx_mod, "fold_into_profile", fake_fold)

    with tempfile.TemporaryDirectory() as tmp:
        store = ConversationStore(Path(tmp) / "test.db")
        s = store.new_session()
        store.add_message(s, "user", "hello")
        await ctx_mod.finalize_session(store, s)
        assert store.get_summary(s) == "Summary of the chat."
        assert "Summary of the chat." in store.get_profile()

        # Idempotent: second call doesn't re-summarize.
        async def boom(messages):
            raise AssertionError("should not re-summarize")

        monkeypatch.setattr(ctx_mod, "summarize_conversation", boom)
        await ctx_mod.finalize_session(store, s)
        store.close()
