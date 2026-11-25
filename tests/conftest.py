"""Shared fixtures: keep every test's sqlite state (conversation store,
activity journal, conversation threads) in a temp dir so nothing writes data/oracle.db in the
repo, and reset the journal singleton between tests."""

from __future__ import annotations

import pytest

from config.settings import settings
from oracle.memory import journal, threads


@pytest.fixture(autouse=True)
def _isolated_memory(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "db_path", tmp_path / "oracle.db")
    monkeypatch.setattr(settings, "books_db_path", tmp_path / "books.db")
    journal._journal = None
    threads._store = None
    yield
    if threads._store is not None:
        threads._store.close()
    threads._store = None
    j = journal._journal
    if j is not None:
        j.close()
    journal._journal = None
