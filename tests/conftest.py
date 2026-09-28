"""Shared fixtures: keep every test's sqlite state (conversation store,
activity journal) in a temp dir so nothing writes data/oracle.db in the
repo, and reset the journal singleton between tests."""

from __future__ import annotations

import pytest

from config.settings import settings
from oracle.memory import journal


@pytest.fixture(autouse=True)
def _isolated_memory(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "db_path", tmp_path / "oracle.db")
    monkeypatch.setattr(settings, "books_db_path", tmp_path / "books.db")
    journal._journal = None
    yield
    j = journal._journal
    if j is not None:
        j.close()
    journal._journal = None
