"""Music indexing and playback for Radio Oracle.

Scans a local music library, extracts tags, indexes into SQLite,
and plays tracks through the configured audio output with AM radio
filter and hardware volume control.
"""

from oracle.music.catalog import Catalog, Track
from oracle.music.player import Player

__all__ = ["Catalog", "Player", "Track"]
