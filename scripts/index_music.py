#!/usr/bin/env python3
"""Index a directory of music files into the music catalog.

Usage:
    python scripts/index_music.py [MUSIC_DIR]
    python scripts/index_music.py --list
    python scripts/index_music.py --search "beatles"
    python scripts/index_music.py --stats
"""

import argparse
from pathlib import Path


def _fmt_duration(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}:{s:02d}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Index music for Radio Oracle")
    parser.add_argument("music_dir", nargs="?", type=Path, help="Directory of music files")
    parser.add_argument("--list", action="store_true", help="List all indexed tracks")
    parser.add_argument("--search", type=str, help="Search for a track")
    parser.add_argument("--stats", action="store_true", help="Show catalog statistics")
    args = parser.parse_args()

    from oracle.music.catalog import Catalog

    cat = Catalog()

    if args.list:
        tracks = cat.list_tracks()
        if not tracks:
            print("No tracks indexed yet.")
            return
        for t in tracks:
            dur = _fmt_duration(t.duration)
            print(f"  [{t.id:4d}] {t.artist or 'Unknown'} — {t.title} [{dur}]")
        print(f"\n{len(tracks)} tracks total")
        return

    if args.search:
        results = cat.search(args.search)
        if not results:
            print(f"No tracks matching '{args.search}'")
            return
        for t in results:
            dur = _fmt_duration(t.duration)
            print(f"  [{t.id:4d}] {t.artist or 'Unknown'} — {t.title} [{dur}]")
        print(f"\n{len(results)} matches")
        return

    if args.stats:
        tracks = cat.list_tracks()
        if not tracks:
            print("No tracks indexed.")
            return
        artists = len(set(t.artist for t in tracks if t.artist))
        albums = len(set(t.album for t in tracks if t.album))
        total_dur = sum(t.duration for t in tracks)
        hours = total_dur / 3600
        print(f"Tracks:  {len(tracks)}")
        print(f"Artists: {artists}")
        print(f"Albums:  {albums}")
        print(f"Total:   {hours:.1f} hours")
        return

    # Default: index a directory
    music_dir = args.music_dir
    added = cat.index_directory(music_dir)
    if added:
        print(f"Indexed {added} new tracks")

    total = cat.count()
    print(f"\nCatalog now contains {total} tracks")
    cat.close()


if __name__ == "__main__":
    main()
