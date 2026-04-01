#!/usr/bin/env python3
"""Initialize music library by scanning directories and extracting metadata.

Scans configured music directories for MP3 files, extracts ID3 tags with
fallback to filename parsing, and writes a consolidated metadata.json.

Usage:
    python3 scripts/init_library.py [--config path/to/config.json]

Output:
    embeddings/metadata.json — list of SongMetadata dicts
"""

import json
import logging
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from mutagen.easyid3 import EasyID3
    from mutagen.mp3 import MP3
except ImportError:
    print("Missing mutagen. Install: pip install mutagen")
    sys.exit(1)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "config.json"
EMBEDDINGS_DIR = PROJECT_ROOT / "embeddings"
METADATA_FILE = EMBEDDINGS_DIR / "metadata.json"


def load_config(config_path: Path = DEFAULT_CONFIG) -> Dict[str, Any]:
    """Load project configuration from JSON file.

    Args:
        config_path: Path to config.json. Falls back to defaults if missing.

    Returns:
        Parsed configuration dictionary.
    """
    if not config_path.exists():
        logger.warning("Config not found at %s, using defaults", config_path)
        return {"music_sources": [{"path": "../music", "recursive": True}]}
    with open(config_path) as f:
        return json.load(f)


@dataclass
class SongMetadata:
    """Metadata for a single song in the library.

    Attributes:
        filename: Base filename (e.g. "01 - Track.mp3").
        path: Absolute path to the MP3 file.
        artist: Artist name from ID3 or filename.
        album: Album name from ID3 or filename.
        title: Track title from ID3 or filename.
        duration: Duration in seconds.
        bitrate: Bitrate in bps.
        year: Release year.
        genre: Genre tag.
        source: How metadata was obtained — "id3" or "filename".
    """
    filename: str
    path: str
    artist: Optional[str] = None
    album: Optional[str] = None
    title: Optional[str] = None
    duration: Optional[int] = None
    bitrate: Optional[int] = None
    year: Optional[int] = None
    genre: Optional[str] = None
    source: str = "id3"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dictionary for JSON storage."""
        return asdict(self)


def extract_from_id3(mp3_path: str) -> Optional[SongMetadata]:
    """Extract metadata from ID3 tags embedded in an MP3 file.

    Args:
        mp3_path: Filesystem path to the MP3 file.

    Returns:
        SongMetadata if ID3 tags could be read, None on failure.
    """
    try:
        audio = EasyID3(mp3_path)
    except Exception as exc:
        logger.debug("No ID3 tags in %s: %s", mp3_path, exc)
        return None

    metadata = SongMetadata(
        filename=Path(mp3_path).name,
        path=mp3_path,
        artist=audio.get("artist", [None])[0],
        album=audio.get("album", [None])[0],
        title=audio.get("title", [None])[0],
        genre=audio.get("genre", [None])[0],
        source="id3",
    )

    try:
        mp3_file = MP3(mp3_path)
        metadata.duration = int(mp3_file.info.length)
        metadata.bitrate = mp3_file.info.bitrate
    except Exception as exc:
        logger.debug("Could not read MP3 info for %s: %s", mp3_path, exc)

    return metadata


def parse_from_filename(mp3_path: str) -> SongMetadata:
    """Parse metadata from filename and directory structure.

    Tries these strategies in order:
        1. Filename: "Artist - Album - Title.mp3"
        2. Filename: "Artist - Title.mp3"
        3. Filename: "artist_name-song_title.mp3" (underscore/hyphen format)
        4. Directory: Artist/Album/Title.mp3 (most common library layout)
        5. Directory: Artist/Title.mp3

    Args:
        mp3_path: Filesystem path to the MP3 file.

    Returns:
        SongMetadata with fields populated from filename/directory parts.
    """
    path = Path(mp3_path)
    filename = path.stem
    parts = filename.split(" - ")

    metadata = SongMetadata(
        filename=path.name,
        path=mp3_path,
        source="filename",
    )

    if len(parts) >= 3:
        metadata.artist = parts[0].strip()
        metadata.album = parts[1].strip()
        metadata.title = parts[2].strip()
    elif len(parts) == 2:
        metadata.artist = parts[0].strip()
        metadata.title = parts[1].strip()
    elif "-" in filename or "_" in filename:
        # Handle "artist_name-song_title" or "artist-song_title" formats
        sep = "-" if "-" in filename else "_"
        fname_parts = filename.split(sep, 1)
        if len(fname_parts) == 2:
            metadata.artist = fname_parts[0].replace("_", " ").strip().title()
            metadata.title = fname_parts[1].replace("_", " ").strip().title()
    else:
        metadata.title = filename

    # Use directory structure as fallback/supplement: Artist/Album/Title.mp3
    dir_parts = path.parts
    music_idx = None
    for i, part in enumerate(dir_parts):
        if part.lower() == "music":
            music_idx = i
            break

    if music_idx is not None:
        remaining = dir_parts[music_idx + 1:-1]  # dirs between music/ and filename
        if len(remaining) >= 2 and not metadata.artist:
            metadata.artist = remaining[0]
            metadata.album = remaining[1]
        elif len(remaining) == 1 and not metadata.artist:
            metadata.artist = remaining[0]
        # Fill in album from directory even if artist came from filename
        if len(remaining) >= 2 and not metadata.album:
            metadata.album = remaining[1]

    if not metadata.title:
        metadata.title = filename

    return metadata


def scan_directory(directory: Path, recursive: bool = True) -> List[SongMetadata]:
    """Scan a directory for MP3 files and extract metadata.

    Tries ID3 extraction first; falls back to filename parsing if ID3
    tags are missing or lack an artist field.

    Args:
        directory: Root directory to scan.
        recursive: Whether to scan subdirectories.

    Returns:
        List of SongMetadata for every MP3 found.
    """
    songs: List[SongMetadata] = []

    if not directory.exists():
        logger.warning("Music directory does not exist: %s", directory)
        return songs

    glob_fn = directory.rglob if recursive else directory.glob
    mp3_files = sorted(
        list(glob_fn("*.mp3")) + list(glob_fn("*.MP3")),
        key=lambda p: p.name,
    )

    logger.info("Found %d MP3 files in %s", len(mp3_files), directory)

    for mp3_path in mp3_files:
        metadata = extract_from_id3(str(mp3_path))

        if not metadata or not metadata.artist:
            metadata = parse_from_filename(str(mp3_path))

        songs.append(metadata)
        logger.info(
            "  %s - %s [%s]",
            metadata.artist or "Unknown",
            metadata.title or metadata.filename,
            metadata.source,
        )

    return songs


def init_library(config_path: Path = DEFAULT_CONFIG) -> List[SongMetadata]:
    """Main entry point: scan all configured sources and persist metadata.

    Args:
        config_path: Path to config.json.

    Returns:
        List of all discovered SongMetadata.
    """
    config = load_config(config_path)
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

    all_songs: List[SongMetadata] = []
    for source in config.get("music_sources", []):
        source_path = (PROJECT_ROOT / source["path"]).resolve()
        recursive = source.get("recursive", True)
        logger.info("Scanning source '%s' at %s", source.get("name", ""), source_path)
        all_songs.extend(scan_directory(source_path, recursive=recursive))

    if not all_songs:
        logger.warning("No songs found. Add MP3s to your configured music directories.")
        return all_songs

    metadata_list = [song.to_dict() for song in all_songs]
    with open(METADATA_FILE, "w") as f:
        json.dump(metadata_list, f, indent=2)

    artists = {s.artist for s in all_songs if s.artist}
    logger.info("Extracted %d songs from %d artists", len(all_songs), len(artists))
    logger.info("Metadata saved to %s", METADATA_FILE)

    return all_songs


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    songs = init_library()
    if songs:
        artists = {s.artist for s in songs if s.artist}
        print(f"\n  Artists: {len(artists)}")
        print(f"  Songs:   {len(songs)}")
        top = sorted(artists)[:5]
        print(f"  Sample:  {', '.join(top)}{'...' if len(artists) > 5 else ''}")


if __name__ == "__main__":
    main()
