#!/usr/bin/env python3
"""Musique Automatique playback engine with crossfade DJ transitions.

Loads a generated playlist, crossfades between songs using detected
fade points, and outputs a continuous mix or plays it directly.

Usage:
    python3 scripts/play.py                           # play current playlist
    python3 scripts/play.py --output mix.mp3          # export mixed file
    python3 scripts/play.py --seed "artist:Pink Floyd" # generate + play
    python3 scripts/play.py --crossfade 5000          # 5s crossfade

Requires:
    pip install pydub
    ffmpeg (system package)
"""

import argparse
import json
import logging
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from pydub import AudioSegment
except ImportError:
    print("Missing pydub. Install: pip install pydub")
    sys.exit(1)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
EMBEDDINGS_DIR = PROJECT_ROOT / "embeddings"
VECTORS_FILE = EMBEDDINGS_DIR / "vectors.json"
PLAYLIST_FILE = EMBEDDINGS_DIR / "playlist.json"


def load_playlist() -> List[Dict[str, Any]]:
    """Load the current playlist from playlist.json."""
    if not PLAYLIST_FILE.exists():
        logger.error("No playlist found. Run dj.py first.")
        sys.exit(1)
    with open(PLAYLIST_FILE) as f:
        return json.load(f)


def resolve_path(song_path: str) -> Path:
    """Resolve a song path relative to the project root."""
    p = Path(song_path)
    if p.is_absolute() and p.exists():
        return p
    resolved = PROJECT_ROOT / p
    if resolved.exists():
        return resolved
    # Try relative to project root without leading dirs
    for base in [PROJECT_ROOT, PROJECT_ROOT / "music"]:
        candidate = base / p.name
        if candidate.exists():
            return candidate
    return resolved


def load_audio(song_path: str) -> Optional[AudioSegment]:
    """Load an audio file into a pydub AudioSegment."""
    path = resolve_path(song_path)
    if not path.exists():
        logger.warning("File not found: %s", path)
        return None
    try:
        return AudioSegment.from_file(str(path))
    except Exception as exc:
        logger.error("Could not load %s: %s", path, exc)
        return None


_vectors_cache: Optional[Dict[str, Any]] = None


def get_audio_features(song_metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Look up audio features from vectors.json for a song."""
    global _vectors_cache
    if _vectors_cache is None:
        if not VECTORS_FILE.exists():
            return {}
        with open(VECTORS_FILE) as f:
            _vectors_cache = json.load(f)
    filename = song_metadata.get("filename", "")
    for data in _vectors_cache.values():
        if data.get("metadata", {}).get("filename") == filename:
            return data.get("audio_features", {})
    return {}


def crossfade_songs(
    accumulated_mix: AudioSegment,
    next_song: AudioSegment,
    crossfade_ms: int = 3000,
) -> AudioSegment:
    """Append a song to the mix with a crossfade transition.

    The crossfade overlaps the last crossfade_ms of the accumulated mix
    with the first crossfade_ms of the next song.

    Args:
        accumulated_mix: The mix built so far.
        next_song: The incoming song to append.
        crossfade_ms: Duration of the crossfade overlap in ms.

    Returns:
        Combined AudioSegment with smooth crossfade.
    """
    crossfade_ms = min(crossfade_ms, len(accumulated_mix), len(next_song))
    return accumulated_mix.append(next_song, crossfade=crossfade_ms)


def build_continuous_mix(
    playlist: List[Dict[str, Any]],
    crossfade_ms: int = 3000,
) -> Optional[AudioSegment]:
    """Build a continuous DJ mix from a playlist with crossfade transitions.

    Args:
        playlist: List of playlist entries (from playlist.json).
        crossfade_ms: Default crossfade duration in ms.

    Returns:
        Combined AudioSegment, or None if no songs could be loaded.
    """
    mix = None
    prev_features: Dict[str, Any] = {}

    for i, entry in enumerate(playlist):
        metadata = entry.get("metadata", entry)
        song_path = metadata.get("path", "")
        artist = metadata.get("artist", "Unknown")
        title = metadata.get("title", metadata.get("filename", "Unknown"))

        audio = load_audio(song_path)
        if audio is None:
            logger.warning("Skipping [%d] %s - %s (could not load)", i + 1, artist, title)
            continue

        # Get audio features for intelligent crossfade placement
        features = get_audio_features(metadata)

        if mix is None:
            mix = audio
            logger.info("[%2d] %s - %s  (%.1fs)", i + 1, artist, title, len(audio) / 1000)
        else:
            # Adjust crossfade based on BPM similarity between adjacent songs
            effective_crossfade = crossfade_ms
            prev_bpm = prev_features.get("bpm", 0)
            curr_bpm = features.get("bpm", 0)
            if prev_bpm > 0 and curr_bpm > 0:
                bpm_diff = abs(prev_bpm - curr_bpm)
                if bpm_diff < 5:
                    effective_crossfade = int(crossfade_ms * 1.5)  # longer blend for similar tempos
                elif bpm_diff > 30:
                    effective_crossfade = int(crossfade_ms * 0.6)  # quick cut for big tempo change

            mix = crossfade_songs(mix, audio, crossfade_ms=effective_crossfade)
            logger.info(
                "[%2d] %s - %s  (%.1fs, xfade=%dms, mix=%.1fs)",
                i + 1, artist, title, len(audio) / 1000,
                effective_crossfade, len(mix) / 1000,
            )

        prev_features = features

    return mix


def play_audio(audio: AudioSegment) -> None:
    """Play an AudioSegment using ffplay."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
        audio.export(tmp_path, format="wav")

    logger.info("Playing via ffplay... (Ctrl+C to stop)")
    try:
        subprocess.run(
            ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", tmp_path],
            check=True,
        )
    except FileNotFoundError:
        logger.error("ffplay not found. Install ffmpeg or use --output to export.")
    except KeyboardInterrupt:
        logger.info("Playback stopped.")
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Musique Automatique DJ Playback")
    parser.add_argument("--seed", help="Generate a new playlist from seed query first")
    parser.add_argument("--length", type=int, default=10, help="Playlist length (with --seed)")
    parser.add_argument("--crossfade", type=int, default=3000, help="Crossfade duration in ms")
    parser.add_argument("--output", help="Export mix to file (mp3/wav) instead of playing")
    parser.add_argument("--dry-run", action="store_true", help="Show playlist without playing")
    args = parser.parse_args()

    # Optionally generate a fresh playlist
    if args.seed:
        from dj import generate_dj_set, save_playlist
        with open(VECTORS_FILE) as f:
            vectors = json.load(f)
        playlist_data = generate_dj_set(vectors, args.seed, set_length=args.length)
        if not playlist_data:
            logger.error("Could not generate playlist from seed: %s", args.seed)
            sys.exit(1)
        save_playlist(playlist_data, PLAYLIST_FILE)

    playlist = load_playlist()

    if args.dry_run:
        print(f"\nDJ Set ({len(playlist)} songs):")
        for entry in playlist:
            meta = entry.get("metadata", entry)
            print(f"  [{entry.get('position', '?')}] {meta.get('artist', '?')} - {meta.get('title', '?')}  "
                  f"(mood={entry.get('mood', '?')}, energy={entry.get('energy', '?')})")
        return

    print(f"\nBuilding continuous mix from {len(playlist)} songs...")
    mix = build_continuous_mix(playlist, crossfade_ms=args.crossfade)

    if mix is None:
        logger.error("No audio could be loaded.")
        sys.exit(1)

    print(f"Mix duration: {len(mix) / 1000:.1f}s ({len(mix) / 60000:.1f} min)")

    if args.output:
        out_path = Path(args.output)
        fmt = out_path.suffix.lstrip(".") or "mp3"
        mix.export(str(out_path), format=fmt, bitrate="320k")
        print(f"Exported to {out_path}")
    else:
        play_audio(mix)


if __name__ == "__main__":
    main()
