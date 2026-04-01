#!/usr/bin/env python3
"""Generate semantic embeddings for each song using Claude.

Reads metadata.json, sends each song's metadata to Claude for semantic
analysis, and writes an n-dimensional embedding vector per song to
vectors.json.

Usage:
    python3 scripts/embed_songs.py [--config path/to/config.json]

Output:
    embeddings/vectors.json — {song_id: {metadata, embedding}} mapping
"""

import hashlib
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import anthropic
except ImportError:
    print("Missing anthropic SDK. Install: pip install anthropic")
    sys.exit(1)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "config.json"
EMBEDDINGS_DIR = PROJECT_ROOT / "embeddings"
METADATA_FILE = EMBEDDINGS_DIR / "metadata.json"
VECTORS_FILE = EMBEDDINGS_DIR / "vectors.json"

# Retry settings for transient API errors
MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 2.0

# Default model if config is missing
DEFAULT_MODEL = "claude-sonnet-4-20250514"


def load_config(config_path: Path = DEFAULT_CONFIG) -> Dict[str, Any]:
    """Load project configuration.

    Args:
        config_path: Path to config.json.

    Returns:
        Parsed config dict, or sensible defaults.
    """
    if not config_path.exists():
        logger.warning("Config not found at %s, using defaults", config_path)
        return {}
    with open(config_path) as f:
        return json.load(f)


def song_id(song: Dict[str, Any]) -> str:
    """Compute a stable short ID for a song based on artist+title.

    Args:
        song: Song metadata dict (needs "artist" and/or "title"/"filename").

    Returns:
        8-character hex digest.
    """
    key = f"{song.get('artist', 'unknown')}{song.get('title', song.get('filename', ''))}"
    return hashlib.md5(key.encode()).hexdigest()[:8]


def _build_prompt(song: Dict[str, Any]) -> str:
    """Build the Claude prompt for a single song embedding.

    Args:
        song: Song metadata dict.

    Returns:
        Prompt string requesting a JSON embedding.
    """
    artist = song.get("artist") or "Unknown Artist"
    album = song.get("album") or "Unknown Album"
    title = song.get("title") or song.get("filename", "Unknown Title")
    genre = song.get("genre") or ""

    return f"""Analyze this song and generate a semantic embedding profile.

Song: {title}
Artist: {artist}
Album: {album}
Genre: {genre}

Return a JSON object with these dimensions (each 0-10 scale):
{{
  "energy": <0-10, high for upbeat/intense, low for ambient/calm>,
  "mood": "<primary mood: happy, sad, melancholic, energetic, introspective, etc>",
  "tempo_feel": <0-10, slow to fast>,
  "instrumentation": "<primary instruments: acoustic, electronic, orchestral, rock, hip-hop, etc>",
  "complexity": <0-10, simple to intricate>,
  "danceability": <0-10, how suitable for dancing>,
  "vocality": <0-10, prominence of vocals>,
  "era": "<approximate era: classical, 80s, 90s, 2000s, 2010s, 2020s, contemporary>",
  "lyrical_themes": ["<theme1>", "<theme2>", "<theme3>"],
  "similar_artists": ["<artist1>", "<artist2>", "<artist3>"],
  "transition_compatibility": ["<compatible_genre1>", "<compatible_genre2>"]
}}

Be creative but consistent with the song's actual characteristics."""


def _parse_embedding(response_text: str) -> Optional[Dict[str, Any]]:
    """Extract the JSON embedding object from Claude's response.

    Args:
        response_text: Raw text from Claude.

    Returns:
        Parsed embedding dict, or None if extraction failed.
    """
    json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError as exc:
            logger.warning("JSON parse error in Claude response: %s", exc)
    return None


def _fallback_embedding(song: Dict[str, Any]) -> Dict[str, Any]:
    """Return a neutral embedding when the API call or parse fails.

    Args:
        song: Song metadata dict.

    Returns:
        Embedding dict with neutral/default values.
    """
    return {
        "energy": 5,
        "mood": song.get("genre") or "neutral",
        "tempo_feel": 5,
        "instrumentation": song.get("genre") or "unknown",
        "complexity": 5,
        "danceability": 5,
        "vocality": 5,
        "era": "contemporary",
        "lyrical_themes": [],
        "similar_artists": [song.get("artist", "Unknown")],
        "transition_compatibility": [],
    }


def generate_song_embedding(
    song: Dict[str, Any],
    client: anthropic.Anthropic,
    model: str = DEFAULT_MODEL,
) -> Dict[str, Any]:
    """Generate a semantic embedding for one song via the Claude API.

    Retries transient errors with exponential backoff. Falls back to a
    neutral embedding if all retries are exhausted.

    Args:
        song: Song metadata dict.
        client: Initialized Anthropic client.
        model: Model identifier to use.

    Returns:
        Embedding dict (always returns — fallback on failure).
    """
    prompt = _build_prompt(song)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            message = client.messages.create(
                model=model,
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}],
            )
            embedding = _parse_embedding(message.content[0].text)
            if embedding:
                return embedding
            logger.warning("Could not parse embedding on attempt %d", attempt)
        except anthropic.RateLimitError:
            wait = RETRY_BACKOFF_BASE ** attempt
            logger.warning("Rate limited, retrying in %.1fs (attempt %d/%d)", wait, attempt, MAX_RETRIES)
            time.sleep(wait)
        except anthropic.APIError as exc:
            logger.error("API error on attempt %d/%d: %s", attempt, MAX_RETRIES, exc)
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF_BASE ** attempt)

    logger.warning("All retries exhausted for '%s', using fallback embedding",
                    song.get("title", song.get("filename")))
    return _fallback_embedding(song)


def embed_library(config_path: Path = DEFAULT_CONFIG) -> Dict[str, Any]:
    """Embed all songs in the metadata file.

    Args:
        config_path: Path to config.json (reads model name from embedding section).

    Returns:
        The full vectors dict: {song_id: {metadata, embedding}}.
    """
    if not METADATA_FILE.exists():
        logger.error("Metadata file not found: %s — run init_library.py first", METADATA_FILE)
        sys.exit(1)

    with open(METADATA_FILE) as f:
        songs = json.load(f)

    config = load_config(config_path)
    model = config.get("embedding", {}).get("model", DEFAULT_MODEL)
    logger.info("Embedding %d songs with model %s", len(songs), model)

    client = anthropic.Anthropic()

    # Load existing vectors for incremental embedding
    vectors: Dict[str, Any] = {}
    if VECTORS_FILE.exists():
        with open(VECTORS_FILE) as f:
            vectors = json.load(f)
        logger.info("Loaded %d existing embeddings", len(vectors))

    new_count = 0
    for i, song in enumerate(songs, 1):
        sid = song_id(song)

        if sid in vectors:
            logger.debug("Skipping already-embedded song %s", sid)
            continue

        label = f"{song.get('artist', 'Unknown')} - {song.get('title', song.get('filename'))}"
        logger.info("[%d/%d] %s", i, len(songs), label)

        embedding = generate_song_embedding(song, client, model=model)
        vectors[sid] = {"metadata": song, "embedding": embedding}
        new_count += 1

        logger.info("    Mood: %s, Energy: %s/10", embedding.get("mood"), embedding.get("energy"))

    with open(VECTORS_FILE, "w") as f:
        json.dump(vectors, f, indent=2)

    logger.info("Generated %d new embeddings (%d total)", new_count, len(vectors))
    logger.info("Vectors saved to %s", VECTORS_FILE)
    return vectors


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    embed_library()


if __name__ == "__main__":
    main()
