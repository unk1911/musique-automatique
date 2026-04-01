#!/usr/bin/env python3
"""Music discovery engine -- finds, scores, and downloads new songs.

Builds a taste profile from the existing library, asks Claude for
recommendations, scores them against the profile, and downloads the
best match via yt-dlp.

Usage:
    python3 scripts/discover.py                  # discover + download 1 song
    python3 scripts/discover.py --dry-run        # show candidates only
    python3 scripts/discover.py --count 3        # discover 3 songs

Requires:
    - ANTHROPIC_API_KEY in environment or ../.env
    - yt-dlp (system package)
"""

import argparse
import hashlib
import json
import logging
import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent))

try:
    import anthropic
except ImportError:
    print("Missing anthropic SDK. Install: pip install anthropic")
    sys.exit(1)

from dj import cosine_similarity
from embed_songs import generate_song_embedding

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
EMBEDDINGS_DIR = PROJECT_ROOT / "embeddings"
VECTORS_FILE = EMBEDDINGS_DIR / "vectors.json"
MUSIC_DIR = PROJECT_ROOT / "music"
BROWSER_PROFILE = PROJECT_ROOT.parent.parent / "browser" / "openclaw" / "user-data" / "Default"

DEFAULT_MODEL = "claude-sonnet-4-20250514"


def load_env():
    """Load ANTHROPIC_API_KEY from .env if not already set."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        return
    env_file = PROJECT_ROOT.parent / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if line.startswith("ANTHROPIC_API_KEY"):
                key, _, val = line.partition("=")
                os.environ[key.strip()] = val.strip().strip('"')


def load_vectors() -> Dict[str, Any]:
    if not VECTORS_FILE.exists():
        logger.error("No vectors.json found. Run embed_songs.py first.")
        sys.exit(1)
    with open(VECTORS_FILE) as f:
        return json.load(f)


def save_vectors(vectors: Dict[str, Any]):
    with open(VECTORS_FILE, "w") as f:
        json.dump(vectors, f, indent=2)


def song_id(artist: str, title: str) -> str:
    key = f"{artist}{title}"
    return hashlib.md5(key.encode()).hexdigest()[:8]


# ---------------------------------------------------------------------------
# Taste profile
# ---------------------------------------------------------------------------

def build_taste_profile(vectors: Dict[str, Any]) -> Dict[str, Any]:
    """Build a taste profile from all songs in the library."""
    numeric_keys = ["energy", "tempo_feel", "complexity", "danceability", "vocality"]
    sums = {k: 0.0 for k in numeric_keys}
    count = 0

    moods = Counter()
    similar_artists = Counter()
    genres = Counter()
    themes = Counter()
    library_artists = set()

    for data in vectors.values():
        emb = data.get("embedding", {})
        meta = data.get("metadata", {})

        for k in numeric_keys:
            if isinstance(emb.get(k), (int, float)):
                sums[k] += emb[k]
        count += 1

        if emb.get("mood"):
            moods[emb["mood"]] += 1
        for a in emb.get("similar_artists", []):
            similar_artists[a] += 1
        for g in emb.get("transition_compatibility", []):
            genres[g] += 1
        for t in emb.get("lyrical_themes", []):
            themes[t] += 1
        if meta.get("artist"):
            library_artists.add(meta["artist"])

    profile = {
        "numeric": {k: round(v / max(count, 1), 1) for k, v in sums.items()},
        "top_moods": [m for m, _ in moods.most_common(8)],
        "top_similar_artists": [a for a, _ in similar_artists.most_common(15)],
        "top_genres": [g for g, _ in genres.most_common(10)],
        "top_themes": [t for t, _ in themes.most_common(8)],
        "library_artists": sorted(library_artists),
        "song_count": count,
    }
    return profile


# ---------------------------------------------------------------------------
# Claude-based discovery
# ---------------------------------------------------------------------------

def get_candidates(
    taste_profile: Dict[str, Any],
    current_song: Optional[Dict[str, Any]],
    client: anthropic.Anthropic,
) -> List[Dict[str, str]]:
    """Ask Claude to suggest new songs based on the taste profile."""
    num = taste_profile["numeric"]
    artists = ", ".join(taste_profile["library_artists"])
    similar = ", ".join(taste_profile["top_similar_artists"][:10])
    genres = ", ".join(taste_profile["top_genres"][:8])
    moods = ", ".join(taste_profile["top_moods"][:6])
    themes = ", ".join(taste_profile["top_themes"][:6])

    current_ctx = ""
    if current_song:
        meta = current_song.get("metadata", {})
        emb = current_song.get("embedding", {})
        current_ctx = (
            f"\nThe last song playing was: {meta.get('artist', '?')} - {meta.get('title', '?')} "
            f"(mood: {emb.get('mood', '?')}, energy: {emb.get('energy', '?')}). "
            f"The next song should flow naturally from it."
        )

    prompt = f"""You are a music discovery engine. A listener has a library of {taste_profile['song_count']} songs.

Their taste profile:
- Artists in library: {artists}
- Energy: {num['energy']}/10, Tempo feel: {num['tempo_feel']}/10
- Complexity: {num['complexity']}/10, Danceability: {num['danceability']}/10
- Vocality: {num['vocality']}/10
- Preferred moods: {moods}
- Preferred genres: {genres}
- Lyrical themes: {themes}
- Artists they'd probably like: {similar}
{current_ctx}

Suggest 5 songs by artists NOT already in their library that they would love.
Pick songs that are available on YouTube (well-known enough to find).
Mix discovery (new artists) with deeper cuts from familiar-adjacent artists.

Return ONLY a JSON array:
[{{"artist": "Artist Name", "title": "Song Title"}}, ...]"""

    msg = client.messages.create(
        model=DEFAULT_MODEL,
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}],
    )

    match = re.search(r"\[.*\]", msg.content[0].text, re.DOTALL)
    if match:
        try:
            candidates = json.loads(match.group())
            logger.info("Got %d candidates from Claude", len(candidates))
            return candidates
        except json.JSONDecodeError:
            logger.warning("Failed to parse candidates JSON")
    return []


def score_candidates(
    candidates: List[Dict[str, str]],
    taste_profile: Dict[str, Any],
    client: anthropic.Anthropic,
) -> List[Tuple[Dict[str, str], Dict[str, Any], float]]:
    """Embed each candidate and score against taste profile."""
    scored = []

    for candidate in candidates:
        song_meta = {
            "artist": candidate["artist"],
            "title": candidate["title"],
            "album": "",
            "genre": "",
        }

        embedding = generate_song_embedding(song_meta, client, model=DEFAULT_MODEL)

        sim = cosine_similarity(embedding, taste_profile["numeric"])

        # Boost if candidate's similar_artists overlap with library artists
        artist_overlap = len(
            set(embedding.get("similar_artists", []))
            & set(taste_profile["library_artists"])
        )
        affinity = 1.0 + 0.15 * min(artist_overlap, 3)

        # Boost if genres overlap
        genre_overlap = len(
            set(embedding.get("transition_compatibility", []))
            & set(taste_profile["top_genres"])
        )
        genre_boost = 1.0 + 0.1 * min(genre_overlap, 3)

        score = (sim ** 2) * affinity * genre_boost

        scored.append((candidate, embedding, score))
        logger.info(
            "  %s - %s: sim=%.3f affinity=%.2f genre=%.2f score=%.4f",
            candidate["artist"], candidate["title"],
            sim, affinity, genre_boost, score,
        )

    scored.sort(key=lambda x: x[2], reverse=True)
    return scored


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_song(artist: str, title: str) -> Optional[str]:
    """Download a song from YouTube via yt-dlp. Returns path to MP3."""
    artist_dir = MUSIC_DIR / artist
    artist_dir.mkdir(parents=True, exist_ok=True)

    safe_title = re.sub(r'[<>:"/\\|?*]', '', title)
    output_template = str(artist_dir / f"{safe_title}.%(ext)s")
    output_path = artist_dir / f"{safe_title}.mp3"

    if output_path.exists():
        logger.info("Already downloaded: %s", output_path)
        return str(output_path)

    search_query = f"{artist} - {title}"
    logger.info("Downloading: %s", search_query)

    try:
        result = subprocess.run(
            [
                "yt-dlp",
                f"ytsearch1:{search_query}",
                "-x", "--audio-format", "mp3",
                "--audio-quality", "0",
                "-o", output_template,
                "--no-playlist",
                "--quiet",
                "--js-runtimes", "node",
                "--cookies-from-browser", f"chromium:{BROWSER_PROFILE}",
            ],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode != 0:
            logger.error("yt-dlp failed: %s", result.stderr[:200])
            return None

        # yt-dlp may produce the file with .mp3 extension
        if output_path.exists():
            return str(output_path)

        # Check for the file (yt-dlp might add different extension handling)
        for f in artist_dir.glob(f"{safe_title}.*"):
            if f.suffix in (".mp3", ".m4a", ".opus", ".webm"):
                return str(f)

        logger.error("Download completed but file not found")
        return None

    except subprocess.TimeoutExpired:
        logger.error("yt-dlp timed out")
        return None


# ---------------------------------------------------------------------------
# Audio analysis (reuse existing)
# ---------------------------------------------------------------------------

def analyse_downloaded(file_path: str) -> Dict[str, Any]:
    """Run audio analysis on a downloaded file."""
    try:
        from playback_analysis import analyse_file
        return analyse_file(file_path)
    except Exception as exc:
        logger.warning("Audio analysis failed for %s: %s", file_path, exc)
        return {}


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def discover_next(
    vectors: Dict[str, Any],
    current_data: Optional[Dict[str, Any]],
    client: anthropic.Anthropic,
) -> Optional[Tuple[str, Dict[str, Any]]]:
    """Discover, download, embed, and return a new song.

    Returns (song_id, song_data) or None on failure.
    """
    profile = build_taste_profile(vectors)

    print(f"  Taste: energy={profile['numeric']['energy']}, "
          f"tempo={profile['numeric']['tempo_feel']}, "
          f"moods={', '.join(profile['top_moods'][:3])}")

    # Get candidates from Claude
    candidates = get_candidates(profile, current_data, client)
    if not candidates:
        logger.warning("No candidates returned")
        return None

    print(f"  Candidates: {len(candidates)}")
    for c in candidates:
        print(f"    - {c['artist']} - {c['title']}")

    # Score candidates
    scored = score_candidates(candidates, profile, client)
    if not scored:
        return None

    # Try to download the top scorer, fall back to next if it fails
    for candidate, embedding, score in scored:
        artist = candidate["artist"]
        title = candidate["title"]
        print(f"  Winner: {artist} - {title} (score={score:.4f})")

        file_path = download_song(artist, title)
        if file_path is None:
            print(f"  Download failed, trying next candidate...")
            continue

        # Compute relative path from project root
        rel_path = str(Path(file_path).relative_to(PROJECT_ROOT))

        # Analyse audio
        print(f"  Analyzing audio...")
        audio_features = analyse_downloaded(file_path)

        # Build song data
        sid = song_id(artist, title)
        song_data = {
            "metadata": {
                "filename": Path(file_path).name,
                "path": rel_path,
                "artist": artist,
                "title": title,
                "album": None,
                "duration": audio_features.get("duration_sec"),
                "source": "discovered",
            },
            "embedding": embedding,
            "audio_features": audio_features,
        }

        # Persist to vectors.json
        vectors[sid] = song_data
        save_vectors(vectors)

        bpm = audio_features.get("bpm", "?")
        key = audio_features.get("key", "?")
        mode = audio_features.get("mode", "")
        print(f"  Added to library: {artist} - {title} (bpm={bpm}, key={key} {mode})")

        return sid, song_data

    logger.warning("All candidates failed to download")
    return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Musique Automatique Discovery Engine")
    parser.add_argument("--dry-run", action="store_true", help="Show candidates without downloading")
    parser.add_argument("--count", type=int, default=1, help="Number of songs to discover")
    args = parser.parse_args()

    load_env()
    client = anthropic.Anthropic()
    vectors = load_vectors()

    profile = build_taste_profile(vectors)
    print(f"\nTaste Profile ({profile['song_count']} songs):")
    print(f"  Energy: {profile['numeric']['energy']}, Tempo: {profile['numeric']['tempo_feel']}")
    print(f"  Complexity: {profile['numeric']['complexity']}, Danceability: {profile['numeric']['danceability']}")
    print(f"  Moods: {', '.join(profile['top_moods'][:5])}")
    print(f"  Genres: {', '.join(profile['top_genres'][:5])}")
    print(f"  Artists ({len(profile['library_artists'])}): {', '.join(profile['library_artists'][:8])}...")
    print(f"  Similar artists: {', '.join(profile['top_similar_artists'][:8])}...")

    if args.dry_run:
        print(f"\nDiscovering candidates...")
        candidates = get_candidates(profile, None, client)
        scored = score_candidates(candidates, profile, client)
        print(f"\nRanked candidates:")
        for candidate, embedding, score in scored:
            print(f"  {score:.4f}  {candidate['artist']} - {candidate['title']}  "
                  f"(mood={embedding.get('mood', '?')}, energy={embedding.get('energy', '?')})")
        return

    for i in range(args.count):
        print(f"\n--- Discovery {i + 1}/{args.count} ---")
        result = discover_next(vectors, None, client)
        if result:
            sid, data = result
            meta = data["metadata"]
            print(f"  OK: {meta['artist']} - {meta['title']}")
        else:
            print("  Failed to discover a song")
            break


if __name__ == "__main__":
    main()
