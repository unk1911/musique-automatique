#!/usr/bin/env python3
"""The DJ Engine: intelligent playlist generation and song sequencing.

Loads pre-computed semantic embeddings, finds songs similar to a seed,
and builds a coherent DJ set by walking the embedding space.

Usage:
    python3 scripts/dj.py --seed "artist:Pink Floyd,album:Division Bell"
    python3 scripts/dj.py --seed "title:Coming Back to Life" --length 15
    python3 scripts/dj.py --seed "Radiohead" --variety 0.5

Output:
    embeddings/playlist.json — ordered list with transition metadata
"""

import argparse
import json
import logging
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "config.json"
EMBEDDINGS_DIR = PROJECT_ROOT / "embeddings"
VECTORS_FILE = EMBEDDINGS_DIR / "vectors.json"


# ---------------------------------------------------------------------------
# Similarity
# ---------------------------------------------------------------------------

def cosine_similarity(vec1: Dict[str, Any], vec2: Dict[str, Any]) -> float:
    """Compute cosine similarity between two embedding dicts.

    Only numeric fields (int/float) are compared; string and list fields
    are ignored.  Returns 0.0 when vectors share no numeric dimensions
    or either norm is zero.

    Args:
        vec1: First embedding dict.
        vec2: Second embedding dict.

    Returns:
        Cosine similarity in [-1, 1] (typically [0, 1] for these embeddings).
    """
    dims1 = {k: v for k, v in vec1.items() if isinstance(v, (int, float))}
    dims2 = {k: v for k, v in vec2.items() if isinstance(v, (int, float))}

    common_keys = set(dims1) & set(dims2)
    if not common_keys:
        return 0.0

    dot = sum(dims1[k] * dims2[k] for k in common_keys)
    norm1 = math.sqrt(sum(dims1[k] ** 2 for k in common_keys))
    norm2 = math.sqrt(sum(dims2[k] ** 2 for k in common_keys))

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot / (norm1 * norm2)


# ---------------------------------------------------------------------------
# Seed lookup
# ---------------------------------------------------------------------------

def find_seed_song(
    vectors: Dict[str, Any],
    seed_query: str,
) -> Optional[Tuple[str, Dict[str, Any]]]:
    """Find a song matching a comma-separated query of field:value pairs.

    Query format examples:
        "artist:Pink Floyd,album:Division Bell"
        "title:Coming Back to Life"
        "Pink Floyd"          (defaults to title search)

    Args:
        vectors: Full vectors dict {song_id: {metadata, embedding}}.
        seed_query: Comma-separated search terms.

    Returns:
        (song_id, song_data) tuple for the best match, or None.
    """
    query_parts = [p.strip() for p in seed_query.split(",") if p.strip()]
    best_match: Optional[Tuple[str, Dict[str, Any]]] = None
    best_score = 0

    for song_id, data in vectors.items():
        metadata = data.get("metadata", {})
        match_count = 0

        for part in query_parts:
            if ":" in part:
                key, val = part.split(":", 1)
                key, val = key.strip(), val.strip().lower()
            else:
                key, val = "title", part.strip().lower()

            song_val = metadata.get(key)
            if song_val is None:
                continue
            if val in str(song_val).lower():
                match_count += 1

        if match_count > best_score:
            best_score = match_count
            best_match = (song_id, data)

    return best_match


# ---------------------------------------------------------------------------
# Similar-song search
# ---------------------------------------------------------------------------

def find_similar_songs(
    seed_id: str,
    seed_embedding: Dict[str, Any],
    all_vectors: Dict[str, Any],
    limit: int = 5,
    exclude: Optional[set] = None,
) -> List[Tuple[str, Dict[str, Any], float]]:
    """Find songs most similar to a seed embedding.

    Args:
        seed_id: ID of the seed song (excluded from results).
        seed_embedding: The embedding dict of the seed song (the inner
            dict with energy/mood/etc., NOT the outer {metadata, embedding}).
        all_vectors: Full vectors dict.
        limit: Maximum number of results.
        exclude: Additional song IDs to skip.

    Returns:
        List of (song_id, song_data, similarity_score) sorted descending.
    """
    exclude = exclude or set()

    similarities: List[Tuple[str, Dict[str, Any], float]] = []
    for sid, data in all_vectors.items():
        if sid == seed_id or sid in exclude:
            continue
        sim = cosine_similarity(seed_embedding, data["embedding"])
        similarities.append((sid, data, sim))

    similarities.sort(key=lambda x: x[2], reverse=True)
    return similarities[:limit]


# ---------------------------------------------------------------------------
# DJ set generation
# ---------------------------------------------------------------------------

def generate_dj_set(
    vectors: Dict[str, Any],
    seed_query: str,
    set_length: int = 10,
    variety_factor: float = 0.3,
) -> List[Dict[str, Any]]:
    """Generate a DJ set starting from a seed song.

    At each step, the engine finds the top candidates by cosine similarity
    and selects one with weighted randomness controlled by *variety_factor*
    (0.0 = always pick most similar, 1.0 = uniform random among top-N).

    Args:
        vectors: Full vectors dict.
        seed_query: Search query for the starting song.
        set_length: Desired number of songs in the set.
        variety_factor: How much randomness to inject (0.0–1.0).

    Returns:
        Ordered list of song data dicts [{metadata, embedding}, ...].
    """
    seed_result = find_seed_song(vectors, seed_query)
    if not seed_result:
        logger.error("Seed song not found for query: %s", seed_query)
        return []

    seed_id, seed_data = seed_result
    playlist: List[Dict[str, Any]] = [seed_data]
    played = {seed_id}

    meta = seed_data["metadata"]
    logger.info(
        "DJ Set starting from: %s - %s",
        meta.get("artist", "Unknown"),
        meta.get("title", meta.get("filename", "Unknown")),
    )

    candidate_pool = max(3, int(5 * (1 + variety_factor)))

    while len(playlist) < set_length:
        current = playlist[-1]
        current_id = _find_id_for_data(current, vectors, played)

        similar = find_similar_songs(
            seed_id=current_id or "",
            seed_embedding=current["embedding"],
            all_vectors=vectors,
            limit=candidate_pool,
            exclude=played,
        )

        if not similar:
            logger.warning("No more candidate songs available")
            break

        next_id, next_data, similarity = _pick_candidate(similar, variety_factor)

        playlist.append(next_data)
        played.add(next_id)

        nmeta = next_data["metadata"]
        logger.info(
            "[%2d] %s - %s  (mood=%s, sim=%.2f)",
            len(playlist),
            nmeta.get("artist", "Unknown"),
            nmeta.get("title", nmeta.get("filename", "Unknown")),
            next_data["embedding"].get("mood"),
            similarity,
        )

    return playlist


def _find_id_for_data(
    data: Dict[str, Any],
    vectors: Dict[str, Any],
    played: set,
) -> Optional[str]:
    """Reverse-lookup a song's ID in the vectors dict.

    Falls back to the last-played ID.
    """
    for sid, sdata in vectors.items():
        if sdata is data:
            return sid
    # fallback: last added to played set
    return next(iter(played)) if played else None


def _pick_candidate(
    candidates: List[Tuple[str, Dict[str, Any], float]],
    variety_factor: float,
) -> Tuple[str, Dict[str, Any], float]:
    """Select a candidate with weighted randomness.

    When variety_factor == 0.0 always returns the top candidate.
    Higher values increase the chance of picking a lower-ranked match.

    Args:
        candidates: Sorted list of (id, data, similarity).
        variety_factor: Randomness control (0.0–1.0).

    Returns:
        Selected (id, data, similarity) tuple.
    """
    if variety_factor <= 0.0 or len(candidates) == 1:
        return candidates[0]

    # Weight by similarity score, flattened by variety_factor
    weights = []
    for _, _, sim in candidates:
        w = sim ** (1.0 / max(variety_factor, 0.01))
        weights.append(max(w, 1e-9))

    chosen = random.choices(candidates, weights=weights, k=1)[0]
    return chosen


# ---------------------------------------------------------------------------
# Playlist persistence
# ---------------------------------------------------------------------------

def save_playlist(playlist: List[Dict[str, Any]], path: Path) -> None:
    """Write a playlist to JSON.

    Args:
        playlist: List of song data dicts.
        path: Output file path.
    """
    out = []
    for i, song in enumerate(playlist):
        entry = {
            "position": i + 1,
            "metadata": song["metadata"],
            "mood": song["embedding"].get("mood"),
            "energy": song["embedding"].get("energy"),
        }
        out.append(entry)

    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    logger.info("Playlist saved to %s", path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list (defaults to sys.argv[1:]).

    Returns:
        Parsed namespace.
    """
    parser = argparse.ArgumentParser(
        description="Musique Automatique DJ Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            '  %(prog)s --seed "artist:Pink Floyd,album:Division Bell"\n'
            '  %(prog)s --seed "title:Paranoid Android" --length 15\n'
            '  %(prog)s --seed "Radiohead" --variety 0.5\n'
        ),
    )
    parser.add_argument(
        "--seed", required=True,
        help='Seed query, e.g. "artist:Pink Floyd,album:Division Bell"',
    )
    parser.add_argument(
        "--length", type=int, default=10,
        help="Number of songs in the set (default: 10)",
    )
    parser.add_argument(
        "--variety", type=float, default=None,
        help="Variety factor 0.0–1.0 (default: from config or 0.3)",
    )
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG,
        help="Path to config.json",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    """CLI entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    args = parse_args(argv)

    if not VECTORS_FILE.exists():
        logger.error("Vectors file not found: %s — run embed_songs.py first", VECTORS_FILE)
        sys.exit(1)

    with open(VECTORS_FILE) as f:
        vectors = json.load(f)

    # Read variety from config if not overridden on CLI
    variety = args.variety
    if variety is None:
        if args.config.exists():
            with open(args.config) as f:
                cfg = json.load(f)
            variety = cfg.get("dj_config", {}).get("variety_factor", 0.3)
        else:
            variety = 0.3

    playlist = generate_dj_set(
        vectors,
        seed_query=args.seed,
        set_length=args.length,
        variety_factor=variety,
    )

    if playlist:
        output_path = EMBEDDINGS_DIR / "playlist.json"
        save_playlist(playlist, output_path)
        print(f"\nDJ Set: {len(playlist)} songs")


if __name__ == "__main__":
    main()
