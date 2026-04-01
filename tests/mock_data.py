"""Generate realistic mock data for testing the DJ engine.

Provides factories for song metadata and embedding vectors so tests can
run without real MP3 files or Claude API calls.

Usage:
    from tests.mock_data import make_song, make_vectors, SAMPLE_LIBRARY
"""

from typing import Any, Dict, List, Optional


def make_embedding(
    energy: float = 5,
    mood: str = "neutral",
    tempo_feel: float = 5,
    instrumentation: str = "rock",
    complexity: float = 5,
    danceability: float = 5,
    vocality: float = 5,
    era: str = "2000s",
    lyrical_themes: Optional[List[str]] = None,
    similar_artists: Optional[List[str]] = None,
    transition_compatibility: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Create a single embedding dict with explicit control over every field.

    All numeric fields default to 5 (midpoint). Override any field via kwargs.
    """
    return {
        "energy": energy,
        "mood": mood,
        "tempo_feel": tempo_feel,
        "instrumentation": instrumentation,
        "complexity": complexity,
        "danceability": danceability,
        "vocality": vocality,
        "era": era,
        "lyrical_themes": lyrical_themes or [],
        "similar_artists": similar_artists or [],
        "transition_compatibility": transition_compatibility or [],
    }


def make_song(
    artist: str = "Test Artist",
    title: str = "Test Song",
    album: str = "Test Album",
    genre: str = "rock",
    filename: Optional[str] = None,
    **embedding_kwargs: Any,
) -> Dict[str, Any]:
    """Create a complete song entry (metadata + embedding) for vectors.json.

    Args:
        artist: Artist name.
        title: Track title.
        album: Album name.
        genre: Genre tag.
        filename: Override filename (defaults to "{title}.mp3").
        **embedding_kwargs: Forwarded to make_embedding().

    Returns:
        Dict matching the vectors.json structure: {metadata, embedding}.
    """
    if filename is None:
        filename = f"{title}.mp3"
    return {
        "metadata": {
            "filename": filename,
            "path": f"/mock/music/{filename}",
            "artist": artist,
            "album": album,
            "title": title,
            "genre": genre,
            "duration": 240,
            "bitrate": 320000,
            "year": 2000,
            "source": "mock",
        },
        "embedding": make_embedding(**embedding_kwargs),
    }


# ---------------------------------------------------------------------------
# Pre-built sample library (10 songs spanning different styles)
# ---------------------------------------------------------------------------

SAMPLE_SONGS: List[Dict[str, Any]] = [
    make_song(
        artist="Pink Floyd", title="Coming Back to Life",
        album="The Division Bell", genre="progressive rock",
        energy=4, mood="melancholic", tempo_feel=3, instrumentation="rock",
        complexity=7, danceability=2, vocality=6, era="90s",
        lyrical_themes=["loss", "hope", "renewal"],
        similar_artists=["David Gilmour", "Roger Waters"],
    ),
    make_song(
        artist="Pink Floyd", title="Keep Talking",
        album="The Division Bell", genre="progressive rock",
        energy=6, mood="introspective", tempo_feel=5, instrumentation="electronic",
        complexity=7, danceability=3, vocality=7, era="90s",
        lyrical_themes=["communication", "isolation"],
        similar_artists=["David Gilmour", "Stephen Hawking"],
    ),
    make_song(
        artist="Radiohead", title="Paranoid Android",
        album="OK Computer", genre="alternative rock",
        energy=7, mood="anxious", tempo_feel=6, instrumentation="rock",
        complexity=9, danceability=3, vocality=8, era="90s",
        lyrical_themes=["alienation", "technology", "paranoia"],
        similar_artists=["Muse", "Pixies"],
    ),
    make_song(
        artist="Radiohead", title="No Surprises",
        album="OK Computer", genre="alternative rock",
        energy=2, mood="melancholic", tempo_feel=2, instrumentation="acoustic",
        complexity=4, danceability=1, vocality=7, era="90s",
        lyrical_themes=["suburban life", "resignation"],
        similar_artists=["Thom Yorke", "Coldplay"],
    ),
    make_song(
        artist="Daft Punk", title="Around the World",
        album="Homework", genre="electronic",
        energy=8, mood="energetic", tempo_feel=8, instrumentation="electronic",
        complexity=3, danceability=10, vocality=4, era="90s",
        lyrical_themes=["repetition", "dance"],
        similar_artists=["Chemical Brothers", "Justice"],
    ),
    make_song(
        artist="Daft Punk", title="Digital Love",
        album="Discovery", genre="electronic",
        energy=7, mood="happy", tempo_feel=7, instrumentation="electronic",
        complexity=4, danceability=9, vocality=5, era="2000s",
        lyrical_themes=["love", "nostalgia"],
        similar_artists=["Breakbot", "Kavinsky"],
    ),
    make_song(
        artist="Miles Davis", title="So What",
        album="Kind of Blue", genre="jazz",
        energy=3, mood="cool", tempo_feel=4, instrumentation="acoustic",
        complexity=8, danceability=2, vocality=0, era="classical",
        lyrical_themes=[],
        similar_artists=["John Coltrane", "Bill Evans"],
    ),
    make_song(
        artist="Beethoven", title="Moonlight Sonata",
        album="Piano Sonatas", genre="classical",
        energy=2, mood="melancholic", tempo_feel=2, instrumentation="acoustic",
        complexity=7, danceability=0, vocality=0, era="classical",
        lyrical_themes=[],
        similar_artists=["Chopin", "Mozart"],
    ),
    make_song(
        artist="Kendrick Lamar", title="Alright",
        album="To Pimp a Butterfly", genre="hip-hop",
        energy=8, mood="defiant", tempo_feel=7, instrumentation="hip-hop",
        complexity=7, danceability=7, vocality=9, era="2010s",
        lyrical_themes=["resilience", "social justice", "hope"],
        similar_artists=["J. Cole", "Chance the Rapper"],
    ),
    make_song(
        artist="Tame Impala", title="Let It Happen",
        album="Currents", genre="psychedelic rock",
        energy=6, mood="euphoric", tempo_feel=6, instrumentation="electronic",
        complexity=6, danceability=7, vocality=6, era="2010s",
        lyrical_themes=["change", "surrender", "transcendence"],
        similar_artists=["MGMT", "Pond"],
    ),
]


def make_vectors(songs: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """Build a vectors dict keyed by sequential IDs.

    Args:
        songs: List of song dicts (default: SAMPLE_SONGS).

    Returns:
        Dict mapping "s001".."sNNN" → song data, matching vectors.json format.
    """
    if songs is None:
        songs = SAMPLE_SONGS
    return {f"s{i:03d}": song for i, song in enumerate(songs, 1)}


# Convenience: ready-to-use vector dict
SAMPLE_LIBRARY = make_vectors()
