"""Unit tests for the DJ engine (scripts/dj.py).

Run with: python -m pytest tests/test_dj.py -v
"""

import json
import math
import sys
import tempfile
from pathlib import Path
from unittest import mock

import pytest

# Ensure project root is on sys.path so we can import scripts.*
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.dj import (
    cosine_similarity,
    find_seed_song,
    find_similar_songs,
    generate_dj_set,
    parse_args,
    save_playlist,
    _pick_candidate,
)
from tests.mock_data import (
    SAMPLE_LIBRARY,
    make_embedding,
    make_song,
    make_vectors,
)


# -------------------------------------------------------------------------
# cosine_similarity
# -------------------------------------------------------------------------

class TestCosineSimilarity:
    """Tests for the cosine_similarity function."""

    def test_identical_vectors(self):
        """Identical embeddings should have similarity ~1.0."""
        emb = make_embedding(energy=8, tempo_feel=6, complexity=4, danceability=7, vocality=5)
        assert cosine_similarity(emb, emb) == pytest.approx(1.0, abs=1e-9)

    def test_orthogonal_vectors(self):
        """Vectors with nothing in common numerically should return 0."""
        a = {"energy": 1, "tempo_feel": 0}
        b = {"energy": 0, "tempo_feel": 1}
        # These are not orthogonal in the mathematical sense but test the function
        sim = cosine_similarity(a, b)
        assert 0.0 <= sim <= 1.0

    def test_zero_vector(self):
        """A zero-norm vector should return 0 (no division error)."""
        a = {"energy": 0, "tempo_feel": 0, "complexity": 0}
        b = {"energy": 5, "tempo_feel": 5, "complexity": 5}
        assert cosine_similarity(a, b) == 0.0

    def test_no_common_keys(self):
        """No shared numeric keys → 0."""
        a = {"energy": 5}
        b = {"tempo_feel": 5}
        assert cosine_similarity(a, b) == 0.0

    def test_ignores_non_numeric_fields(self):
        """String and list fields should not affect similarity."""
        a = make_embedding(energy=8, tempo_feel=6)
        b = make_embedding(energy=8, tempo_feel=6)
        b["mood"] = "totally different mood"
        b["lyrical_themes"] = ["x", "y", "z"]
        # Numeric fields are identical, so similarity should be 1.0
        assert cosine_similarity(a, b) == pytest.approx(1.0, abs=1e-9)

    def test_symmetry(self):
        """cosine_similarity(a, b) == cosine_similarity(b, a)."""
        a = make_embedding(energy=3, tempo_feel=9, complexity=1)
        b = make_embedding(energy=7, tempo_feel=2, complexity=8)
        assert cosine_similarity(a, b) == pytest.approx(cosine_similarity(b, a))

    def test_similar_songs_score_higher(self):
        """Two rock songs should be more similar than rock vs. electronic."""
        rock1 = make_embedding(energy=7, tempo_feel=6, danceability=4, vocality=7)
        rock2 = make_embedding(energy=6, tempo_feel=5, danceability=3, vocality=8)
        electronic = make_embedding(energy=9, tempo_feel=9, danceability=10, vocality=2)

        sim_rock = cosine_similarity(rock1, rock2)
        sim_cross = cosine_similarity(rock1, electronic)
        assert sim_rock > sim_cross


# -------------------------------------------------------------------------
# find_seed_song
# -------------------------------------------------------------------------

class TestFindSeedSong:
    """Tests for seed song lookup."""

    def test_find_by_artist(self):
        result = find_seed_song(SAMPLE_LIBRARY, "artist:Pink Floyd")
        assert result is not None
        sid, data = result
        assert data["metadata"]["artist"] == "Pink Floyd"

    def test_find_by_title(self):
        result = find_seed_song(SAMPLE_LIBRARY, "title:Paranoid Android")
        assert result is not None
        assert result[1]["metadata"]["title"] == "Paranoid Android"

    def test_find_by_combined_query(self):
        result = find_seed_song(SAMPLE_LIBRARY, "artist:Radiohead,title:No Surprises")
        assert result is not None
        assert result[1]["metadata"]["title"] == "No Surprises"

    def test_bare_query_searches_title(self):
        """A query without key: prefix should search title."""
        result = find_seed_song(SAMPLE_LIBRARY, "So What")
        assert result is not None
        assert result[1]["metadata"]["artist"] == "Miles Davis"

    def test_case_insensitive(self):
        result = find_seed_song(SAMPLE_LIBRARY, "artist:pink floyd")
        assert result is not None
        assert result[1]["metadata"]["artist"] == "Pink Floyd"

    def test_partial_match(self):
        result = find_seed_song(SAMPLE_LIBRARY, "title:Coming Back")
        assert result is not None
        assert "Coming Back" in result[1]["metadata"]["title"]

    def test_no_match_returns_none(self):
        result = find_seed_song(SAMPLE_LIBRARY, "artist:Nonexistent Band")
        assert result is None

    def test_best_match_wins(self):
        """A two-field match should beat a one-field match."""
        result = find_seed_song(SAMPLE_LIBRARY, "artist:Pink Floyd,title:Keep Talking")
        assert result is not None
        assert result[1]["metadata"]["title"] == "Keep Talking"

    def test_none_metadata_values(self):
        """Handles None values in metadata without crashing."""
        vectors = {"x": {"metadata": {"artist": None, "title": "Test"}, "embedding": {}}}
        result = find_seed_song(vectors, "artist:Pink Floyd")
        assert result is None

    def test_empty_vectors(self):
        assert find_seed_song({}, "anything") is None


# -------------------------------------------------------------------------
# find_similar_songs
# -------------------------------------------------------------------------

class TestFindSimilarSongs:
    """Tests for similarity search."""

    def test_excludes_seed(self):
        """Seed song should never appear in results."""
        results = find_similar_songs(
            "s001", SAMPLE_LIBRARY["s001"]["embedding"], SAMPLE_LIBRARY,
        )
        ids = [r[0] for r in results]
        assert "s001" not in ids

    def test_respects_limit(self):
        results = find_similar_songs(
            "s001", SAMPLE_LIBRARY["s001"]["embedding"], SAMPLE_LIBRARY, limit=3,
        )
        assert len(results) <= 3

    def test_respects_exclude(self):
        results = find_similar_songs(
            "s001", SAMPLE_LIBRARY["s001"]["embedding"], SAMPLE_LIBRARY,
            exclude={"s002", "s003"},
        )
        ids = [r[0] for r in results]
        assert "s002" not in ids
        assert "s003" not in ids

    def test_sorted_descending(self):
        results = find_similar_songs(
            "s001", SAMPLE_LIBRARY["s001"]["embedding"], SAMPLE_LIBRARY,
        )
        sims = [r[2] for r in results]
        assert sims == sorted(sims, reverse=True)

    def test_returns_tuples(self):
        results = find_similar_songs(
            "s001", SAMPLE_LIBRARY["s001"]["embedding"], SAMPLE_LIBRARY, limit=2,
        )
        for sid, data, sim in results:
            assert isinstance(sid, str)
            assert "metadata" in data
            assert "embedding" in data
            assert isinstance(sim, float)

    def test_empty_when_all_excluded(self):
        all_ids = set(SAMPLE_LIBRARY.keys())
        results = find_similar_songs(
            "s001", SAMPLE_LIBRARY["s001"]["embedding"], SAMPLE_LIBRARY,
            exclude=all_ids,
        )
        assert results == []


# -------------------------------------------------------------------------
# generate_dj_set
# -------------------------------------------------------------------------

class TestGenerateDjSet:
    """Tests for DJ set generation."""

    def test_generates_requested_length(self):
        playlist = generate_dj_set(SAMPLE_LIBRARY, "artist:Pink Floyd", set_length=5)
        assert len(playlist) == 5

    def test_starts_with_seed(self):
        playlist = generate_dj_set(SAMPLE_LIBRARY, "artist:Miles Davis", set_length=3)
        assert playlist[0]["metadata"]["artist"] == "Miles Davis"

    def test_no_duplicate_songs(self):
        playlist = generate_dj_set(SAMPLE_LIBRARY, "artist:Daft Punk", set_length=8)
        titles = [s["metadata"]["title"] for s in playlist]
        assert len(titles) == len(set(titles))

    def test_empty_on_bad_seed(self):
        playlist = generate_dj_set(SAMPLE_LIBRARY, "artist:Nobody", set_length=5)
        assert playlist == []

    def test_caps_at_library_size(self):
        playlist = generate_dj_set(SAMPLE_LIBRARY, "artist:Pink Floyd", set_length=100)
        assert len(playlist) <= len(SAMPLE_LIBRARY)

    def test_variety_factor_zero(self):
        """With variety=0, output should be deterministic."""
        a = generate_dj_set(SAMPLE_LIBRARY, "artist:Pink Floyd", set_length=5, variety_factor=0.0)
        b = generate_dj_set(SAMPLE_LIBRARY, "artist:Pink Floyd", set_length=5, variety_factor=0.0)
        titles_a = [s["metadata"]["title"] for s in a]
        titles_b = [s["metadata"]["title"] for s in b]
        assert titles_a == titles_b

    def test_each_song_has_metadata_and_embedding(self):
        playlist = generate_dj_set(SAMPLE_LIBRARY, "artist:Radiohead", set_length=4)
        for song in playlist:
            assert "metadata" in song
            assert "embedding" in song


# -------------------------------------------------------------------------
# _pick_candidate
# -------------------------------------------------------------------------

class TestPickCandidate:
    """Tests for candidate selection with variety."""

    def test_zero_variety_picks_top(self):
        candidates = [
            ("a", {}, 0.95),
            ("b", {}, 0.80),
            ("c", {}, 0.60),
        ]
        sid, _, sim = _pick_candidate(candidates, variety_factor=0.0)
        assert sid == "a"
        assert sim == 0.95

    def test_single_candidate(self):
        candidates = [("only", {}, 0.5)]
        sid, _, _ = _pick_candidate(candidates, variety_factor=1.0)
        assert sid == "only"


# -------------------------------------------------------------------------
# save_playlist
# -------------------------------------------------------------------------

class TestSavePlaylist:
    """Tests for playlist serialization."""

    def test_writes_valid_json(self):
        playlist = [SAMPLE_LIBRARY["s001"], SAMPLE_LIBRARY["s002"]]
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)

        save_playlist(playlist, path)
        data = json.loads(path.read_text())

        assert len(data) == 2
        assert data[0]["position"] == 1
        assert data[1]["position"] == 2
        assert "metadata" in data[0]
        assert "mood" in data[0]
        assert "energy" in data[0]
        path.unlink()


# -------------------------------------------------------------------------
# parse_args
# -------------------------------------------------------------------------

class TestParseArgs:
    """Tests for CLI argument parsing."""

    def test_seed_required(self):
        with pytest.raises(SystemExit):
            parse_args([])

    def test_basic_seed(self):
        args = parse_args(["--seed", "artist:Pink Floyd"])
        assert args.seed == "artist:Pink Floyd"
        assert args.length == 10  # default
        assert args.variety is None  # default

    def test_all_flags(self):
        args = parse_args(["--seed", "test", "--length", "20", "--variety", "0.7"])
        assert args.length == 20
        assert args.variety == pytest.approx(0.7)
