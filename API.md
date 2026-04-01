# Musique Automatique — API Reference

Module-level documentation for the Phase 1 codebase. All public functions
are importable from `scripts.*`.

---

## scripts/init_library.py

Library scanner. Reads MP3 files, extracts ID3 metadata, and writes
`embeddings/metadata.json`.

### Classes

#### `SongMetadata`
Dataclass holding per-song metadata.

| Field      | Type           | Description                         |
|------------|----------------|-------------------------------------|
| `filename` | `str`          | Base filename                       |
| `path`     | `str`          | Absolute filesystem path            |
| `artist`   | `Optional[str]`| Artist name                         |
| `album`    | `Optional[str]`| Album name                          |
| `title`    | `Optional[str]`| Track title                         |
| `duration` | `Optional[int]`| Duration in seconds                 |
| `bitrate`  | `Optional[int]`| Bitrate in bps                      |
| `year`     | `Optional[int]`| Release year                        |
| `genre`    | `Optional[str]`| Genre tag                           |
| `source`   | `str`          | `"id3"` or `"filename"`             |

### Functions

| Function | Signature | Description |
|----------|-----------|-------------|
| `extract_from_id3` | `(mp3_path: str) → Optional[SongMetadata]` | Extract metadata from ID3 tags. Returns `None` on failure. |
| `parse_from_filename` | `(mp3_path: str) → SongMetadata` | Parse artist/title from filename. Supports `Artist - Album - Title.mp3` and `Artist - Title.mp3`. |
| `scan_directory` | `(directory: Path, recursive: bool) → List[SongMetadata]` | Scan a directory for MP3s. Tries ID3 first, falls back to filename. |
| `init_library` | `(config_path: Path) → List[SongMetadata]` | Top-level entry point. Scans all configured sources, writes `metadata.json`. |

---

## scripts/embed_songs.py

Semantic embedding generator. Sends song metadata to the Claude API and
stores n-dimensional profiles in `embeddings/vectors.json`.

### Functions

| Function | Signature | Description |
|----------|-----------|-------------|
| `song_id` | `(song: Dict) → str` | Stable 8-char hex ID from artist+title. |
| `generate_song_embedding` | `(song: Dict, client: Anthropic, model: str) → Dict` | Call Claude to generate embedding. Retries on rate limits. Falls back to neutral embedding. |
| `embed_library` | `(config_path: Path) → Dict` | Embed all songs in metadata.json. Skips already-embedded songs (incremental). |

### Embedding Schema

Each embedding is a dict with these fields:

| Field | Type | Range | Description |
|-------|------|-------|-------------|
| `energy` | `int` | 0–10 | Calm (0) to intense (10) |
| `mood` | `str` | — | Primary mood descriptor |
| `tempo_feel` | `int` | 0–10 | Slow (0) to fast (10) |
| `instrumentation` | `str` | — | Primary instrument family |
| `complexity` | `int` | 0–10 | Simple (0) to intricate (10) |
| `danceability` | `int` | 0–10 | How suitable for dancing |
| `vocality` | `int` | 0–10 | Instrumental (0) to vocal-heavy (10) |
| `era` | `str` | — | Approximate era label |
| `lyrical_themes` | `List[str]` | — | Up to 3 theme keywords |
| `similar_artists` | `List[str]` | — | Up to 3 similar artists |
| `transition_compatibility` | `List[str]` | — | Compatible genres for transitions |

---

## scripts/dj.py

DJ engine. Builds playlists by walking the embedding space from a seed song.

### Functions

| Function | Signature | Description |
|----------|-----------|-------------|
| `cosine_similarity` | `(vec1: Dict, vec2: Dict) → float` | Cosine similarity over shared numeric fields. |
| `find_seed_song` | `(vectors: Dict, seed_query: str) → Optional[Tuple]` | Find a song matching a comma-separated `field:value` query. Returns best match. |
| `find_similar_songs` | `(seed_id, seed_embedding, all_vectors, limit, exclude) → List[Tuple]` | Top-N most similar songs by cosine distance. |
| `generate_dj_set` | `(vectors, seed_query, set_length, variety_factor) → List[Dict]` | Generate a full DJ set. Uses weighted random selection controlled by `variety_factor`. |
| `save_playlist` | `(playlist: List[Dict], path: Path) → None` | Serialize playlist to JSON with position, metadata, mood, energy. |
| `parse_args` | `(argv) → Namespace` | Argument parser: `--seed`, `--length`, `--variety`, `--config`. |

### Seed Query Syntax

Comma-separated `key:value` pairs. Omitting the key searches `title`.

```
"artist:Pink Floyd,album:Division Bell"
"title:Coming Back to Life"
"Pink Floyd"                              # searches title
```

---

## scripts/playback_analysis.py

Audio feature extraction for Phase 2 harmonic mixing. Requires `librosa`.

### Functions

| Function | Signature | Description |
|----------|-----------|-------------|
| `detect_bpm` | `(y: ndarray, sr: int) → float` | Tempo estimation via librosa beat tracking. |
| `detect_key` | `(y: ndarray, sr: int) → Dict` | Key detection using Krumhansl–Schmuckler profiles. Returns key, mode, confidence, Camelot code. |
| `compute_spectral_features` | `(y: ndarray, sr: int) → Dict` | Spectral centroid, rolloff, ZCR, RMS energy. |
| `detect_fade_points` | `(y: ndarray, sr: int) → Dict` | Intro/outro timestamps for crossfade placement. |
| `analyse_file` | `(file_path: str) → Dict` | Full analysis of one audio file. |
| `analyse_library` | `() → None` | Batch-analyse all songs, merging results into vectors.json. |

### Audio Features Schema

| Field | Type | Description |
|-------|------|-------------|
| `bpm` | `float` | Estimated beats per minute |
| `key` | `str` | Musical key (e.g. "C", "F#") |
| `mode` | `str` | "major" or "minor" |
| `confidence` | `float` | Key detection confidence (0–1) |
| `camelot` | `str` | Camelot Wheel code (e.g. "8B") |
| `spectral_centroid` | `float` | Mean spectral centroid (Hz) |
| `spectral_rolloff` | `float` | Mean spectral rolloff (Hz) |
| `zero_crossing_rate` | `float` | Mean ZCR |
| `rms_energy` | `float` | Mean RMS energy |
| `intro_end_sec` | `float` | Timestamp where intro energy rises |
| `outro_start_sec` | `float` | Timestamp where outro energy falls |
| `duration_sec` | `float` | Total duration in seconds |

---

## tests/mock_data.py

Test data factories. No external dependencies.

| Function | Signature | Description |
|----------|-----------|-------------|
| `make_embedding` | `(**kwargs) → Dict` | Create an embedding dict with defaults. |
| `make_song` | `(artist, title, album, ...) → Dict` | Create a full song entry (metadata + embedding). |
| `make_vectors` | `(songs) → Dict` | Build a vectors dict keyed by sequential IDs. |

| Constant | Description |
|----------|-------------|
| `SAMPLE_SONGS` | List of 10 pre-built songs spanning rock, electronic, jazz, classical, hip-hop. |
| `SAMPLE_LIBRARY` | `make_vectors(SAMPLE_SONGS)` — ready-to-use vectors dict. |

---

## Configuration (config/config.json)

| Section | Key | Type | Description |
|---------|-----|------|-------------|
| `embedding` | `model` | `str` | Claude model for embeddings |
| `embedding` | `max_retries` | `int` | API retry count |
| `dj_config` | `variety_factor` | `float` | Randomness in song selection (0–1) |
| `dj_config` | `default_set_length` | `int` | Default playlist length |
| `dj_config` | `similarity_threshold` | `float` | Minimum similarity to consider |
| `dj_config` | `crossfade_duration_ms` | `int` | Crossfade length (Phase 2) |
| `audio_analysis` | `sample_rate` | `int` | librosa sample rate |
| `audio_analysis` | `fade_threshold` | `float` | RMS fraction for fade detection |
| `audio_analysis` | `fade_window_sec` | `int` | Intro/outro scan window |
