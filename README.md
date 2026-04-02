# Musique Automatique

A taste-aware, AI-powered music DJ that analyzes your library, builds semantic embeddings of each song, and streams continuous crossfaded mixes directly to your phone via ADB.

## How It Works

1. **Scan** your MP3 library -- extracts ID3 tags, falls back to directory/filename parsing
2. **Embed** each song via Claude API -- generates n-dimensional semantic vectors (mood, energy, tempo, complexity, danceability, etc.)
3. **Analyze** audio features via librosa -- BPM, musical key, spectral profile, fade points
4. **DJ** -- walks the embedding space using cosine similarity, picking songs that flow naturally from the current track
5. **Stream** -- pipes continuous crossfaded PCM audio to a headless Android AudioTrack player over ADB

## Quick Start

```bash
# 1. Scan and extract metadata
python3 scripts/init_library.py

# 2. Generate semantic embeddings (requires ANTHROPIC_API_KEY)
python3 scripts/embed_songs.py

# 3. Analyze audio features (BPM, key, fade points)
python3 scripts/playback_analysis.py

# 4. Stream live DJ to phone (random seed)
python3 scripts/stream_dj.py
```

## Live Streaming DJ

Streams crossfaded audio directly to your phone speaker via ADB. Picks each next song dynamically based on semantic similarity.

```bash
# Start with a random song (default)
python3 scripts/stream_dj.py

# Start from a specific seed
python3 scripts/stream_dj.py --seed "artist:Portishead,title:Glory Box"
python3 scripts/stream_dj.py --seed "Chemical Brothers" --variety 0.5
python3 scripts/stream_dj.py --seed "title:Dope Coil" --crossfade 5000 --volume 80

# Queue a specific song as next (downloads from YouTube if not in library)
python3 scripts/stream_dj.py --next "Lou Reed - Coney Island Baby"
python3 scripts/stream_dj.py --next "artist:Radiohead,title:Creep"

# Stop the DJ
python3 scripts/stream_dj.py --stop
```

**Options:**
- `--seed` -- seed query (artist, title, or field:value pairs); omit for random
- `--next` -- queue a song as the next track; downloads from YouTube if not in library
- `--variety` -- 0.0 (always most similar) to 1.0 (more random), default 0.3
- `--crossfade` -- crossfade duration in ms, default 3000
- `--volume` -- phone volume 0-100, default 75

**Stopping playback:**
- From server: `python3 scripts/stream_dj.py --stop`
- From phone (Termux): `touch /data/local/tmp/stream.stop`
- Physical: disconnect ADB

## Batch Mix Export

Generate a playlist and export a crossfaded MP3 file:

```bash
# Generate playlist
python3 scripts/dj.py --seed "artist:Portishead" --length 12

# Export crossfaded mix
python3 scripts/play.py --output my_mix.mp3 --crossfade 5000

# Or dry-run to see the playlist
python3 scripts/play.py --dry-run
```

## Directory Structure

```
musique-automatique/
├── music/              # MP3 library (flat directory of MP3s)
├── embeddings/
│   ├── metadata.json   # Song metadata (artist, album, title, etc.)
│   ├── vectors.json    # Semantic embeddings + audio features per song
│   └── playlist.json   # Last generated DJ set
├── config/
│   └── config.json     # Model, crossfade, variety settings
└── scripts/
    ├── init_library.py       # Scan music, extract metadata
    ├── embed_songs.py        # Generate semantic vectors (Claude API)
    ├── playback_analysis.py  # BPM, key, spectral analysis (librosa)
    ├── dj.py                 # Playlist generator (cosine similarity walk)
    ├── play.py               # Batch mix builder + export
    └── stream_dj.py          # Live streaming DJ via ADB
```

## Requirements

- Python: `pydub`, `mutagen`, `anthropic`, `librosa`, `numpy`, `soundfile`
- System: `ffmpeg`, `adb`
- Android: `stream.dex` (built from `adb-audio-player/build.sh`)
- API: `ANTHROPIC_API_KEY` for embedding generation

## Embedding Dimensions

Each song is characterized across these semantic axes (0-10 scale where applicable):

| Dimension | Type | Example |
|-----------|------|---------|
| energy | numeric | 7 |
| mood | string | "melancholic", "dark hypnotic" |
| tempo_feel | numeric | 5 |
| instrumentation | string | "synthesizers, drum machines" |
| complexity | numeric | 8 |
| danceability | numeric | 3 |
| vocality | numeric | 9 |
| era | string | "1990s" |
| lyrical_themes | list | ["alienation", "love"] |
| similar_artists | list | ["Massive Attack", "Tricky"] |
| transition_compatibility | list | ["trip-hop", "downtempo"] |

Audio features (from librosa analysis):

| Feature | Example |
|---------|---------|
| bpm | 123.0 |
| key / mode | A minor |
| camelot | 8A |
| spectral_centroid | 1989.9 |
| intro_end_sec / outro_start_sec | 0.5 / 303.8 |
