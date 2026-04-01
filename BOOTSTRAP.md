# Musique Automatique - Bootstrap Guide

Welcome to your taste-aware DJ engine! Here's how to get started.

## Step 1: Add Music

Place 10 test songs (MP3s) in:
```
musique-atomatique/music/
```

**Recommended format:**
```
music/
├── Pink Floyd/
│   └── Division Bell/
│       ├── 01 - Coming Back to Life.mp3
│       ├── 02 - Keep Talking.mp3
│       └── ...
├── Radiohead/
│   └── OK Computer/
│       └── ...
└── ...
```

**Minimal format (filename parsing):**
```
music/
├── Pink Floyd - Coming Back to Life.mp3
├── Radiohead - Paranoid Android.mp3
└── ...
```

## Step 2: Initialize Library

Extract metadata from ID3 tags + filenames:

```bash
python3 scripts/init_library.py
```

Output: `embeddings/metadata.json`

## Step 3: Generate Embeddings

Create semantic vectors for each song using Claude:

```bash
python3 scripts/embed_songs.py
```

This creates an n-dimensional profile:
- **Energy** (0-10): calm → intense
- **Mood**: happy, sad, melancholic, energetic, etc.
- **Tempo Feel** (0-10): slow → fast
- **Instrumentation**: acoustic, electronic, orchestral, rock, etc.
- **Complexity** (0-10): simple → intricate
- **Danceability** (0-10): not danceable → very danceable
- **Vocality** (0-10): instrumental → vocal-heavy
- **Era**: classical, 80s, 90s, 2000s, 2010s, 2020s
- **Lyrical Themes**: ["love", "nature", "politics", ...]
- **Similar Artists**: ["artist1", "artist2", ...]
- **Transition Compatibility**: ["genre1", "genre2", ...]

Output: `embeddings/vectors.json`

## Step 4: Generate a DJ Set

Start from a seed song and let the system build an intelligent playlist:

```bash
python3 scripts/dj.py --seed "Pink Floyd,Coming Back to Life"
```

The DJ engine will:
1. Find your seed song
2. Use embeddings to find thematically similar songs
3. Build a coherent 10-song set
4. Create smooth transitions between tracks

Output: `embeddings/playlist.json`

## Example Seed Queries

```bash
# Search by artist + album
python3 scripts/dj.py --seed "artist:Pink Floyd,album:Division Bell"

# Search by title
python3 scripts/dj.py --seed "title:Coming Back to Life"

# Just artist
python3 scripts/dj.py --seed "Pink Floyd"

# Short hand
python3 scripts/dj.py --seed "Pink Floyd,Coming Back"
```

## Next Steps (Phase 2)

Once the embeddings work, we'll add:

1. **Audio Blending**: Analyze song endings for crossfade points
2. **Harmonic Mixing**: Key/BPM analysis for seamless transitions
3. **Real-time Playback**: Queue songs with auto-blending
4. **User Feedback Loop**: Learn from skips/repeats to refine taste model

## Architecture

```
Input (MP3s with metadata)
         ↓
   [init_library.py]
         ↓
   metadata.json (artist, album, title, duration, genre)
         ↓
  [embed_songs.py] ← Claude API
         ↓
   vectors.json (semantic embeddings + n-dim profiles)
         ↓
     [dj.py] ← User selects seed
         ↓
   playlist.json (ordered songs with transitions)
         ↓
  [playback engine] (future: audio blending)
         ↓
   ♫ Your taste, automated ♫
```

---

**Ready to build the future of music?** Let's start with 10 songs and iterate.
