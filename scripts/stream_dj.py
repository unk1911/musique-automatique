#!/usr/bin/env python3
"""Live streaming DJ -- plays crossfaded music directly to phone via ADB.

Streams continuous PCM audio through an ADB pipe to a headless Android
AudioTrack player.  Picks each next song dynamically using semantic
embeddings, crossfading seamlessly between tracks.

Usage:
    python3 scripts/stream_dj.py --seed "artist:Portishead"
    python3 scripts/stream_dj.py --seed "title:Dope Coil" --variety 0.5
    python3 scripts/stream_dj.py --seed "Chemical Brothers" --crossfade 5000

Requires:
    - adb connected to phone
    - stream.dex built and available (from adb-audio-player)
    - pip install pydub
"""

import argparse
import json
import random
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

try:
    from pydub import AudioSegment
except ImportError:
    print("Missing pydub. Install: pip install pydub")
    sys.exit(1)

# Allow importing from scripts/
sys.path.insert(0, str(Path(__file__).parent))
from dj import find_seed_song, find_similar_songs, _pick_candidate
from discover import discover_next, load_env

PROJECT_ROOT = Path(__file__).parent.parent
VECTORS_FILE = PROJECT_ROOT / "embeddings" / "vectors.json"
CONFIG_FILE = PROJECT_ROOT / "config" / "config.json"

# Audio format for streaming (music quality)
SAMPLE_RATE = 44100
CHANNELS = 2
SAMPLE_WIDTH = 2  # 16-bit
CHUNK_SIZE = 32768  # ~185ms of audio at 44100/stereo/16bit

# ADB paths
ADB_AUDIO_PLAYER = Path(__file__).parent.parent.parent / "adb-audio-player"
LOCAL_DEX = ADB_AUDIO_PLAYER / "bin" / "stream.dex"
REMOTE_DEX = "/data/local/tmp/stream.dex"
REQUEST_FILE = Path("/tmp/musique.next")

# Global for signal handler
_adb_proc: Optional[subprocess.Popen] = None
_stopping = False


def adb(*args, quiet=False):
    """Run an ADB command."""
    cmd = ["adb"] + list(args)
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if not quiet and result.returncode != 0:
        sys.stderr.write(f"adb error: {result.stderr}\n")
    return result.returncode, result.stdout, result.stderr


def ensure_dex():
    """Push stream.dex to phone if needed."""
    if not LOCAL_DEX.exists():
        print(f"stream.dex not found at {LOCAL_DEX}")
        print("Build it: cd adb-audio-player && ./build.sh")
        sys.exit(1)
    print(f"Pushing stream.dex to device...")
    rc, _, err = adb("push", str(LOCAL_DEX), REMOTE_DEX)
    if rc != 0:
        print(f"Failed to push DEX: {err}")
        sys.exit(1)


def set_volume(pct: int):
    """Set media volume on device (0-100 -> 0-15)."""
    level = max(0, min(15, int(pct * 15 / 100)))
    adb("shell", "cmd", "media_session", "volume", "--stream", "3",
        "--set", str(level), quiet=True)


def kill_existing():
    """Kill any existing StreamAudio process on device."""
    adb("shell", "pkill", "-f", "app_process.*StreamAudio", quiet=True)
    time.sleep(0.3)


def open_stream() -> subprocess.Popen:
    """Open ADB streaming subprocess."""
    cmd = [
        "adb", "shell",
        f"app_process -Djava.class.path={REMOTE_DEX} / StreamAudio {SAMPLE_RATE} {CHANNELS}"
    ]
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Drain stdout/stderr in background to prevent pipe deadlock
    def drain(pipe, label):
        for line in pipe:
            line = line.decode("utf-8", errors="replace").strip()
            if line:
                sys.stderr.write(f"  [{label}] {line}\n")

    threading.Thread(target=drain, args=(proc.stdout, "device"), daemon=True).start()
    threading.Thread(target=drain, args=(proc.stderr, "device-err"), daemon=True).start()

    time.sleep(0.5)  # let AudioTrack initialize
    if proc.poll() is not None:
        print("StreamAudio failed to start on device")
        sys.exit(1)

    return proc


def load_and_normalize(song_path: str) -> Optional[AudioSegment]:
    """Load an MP3 and normalize to streaming format."""
    path = PROJECT_ROOT / song_path
    if not path.exists():
        return None
    try:
        audio = AudioSegment.from_file(str(path))
        audio = audio.set_frame_rate(SAMPLE_RATE)
        audio = audio.set_channels(CHANNELS)
        audio = audio.set_sample_width(SAMPLE_WIDTH)
        return audio
    except Exception as exc:
        sys.stderr.write(f"Could not load {path}: {exc}\n")
        return None


def write_pcm(pipe, raw_data: bytes):
    """Write raw PCM bytes to the ADB pipe in chunks."""
    offset = 0
    while offset < len(raw_data) and not _stopping:
        end = min(offset + CHUNK_SIZE, len(raw_data))
        chunk = raw_data[offset:end]
        try:
            pipe.write(chunk)
            pipe.flush()
        except BrokenPipeError:
            raise ConnectionError("ADB pipe broken -- phone disconnected?")
        offset = end


def calculate_crossfade(current_data: Dict, next_data: Dict, default_ms: int) -> int:
    """BPM-aware crossfade duration."""
    prev_bpm = current_data.get("audio_features", {}).get("bpm", 0)
    curr_bpm = next_data.get("audio_features", {}).get("bpm", 0)
    if prev_bpm > 0 and curr_bpm > 0:
        bpm_diff = abs(prev_bpm - curr_bpm)
        if bpm_diff < 5:
            return int(default_ms * 1.5)
        elif bpm_diff > 30:
            return int(default_ms * 0.6)
    return default_ms


def pick_next(
    current_data: Dict,
    current_id: str,
    vectors: Dict,
    played: set,
    variety: float,
) -> Optional[Tuple[str, Dict, float]]:
    """Pick the next song using embedding similarity."""
    candidate_pool = max(3, int(5 * (1 + variety)))
    similar = find_similar_songs(
        seed_id=current_id,
        seed_embedding=current_data["embedding"],
        all_vectors=vectors,
        limit=candidate_pool,
        exclude=played,
    )
    if not similar:
        return None
    return _pick_candidate(similar, variety)


def print_now_playing(song_data: Dict, position: int, similarity: float = 0, is_seed: bool = False):
    """Print current song info."""
    meta = song_data["metadata"]
    emb = song_data["embedding"]
    af = song_data.get("audio_features", {})

    artist = meta.get("artist", "Unknown")
    title = meta.get("title", meta.get("filename", "Unknown"))
    mood = emb.get("mood", "?")
    bpm = af.get("bpm")
    key_str = af.get("key", "")
    mode = af.get("mode", "")

    bpm_str = f"  bpm={bpm:.0f}" if bpm else ""
    key_full = f"  key={key_str} {mode}" if key_str else ""
    sim_str = f"  sim={similarity:.2f}" if not is_seed else "  [seed]"

    print(f"\n  [{position:2d}] {artist} - {title}")
    print(f"       mood={mood}{bpm_str}{key_full}{sim_str}")


def stream_dj(seed_query: str, variety: float, crossfade_ms: int, volume: int):
    """Main DJ loop -- streams crossfaded audio to phone."""
    global _adb_proc, _stopping

    # Load library
    if not VECTORS_FILE.exists():
        print("No vectors.json found. Run embed_songs.py first.")
        sys.exit(1)
    with open(VECTORS_FILE) as f:
        vectors = json.load(f)
    print(f"Library: {len(vectors)} songs")

    # Init Anthropic client for discovery
    load_env()
    try:
        import anthropic
        _anthropic_client = anthropic.Anthropic()
    except Exception:
        _anthropic_client = None
        print("(Discovery disabled -- no ANTHROPIC_API_KEY)")

    # Find seed
    seed_result = find_seed_song(vectors, seed_query)
    if not seed_result:
        print(f"Seed not found: {seed_query}")
        sys.exit(1)
    current_id, current_data = seed_result

    # Setup device
    ensure_dex()
    kill_existing()
    set_volume(volume)

    # Open stream
    print(f"\nStarting stream (Ctrl+C to stop)...")
    _adb_proc = open_stream()
    proc = _adb_proc

    played = {current_id}
    position = 1

    # Load first song
    current_audio = load_and_normalize(current_data["metadata"]["path"])
    if current_audio is None:
        print(f"Could not load seed song")
        proc.stdin.close()
        sys.exit(1)

    print_now_playing(current_data, position, is_seed=True)
    print(f"       duration={len(current_audio) / 1000:.0f}s")

    try:
        while not _stopping:
            # Check for a manually requested next song
            result = None
            if REQUEST_FILE.exists():
                try:
                    req = json.loads(REQUEST_FILE.read_text())
                    REQUEST_FILE.unlink(missing_ok=True)
                    req_id = req["id"]
                    req_data = req["data"]
                    if req_id == current_id:
                        pass  # already playing this song, ignore stale request
                    else:
                        # Inject into in-memory vectors so it persists
                        vectors[req_id] = req_data
                        result = (req_id, req_data, 1.0)
                        print("\n  --- Transitioning to requested song ---")
                except Exception as _req_err:
                    sys.stderr.write(f"  Request file error: {_req_err}\n")

            # Pick next song
            if result is None:
                result = pick_next(current_data, current_id, vectors, played, variety)
            if result is None:
                # Library exhausted -- discover new music
                if _anthropic_client:
                    print("\n  --- Discovering new music... ---")
                    discovery = discover_next(vectors, current_data, _anthropic_client)
                    if discovery:
                        new_sid, new_data = discovery
                        result = (new_sid, new_data, 0.0)
                    else:
                        print("  Discovery failed, wrapping library")
                        played = {current_id}
                        result = pick_next(current_data, current_id, vectors, played, variety)
                else:
                    print("\n  --- Library wrap (no API key for discovery) ---")
                    played = {current_id}
                    result = pick_next(current_data, current_id, vectors, played, variety)
                if result is None:
                    print("No candidates available")
                    break

            next_id, next_data, similarity = result

            # Load next song (skip if can't load)
            next_audio = load_and_normalize(next_data["metadata"]["path"])
            if next_audio is None:
                played.add(next_id)
                continue

            # Calculate crossfade
            xfade_ms = calculate_crossfade(current_data, next_data, crossfade_ms)
            # Clamp to half the shorter song
            xfade_ms = min(xfade_ms, len(current_audio) // 2, len(next_audio) // 2)

            # Stream current song body (everything except the crossfade tail)
            body = current_audio[:-xfade_ms]
            write_pcm(proc.stdin, body.raw_data)

            # Crossfade: blend tail of current with head of next
            tail = current_audio[-xfade_ms:]
            head = next_audio[:xfade_ms]
            blended = tail.fade_out(xfade_ms).overlay(head.fade_in(xfade_ms))
            write_pcm(proc.stdin, blended.raw_data)

            # Advance
            current_audio = next_audio[xfade_ms:]  # remaining part of next song
            current_data = next_data
            current_id = next_id
            played.add(next_id)
            position += 1

            print_now_playing(current_data, position, similarity)
            print(f"       duration={len(next_audio) / 1000:.0f}s  xfade={xfade_ms}ms")

        # Flush the last song's remaining audio
        if current_audio and not _stopping:
            write_pcm(proc.stdin, current_audio.raw_data)

    except ConnectionError as e:
        print(f"\n{e}")
    except KeyboardInterrupt:
        pass
    finally:
        print("\n\nStopping DJ...")
        _stopping = True
        if proc.stdin:
            try:
                proc.stdin.close()
            except Exception:
                pass
        proc.wait(timeout=5)
        print("Done.")


def fetch_on_demand(query: str) -> Optional[Tuple[str, Dict]]:
    """Find or download a song by query. Returns (song_id, song_data) or None."""
    if not VECTORS_FILE.exists():
        return None
    with open(VECTORS_FILE) as f:
        vectors = json.load(f)

    # Already in library?
    result = find_seed_song(vectors, query)
    if result:
        return result

    # Parse "Artist - Title" or "artist:X,title:Y"
    artist, title = None, None
    if ":" in query and "," in query:
        for part in query.split(","):
            if ":" in part:
                k, v = part.split(":", 1)
                k, v = k.strip(), v.strip()
                if k == "artist":
                    artist = v
                if k == "title":
                    title = v
    elif " - " in query:
        parts = query.split(" - ", 1)
        artist, title = parts[0].strip(), parts[1].strip()

    if not artist or not title:
        print(f"Could not parse artist/title from: {query!r}")
        print('Use format: "Artist - Title" or "artist:X,title:Y"')
        return None

    # Download
    load_env()
    from discover import download_song, analyse_downloaded
    from discover import song_id as make_sid, save_vectors
    from embed_songs import generate_song_embedding

    print(f"  Downloading: {artist} - {title}")
    file_path = download_song(artist, title)
    if not file_path:
        print(f"  Download failed.")
        return None

    rel_path = str(Path(file_path).relative_to(PROJECT_ROOT))
    audio_features = analyse_downloaded(file_path)

    try:
        import anthropic as _anthropic
        _client = _anthropic.Anthropic()
        embedding = generate_song_embedding({"artist": artist, "title": title, "album": None}, _client)
    except Exception as exc:
        print(f"  Embedding failed: {exc}")
        embedding = {}

    sid = make_sid(artist, title)
    song_data = {
        "metadata": {
            "filename": Path(file_path).name,
            "path": rel_path,
            "artist": artist,
            "title": title,
            "album": None,
            "duration": audio_features.get("duration_sec"),
            "source": "on_demand",
        },
        "embedding": embedding,
        "audio_features": audio_features,
    }
    vectors[sid] = song_data
    save_vectors(vectors)
    print(f"  Added to library: {artist} - {title}")
    return sid, song_data


def request_next(query: str):
    """Queue a specific song as the next track (downloads if needed)."""
    print(f"Fetching: {query}")
    result = fetch_on_demand(query)
    if result:
        song_id, song_data = result
        meta = song_data["metadata"]
        REQUEST_FILE.write_text(json.dumps({"id": song_id, "data": song_data}))
        print(f"Queued: {meta.get('artist')} - {meta.get('title')}")
    else:
        print("Failed to queue song.")


def handle_sigint(signum, frame):
    global _stopping
    _stopping = True


def stop_dj():
    """Stop the DJ from here or from the phone."""
    # Signal phone-side stop
    adb("shell", "touch", "/data/local/tmp/stream.stop", quiet=True)
    # Kill server-side process
    kill_existing()
    print("DJ stopped.")


def main():
    signal.signal(signal.SIGINT, handle_sigint)

    parser = argparse.ArgumentParser(
        description="Musique Automatique Live Streaming DJ",
        epilog='Example: %(prog)s --seed "artist:Portishead"',
    )
    parser.add_argument("--seed", help='Seed query, e.g. "artist:Portishead" (default: random)')
    parser.add_argument("--stop", action="store_true", help="Stop the currently running DJ")
    parser.add_argument("--next", metavar="QUERY", help='Queue a song as next, e.g. "Lou Reed - Coney Island Baby"')
    parser.add_argument("--variety", type=float, default=None, help="Variety factor 0.0-1.0")
    parser.add_argument("--crossfade", type=int, default=None, help="Base crossfade duration in ms")
    parser.add_argument("--volume", type=int, default=75, help="Volume 0-100 (default: 75)")
    args = parser.parse_args()

    if args.stop:
        stop_dj()
        return

    if args.next:
        request_next(args.next)
        return

    if not args.seed:
        if not VECTORS_FILE.exists():
            parser.error("No vectors.json found. Run embed_songs.py first.")
        with open(VECTORS_FILE) as f:
            vectors = json.load(f)
        if not vectors:
            parser.error("Library is empty.")
        random_id = random.choice(list(vectors.keys()))
        meta = vectors[random_id]["metadata"]
        args.seed = f"artist:{meta.get('artist', '')},title:{meta.get('title', meta.get('filename', ''))}"
        print(f"Random seed: {meta.get('artist', '?')} - {meta.get('title', meta.get('filename', '?'))}")

    # Read defaults from config
    variety = args.variety
    crossfade = args.crossfade
    if variety is None or crossfade is None:
        if CONFIG_FILE.exists():
            with open(CONFIG_FILE) as f:
                cfg = json.load(f)
            if variety is None:
                variety = cfg.get("dj_config", {}).get("variety_factor", 0.3)
            if crossfade is None:
                crossfade = cfg.get("dj_config", {}).get("crossfade_duration_ms", 3000)
        else:
            variety = variety or 0.3
            crossfade = crossfade or 3000

    stream_dj(args.seed, variety, crossfade, args.volume)


if __name__ == "__main__":
    main()

