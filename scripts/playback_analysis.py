#!/usr/bin/env python3
"""Audio feature extraction for Phase 2 harmonic mixing and crossfade.

Uses librosa to extract BPM, musical key, onset envelope, and spectral
features from MP3 files.  Results are stored alongside semantic embeddings
in vectors.json to enable beat-matched transitions and harmonic mixing.

Usage:
    python3 scripts/playback_analysis.py                  # analyse whole library
    python3 scripts/playback_analysis.py --file song.mp3  # single file

Output:
    Merges an "audio_features" key into each entry in vectors.json.

Requires:
    pip install librosa numpy soundfile
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import librosa
    import numpy as np
except ImportError:
    print("Missing librosa/numpy. Install: pip install librosa numpy soundfile")
    sys.exit(1)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
EMBEDDINGS_DIR = PROJECT_ROOT / "embeddings"
VECTORS_FILE = EMBEDDINGS_DIR / "vectors.json"

# Chroma-index-to-key mapping (Krumhansl–Schmuckler profiles)
_PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def detect_bpm(y: np.ndarray, sr: int) -> float:
    """Estimate tempo (BPM) from an audio signal.

    Args:
        y: Audio time-series (mono).
        sr: Sample rate.

    Returns:
        Estimated BPM as a float.
    """
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    # librosa >= 0.10 returns an array; older versions return a scalar
    if hasattr(tempo, "__len__"):
        return float(tempo[0])
    return float(tempo)


def detect_key(y: np.ndarray, sr: int) -> Dict[str, Any]:
    """Estimate musical key using chroma features.

    Uses the Krumhansl–Schmuckler key-finding algorithm: computes chroma
    energy, then correlates with major and minor profiles.

    Args:
        y: Audio time-series (mono).
        sr: Sample rate.

    Returns:
        Dict with "key" (e.g. "C"), "mode" ("major"/"minor"),
        "confidence" (0-1), and "camelot" (Camelot Wheel code).
    """
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_energy = chroma.mean(axis=1)

    # Krumhansl major/minor profiles
    major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                              2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                              2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

    best_key = "C"
    best_mode = "major"
    best_corr = -1.0

    for shift in range(12):
        rotated = np.roll(chroma_energy, -shift)
        corr_major = float(np.corrcoef(rotated, major_profile)[0, 1])
        corr_minor = float(np.corrcoef(rotated, minor_profile)[0, 1])

        if corr_major > best_corr:
            best_corr = corr_major
            best_key = _PITCH_CLASSES[shift]
            best_mode = "major"
        if corr_minor > best_corr:
            best_corr = corr_minor
            best_key = _PITCH_CLASSES[shift]
            best_mode = "minor"

    return {
        "key": best_key,
        "mode": best_mode,
        "confidence": round(max(best_corr, 0.0), 3),
        "camelot": _to_camelot(best_key, best_mode),
    }


def _to_camelot(key: str, mode: str) -> str:
    """Convert key + mode to Camelot Wheel notation.

    The Camelot Wheel is widely used by DJs for harmonic mixing.
    Adjacent codes (e.g. 8A → 9A, 8A → 8B) are harmonically compatible.

    Args:
        key: Note name (e.g. "C", "F#").
        mode: "major" or "minor".

    Returns:
        Camelot code (e.g. "8B", "5A").
    """
    major_wheel = {
        "B": "1B", "F#": "2B", "C#": "3B", "G#": "4B",
        "D#": "5B", "A#": "6B", "F": "7B", "C": "8B",
        "G": "9B", "D": "10B", "A": "11B", "E": "12B",
    }
    minor_wheel = {
        "G#": "1A", "D#": "2A", "A#": "3A", "F": "4A",
        "C": "5A", "G": "6A", "D": "7A", "A": "8A",
        "E": "9A", "B": "10A", "F#": "11A", "C#": "12A",
    }
    if mode == "minor":
        return minor_wheel.get(key, "?")
    return major_wheel.get(key, "?")


def compute_spectral_features(y: np.ndarray, sr: int) -> Dict[str, float]:
    """Compute spectral summary statistics useful for transition matching.

    Args:
        y: Audio time-series (mono).
        sr: Sample rate.

    Returns:
        Dict with spectral_centroid, spectral_rolloff, zero_crossing_rate,
        and rms_energy — all mean values over the full track.
    """
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    rms = librosa.feature.rms(y=y)

    return {
        "spectral_centroid": float(centroid.mean()),
        "spectral_rolloff": float(rolloff.mean()),
        "zero_crossing_rate": float(zcr.mean()),
        "rms_energy": float(rms.mean()),
    }


def detect_fade_points(y: np.ndarray, sr: int) -> Dict[str, float]:
    """Detect intro/outro energy ramps for crossfade placement.

    Scans the first and last 30 seconds of the track and reports the
    timestamp where energy exceeds (intro) or drops below (outro) 25%
    of peak RMS.

    Args:
        y: Audio time-series (mono).
        sr: Sample rate.

    Returns:
        Dict with "intro_end_sec" and "outro_start_sec".
    """
    window = sr * 30  # 30-second windows
    frame_length = 2048
    hop_length = 512

    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    peak_rms = rms.max()
    threshold = peak_rms * 0.25

    # Intro: first frame above threshold
    intro_end = 0.0
    intro_frames = rms[: window // hop_length]
    for i, val in enumerate(intro_frames):
        if val >= threshold:
            intro_end = librosa.frames_to_time(i, sr=sr, hop_length=hop_length)
            break

    # Outro: last frame above threshold
    outro_start = librosa.get_duration(y=y, sr=sr)
    outro_frames = rms[-(window // hop_length) :]
    offset = len(rms) - len(outro_frames)
    for i in range(len(outro_frames) - 1, -1, -1):
        if outro_frames[i] >= threshold:
            outro_start = librosa.frames_to_time(offset + i, sr=sr, hop_length=hop_length)
            break

    return {
        "intro_end_sec": round(float(intro_end), 2),
        "outro_start_sec": round(float(outro_start), 2),
    }


def analyse_file(file_path: str) -> Dict[str, Any]:
    """Run full audio analysis on a single MP3/audio file.

    Args:
        file_path: Path to the audio file.

    Returns:
        Dict with bpm, key_info, spectral, and fade_points.

    Raises:
        FileNotFoundError: If file_path does not exist.
        RuntimeError: If librosa cannot decode the file.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    logger.info("Analysing %s", path.name)
    try:
        y, sr = librosa.load(str(path), sr=22050, mono=True)
    except Exception as exc:
        raise RuntimeError(f"Could not decode {file_path}: {exc}") from exc

    bpm = detect_bpm(y, sr)
    key_info = detect_key(y, sr)
    spectral = compute_spectral_features(y, sr)
    fades = detect_fade_points(y, sr)

    features = {
        "bpm": round(bpm, 1),
        **key_info,
        **spectral,
        **fades,
        "duration_sec": round(float(librosa.get_duration(y=y, sr=sr)), 2),
    }

    logger.info(
        "  BPM=%.1f  Key=%s %s (%s)  dur=%.0fs",
        features["bpm"], features["key"], features["mode"],
        features["camelot"], features["duration_sec"],
    )
    return features


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------

def analyse_library() -> None:
    """Analyse all songs in vectors.json and merge audio features.

    Skips songs that already have an "audio_features" key or whose
    file path is missing from disk.
    """
    if not VECTORS_FILE.exists():
        logger.error("Vectors file not found: %s — run embed_songs.py first", VECTORS_FILE)
        sys.exit(1)

    with open(VECTORS_FILE) as f:
        vectors = json.load(f)

    updated = 0
    for sid, data in vectors.items():
        if "audio_features" in data:
            logger.debug("Skipping %s (already analysed)", sid)
            continue

        file_path = data.get("metadata", {}).get("path")
        if not file_path or not Path(file_path).exists():
            logger.warning("Skipping %s — file not found: %s", sid, file_path)
            continue

        try:
            features = analyse_file(file_path)
            data["audio_features"] = features
            updated += 1
        except (FileNotFoundError, RuntimeError) as exc:
            logger.error("Failed to analyse %s: %s", sid, exc)

    with open(VECTORS_FILE, "w") as f:
        json.dump(vectors, f, indent=2)

    logger.info("Analysed %d new tracks (%d total in library)", updated, len(vectors))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if len(sys.argv) > 2 and sys.argv[1] == "--file":
        features = analyse_file(sys.argv[2])
        print(json.dumps(features, indent=2))
    else:
        analyse_library()


if __name__ == "__main__":
    main()
