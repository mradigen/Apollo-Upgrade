#!/usr/bin/env python3
"""Introduce audio degradations to a WAV file."""

import argparse
import pathlib

import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve


def parse_args():
    p = argparse.ArgumentParser(description="Degrade a WAV file.")
    p.add_argument("input", type=pathlib.Path, help="Input .wav file")
    p.add_argument(
        "--degradations",
        nargs="+",
        choices=["clip", "missing", "reverb"],
        default=["clip", "missing", "reverb"],
        help="Degradations to apply",
    )
    p.add_argument(
        "--mode",
        choices=["separate", "combined", "both"],
        default="both",
        help="Output mode",
    )
    p.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=pathlib.Path("dataset/output"),
        help="Output directory",
    )
    p.add_argument("--clip-threshold", type=float, default=0.5,
                   help="Clipping threshold as fraction of peak amplitude (0-1)")
    p.add_argument("--missing-rate", type=float, default=0.01,
                   help="Fraction of samples to zero out as contiguous blocks")
    p.add_argument("--reverb-decay", type=float, default=2.0,
                   help="Reverberation decay time in seconds")
    return p.parse_args()


def apply_clipping(audio: np.ndarray, threshold: float) -> np.ndarray:
    """Hard-clip audio at ±threshold * peak."""
    peak = np.max(np.abs(audio))
    limit = threshold * peak
    return np.clip(audio, -limit, limit)


def apply_missing_samples(audio: np.ndarray, rate: float, rng: np.random.Generator) -> np.ndarray:
    """Zero out random contiguous blocks totalling ~rate fraction of samples."""
    out = audio.copy()
    total_to_zero = int(len(audio) * rate)
    block_size = max(1, len(audio) // 1000)  # ~0.1% per block
    n_blocks = max(1, total_to_zero // block_size)
    starts = rng.integers(0, len(audio) - block_size, size=n_blocks)
    for s in starts:
        out[s : s + block_size] = 0.0
    return out


def apply_reverb(audio: np.ndarray, sr: int, decay: float) -> np.ndarray:
    """Convolve with a synthetic exponentially-decaying RIR."""
    rng = np.random.default_rng(42)
    n = int(sr * decay)
    t = np.linspace(0, decay, n)
    rir = rng.standard_normal(n) * np.exp(-6.0 * t / decay)
    rir /= np.max(np.abs(rir))
    if audio.ndim == 2:  # multichannel: convolve each channel
        channels = [
            fftconvolve(audio[:, c], rir, mode="full")[: len(audio)]
            for c in range(audio.shape[1])
        ]
        wet = np.stack(channels, axis=1)
    else:
        wet = fftconvolve(audio, rir, mode="full")[: len(audio)]
    # Normalise to avoid clipping after reverb
    peak = np.max(np.abs(wet))
    if peak > 0:
        wet = wet / peak * np.max(np.abs(audio))
    return wet


DEGRADATION_MAP = {
    "clip": lambda audio, sr, args: apply_clipping(audio, args.clip_threshold),
    "missing": lambda audio, sr, args: apply_missing_samples(
        audio, args.missing_rate, np.random.default_rng(0)
    ),
    "reverb": lambda audio, sr, args: apply_reverb(audio, sr, args.reverb_decay),
}


def save(path: pathlib.Path, audio: np.ndarray, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, sr)
    print(f"  wrote {path}")


def main() -> None:
    args = parse_args()
    audio, sr = sf.read(str(args.input))
    stem = args.input.stem
    out = args.output_dir

    if args.mode in ("separate", "both"):
        for name in args.degradations:
            degraded = DEGRADATION_MAP[name](audio, sr, args)
            save(out / f"{stem}_{name}.wav", degraded, sr)

    if args.mode in ("combined", "both"):
        combined = audio.copy()
        for name in args.degradations:
            combined = DEGRADATION_MAP[name](combined, sr, args)
        save(out / f"{stem}_degraded.wav", combined, sr)


if __name__ == "__main__":
    main()
