#!/usr/bin/env python3
import argparse
import importlib.util
import os
from typing import List

import numpy as np
import soundfile as sf


def load_turns(turns_path: str):
    spec = importlib.util.spec_from_file_location("turns_module", turns_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {turns_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    if not hasattr(module, "turns"):
        raise RuntimeError(f"{turns_path} does not define a 'turns' variable")
    turns_list = getattr(module, "turns")
    if not isinstance(turns_list, list):
        raise RuntimeError("'turns' must be a list")
    return turns_list


def resample_linear(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """Simple linear resampler for speech (upsampling-focused).

    - Expects x shape (frames, channels) float32 in [-1, 1].
    - Uses per-channel linear interpolation.
    """
    if sr_in == sr_out:
        return x
    n_in = x.shape[0]
    n_out = int(round(n_in * (sr_out / sr_in)))
    if n_in == 0 or n_out == 0:
        return np.zeros((0, x.shape[1]), dtype=np.float32)
    t_in = np.arange(n_in, dtype=np.float64)
    t_out = np.linspace(0.0, n_in - 1, num=n_out, dtype=np.float64)
    y = np.empty((n_out, x.shape[1]), dtype=np.float32)
    for ch in range(x.shape[1]):
        y[:, ch] = np.interp(t_out, t_in, x[:, ch].astype(np.float64)).astype(np.float32)
    return y


def main():
    parser = argparse.ArgumentParser(description="Resample existing turn audio files to a target sample rate (in-place)")
    parser.add_argument("--turns-path", default="turns.py", help="Path to turns.py")
    parser.add_argument("--target-rate", type=int, default=24000, help="Target sample rate (Hz)")
    parser.add_argument("--start-index", type=int, default=0, help="Start turn index")
    parser.add_argument("--end-index", type=int, default=None, help="End turn index (inclusive); defaults to last")
    parser.add_argument("--only-existing", action="store_true", help="Skip turns missing audio_file (default: skip silently)")
    args = parser.parse_args()

    turns = load_turns(args.turns_path)
    n = len(turns)
    start = max(0, args.start_index)
    end = n - 1 if args.end_index is None else min(args.end_index, n - 1)

    print(f"Resampling turns {start}..{end} to {args.target_rate} Hz (in-place)")

    updated: List[int] = []
    skipped: List[int] = []
    errors: List[int] = []

    for i in range(start, end + 1):
        t = turns[i]
        p = t.get("audio_file")
        if not p:
            if args.only_existing:
                print(f"[{i:03d}] no audio_file; skipping (only-existing)")
            else:
                print(f"[{i:03d}] no audio_file; skipping")
            skipped.append(i)
            continue
        if not os.path.exists(p):
            print(f"[{i:03d}] missing file: {p}; skipping")
            errors.append(i)
            continue

        try:
            data, sr_in = sf.read(p, dtype="float32", always_2d=True)
            if sr_in == args.target_rate:
                print(f"[{i:03d}] {p} already {sr_in} Hz; skipping")
                skipped.append(i)
                continue

            y = resample_linear(data, sr_in=sr_in, sr_out=args.target_rate)
            # Write back as 16-bit PCM
            sf.write(p, y, args.target_rate, subtype="PCM_16")
            print(f"[{i:03d}] wrote {p} @ {args.target_rate} Hz (was {sr_in} Hz)")
            updated.append(i)
        except Exception as e:  # noqa: BLE001
            print(f"[{i:03d}] ERROR {p}: {e}")
            errors.append(i)

    print("Done.")
    print(f"Updated: {len(updated)} | Skipped: {len(skipped)} | Errors: {len(errors)}")


if __name__ == "__main__":
    main()

