"""Generate demo guitar tracks (synthesised plucks) and analyze them.

Usage:  python examples/make_demo.py [output_dir]
"""

import sys
from pathlib import Path

import soundfile as sf

from fretmap.music import name_to_midi
from fretmap.synth import render_sequence

SR = 44100


def n(*names: str) -> list[int]:
    return [name_to_midi(x) for x in names]


DEMOS = {
    # A simple melody line (single notes).
    "melody": [
        (0.0, n("E2"), 0.45), (0.5, n("G2"), 0.45), (1.0, n("A2"), 0.45),
        (1.5, n("C3"), 0.45), (2.0, n("D3"), 0.45), (2.5, n("E3"), 0.9),
    ],
    # Rhythm part mixing chords, a power-chord interval and single notes.
    "rhythm": [
        (0.0, n("E2", "B2", "E3", "G#3", "B3", "E4"), 0.9),   # E major
        (1.0, n("A2", "E3", "A3", "C4", "E4"), 0.9),          # A minor
        (2.0, n("G2", "D3"), 0.45),                            # G5 power chord
        (2.5, n("B3"), 0.4),                                   # single note
        (3.0, n("C4"), 0.4),                                   # single note
        (3.5, n("D3", "A3", "D4", "F#4"), 0.9),               # D major
    ],
}


def main() -> None:
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, events in DEMOS.items():
        path = out_dir / f"{name}.wav"
        sf.write(path, render_sequence(events, sr=SR), SR)
        print(f"wrote {path}")
    print(f"\nTry:  fretmap {out_dir / 'rhythm.wav'}")


if __name__ == "__main__":
    main()
