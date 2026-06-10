"""Command-line interface for fretmap."""

import argparse
import json
import sys

from fretmap.analyze import analyze_file
from fretmap.fretboard import DEFAULT_MAX_FRET, STANDARD_TUNING, parse_tuning
from fretmap.render import (
    render_event_fretboard,
    render_event_list,
    render_summary_fretboard,
    render_tab,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="fretmap",
        description=(
            "Transcribe an isolated guitar track: detect notes, intervals and "
            "chords, and map them to guitar tab and fretboard diagrams."
        ),
    )
    p.add_argument("input", help="audio file (wav/flac/ogg/...)")
    p.add_argument("--tab", action="store_true", help="print ASCII tab")
    p.add_argument("--fretboard", action="store_true", help="print a fretboard diagram per event")
    p.add_argument("--summary", action="store_true", help="print a fretboard map of all positions")
    p.add_argument("--events", action="store_true", help="print the timed event list")
    p.add_argument("--json", action="store_true", help="print machine-readable JSON")
    p.add_argument(
        "--tuning",
        default="E2,A2,D3,G3,B3,E4",
        help="comma-separated open-string notes, low to high (default: standard)",
    )
    p.add_argument("--max-fret", type=int, default=DEFAULT_MAX_FRET, help="highest fret (default 24)")
    p.add_argument(
        "--max-polyphony", type=int, default=6, help="max simultaneous notes to detect (default 6)"
    )
    p.add_argument("-o", "--output", help="write output to file instead of stdout")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        tuning = parse_tuning(args.tuning) if args.tuning else STANDARD_TUNING
    except (ValueError, KeyError, IndexError):
        print(f"error: cannot parse tuning {args.tuning!r}", file=sys.stderr)
        return 2

    try:
        events, sr, duration = analyze_file(
            args.input,
            tuning=tuning,
            max_fret=args.max_fret,
            max_polyphony=args.max_polyphony,
        )
    except Exception as exc:  # soundfile raises various error types
        print(f"error: could not analyze {args.input!r}: {exc}", file=sys.stderr)
        return 1

    # Default view if no sections were requested explicitly.
    if not any([args.tab, args.fretboard, args.summary, args.events, args.json]):
        args.tab = args.events = args.summary = True

    out: list[str] = []
    if args.json:
        out.append(
            json.dumps(
                {
                    "input": args.input,
                    "sample_rate": sr,
                    "duration": round(duration, 3),
                    "events": [e.to_dict() for e in events],
                },
                indent=2,
            )
        )
    else:
        n_notes = sum(1 for e in events if e.kind == "note")
        n_int = sum(1 for e in events if e.kind == "interval")
        n_chords = sum(1 for e in events if e.kind == "chord")
        out.append(
            f"{args.input}: {duration:.2f}s @ {sr} Hz — "
            f"{len(events)} events ({n_notes} notes, {n_int} intervals, {n_chords} chords)"
        )
        if args.events:
            out.append("")
            out.append(render_event_list(events))
        if args.tab:
            out.append("")
            out.append(render_tab(events, tuning))
        if args.fretboard:
            for ev in events:
                out.append("")
                out.append(render_event_fretboard(ev, tuning))
        if args.summary:
            out.append("")
            out.append(render_summary_fretboard(events, tuning))

    text = "\n".join(out) + "\n"
    if args.output:
        with open(args.output, "w") as fh:
            fh.write(text)
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
