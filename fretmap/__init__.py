"""fretmap: guitar transcription from an isolated guitar track.

Pipeline: audio -> onset detection -> per-segment polyphonic pitch
detection -> note/interval/chord classification -> fretboard position
assignment -> tab / fretboard rendering.
"""

from fretmap.analyze import Event, analyze, analyze_file

__all__ = ["Event", "analyze", "analyze_file"]
__version__ = "0.2.0"
