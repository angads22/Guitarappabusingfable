"""The GUI module must import (and report-building work) without tkinter."""

import soundfile as sf

from fretmap import gui
from fretmap.music import name_to_midi
from fretmap.synth import render_sequence


def test_gui_module_imports_without_tk():
    assert callable(gui.main)
    assert callable(gui.App)


def test_build_report(tmp_path):
    audio = render_sequence([(0.0, [name_to_midi("E2")], 0.5)], sr=44100)
    path = tmp_path / "note.wav"
    sf.write(path, audio, 44100)
    report = gui.build_report(str(path))
    assert "E2" in report
    assert "Fretboard map" in report
