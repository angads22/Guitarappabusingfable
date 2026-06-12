# fretmap

A digital-signal-processing app that takes an **isolated guitar track** as
input, determines the **note names** being played and their **locations on
the fretboard**, and maps the result out as **guitar tab** and **fretboard
diagrams**.

For rhythm guitar input it distinguishes **chords** from single **notes**,
and it handles **individual notes or intervals played in between chords**
(e.g. a power-chord stab followed by two melody notes followed by a full
strummed chord).

```text
$ fretmap rhythm.wav

rhythm.wav: 4.70s @ 44100 Hz — 6 events (2 notes, 1 intervals, 3 chords)

[   0.00s] chord    E                        notes: E2 B2 E3 G#3
[   0.99s] chord    Am                       notes: A2 E3 A3 C4 E4
[   1.97s] interval G5 (G2+D3)               notes: G2 D3
[   2.47s] note     B3                       notes: B3
[   2.97s] note     C4                       notes: C4
[   3.48s] chord    D                        notes: D3 A3 D4 F#4

   E     Am    G5  B   C   D
e |-------0-----------------2----|
B |-------1---------0---1---3----|
G |-1-----2-----------------2----|
D |-2-----2-----0-----------0----|
A |-2-----0----------------------|
E |-0-----------3----------------|

Fretboard map of all detected positions:
e o|---|-F#|---|---|
B o|-C-|---|-D-|---|
G  |-G#|-A-|---|---|
D o|---|-E-|---|---|
A o|---|-B-|---|---|
E o|---|---|-G-|---|
    1   2   3   4
```

## Download (no Python needed)

Standalone single-file executables for **Windows, macOS and Linux** are on
the [Releases page](https://github.com/angads22/Guitarappabusingfable/releases/latest).
Two flavours per platform:

- **`fretmap-gui-…`** — windowed app: open a file with a button, read the
  transcription, save it to a text file. It **checks the repo for updates
  on launch** (and every 6 hours); when a new release exists the
  "Check for updates" button becomes **"Update to vX.Y.Z"** and one click
  downloads the new executable and restarts into it.
- **`fretmap-…`** — command-line version: double-click and type a path,
  drag an audio file onto it, or use it from a terminal with all options.
  `fretmap --check-updates` reports whether a newer release exists.

On macOS/Linux make them executable first: `chmod +x fretmap-gui-macos`.
Every push also uploads fresh builds to the workflow run's Artifacts
section; releases (tags `v*`) are what the self-updater follows.

## Installation (from source)

```bash
pip install -e .          # runtime: numpy + soundfile
pip install -e ".[dev]"   # + pytest for development
```

## Usage

```bash
fretmap input.wav                  # default view: events + tab + fretboard map
fretmap input.wav --tab            # ASCII tab only
fretmap input.wav --fretboard      # one fretboard diagram per detected event
fretmap input.wav --summary        # single map of every position used
fretmap input.wav --json           # machine-readable output
fretmap input.wav --tuning D2,A2,D3,G3,B3,E4   # drop-D (any tuning, low→high)
fretmap input.wav --max-fret 21 --max-polyphony 5
fretmap input.wav --lead            # isolate the lead line of a 2-guitar mix
fretmap input.wav -o transcription.txt
```

Any format libsndfile can read works (wav, flac, ogg, ...). Try it without
a recording by generating synthesized demo tracks:

```bash
python examples/make_demo.py
fretmap examples/output/rhythm.wav
```

## How it works (the DSP pipeline)

1. **Onset detection** — an STFT (2048-sample Hann window, 512 hop) is
   computed and note/chord attacks are found as peaks in the positive
   **spectral flux** of the log-magnitude spectrogram, using an adaptive
   local-mean threshold, an RMS gate, and minimum peak separation.
2. **Polyphonic pitch detection** — for each inter-onset segment the pluck
   transient is skipped and a ~200 ms window is analyzed with a
   zero-padded FFT. Fundamental candidates on a quarter-semitone grid
   (spanning the tuning's lowest open string up to its highest fret) are
   scored by a **harmonically weighted sum of spectral peaks**, where each
   captured peak is also weighted by how exactly it sits on the
   candidate's harmonic comb (this stops low candidates from "vacuuming
   up" other notes' harmonics). The best candidate is taken, refined by
   parabolic peak interpolation, its harmonics are cancelled from the
   working spectrum, and the search repeats — yielding one pitch for a
   single note and several for a chord. A sub-octave guard (odd vs. even
   harmonic energy) suppresses half-pitch errors. A **global tuning
   offset** (circular mean of every note's deviation from equal
   temperament) is then estimated for the whole track, so a uniformly
   detuned guitar still rounds to consistent semitones. Finally, notes
   that are only still **ringing** from the previous event — no energy
   rise in their harmonic bands at the onset — are dropped instead of
   being re-reported as new notes.
   Three refinements make voicings come out right on real strings:
   a **stiff-string stretch** is fitted per note (real partials sit at
   h·f0·√(1+Bh²), progressively sharp) and the comb evaluated against the
   stretched positions; a **hidden-note recovery pass** re-examines the
   uncancelled spectrum for octave/twelfth/double-octave doublings whose
   every partial coincides with a lower note's comb (e.g. the A3 inside an
   open Am, or the top E4 of an open Em7) — they are accepted only when
   the energy at the hidden note's partials exceeds the lower note's own
   interpolated envelope on several harmonics at once, with a decay-rate
   veto so a pickup/body resonance can't fake a note; and a time-domain
   **YIN** estimate arbitrates octave errors on single notes.
3. **Classification** — 1 pitch → *note*; 2 pitches → *interval* (named,
   e.g. "minor 3rd", with root+fifth labelled as a power chord like `G5`);
   3+ pitch classes → *chord*, identified by tolerant template matching:
   each template declares required and optional tones (the perfect 5th is
   optional on most qualities), so shell voicings get their real name —
   E2+D3+G3 is `Em7 (no 5th)`, not an anonymous note cluster. Templates
   cover maj, min, 7, m7, maj7, sus2/4, dim, aug, 6, m6, dim7, m7b5,
   add9/add11, 9, m9, maj9, 7#9, 7b9, 11, m11, 13 and 6/9, with
   slash-chord naming when the bass isn't the root.
4. **Fretboard mapping** — every pitch has up to six (string, fret)
   candidates. Single notes pick the position closest to the current hand
   position (open strings are cheap); chords are solved by backtracking
   for a playable shape: distinct strings, fretted span ≤ 4 frets,
   minimal span/height/movement. A built-in **catalogue of real guitar
   shapes** (open chords, barres, shell voicings, triads, drone shapes —
   each transposable up the neck) competes with the backtracker, so known
   voicings come out fingered the way a guitarist plays them. The
   catalogue also performs **voicing completion**: when a near-full strum
   is exactly one note short of exactly one known shape and the missing
   note is an octave doubling sitting on two detected notes' harmonics at
   once (spectrally unattributable — e.g. the B3 of an open E chord), it
   is restored, marked as inferred and shown in parentheses in the tab.
5. **Rendering** — ASCII tab (chord labels above the columns, column
   spacing proportional to the time until the next onset), per-event
   fretboard diagrams with note names on the grid, a whole-track fretboard
   map, and JSON for downstream tools.

## Accuracy notes & limitations

- Designed for **isolated** (clean/DI or lightly processed) guitar.
  Heavy distortion smears harmonics and will reduce accuracy.
- **Octave doublings inside a strummed chord** (e.g. A2 and A3 in an open
  Am) share every harmonic with the lower note; the recovery pass brings
  them back when the spectral evidence is attributable, so full open-chord
  voicings render with all their strings. The B3 inside an open E-shape
  chord is unrecoverable from the spectrum in principle (its fundamental
  sits 2 cents from E2's 3rd harmonic *and* on B2's 2nd) — the chord-shape
  catalogue restores it when the other five strings match a known shape,
  shown parenthesized in the tab (e.g. `(0)`) because it is inferred from
  fretboard knowledge rather than detected: the player may genuinely have
  muted that string. One honest gap remains: a doubling more than ~12 dB
  quieter than the rest of the strum may be missed.
- `--lead` isolates the dominant melodic line by register tracking; it is
  a heuristic for two-guitar mixes, not full source separation.
- Fretboard positions are inherently ambiguous on a guitar (the same pitch
  exists in up to six places); the mapper picks an ergonomic choice, which
  may differ from what was actually fingered.
- Events are onset-based: hammer-ons/slides without a clear attack merge
  into the preceding event.

## Development

```bash
python -m pytest tests   # unit + end-to-end tests on synthesized plucks
```

The test suite synthesizes plucked-string audio (`fretmap/synth.py`) with
known ground truth and checks the whole pipeline: onsets, multi-pitch,
chord/interval naming, fretboard assignment, and rendering.
