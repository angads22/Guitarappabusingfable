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
[   0.99s] chord    Am                       notes: A2 E3 C4 E4
[   1.97s] interval G5 (G2+D3)               notes: G2 D3
[   2.47s] note     B3                       notes: B3
[   2.97s] note     C4                       notes: C4
[   3.48s] chord    D                        notes: D3 A3 F#4

   E   Am  G5  B   C   D
e |-----0---------------2--|
B |-----1-------0---1------|
G |-1-------------------2--|
D |-2---2---0-----------0--|
A |-2---0------------------|
E |-0-------3--------------|

Fretboard map of all detected positions:
e o|---|-F#|---|---|
B o|-C-|---|---|---|
G  |-G#|-A-|---|---|
D o|---|-E-|---|---|
A o|---|-B-|---|---|
E o|---|---|-G-|---|
    1   2   3   4
```

## Installation

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
   (E2…fret-24 range) are scored by a **harmonically weighted sum of
   spectral peaks**, where each captured peak is also weighted by how
   exactly it sits on the candidate's harmonic comb (this stops low
   candidates from "vacuuming up" other notes' harmonics). The best
   candidate is taken, refined by parabolic peak interpolation, its
   harmonics are cancelled from the working spectrum, and the search
   repeats — yielding one pitch for a single note and several for a chord.
   A sub-octave guard (odd vs. even harmonic energy) suppresses
   half-pitch errors.
3. **Classification** — 1 pitch → *note*; 2 pitches → *interval* (named,
   e.g. "minor 3rd", with root+fifth labelled as a power chord like `G5`);
   3+ pitch classes → *chord*, identified by matching interval sets
   against templates (maj, min, 7, m7, maj7, sus2/4, dim, aug, 6, add9, …)
   with slash-chord naming when the bass isn't the root.
4. **Fretboard mapping** — every pitch has up to six (string, fret)
   candidates. Single notes pick the position closest to the current hand
   position (open strings are cheap); chords are solved by backtracking
   for a playable shape: distinct strings, fretted span ≤ 4 frets,
   minimal span/height/movement.
5. **Rendering** — ASCII tab (chord labels above the columns), per-event
   fretboard diagrams with note names on the grid, a whole-track fretboard
   map, and JSON for downstream tools.

## Accuracy notes & limitations

- Designed for **isolated** (clean/DI or lightly processed) guitar.
  Heavy distortion smears harmonics and will reduce accuracy.
- **Octave doublings inside a strummed chord** (e.g. A2 and A3 in an open
  Am) share every harmonic, so the doubled note may be absorbed into the
  lower one. Chord names are derived from pitch classes and survive this;
  the tab then shows the essential voicing rather than every doubled string.
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
