"""Tkinter GUI: open a guitar track, see the transcription, self-update.

Update checks run on launch, every 6 hours, and on demand via the
"Check for updates" button; when a newer release exists the button turns
into a one-click "Update to vX.Y.Z" that downloads the new executable
and restarts into it.
"""

import queue
import threading

from fretmap import __version__, updater
from fretmap.analyze import analyze_file
from fretmap.fretboard import STANDARD_TUNING
from fretmap.render import render_event_list, render_summary_fretboard, render_tab

RECHECK_MS = 6 * 60 * 60 * 1000  # periodic background update check


def build_report(path: str) -> str:
    events, sr, duration = analyze_file(path, tuning=STANDARD_TUNING)
    n_notes = sum(1 for e in events if e.kind == "note")
    n_int = sum(1 for e in events if e.kind == "interval")
    n_chords = sum(1 for e in events if e.kind == "chord")
    parts = [
        f"{path}\n{duration:.2f}s @ {sr} Hz — {len(events)} events "
        f"({n_notes} notes, {n_int} intervals, {n_chords} chords)",
        render_event_list(events),
        render_tab(events, STANDARD_TUNING),
        render_summary_fretboard(events, STANDARD_TUNING),
    ]
    return "\n\n".join(parts)


class App:
    def __init__(self, root):
        import tkinter as tk
        from tkinter import font as tkfont
        from tkinter import ttk

        self.tk = tk
        self.root = root
        self.update_info = None
        self.results = queue.Queue()

        root.title(f"fretmap {__version__} — guitar track to tab & fretboard")
        root.geometry("900x620")

        bar = ttk.Frame(root, padding=6)
        bar.pack(fill="x")
        self.open_btn = ttk.Button(bar, text="Open audio file…", command=self.open_file)
        self.open_btn.pack(side="left")
        self.save_btn = ttk.Button(bar, text="Save output…", command=self.save_output, state="disabled")
        self.save_btn.pack(side="left", padx=(6, 0))
        self.update_btn = ttk.Button(bar, text="Check for updates", command=self.on_update_clicked)
        self.update_btn.pack(side="right")
        ttk.Label(bar, text=f"v{__version__}").pack(side="right", padx=(0, 8))

        mono = tkfont.nametofont("TkFixedFont").copy()
        mono.configure(size=11)
        self.text = tk.Text(root, font=mono, wrap="none", state="disabled")
        yscroll = ttk.Scrollbar(root, orient="vertical", command=self.text.yview)
        xscroll = ttk.Scrollbar(root, orient="horizontal", command=self.text.xview)
        self.text.configure(yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)
        yscroll.pack(side="right", fill="y")
        xscroll.pack(side="bottom", fill="x")
        self.text.pack(fill="both", expand=True, padx=6)

        self.status = ttk.Label(root, text="Open an isolated guitar track (wav/flac/ogg) to transcribe it.", padding=4)
        self.status.pack(fill="x")

        self.set_text(
            "fretmap — guitar transcription\n\n"
            "1. Click 'Open audio file…' and pick an isolated guitar recording.\n"
            "2. Read the detected notes, chords, tab and fretboard map here.\n"
            "3. 'Save output…' writes the transcription to a text file.\n\n"
            "Updates are checked automatically on launch; use the button in the\n"
            "top-right corner to check or install at any time."
        )

        self.check_updates(manual=False)
        root.after(RECHECK_MS, self.periodic_check)

    # ---------- transcription ----------

    def open_file(self):
        from tkinter import filedialog

        path = filedialog.askopenfilename(
            title="Choose an isolated guitar track",
            filetypes=[
                ("Audio files", "*.wav *.flac *.ogg *.aiff *.aif *.mp3"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return
        self.open_btn.configure(state="disabled")
        self.set_status(f"Analyzing {path} …")
        threading.Thread(target=self._analyze_worker, args=(path,), daemon=True).start()
        self.root.after(100, self._poll_results)

    def _analyze_worker(self, path: str):
        try:
            self.results.put(("report", build_report(path)))
        except Exception as exc:
            self.results.put(("error", f"Could not analyze {path}: {exc}"))

    def _poll_results(self):
        try:
            kind, payload = self.results.get_nowait()
        except queue.Empty:
            self.root.after(100, self._poll_results)
            return
        self.open_btn.configure(state="normal")
        if kind == "report":
            self.set_text(payload)
            self.save_btn.configure(state="normal")
            self.set_status("Done.")
        else:
            self.set_status(payload)

    def save_output(self):
        from tkinter import filedialog

        path = filedialog.asksaveasfilename(
            defaultextension=".txt", filetypes=[("Text files", "*.txt")]
        )
        if not path:
            return
        with open(path, "w") as fh:
            fh.write(self.text.get("1.0", "end-1c") + "\n")
        self.set_status(f"Saved {path}")

    # ---------- updates ----------

    def check_updates(self, manual: bool):
        if manual:
            self.set_status("Checking for updates…")
        threading.Thread(target=self._update_check_worker, args=(manual,), daemon=True).start()

    def _update_check_worker(self, manual: bool):
        info = updater.check_for_update(gui=True)
        self.root.after(0, lambda: self._update_check_done(info, manual))

    def _update_check_done(self, info, manual: bool):
        self.update_info = info
        if info:
            self.update_btn.configure(text=f"Update to v{info.version}")
            self.set_status(
                f"Update available: v{info.version} (you have v{__version__}). "
                "Click the update button to install."
            )
        elif manual:
            self.update_btn.configure(text="Check for updates")
            self.set_status(f"You are up to date (v{__version__}).")

    def periodic_check(self):
        if not self.update_info:
            self.check_updates(manual=False)
        self.root.after(RECHECK_MS, self.periodic_check)

    def on_update_clicked(self):
        if not self.update_info:
            self.check_updates(manual=True)
            return
        info = self.update_info
        if not updater.can_self_update():
            from tkinter import messagebox

            messagebox.showinfo(
                "Update available",
                f"v{info.version} is available, but self-update only works in the "
                f"packaged executable.\n\nDownload it from:\n{info.release_url}",
            )
            return
        self.update_btn.configure(state="disabled")
        self.set_status(f"Downloading v{info.version} …")
        threading.Thread(target=self._update_worker, args=(info,), daemon=True).start()

    def _update_worker(self, info):
        try:
            def progress(done, total):
                if total:
                    pct = 100 * done // total
                    self.root.after(0, lambda: self.set_status(f"Downloading v{info.version} … {pct}%"))

            new_file = updater.download_update(info, progress=progress)
            self.root.after(0, lambda: self.set_status("Restarting into the new version…"))
            updater.apply_and_restart(new_file)
        except Exception as exc:
            self.root.after(0, lambda: self._update_failed(exc, info))

    def _update_failed(self, exc, info):
        self.update_btn.configure(state="normal")
        self.set_status(f"Update failed: {exc} — you can download manually: {info.release_url}")

    # ---------- helpers ----------

    def set_text(self, content: str):
        self.text.configure(state="normal")
        self.text.delete("1.0", "end")
        self.text.insert("1.0", content)
        self.text.configure(state="disabled")

    def set_status(self, msg: str):
        self.status.configure(text=msg)


def main() -> int:
    try:
        import tkinter as tk
    except ImportError:
        print("error: tkinter is not available; use the 'fretmap' CLI instead")
        return 1
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        print(f"error: cannot open a window ({exc}); use the 'fretmap' CLI instead")
        return 1
    App(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
