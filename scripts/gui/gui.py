"""Desktop GUI (Tkinter) for browsing the data, running steps and viewing results.

A thin Tkinter front end over src/nsclc_pipeline/*.py -- every button here
just calls the same functions the CLI scripts call, so anything you can do
by hand with `python -m nsclc_pipeline.<module>` you can do by
clicking here instead. Long-running work (network fetches, image
downloads) runs on a background thread so the window never freezes; each
module's print() output is redirected into the on-screen log.

Run with:
    python scripts/gui/gui.py
"""

from __future__ import annotations

import os
import queue
import sys
import threading
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk

import matplotlib

matplotlib.use("TkAgg")

# Publication-style rcParams (Arial/Helvetica, compact type, thin lines, no top/right spines), set once
# so every embedded chart and exported figure shares them; defined in scripts/figures/palette.py.
from scripts.figures import palette  # noqa: E402

palette.apply_style()

import mplcursors
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import SimpleITK as sitk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from PIL import ImageTk
from statannotations.Annotator import Annotator

from nsclc_pipeline import (
    aim_annotations,
    clinical,
    config,
    download,
    embeddings,
    interpretation,
    modeling,
    perturbation,
    radiomics,
    rna_seq,
    seg_to_mask,
    series_index,
    survival,
    viz,
)


class QueueWriter:
    """A file-like object that pushes print() output into a queue instead
    of stdout, so background-thread prints show up in the GUI log."""

    def __init__(self, q: queue.Queue):
        self.q = q

    def write(self, text):
        if text.strip():
            self.q.put(("log", text.rstrip("\n")))

    def flush(self):
        pass


class PipelineGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("NSCLC Radiogenomics Pipeline")
        self.geometry("780x580")
        self.minsize(660, 480)

        self.queue: queue.Queue = queue.Queue()
        self.busy = False
        self.image_viewer_dialog = None

        self._build_widgets()
        self.after(100, self._poll_queue)

    # UI

    def _build_widgets(self):
        pad = {"padx": 8, "pady": 6}

        actions = ttk.LabelFrame(self, text="Pipeline stages")
        actions.pack(fill="x", **pad)

        self.btn_metadata = ttk.Button(
            actions, text="Build metadata tables", command=self.run_build_metadata
        )
        self.btn_rnaseq = ttk.Button(actions, text="Fetch RNA-seq", command=self.run_fetch_rna_seq)
        self.btn_download = ttk.Button(actions, text="Download images...", command=self.open_download_dialog)
        self.btn_convert = ttk.Button(
            actions, text="Convert SEG -> mask...", command=self.open_convert_dialog
        )
        self.btn_radiomics = ttk.Button(
            actions, text="Extract radiomics...", command=self.open_radiomics_dialog
        )
        self.btn_modeling = ttk.Button(
            actions, text="Run modeling...", command=self.open_modeling_dialog
        )
        self.btn_perturbation = ttk.Button(
            actions, text="Segmentation adjustment study...", command=self.open_perturbation_dialog
        )
        self.btn_survival = ttk.Button(
            actions, text="Run survival model...", command=self.open_survival_dialog
        )
        self.btn_embeddings = ttk.Button(
            actions, text="Extract embeddings...", command=self.open_embeddings_dialog
        )

        self.action_buttons = [
            self.btn_metadata,
            self.btn_rnaseq,
            self.btn_download,
            self.btn_convert,
            self.btn_radiomics,
            self.btn_modeling,
            self.btn_perturbation,
            self.btn_survival,
            self.btn_embeddings,
        ]
        for i, b in enumerate(self.action_buttons):
            row, col = divmod(i, 3)
            b.grid(row=row, column=col, padx=6, pady=6, sticky="ew")
        for c in range(3):
            actions.grid_columnconfigure(c, weight=1)

        status_frame = ttk.Frame(self)
        status_frame.pack(fill="x", **pad)
        self.status_var = tk.StringVar(value="Ready.")
        ttk.Label(status_frame, textvariable=self.status_var).pack(side="left")

        self.progress = ttk.Progressbar(self, mode="determinate", maximum=100)
        self.progress.pack(fill="x", padx=8, pady=(0, 6))

        # packed before the log so the buttons stay visible when the window is small
        bottom = ttk.Frame(self)
        bottom.pack(side="bottom", fill="x", **pad)

        log_frame = ttk.LabelFrame(self, text="Log")
        log_frame.pack(fill="both", expand=True, **pad)
        self.log_text = scrolledtext.ScrolledText(log_frame, state="disabled", wrap="word", height=18)
        self.log_text.pack(fill="both", expand=True)

        ttk.Button(bottom, text="Open output folder", command=self.open_output_folder).pack(
            side="left", padx=4
        )
        ttk.Button(bottom, text="View a table...", command=self.view_table).pack(side="left", padx=4)
        ttk.Button(bottom, text="View images...", command=self.open_image_viewer).pack(side="left", padx=4)
        ttk.Button(bottom, text="Dashboard...", command=self.open_dashboard).pack(side="left", padx=4)

    # log / progress glue

    def log(self, text: str):
        self.log_text.configure(state="normal")
        ts = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert("end", f"[{ts}] {text}\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _poll_queue(self):
        try:
            while True:
                kind, payload = self.queue.get_nowait()
                if kind == "log":
                    self.log(payload)
                elif kind == "progress":
                    self.progress["value"] = payload
                elif kind == "status":
                    self.status_var.set(payload)
                elif kind == "done":
                    self._finish(payload or "Done.")
                elif kind == "error":
                    self._finish("Error.")
                    messagebox.showerror("Pipeline error", payload)
                elif kind == "view_ready":
                    if self.image_viewer_dialog is not None and self.image_viewer_dialog.winfo_exists():
                        self.image_viewer_dialog.show_volume(*payload)
        except queue.Empty:
            pass
        self.after(100, self._poll_queue)

    def _finish(self, status_text: str):
        self._set_busy(False)
        self.status_var.set(status_text)
        self.progress.stop()
        self.progress.configure(mode="determinate")
        self.progress["value"] = 100

    def _set_busy(self, busy: bool):
        self.busy = busy
        state = "disabled" if busy else "normal"
        for b in self.action_buttons:
            b.configure(state=state)

    def run_in_thread(self, target, *, indeterminate: bool = False):
        """Run `target` (a zero-arg callable) on a background thread,
        disabling the action buttons and wiring its print() output + any
        exception into the log/status/progress widgets."""
        if self.busy:
            messagebox.showinfo("Busy", "Another operation is already running -- wait for it to finish.")
            return
        self._set_busy(True)
        self.progress["value"] = 0
        if indeterminate:
            self.progress.configure(mode="indeterminate")
            self.progress.start(12)
        else:
            self.progress.configure(mode="determinate")

        def worker():
            old_stdout = sys.stdout
            sys.stdout = QueueWriter(self.queue)
            try:
                target()
                self.queue.put(("done", "Done."))
            except Exception as e:  # noqa: BLE001 -- surface any pipeline error in the GUI, don't crash it
                self.queue.put(("error", f"{type(e).__name__}: {e}"))
            finally:
                sys.stdout = old_stdout

        threading.Thread(target=worker, daemon=True).start()

    # actions

    def run_build_metadata(self):
        def task():
            print("Building clinical table...")
            clin = clinical.build()
            self.queue.put(("progress", 33))
            print(f"clinical.csv: {len(clin)} patients")

            print("Building series index...")
            idx = series_index.build()
            self.queue.put(("progress", 66))
            print(f"series_index.csv: {len(idx)} series, {idx['case_id'].nunique()} patients")

            print("Parsing AIM annotations...")
            annotations, locators = aim_annotations.build()
            self.queue.put(("progress", 100))
            print(
                f"aim_annotations.csv: {len(annotations)} rows, "
                f"aim_locators.csv: {len(locators)} rows, "
                f"{annotations['case_id'].nunique()} patients with AIM files"
            )

        self.run_in_thread(task)

    def run_fetch_rna_seq(self):
        def task():
            print("Fetching + tidying GEO GSE103584 RNA-seq matrix (network fetch, ~7MB)...")
            expr = rna_seq.build()
            print(f"rna_seq_expression.csv.gz: {expr.shape[0]} genes x {expr.shape[1]} patients")

        self.run_in_thread(task, indeterminate=True)

    def open_download_dialog(self):
        if self.busy:
            messagebox.showinfo("Busy", "Another operation is already running -- wait for it to finish.")
            return
        if not config.SERIES_INDEX_OUT.exists():
            messagebox.showwarning("Series index missing", "Run 'Build metadata tables' first.")
            return
        DownloadDialog(self)

    def open_convert_dialog(self):
        if self.busy:
            messagebox.showinfo("Busy", "Another operation is already running -- wait for it to finish.")
            return
        ConvertDialog(self)

    def open_radiomics_dialog(self):
        if self.busy:
            messagebox.showinfo("Busy", "Another operation is already running -- wait for it to finish.")
            return
        if not config.SERIES_INDEX_OUT.exists():
            messagebox.showwarning("Series index missing", "Run 'Build metadata tables' first.")
            return
        RadiomicsDialog(self)

    def open_modeling_dialog(self):
        if self.busy:
            messagebox.showinfo("Busy", "Another operation is already running -- wait for it to finish.")
            return
        if not config.RADIOMICS_OUT.exists():
            messagebox.showwarning("No radiomics features yet", "Run 'Extract radiomics...' first.")
            return
        ModelingDialog(self)

    def open_perturbation_dialog(self):
        if self.busy:
            messagebox.showinfo("Busy", "Another operation is already running -- wait for it to finish.")
            return
        if not config.RADIOMICS_OUT.exists():
            messagebox.showwarning("No radiomics features yet", "Run 'Extract radiomics...' first.")
            return
        PerturbationDialog(self)

    def open_survival_dialog(self):
        if self.busy:
            messagebox.showinfo("Busy", "Another operation is already running -- wait for it to finish.")
            return
        if not config.RADIOMICS_OUT.exists():
            messagebox.showwarning("No radiomics features yet", "Run 'Extract radiomics...' first.")
            return
        SurvivalDialog(self)

    def open_embeddings_dialog(self):
        if self.busy:
            messagebox.showinfo("Busy", "Another operation is already running -- wait for it to finish.")
            return
        if not config.SERIES_INDEX_OUT.exists():
            messagebox.showwarning("Series index missing", "Run 'Build metadata tables' first.")
            return
        EmbeddingsDialog(self)

    def open_dashboard(self):
        if not config.CLINICAL_OUT.exists():
            messagebox.showwarning("No data yet", "Run 'Build metadata tables' first.")
            return
        DashboardDialog(self)

    def open_image_viewer(self):
        if self.busy:
            messagebox.showinfo("Busy", "Another operation is already running -- wait for it to finish.")
            return
        if not config.SERIES_INDEX_OUT.exists():
            messagebox.showwarning("Series index missing", "Run 'Build metadata tables' first.")
            return
        ImageViewerDialog(self)

    def open_output_folder(self):
        config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        os.startfile(config.PROCESSED_DIR)  # noqa: S606 -- Windows-only tool, by design

    def view_table(self):
        path = filedialog.askopenfilename(
            initialdir=config.PROCESSED_DIR,
            title="Choose a processed table",
            filetypes=[("CSV / CSV.GZ", "*.csv *.csv.gz"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            df = pd.read_csv(path)
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("Could not read file", str(e))
            return
        TablePreview(self, Path(path).name, df)


class DownloadDialog(tk.Toplevel):
    """Filter series_index.csv by modality + a validated multi-select list
    of case IDs -- not free-typed comma-separated text, which silently
    matches zero series on a typo with no clear signal why. Same fix as
    ConvertDialog: pick from what actually exists instead of typing it."""

    MODALITIES = ["ALL", "CT", "PT", "SEG"]

    def __init__(self, app: PipelineGUI):
        super().__init__(app)
        self.app = app
        self.title("Download images")
        self.resizable(False, False)
        self.grab_set()

        self.series_index = pd.read_csv(config.SERIES_INDEX_OUT)

        pad = {"padx": 8, "pady": 6}

        ttk.Label(self, text="Modality:").grid(row=0, column=0, sticky="w", **pad)
        self.modality_var = tk.StringVar(value="SEG")
        ttk.OptionMenu(
            self,
            self.modality_var,
            self.modality_var.get(),
            *self.MODALITIES,
            command=self._refresh_case_list,
        ).grid(row=0, column=1, sticky="ew", **pad)

        ttk.Label(self, text="Filter case IDs:").grid(row=1, column=0, sticky="w", **pad)
        self.filter_var = tk.StringVar()
        filter_entry = ttk.Entry(self, textvariable=self.filter_var, width=20)
        filter_entry.grid(row=1, column=1, sticky="ew", **pad)
        filter_entry.bind("<KeyRelease>", lambda _e: self._refresh_case_list())

        ttk.Label(
            self,
            text="Case ID(s) -- select none for ALL patients,\nctrl/shift-click for multiple:",
        ).grid(row=2, column=0, columnspan=2, sticky="w", padx=8)

        list_frame = ttk.Frame(self)
        list_frame.grid(row=3, column=0, columnspan=2, padx=8, pady=4, sticky="ew")
        self.case_listbox = tk.Listbox(list_frame, selectmode="extended", height=8, exportselection=False)
        vsb = ttk.Scrollbar(list_frame, orient="vertical", command=self.case_listbox.yview)
        self.case_listbox.configure(yscrollcommand=vsb.set)
        self.case_listbox.pack(side="left", fill="both", expand=True)
        vsb.pack(side="left", fill="y")

        sel_btns = ttk.Frame(self)
        sel_btns.grid(row=4, column=0, columnspan=2)
        ttk.Button(sel_btns, text="Select all shown", command=self._select_all).pack(side="left", padx=4)
        ttk.Button(
            sel_btns, text="Clear selection", command=lambda: self.case_listbox.selection_clear(0, "end")
        ).pack(side="left", padx=4)

        ttk.Label(self, text="Limit (blank = no limit):").grid(row=5, column=0, sticky="w", **pad)
        self.limit_var = tk.StringVar()
        ttk.Entry(self, textvariable=self.limit_var, width=10).grid(row=5, column=1, sticky="w", **pad)

        ttk.Button(self, text="Preview", command=self.preview).grid(row=6, column=0, **pad)
        self.preview_var = tk.StringVar(value="Click Preview to see how many series / how large.")
        ttk.Label(self, textvariable=self.preview_var, wraplength=320, justify="left").grid(
            row=6, column=1, sticky="w", **pad
        )

        btns = ttk.Frame(self)
        btns.grid(row=7, column=0, columnspan=2, pady=(4, 10))
        self.start_btn = ttk.Button(btns, text="Start download", command=self.start, state="disabled")
        self.start_btn.pack(side="left", padx=6)
        ttk.Button(btns, text="Cancel", command=self.destroy).pack(side="left", padx=6)

        self._refresh_case_list()

    def _current_case_ids_for_modality(self):
        modality = None if self.modality_var.get() == "ALL" else self.modality_var.get()
        df = self.series_index
        if modality:
            df = df[df["modality"] == modality]
        return sorted(df["case_id"].unique())

    def _refresh_case_list(self, *_args):
        all_ids = self._current_case_ids_for_modality()
        needle = self.filter_var.get().strip().lower()
        shown = [c for c in all_ids if needle in c.lower()] if needle else all_ids
        previously_selected = {self.case_listbox.get(i) for i in self.case_listbox.curselection()}
        self.case_listbox.delete(0, "end")
        for cid in shown:
            self.case_listbox.insert("end", cid)
        for i, cid in enumerate(shown):
            if cid in previously_selected:
                self.case_listbox.selection_set(i)

    def _select_all(self):
        self.case_listbox.selection_set(0, "end")

    def _selected_case_ids(self):
        return [self.case_listbox.get(i) for i in self.case_listbox.curselection()] or None

    def _parse_filters(self):
        modality = None if self.modality_var.get() == "ALL" else self.modality_var.get()
        case_ids = self._selected_case_ids()
        raw_limit = self.limit_var.get().strip()
        limit = int(raw_limit) if raw_limit else None
        return modality, case_ids, limit

    def preview(self):
        modality, case_ids, limit = self._parse_filters()
        df = download.filter_series(modality=modality, case_ids=case_ids, limit=limit)
        n = len(df)
        size_gb = df["file_size_bytes"].sum() / 1e9 if "file_size_bytes" in df.columns else float("nan")
        self.preview_var.set(f"{n} series, ~{size_gb:.2f} GB")
        self.start_btn.configure(state="normal" if n > 0 else "disabled")
        if n > 300:
            messagebox.showwarning(
                "Large download",
                f"This will pull {n} series (~{size_gb:.1f} GB). That's a lot -- consider "
                f"narrowing by case ID or modality first.",
            )

    def start(self):
        modality, case_ids, limit = self._parse_filters()
        self.destroy()

        def task():
            def progress_cb(i, total, row):
                self.app.queue.put(("progress", int(100 * i / total)))
                self.app.queue.put(
                    ("status", f"Downloading {i}/{total}: {row['case_id']} / {row['modality']}")
                )

            paths = download.download_by_filter(
                modality=modality, case_ids=case_ids, limit=limit, progress_cb=progress_cb
            )
            print(f"Downloaded {len(paths)} series -> {config.DICOM_DIR}")

        self.app.run_in_thread(task)


class ConvertDialog(tk.Toplevel):
    """Pick a case ID from the list of patients that actually HAVE a SEG
    series (not every patient does -- only 144/211), then convert it.

    The case ID -> SEG series UID -> download folder mapping is resolved
    automatically from series_index.csv, and downloaded on demand if it
    isn't already on disk. Earlier versions let you type any case ID next
    to a manually-browsed folder with no cross-check, which made it easy
    to save one patient's mask under a different patient's name -- this
    dropdown makes that class of mistake structurally impossible."""

    def __init__(self, app: PipelineGUI):
        super().__init__(app)
        self.app = app
        self.title("Convert SEG -> mask")
        self.resizable(False, False)
        self.grab_set()

        pad = {"padx": 8, "pady": 6}

        if not config.SERIES_INDEX_OUT.exists():
            ttk.Label(self, text="Run 'Build metadata tables' first.").grid(row=0, column=0, **pad)
            ttk.Button(self, text="Close", command=self.destroy).grid(row=1, column=0, **pad)
            return

        seg = pd.read_csv(config.SERIES_INDEX_OUT)
        self.seg = seg[seg["modality"] == "SEG"].sort_values("case_id").reset_index(drop=True)
        case_ids = self.seg["case_id"].tolist()

        ttk.Label(self, text=f"Case ID ({len(case_ids)} patients have a SEG series):").grid(
            row=0, column=0, columnspan=2, sticky="w", **pad
        )
        self.case_id_var = tk.StringVar()
        self.combo = ttk.Combobox(
            self, textvariable=self.case_id_var, values=case_ids, state="readonly", width=20
        )
        self.combo.grid(row=1, column=0, sticky="w", **pad)
        self.combo.bind("<<ComboboxSelected>>", self._on_select)

        self.status_var = tk.StringVar(value="Select a case ID above.")
        ttk.Label(self, textvariable=self.status_var, wraplength=320, justify="left").grid(
            row=1, column=1, sticky="w", **pad
        )

        self.batch_var = tk.BooleanVar(value=False)
        already = sum(
            (config.MASKS_DIR / f"{cid}_seg.nii.gz").exists() for cid in case_ids
        )
        ttk.Checkbutton(
            self,
            text=f"Convert ALL {len(case_ids)} available patients "
            f"({already} already done, will be skipped)",
            variable=self.batch_var,
            command=self._on_batch_toggle,
        ).grid(row=2, column=0, columnspan=2, sticky="w", **pad)

        btns = ttk.Frame(self)
        btns.grid(row=3, column=0, columnspan=2, pady=(4, 10))
        ttk.Button(btns, text="Convert", command=self.start).pack(side="left", padx=6)
        ttk.Button(btns, text="Cancel", command=self.destroy).pack(side="left", padx=6)

    def _on_batch_toggle(self):
        if self.batch_var.get():
            self.combo.configure(state="disabled")
            self.status_var.set("Will convert every patient with a SEG series, downloading any missing ones.")
        else:
            self.combo.configure(state="readonly")
            self.status_var.set("Select a case ID above.")

    def _series_uid_for(self, case_id: str) -> str:
        return self.seg.loc[self.seg["case_id"] == case_id, "series_uid"].iloc[0]

    def _on_select(self, _event=None):
        case_id = self.case_id_var.get()
        series_uid = self._series_uid_for(case_id)
        seg_dir = config.DICOM_DIR / series_uid
        downloaded = seg_dir.exists() and any(seg_dir.iterdir())
        self.status_var.set(
            f"Series {series_uid[:24]}...\n"
            + ("Already downloaded -- will convert directly." if downloaded else "Not downloaded yet -- will download first (small, this is a SEG file).")
        )

    def start(self):
        if self.batch_var.get():
            self._start_batch()
            return

        case_id = self.case_id_var.get().strip()
        if not case_id:
            messagebox.showwarning("Missing input", "Pick a case ID first.")
            return
        series_uid = self._series_uid_for(case_id)
        self.destroy()

        def task():
            print(f"Resolved {case_id} -> SEG series {series_uid}")
            seg_dir = download.download_series(series_uid)
            print(f"SEG DICOM in {seg_dir}")
            out_path = seg_to_mask.build_one(case_id, seg_dir)
            print(f"Wrote {out_path}")

        self.app.run_in_thread(task, indeterminate=True)

    def _start_batch(self):
        rows = list(self.seg.itertuples())
        total = len(rows)
        self.destroy()

        def task():
            written, skipped, failed = 0, 0, 0
            for i, row in enumerate(rows, start=1):
                out_path = config.MASKS_DIR / f"{row.case_id}_seg.nii.gz"
                self.app.queue.put(("progress", int(100 * i / total)))
                self.app.queue.put(("status", f"{i}/{total}: {row.case_id}"))
                if out_path.exists():
                    skipped += 1
                    continue
                try:
                    seg_dir = download.download_series(row.series_uid)
                    seg_to_mask.build_one(row.case_id, seg_dir)
                    written += 1
                except Exception as e:  # noqa: BLE001 -- keep going, report at the end
                    print(f"FAILED {row.case_id}: {type(e).__name__}: {e}")
                    failed += 1
            print(f"Batch done: {written} written, {skipped} already existed, {failed} failed.")

        self.app.run_in_thread(task)


class RadiomicsDialog(tk.Toplevel):
    """Pick a case ID (or all of them) to extract shape/intensity/texture
    features for. Downloads whatever CT/SEG is missing automatically --
    same resolve-the-right-series-then-fetch-if-needed pattern as Convert
    and the image viewer. Results are saved incrementally to
    radiomics_features.csv so a long batch run can be resumed if
    interrupted (already-done case IDs are skipped on a re-run)."""

    def __init__(self, app: PipelineGUI):
        super().__init__(app)
        self.app = app
        self.title("Extract radiomics")
        self.resizable(False, False)
        self.grab_set()

        pad = {"padx": 8, "pady": 6}

        seg = pd.read_csv(config.SERIES_INDEX_OUT)
        self.case_ids = sorted(seg[seg["modality"] == "SEG"]["case_id"].unique())
        already = 0
        if config.RADIOMICS_OUT.exists():
            already = pd.read_csv(config.RADIOMICS_OUT)["case_id"].nunique()

        ttk.Label(self, text=f"Case ID ({len(self.case_ids)} patients have a SEG series):").grid(
            row=0, column=0, columnspan=2, sticky="w", **pad
        )
        self.case_id_var = tk.StringVar()
        self.combo = ttk.Combobox(
            self, textvariable=self.case_id_var, values=self.case_ids, state="readonly", width=20
        )
        self.combo.grid(row=1, column=0, sticky="w", **pad)

        self.batch_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            self,
            text=f"Extract ALL {len(self.case_ids)} patients "
            f"({already} already in radiomics_features.csv, will be skipped)",
            variable=self.batch_var,
            command=self._on_batch_toggle,
        ).grid(row=2, column=0, columnspan=2, sticky="w", **pad)

        note = (
            "Each new patient may need its CT series downloaded "
            "(usually 50-300 MB) if it is not already present."
        )
        ttk.Label(self, text=note, wraplength=380, justify="left").grid(
            row=3, column=0, columnspan=2, sticky="w", padx=8
        )

        btns = ttk.Frame(self)
        btns.grid(row=4, column=0, columnspan=2, pady=(8, 10))
        ttk.Button(btns, text="Extract", command=self.start).pack(side="left", padx=6)
        ttk.Button(btns, text="Cancel", command=self.destroy).pack(side="left", padx=6)

    def _on_batch_toggle(self):
        self.combo.configure(state="disabled" if self.batch_var.get() else "readonly")

    def start(self):
        batch = self.batch_var.get()
        case_id = self.case_id_var.get().strip()
        if not batch and not case_id:
            messagebox.showwarning("Missing input", "Pick a case ID, or check Extract ALL.")
            return
        targets = None if batch else [case_id]
        self.destroy()

        def task():
            def progress_cb(i, total, cid):
                self.app.queue.put(("progress", int(100 * i / total)))
                self.app.queue.put(("status", f"Radiomics {i}/{total}: {cid}"))

            df = radiomics.build(case_ids=targets, progress_cb=progress_cb)
            print(f"radiomics_features.csv now has {len(df)} patients -> {config.RADIOMICS_OUT}")

        self.app.run_in_thread(task)


class EmbeddingsDialog(tk.Toplevel):
    """Pick a case ID (or all of them) to extract ImageNet-pretrained
    ResNet18 embeddings for -- see embeddings.py's docstring for exactly
    what this is (and isn't: not a specialized medical-imaging foundation
    model). Same CT/SEG auto-download and incremental-save/resume pattern
    as Extract radiomics."""

    def __init__(self, app: PipelineGUI):
        super().__init__(app)
        self.app = app
        self.title("Extract embeddings")
        self.resizable(False, False)
        self.grab_set()

        pad = {"padx": 8, "pady": 6}

        seg = pd.read_csv(config.SERIES_INDEX_OUT)
        self.case_ids = sorted(seg[seg["modality"] == "SEG"]["case_id"].unique())
        embeddings_path = config.PROCESSED_DIR / "embeddings.csv"
        already = pd.read_csv(embeddings_path)["case_id"].nunique() if embeddings_path.exists() else 0

        note = (
            "ImageNet-pretrained ResNet18 applied to the tumor's central "
            "axial slice (512-dim embedding): a transfer-learning "
            "baseline, not a 3D medical-imaging foundation model. "
            "The embeddings are compared with radiomic features "
            "using the same nested cross-validation."
        )
        ttk.Label(self, text=note, wraplength=380, justify="left").grid(
            row=0, column=0, columnspan=2, sticky="w", **pad
        )

        ttk.Label(self, text=f"Case ID ({len(self.case_ids)} patients have a SEG series):").grid(
            row=1, column=0, columnspan=2, sticky="w", **pad
        )
        self.case_id_var = tk.StringVar()
        self.combo = ttk.Combobox(
            self, textvariable=self.case_id_var, values=self.case_ids, state="readonly", width=20
        )
        self.combo.grid(row=2, column=0, sticky="w", **pad)

        self.batch_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            self,
            text=f"Extract ALL {len(self.case_ids)} patients "
            f"({already} already in embeddings.csv, will be skipped)",
            variable=self.batch_var,
            command=self._on_batch_toggle,
        ).grid(row=3, column=0, columnspan=2, sticky="w", **pad)

        btns = ttk.Frame(self)
        btns.grid(row=4, column=0, columnspan=2, pady=(8, 10))
        ttk.Button(btns, text="Extract", command=self.start).pack(side="left", padx=6)
        ttk.Button(btns, text="Cancel", command=self.destroy).pack(side="left", padx=6)

    def _on_batch_toggle(self):
        self.combo.configure(state="disabled" if self.batch_var.get() else "readonly")

    def start(self):
        batch = self.batch_var.get()
        case_id = self.case_id_var.get().strip()
        if not batch and not case_id:
            messagebox.showwarning("Missing input", "Pick a case ID, or check Extract ALL.")
            return
        targets = None if batch else [case_id]
        self.destroy()

        def task():
            def progress_cb(i, total, cid):
                self.app.queue.put(("progress", int(100 * i / total)))
                self.app.queue.put(("status", f"Embeddings {i}/{total}: {cid}"))

            df = embeddings.extract_all(case_ids=targets, progress_cb=progress_cb)
            print(f"embeddings.csv now has {len(df)} patients -> {config.PROCESSED_DIR / 'embeddings.csv'}")

        self.app.run_in_thread(task)


class ModelingDialog(tk.Toplevel):
    """Run nested cross-validation (see modeling.py's docstring for why
    nested, not plain k-fold) to estimate how well the radiomics features
    predict a chosen clinical/genomic target. Results print fold-by-fold
    to the main log, same as everything else."""

    def __init__(self, app: PipelineGUI):
        super().__init__(app)
        self.app = app
        self.title("Run modeling")
        self.resizable(False, False)
        self.grab_set()

        pad = {"padx": 8, "pady": 6}

        clin = pd.read_csv(config.CLINICAL_OUT)
        rad_case_ids = set(pd.read_csv(config.RADIOMICS_OUT)["case_id"])
        clin = clin[clin["case_id"].isin(rad_case_ids)]

        ttk.Label(self, text="Target:").grid(row=0, column=0, sticky="w", **pad)
        self.target_var = tk.StringVar(value="recurrence")
        target_combo = ttk.Combobox(
            self, textvariable=self.target_var, values=list(modeling.TARGETS), state="readonly", width=16
        )
        target_combo.grid(row=0, column=1, sticky="w", **pad)
        target_combo.bind("<<ComboboxSelected>>", lambda _e: self._update_counts())

        ttk.Label(self, text="Feature set:").grid(row=1, column=0, sticky="w", **pad)
        self.feature_set_var = tk.StringVar(value="radiomics")
        fs_combo = ttk.Combobox(
            self, textvariable=self.feature_set_var, values=list(modeling.FEATURE_SETS),
            state="readonly", width=24,
        )
        fs_combo.grid(row=1, column=1, sticky="w", **pad)
        fs_combo.bind("<<ComboboxSelected>>", lambda _e: self._update_counts())

        self.counts_var = tk.StringVar()
        ttk.Label(self, textvariable=self.counts_var, wraplength=320, justify="left").grid(
            row=2, column=0, columnspan=2, sticky="w", padx=8
        )
        self._clin = clin
        self._rad_case_ids = rad_case_ids
        self._rna_case_ids = (
            set(pd.read_csv(config.RNA_SEQ_OUT, nrows=0).columns) if config.RNA_SEQ_OUT.exists() else set()
        )
        embeddings_path = config.PROCESSED_DIR / "embeddings.csv"
        self._embed_case_ids = (
            set(pd.read_csv(embeddings_path, usecols=["case_id"])["case_id"]) if embeddings_path.exists() else set()
        )
        self._update_counts()

        ttk.Label(self, text="Outer folds:").grid(row=3, column=0, sticky="w", **pad)
        self.outer_var = tk.IntVar(value=5)
        ttk.Spinbox(self, from_=3, to=10, textvariable=self.outer_var, width=5).grid(
            row=3, column=1, sticky="w", **pad
        )

        ttk.Label(self, text="Inner folds:").grid(row=4, column=0, sticky="w", **pad)
        self.inner_var = tk.IntVar(value=5)
        ttk.Spinbox(self, from_=3, to=10, textvariable=self.inner_var, width=5).grid(
            row=4, column=1, sticky="w", **pad
        )

        ttk.Label(self, text="Random seed:").grid(row=5, column=0, sticky="w", **pad)
        self.seed_var = tk.IntVar(value=0)
        ttk.Entry(self, textvariable=self.seed_var, width=7).grid(row=5, column=1, sticky="w", **pad)

        note = (
            "Results are an exploratory signal at this sample size, not a "
            "validated biomarker -- see the log for the per-fold spread. "
            "The \"+rnaseq\" feature set is noticeably slower (PCA refit "
            "inside every fold x hyperparameter combination)."
        )
        ttk.Label(self, text=note, wraplength=340, justify="left").grid(
            row=6, column=0, columnspan=2, sticky="w", padx=8, pady=(0, 6)
        )

        btns = ttk.Frame(self)
        btns.grid(row=7, column=0, columnspan=2, pady=(4, 10))
        ttk.Button(btns, text="Run", command=self.start).pack(side="left", padx=6)
        ttk.Button(btns, text="Cancel", command=self.destroy).pack(side="left", padx=6)

    def _update_counts(self):
        spec = modeling.TARGETS[self.target_var.get()]
        use = modeling.FEATURE_SETS[self.feature_set_var.get()]
        eligible_ids = self._rad_case_ids
        if use["rnaseq"]:
            eligible_ids = eligible_ids & self._rna_case_ids
        if use["embeddings"]:
            eligible_ids = eligible_ids & self._embed_case_ids
        col = self._clin[self._clin["case_id"].isin(eligible_ids)][spec["column"]]
        n = col.notna().sum()
        n_pos = (col == spec["positive"]).sum()
        self.counts_var.set(f"{spec['description']}: n={n}, positive={n_pos} ({n_pos / n:.1%})" if n else "No data")

    def start(self):
        target = self.target_var.get()
        feature_set = self.feature_set_var.get()
        outer_folds = self.outer_var.get()
        inner_folds = self.inner_var.get()
        seed = self.seed_var.get()
        self.destroy()

        def task():
            modeling.run(
                target=target, feature_set=feature_set,
                outer_folds=outer_folds, inner_folds=inner_folds, random_state=seed,
            )

        self.app.run_in_thread(task, indeterminate=True)


class ImageViewerDialog(tk.Toplevel):
    """Browse CT slices with the tumor mask overlaid, for any case that
    already has a mask built via Convert SEG -> mask. Resolves the exact
    CT series the mask was drawn on from the SEG object itself (not
    guessed from series_description -- a patient can have several CT
    series and only one matches the mask, see
    seg_to_mask.referenced_ct_series_uid), downloading it on demand."""

    def __init__(self, app: PipelineGUI):
        super().__init__(app)
        self.app = app
        self.app.image_viewer_dialog = self
        self.title("View images")
        self.resizable(False, False)

        self.seg = pd.read_csv(config.SERIES_INDEX_OUT)
        self.seg = self.seg[self.seg["modality"] == "SEG"]
        mask_ids = sorted(
            p.name.removesuffix("_seg.nii.gz") for p in config.MASKS_DIR.glob("*_seg.nii.gz")
        )

        pad = {"padx": 8, "pady": 6}

        top = ttk.Frame(self)
        top.pack(fill="x", **pad)
        ttk.Label(top, text="Case ID (has a mask already):").pack(side="left")
        self.case_id_var = tk.StringVar()
        ttk.Combobox(
            top, textvariable=self.case_id_var, values=mask_ids, state="readonly", width=16
        ).pack(side="left", padx=6)
        ttk.Button(top, text="Load", command=self.load_case).pack(side="left", padx=6)

        self.info_var = tk.StringVar(
            value=f"{len(mask_ids)} patients have a mask -- pick one and click Load."
            if mask_ids
            else "No masks found yet -- run Convert SEG -> mask first."
        )
        ttk.Label(self, textvariable=self.info_var, wraplength=560, justify="left").pack(
            anchor="w", padx=8
        )

        self.canvas = tk.Canvas(self, width=512, height=512, bg="black")
        self.canvas.pack(padx=8, pady=6)
        self._tk_img = None  # keep a reference so Tk doesn't garbage-collect it mid-display

        controls = ttk.Frame(self)
        controls.pack(fill="x", padx=8, pady=(0, 10))

        self.slice_scale = ttk.Scale(
            controls, from_=0, to=0, orient="horizontal", command=self._on_slice_change, state="disabled"
        )
        self.slice_scale.pack(fill="x")
        self.slice_label_var = tk.StringVar(value="Slice: -")
        ttk.Label(controls, textvariable=self.slice_label_var).pack(anchor="w")

        opts = ttk.Frame(controls)
        opts.pack(fill="x", pady=(6, 0))
        ttk.Label(opts, text="Window:").pack(side="left")
        self.window_var = tk.StringVar(value="Lung")
        for name in viz.WINDOW_PRESETS:
            ttk.Radiobutton(
                opts, text=name, value=name, variable=self.window_var, command=self._redraw
            ).pack(side="left", padx=4)

        self.overlay_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            opts, text="Show tumor mask", variable=self.overlay_var, command=self._redraw
        ).pack(side="left", padx=12)

        self.ct_array = None
        self.mask_array = None

    def load_case(self):
        case_id = self.case_id_var.get().strip()
        if not case_id:
            messagebox.showwarning("Missing input", "Pick a case ID first.")
            return
        mask_path = config.MASKS_DIR / f"{case_id}_seg.nii.gz"
        matches = self.seg.loc[self.seg["case_id"] == case_id, "series_uid"]
        if matches.empty:
            messagebox.showerror("No SEG series", f"{case_id} has no SEG series in the series index.")
            return
        seg_series_uid = matches.iloc[0]

        def task():
            seg_dir = config.DICOM_DIR / seg_series_uid
            if not (seg_dir.exists() and any(seg_dir.iterdir())):
                print(f"Downloading SEG for {case_id}...")
                seg_dir = download.download_series(seg_series_uid)
            seg_dicom = seg_to_mask.find_seg_dicom(seg_dir)
            ct_series_uid = seg_to_mask.referenced_ct_series_uid(seg_dicom)
            print(f"{case_id}: mask references CT series {ct_series_uid}")

            ct_dir = config.DICOM_DIR / ct_series_uid
            if not (ct_dir.exists() and any(ct_dir.iterdir())):
                print(f"Downloading CT series {ct_series_uid} (typically 50-300MB)...")
                ct_dir = download.download_series(ct_series_uid)

            print("Loading CT volume...")
            ct_image = viz.load_ct_volume(ct_dir)
            print("Resampling mask onto the CT grid...")
            mask_image = viz.load_mask_on_ct(mask_path, ct_image)

            ct_array = sitk.GetArrayFromImage(ct_image)
            mask_array = sitk.GetArrayFromImage(mask_image)
            self.app.queue.put(("view_ready", (case_id, ct_array, mask_array)))

        self.app.run_in_thread(task, indeterminate=True)

    def show_volume(self, case_id, ct_array, mask_array):
        self.ct_array = ct_array
        self.mask_array = mask_array
        n = ct_array.shape[0]
        nz = sorted(set((mask_array > 0).nonzero()[0].tolist()))
        mid = nz[len(nz) // 2] if nz else n // 2

        self.slice_scale.configure(from_=0, to=n - 1, state="normal")
        self.slice_scale.set(mid)
        self.info_var.set(f"{case_id}: {n} slices total, tumor mask present on {len(nz)} of them.")
        self._redraw()

    def _on_slice_change(self, _value):
        self._redraw()

    def _redraw(self):
        if self.ct_array is None:
            return
        z = int(round(self.slice_scale.get()))
        level, width = viz.WINDOW_PRESETS[self.window_var.get()]
        mask = self.mask_array if self.overlay_var.get() else None
        img = viz.render_slice(self.ct_array, z, level, width, mask_array=mask)
        self._tk_img = ImageTk.PhotoImage(img)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self._tk_img)
        total = self.ct_array.shape[0]
        has_tumor = mask is not None and (self.mask_array[z] > 0).any()
        self.slice_label_var.set(f"Slice: {z + 1} / {total}" + ("  [tumor here]" if has_tumor else ""))


_INK = "#0b0b0b"
_INK2 = "#52514e"
_MUTED = "#898781"
_GRID = "#e1e0d9"
# Same colorblind-safe categorical order used elsewhere in this project's
# charts (validated per the dataviz skill's reference palette): blue,
# orange, aqua, amber, magenta -- adjacent pairs and (for the first 3) all
# pairs clear the palette's CVD/normal-vision separation floors.
_CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
_SEQ = "#2a78d6"
# Same sequential blue ramp documented in the dataviz skill's reference
# palette (steps 100->700), for magnitude encodings (e.g. instability
# ranked bars) where a flat categorical fill would be the wrong signal.
_SEQ_CMAP = LinearSegmentedColormap.from_list("nsclc_seq_blue", ["#cde2fb", "#0d366b"])


def _seq_color(t: float) -> tuple:
    """t in [0,1] -> a color along the sequential blue ramp."""
    return _SEQ_CMAP(float(np.clip(t, 0, 1)))

_SLOT_MAP = {
    "Wildtype": 0, "Mutant": 1, "Translocated": 1, "Unknown": 2,
    "no": 0, "yes": 1,
    "Alive": 0, "Dead": 1,
    "Male": 0, "Female": 1,
    "Stanford": 0, "VA": 1,
    "Former": 0, "Nonsmoker": 1, "Current": 2,
    "Adenocarcinoma": 0, "Squamous cell carcinoma": 1,
    "NSCLC NOS (not otherwise specified)": 2,
}


def _slot(label, fallback_index):
    return _SLOT_MAP.get(label, fallback_index % len(_CAT))


def _style_axes(ax):
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(_GRID)
    ax.tick_params(colors=_INK2, labelsize=8)
    ax.set_axisbelow(True)


def _panel_label(ax, letter):
    """Bold lowercase panel letter at the top-left of an axes, outside the
    plot area -- standard convention for multi-panel journal figures
    (Nature/Nature Communications style: 'a', 'b', 'c', ... not 'A/B/C')."""
    ax.text(
        -0.14, 1.12, letter, transform=ax.transAxes, fontsize=12, fontweight="bold",
        va="top", ha="left", color=_INK,
    )


def _hbar(ax, rows, title):
    """rows: list of (label, count) tuples, most-common first."""
    labels = [str(r[0]) for r in rows]
    values = [r[1] for r in rows]
    total = sum(values)
    colors = [_CAT[_slot(lbl, i)] for i, lbl in enumerate(labels)]
    y = np.arange(len(labels))
    bars = ax.barh(y, values, color=colors, height=0.62, zorder=3, edgecolor="white", linewidth=1.1)
    ax.set_yticks(y)
    ax.set_yticklabels([lbl if len(lbl) <= 28 else lbl[:26] + "…" for lbl in labels], fontsize=8.5)
    ax.invert_yaxis()
    ax.set_title(title, fontsize=10, fontweight="bold", loc="left", color=_INK)
    xmax = max(values) if values else 1
    for b, v in zip(bars, values):
        ax.text(
            b.get_width() + xmax * 0.02, b.get_y() + b.get_height() / 2,
            f"{v} ({100 * v / total:.0f}%)", va="center", fontsize=7.5, color=_INK2,
        )
    ax.set_xlim(0, xmax * 1.32)
    ax.grid(axis="x", color=_GRID, linewidth=0.8, zorder=0)
    _style_axes(ax)
    ax.spines["left"].set_visible(False)
    ax.tick_params(left=False)


def _hist(ax, values, title, unit=""):
    values = np.asarray(values, dtype=float)
    sns.histplot(
        values, bins=14, ax=ax, color=_SEQ, edgecolor="white", linewidth=0.7,
        alpha=0.85, kde=True, line_kws={"linewidth": 2, "color": "#0d366b"}, zorder=3,
    )
    mean_v = float(np.mean(values))
    ax.axvline(mean_v, color=_INK2, linewidth=1.1, linestyle="--", zorder=4, alpha=0.8)
    ax.set_title(title, fontsize=10, fontweight="bold", loc="left", color=_INK)
    ax.set_ylabel("")
    if unit:
        ax.set_xlabel(unit, fontsize=8, color=_MUTED)
    else:
        ax.set_xlabel("")
    ax.grid(axis="y", color=_GRID, linewidth=0.8, zorder=0)
    _style_axes(ax)


def _violin_by_group(ax, df, value_col, group_col, group_order, title, ylabel, group_labels=None):
    """Violin plot of value_col split into exactly two groups, annotated
    with a real Mann-Whitney U test via statannotations -- a computed
    result, not a decorative bracket. Shows the distribution's actual
    shape (seaborn violinplot), which a bare histogram collapses away."""
    plot_df = df[[value_col, group_col]].dropna()
    palette = {g: _CAT[i % len(_CAT)] for i, g in enumerate(group_order)}
    sns.violinplot(
        data=plot_df, x=group_col, y=value_col, order=group_order, ax=ax,
        palette=palette, linewidth=1.1, inner="quartile", cut=0, hue=group_col, legend=False,
    )
    for coll in ax.collections:
        coll.set_edgecolor("white")

    pairs = [(group_order[0], group_order[1])]
    annot = Annotator(ax, pairs, data=plot_df, x=group_col, y=value_col, order=group_order)
    annot.configure(test="Mann-Whitney", text_format="star", loc="inside", verbose=0, fontsize=9)
    annot.apply_and_annotate()

    counts = plot_df[group_col].value_counts()
    tick_labels = [
        f"{(group_labels or {}).get(g, g)}\n(n={counts.get(g, 0)})" for g in group_order
    ]
    ax.set_xticks(range(len(group_order)))
    ax.set_xticklabels(tick_labels, fontsize=8.5)
    ax.set_xlabel("")
    ax.set_ylabel(ylabel, fontsize=8.5, color=_MUTED)
    ax.set_title(title, fontsize=10, fontweight="bold", loc="left", color=_INK)
    ax.grid(axis="y", color=_GRID, linewidth=0.8, zorder=0)
    _style_axes(ax)


def _scatter(ax, df):
    order = {"no": 0, "yes": 1}
    colors = [_CAT[order.get(r, 2)] for r in df["recurrence"]]
    sc = ax.scatter(
        df["shape_volume_mm3"], df["texture_contrast"], c=colors, s=42,
        alpha=0.78, edgecolors="white", linewidths=0.8, zorder=3,
    )
    ax.set_xscale("log")
    ax.set_xlabel("Tumor volume (mm³, log scale)", fontsize=8.5, color=_MUTED)
    ax.set_ylabel("GLCM texture contrast", fontsize=8.5, color=_MUTED)
    ax.set_title("Tumor volume vs. texture contrast", fontsize=10, fontweight="bold", loc="left", color=_INK)
    ax.grid(True, color=_GRID, linewidth=0.7, zorder=0, alpha=0.7)
    _style_axes(ax)
    handles = [
        Line2D(
            [0], [0], marker="o", linestyle="", color=_CAT[order.get(k, 2)],
            markeredgecolor="white", markeredgewidth=0.8, markersize=7, label=lbl,
        )
        for k, lbl in [("no", "No recurrence"), ("yes", "Recurrence"), ("unknown", "Unknown")]
    ]
    legend = ax.legend(
        handles=handles, fontsize=8, frameon=True, loc="upper right",
        facecolor="white", edgecolor=_GRID, framealpha=0.92,
    )
    legend.get_frame().set_linewidth(0.6)
    ax.text(
        0.015, 0.03, f"n={len(df)}", transform=ax.transAxes, fontsize=7.5, color=_MUTED,
    )
    return sc


def _roc(ax, spec):
    fpr, tpr = spec["fpr"], spec["tpr"]
    ax.plot([0, 1], [0, 1], color=_MUTED, linewidth=1, linestyle=":", zorder=2)
    ax.fill_between(fpr, tpr, fpr, color=_CAT[0], alpha=0.12, zorder=2, linewidth=0)
    ax.plot(fpr, tpr, color=_CAT[0], linewidth=2.3, zorder=3, solid_capstyle="round")

    # Direct AUC readout on the curve itself, not just the title -- find
    # the point furthest from the diagonal (the curve's most informative
    # operating point) and label it there.
    fpr_a, tpr_a = np.asarray(fpr), np.asarray(tpr)
    j = int(np.argmax(tpr_a - fpr_a))
    ax.scatter([fpr_a[j]], [tpr_a[j]], s=26, color=_CAT[0], zorder=4, edgecolors="white", linewidths=0.8)
    ax.annotate(
        f"AUC = {spec['mean_auc']:.2f}",
        xy=(fpr_a[j], tpr_a[j]), xytext=(fpr_a[j] + 0.22, tpr_a[j] - 0.16),
        fontsize=8.5, fontweight="bold", color=_CAT[0],
        arrowprops={"arrowstyle": "-", "color": _CAT[0], "linewidth": 0.8},
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xlabel("False positive rate", fontsize=8.5, color=_MUTED)
    ax.set_ylabel("True positive rate", fontsize=8.5, color=_MUTED)
    # AUC itself is already shown in-plot via the annotation above; the
    # title only needs to say what this row IS. Including "mean AUC
    # X +/- Y" here too made this the longest string in either panel of
    # a row, and at loc="left" a long title isn't clipped to its own
    # axes -- it overflows rightward into the neighboring fold-strip
    # panel's title, visibly colliding with it in a multi-row figure
    # (Supplementary Fig. S3, one row per target x feature-set).
    ax.set_title(spec["title"], fontsize=10, fontweight="bold", loc="left", color=_INK)
    ax.grid(color=_GRID, linewidth=0.7, zorder=0, alpha=0.7)
    _style_axes(ax)


def _fold_strip(ax, fold_aucs, mean_auc, title):
    ax.axvspan(mean_auc - np.std(fold_aucs), mean_auc + np.std(fold_aucs), color=_CAT[0], alpha=0.08, zorder=1)
    ax.axvline(mean_auc, color=_INK2, linestyle="--", linewidth=1.1, zorder=2)
    # Small deterministic vertical jitter so folds landing at nearly the
    # same AUC (common with only 5 folds) don't render as one indistinguishable dot.
    rng = np.random.RandomState(0)
    jitter = rng.uniform(-0.12, 0.12, size=len(fold_aucs))
    ax.scatter(
        fold_aucs, jitter, s=100, color=_CAT[0], alpha=0.88,
        edgecolors="white", linewidths=1.2, zorder=3,
    )
    for i, (a, j) in enumerate(zip(fold_aucs, jitter), start=1):
        ax.annotate(str(i), (a, j), fontsize=6, color="white", ha="center", va="center", zorder=4, fontweight="bold")
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.35, 0.35)
    ax.set_yticks([])
    ax.set_xlabel("AUC per outer fold (dashed = mean, band = ±1 std)", fontsize=8.5, color=_MUTED)
    ax.set_title(title, fontsize=9, loc="left", color=_INK2)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.spines["bottom"].set_visible(True)
    ax.spines["bottom"].set_color(_GRID)
    ax.tick_params(colors=_INK2, labelsize=8)


_FEATURE_SET_LABELS = {
    "radiomics": "Radiomics",
    "radiomics+clinical": "Radiomics + clinical",
    "radiomics+clinical+rnaseq": "Radiomics + clinical + RNA-seq",
    "embeddings": "CNN embeddings",
    "radiomics+embeddings": "Radiomics + CNN embeddings",
}


def _feature_set_comparison(ax, specs):
    """The headline figure: mean AUC (+/- fold std) grouped by target,
    one bar per feature set actually run, in a fixed color per feature
    set so it means the same thing in every group. This is the figure
    that shows the project's actual finding -- adding more data sources
    doesn't uniformly help, and the picture only becomes visible by
    comparing feature sets side by side rather than reporting one number."""
    targets = sorted({s["target"] for s in specs}, key=lambda t: list(modeling.TARGETS).index(t))
    feature_sets = sorted(
        {s["feature_set"] for s in specs},
        key=lambda f: list(modeling.FEATURE_SETS).index(f) if f in modeling.FEATURE_SETS else 99,
    )
    by_key = {(s["target"], s["feature_set"]): s for s in specs}

    n_fs = len(feature_sets)
    group_w = 0.8
    bar_w = group_w / max(n_fs, 1)
    x = np.arange(len(targets))

    for i, fs in enumerate(feature_sets):
        means = [by_key[(t, fs)]["mean_auc"] if (t, fs) in by_key else np.nan for t in targets]
        stds = [by_key[(t, fs)]["std_auc"] if (t, fs) in by_key else 0 for t in targets]
        offset = (i - (n_fs - 1) / 2) * bar_w
        bars = ax.bar(
            x + offset, means, bar_w * 0.92, yerr=stds, capsize=2.5,
            color=_CAT[i % len(_CAT)], label=_FEATURE_SET_LABELS.get(fs, fs), zorder=3,
            error_kw={"elinewidth": 0.9, "ecolor": _INK2}, edgecolor="white", linewidth=0.9,
        )
        for b, m, s in zip(bars, means, stds):
            if np.isnan(m):
                continue
            ax.text(
                b.get_x() + b.get_width() / 2, m + s + 0.02, f"{m:.2f}",
                ha="center", va="bottom", fontsize=6.5, color=_INK2,
            )

    ax.axhline(0.5, color=_MUTED, linestyle="--", linewidth=1, zorder=2)
    ax.text(len(targets) - 0.45, 0.505, "chance", fontsize=7.5, color=_MUTED, va="bottom")
    ax.set_xticks(x)
    ax.set_xticklabels([modeling.TARGETS[t]["description"] for t in targets], fontsize=8.5)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Nested-CV AUC (mean ± fold std)", fontsize=8.5, color=_MUTED)
    ax.set_title(
        "More data isn't automatically better: AUC by feature set and target",
        fontsize=10.5, fontweight="bold", loc="left", color=_INK,
    )
    ax.grid(axis="y", color=_GRID, linewidth=0.8, zorder=0)
    ax.legend(loc="upper right", ncol=1, fontsize=7.5, frameon=False)
    _style_axes(ax)


def _stability_bar(ax, stability_df, top_n=12):
    top = stability_df.head(top_n).iloc[::-1]  # smallest at bottom, largest at top when drawn
    y = np.arange(len(top))
    # This ranks features by *magnitude* of instability -- a sequential
    # (one-hue, light-to-dark) ramp is the semantically correct encoding
    # for magnitude, not a flat categorical fill. Normalized in log-space
    # since the axis is log-scale (a 100x-unstable feature should read as
    # far more saturated than a 2x one, not just "on the same bar color").
    vals = top["median_abs_rel_change"].to_numpy()
    log_vals = np.log10(np.clip(vals, 1e-6, None))
    span = max(log_vals.max() - log_vals.min(), 1e-9)
    norm = (log_vals - log_vals.min()) / span
    colors = [_seq_color(0.3 + 0.7 * n) for n in norm]
    ax.barh(y, vals, color=colors, height=0.62, zorder=3, edgecolor="white", linewidth=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels(top["feature"], fontsize=8, fontfamily="monospace")
    ax.set_xscale("log")
    ax.set_xlabel(
        "Median |relative change| between -1mm and +1mm (log scale)", fontsize=8.5, color=_MUTED
    )
    ax.set_title(
        "Least stable features under segmentation perturbation", fontsize=10, fontweight="bold",
        loc="left", color=_INK,
    )
    ax.grid(axis="x", color=_GRID, linewidth=0.8, zorder=0, which="both")
    _style_axes(ax)
    ax.spines["left"].set_visible(False)
    ax.tick_params(left=False)


def _perturbation_auc_bars(ax, comparison_df, target):
    sub = comparison_df[comparison_df["target"] == target].sort_values("perturbation_mm")
    labels = [f"{mm:+g}mm" for mm in sub["perturbation_mm"]]
    values = sub["mean_auc"].to_numpy()
    original = sub["original_auc"].iloc[0] if len(sub) else None

    # Color encodes direction (erosion vs. dilation), not an arbitrary
    # category -- shrinking the mask (blue) and growing it (orange) are a
    # meaningfully different manipulation, and coloring them differently
    # reinforces that even before reading the x-axis labels.
    x = np.arange(len(labels))
    colors = [_CAT[0] if mm < 0 else _CAT[1] for mm in sub["perturbation_mm"]]
    bars = ax.bar(x, values, color=colors, width=0.5, zorder=3, edgecolor="white", linewidth=1.0)
    # Value label inside the bar (white, near the top) rather than above
    # it -- avoids colliding with the horizontal "original" reference
    # line, which can sit right at a bar's height.
    for b, v in zip(bars, values):
        ax.text(
            b.get_x() + b.get_width() / 2, v - 0.03, f"{v:.2f}",
            ha="center", va="top", fontsize=8, color="white", fontweight="bold", zorder=4,
        )
    if original is not None:
        ax.axhline(original, color=_MUTED, linestyle="--", linewidth=1.3, zorder=2)
        ax.text(
            len(labels) - 0.4, min(original + 0.04, 0.97), f"original: {original:.2f}",
            va="bottom", fontsize=8, color=_INK2,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylim(0, 1)
    ax.set_ylabel("AUC", fontsize=8.5, color=_MUTED)
    ax.set_title(
        f"{modeling.TARGETS[target]['description']}: AUC using ONLY perturbed features",
        fontsize=10, fontweight="bold", loc="left", color=_INK,
    )
    ax.grid(axis="y", color=_GRID, linewidth=0.8, zorder=0)
    _style_axes(ax)


def _km_plot(ax, ax_risk, oof: pd.DataFrame):
    """Kaplan-Meier curves for patients above/below median predicted risk
    (out-of-fold risk scores only -- no patient's curve uses a risk score
    from a model trained on them), with a log-rank test for separation
    and a "number at risk" table underneath -- standard in a published KM
    figure, since the curve's right-hand tail is drawn from very few
    remaining patients and reads as more certain than it is without one."""
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import logrank_test

    median_risk = oof["risk_score"].median()
    high = oof[oof["risk_score"] >= median_risk]
    low = oof[oof["risk_score"] < median_risk]

    kmf_high = KaplanMeierFitter()
    kmf_high.fit(high["duration_days"], high["event"], label=f"Predicted high risk (n={len(high)})")
    kmf_high.plot_survival_function(
        ax=ax, color=_CAT[1], ci_show=True, ci_alpha=0.15, linewidth=2.2, censor_styles={"marker": "|", "ms": 7}
    )

    kmf_low = KaplanMeierFitter()
    kmf_low.fit(low["duration_days"], low["event"], label=f"Predicted low risk (n={len(low)})")
    kmf_low.plot_survival_function(
        ax=ax, color=_CAT[0], ci_show=True, ci_alpha=0.15, linewidth=2.2, censor_styles={"marker": "|", "ms": 7}
    )

    result = logrank_test(
        high["duration_days"], low["duration_days"],
        event_observed_A=high["event"], event_observed_B=low["event"],
    )
    p_text = "p < 0.001" if result.p_value < 0.001 else f"p = {result.p_value:.3f}"
    ax.set_xlabel("", fontsize=8.5, color=_MUTED)
    ax.set_ylabel("Survival probability", fontsize=8.5, color=_MUTED)
    ax.set_title(
        f"Kaplan-Meier by out-of-fold predicted risk group (log-rank {p_text})",
        fontsize=10, fontweight="bold", loc="left", color=_INK,
    )
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1.02)
    ax.grid(color=_GRID, linewidth=0.7, zorder=0, alpha=0.7)
    legend = ax.legend(fontsize=8.5, frameon=True, facecolor="white", edgecolor=_GRID, framealpha=0.92)
    legend.get_frame().set_linewidth(0.6)
    _style_axes(ax)
    ax.tick_params(labelbottom=False)  # x tick labels live on the risk table below instead

    xmax = ax.get_xlim()[1]
    time_points = np.linspace(0, xmax, 6)
    ax_risk.set_xlim(0, xmax)
    ax_risk.set_ylim(-1.9, 1.6)  # room below y=0 for tick numbers + "Days since CT" caption
    ax_risk.axis("off")
    for row_i, (label, grp, color) in enumerate([("High risk", high, _CAT[1]), ("Low risk", low, _CAT[0])]):
        y = 1 - row_i
        ax_risk.text(
            -0.01, y, label, transform=ax_risk.get_yaxis_transform(), ha="right", va="center",
            fontsize=7.5, color=color, fontweight="bold", clip_on=False,
        )
        for t in time_points:
            n_at_risk = int((grp["duration_days"] >= t).sum())
            ax_risk.text(t, y, str(n_at_risk), ha="center", va="center", fontsize=7.5, color=_INK2)
    ax_risk.text(
        -0.01, 1.55, "No. at risk", transform=ax_risk.get_yaxis_transform(), ha="right", va="center",
        fontsize=7.5, color=_MUTED, style="italic", clip_on=False,
    )
    for t in time_points:
        ax_risk.text(t, -0.85, f"{t:.0f}", ha="center", va="top", fontsize=7.5, color=_MUTED)
    ax_risk.text(xmax / 2, -1.5, "Days since CT", ha="center", va="top", fontsize=8.5, color=_MUTED)


class SurvivalDialog(tk.Toplevel):
    """Cox proportional-hazards survival model -- see survival.py's
    docstring for why this uses actual follow-up time + censoring instead
    of collapsing outcome to a binary recurrence/no-recurrence label."""

    def __init__(self, app: PipelineGUI):
        super().__init__(app)
        self.app = app
        self.title("Run survival model")
        self.resizable(False, False)
        self.grab_set()

        pad = {"padx": 8, "pady": 6}

        rad_n = pd.read_csv(config.RADIOMICS_OUT)["case_id"].nunique()
        clin = pd.read_csv(config.CLINICAL_OUT)
        n_events = int(clin["deceased"].sum())
        note = (
            f"Overall survival (Cox PH): {rad_n} patients with radiomics, "
            f"{n_events} deaths observed cohort-wide. Duration measured from "
            f"CT date (see survival.py for why, not the dataset's own "
            f"time_to_death_days column). Scored by concordance index (same "
            f"0.5=chance/1.0=perfect scale as AUC)."
        )
        ttk.Label(self, text=note, wraplength=360, justify="left").grid(
            row=0, column=0, columnspan=2, sticky="w", **pad
        )

        ttk.Label(self, text="Outer folds:").grid(row=1, column=0, sticky="w", **pad)
        self.outer_var = tk.IntVar(value=5)
        ttk.Spinbox(self, from_=3, to=10, textvariable=self.outer_var, width=5).grid(
            row=1, column=1, sticky="w", **pad
        )

        ttk.Label(self, text="Inner folds:").grid(row=2, column=0, sticky="w", **pad)
        self.inner_var = tk.IntVar(value=5)
        ttk.Spinbox(self, from_=3, to=10, textvariable=self.inner_var, width=5).grid(
            row=2, column=1, sticky="w", **pad
        )

        ttk.Label(self, text="Random seed:").grid(row=3, column=0, sticky="w", **pad)
        self.seed_var = tk.IntVar(value=0)
        ttk.Entry(self, textvariable=self.seed_var, width=7).grid(row=3, column=1, sticky="w", **pad)

        btns = ttk.Frame(self)
        btns.grid(row=4, column=0, columnspan=2, pady=(8, 10))
        ttk.Button(btns, text="Run", command=self.start).pack(side="left", padx=6)
        ttk.Button(btns, text="Cancel", command=self.destroy).pack(side="left", padx=6)

    def start(self):
        outer_folds = self.outer_var.get()
        inner_folds = self.inner_var.get()
        seed = self.seed_var.get()
        self.destroy()

        def task():
            survival.run(outer_folds=outer_folds, inner_folds=inner_folds, random_state=seed)

        self.app.run_in_thread(task, indeterminate=True)


class PerturbationDialog(tk.Toplevel):
    """Two-step segmentation-perturbation robustness study: (1) re-extract
    radiomics features from eroded/dilated versions of every tumor mask,
    (2) compare which features are least stable under that boundary shift
    and re-run the same nested-CV model on the perturbed-only feature
    tables to see how much the reported AUC moves from boundary variation
    alone -- see perturbation.py's docstring for the full rationale."""

    def __init__(self, app: PipelineGUI):
        super().__init__(app)
        self.app = app
        self.title("Segmentation adjustment study")
        self.resizable(False, False)
        self.grab_set()

        pad = {"padx": 8, "pady": 6}

        seg = pd.read_csv(config.SERIES_INDEX_OUT)
        n_available = seg[seg["modality"] == "SEG"]["case_id"].nunique()
        existing = 0
        feat_path = config.PROCESSED_DIR / "perturbation_features.csv"
        if feat_path.exists():
            existing = pd.read_csv(feat_path)["case_id"].nunique()

        mm_str = ", ".join(f"{m:+g}" for m in perturbation.DEFAULT_MM_LEVELS)
        intro = (
            f"Step 1 re-extracts radiomics from masks perturbed by "
            f"{mm_str} mm for each of {n_available} patients "
            f"(reuses already-downloaded CTs, no new downloads) -- "
            f"{'not started' if existing == 0 else f'{existing}/{n_available} patients done'} yet. "
            f"This takes a while (~10s/patient). Step 2 needs step 1 finished."
        )
        ttk.Label(self, text=intro, wraplength=380, justify="left").grid(
            row=0, column=0, columnspan=2, sticky="w", **pad
        )

        btns1 = ttk.Frame(self)
        btns1.grid(row=1, column=0, columnspan=2, pady=(0, 10))
        ttk.Button(
            btns1, text="1. Extract perturbed features (all patients)", command=self.start_extraction
        ).pack(side="left", padx=6)

        ttk.Label(
            self,
            text="Step 2: compare feature stability between -1mm and +1mm "
            "(a realistic inter-radiologist boundary disagreement), then "
            "re-run nested CV using only the eroded or only the dilated "
            "features, for each target below.",
            wraplength=380, justify="left",
        ).grid(row=2, column=0, columnspan=2, sticky="w", padx=8)

        self.target_vars = {}
        for i, target in enumerate(modeling.TARGETS):
            var = tk.BooleanVar(value=(target in ("recurrence", "egfr")))
            ttk.Checkbutton(self, text=target, variable=var).grid(
                row=3 + i, column=0, sticky="w", padx=16
            )
            self.target_vars[target] = var

        btns2 = ttk.Frame(self)
        btns2.grid(row=3 + len(modeling.TARGETS), column=0, columnspan=2, pady=(10, 10))
        ttk.Button(
            btns2, text="2. Compare stability + AUC", command=self.start_comparison
        ).pack(side="left", padx=6)
        ttk.Button(btns2, text="Close", command=self.destroy).pack(side="left", padx=6)

    def start_extraction(self):
        def task():
            perturbation.run_perturbation_study(mm_levels=perturbation.DEFAULT_MM_LEVELS)

        self.app.run_in_thread(task, indeterminate=True)
        self.destroy()

    def start_comparison(self):
        targets = tuple(t for t, v in self.target_vars.items() if v.get())
        if not targets:
            messagebox.showwarning("Missing input", "Pick at least one target.")
            return
        feat_path = config.PROCESSED_DIR / "perturbation_features.csv"
        if not feat_path.exists():
            messagebox.showwarning("No data yet", "Run step 1 first.")
            return

        seg = pd.read_csv(config.SERIES_INDEX_OUT)
        n_available = seg[seg["modality"] == "SEG"]["case_id"].nunique()
        n_done = pd.read_csv(feat_path)["case_id"].nunique()
        if n_done < n_available:
            proceed = messagebox.askyesno(
                "Extraction not finished",
                f"Only {n_done}/{n_available} patients have perturbed features so far -- "
                f"step 1 is still running or was stopped early. Comparing on this few "
                f"patients isn't meaningful (tiny, unrepresentative sample). Proceed anyway?",
                default="no",
            )
            if not proceed:
                return
        self.destroy()

        def task():
            df = pd.read_csv(feat_path)
            print("Top 10 least-stable features between -1mm and +1mm perturbation:")
            stability = perturbation.feature_stability_summary(df, -1, 1)
            print(stability.head(10).to_string(index=False))
            print()
            perturbation.run_auc_comparison(targets=targets)

        self.app.run_in_thread(task, indeterminate=True)


# Figures drawn by `python pipeline.py run figures`.
FIGURES_DIR = Path(__file__).resolve().parents[2] / "outputs" / "figures"
# Optional: the built manuscript (manuscript/PAPER.md, supplementary/SUPPLEMENTARY.md), kept outside this
# repository. If NSCLC_PAPER_DIR points at it, the figure browser also shows the paper's numbers and legends;
# otherwise it lists the figure files in FIGURES_DIR.
PAPER_DIR = Path(os.environ.get("NSCLC_PAPER_DIR", Path(__file__).resolve().parents[2] / "paper"))


def _plain(text: str) -> str:
    """Markdown legend -> plain text for display."""
    import re
    text = re.sub(r"<sup>(.*?)</sup>", r"\1", text)
    return re.sub(r"\*\*|\*", "", text).strip()


def figure_files(kind: str) -> list[dict]:
    """Figure files in outputs/figures/ when no built manuscript is available. kind = 'main' or 'supp'."""
    import re
    items = []
    if kind == "main":
        for path in sorted((FIGURES_DIR / "main").glob("figure*.png"),
                           key=lambda p: int(re.match(r"figure(\d+)", p.stem).group(1))):
            num = re.match(r"figure(\d+)", path.stem).group(1)
            title = path.stem.split("_", 1)[1].replace("_", " ") if "_" in path.stem else ""
            items.append({"number": f"Fig. {num}", "title": title, "legend": "", "path": path})
    else:
        for path in sorted((FIGURES_DIR / "supplementary").glob("*.png")):
            items.append({"number": "Suppl. Fig.", "title": path.stem.replace("_", " "), "legend": "", "path": path})
    return items


def paper_figures(kind: str) -> list[dict]:
    """The paper's figures, in order, read from the built documents so the
    dashboard always matches the current manuscript. kind = 'main' or 'supp'.
    Each item: number, title, legend, image path. Without a built manuscript,
    the figure files in outputs/figures/ are listed instead."""
    import re
    items = []
    if kind == "main":
        md_path = PAPER_DIR / "manuscript" / "PAPER.md"
        if not md_path.exists():
            return figure_files(kind)
        md = md_path.read_text(encoding="utf-8")
        images = dict(re.findall(r"^!\[Figure (\d+)\]\(([^)]+)\)", md, re.M))
        for num, title, legend in re.findall(r"^\*\*Fig\. (\d+) \| (.*?)\*\*(.*?)(?=\n\n|\Z)", md, re.M | re.S):
            if num in images:
                items.append({"number": f"Fig. {num}", "title": _plain(title), "legend": _plain(legend),
                              "path": (md_path.parent / images[num]).resolve()})
    else:
        md_path = PAPER_DIR / "supplementary" / "SUPPLEMENTARY.md"
        if not md_path.exists():
            return figure_files(kind)
        md = md_path.read_text(encoding="utf-8")
        for num, title, legend, img in re.findall(
                r"^\*\*Supplementary Fig\. S?(\d+) \| (.*?)\*\*(.*?)\n\n!\[[^\]]*\]\(([^)]+)\)", md, re.M | re.S):
            items.append({"number": f"Suppl. Fig. {num}", "title": _plain(title), "legend": _plain(legend),
                          "path": (md_path.parent / img).resolve()})
    return items


class PaperFiguresPanel(ttk.Frame):
    """The paper's own figure files (outputs/figures/main, outputs/figures/supplementary)
    with their numbers, titles and legends: a list on the left, the image on
    the right (fit to window, or zoom and scroll), and the full legend below."""

    def __init__(self, parent, kind: str):
        super().__init__(parent)
        self.items = paper_figures(kind)
        self.zoom = None  # None = fit to window width
        self._photo = None
        self._image = None

        if not self.items:
            ttk.Label(self, text="No figures found. Run: python pipeline.py run figures",
                      foreground=_MUTED).pack(padx=12, pady=12, anchor="w")
            return

        left = ttk.Frame(self)
        left.pack(side="left", fill="y", padx=(8, 4), pady=8)
        self.listbox = tk.Listbox(left, width=44, activestyle="none", exportselection=False)
        for it in self.items:
            self.listbox.insert("end", f"{it['number']}  {it['title']}")
        self.listbox.pack(fill="y", expand=True)
        self.listbox.bind("<<ListboxSelect>>", lambda e: self._show())

        right = ttk.Frame(self)
        right.pack(side="left", fill="both", expand=True, padx=(4, 8), pady=8)
        self.heading = ttk.Label(right, font=("Arial", 10, "bold"), anchor="w", wraplength=880, justify="left")
        self.heading.pack(fill="x")
        bar = ttk.Frame(right)
        bar.pack(fill="x", pady=(4, 0))
        for label, cmd in (("Fit to width", lambda: self._set_zoom(None)), ("Zoom −", lambda: self._set_zoom(0.8)),
                           ("Zoom +", lambda: self._set_zoom(1.25)), ("Open image", self._open_image),
                           ("Open folder", self._open_folder)):
            ttk.Button(bar, text=label, command=cmd).pack(side="left", padx=(0, 4))

        view = ttk.Frame(right)
        view.pack(fill="both", expand=True, pady=(6, 4))
        self.canvas = tk.Canvas(view, background="white", highlightthickness=0)
        vsb = ttk.Scrollbar(view, orient="vertical", command=self.canvas.yview)
        hsb = ttk.Scrollbar(view, orient="horizontal", command=self.canvas.xview)
        self.canvas.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        vsb.pack(side="right", fill="y")
        hsb.pack(side="bottom", fill="x")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda e: self.zoom is None and self._render())
        self.canvas.bind("<Enter>", lambda e: self.canvas.bind_all("<MouseWheel>", self._wheel))
        self.canvas.bind("<Leave>", lambda e: self.canvas.unbind_all("<MouseWheel>"))

        self.legend = scrolledtext.ScrolledText(right, height=6, wrap="word", font=("Arial", 9))
        self.legend.pack(fill="x")

        self.listbox.selection_set(0)
        self.after(50, self._show)

    def _current(self):
        sel = self.listbox.curselection()
        return self.items[sel[0] if sel else 0]

    def _show(self):
        from PIL import Image
        it = self._current()
        self.heading.configure(text=f"{it['number']} | {it['title']}")
        self.legend.configure(state="normal")
        self.legend.delete("1.0", "end")
        self.legend.insert("1.0", f"{it['number']} | {it['title']}\n\n{it['legend']}\n\nFile: {it['path']}")
        self.legend.configure(state="disabled")
        self._image = Image.open(it["path"]) if it["path"].exists() else None
        self._render()

    def _render(self):
        from PIL import Image
        self.canvas.delete("all")
        if self._image is None:
            self.canvas.create_text(20, 20, anchor="nw", text="Image file not found - run: python pipeline.py run figures")
            return
        w = max(self.canvas.winfo_width() - 4, 200)
        scale = (w / self._image.width) if self.zoom is None else self.zoom
        size = (max(int(self._image.width * scale), 1), max(int(self._image.height * scale), 1))
        self._photo = ImageTk.PhotoImage(self._image.convert("RGB").resize(size, Image.LANCZOS))
        self.canvas.create_image(0, 0, anchor="nw", image=self._photo)
        self.canvas.configure(scrollregion=(0, 0, *size))

    def _set_zoom(self, factor):
        if factor is None or self._image is None:
            self.zoom = None
        else:
            current = self.zoom or (max(self.canvas.winfo_width() - 4, 200) / self._image.width)
            self.zoom = min(max(current * factor, 0.1), 4.0)
        self._render()

    def _wheel(self, event):
        self.canvas.yview_scroll(int(-event.delta / 120), "units")

    def _open_image(self):
        _open_path(self._current()["path"])

    def _open_folder(self):
        _open_path(self._current()["path"].parent)


def _open_path(path):
    """Open a file or folder with the operating system's default application."""
    import subprocess
    if sys.platform == "win32":
        os.startfile(path)  # noqa: S606 -- local file chosen by the user
    elif sys.platform == "darwin":
        subprocess.run(["open", str(path)], check=False)
    else:
        subprocess.run(["xdg-open", str(path)], check=False)


class DashboardDialog(tk.Toplevel):
    """The paper's figures, natively in this window -- no browser involved.

    Two tabs: the main figures and the supplementary figures, i.e. the files
    in outputs/figures/main and outputs/figures/supplementary with the numbers,
    titles and legends of the current manuscript. The older live-chart tabs
    (_build_*_tab below) are no longer shown; their drawing helpers are kept
    because archived figure scripts import them."""

    def __init__(self, app: PipelineGUI):
        super().__init__(app)
        self.app = app
        self.title("Dashboard")
        self.geometry("1320x900")
        self._cursors = []  # keep mplcursors objects alive (they detach if GC'd)

        notebook = ttk.Notebook(self)
        notebook.pack(fill="both", expand=True)
        notebook.add(PaperFiguresPanel(notebook, "main"), text="Paper: main figures")
        notebook.add(PaperFiguresPanel(notebook, "supp"), text="Paper: supplementary figures")

    def _scroll_frame(self, parent):
        canvas = tk.Canvas(parent, highlightthickness=0)
        vsb = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        inner = ttk.Frame(canvas)
        inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.configure(yscrollcommand=vsb.set)
        canvas.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")

        def _wheel(event):
            canvas.yview_scroll(int(-event.delta / 120), "units")

        canvas.bind("<Enter>", lambda e: canvas.bind_all("<MouseWheel>", _wheel))
        canvas.bind("<Leave>", lambda e: canvas.unbind_all("<MouseWheel>"))
        return inner

    def _embed(self, parent, fig):
        toolbar_frame = ttk.Frame(parent)
        toolbar_frame.pack(fill="x")
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        NavigationToolbar2Tk(canvas, toolbar_frame).update()
        ttk.Button(
            toolbar_frame, text="Export figure (PNG 300dpi + PDF)...",
            command=lambda: self._export_figure(fig),
        ).pack(side="right", padx=6)
        canvas.get_tk_widget().pack(fill="both", expand=True)
        return canvas

    def _export_figure(self, fig):
        path = filedialog.asksaveasfilename(
            title="Export figure",
            defaultextension=".png",
            initialfile="figure",
            filetypes=[("PNG + PDF (both)", "*.png")],
        )
        if not path:
            return
        base = path[:-4] if path.lower().endswith(".png") else path
        png_path, pdf_path = f"{base}.png", f"{base}.pdf"
        fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
        fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
        messagebox.showinfo("Exported", f"Saved:\n{png_path}\n{pdf_path}")

    def _build_summary_tab(self, notebook, specs):
        tab = ttk.Frame(notebook)
        notebook.add(tab, text="Explore: feature sets")
        inner = self._scroll_frame(tab)

        fig = Figure(figsize=(9.6, 5.6), dpi=100)
        ax = fig.subplots(1, 1)
        _feature_set_comparison(ax, specs)
        fig.suptitle(
            "Radiomics, clinical, RNA-seq and CNN embeddings compared under identical nested cross-validation (exploratory)",
            fontsize=10.5, fontweight="bold", x=0.02, ha="left",
        )
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        self._embed(inner, fig)

        ttk.Label(
            inner,
            text="Every bar is the same model (logistic regression, nested cross-validation) with a "
            "different set of input features. Exploratory; see the paper for the reported results.",
            foreground=_MUTED, wraplength=880, justify="left",
        ).pack(anchor="w", padx=8, pady=(0, 12))

    def _build_cohort_tab(self, notebook, clin):
        tab = ttk.Frame(notebook)
        notebook.add(tab, text="Explore: cohort")
        inner = self._scroll_frame(tab)

        panels = [
            ("Site", clin["site"]),
            ("Gender", clin["gender"]),
            ("Histology", clin["histology"]),
            ("Smoking status", clin["smoking_status"]),
            ("EGFR mutation", clin["egfr_mutation_status"]),
            ("KRAS mutation", clin["kras_mutation_status"]),
            ("ALK translocation", clin["alk_translocation_status"]),
            ("Recurrence", clin["recurrence"]),
            ("Survival status", clin["deceased"].map({True: "Dead", False: "Alive"})),
        ]

        fig = Figure(figsize=(9.6, 17), dpi=100)
        axes = fig.subplots(5, 2)
        flat = axes.flatten()
        letters = "abcdefghij"
        for ax, (title, series), letter in zip(flat, panels, letters):
            vc = series.fillna("Unknown").value_counts()
            _hbar(ax, list(vc.items()), title)
            _panel_label(ax, letter)
        _hist(flat[9], clin["age_at_histological_diagnosis"].dropna(), "Age at diagnosis", "years")
        _panel_label(flat[9], letters[9])
        fig.suptitle(f"Cohort characteristics: n={len(clin)} patients (exploratory)", fontsize=12, fontweight="bold", x=0.02, ha="left")
        fig.tight_layout(rect=[0, 0.01, 1, 0.96])
        self._embed(inner, fig)

    def _build_radiomics_tab(self, notebook, rad, clin):
        tab = ttk.Frame(notebook)
        notebook.add(tab, text="Explore: radiomics")
        inner = self._scroll_frame(tab)

        merged = rad.merge(clin[["case_id", "recurrence"]], on="case_id", how="left")
        merged["recurrence"] = merged["recurrence"].fillna("unknown")
        # yes/no only for the violins -- statannotations' Mann-Whitney
        # needs exactly two groups, and the 1 patient with unknown
        # recurrence status has no group to compare against anyway.
        known = merged[merged["recurrence"].isin(["no", "yes"])]

        fig = Figure(figsize=(9.6, 12), dpi=100)
        mosaic = fig.subplot_mosaic([["A", "B"], ["C", "D"], ["E", "E"]])
        _violin_by_group(
            mosaic["A"], known, "shape_volume_mm3", "recurrence", ["no", "yes"],
            "Tumor volume by recurrence", "Volume (mm³)",
            group_labels={"no": "No recurrence", "yes": "Recurrence"},
        )
        _violin_by_group(
            mosaic["B"], known, "firstorder_mean_hu", "recurrence", ["no", "yes"],
            "Mean intensity by recurrence", "Mean HU",
            group_labels={"no": "No recurrence", "yes": "Recurrence"},
        )
        _hist(mosaic["C"], rad["shape_sphericity"], "Sphericity")
        _hist(mosaic["D"], rad["texture_contrast"], "GLCM texture contrast")
        for key, letter in zip(["A", "B", "C", "D"], "abcd"):
            _panel_label(mosaic[key], letter)

        sc = _scatter(mosaic["E"], merged)
        _panel_label(mosaic["E"], "e")

        cursor = mplcursors.cursor(sc, hover=True)

        @cursor.connect("add")
        def _(sel, _df=merged):
            row = _df.iloc[sel.index]
            sel.annotation.set_text(
                f"{row.case_id}\nvolume {row.shape_volume_mm3:.0f} mm³\n"
                f"contrast {row.texture_contrast:.1f}\nrecurrence: {row.recurrence}"
            )

        self._cursors.append(cursor)

        fig.suptitle(
            f"Radiomic feature distributions: n={len(rad)} patients with a tumor mask (exploratory)",
            fontsize=12, fontweight="bold", x=0.02, ha="left",
        )
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        self._embed(inner, fig)

    def _load_model_specs(self):
        from sklearn.metrics import roc_curve

        # modeling.py names files "modeling_results_{target}.csv" for the
        # default "radiomics" feature set, or
        # "modeling_results_{target}__{feature_set with + -> _}.csv"
        # otherwise -- discover whatever combinations have actually been
        # run, rather than assuming a fixed list of targets.
        suffix_to_feature_set = {fs.replace("+", "_"): fs for fs in modeling.FEATURE_SETS}

        specs = []
        for results_path in sorted(config.PROCESSED_DIR.glob("modeling_results_*.csv")):
            stem = results_path.stem.removeprefix("modeling_results_")
            if "__" in stem:
                target, fs_suffix = stem.split("__", 1)
                if fs_suffix.startswith("perturb_"):
                    # These are segmentation-perturbation runs (Perturbation
                    # tab / Fig. 3-6), not a feature-set choice -- globbing
                    # modeling_results_*.csv otherwise sweeps them into this
                    # feature-fusion comparison and clutters it with bars
                    # like "perturb_m2mm" that don't belong here.
                    continue
                feature_set = suffix_to_feature_set.get(fs_suffix, fs_suffix)
            else:
                target, feature_set = stem, "radiomics"
            if target not in modeling.TARGETS:
                continue
            preds_path = config.PROCESSED_DIR / f"modeling_predictions_{stem}.csv"
            if not preds_path.exists():
                continue

            folds = pd.read_csv(results_path)
            preds = pd.read_csv(preds_path)
            fpr, tpr, _ = roc_curve(preds["y_true"], preds["y_proba"])
            specs.append(
                {
                    "target": target,
                    "feature_set": feature_set,
                    "title": f"{modeling.TARGETS[target]['description']} ({feature_set})",
                    "fpr": fpr,
                    "tpr": tpr,
                    "fold_aucs": folds["auc"].to_numpy(),
                    "mean_auc": folds["auc"].mean(),
                    "std_auc": folds["auc"].std(),
                    "n": len(preds),
                    "n_positive": int(preds["y_true"].sum()),
                }
            )

        target_order = list(modeling.TARGETS)
        fs_order = list(modeling.FEATURE_SETS)
        specs.sort(
            key=lambda s: (
                target_order.index(s["target"]) if s["target"] in target_order else 99,
                fs_order.index(s["feature_set"]) if s["feature_set"] in fs_order else 99,
            )
        )
        return specs

    def _build_modeling_tab(self, notebook, specs):
        tab = ttk.Frame(notebook)
        notebook.add(tab, text="Explore: modeling")
        inner = self._scroll_frame(tab)

        fig = Figure(figsize=(9.6, 4.2 * len(specs)), dpi=100)
        axes = fig.subplots(len(specs), 2, squeeze=False)
        target_order = list(modeling.TARGETS)
        for row, spec in zip(axes, specs):
            # Very light background tint per target (not per row) so a
            # long auto-discovered list visually groups by target at a
            # glance, without needing a separate header row per group.
            tint = _CAT[target_order.index(spec["target"]) % len(_CAT)]
            for ax in row:
                ax.set_facecolor(tint + "14")  # ~8% alpha via hex suffix
            _roc(row[0], spec)
            _fold_strip(
                row[1], spec["fold_aucs"], spec["mean_auc"],
                f"mean AUC {spec['mean_auc']:.2f} ± {spec['std_auc']:.2f}, "
                f"n={spec['n']}, {spec['n_positive']} positive",
            )
        fig.suptitle(
            "Nested cross-validation detail, by target and feature set (exploratory; the AUC moves with --seed)",
            fontsize=12, fontweight="bold", x=0.02, ha="left",
        )
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        self._embed(inner, fig)

    def _build_interpretation_tab(self, notebook, specs):
        """SHAP beeswarm per target, always with the 'radiomics' feature set so
        the targets can be compared. Computed when the tab opens (under a
        second); see interpretation.py for how this differs from the
        nested cross-validation results."""
        targets_with_radiomics = sorted(
            {s["target"] for s in specs if s["feature_set"] == "radiomics"},
            key=lambda t: list(modeling.TARGETS).index(t),
        )
        if not targets_with_radiomics:
            return

        tab = ttk.Frame(notebook)
        notebook.add(tab, text="Explore: interpretation")
        inner = self._scroll_frame(tab)

        results = []
        for target in targets_with_radiomics:
            try:
                sv, case_ids = interpretation.compute_shap(target, "radiomics")
                results.append((target, sv))
            except Exception as e:  # noqa: BLE001 -- skip a target SHAP can't handle, don't crash the tab
                print(f"SHAP failed for {target}: {type(e).__name__}: {e}")

        if not results:
            return

        fig = Figure(figsize=(8.5, 5.2 * len(results)), dpi=100)
        axes = fig.subplots(len(results), 1, squeeze=False)
        for (target, sv), letter, row in zip(results, "abcdefgh", axes):
            ax = row[0]
            shap.plots.beeswarm(sv, ax=ax, show=False, max_display=12, plot_size=None, color_bar=True)
            ax.set_title(
                f"{modeling.TARGETS[target]['description']} (radiomics)",
                fontsize=10, fontweight="bold", loc="left", color=_INK,
            )
            _panel_label(ax, letter)

        fig.suptitle(
            "SHAP feature importance: which radiomic features drive each model, "
            "and in which direction (fit on all data; descriptive, not a CV performance claim)",
            fontsize=11.5, fontweight="bold", x=0.02, ha="left",
        )
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        self._embed(inner, fig)

    def _build_perturbation_tab(self, notebook):
        tab = ttk.Frame(notebook)
        notebook.add(tab, text="Explore: segmentation adjustment")
        inner = self._scroll_frame(tab)

        feat_path = config.PROCESSED_DIR / "perturbation_features.csv"
        pert = pd.read_csv(feat_path)
        n_done = pert["case_id"].nunique()
        seg = pd.read_csv(config.SERIES_INDEX_OUT)
        n_available = seg[seg["modality"] == "SEG"]["case_id"].nunique()

        comparison_path = config.PROCESSED_DIR / "perturbation_auc_comparison.csv"
        comparison = pd.read_csv(comparison_path) if comparison_path.exists() else None
        targets_shown = sorted(comparison["target"].unique()) if comparison is not None else []

        n_rows = 1 + len(targets_shown)
        fig = Figure(figsize=(9.6, 3.6 + 3.2 * len(targets_shown)), dpi=100)
        if n_rows == 1:
            ax_stability = fig.subplots(1, 1)
        else:
            gs = fig.add_gridspec(n_rows, 1, height_ratios=[1.3] + [1] * len(targets_shown))
            ax_stability = fig.add_subplot(gs[0])

        have_both = {-1, 1}.issubset(set(pert["perturbation_mm"].unique()))
        if have_both:
            stability = perturbation.feature_stability_summary(pert, -1, 1)
            _stability_bar(ax_stability, stability)
            _panel_label(ax_stability, "a")
        else:
            ax_stability.axis("off")
            ax_stability.text(
                0.02, 0.5, "Need both -1mm and +1mm extracted to compare stability.",
                fontsize=10, color=_INK2, transform=ax_stability.transAxes,
            )

        letters = "bcdefgh"
        for i, (target, letter) in enumerate(zip(targets_shown, letters), start=1):
            ax = fig.add_subplot(gs[i])
            _perturbation_auc_bars(ax, comparison, target)
            _panel_label(ax, letter)

        coverage_note = f"{n_done}/{n_available} patients extracted"
        if n_done < n_available:
            coverage_note += " -- INCOMPLETE, re-run once extraction finishes for a real result"
        fig.suptitle(f"Segmentation adjustment: {coverage_note} (exploratory)", fontsize=12, fontweight="bold", x=0.02, ha="left")
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        self._embed(inner, fig)

        if not targets_shown:
            ttk.Label(
                inner,
                text="No AUC comparison yet. Run step 2 (\"Compare stability + AUC\") "
                "in the Segmentation adjustment study dialog after extraction.",
                foreground=_MUTED, wraplength=600,
            ).pack(anchor="w", padx=8, pady=(0, 10))

    def _build_survival_tab(self, notebook):
        tab = ttk.Frame(notebook)
        notebook.add(tab, text="Explore: survival")
        inner = self._scroll_frame(tab)

        oof = pd.read_csv(config.PROCESSED_DIR / "survival_predictions.csv")
        folds = pd.read_csv(config.PROCESSED_DIR / "survival_results.csv")

        # constrained_layout (not tight_layout) -- tight_layout warns and
        # miscomputes spacing when axes are sharex-linked, which ax_risk is.
        fig = Figure(figsize=(9.6, 9.2), dpi=100, layout="constrained")
        gs = fig.add_gridspec(3, 1, height_ratios=[3, 1.3, 1.3], hspace=0.12)
        ax_km = fig.add_subplot(gs[0])
        ax_risk = fig.add_subplot(gs[1], sharex=ax_km)
        _km_plot(ax_km, ax_risk, oof)
        _panel_label(ax_km, "a")

        ax_strip = fig.add_subplot(gs[2])
        _fold_strip(
            ax_strip, folds["concordance"].to_numpy(), folds["concordance"].mean(),
            f"n={len(oof)}, {int(oof['event'].sum())} events",
        )
        ax_strip.set_xlabel("Concordance index per outer fold (dashed = mean)", fontsize=8.5, color=_MUTED)
        _panel_label(ax_strip, "b")

        fig.suptitle(
            f"Survival (Cox PH, exploratory): mean concordance {folds['concordance'].mean():.2f} "
            f"± {folds['concordance'].std():.2f}",
            fontsize=12, fontweight="bold", x=0.02, ha="left",
        )
        self._embed(inner, fig)


class TablePreview(tk.Toplevel):
    """Read-only scrollable preview of a DataFrame (capped rows for responsiveness)."""

    MAX_ROWS = 500

    def __init__(self, app: PipelineGUI, title: str, df: pd.DataFrame):
        super().__init__(app)
        self.title(title)
        self.geometry("900x500")

        shown = df.head(self.MAX_ROWS)
        info_text = f"{len(df)} rows x {len(df.columns)} columns"
        if len(df) > self.MAX_ROWS:
            info_text += f" (showing first {self.MAX_ROWS})"
        ttk.Label(self, text=info_text).pack(anchor="w", padx=8, pady=4)

        frame = ttk.Frame(self)
        frame.pack(fill="both", expand=True, padx=8, pady=(0, 8))

        cols = list(shown.columns)
        tree = ttk.Treeview(frame, columns=cols, show="headings")
        for c in cols:
            tree.heading(c, text=c)
            tree.column(c, width=120, stretch=False)
        for _, row in shown.iterrows():
            tree.insert("", "end", values=[row[c] for c in cols])

        vsb = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        hsb = ttk.Scrollbar(frame, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        frame.grid_rowconfigure(0, weight=1)
        frame.grid_columnconfigure(0, weight=1)


def main():
    app = PipelineGUI()
    app.mainloop()


if __name__ == "__main__":
    main()
