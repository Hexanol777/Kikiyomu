import tkinter as tk
from tkinter import ttk
import time as _time


# ── Design tokens ──────────────────────────────────────────────────────────────
P = {
    "bg":      "#16171d",   # root / window
    "panel":   "#1e1f29",   # panel backgrounds
    "surface": "#262733",   # inputs, treeview, text areas
    "border":  "#31334a",   # dividers, separators
    "accent":  "#a78bfa",   # interactive, current state
    "text":    "#dddde8",   # primary text
    "muted":   "#636680",   # labels, timestamps, secondary text
    "success": "#86efac",   # model loaded, playing
    "error":   "#f87171",   # errors
    "warn":    "#fbbf24",   # warnings, loading state
    "ocr":     "#67e8f9",   # OCR output
    "force":   "#fb923c",   # force read
}

# Status bar states → (label text, color)
STATUS_CONFIG = {
    "idle":       ("● Idle",        P["muted"]),
    "loading":    ("● Loading…",    P["warn"]),
    "generating": ("● Generating…", P["accent"]),
    "playing":    ("● Playing",     P["success"]),
    "error":      ("● Error",       P["error"]),
}


def setup_theme(root):
    """Apply the dark ttk theme to the root window."""
    root.configure(bg=P["bg"])
    style = ttk.Style(root)
    style.theme_use("clam")

    style.configure("TFrame",
        background=P["panel"],
    )
    style.configure("TLabel",
        background=P["panel"],
        foreground=P["text"],
        font=("Segoe UI", 9),
    )
    style.configure("TButton",
        background=P["surface"],
        foreground=P["accent"],
        borderwidth=0,
        focusthickness=0,
        font=("Segoe UI", 9, "bold"),
        padding=(10, 6),
        relief="flat",
    )
    style.map("TButton",
        background=[("active",   P["accent"]), ("pressed",   P["accent"]), ("disabled", P["surface"])],
        foreground=[("active",   P["bg"]),     ("pressed",   P["bg"]),     ("disabled", P["muted"])],
        relief=[("pressed", "flat")],
    )
    style.configure("TEntry",
        fieldbackground=P["surface"],
        foreground=P["text"],
        insertcolor=P["text"],
        bordercolor=P["border"],
        lightcolor=P["surface"],
        darkcolor=P["surface"],
        selectbackground=P["accent"],
        selectforeground=P["bg"],
        padding=(4, 4),
    )
    style.configure("TCheckbutton",
        background=P["panel"],
        foreground=P["text"],
        font=("Segoe UI", 9),
        focusthickness=0,
        indicatorcolor=P["surface"],
        indicatorrelief="flat",
    )
    style.map("TCheckbutton",
        background=[("active", P["panel"])],
        foreground=[("active", P["accent"])],
        indicatorcolor=[("selected", P["accent"]), ("pressed", P["accent"])],
    )
    style.configure("Treeview",
        background=P["surface"],
        foreground=P["text"],
        fieldbackground=P["surface"],
        borderwidth=0,
        rowheight=26,
        font=("Segoe UI", 9),
    )
    style.configure("Treeview.Heading",
        background=P["panel"],
        foreground=P["muted"],
        borderwidth=0,
        font=("Segoe UI", 8, "bold"),
        relief="flat",
    )
    style.map("Treeview",
        background=[("selected", P["accent"])],
        foreground=[("selected", P["bg"])],
    )
    style.map("Treeview.Heading",
        background=[("active", P["panel"])],
        relief=[("active", "flat")],
    )
    style.configure("Vertical.TScrollbar",
        background=P["border"],
        troughcolor=P["panel"],
        borderwidth=0,
        arrowsize=0,
        width=5,
        relief="flat",
    )
    style.map("Vertical.TScrollbar",
        background=[("active", P["muted"]), ("pressed", P["accent"])],
    )
    style.configure("TScale",
        background=P["panel"],
        troughcolor=P["surface"],
        sliderlength=12,
        sliderrelief="flat",
        borderwidth=0,
    )
    style.map("TScale",
        background=[("active", P["panel"])],
        troughcolor=[("active", P["surface"])],
    )


# ── Widgets ────────────────────────────────────────────────────────────────────

class SignEntry(ttk.Frame):
    def __init__(self, parent, label_text, default_value=""):
        super().__init__(parent)
        tk.Label(
            self, text=label_text,
            fg=P["muted"], bg=P["panel"],
            font=("Segoe UI", 8),
        ).pack(anchor="w")
        self.entry = ttk.Entry(self)
        self.entry.pack(fill="x", ipady=3)
        self.entry.insert(0, default_value)

    def get(self):
        return self.entry.get()


class ModelTreeView(ttk.Frame):
    def __init__(self, parent, models_dir="models"):
        super().__init__(parent)
        self.models_dir = models_dir

        self.tree = ttk.Treeview(
            self, columns=("Model",), show="headings", selectmode="browse"
        )
        self.tree.heading("Model", text="Available Models")
        self.tree.column("Model", anchor="w", stretch=True)
        self.tree.pack(side="left", fill="both", expand=True)

        sb = ttk.Scrollbar(self, orient="vertical", command=self.tree.yview)
        sb.pack(side="right", fill="y")
        self.tree.config(yscrollcommand=sb.set)

        self.load_models()

    def load_models(self):
        import os
        os.makedirs(self.models_dir, exist_ok=True)
        for model in os.listdir(self.models_dir):
            if model.endswith(".pth"):
                self.tree.insert("", "end", values=(model,))

    def get_selected_model(self):
        sel = self.tree.selection()
        return self.tree.item(sel[0])["values"][0] if sel else None


class HistoryTextBox(ttk.Frame):
    """Scrollable log panel with color-coded tags, timestamps, and a status bar."""

    _TAG_COLORS = {
        "info":    P["muted"],
        "tts":     P["accent"],
        "ocr":     P["ocr"],
        "force":   P["force"],
        "error":   P["error"],
        "success": P["success"],
        "warn":    P["warn"],
    }

    def __init__(self, parent):
        super().__init__(parent)

        # Status bar — packed first so it anchors at the bottom
        tk.Frame(self, bg=P["border"], height=1).pack(side="bottom", fill="x")

        self._status_bar = tk.Frame(self, bg=P["panel"], height=24)
        self._status_bar.pack(side="bottom", fill="x")
        self._status_bar.pack_propagate(False)

        self._status_lbl = tk.Label(
            self._status_bar, text="● Idle",
            fg=P["muted"], bg=P["panel"],
            font=("Segoe UI", 8), anchor="w",
        )
        self._status_lbl.pack(side="left", padx=8, pady=4)

        self._time_lbl = tk.Label(
            self._status_bar, text="",
            fg=P["muted"], bg=P["panel"],
            font=("Consolas", 8), anchor="e",
        )
        self._time_lbl.pack(side="right", padx=8, pady=4)

        # Log text area
        inner = tk.Frame(self, bg=P["panel"])
        inner.pack(fill="both", expand=True)

        self.text = tk.Text(
            inner,
            wrap="word",
            state="disabled",
            bg=P["surface"],
            fg=P["text"],
            insertbackground=P["text"],
            relief="flat",
            borderwidth=0,
            font=("Consolas", 9),
            padx=10, pady=8,
            cursor="arrow",
            selectbackground=P["accent"],
            selectforeground=P["bg"],
        )
        sb = ttk.Scrollbar(inner, orient="vertical", command=self.text.yview)
        self.text.config(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.text.pack(side="left", fill="both", expand=True)

        # Register color tags
        for tag, color in self._TAG_COLORS.items():
            self.text.tag_configure(tag, foreground=color)
        # Timestamp tag — dimmed, slightly smaller
        self.text.tag_configure("ts", foreground=P["muted"], font=("Consolas", 8))

    def append_text(self, msg, tag="info"):
        ts = _time.strftime("%H:%M:%S")
        self.text.config(state="normal")
        self.text.insert("end", f"{ts}  ", "ts")
        self.text.insert("end", f"{msg}\n", tag)
        self.text.see("end")
        self.text.config(state="disabled")
        self._time_lbl.config(text=ts)

    def set_status(self, state):
        label, color = STATUS_CONFIG.get(state, ("● Idle", P["muted"]))
        self._status_lbl.config(text=label, fg=color)


class PlaybackSlider(ttk.Frame):
    def __init__(self, parent, from_=0.5, to=2.0, initial=1.0):
        super().__init__(parent)

        tk.Label(
            self, text="Playback Speed",
            fg=P["muted"], bg=P["panel"],
            font=("Segoe UI", 8),
        ).pack(anchor="w")

        self._var = tk.DoubleVar(value=initial)
        ttk.Scale(
            self, from_=from_, to=to,
            variable=self._var, orient="horizontal",
        ).pack(fill="x", pady=(2, 0))

        self._lbl = tk.Label(
            self, text=f"{initial:.2f}×",
            fg=P["accent"], bg=P["panel"],
            font=("Segoe UI", 9, "bold"),
        )
        self._lbl.pack(anchor="center")

        def _update(*_):
            v = round(float(self._var.get()) / 0.05) * 0.05
            self._var.set(v)
            self._lbl.config(text=f"{v:.2f}×")

        self._var.trace_add("write", _update)

    def get(self):
        v = self._var.get()
        return 1.0 / v if v != 0 else 1.0


class ToolTip:
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self._tip = None
        widget.bind("<Enter>", self._show)
        widget.bind("<Leave>", self._hide)

    def _show(self, _=None):
        if self._tip or not self.text:
            return
        try:
            x, y, _, cy = self.widget.bbox("insert")
        except Exception:
            x, y, cy = 0, 0, 0
        x += self.widget.winfo_rootx() + 28
        y += self.widget.winfo_rooty() + cy + 14
        self._tip = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        tk.Label(
            tw, text=self.text, justify="left",
            bg=P["surface"], fg=P["text"],
            relief="flat", borderwidth=0,
            font=("Segoe UI", 8), padx=10, pady=6,
        ).pack()

    def _hide(self, _=None):
        tw, self._tip = self._tip, None
        if tw:
            tw.destroy()


def create_app(app_class):
    root = tk.Tk()
    setup_theme(root)
    app = app_class(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    return root