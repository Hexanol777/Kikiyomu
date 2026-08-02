import tkinter as tk
import customtkinter as ctk
import time as _time
import os

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")

# ── Palette ────────────────────────────────────────────────────────────────────
P = {
    "bg":      "#16171d",
    "panel":   "#1e1f29",
    "surface": "#262733",
    "border":  "#31334a",
    "accent":  "#a78bfa",
    "text":    "#dddde8",
    "muted":   "#636680",
    "success": "#86efac",
    "error":   "#f87171",
    "warn":    "#fbbf24",
    "ocr":     "#67e8f9",
    "force":   "#fb923c",
}

STATUS_CONFIG = {
    "idle":       ("● Idle",        P["muted"]),
    "loading":    ("● Loading…",    P["warn"]),
    "generating": ("● Generating…", P["accent"]),
    "playing":    ("● Playing",     P["success"]),
    "error":      ("● Error",       P["error"]),
}

_FONT_UI   = ("Segoe UI", 12)
_FONT_MONO = ("Consolas", 9)
_FONT_SMALL = ("Segoe UI", 10)


# ── Widgets ────────────────────────────────────────────────────────────────────

class SignEntry(ctk.CTkFrame):
    def __init__(self, parent, label_text, default_value=""):
        super().__init__(parent, fg_color="transparent")
        ctk.CTkLabel(
            self, text=label_text,
            text_color=P["muted"], fg_color="transparent",
            font=_FONT_SMALL, anchor="w",
        ).pack(fill="x")
        self._entry = ctk.CTkEntry(
            self,
            fg_color=P["surface"],
            border_color=P["border"],
            text_color=P["text"],
            border_width=1,
            corner_radius=4,
            font=_FONT_UI,
        )
        self._entry.pack(fill="x", pady=(2, 0), ipady=2)
        self._entry.insert(0, default_value)

    def get(self):
        return self._entry.get()


class ModelListView(ctk.CTkScrollableFrame):
    """Replaces ttk.Treeview — clean clickable rows, no border artifacts."""

    def __init__(self, parent, models_dir="models"):
        super().__init__(
            parent,
            fg_color=P["surface"],
            corner_radius=6,
            scrollbar_button_color=P["border"],
            scrollbar_button_hover_color=P["muted"],
        )
        self._models_dir = models_dir
        self._selected = None
        self._buttons = {}
        self._load()

    def _load(self):
        os.makedirs(self._models_dir, exist_ok=True)
        for model in sorted(os.listdir(self._models_dir)):
            if model.endswith(".pth"):
                btn = ctk.CTkButton(
                    self,
                    text=model,
                    anchor="w",
                    fg_color="transparent",
                    text_color=P["text"],
                    hover_color=P["border"],
                    font=_FONT_UI,
                    corner_radius=4,
                    height=30,
                    command=lambda m=model: self._select(m),
                )
                btn.pack(fill="x", padx=4, pady=1)
                self._buttons[model] = btn

    def _select(self, model):
        if self._selected and self._selected in self._buttons:
            self._buttons[self._selected].configure(
                fg_color="transparent", text_color=P["text"]
            )
        self._selected = model
        self._buttons[model].configure(
            fg_color=P["accent"], text_color=P["bg"]
        )

    def get_selected_model(self):
        return self._selected


class HistoryTextBox(ctk.CTkFrame):
    """Scrollable log with color-coded tags, timestamps, and a live status bar."""

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
        super().__init__(parent, fg_color=P["panel"], corner_radius=6)

        # Status bar anchors at the bottom — pack before expanding area
        tk.Frame(self, bg=P["border"], height=1).pack(side="bottom", fill="x")

        _sb_row = tk.Frame(self, bg=P["panel"], height=26)
        _sb_row.pack(side="bottom", fill="x")
        _sb_row.pack_propagate(False)

        self._status_lbl = tk.Label(
            _sb_row, text="● Idle",
            fg=P["muted"], bg=P["panel"],
            font=("Segoe UI", 8), anchor="w",
        )
        self._status_lbl.pack(side="left", padx=10, pady=4)

        self._time_lbl = tk.Label(
            _sb_row, text="",
            fg=P["muted"], bg=P["panel"],
            font=("Consolas", 8), anchor="e",
        )
        self._time_lbl.pack(side="right", padx=10, pady=4)

        # Log text area
        inner = tk.Frame(self, bg=P["surface"])
        inner.pack(fill="both", expand=True, padx=1, pady=(1, 0))

        self.text = tk.Text(
            inner,
            wrap="word",
            state="disabled",
            bg=P["surface"],
            fg=P["text"],
            insertbackground=P["text"],
            relief="flat",
            borderwidth=0,
            font=_FONT_MONO,
            padx=10, pady=8,
            cursor="arrow",
            selectbackground=P["accent"],
            selectforeground=P["bg"],
        )
        sb = ctk.CTkScrollbar(
            inner,
            command=self.text.yview,
            button_color=P["border"],
            button_hover_color=P["muted"],
            minimum_pixel_length=20,
        )
        self.text.config(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y", pady=2)
        self.text.pack(side="left", fill="both", expand=True)

        for tag, color in self._TAG_COLORS.items():
            self.text.tag_configure(tag, foreground=color)
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


class PlaybackSlider(ctk.CTkFrame):
    def __init__(self, parent, from_=0.5, to=2.0, initial=1.0):
        super().__init__(parent, fg_color="transparent")

        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(fill="x")
        ctk.CTkLabel(
            header, text="Playback Speed",
            text_color=P["muted"], fg_color="transparent",
            font=_FONT_SMALL, anchor="w",
        ).pack(side="left")
        self._lbl = ctk.CTkLabel(
            header, text=f"{initial:.2f}×",
            text_color=P["accent"], fg_color="transparent",
            font=("Segoe UI", 11, "bold"), anchor="e",
            width=50,
        )
        self._lbl.pack(side="right")

        self._var = tk.DoubleVar(value=initial)
        ctk.CTkSlider(
            self,
            from_=from_, to=to,
            variable=self._var,
            button_color=P["accent"],
            button_hover_color=P["accent"],
            progress_color=P["accent"],
            fg_color=P["surface"],
            number_of_steps=30,
        ).pack(fill="x", pady=(4, 0))

        def _update(*_):
            v = round(float(self._var.get()) / 0.05) * 0.05
            self._var.set(v)
            self._lbl.configure(text=f"{v:.2f}×")

        self._var.trace_add("write", _update)

    def get(self):
        v = self._var.get()
        return 1.0 / v if v != 0 else 1.0


class KikiCheckBox(ctk.CTkCheckBox):
    """Checkbox styled as a filled-square indicator in accent color."""

    def __init__(self, parent, text, variable, command=None):
        super().__init__(
            parent,
            text=text,
            variable=variable,
            command=command,
            # Checked state: filled accent square
            fg_color=P["accent"],
            hover_color=P["accent"],
            checkmark_color=P["bg"],   # dark mark on purple = subtle tick
            # Unchecked state
            border_color=P["muted"],
            border_width=2,
            corner_radius=3,           # square-ish
            # Text
            text_color=P["text"],
            font=_FONT_UI,
        )


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
        x = self.widget.winfo_rootx() + self.widget.winfo_width() + 6
        y = self.widget.winfo_rooty() + (self.widget.winfo_height() // 2) - 14
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
    root = ctk.CTk()
    root.configure(fg_color=P["bg"])
    app = app_class(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    return root