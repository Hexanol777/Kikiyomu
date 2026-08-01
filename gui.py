import tkinter as tk
from tkinter import ttk


class SignEntry(ttk.Frame):
    def __init__(self, parent, label_text, default_value=""):
        super().__init__(parent)
        ttk.Label(self, text=label_text).pack(anchor="w")
        self.entry = ttk.Entry(self)
        self.entry.pack(fill="x")
        self.entry.insert(0, default_value)

    def get(self):
        return self.entry.get()


class ModelTreeView(ttk.Frame):
    def __init__(self, parent, models_dir="models"):
        super().__init__(parent)
        self.models_dir = models_dir

        self.tree = ttk.Treeview(
            self,
            columns=("Model",),
            show="headings",
            selectmode="browse"
        )
        self.tree.heading("Model", text="Model")
        self.tree.pack(side="left", fill="both", expand=True)

        scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.tree.yview)
        scrollbar.pack(side="right", fill="y")
        self.tree.config(yscrollcommand=scrollbar.set)

        self.load_models()

    def load_models(self):
        import os
        os.makedirs(self.models_dir, exist_ok=True)
        for model in os.listdir(self.models_dir):
            if model.endswith(".pth"):
                self.tree.insert("", "end", values=(model,))

    def get_selected_model(self):
        selected = self.tree.selection()
        if selected:
            return self.tree.item(selected[0])["values"][0]
        return None


class HistoryTextBox(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)

        self.text = tk.Text(self, wrap="word", state="disabled")
        self.text.pack(side="left", fill="both", expand=True)

        scrollbar = ttk.Scrollbar(self, command=self.text.yview)
        scrollbar.pack(side="right", fill="y")

        self.text.config(yscrollcommand=scrollbar.set)

    def append_text(self, msg):
        self.text.config(state="normal")
        self.text.insert("end", msg + "\n")
        self.text.see("end")
        self.text.config(state="disabled")


class PlaybackSlider(ttk.Frame):
    def __init__(self, parent, from_=0.5, to=2.0, initial=1.0):
        super().__init__(parent)

        ttk.Label(self, text="Playback Speed:").pack(anchor="w")

        self.var = tk.DoubleVar(value=initial)
        self.scale = ttk.Scale(
            self,
            from_=from_,
            to=to,
            variable=self.var,
            orient="horizontal"
        )
        self.scale.pack(fill="x")

        self.label = ttk.Label(self, text=f"{initial:.2f}")
        self.label.pack(anchor="w")

        def update(*_):
            rounded = round(float(self.var.get()) / 0.05) * 0.05
            self.var.set(rounded)
            self.label.config(text=f"{rounded:.2f}")

        self.var.trace_add("write", update)

    def get(self):
        val = self.var.get()
        return 1.0 / val if val != 0 else 1.0


class ToolTip:
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tip_window = None

        widget.bind("<Enter>", self.show_tip)
        widget.bind("<Leave>", self.hide_tip)

    def show_tip(self, event=None):
        if self.tip_window or not self.text:
            return

        x, y, _cx, cy = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 30
        y += self.widget.winfo_rooty() + cy + 15

        self.tip_window = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")

        label = tk.Label(
            tw,
            text=self.text,
            justify="left",
            background="#ffffe0",
            relief="solid",
            borderwidth=1,
            font=("Segoe UI", 9)
        )
        label.pack(ipadx=1)


    def hide_tip(self, event=None):
        tw = self.tip_window
        self.tip_window = None
        if tw:
            tw.destroy()


def create_app(app_class):
    root = tk.Tk()
    app = app_class(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    return root