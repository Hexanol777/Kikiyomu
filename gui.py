import os
import tkinter as tk
from tkinter import ttk, messagebox
import threading
import time
import pyperclip
import torch
import numpy as np
import sounddevice as sd
import re

from models import SynthesizerTrn
import utils
import commons
from text import text_to_sequence

# --- Configuration ---
CONFIG_PATH = "config/config.json"
SAMPLE_RATE = 22050
SPEAKER_ID = 0


# --- TTS and Clipboard Handling Functions ---

def is_valid_text(text, open_sign="「", close_sign="」"):
    if not text or not isinstance(text, str):
        return False
    
    if len(text.strip()) > 100:
        return False
    
    image_exts = [".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff"]
    if any(ext in text.lower() for ext in image_exts):
        return False
    
    if text.startswith(open_sign) and text.endswith(close_sign):
        return False

    jp_ranges = [
        (0x3040, 0x309F), (0x30A0, 0x30FF), (0x4E00, 0x9FAF),
        (0x3400, 0x4DBF), (0x3000, 0x303F)
    ]
    
    return any(any(start <= ord(c) <= end for start, end in jp_ranges) for c in text[:20])


def generate_audio(text, model, hps, speaker_id, length_scale=1.1):
    text = text.replace('\n', '').replace('\r', '').replace(" ", "")
    
    text = f"_[JA]{text}__[JA]"
    stn_tst, _ = text_to_sequence(text, hps.symbols, hps.data.text_cleaners)
    if hps.data.add_blank:
        stn_tst = commons.intersperse(stn_tst, 0)
    stn_tst = torch.LongTensor(stn_tst).unsqueeze(0).to(model.device)
    lengths = torch.LongTensor([stn_tst.size(1)]).to(model.device)
    sid = torch.LongTensor([speaker_id]).to(model.device)

    with torch.no_grad():
        audio = model.infer(
            stn_tst, lengths, sid=sid,
            noise_scale=0.6, noise_scale_w=0.668,
            length_scale=length_scale
        )[0][0, 0].data.cpu().float().numpy()
    return np.clip(audio, -1.0, 1.0)


def play_audio(audio):
    audio = audio.astype(np.float32)
    sd.play(audio, SAMPLE_RATE)
    sd.wait()


# --- GUI Widgets ---

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
        self.tree = ttk.Treeview(self, columns=("Model",), show="headings", selectmode="browse")
        self.tree.heading("Model", text="Model")
        self.tree.pack(side="left", fill="both", expand=True)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.tree.yview)
        scrollbar.pack(side="right", fill="y")
        self.tree.config(yscrollcommand=scrollbar.set)
        self.load_models()

    def load_models(self):
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
        self.scale = ttk.Scale(self, from_=from_, to=to, variable=self.var, orient="horizontal")
        self.scale.pack(fill="x")
        self.label = ttk.Label(self, text=f"{initial:.2f}")
        self.label.pack(anchor="w")

        def update(val):
            rounded = round(float(self.var.get()) / 0.05) * 0.05
            self.var.set(rounded)
            self.label.config(text=f"{rounded:.2f}")

        self.var.trace("w", lambda *args: update(self.var.get()))

    def get(self):
        return self.var.get()


# --- Main Application ---

class KikiYomuApp:
    def __init__(self, root):
        self.root = root
        self.root.title("KikiYomu")
        self.root.geometry("1000x400")
        self.root.resizable(False, False)

        # Layout frames
        self.root.columnconfigure((0, 1, 2), weight=1)
        self.root.rowconfigure(0, weight=1)

        self.left = ttk.Frame(root, padding=10, relief="groove", borderwidth=2)
        self.middle = ttk.Frame(root, padding=10, relief="groove", borderwidth=2)
        self.right = ttk.Frame(root, padding=10, relief="groove", borderwidth=2)

        self.left.grid(row=0, column=0, sticky="nsew", padx=2, pady=2)
        self.middle.grid(row=0, column=1, sticky="nsew", padx=2, pady=2)
        self.right.grid(row=0, column=2, sticky="nsew", padx=2, pady=2)

        # Left
        ttk.Label(self.left, text="Models", font=("Segoe UI", 10, "bold")).pack()
        self.model_tree = ModelTreeView(self.left)
        self.model_tree.pack(fill="both", expand=True, pady=5)
        ttk.Button(self.left, text="Select Model", command=self.load_model).pack(fill="x")

        # Middle
        ttk.Label(self.middle, text="Clipboard History", font=("Segoe UI", 10, "bold")).pack()
        self.history = HistoryTextBox(self.middle)
        self.history.pack(fill="both", expand=True, pady=5)

        # Right
        ttk.Label(self.right, text="Options", font=("Segoe UI", 10, "bold")).pack()
        self.open_sign = SignEntry(self.right, "Opening Sign:", "「")
        self.open_sign.pack(fill="x", pady=5)
        self.close_sign = SignEntry(self.right, "Closing Sign:", "」")
        self.close_sign.pack(fill="x", pady=5)
        self.playback_slider = PlaybackSlider(self.right)
        self.playback_slider.pack(fill="x", pady=10)

        # Checkbox for removing speaker names
        self.remove_speaker_var = tk.BooleanVar(value=True)
        self.remove_speaker_checkbox = ttk.Checkbutton(
            self.right,
            text="RPGMaker\n WolfRPG",
            variable=self.remove_speaker_var
        )
        self.remove_speaker_checkbox.pack(anchor="w", pady=(10, 0))


        self.model = None
        self.hps = None
        self.last_clip = ""
        self.running = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.start_monitoring()

    def load_model(self):
        model_file = self.model_tree.get_selected_model()
        if not model_file:
            self.history.append_text("No model selected.")
            return

        model_path = os.path.join("models", model_file)
        self.hps = utils.get_hparams_from_file(CONFIG_PATH)
        self.model = SynthesizerTrn(
            len(self.hps.symbols),
            self.hps.data.filter_length // 2 + 1,
            self.hps.train.segment_size // self.hps.data.hop_length,
            n_speakers=self.hps.data.n_speakers,
            **self.hps.model
        ).to(self.device)
        utils.load_checkpoint(model_path, self.model, None)
        self.model.eval()
        self.model.device = self.device
        self.history.append_text(f"Model loaded: {model_file}")

    def remove_speaker_name(self, text):
        """remove 【Speaker】 patterns from the beginning if the checkbox is enabled."""
        if self.remove_speaker_var.get():
            if text.startswith("【"):
                closing_index = text.find("】")
                if closing_index != -1 and closing_index != len(text) - 1:
                    return text[closing_index + 1:].lstrip()
        return text

    def start_monitoring(self):
        def loop():
            self.running = True
            while self.running:
                time.sleep(0.2)
                text = pyperclip.paste()
                if (
                    text != self.last_clip and is_valid_text(
                        text,
                        self.open_sign.get(),
                        self.close_sign.get()
                    )
                ):
                    self.last_clip = text
                    self.history.append_text(text)
                    try:
                        if self.model and self.hps:
                            audio = generate_audio(
                                text, self.model, self.hps, SPEAKER_ID,
                                length_scale=self.playback_slider.get()
                            )
                            play_audio(audio)
                        else:
                            self.history.append_text("Model not loaded.")
                    except Exception as e:
                        self.history.append_text(f"Error: {e}")
                time.sleep(0.2)

        threading.Thread(target=loop, daemon=True).start()


# --- Entry Point ---

def main():
    root = tk.Tk()
    app = KikiYomuApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
