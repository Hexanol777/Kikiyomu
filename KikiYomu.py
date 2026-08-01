import tkinter as tk
from tkinter import ttk

import threading
import time
import os

import pyperclip
import torch

from models import SynthesizerTrn
import utils

from pynput import keyboard as pynput_keyboard

from ocr import get_clipboard_image, OCR
from tts import generate_audio, play_audio, SPEAKER_ID
from processor import (
    is_valid_text,
    remove_speaker_name,
    remove_consecutive_kanji_duplicates,
    collapse_repetitions,
    word_filter,
)
from gui import (
    SignEntry,
    ModelTreeView,
    HistoryTextBox,
    PlaybackSlider,
    ToolTip,
    create_app,
)


# --- Configuration ---

BASE_DIR = os.path.dirname(__file__)
CONFIG_PATH = os.path.join(BASE_DIR, "config", "config.json")
ICON = os.path.join(BASE_DIR, "config", "icon.png")


# --- Application ---

class KikiYomuApp:
    def __init__(self, root):
        self.root = root
        self.root.title("KikiYomu")
        self.root.geometry("1000x450")
        self.root.resizable(False, False)

        # Layout
        self.root.columnconfigure(0, weight=7)
        self.root.columnconfigure(1, weight=3)
        self.root.columnconfigure(2, weight=5)
        self.root.rowconfigure(0, weight=1)

        self.left   = ttk.Frame(root, padding=10, relief="groove", borderwidth=2)
        self.middle = ttk.Frame(root, padding=10, relief="groove", borderwidth=2)
        self.right  = ttk.Frame(root, padding=10, relief="groove", borderwidth=2)

        self.left.grid(  row=0, column=0, sticky="nsew", padx=2, pady=2)
        self.middle.grid(row=0, column=1, sticky="nsew", padx=2, pady=2)
        self.right.grid( row=0, column=2, sticky="nsew", padx=2, pady=2)

        # Left — model selector
        ttk.Label(self.left, text="Models", font=("Segoe UI", 10, "bold")).pack()
        self.model_tree = ModelTreeView(self.left)
        self.model_tree.pack(fill="both", expand=True, pady=5)
        ttk.Button(self.left, text="Select Model", command=self.load_model).pack(fill="x")

        # Middle — log
        ttk.Label(self.middle, text="Log", font=("Segoe UI", 10, "bold")).pack()
        self.history = HistoryTextBox(self.middle)
        self.history.pack(fill="both", expand=True, pady=5)

        # Right — options
        ttk.Label(self.right, text="Options", font=("Segoe UI", 10, "bold")).pack()

        self.open_sign = SignEntry(self.right, "Opening Sign:", "「")
        self.open_sign.pack(fill="x", pady=5)
        ToolTip(self.open_sign, "Character marking the start of spoken dialogue. Default: 「")

        self.close_sign = SignEntry(self.right, "Closing Sign:", "」")
        self.close_sign.pack(fill="x", pady=5)
        ToolTip(self.close_sign, "Character marking the end of spoken dialogue. Default: 」")

        self.playback_slider = PlaybackSlider(self.right)
        self.playback_slider.pack(fill="x", pady=10)

        self.remove_speaker_var = tk.BooleanVar(value=False)
        self.remove_speaker_checkbox = ttk.Checkbutton(
            self.right, text="RPGMaker\n WolfRPG",
            variable=self.remove_speaker_var
        )
        self.remove_speaker_checkbox.pack(anchor="w", pady=(10, 0))
        ToolTip(self.remove_speaker_checkbox, "Removes 【Name】 speaker tags from dialogue.")

        self.ocr_var = tk.BooleanVar(value=False)
        self.ocr_checkbox = ttk.Checkbutton(
            self.right, text="Image OCR",
            variable=self.ocr_var
        )
        self.ocr_checkbox.pack(anchor="w", pady=(10, 0))
        ToolTip(self.ocr_checkbox, "Extracts Japanese text from clipboard images using OCR.")

        self.remove_repetition_var = tk.BooleanVar(value=False)
        self.remove_repetition_checkbox = ttk.Checkbutton(
            self.right, text="Repeated\nText Filter",
            variable=self.remove_repetition_var,
            command=self._toggle_word_filter_entry
        )
        self.remove_repetition_checkbox.pack(anchor="w", pady=(10, 0))
        ToolTip(self.remove_repetition_checkbox, "Removes repetitions from extracted text. Use when Textractor can't filter them.")

        self.custom_filter_label = ttk.Label(self.right, text="Words to filter\n(comma separated):")
        self.custom_filter_entry = tk.Text(self.right, height=3, width=25)
        # Hidden by default — revealed when repetition filter is toggled on
        self.custom_filter_label.pack_forget()
        self.custom_filter_entry.pack_forget()

        # State
        self.model = None
        self.hps = None
        self.last_clip = ""
        self.running = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.start_key_listener()
        self.start_monitoring()

    # --- Model ---

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

    # --- TTS event handlers ---

    def force_read(self, text):
        """Synthesize and play text unconditionally (hotkey-triggered)."""
        if self.model and self.hps:
            self.history.append_text(f"[Force Read]: {text}")
            audio = generate_audio(
                text, self.model, self.hps, SPEAKER_ID,
                length_scale=self.playback_slider.get()
            )
            play_audio(audio)
        else:
            self.history.append_text("[Force Read]: Model not loaded.")

    def on_force_read(self, event=None):
        """Hotkey handler — strips dialogue markers then force-reads."""
        text = pyperclip.paste()
        open_sign  = self.open_sign.get()
        close_sign = self.close_sign.get()
        if text.startswith(open_sign) and text.endswith(close_sign):
            text = text[len(open_sign):-len(close_sign)].strip()
        if is_valid_text(text, open_sign, close_sign):
            self.force_read(text)

    # --- Text processor pipeline ---

    def _get_filter_words(self):
        raw = self.custom_filter_entry.get("1.0", "end").strip()
        return [w.strip() for w in raw.split(",") if w.strip()]

    def _build_processors(self):
        """Return the ordered list of text processor callables for the current settings."""
        return [
            lambda t: remove_speaker_name(t, self.remove_speaker_var.get()),
            remove_consecutive_kanji_duplicates,
            lambda t: collapse_repetitions(t, self.remove_repetition_var.get()),
            lambda t: word_filter(t, self._get_filter_words()),
        ]

    # --- UI helpers ---

    def _toggle_word_filter_entry(self):
        if self.remove_repetition_var.get():
            self.custom_filter_label.pack(anchor="w", pady=(5, 0))
            self.custom_filter_entry.pack(fill="x")
            self.history.append_text("WordFilter Enabled")
        else:
            self.custom_filter_label.pack_forget()
            self.custom_filter_entry.pack_forget()
            self.history.append_text("WordFilter Disabled")

    # --- Hotkey listener ---

    def start_key_listener(self):
        def on_press(key):
            try:
                if key == pynput_keyboard.Key.shift_r:
                    self.root.after(0, self.on_force_read)
            except Exception as e:
                self.history.append_text(f"[KeyListener Error]: {e}")

        self.key_listener = pynput_keyboard.Listener(on_press=on_press)
        self.key_listener.daemon = True
        self.key_listener.start()

    # --- Clipboard monitoring loop ---

    def start_monitoring(self):
        self.history.append_text("Monitoring has started.")
        self.history.append_text(f"Inference device: {self.device}\n")
        self.history.append_text("Available Hotkeys:\n  Right Shift  →  Force read current clipboard text")

        def loop():
            self.running = True
            while self.running:
                time.sleep(0.2)
                text = pyperclip.paste()

                if self.ocr_var.get():
                    image = get_clipboard_image(text)
                    if image:
                        text = OCR(image)
                        self.history.append_text(f"[OCR]: {text}")

                open_sign  = self.open_sign.get()
                close_sign = self.close_sign.get()

                if text != self.last_clip and is_valid_text(text, open_sign, close_sign):
                    self.last_clip = text
                    pyperclip.copy(text)
                    self.history.append_text(text)
                    try:
                        if self.model and self.hps:
                            processed = text
                            for fn in self._build_processors():
                                processed = fn(processed)
                            audio = generate_audio(
                                processed, self.model, self.hps, SPEAKER_ID,
                                length_scale=self.playback_slider.get()
                            )
                            play_audio(audio)
                        else:
                            self.history.append_text("Model not loaded.")
                    except Exception as e:
                        self.history.append_text(f"Error: {e}")

        threading.Thread(target=loop, daemon=True).start()

    # --- Cleanup ---

    def on_close(self):
        try:
            if hasattr(self, "key_listener"):
                self.key_listener.stop()
        except Exception:
            pass
        self.running = False
        self.root.destroy()


# --- Entry point ---

def main():
    root = create_app(KikiYomuApp)
    root.mainloop()


if __name__ == "__main__":
    main()