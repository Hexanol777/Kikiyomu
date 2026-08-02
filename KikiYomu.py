import tkinter as tk
import customtkinter as ctk

import threading
import time
import os

import pyperclip
import torch
import sounddevice as sd

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
    ModelListView,
    HistoryTextBox,
    PlaybackSlider,
    KikiCheckBox,
    ToolTip,
    create_app,
    P,
)


BASE_DIR   = os.path.dirname(__file__)
CONFIG_PATH = os.path.join(BASE_DIR, "config", "config.json")


class KikiYomuApp:
    def __init__(self, root):
        self.root = root
        self.root.title("KikiYomu")
        self.root.geometry("1060x520")
        self.root.resizable(False, False)

        self.root.grid_columnconfigure(0, weight=2, minsize=190)
        self.root.grid_columnconfigure(1, weight=5, minsize=420)
        self.root.grid_columnconfigure(2, weight=3, minsize=240)
        self.root.grid_rowconfigure(0, weight=1)

        self.left   = ctk.CTkFrame(root, fg_color=P["panel"], corner_radius=8)
        self.middle = ctk.CTkFrame(root, fg_color=P["panel"], corner_radius=8)
        self.right  = ctk.CTkFrame(root, fg_color=P["panel"], corner_radius=8)

        self.left.grid(  row=0, column=0, padx=(8, 3), pady=8, sticky="nsew")
        self.middle.grid(row=0, column=1, padx=3,       pady=8, sticky="nsew")
        self.right.grid( row=0, column=2, padx=(3, 8), pady=8, sticky="nsew")

        self._build_left()
        self._build_center()
        self._build_right()

        # State
        self.model    = None
        self.hps      = None
        self.last_clip = ""
        self.running  = False
        self.device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.start_key_listener()
        self.start_monitoring()

    # ── Layout ────────────────────────────────────────────────────────────────

    def _section_label(self, parent, text):
        ctk.CTkLabel(
            parent, text=text,
            text_color=P["muted"], fg_color="transparent",
            font=("Segoe UI", 9, "bold"), anchor="w",
        ).pack(fill="x", padx=12, pady=(10, 6))

    def _build_left(self):
        self._section_label(self.left, "MODELS")

        self.model_list = ModelListView(self.left)
        self.model_list.pack(fill="both", expand=True, padx=8, pady=(0, 8))

        self.load_btn = ctk.CTkButton(
            self.left, text="Load Model",
            fg_color=P["accent"],
            hover_color="#b8a3fb",
            text_color=P["bg"],
            font=("Segoe UI", 12, "bold"),
            corner_radius=6,
            height=34,
            command=self.load_model,
        )
        self.load_btn.pack(fill="x", padx=8, pady=(0, 10))

    def _build_center(self):
        # 3-px accent stripe on the left edge of the log panel
        tk.Frame(self.middle, bg=P["accent"], width=3).pack(side="left", fill="y")

        inner = ctk.CTkFrame(self.middle, fg_color="transparent")
        inner.pack(side="left", fill="both", expand=True, padx=(6, 8), pady=0)

        self._section_label(inner, "ACTIVITY LOG")
        self.history = HistoryTextBox(inner)
        self.history.pack(fill="both", expand=True, pady=(0, 8))

    def _build_right(self):
        self._section_label(self.right, "OPTIONS")

        self.open_sign = SignEntry(self.right, "Opening sign", "「")
        self.open_sign.pack(fill="x", padx=12, pady=(0, 6))
        ToolTip(self.open_sign, "Start-of-dialogue marker.\nDefault: 「")

        self.close_sign = SignEntry(self.right, "Closing sign", "」")
        self.close_sign.pack(fill="x", padx=12, pady=(0, 10))
        ToolTip(self.close_sign, "End-of-dialogue marker.\nDefault: 」")

        self.playback_slider = PlaybackSlider(self.right)
        self.playback_slider.pack(fill="x", padx=12, pady=(0, 12))

        tk.Frame(self.right, bg=P["border"], height=1).pack(fill="x", padx=12, pady=(0, 10))

        self.remove_speaker_var = tk.BooleanVar(value=False)
        cb_rpg = KikiCheckBox(self.right, "RPGMaker / WolfRPG", self.remove_speaker_var)
        cb_rpg.pack(anchor="w", padx=12, pady=(0, 6))
        ToolTip(cb_rpg, "Strips 【Name】 speaker tags\nfrom the start of dialogue lines.")

        self.ocr_var = tk.BooleanVar(value=False)
        cb_ocr = KikiCheckBox(self.right, "Image OCR", self.ocr_var)
        cb_ocr.pack(anchor="w", padx=12, pady=(0, 6))
        ToolTip(cb_ocr, "Extract Japanese text from clipboard images.\nBest used with a snipping tool.")

        self.remove_repetition_var = tk.BooleanVar(value=False)
        cb_rep = KikiCheckBox(
            self.right, "Repeated Text Filter",
            self.remove_repetition_var,
            command=self._toggle_word_filter,
        )
        cb_rep.pack(anchor="w", padx=12, pady=(0, 6))
        ToolTip(cb_rep, "Collapses repeated substrings.\nUse when Textractor can't filter them.")

        self._filter_label = ctk.CTkLabel(
            self.right, text="Words to filter (comma-separated):",
            text_color=P["muted"], fg_color="transparent",
            font=("Segoe UI", 10), anchor="w",
        )
        self.custom_filter_entry = tk.Text(
            self.right, height=3,
            bg=P["surface"], fg=P["text"],
            insertbackground=P["text"],
            relief="flat", borderwidth=0,
            font=("Segoe UI", 10),
            padx=8, pady=6,
        )
        # Hidden until repetition filter is enabled
        self._filter_label.pack_forget()
        self.custom_filter_entry.pack_forget()

    # ── Thread-safe helpers ───────────────────────────────────────────────────

    def _log(self, msg, tag="info"):
        self.root.after(0, lambda m=msg, t=tag: self.history.append_text(m, t))

    def _set_status(self, state):
        self.root.after(0, lambda s=state: self.history.set_status(s))

    # ── Model loading ─────────────────────────────────────────────────────────

    def load_model(self):
        model_file = self.model_list.get_selected_model()
        if not model_file:
            self._log("No model selected.", "warn")
            return

        self.load_btn.configure(state="disabled")
        self._log(f"Loading {model_file}…", "info")
        self._set_status("loading")

        def _load():
            try:
                model_path = os.path.join("models", model_file)
                hps = utils.get_hparams_from_file(CONFIG_PATH)
                model = SynthesizerTrn(
                    len(hps.symbols),
                    hps.data.filter_length // 2 + 1,
                    hps.train.segment_size // hps.data.hop_length,
                    n_speakers=hps.data.n_speakers,
                    **hps.model,
                ).to(self.device)
                utils.load_checkpoint(model_path, model, None)
                model.eval()
                model.device = self.device
                self.hps   = hps
                self.model = model
                self._log(f"Ready — {model_file}", "success")
            except Exception as e:
                self._log(f"Load failed: {e}", "error")
                self._set_status("error")
                time.sleep(2)
            finally:
                self._set_status("idle")
                self.root.after(0, lambda: self.load_btn.configure(state="normal"))

        threading.Thread(target=_load, daemon=True).start()

    # ── TTS handlers ──────────────────────────────────────────────────────────

    def force_read(self, text):
        if not (self.model and self.hps):
            self._log("No model loaded.", "warn")
            return

        def _play():
            try:
                sd.stop()
                self._log(f"[Force] {text}", "force")
                self._set_status("generating")
                audio = generate_audio(
                    text, self.model, self.hps, SPEAKER_ID,
                    length_scale=self.playback_slider.get(),
                )
                self._set_status("playing")
                play_audio(audio)
            except Exception as e:
                self._log(f"Force read error: {e}", "error")
                self._set_status("error")
                time.sleep(1)
            finally:
                self._set_status("idle")

        threading.Thread(target=_play, daemon=True).start()

    def on_force_read(self, event=None):
        text       = pyperclip.paste()
        open_sign  = self.open_sign.get()
        close_sign = self.close_sign.get()
        if text.startswith(open_sign) and text.endswith(close_sign):
            text = text[len(open_sign):-len(close_sign)].strip()
        if is_valid_text(text, open_sign, close_sign):
            self.force_read(text)

    # ── Text processor pipeline ───────────────────────────────────────────────

    def _get_filter_words(self):
        raw = self.custom_filter_entry.get("1.0", "end").strip()
        return [w.strip() for w in raw.split(",") if w.strip()]

    def _build_processors(self):
        return [
            lambda t: remove_speaker_name(t, self.remove_speaker_var.get()),
            remove_consecutive_kanji_duplicates,
            lambda t: collapse_repetitions(t, self.remove_repetition_var.get()),
            lambda t: word_filter(t, self._get_filter_words()),
        ]

    # ── UI helpers ────────────────────────────────────────────────────────────

    def _toggle_word_filter(self):
        if self.remove_repetition_var.get():
            self._filter_label.pack(fill="x", padx=12, pady=(6, 2))
            self.custom_filter_entry.pack(fill="x", padx=12)
            self._log("Word filter enabled.", "info")
        else:
            self._filter_label.pack_forget()
            self.custom_filter_entry.pack_forget()
            self._log("Word filter disabled.", "info")

    # ── Hotkey listener ───────────────────────────────────────────────────────

    def start_key_listener(self):
        def on_press(key):
            try:
                if key == pynput_keyboard.Key.shift_r:
                    self.root.after(0, self.on_force_read)
            except Exception as e:
                self._log(f"[KeyListener] {e}", "error")

        self.key_listener = pynput_keyboard.Listener(on_press=on_press)
        self.key_listener.daemon = True
        self.key_listener.start()

    # ── Clipboard monitoring loop ─────────────────────────────────────────────

    def start_monitoring(self):
        self._log("Monitoring started.", "info")
        self._log(f"Inference device: {self.device}", "info")
        self._log("Right Shift → force-read current clipboard text.", "info")

        def loop():
            self.running = True
            while self.running:
                time.sleep(0.2)
                text       = pyperclip.paste()
                open_sign  = self.open_sign.get()
                close_sign = self.close_sign.get()

                if self.ocr_var.get():
                    image = get_clipboard_image(text)
                    if image:
                        text = OCR(image)
                        self._log(f"[OCR] {text}", "ocr")

                if text != self.last_clip and is_valid_text(text, open_sign, close_sign):
                    self.last_clip = text
                    pyperclip.copy(text)
                    self._log(text, "tts")

                    if self.model and self.hps:
                        try:
                            processed = text
                            for fn in self._build_processors():
                                processed = fn(processed)
                            self._set_status("generating")
                            audio = generate_audio(
                                processed, self.model, self.hps, SPEAKER_ID,
                                length_scale=self.playback_slider.get(),
                            )
                            self._set_status("playing")
                            play_audio(audio)
                        except Exception as e:
                            self._log(f"Error: {e}", "error")
                            self._set_status("error")
                            time.sleep(1)
                        finally:
                            self._set_status("idle")
                    else:
                        self._log("No model loaded.", "warn")

        threading.Thread(target=loop, daemon=True).start()

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def on_close(self):
        try:
            if hasattr(self, "key_listener"):
                self.key_listener.stop()
        except Exception:
            pass
        self.running = False
        self.root.destroy()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    root = create_app(KikiYomuApp)
    root.mainloop()


if __name__ == "__main__":
    main()