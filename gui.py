import os
import tkinter as tk
from tkinter import ttk, messagebox

# --- Individual widget classes ---

class SignEntry(ttk.Frame):
    """A widget for entering a sign (opening or closing)."""
    def __init__(self, parent, label_text, default_value="", **kwargs):
        super().__init__(parent, **kwargs)
        self.label = ttk.Label(self, text=label_text)
        self.label.pack(side="top", anchor="w", pady=(0, 2))
        self.entry = ttk.Entry(self)
        self.entry.pack(side="top", fill="x")
        self.entry.insert(0, default_value)
        
    def get(self):
        return self.entry.get()

class ModelTreeView(ttk.Frame):
    """A widget that wraps a Treeview to show available models from the /models folder."""
    def __init__(self, parent, models_dir="models", **kwargs):
        super().__init__(parent, **kwargs)
        self.models_dir = models_dir
        self.tree = ttk.Treeview(self, columns=("Model",), show="headings", selectmode="browse")
        self.tree.heading("Model", text="Model", anchor="w")
        self.tree.pack(side="left", fill="both", expand=True)
        
        # Vertical scrollbar
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.tree.yview)
        scrollbar.pack(side="right", fill="y")
        self.tree.configure(yscrollcommand=scrollbar.set)

        self.load_models()

    def load_models(self):
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)

        model_files = [f for f in os.listdir(self.models_dir) if f.endswith(".pth")]

        for model in model_files:
            self.tree.insert("", "end", values=(model,))
    
    def get_selected_model(self):
        selected = self.tree.selection()
        if selected:
            return self.tree.item(selected[0])["values"][0]
        return None

class HistoryTextBox(ttk.Frame):
    """A read-only text widget to display clipboard history."""
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self.text = tk.Text(self, wrap="word", state="disabled")
        self.text.pack(side="left", fill="both", expand=True)
        
        # Vertical scrollbar
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.text.yview)
        scrollbar.pack(side="right", fill="y")
        self.text.configure(yscrollcommand=scrollbar.set)
        
    def append_text(self, new_text):
        self.text.configure(state="normal")
        self.text.insert("end", new_text + "\n")
        self.text.configure(state="disabled")

class PlaybackSlider(ttk.Frame):
    """A widget wrapping a Scale for playback speed with a label."""
    def __init__(self, parent, from_=0.5, to=2.0, initial=1.0, **kwargs):
        super().__init__(parent, **kwargs)
        self.label = ttk.Label(self, text="Playback Speed:")
        self.label.pack(side="top", anchor="w")
        self.speed_var = tk.DoubleVar(value=initial)
        self.scale = ttk.Scale(self, variable=self.speed_var, from_=from_, to=to, orient="horizontal", command=self._update_label)
        self.scale.pack(side="top", fill="x")
        self.value_label = ttk.Label(self, text=f"{initial:.2f}")
        self.value_label.pack(side="top", anchor="w", pady=(2, 0))
        
    def _update_label(self, event):
        self.value_label.config(text=f"{self.speed_var.get():.2f}")
        
    def get(self):
        return self.speed_var.get()

# --- Main Application Layout ---

class KikiYomuApp:
    def __init__(self, root):
        self.root = root
        self.root.title("KikiYomu")
        self.root.geometry("900x400")
        
        # Configure the grid to have three vertical sections
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)  # Left section
        self.root.grid_columnconfigure(1, weight=2)  # Middle section
        self.root.grid_columnconfigure(2, weight=1)  # Right section
        
        # Left Section: Model selection
        self.left_frame = ttk.Frame(root, padding=10, relief="groove")
        self.left_frame.grid(row=0, column=0, sticky="nsew")
        ttk.Label(self.left_frame, text="Models", font=("Segoe UI", 10, "bold")).pack(anchor="nw")
        self.model_tree = ModelTreeView(self.left_frame)
        self.model_tree.pack(fill="both", expand=True, pady=(5, 5))
        self.select_model_btn = ttk.Button(self.left_frame, text="Select Model", command=self.select_model)
        self.select_model_btn.pack(fill="x", pady=(5, 0))
        
        # Middle Section: Clipboard history
        self.middle_frame = ttk.Frame(root, padding=10, relief="groove")
        self.middle_frame.grid(row=0, column=1, sticky="nsew")
        ttk.Label(self.middle_frame, text="Clipboard History", font=("Segoe UI", 10, "bold")).pack(anchor="nw")
        self.history_text = HistoryTextBox(self.middle_frame)
        self.history_text.pack(fill="both", expand=True, pady=(5, 5))
        
        # Right Section: Options
        self.right_frame = ttk.Frame(root, padding=10, relief="groove")
        self.right_frame.grid(row=0, column=2, sticky="nsew")
        ttk.Label(self.right_frame, text="Options", font=("Segoe UI", 10, "bold")).pack(anchor="nw")
        
        # Use SignEntry class for opening and closing signs
        self.opening_sign = SignEntry(self.right_frame, label_text="Opening Sign:", default_value="[")
        self.opening_sign.pack(fill="x", pady=(10, 5))
        self.closing_sign = SignEntry(self.right_frame, label_text="Closing Sign:", default_value="]")
        self.closing_sign.pack(fill="x", pady=(10, 5))
        
        # Playback slider
        self.playback_slider = PlaybackSlider(self.right_frame, from_=0.5, to=2.0, initial=1.0)
        self.playback_slider.pack(fill="x", pady=(10, 5))
        
    def select_model(self):
        model = self.model_tree.get_selected_model()
        if model:
            self.history_text.append_text(f"Selected Model: {model}")
        else:
            self.history_text.append_text("No model selected. Please select a model from the list.")

def main():
    root = tk.Tk()
    # Optionally set a native Windows theme if available
    style = ttk.Style()
    if 'winnative' in style.theme_names():
        style.theme_use('winnative')
    app = KikiYomuApp(root)
    root.mainloop()

if __name__ == '__main__':
    main()
