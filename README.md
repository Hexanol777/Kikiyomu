# KikiYomu

**KikiYomu** is a lightweight, real-time Text-to-Speech (TTS) application that monitors your clipboard and instantly uses AI voice models to have anime-style characters narrate Japanese text from games or visual novels.

---

## Features

-  **Real-Time TTS**: Automatically reads aloud Japanese text copied to your clipboard.
-  **Speaker Tag Handling**: Option to remove speaker tags like `【Name】` commonly found in RPGMaker and WolfRPG games.
-  **Game Compatibility**: Designed to work well with most visual novels and games that use stylized dialogue formatting.
-  **User-Friendly GUI**: Simple GUI

---

## Installation

### Prerequisites

- Python 3.8 or later
- [PyTorch](https://pytorch.org/get-started/locally/) (with CUDA if using GPU)
- [SoundDevice](https://python-sounddevice.readthedocs.io/) (for audio playback)

### Setup Instructions

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/KikiYomu.git
   cd KikiYomu

2. **Install Python Dependencies**
```bash
pip install -r requirements.txt
```
3. **Download Pretrained Voice Models**

Visit the following Hugging Face repository to download the Pretrained AI voice models:

[AI-Voice Models](https://huggingface.co/spaces/zomehwh/vits-models/tree/main/pretrained_models)

Place the .pth model files into the models/ directory.

## Usage

1. Start the App
```bash
    python gui.py
```
- Additionally you can just you run the `KikiYomu.py` file in command line as it still offers most utilities.
2. Load a Model

- In the "Models" panel, select a .pth model and click "Select Model".

3. Configure Settings (if needed)

- Set the opening/closing signs used for spoken text (e.g., 「 and 」).

    - If your are playing an RPGMaker game, Enable the checkbox to remove RPGMaker/WolfRPG-style speaker tags (【Name】) at the start of lines.

- Adjust playback speed with the slider.

4. Copy Text to Speak

Copy any Japanese line of text to the clipboard.
If it passes the filters, KikiYomu will automatically speak it aloud using the selected AI voice.

## Credits

- Voice Models: zomehwh's VITS Models on Hugging Face


