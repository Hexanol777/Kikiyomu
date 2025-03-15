import torch
import sounddevice as sd  
import numpy as np
import time
import pyperclip
from models import SynthesizerTrn
import utils
import commons
from text import text_to_sequence

# Configuration
MODEL_PATH = "models/herta.pth"  # Set your model path here
CONFIG_PATH = "config/config.json"  # Set your config path here
SPEAKER_ID = 0  # Set your speaker ID here
LANGUAGE = "JA"  # Always Japanese
SAMPLE_RATE = 22050  # Sample rate for audio playback

def generate_audio(text, model, speaker_id, noise_scale=0.6, noise_scale_w=0.668, length_scale=1.0):
    # Preprocess text
    text = text.replace('\n', ' ').replace('\r', '').replace(" ", "")
    text = f"_[JA]{text}___[JA]"  # Wrap with Japanese tags
    
    # Convert text to sequence
    stn_tst, _ = text_to_sequence(text, hps.symbols, hps.data.text_cleaners)
    if hps.data.add_blank:
        stn_tst = commons.intersperse(stn_tst, 0)
    stn_tst = torch.LongTensor(stn_tst)
    
    # Run inference
    with torch.no_grad():
        x_tst = stn_tst.unsqueeze(0).to(device)
        x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).to(device)
        sid = torch.LongTensor([speaker_id]).to(device)
        
        audio = model.infer(
            x_tst, x_tst_lengths, 
            sid=sid, 
            noise_scale=noise_scale,
            noise_scale_w=noise_scale_w,
            length_scale=length_scale
        )[0][0, 0].data.cpu().float().numpy()
    
    return audio

def play_audio(audio, sample_rate):
    # Ensure audio is in the range [-1, 1]
    audio = np.clip(audio, -1.0, 1.0)  # Clip to avoid invalid values
    audio = audio.astype(np.float32)  # Ensure correct data type for sounddevice
    
    # Play audio
    sd.play(audio, sample_rate)
    sd.wait()  # Wait until the audio is finished playing

def main():
    # Initialize model
    global hps, device
    hps = utils.get_hparams_from_file(CONFIG_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = SynthesizerTrn(
        len(hps.symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model
    ).to(device)
    
    utils.load_checkpoint(MODEL_PATH, model, None)
    model.eval()

    # Clipboard monitoring
    last_clipboard_content = pyperclip.paste()  # Initialize with current clipboard content

    print("Clipboard monitoring started. Copy Japanese text to synthesize and play audio...")

    while True:
        time.sleep(0.2)
        current_clipboard_content = pyperclip.paste()

        if current_clipboard_content != last_clipboard_content:
            # Generate and play audio
            try:
                audio = generate_audio(current_clipboard_content, model, SPEAKER_ID)
                play_audio(audio, SAMPLE_RATE)
            except Exception as e:
                print(f"Error generating or playing audio: {e}")

            last_clipboard_content = current_clipboard_content  # Update last clipboard content

        time.sleep(1)

if __name__ == "__main__":
    main()