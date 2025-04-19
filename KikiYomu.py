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
SAMPLE_RATE = 22050  # Sample rate for audio playback used by sd

def is_valid_text(text):
    """Check if clipboard content is valid text for TTS"""

    # Skip if text is empty
    if not text or not isinstance(text, str):
        return False
    
    # Skip if text is too long (might be something user decided to copy while program was running)
    if len(text.strip()) > 100:
        return False
        
    # Skip if text contains common image file extensions
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff']
    if any(ext in text.lower() for ext in image_extensions):
        return False
        
    # Skip if text contains opening and closing signs since they are usually voiced (besides the MC :( )
    if text[0] == "「" and text[-1] == "」":
        return False

    # Check for jp chars (hiragana, katakana, kanji and punctuations)
    jp_ranges = [
        (0x3040, 0x309F),  # Hiragana
        (0x30A0, 0x30FF),  # Katakana
        (0x4E00, 0x9FAF),  # Common Kanji
        (0x3400, 0x4DBF),  # Rare Kanji
        (0x3000, 0x303F)   # Japanese punctuation/symbols
    ]
    
    # Check each character in the text
    for char in text[:20]:  # Only check first 20 chars for performance
        char_code = ord(char)
        for start, end in jp_ranges:
            if start <= char_code <= end:
                return True  # Found at least one Japanese character
    
    return False  # No Japanese characters found
    

def generate_audio(text, model, speaker_id, 
                   noise_scale=0.6,
                   noise_scale_w=0.668, 
                   length_scale=1.1): # Playback speed
    
    # Preprocess text
    text = text.replace('\n', ' ').replace('\r', '').replace(" ", "")
    text = f"_[JA]{text}__[JA]"  # Wrap with Japanese tags
    
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

        # Skip if clipboard unchanged or invalid
        if current_clipboard_content == last_clipboard_content or not is_valid_text(current_clipboard_content):
            time.sleep(0.2)
            continue

        else:
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