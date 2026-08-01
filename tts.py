import numpy as np
import sounddevice as sd
import torch

import commons
from text import text_to_sequence

SAMPLE_RATE = 22050
SPEAKER_ID = 0


def generate_audio(text, model, hps, speaker_id=SPEAKER_ID, length_scale=1.1):
    """Convert Japanese text to a numpy audio array using the loaded VITS model."""
    text = text.replace('\n', '').replace('\r', '').replace(' ', '')
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
    """Play a float32 numpy audio array at the app sample rate."""
    sd.play(audio.astype(np.float32), SAMPLE_RATE)
    sd.wait()