import numpy as np
import sounddevice as sd
import torch

import commons
from text import text_to_sequence

SAMPLE_RATE = 22050
SPEAKER_ID = 0

# Silence padding in frames added to each edge of the generated audio.
# Prevents the model from clipping the first/last syllables, which can
# happen even with text-level padding when inference truncates early.
_PAD_FRAMES = int(SAMPLE_RATE * 0.18)   # 180 ms each side
_SILENCE = np.zeros(_PAD_FRAMES, dtype=np.float32)


def generate_audio(text, model, hps, speaker_id=SPEAKER_ID, length_scale=1.1):
    """Convert Japanese text to a float32 numpy audio array via VITS inference."""
    text = text.replace('\n', '').replace('\r', '').replace(' ', '')
    # Underscore pads silence at phoneme level; [JA] tags scope the language.
    # Leading _ and trailing __ give the model a breath before and after.
    text = f"_[JA]{text}__[JA]"

    stn_tst, _ = text_to_sequence(text, hps.symbols, hps.data.text_cleaners)
    if hps.data.add_blank:
        stn_tst = commons.intersperse(stn_tst, 0)

    stn_tst = torch.LongTensor(stn_tst).unsqueeze(0).to(model.device)
    lengths  = torch.LongTensor([stn_tst.size(1)]).to(model.device)
    sid      = torch.LongTensor([speaker_id]).to(model.device)

    with torch.no_grad():
        audio = model.infer(
            stn_tst, lengths, sid=sid,
            noise_scale=0.6, noise_scale_w=0.668,
            length_scale=length_scale,
        )[0][0, 0].data.cpu().float().numpy()

    audio = np.clip(audio, -1.0, 1.0)

    # Numpy-level silence padding: catches any remaining edge clipping that
    # the text-level underscores don't fully cover on shorter utterances.
    return np.concatenate([_SILENCE, audio, _SILENCE])


def play_audio(audio):
    """Play a float32 numpy array at the app sample rate, blocking until done."""
    sd.play(audio.astype(np.float32), SAMPLE_RATE)
    sd.wait()