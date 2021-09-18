import librosa
import numpy as np

import settings as s

def wav_to_mfcc(audio, n_mfcc=20, max_len=s.img_width):
    wave = np.asfortranarray(audio[::2])
    mfcc = librosa.feature.mfcc(wave, sr=16000, n_mfcc=n_mfcc)
    mfcc = mfcc[1:]
    # If maximum length exceeds mfcc lengths then pad the remaining ones
    if (max_len > mfcc.shape[1]):
        pad_width = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')

    # Else cutoff the remaining parts
    else:
        mfcc = mfcc[:, :max_len]

    return mfcc
