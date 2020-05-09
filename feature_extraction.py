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

# train_audio_path = r'C:\Users\Ramonito\Datasets\speech_commands_train_audio'
# new_sampling_rate = 8000
# labels = ["left", "right", "go", "stop"]
# counter = 1
# for label in labels:
#
#     print(label)
#     wave_files = [f for f in os.listdir(train_audio_path + '\\' + label) if f.endswith('.wav')]
#     print(len(wave_files))
#
#     for wave_file in wave_files:
#         if counter == 1:
#             samples, sampling_rate = librosa.load(os.path.join(train_audio_path, label, wave_file), sr=16000)
#             samples = librosa.resample(samples, sampling_rate, new_sampling_rate)
#             mfcc = wav2mfcc(os.path.join(train_audio_path, label, wave_file))
#             print("")

