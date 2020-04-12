import os
from os.path import join
import librosa   #for audio processing
import IPython.display as ipd
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile #for audio processing
import warnings
warnings.filterwarnings("ignore")

folder = "my_sounds"
filename = "up.wav"
new_sample_rate = 8000
sample_audio_file = join(folder, filename)
samples, sample_rate = librosa.load(sample_audio_file, sr = 16000)
samples = librosa.resample(samples, sample_rate, new_sample_rate)
fig = plt.figure(figsize=(14, 8))
ax1 = fig.add_subplot(211)
ax1.set_title('Raw wave of ' + sample_audio_file)
ax1.set_xlabel('time')
ax1.set_ylabel('Amplitude')
ax1.plot(np.linspace(0, new_sample_rate/len(samples), new_sample_rate), samples)
plt.show()
