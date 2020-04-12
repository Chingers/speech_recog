import sounddevice as sd
import soundfile as sf
from os.path import join

sample_rate = 16000
duration = 1 # seconds
folder = "my_sounds"
filename = input("Enter filename: ")
input("Enter to start ")
my_data = sd.rec(int(sample_rate * duration), samplerate=sample_rate,
                channels=1, blocking=True)
print("Done recording")
sd.wait()
sf.write(join(folder, filename + ".wav"), my_data, sample_rate)