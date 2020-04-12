import sounddevice as sd
import soundfile as sf
from os.path import join

samplerate = 16000
duration = 1 # seconds
folder = "my_sounds"
filename = input("Enter filename: ")
input("Enter to start ")
mydata = sd.rec(int(samplerate * duration), samplerate=samplerate,
    channels=1, blocking=True)
print("Done recording")
sd.wait()
sf.write(join(folder, filename + ".wav"), mydata, samplerate)