import os
import pyaudio
import librosa
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import time
import numpy as np
from queue import Queue

import settings as s
from feature_extraction import wav_to_mfcc

silence_thresh = 200
q = Queue()

le = LabelEncoder()
y = le.fit_transform(s.labels)
classes = list(le.classes_)

def callback(in_data, frame_count, time_info, status):
    data_int = np.frombuffer(in_data, dtype="int16")
    if np.abs(data_int).mean() > silence_thresh:
        data_float = np.frombuffer(in_data,  dtype="float32")
        q.put(data_float)
        time.sleep(1)
        return (in_data, pyaudio.paContinue)
    return (in_data, pyaudio.paContinue)

def predict(audio, model_num, model):
    if (model_num < 3):
        prob = model.predict(audio.reshape(1, s.img_height, s.img_width, 1))
    else:
        prob = model.predict(audio.reshape(1, s.new_sampling_rate, 1))

    index = np.argmax(prob[0])

    return classes[index]

def predict_real_time(model_num):

    print("Welcome to the real time inference program. Here you can speak into the mic and say any of these words:")
    for label in s.labels:
        print("-" + label)

    print("The models will try their best to predict what you are saying!")
    run = input("Enter [s] to start or [Enter] to end: ") == "s"
    model = load_model(os.path.join("models", "model_" + str(model_num) + ".h5"))

    p = pyaudio.PyAudio()

    stream = p.open(format=pyaudio.paFloat32,
                    channels=1,
                    rate=s.sampling_rate,
                    frames_per_buffer=s.sampling_rate,
                    input=True,
                    stream_callback=callback)

    while(run == True):
        stream.start_stream()
        print("Listening...")
        while stream.is_active():
            data = q.get()
            if(model_num < 3):
                mfccs = wav_to_mfcc(data)
                output = predict(mfccs, model_num, model)
            else:
                samples = librosa.resample(data, s.sampling_rate, s.new_sampling_rate)
                output = predict(samples, model_num, model)

            if(output != "noise"):
                print("Is your word: " + output)
                break

        stream.stop_stream()
        run = input("Enter [s] to try again or [Enter] to end: ") == "s"

    stream.close()
