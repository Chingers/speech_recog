import os
import librosa
import numpy as np
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.layers import Dense, Dropout, Flatten, Conv1D, Input, MaxPooling1D, Conv2D, MaxPooling2D, BatchNormalization, Bidirectional, GRU
from keras.models import Model, Sequential
from keras.callbacks import EarlyStopping
from keras.utils import np_utils

from feature_extraction import wav_to_mfcc
import settings as s

def prepare_data(is_mfcc):
    all_waves = []
    all_mfccs = []
    all_labels = []

    for label in s.labels:
        print("Setting up input data for label: " + label)
        wave_files = [f for f in os.listdir(s.train_audio_path + '\\' + label) if f.endswith('.wav')]
        print("Number of training examples: " + str(len(wave_files)))

        for wave_file in wave_files:
            samples, sampling_rate = librosa.load(os.path.join(s.train_audio_path, label, wave_file), sr=16000)

            if(is_mfcc == False):
                new_samples = librosa.resample(samples, sampling_rate, s.new_sampling_rate)

            mfcc = np.array(wav_to_mfcc(samples))

            if (len(samples) == 16000):
                if(is_mfcc):
                    all_mfccs.append(mfcc.reshape(s.img_height, s.img_width, 1))
                else:
                    all_waves.append(new_samples)
                all_labels.append(label)

    print("total waves: " + str(len(all_waves)))
    print("total labels: " + str(len(all_waves)))

    le = LabelEncoder()
    y = le.fit_transform(all_labels)
    y = np_utils.to_categorical(y, num_classes=len(s.labels))

    if not is_mfcc:
        all_waves = np.array(all_waves).reshape(-1, 8000, 1)
        x_tr, x_val, y_tr, y_val = train_test_split(np.array(all_waves), np.array(y), stratify=y, test_size=0.2,
                                                    random_state=777, shuffle=True)
    else:
        x_tr, x_val, y_tr, y_val = train_test_split(np.array(all_mfccs), np.array(y), stratify=y, test_size=0.2,
                                                    random_state=777, shuffle=True)

    return x_tr, x_val, y_tr, y_val


def train_and_save_model(model_num, x_tr, x_val, y_tr, y_val):

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10, min_delta=0.0001)

    if (model_num == 1):
        model = Sequential()
        model.add(Conv2D(32, (3, 3), input_shape=(s.img_height, s.img_width, 1), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.1))
        model.add(Flatten())
        model.add(Dense(100, activation='relu'))
        model.add(Dense(len(s.labels), activation='softmax'))

    elif(model_num == 2):
        model = Sequential()
        model.add(Conv2D(52, (3, 3), input_shape=(s.img_height, s.img_width, 1), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.08))
        model.add(Conv2D(52, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.08))
        model.add(Flatten())
        model.add(Dense(100, activation='relu'))
        model.add(Dense(len(s.labels), activation='softmax'))

    elif(model_num == 3):

        inputs = Input(shape=(s.new_sampling_rate, 1))
        x = BatchNormalization(axis=-1, momentum=0.99, epsilon=1e-3, center=True, scale=True)(inputs)
        #First Conv1D layer
        x = Conv1D(8,13, padding='valid', activation='relu', strides=1)(x)
        x = MaxPooling1D(3)(x)
        x = Dropout(0.3)(x)
        #Second Conv1D layer
        x = Conv1D(16, 11, padding='valid', activation='relu', strides=1)(x)
        x = MaxPooling1D(3)(x)
        x = Dropout(0.3)(x)
        #Third Conv1D layer
        x = Conv1D(32, 9, padding='valid', activation='relu', strides=1)(x)
        x = MaxPooling1D(3)(x)
        x = Dropout(0.3)(x)
        x = BatchNormalization(axis=-1, momentum=0.99, epsilon=1e-3, center=True, scale=True)(x)
        x = Bidirectional(GRU(128, return_sequences=True), merge_mode='sum')(x)
        x = Bidirectional(GRU(128, return_sequences=True), merge_mode='sum')(x)
        x = Bidirectional(GRU(128, return_sequences=False), merge_mode='sum')(x)
        x = BatchNormalization(axis=-1, momentum=0.99, epsilon=1e-3, center=True, scale=True)(x)
        #Dense Layer 1
        x = Dense(256, activation='relu')(x)
        outputs = Dense(len(s.labels), activation="softmax")(x)
        model = Model(inputs, outputs)

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = model.fit(x=x_tr, y=y_tr, epochs=100, callbacks=[es], batch_size=32, validation_data=(x_val, y_val))

    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    pyplot.legend()
    pyplot.show()

    model.save(os.path.join("models", "model_" + str(model_num) + ".h5"))