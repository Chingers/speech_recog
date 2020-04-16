import os
import librosa
import numpy as np
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.layers import Dense, Dropout, Flatten, Conv1D, Input, MaxPooling1D
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
from keras.utils import np_utils


labels = ["left", "right", "go", "stop"]

train_audio_path = r'C:\Users\Ramonito\Datasets\speech_commands_train_audio'
new_sampling_rate = 8000

all_waves = []
all_labels = []

for label in labels:
    print(label)
    wave_files = [f for f in os.listdir(train_audio_path + '\\' + label) if f.endswith('.wav')]
    print(len(wave_files))

    for wave_file in wave_files:
        samples, sampling_rate = librosa.load(os.path.join(train_audio_path, label, wave_file), sr=16000)
        samples = librosa.resample(samples, sampling_rate, new_sampling_rate)

        if (len(samples) == 8000):
            all_waves.append(samples)
            all_labels.append(label)

print("total waves: " + str(len(all_waves)))
print("total labels: " + str(len(all_waves)))



le = LabelEncoder()
y=le.fit_transform(all_labels)
classes = list(le.classes_)

y=np_utils.to_categorical(y, num_classes=len(labels))

all_waves = np.array(all_waves).reshape(-1, 8000, 1)

x_tr, x_val, y_tr, y_val = train_test_split(np.array(all_waves), np.array(y), stratify=y, test_size=0.2,
                                            random_state=777, shuffle=True)

K.clear_session()

inputs = Input(shape=(8000,1))

#First Conv1D layer
conv = Conv1D(8,13, padding='valid', activation='relu', strides=1)(inputs)
conv = MaxPooling1D(3)(conv)
conv = Dropout(0.3)(conv)

#Second Conv1D layer
conv = Conv1D(16, 11, padding='valid', activation='relu', strides=1)(conv)
conv = MaxPooling1D(3)(conv)
conv = Dropout(0.3)(conv)

#Third Conv1D layer
conv = Conv1D(32, 9, padding='valid', activation='relu', strides=1)(conv)
conv = MaxPooling1D(3)(conv)
conv = Dropout(0.3)(conv)

#Fourth Conv1D layer
conv = Conv1D(64, 7, padding='valid', activation='relu', strides=1)(conv)
conv = MaxPooling1D(3)(conv)
conv = Dropout(0.3)(conv)

#Flatten layer
conv = Flatten()(conv)

#Dense Layer 1
conv = Dense(256, activation='relu')(conv)
conv = Dropout(0.3)(conv)

#Dense Layer 2
conv = Dense(128, activation='relu')(conv)
conv = Dropout(0.3)(conv)

outputs = Dense(len(labels), activation='softmax')(conv)

model = Model(inputs, outputs)
model.summary()
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10, min_delta=0.0001)
mc = ModelCheckpoint('best_model.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='max')

history = model.fit(x_tr, y_tr, epochs=10, callbacks=[es,mc], batch_size=32, validation_data=(x_val,y_val))

pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()


