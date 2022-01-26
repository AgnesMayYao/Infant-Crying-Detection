##Author: Xuewen Yao
#X. Yao, M. Micheletti, M. Johnson, E. Thomaz, and K. de Barbaro, "Infant Crying Detection in Real-World Environments," in ICASSP 2022 (Accepted)
#Expected input is an audio file (audio file format tested is waveform audio file at 22050Hz) and a preprocessed csv file (see preprocessing.py)
#Output is a CSV file with two columns (timestamp-seconds, prediction) predictions of crying (1) or not (0) at each second

import librosa
import os
import numpy as np
import csv
import random
from sklearn.svm import SVC
import h5py
import copy
from pyAudioAnalysis import ShortTermFeatures
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.models import Sequential, model_from_json, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras import backend as K



def whatIsAnEvent(data, event_thre):
	'''
	This functions takes a list of 0/1 with its timestamp and remove continous 1s shorter than event_thre
	Input: data contains two columns (timestamp, 0/1)
		   event_thre: the minimun threshold for an event (continuous 1s) to be kept in the output 
	Output: data with continous 1s shorter than event_thre changeed to 0s
	'''
    previous = (-1, -1)
    start = (-1, -1)
    for i in range(len(data)):
        if data[i, 1] == 1 and previous[1] == -1:
            previous = (i, data[i, 0])
        elif data[i, 1] == 0 and previous[1] != -1 and data[i - 1, 1] == 1:
            start = (i, data[i, 0])
            if start[1] - previous[1] <= event_thre:
                data[previous[0] : start[0], 1] = 0
            previous = (-1, -1)
            start = (-1, -1)

    if previous[1] != -1 and data[-1, 0] - previous[1] + 1 <= event_thre:
        data[previous[0] :, 1] = 0
    return data


def combineIntoEvent(data, time_thre):
	'''
	This functions takes a list of 0/1 with its timestamp and combine neighbouring 1s within a time_thre
	Input: data contains two columns (timestamp, 0/1)
		   time_thre: the maximun threshold for neighbouring events (continuous 1s) to be combined in the output (0s between them become 1s)
	Output: data with continous 0s shorter than time_thre changed to 1s
	'''
    previous = (-1, -1)
    for i in range(len(data)):
        if data[i, 1] == 1:
            start = (i, data[i, 0])
            if previous[1] > 0 and start[1] - previous[1] <= time_thre:
                data[previous[0] : start[0], 1] = 1
            previous = start

    if previous[1] > 0 and data[i - 1, 0] - previous[1] <= time_thre:
        data[previous[0] : i, 1] = 1

    return data




##hyperparameters
n_fft = 980
hop_length = 490
n_mels = 225
img_rows, img_cols = 225, 225
batch_size = 128
num_classes = 2


##files
preprocessed_file = "preprocessed.csv"
audio_filename = "P34_2.wav"
output_file = "predictions.csv"


##trained models
#deep_spectrum.h5 can be found at https://utexas.box.com/s/64ecwy5wo0zzla4sax3j30dog0f4k8kv
#deep_spectrum.h5 is a complete AlexNet, but we use the second-to-last layer as deep spectrum features instead of the last prediction layer
saved_model1 = load_model('deep_spectrum.h5')
model1 = Sequential()
for layer in saved_model1.layers[:-1]:
	model1.add(layer)

for layer in model1.layers:
	layer.trainable = False

#load svm model, svm model is at the same repository
from joblib import dump, load
clf1 = load('svm.joblib')

##read audio file
y, sr = librosa.load(audio_filename, offset = 0)
duration = librosa.get_duration(y = y, sr = sr)


##read preprossed file
##preprocssed file is in format (start_time, end_time) of signals having power higher than 350Hz, like: [[0, 2], [31,55], ..]
##filtered_annotations is in format [0, 1, 1, 1, ...] with length equal to audio file, with 1 meaning signals having power higher than 350Hz at that second
previous = 0
filtered_annotations = []
with open(preprocessed_file, 'r') as csvfile:
	csvreader = csv.reader(csvfile, delimiter=',')
	for row in csvreader:
		if float(row[0]) - previous > 0:
			filtered_annotations.extend([0] * int(float(row[0]) - previous))
		previous = float(row[1])
		filtered_annotations.extend([1] * int(float(row[1]) - float(row[0])))
if duration - previous > 0:
	filtered_annotations.extend([0] * int(duration - previous))


##windowing, 5 second windows with 4 second overlap
##windows = [[0, 5], [1, 6], [2, 7], ...]
windows = []
for i in range(0, int(duration) - 4):
	windows.append([i, i + 5])

##Features extraction to get melspectrograms and all acoustic features from pyAudioAnalysis
S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, fmax=None, n_fft = n_fft, hop_length = hop_length)
S = librosa.power_to_db(S, ref=np.max) + 80
F, _ = ShortTermFeatures.feature_extraction(y, sr, 1 * sr, 0.5 * sr)
F = F[:, 0::2]


##image_windows contain melspectrograms for each 5-second window
##feature_windows contain acoustic features (mean, median, std) for each 5-second window
image_windows = []
feature_windows = []
for item in windows:
	image_windows.append(S[:, int(item[0] * sr / hop_length) : int(item[1] * sr / hop_length)])
	F_window = F[:, item[0] : item[1]]
	F_feature = np.concatenate((np.mean(F_window, axis = 1), np.median(F_window, axis = 1), np.std(F_window, axis = 1)), axis = None)
	feature_windows.append(F_feature)


##preprocess before put 5-second melspectrograms into deep spectrum (AlexNet) model
image_windows = np.asarray(image_windows)
image_windows = image_windows.reshape(image_windows.shape[0], img_rows, img_cols, 1) 
image_windows = image_windows.astype('float32')
image_windows /= 80.0

##image_features is deep spectrum features from the model with melspectrograms as input
image_features = model1.predict(image_windows, batch_size=batch_size, verbose=1, steps=None)
##concatenate deep spectrum features and acoustic features and make a prediction using trained SVM model
##the prediction is for each 5 second windows with 4 second overlap
svm_test_input = np.concatenate((image_features, feature_windows), axis = 1)
predictions = clf1.predict(svm_test_input)


##using preprocessed file to filter the predictions
##only consider those seconds with power for frequencies higher than 350Hz
##each second can be predicted 5 times because of the window overlap, if it's predicted as crying for at least one time, then it's crying (1), otherwise it's not
for ind, val in enumerate(filtered_annotations):
	if val >= 1:
		min_ind = max(ind - 4, 0)
		max_ind = min(len(predictions), ind + 1)
		if sum(predictions[min_ind : max_ind]) >= 1:
			filtered_annotations[ind] = 1
		else:
			filtered_annotations[ind] = 0

##add a timestamp for predictions
timed_filted = np.stack([np.arange(len(filtered_annotations)), filtered_annotations], axis = 1)
##Combine neighbouring crying(1s) within 5 seconds of each other
timed_filted = combineIntoEvent(timed_filted, 5)
##Remove isolated 1s shorter than 5 seconds
timed_filted = whatIsAnEvent(timed_filted, 5)

##write predictions into a file
print(timed_filted)
with open(output_file, 'w', newline = '') as f:
    writer = csv.writer(f)
    writer.writerows(timed_filted)


		
