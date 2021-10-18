import librosa
import os
import numpy as np
import librosa.display
import matplotlib.pyplot as plt
import csv
from scipy.signal import savgol_filter
from sklearn.cluster import KMeans
from sklearn.preprocessing import minmax_scale
from scipy.signal import find_peaks
from sklearn.metrics import classification_report, precision_recall_fscore_support, confusion_matrix, accuracy_score,roc_auc_score
from math import sqrt, pi, exp
from collections import Counter
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
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




n_fft = 980
hop_length = 490
n_mels = 225
img_rows, img_cols = 225, 225
batch_size = 128
num_classes = 2





def time_masking(mel_spectrogram, tau, time_masking_para=100, time_mask_num=2):
	mel_spectrogram = np.asarray(mel_spectrogram)
	for i in range(time_mask_num):
		t = np.random.randint(low = 0, high = time_masking_para)
		t0 = np.random.randint(low = 0, high = tau - t)
		mel_spectrogram[:, t0:(t0 + t)] = 0
	return list(mel_spectrogram)



def whatIsAnEvent(data, event_thre):
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



def label_to_num(input_label):
	if input_label == 'other':
		return 0
	elif input_label == 'fuss':
		return 1
	elif input_label == 'cry':
		return 2
	elif input_label == 'scream':
		return 3
	else:
		return 4



data_folder = './24hour_data/'
label_folder = './24hour_labels_filtered/'
real_label_folder = './24hour_labels/'


test_folders = [folder for folder in os.listdir(data_folder) if folder.startswith('P')]


saved_model1 = load_model('deep_spectrum.h5')

model1 = Sequential()
for layer in saved_model1.layers[:-1]:
	model1.add(layer)

for layer in model1.layers:
	layer.trainable = False

from joblib import dump, load
clf1 = load('svm.joblib')


for test_folder in test_folders:
	test_episodes = []
	episodes = [file for file in os.listdir(data_folder + test_folder) if file.endswith('.wav')]
	for episode in episodes:
		test_episodes.append(test_folder + '/' + episode[:-4])

	#get 5s windows with 1s overlap

	all_groundtruth = []
	all_predictions = []

	for ind, test_episode in enumerate(test_episodes):
		audio_filename = data_folder + test_episode + ".wav"
		annotation_filename_ra =  real_label_folder + test_episode + ".csv"
		annotation_filename_filtered = label_folder + test_episode + ".csv"

		y, sr = librosa.load(audio_filename, offset = 0)
		duration = librosa.get_duration(y = y, sr = sr)
		
		previous = 0
		ra_annotations = []
		with open(annotation_filename_ra, 'r') as csvfile:
			csvreader = csv.reader(csvfile, delimiter=',')
			for row in csvreader:
				if float(row[0]) - previous > 0:
					ra_annotations.extend([0] * int(float(row[0]) - previous))
				previous = float(row[1])
				ra_annotations.extend([1] * int(float(row[1]) - float(row[0])))
		if duration - previous > 0:
			ra_annotations.extend([0] * int(duration - previous))
		print(duration, len(ra_annotations))


		previous = 0
		filtered_annotations = []
		with open(annotation_filename_filtered, 'r') as csvfile:
			csvreader = csv.reader(csvfile, delimiter=',')
			for row in csvreader:
				if float(row[0]) - previous > 0:
					filtered_annotations.extend([0] * int(float(row[0]) - previous))
				previous = float(row[1])
				filtered_annotations.extend([1] * int(float(row[1]) - float(row[0])))
		if duration - previous > 0:
			filtered_annotations.extend([0] * int(duration - previous))


		windows = []
		for i in range(0, int(duration) - 4):
			windows.append([i, i + 5])


		S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, fmax=None, n_fft = n_fft, hop_length = hop_length)
		S = librosa.power_to_db(S, ref=np.max) + 80
		F, _ = ShortTermFeatures.feature_extraction(y, sr, 1 * sr, 0.5 * sr)
		F = F[:, 0::2]

		image_windows = []
		feature_windows = []
		for item in windows:
			image_windows.append(S[:, int(item[0] * sr / hop_length) : int(item[1] * sr / hop_length)])
			F_window = F[:, item[0] : item[1]]
			F_feature = np.concatenate((np.mean(F_window, axis = 1), np.median(F_window, axis = 1), np.std(F_window, axis = 1)), axis = None)
			feature_windows.append(F_feature)

		image_windows = np.asarray(image_windows)
		image_windows = image_windows.reshape(image_windows.shape[0], img_rows, img_cols, 1) 
		image_windows = image_windows.astype('float32')
		image_windows /= 80.0
		
	
		image_features = model1.predict(image_windows, batch_size=batch_size, verbose=1, steps=None)
		svm_test_input = np.concatenate((image_features, feature_windows), axis = 1)
		predictions = clf1.predict(svm_test_input)


		for ind, val in enumerate(filtered_annotations):
			if val >= 1:
				min_ind = max(ind - 4, 0)
				max_ind = min(len(predictions), ind + 1)
				if sum(predictions[min_ind : max_ind]) >= 1:
					filtered_annotations[ind] = 1
				else:
					filtered_annotations[ind] = 0


		timed_filted = np.stack([np.arange(len(filtered_annotations)), filtered_annotations], axis = 1)
		timed_filted = combineIntoEvent(timed_filted, 5 )
		timed_filted = whatIsAnEvent(timed_filted, 5 )

		filtered_annotations = timed_filted[:, 1]
		print(confusion_matrix(ra_annotations, filtered_annotations))
		print(accuracy_score(ra_annotations, filtered_annotations))
		print(classification_report(ra_annotations, filtered_annotations, target_names=['other', 'distress']))



		
