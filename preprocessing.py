#filter accuracy
import librosa
import os
import numpy as np
import librosa.display
import matplotlib.pyplot as plt
import csv
from scipy.signal import savgol_filter
from scipy import signal
from sklearn.cluster import KMeans
from sklearn.preprocessing import minmax_scale
from scipy.signal import find_peaks
from sklearn.metrics import classification_report, precision_recall_fscore_support, confusion_matrix, accuracy_score


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



n_fft = 1764
hop_length = 882

data_folder = './24hour_data/'
label_folder = './24hour_labels/'

user_folders = [folder for folder in os.listdir(data_folder) if folder.startswith('P')]
user_folders = ['P12']

all_predictions = []


#get 5s windows with 1s overlap

	audio_filename = data_folder + episode + '.wav'

	y, sr = librosa.load(audio_filename, offset=0, duration=86400)
	duration = librosa.get_duration(y=y, sr=sr)

	sos = signal.butter(10, 350, 'hp', fs=sr, output='sos')
	y  = signal.sosfilt(sos, y)

	S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=None, n_fft = n_fft, hop_length = hop_length)
	print("Spectrogram loaded")
	#print(S.shape)
	S_dB = librosa.power_to_db(S, ref = np.max)
	S[np.where(S_dB < -78)] = 0
	S_sum = np.transpose(np.sum(S, axis = 0))
	#print(S_sum.shape)
	if len(S_sum) >= 121:
		filted = savgol_filter(S_sum, 121, 5)
	else:
		filted = savgol_filter(S_sum, len(S_sum), 3)
	filted = np.asarray([1 if x > 0.05 else 0 for x in filted ])


	timed_filted = np.stack([np.arange(len(filted)), filted], axis = 1)
	timed_filted = combineIntoEvent(timed_filted, 5 / (hop_length * 1. /sr))
	timed_filted = whatIsAnEvent(timed_filted, 5 / (hop_length * 1. /sr))
	predictions = []
	print("Smoothed")

	pointer = 1
	temp = []

	for ind, value in enumerate(np.arange(0, librosa.get_duration(y=y, sr=sr) + hop_length * 1. / sr, hop_length * 1. / sr)):
		if ind < len(timed_filted):
			if value < pointer:
				temp.append(timed_filted[ind, 1])
			else:
				if sum(temp) > 0:
					predictions.append(1)
				else:
					predictions.append(0)
				temp = [timed_filted[ind, 1]]
				pointer += 1
	print(len(predictions), len(ra_annotations))
	begin = False
	start_time = 0
	prv_label = None
	output = []
	for ind, item in enumerate(predictions):
		if item == 1:
			if begin:
				if ra_annotations[ind] != prv_label:
					output.append([start_time, ind, prv_label])
					start_time = ind
					prv_label = ra_annotations[ind]
			else:
				start_time = ind
				prv_label = ra_annotations[ind]
				begin = True
		else:
			if begin:
				output.append([start_time, ind, prv_label])
				begin = False
	if begin:
		output.append([start_time, len(predictions), prv_label])


	if not os.path.exists('./24hour_labels_filtered2/' + episode.split("/")[0]):
		os.mkdir('./24hour_labels_filtered2/' + episode.split("/")[0])
	output_folder = './24hour_labels_filtered2/' + episode.split("/")[0] + '/'

	print(output)
	with open(output_folder + episode.split("/")[1] + '.csv', 'w', newline = '') as f:
	    writer = csv.writer(f)
	    writer.writerows(output)
	all_predictions.extend(predictions)

	max_length = min(len(all_labels), len(all_predictions))

	print(confusion_matrix(all_labels[:max_length], all_predictions[:max_length]))

