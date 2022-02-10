#Author Name: Xuewen Yao
#X. Yao, M. Micheletti, M. Johnson, E. Thomaz, and K. de Barbaro, "Infant Crying Detection in Real-World Environments," in ICASSP 2022 (Accepted)
#Expected input is an audio file (audio file format tested is waveform audio file at 22050Hz)
#Output is a CSV file with two columns (start_time, end_time) of data that have signals with frequency higher than 350 Hz (potential infant crying)

import librosa
import os
import numpy as np
import librosa.display
import csv
from scipy.signal import savgol_filter
from scipy import signal


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



##Hyperparameters
n_fft = 1764
hop_length = 882
n_mels = 128

##Set your filenames
audio_filename = 'P34_2.wav'
output_file = "preprocessed.csv"


##Read audio file
y, sr = librosa.load(audio_filename)
duration = librosa.get_duration(y=y, sr=sr)

##Highpass filter to remove signals with frequency lower than 350Hz as the mean F0 of infant crying is higher than 400 Hz
sos = signal.butter(10, 350, 'hp', fs=sr, output='sos')
y  = signal.sosfilt(sos, y)

##spectrogram gives the power of signals higher than 350Hz at each time point
S = librosa.feature.melspectrogram(y = y, sr = sr, n_mels = n_mels, fmax=None, n_fft = n_fft, hop_length = hop_length)
S_dB = librosa.power_to_db(S, ref = np.max)

##noise reduction
S[np.where(S_dB < -78)] = 0

##Sum of power of all frequencies (higher than 350Hz) at each time point
S_sum = np.transpose(np.sum(S, axis = 0))

##smoothing
if len(S_sum) >= 121:
	filted = savgol_filter(S_sum, 121, 5)
else:
	filted = savgol_filter(S_sum, len(S_sum), 3)

##1 for those have sum of power of all frequencies (higher than 350Hz) larger than 5, 0 otherwise
filted = np.asarray([1 if x > 5 else 0 for x in filted])

##Time column ([1, 0, 1, 1, 1, ...]->[[0, 1], [1, 0], [2, 1], [3, 1], [4, 1], ...])
timed_filted = np.stack([np.arange(len(filted)), filted], axis = 1)

##Combine neighbouring 1s within 5 seconds of each other
timed_filted = combineIntoEvent(timed_filted, 5 / (hop_length * 1. /sr))
##Remove isolated 1s shorter than 5 seconds
timed_filted = whatIsAnEvent(timed_filted, 5 / (hop_length * 1. /sr))


##change the timestamp from frame number to seconds
##for all frames within a second, if there is a 1, then the second is 0, otherwise 0
##predictions is a list of 1/0 with length equal to the length of audio file in seconds
predictions = []
pointer = 1
temp = []
for ind, value in enumerate(np.arange(0, duration + hop_length * 1. / sr, hop_length * 1. / sr)):
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


##convert predictions to output
##predictions is a list of 1/0 with length equal to the length of audio file in seconds [1, 0, 1, 1, 1, ...]
##output is the start_time and end_time of continous 1s [[0, 1], [2, 5], ...]
begin = False
start_time = 0
output = []
for ind, item in enumerate(predictions):
	if item == 1:
		if not begin:
			start_time = ind
			begin = True
	else:
		if begin:
			output.append([start_time, ind])
			begin = False
if begin:
	output.append([start_time, len(predictions)])


##write output into a file
print(output)
with open(output_file, 'w', newline = '') as f:
    writer = csv.writer(f)
    writer.writerows(output)

