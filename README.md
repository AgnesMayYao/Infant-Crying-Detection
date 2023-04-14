# Infant Crying Detection
This repository contains code to run an infant crying detection model for continuous audio in real-world environments, as described in this [paper](https://arxiv.org/abs/2005.07036).

## Citation Information
X. Yao, M. Micheletti, M. Johnson, E. Thomaz, and K. de Barbaro, "Infant Crying Detection in Real-World Environments," in ICASSP 2022 
M. Micheletti, X. Yao, M. Johnson, and K. de Barbaro, "Validating a Model to Detect Infant Crying from Naturalistic Audio," in Behavior Research Methods 2022



## Models and Main Package Versions
Trained deep spectrum model can be found at: https://utexas.box.com/s/poe7i9p9lzs6rg7yer85y2m9wh247lz0. 
Trained SVM model is in this repository: svm.joblib  

### Versions
python3/3.6.3  
tensorflow-gpu==1.15.0  
scikit-learn==0.23.0   
pyAudioAnalysis==0.3.7  
librosa==0.8.1  


# Code
There are two scripts: *preprocessing.py* and *predict.py*.

*preprocessing.py* aims to get rid of the seconds that are definitely not infant crying using frequency information. It reads an audio file and outputs a csv file containing the start_time (seconds) and end_time (seconds) of audio where there is some energy for signals higher than 350Hz. Change input/output filennames here:
```
audio_filename = 'P34_2.wav'
output_file = "preprocessed.csv"
```


*predict.py* gives predictions of crying/not crying at every second. It reads an andio file and the preprocessed csv file and outputs a csv file containing the predictions at each second with its timestamp. Change input/output filenames here:

```
preprocessed_file = "preprocessed.csv"
audio_filename = "P34_2.wav"
output_file = "predictions.csv"
```


## Other resources
HomeBank English deBarbaro Cry Corpus (https://homebank.talkbank.org/access/Password/deBarbaroCry.html)  
	It contains part of the RW-Filt dataset that was used to create the model as 2 out of 24 participants did not give us permission to share their data.  
	To protect the privacy of participants, all crying episodes were cut into five second segments (with four-second overlap between neighboring segments). An equal length and number of five second segments of non-cry data was randomly selected from the same recording. The complete dataset totals 61.3h of labelled data with over seven hours of unique annotated crying data. 



