import librosa
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import librosa.display
import pandas as pd
import os
from pydub import AudioSegment
from onsetExtraction import onset_detect, onset_detect_non_normalize
from PyQt5.QtCore import QThread, pyqtSignal

def onsetDetection(x, sr, hop_length):
    onset_frames = onset_detect(x, sr=sr, hop_length=hop_length, backtrack=True)
    onset_times_backtrack = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop_length)
    onset_sample = librosa.frames_to_samples(onset_frames, hop_length=hop_length)
    return onset_times_backtrack, onset_sample

def onsetDetectionNonNormalize(x, sr, hop_length):
    onset_frames = onset_detect_non_normalize(x, sr=sr, hop_length=hop_length, backtrack=True)
    onset_times_backtrack = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop_length)
    onset_sample = librosa.frames_to_samples(onset_frames, hop_length=hop_length)
    return onset_times_backtrack, onset_sample

def plotingWave(x, sr, filename, hop_size, onset_times_backtrack, normalize = False):
    plt.figure(figsize=(14, 5))
    librosa.display.waveplot(y=x, sr=sr)
    plt.vlines(onset_times_backtrack, -0.8, 0.79, color='r', alpha=0.8)
    red_patch = mpatches.Patch(color='red', label='Onset')
    plt.legend(handles=[red_patch])
    plt.xlabel('total onset : {}'.format(len(onset_times_backtrack)))
    if not os.path.isdir("figure"):
        os.mkdir("figure")
    if (normalize):
        plt.title('hop_size : {} dan di normalisasi'.format(hop_size))
        plt.savefig('figure/onset.png'.format(hop_size))
    else:
        plt.title('hop_size : {} tanpa normalisasi'.format(hop_size))
        plt.savefig('figure/onset.png'.format(hop_size))

def convertTimesToSecond(onset_times_backtrack):
    onset_times_in_milliseconds = []
    for value in onset_times_backtrack:
        onset_times_in_milliseconds.append(value * 1000)

    return onset_times_in_milliseconds

def makeDataFame(hop_size,onset_times_in_milliseconds):
    data = pd.read_csv('data.csv')
    columns = "hop_sise = {}".format(hop_size)
    data[hop_size] = onset_times_in_milliseconds
    data.to_csv('data.csv')

def makeWav(filename, onset_times_backtrack):
    onset_times_in_milliseconds = convertTimesToSecond(onset_times_backtrack)
    if not os.path.isdir("split_audio"):
        os.mkdir("split_audio")
    audio = AudioSegment.from_file(filename)
    lengthaudio = len(audio)
    # print("Length of Audio File", lengthaudio)

    start = onset_times_in_milliseconds[0]
    # In Milliseconds, this will cut 10 Sec of audio
    threshold = 0  # (1 Sec = 1000 milliseconds)
    end = 0
    counter = 0
    index = 1
    for threshold in onset_times_in_milliseconds:
        if start == threshold:
            pass
        else:
            end = threshold
            # print(start, end)
            counter = threshold
            chunk = audio[start:end]
            filename = 'split_audio/{}-potongan-{}.wav'.format(index, counter)
            chunk.export(filename, format="wav")
            start = threshold
            index+=1

# filename = 'dataset/data_test/orang1_65bpm_pola1_3.wav'
# x, sr = librosa.load(filename)
# hop_length = 110
#
# onset_times_backtrack = onsetDetection(x, sr, hop_length)
# plotingWave(x, sr, filename, hop_length, onset_times_backtrack)
# makeWav(filename, onset_times_backtrack)
# # makeDataFame(hop_length, onset_times_backtrack)
# onset_times_backtrack_non_normalize = onsetDetectionNonNormalize(x, sr, hop_length)
# plotingWave(filename, hop_length, onset_times_backtrack_non_normalize, normalize=False)