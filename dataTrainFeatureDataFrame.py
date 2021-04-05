import pandas as pd
import featureExtraction
import glob
import librosa
import soundfile as sf
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
from tqdm import tqdm
from PyQt5.QtCore import QThread, pyqtSignal
import time
import logging
log = logging


class DataTrainMaker(QThread):
    countChanged = pyqtSignal(int)

    def run(self):
        count = 0
        all_data, len_data = initialize()
        class_feature2 = ['0', '1', '2', '3', '4', '5', '6']
        all_meta_data = []
        index = 0
        for meta_data in all_data:
            for files in meta_data:
                try:
                    y, sr = librosa.load(files)
                    signal, fs = sf.read(files)
                    try:
                        signal = signal[:, 0]
                    except:
                        pass
                    # val_fft = featureExtraction.freq_from_fft(signal, fs)
                    # val_zcr = featureExtraction.freq_from_crossings(signal, fs)
                    val_zcr = float(np.mean(librosa.feature.zero_crossing_rate(y=y).T, axis=0))
                    val_autocorr = featureExtraction.autocorr(y, sr)
                    # val_hps = featureExtraction.freq_from_HPS(signal, fs)
                    # STEs = featureExtraction.computeSTE(fs, signal)
                    STEs = featureExtraction.getSTE(y)

                    stft = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
                    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sr).T, axis=0)
                    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T, axis=0)
                    rms = librosa.feature.rms(y=y)
                    melspectogram = np.mean(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40).T, axis=0)
                    f_contrast = list()
                    for data in contrast:
                        f_contrast.append(data)

                    prepared_data = {
                        "class": class_feature2[index],
                        # "FFT": val_fft,
                        "ZCR": val_zcr,
                        "Autocorr": val_autocorr,
                        "STE": STEs
                    }
                    # for data in val_hps[0:1]:
                    #     prepared_data.update({
                    #         "hps-1".format(val_hps.index(data)): data
                    #     })
                    # for data in STEs:
                    #     prepared_data.update({
                    #         "ste-{}".format(STEs.index(data)): data
                    #     })
                    for data in f_contrast:
                        prepared_data.update({
                            "contras-{}".format(f_contrast.index(data)): data
                        })
                    for data in mfcc:
                        prepared_data.update({
                            "mfcc-{}".format(str(np.where(mfcc == data))): data
                        })
                    for data in rms[0, 0:8]:
                        prepared_data.update({
                            "rms-{}".format(str(np.where(rms[0] == data))): data
                        })
                    for data in melspectogram:
                        prepared_data.update({
                            "mel-specto-{}".format(str(np.where(melspectogram == data))): data
                        })
                    # print("berhasil di {}".format(files))
                    all_meta_data.append(prepared_data)
                    count += 1
                    time.sleep(0.3)
                    self.countChanged.emit(int(count/len_data * 100))
                except:
                    print("Gagal di {}".format(files))
                    log.exception("What a waste of life")
            index += 1
        data_frame = pd.DataFrame(all_meta_data)
        data_frame.fillna(0, inplace=True)
        data_frame = data_frame.reindex(columns=(list([a for a in data_frame.columns if a != 'class'])) + ['class'])
        feature_columns = list([a for a in data_frame.columns if a != 'class'])
        x = data_frame[feature_columns]
        normalz = StandardScaler()
        normalz.fit(x)
        new_normalz_x = normalz.transform(x)
        new_data_frame = pd.DataFrame(new_normalz_x, columns=feature_columns)
        new_data_frame['class'] = data_frame['class']
        new_data_frame.to_csv('data_train.csv', index=False, header=False)
        print("berhasil")

def initialize():
    cung_kanan = glob.glob('dataset/data_train/1_Cung_kanan/*')
    de_kanan = glob.glob('dataset/data_train/2_De_kanan/*')
    plak_kiri = glob.glob('dataset/data_train/4_Plak_kiri/*')
    pung_kiri = glob.glob('dataset/data_train/5_Pung_kiri/*')
    tek_kanan = glob.glob('dataset/data_train/3_Tek_kanan/*')
    teng_kiri = glob.glob('dataset/data_train/6_Teng_kiri/*')
    yastup = glob.glob('dataset/data_train/7_pepayas_nutup/*')

    all_data = [cung_kanan, de_kanan, tek_kanan, plak_kiri, pung_kiri, teng_kiri, yastup]
    len_data = len(cung_kanan) + len(de_kanan) + len(tek_kanan) + len(plak_kiri) + len(pung_kiri) + len(teng_kiri) + len(yastup)
    return all_data, len_data
