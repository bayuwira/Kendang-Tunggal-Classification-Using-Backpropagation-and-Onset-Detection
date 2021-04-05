import pandas as pd
import featureExtraction
import glob
import librosa
import soundfile as sf
from sklearn.preprocessing import StandardScaler, MinMaxScaler

cung_kanan = glob.glob('dataset/Dataset_Kendang/Cung_kanan/*')
de_kanan = glob.glob('dataset/Dataset_Kendang/De_kanan/*')
plak_kiri = glob.glob('dataset/Dataset_Kendang/Plak_kiri/*')
pung_kiri = glob.glob('dataset/Dataset_Kendang/Pung_kiri/*')
tek_kanan = glob.glob('dataset/Dataset_Kendang/Tek_kanan/*')
teng_kiri = glob.glob('dataset/Dataset_Kendang/Teng_kiri/*')
all_data = [cung_kanan, de_kanan, tek_kanan, plak_kiri, pung_kiri, teng_kiri]
class_feature = ['C', 'D', 'T', 'c', 'd', 't']
class_feature2 = ['1', '2', '3', '4', '5', '6']
all_meta_data = []
index = 0
for meta_data in all_data:
        for files in meta_data:
            try :
                y, sr = librosa.load(files)
                signal, fs = sf.read(files)
                val_fft = featureExtraction.freq_from_fft(signal, fs)
                val_zcr = featureExtraction.freq_from_crossings(signal, fs)
                val_autocorr = featureExtraction.freq_from_autocorr(signal, fs)
                val_hps = featureExtraction.freq_from_HPS(signal, fs)
                STEs = featureExtraction.computeSTE(fs, signal)
                prepared_data = {
                    "class": class_feature2[index],
                    "FFT": val_fft,
                    "ZCR": val_zcr,
                    "Autocorr": val_autocorr
                }
                for data in val_hps[0:1]:
                    prepared_data.update({
                        "hps-1".format(val_hps.index(data)): data
                    })
                for data in STEs:
                    prepared_data.update({
                        "ste-{}".format(STEs.index(data)): data
                    })
                print("berhasil di {}".format(files))
                all_meta_data.append(prepared_data)
            except:
                print("Gagal di {}".format(files))
        index+=1
data_frame = pd.DataFrame(all_meta_data)
data_frame.fillna(0, inplace=True)
data_frame = data_frame.reindex(columns=sorted(data_frame.columns))
data_frame = data_frame.reindex(columns=(list([a for a in data_frame.columns if a != 'class'])) + ['class'])
data_frame.to_csv('kendang_class_feature3.csv', index=False, header=False)