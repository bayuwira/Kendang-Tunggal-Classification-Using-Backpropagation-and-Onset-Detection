import numpy as np
import librosa
import librosa.display
import pysptk
from scipy.io import wavfile
import IPython.display as ipd
import matplotlib.pyplot as plt
import numpy as np
import glob
from pysptk.synthesis import MLSADF, Synthesizer
import soundfile as sf
import os

def mlsa(x, sr):
    os.remove('audio_sintetik/synthesized_audio.wav')

    x = x.astype(np.float64)
    frame_length = 1024
    hop_length = 80

    frames = librosa.util.frame(x, frame_length=frame_length, hop_length=hop_length).astype(np.float64).T
    frames *= pysptk.hanning(frame_length)

    assert frames.shape[1] == frame_length
    pitch = pysptk.swipe(x.astype(np.float64), fs=sr, hopsize=hop_length, min=60, max=2000, otype="pitch")
    source_excitation = pysptk.excite(pitch, hop_length)

    order = 25
    alpha = 0.41

    mc = pysptk.mcep(frames, order, alpha)
    logH = pysptk.mgc2sp(mc, alpha, 0.0, frame_length).real

    b = pysptk.mc2b(mc, alpha);

    synthesizer = Synthesizer(MLSADF(order=order, alpha=alpha), hop_length)
    x_synthesized = synthesizer.synthesis(source_excitation, b)

    plt.figure(figsize=(14, 5))
    librosa.display.waveplot(y=x, sr=sr)
    plt.title('Plot audio yang di sintesis')
    plt.savefig('figure/synthesized_audio.png')

    sf.write('audio_sintetik/synthesized_audio.wav', x_synthesized, sr, subtype='PCM_24')

    print("audio created")