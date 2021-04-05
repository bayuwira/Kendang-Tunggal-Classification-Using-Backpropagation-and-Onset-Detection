from __future__ import division
import numpy
from numpy.fft import rfft
from numpy import argmax, mean, diff, log, nonzero
from scipy.signal import correlate
from scipy.signal.windows import blackmanharris
from time import time
import soundfile as sf
from parabolic import parabolic
import librosa.feature
from pylab import subplot, plot, log, copy, show
import math
import librosa

def freq_from_crossings(sig, fs):
    """
    Estimate frequency by counting zero crossings
    """
    # Find all indices right before a rising-edge zero crossing
    indices = nonzero((sig[1:] >= 0) & (sig[:-1] < 0))[0]

    # Naive (Measures 1000.185 Hz for 1000 Hz, for instance)
    # crossings = indices

    # More accurate, using linear interpolation to find intersample
    # zero-crossings (Measures 1000.000129 Hz for 1000 Hz, for instance)
    crossings = [i - sig[i] / (sig[i+1] - sig[i]) for i in indices]

    # Some other interpolation based on neighboring points might be better.
    # Spline, cubic, whatever

    return fs / mean(diff(crossings))


def freq_from_fft(sig, fs):
    """
    Estimate frequency from peak of FFT
    """
    # Compute Fourier transform of windowed signal
    windowed = sig * blackmanharris(len(sig))
    f = rfft(windowed)

    # Find the peak and interpolate to get a more accurate peak
    i = argmax(abs(f))  # Just use this for less-accurate, naive version
    true_i = parabolic(log(abs(f)), i)[0]

    # Convert to equivalent frequency
    return fs * true_i / len(windowed)

def autocorr(y, sr, fmin=50.0, fmax = 2000.0):
    # Compute autocorrelation of input segment.
    r = librosa.autocorrelate(y)

    # Define lower and upper limits for the autocorrelation argmax.
    i_min = sr / fmax
    i_max = sr / fmin
    r[:int(i_min)] = 0
    r[int(i_max):] = 0

    # Find the location of the maximum autocorrelation.
    i = r.argmax()
    f0 = float(sr) / i
    return f0


def freq_from_autocorr(sig, fs):
    """
    Estimate frequency using autocorrelation
    """
    # Calculate autocorrelation and throw away the negative lags
    corr = correlate(sig, sig, mode='full')
    corr = corr[len(corr)//2:]

    # Find the first low point
    d = diff(corr)
    start = nonzero(d > 0)[0][0]

    # Find the next peak after the low point (other than 0 lag).  This bit is
    # not reliable for long signals, due to the desired peak occurring between
    # samples, and other peaks appearing higher.
    # Should use a weighting function to de-emphasize the peaks at longer lags.
    peak = argmax(corr[start:]) + start
    px, py = parabolic(corr, peak)
    return fs / px


def freq_from_HPS(sig, fs):
    from pylab import subplot, plot, log, copy, show
    windowed = sig * blackmanharris(len(sig))
    HPSs = []
    # harmonic product spectrum:
    c = abs(rfft(windowed))
    maxharms = 6
    for x in range(2, maxharms):
        a = copy(c[::x])  # Should average or maximum instead of decimating
        # max(c[::x],c[1::x],c[2::x],...)
        c = c[:len(a)]
        i = argmax(abs(c))
        true_i = parabolic(abs(c), i)[0]
        HPSs.append(fs * true_i / len(windowed))
    return HPSs

def computeSTE(fs, signal):
    signal = signal / max(abs(signal))  # scale signal
    assert min(signal) >= -1 and max(signal) <= 1
    sampsPerMilli = int(fs / 1000)
    millisPerFrame = 20
    sampsPerFrame = sampsPerMilli * millisPerFrame
    from numpy import zeros
    STEs = []  # list of short-time energies
    for k in range(0, 14):
        startIdx = k * sampsPerFrame
        stopIdx = startIdx + sampsPerFrame
        window = zeros(signal.shape)
        window[startIdx:stopIdx] = 1  # rectangular window
        STE = sum((signal ** 2) * (window ** 2))
        STEs.append(STE)
    return STEs

def getSTE(signal, frame_length=512, hop_size=256):
    energy = numpy.array([
        sum(abs(signal[i:i+frame_length]**2))
        for i in range(0, len(signal), hop_size)
    ])
    return numpy.mean(energy)


def rmse(signal, frame_size, hop_length):
    rmse = []

    # calculate rmse for each frame
    for i in range(0, len(signal), hop_length):
        rmse_current_frame = numpy.sqrt(sum(signal[i:i + frame_size] ** 2) / frame_size)
        rmse.append(rmse_current_frame)
    return numpy.array(rmse)

def calculate_split_frequency_bin(split_frequency, sample_rate, num_frequency_bins):
    """Infer the frequency bin associated to a given split frequency."""

    frequency_range = sample_rate / 2
    frequency_delta_per_bin = frequency_range / num_frequency_bins
    split_frequency_bin = math.floor(split_frequency / frequency_delta_per_bin)
    return int(split_frequency_bin)


def band_energy_ratio(spectrogram, split_frequency, sample_rate):
    """Calculate band energy ratio with a given split frequency."""

    split_frequency_bin = calculate_split_frequency_bin(split_frequency, sample_rate, len(spectrogram[0]))
    band_energy_ratio = []

    # calculate power spectrogram
    power_spectrogram = numpy.abs(spectrogram) ** 2
    power_spectrogram = power_spectrogram.T

    # calculate BER value for each frame
    for frame in power_spectrogram:
        sum_power_low_frequencies = frame[:split_frequency_bin].sum()
        sum_power_high_frequencies = frame[split_frequency_bin:].sum()
        band_energy_ratio_current_frame = sum_power_low_frequencies / sum_power_high_frequencies
        band_energy_ratio.append(band_energy_ratio_current_frame)

    return numpy.array(band_energy_ratio)


# def testFile():
#     filename = 'split_audio/full_lagu_3-potongan-264.39909297052156.wav'
#     filename2 = 'dataset/Dataset_Kendang/Cung_kanan\kendang-cung-kanan-kendang01-6.wav'
#     y, sr = librosa.load(filename)
#
#     print('Reading file "%s"\n' % filename)
#     signal, fs = sf.read(filename)
#
#     print('Calculating frequency from FFT:', end=' ')
#     start_time = time()
#     print('%f Hz' % freq_from_fft(signal, fs))
#     print('Time elapsed: %.3f s\n' % (time() - start_time))
#
#     print('Calculating frequency from zero crossings:', end=' ')
#     start_time = time()
#     print('%f Hz' % freq_from_crossings(signal, fs))
#     print('Time elapsed: %.3f s\n' % (time() - start_time))
#
#     print('Calculating frequency from autocorrelation:', end=' ')
#     start_time = time()
#     print('%f Hz' % freq_from_autocorr(signal, fs))
#     print('Time elapsed: %.3f s\n' % (time() - start_time))
#
#     print('Calculating frequency from harmonic product spectrum:')
#     start_time = time()
#     freq_from_HPS(signal, fs)
#     print('Time elapsed: %.3f s\n' % (time() - start_time))
#
#     hop_length = 256
#     frame_length = 512
#     print('Calculating STE:')
#     STEs = computeSTE(fs, signal)
#     for data in STEs:
#         print(data)
#     print(len(STEs))
# testFile()