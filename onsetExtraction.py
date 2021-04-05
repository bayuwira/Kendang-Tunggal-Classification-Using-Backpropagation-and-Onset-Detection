import numpy as np
import scipy
from librosa._cache import cache
from librosa import core
from librosa import util
from librosa.util.exceptions import ParameterError
from librosa.feature.spectral import melspectrogram

__all__ = ["onset_detect", "onset_detect_non_normalize", "onset_strength", "onset_strength_multi", "onset_backtrack"]


def onset_detect(
    y=None,
    sr=22050,
    onset_envelope=None,
    hop_length=512,
    backtrack=False,
    energy=None,
    units="frames",
    **kwargs,
):


    # First, get the frame->beat strength profile if we don't already have one
    if onset_envelope is None:
        if y is None:
            raise ParameterError("y or onset_envelope must be provided")

        onset_envelope = onset_strength(y=y, sr=sr, hop_length=hop_length)

    # Shift onset envelope up to be non-negative
    # (a common normalization step to make the threshold more consistent)
    onset_envelope -= onset_envelope.min()

    # Do we have any onsets to grab?
    if not onset_envelope.any():
        return np.array([], dtype=np.int)

    # Normalize onset strength function to [0, 1] range
    onset_envelope /= onset_envelope.max()

    # These parameter settings found by large-scale search
    kwargs.setdefault("pre_max", 0.03 * sr // hop_length)  # 30ms
    kwargs.setdefault("post_max", 0.00 * sr // hop_length + 1)  # 0ms
    kwargs.setdefault("pre_avg", 0.10 * sr // hop_length)  # 100ms
    kwargs.setdefault("post_avg", 0.10 * sr // hop_length + 1)  # 100ms
    kwargs.setdefault("wait", 0.03 * sr // hop_length)  # 30ms
    kwargs.setdefault("delta", 0.07)

    # Peak pick the onset envelope
    onsets = util.peak_pick(onset_envelope, **kwargs)

    # Optionally backtrack the events
    if backtrack:
        if energy is None:
            energy = onset_envelope

        onsets = onset_backtrack(onsets, energy)

    if units == "frames":
        pass
    elif units == "samples":
        onsets = core.frames_to_samples(onsets, hop_length=hop_length)
    elif units == "time":
        onsets = core.frames_to_time(onsets, hop_length=hop_length, sr=sr)
    else:
        raise ParameterError("Invalid unit type: {}".format(units))

    return onsets

def onset_detect_non_normalize(
        y=None,
        sr=22050,
        onset_envelope=None,
        hop_length=512,
        backtrack=False,
        energy=None,
        units="frames",
        **kwargs,
):
    # First, get the frame->beat strength profile if we don't already have one
    if onset_envelope is None:
        if y is None:
            raise ParameterError("y or onset_envelope must be provided")

        onset_envelope = onset_strength(y=y, sr=sr, hop_length=hop_length)

    # Shift onset envelope up to be non-negative
    # (a common normalization step to make the threshold more consistent)
    # onset_envelope -= onset_envelope.min()

    # Do we have any onsets to grab?
    if not onset_envelope.any():
        return np.array([], dtype=np.int)

    # Normalize onset strength function to [0, 1] range
    # onset_envelope /= onset_envelope.max()

    # These parameter settings found by large-scale search
    kwargs.setdefault("pre_max", 0.03 * sr // hop_length)  # 30ms
    kwargs.setdefault("post_max", 0.00 * sr // hop_length + 1)  # 0ms
    kwargs.setdefault("pre_avg", 0.10 * sr // hop_length)  # 100ms
    kwargs.setdefault("post_avg", 0.10 * sr // hop_length + 1)  # 100ms
    kwargs.setdefault("wait", 0.03 * sr // hop_length)  # 30ms
    kwargs.setdefault("delta", 0.07)

    # Peak pick the onset envelope
    onsets = util.peak_pick(onset_envelope, **kwargs)

    # Optionally backtrack the events
    if backtrack:
        if energy is None:
            energy = onset_envelope

        onsets = onset_backtrack(onsets, energy)

    if units == "frames":
        pass
    elif units == "samples":
        onsets = core.frames_to_samples(onsets, hop_length=hop_length)
    elif units == "time":
        onsets = core.frames_to_time(onsets, hop_length=hop_length, sr=sr)
    else:
        raise ParameterError("Invalid unit type: {}".format(units))

    return onsets


def onset_strength(
    y=None,
    sr=22050,
    S=None,
    lag=1,
    max_size=1,
    ref=None,
    detrend=False,
    center=True,
    feature=None,
    aggregate=None,
    **kwargs,
):

    if aggregate is False:
        raise ParameterError(
            "aggregate={} cannot be False when computing full-spectrum onset strength."
        )

    odf_all = onset_strength_multi(
        y=y,
        sr=sr,
        S=S,
        lag=lag,
        max_size=max_size,
        ref=ref,
        detrend=detrend,
        center=center,
        feature=feature,
        aggregate=aggregate,
        channels=None,
        **kwargs,
    )

    return odf_all[0]


def onset_backtrack(events, energy):

    minima = np.flatnonzero((energy[1:-1] <= energy[:-2]) & (energy[1:-1] < energy[2:]))

    # Pad on a 0, just in case we have onsets with no preceding minimum
    # Shift by one to account for slicing in minima detection
    minima = util.fix_frames(1 + minima, x_min=0)

    # Only match going left from the detected events
    return minima[util.match_events(events, minima, right=False)]


@cache(level=30)
def onset_strength_multi(
    y=None,
    sr=22050,
    S=None,
    n_fft=2048,
    hop_length=512,
    lag=1,
    max_size=1,
    ref=None,
    detrend=False,
    center=True,
    feature=None,
    aggregate=None,
    channels=None,
    **kwargs,
):

    if feature is None:
        feature = melspectrogram
        kwargs.setdefault("fmax", 11025.0)

    if aggregate is None:
        aggregate = np.mean

    if lag < 1 or not isinstance(lag, int):
        raise ParameterError("lag must be a positive integer")

    if max_size < 1 or not isinstance(max_size, int):
        raise ParameterError("max_size must be a positive integer")

    # First, compute mel spectrogram
    if S is None:
        S = np.abs(feature(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, **kwargs))

        # Convert to dBs
        S = core.power_to_db(S)

    # Ensure that S is at least 2-d
    S = np.atleast_2d(S)

    # Compute the reference spectrogram.
    # Efficiency hack: skip filtering step and pass by reference
    # if max_size will produce a no-op.
    if ref is None:
        if max_size == 1:
            ref = S
        else:
            ref = scipy.ndimage.maximum_filter1d(S, max_size, axis=0)
    elif ref.shape != S.shape:
        raise ParameterError(
            "Reference spectrum shape {} must match input spectrum {}".format(
                ref.shape, S.shape
            )
        )

    # Compute difference to the reference, spaced by lag
    onset_env = S[:, lag:] - ref[:, :-lag]

    # Discard negatives (decreasing amplitude)
    onset_env = np.maximum(0.0, onset_env)

    # Aggregate within channels
    pad = True
    if channels is None:
        channels = [slice(None)]
    else:
        pad = False

    if aggregate:
        onset_env = util.sync(onset_env, channels, aggregate=aggregate, pad=pad, axis=0)

    # compensate for lag
    pad_width = lag
    if center:
        # Counter-act framing effects. Shift the onsets by n_fft / hop_length
        pad_width += n_fft // (2 * hop_length)

    onset_env = np.pad(onset_env, ([0, 0], [int(pad_width), 0]), mode="constant")

    # remove the DC component
    if detrend:
        onset_env = scipy.signal.lfilter([1.0, -1.0], [1.0, -0.99], onset_env, axis=-1)

    # Trim to match the input duration
    if center:
        onset_env = onset_env[:, : S.shape[1]]

    return onset_env

