import numpy as np


def short_time_processor(full_utterance: np.array,
                         func,
                         sr=16000,
                         frame_dur=0.04,  # seconds
                         hop_dur=0.02,  # seconds
                         *args,
                         **kwargs):
    """
    Works like a decorator. Applies func to all frames of full_utterance and returns 2D output.
    """

    frame_outputs = list()
    frame_length = round(sr * frame_dur)  # samples
    hop_length = round(sr * hop_dur)  # samples
    n_frames = (len(full_utterance)-frame_length) // hop_length  # last incomplete frame is ignored.
    beginning_index = 0
    for i in range(n_frames):
        current_frame = full_utterance[beginning_index: beginning_index+frame_length]
        current_output = func(current_frame, *args, **kwargs)
        frame_outputs.append(current_output)
        beginning_index += hop_length

    full_output = np.transpose(np.array(frame_outputs))
    return full_output


def frame_energy(frame):
    return np.mean(np.square(frame))


def frame_magnitude(frame):
    return np.mean(np.abs(frame))


def frame_zero_crossing(frame):
    return len(np.where(np.diff(np.sign(frame)))[0])


def frame_autocorrelation(frame, k):
    results = list()
    for i in range(k):
        if i == 0:
            results.append(np.sum(np.square(frame)))
        else:
            results.append(np.dot(frame[:-i], frame[i:]))
    return np.array(results)


def frame_dft(frame, n_fft=1024):
    exponentials = []
    discrete_index = np.arange(len(frame))
    for k in range(n_fft//2):
        exponentials.append(np.exp(-2j * np.pi * discrete_index * (k / n_fft)))
    return np.abs(np.dot(np.array(exponentials), frame))
