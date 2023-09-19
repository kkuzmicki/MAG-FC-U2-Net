import tqdm
from pathlib import Path
import os
import librosa
import soundfile
from scipy.io import wavfile
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import scipy
import torch
import torchaudio

# legacy torch function
def complex_norm(
        complex_tensor: torch.Tensor,
        power: float = 1.0
) -> torch.Tensor:
    r"""Compute the norm of complex tensor input.

    Args:
        complex_tensor (Tensor): Tensor shape of `(..., complex=2)`
        power (float): Power of the norm. (Default: `1.0`).

    Returns:
        Tensor: Power of the normed input tensor. Shape of `(..., )`
    """

    # Replace by torch.norm once issue is fixed
    # https://github.com/pytorch/pytorch/issues/34279
    return complex_tensor.pow(2.).sum(-1).pow(0.5 * power)

# legacy torch function
def magphase(
        complex_tensor: torch.Tensor,
        power: float = 1.0
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Separate a complex-valued spectrogram with shape `(..., 2)` into its magnitude and phase.

    Args:
        complex_tensor (Tensor): Tensor shape of `(..., complex=2)`
        power (float): Power of the norm. (Default: `1.0`)

    Returns:
        (Tensor, Tensor): The magnitude and phase of the complex tensor
    """
    mag = complex_norm(complex_tensor, power)
    phase = torch.angle(complex_tensor)
    return mag, phase

def calculate_differences_between_arrays(arr1, arr2):
    differences = []
    for val1, val2 in zip(arr1, arr2):
        diff = val1 - val2
        differences.append(diff)
    return differences

def show_plot_for_list(list, header):
    plt.plot(list)
    plt.title(header)
    plt.show()

if __name__ == "__main__":

    # audio read
    origin_sample_rate, origin_audio = wavfile.read('org.wav')
    wavfile.write('out_org.wav', origin_sample_rate, origin_audio[:,0])

    origin_num_samples, origin_num_channels = origin_audio.shape

    # resampling
    target_audio_scipy = scipy.signal.resample(origin_audio[:,0], origin_num_samples).astype(int)
    target_audio_scipy = np.array(target_audio_scipy, np.int16)

    # show differences
    diffArray = calculate_differences_between_arrays(origin_audio[:,0], target_audio_scipy)
    show_plot_for_list(diffArray, 'Differences')

    # write modified audio to WAV file
    wavfile.write('out.wav', origin_sample_rate, target_audio_scipy)