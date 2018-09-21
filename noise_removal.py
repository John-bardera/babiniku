# coding:utf-8

import numpy as np
from scipy import fftpack


def noise_removal(np_data, rate):
    sample_freq = fftpack.fftfreq(np_data.size, d=rate)
    sig_fft = fftpack.fft(np_data)
    # pidxs = np.where(sample_freq > 0)
    # freqs = sample_freq[pidxs]
    # power = np.abs(sig_fft)[pidxs]
    sig_fft[(np.abs(sample_freq) > 10000)] = 0
    main_sig = np.real(fftpack.ifft(sig_fft))

    return main_sig
