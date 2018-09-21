# coding:utf-8
"""
mel_cepstrumでは「zero(s) are found in periodogram, use eps option to floor」
mel_generalized_cepstrumでは「failed to compute mgcep; error occured in theq」
が発生し進めない。そんな記述もほとんど見つけれなかったので断念。
"""
import numpy as np
import librosa
import pysptk as ps

FRAME_LENGTH = 1024
HOP_LENGTH = 64
ORDER = 25
ALPHA = 0.41
STAGE = 5
GAMMA = -1.0 / STAGE


def framing_windowing(np_data):
    frames = librosa.util.frame(np_data, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH).astype(np.float64).T
    frames *= ps.blackman(FRAME_LENGTH)
    return frames


def source_excitation_generation(np_data, rate):
    pitch = ps.swipe(np_data.astype(np.float64), fs=rate, hopsize=HOP_LENGTH, min=60, max=240, otype="pitch")
    source_excitation = ps.excite(pitch, HOP_LENGTH)
    return source_excitation


def mel_cepstrum(frames):
    mc = ps.mcep(frames, ORDER, ALPHA, eps=0, etype=1)
    # logH = ps.mgc2sp(mc, ALPHA, 0.0, FRAME_LENGTH).real
    return mc


def mel_generalized_cepstrum(frames):
    mgc = ps.mgcep(frames, ORDER, ALPHA, GAMMA)
    return mgc


def synthesis_mel_cepstrum(mc, source_excitation):
    b = ps.mc2b(mc, ALPHA)
    synthesizer = ps.synthesis.Synthesizer(ps.synthesis.MLSADF(order=ORDER, alpha=ALPHA), HOP_LENGTH)
    synthesized = synthesizer.synthesis(source_excitation, b)
    return synthesized


def synthesis_mel_generalized_cepstrum(mgc, source_excitation):
    b = ps.mgc2b(mgc, ALPHA, GAMMA)
    synthesizer = ps.synthesis.Synthesizer(ps.synthesis.MGLSADF(order=ORDER, alpha=ALPHA, stage=STAGE), HOP_LENGTH)
    synthesized = synthesizer.synthesis(source_excitation, b)
    return synthesized


def babiniku(np_data, rate):
    frames = framing_windowing(np_data)
    source_excitation = source_excitation_generation(np_data, rate)
    mc = mel_generalized_cepstrum(frames)
    synthesized = synthesis_mel_generalized_cepstrum(mc, source_excitation)
    return synthesized

