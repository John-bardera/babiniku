# coding:utf-8

import numpy as np
import pyaudio
# import change_voice_sptk as cvs
import change_voice_world as cvw
import noise_removal as nr

CHUNK = 1024 * 16
# RATE = 44100
RATE = 16000
NO_TRANS = False
# Use world
# CHUNK = 1024 * 16
# RATE = 16000

# Max 100000: No trans(Rate = 32000)
MAGNIFICATION = 1.0

INPUT_DEVICE_INDEX = 0
OUTPUT_DEVICE_INDEX = 1

p = pyaudio.PyAudio()

stream = p.open(format=pyaudio.paInt16,
                channels=1,
                rate=RATE,
                frames_per_buffer=CHUNK,
                input=True,
                output=True,
                input_device_index=INPUT_DEVICE_INDEX,
                output_device_index=OUTPUT_DEVICE_INDEX)


def audio_trans(tar_data):
    np_data = np.frombuffer(tar_data, dtype=np.int16) * MAGNIFICATION

    # np_data = np.where(np_data == 0, , np_data)
    # synthesized_data = cv.babiniku(np_data, RATE)
    # bin_data = synthesized_data.tobytes()
    # 詳しくはchange_voice.py参照

    # np_data = np_data * MAGNIFICATION

    f0, sp, ap = cvw.data_extraction(np_data, RATE)
    f0, sp, ap = cvw.parameter_setting(f0, sp, ap, 0)
    synthesized_data = cvw.synthesize(f0, sp, ap, RATE)

    nr_synthesized_data = nr.noise_removal(synthesized_data, RATE).astype(np.int16)

    bin_data = nr_synthesized_data.tobytes()

    return bin_data


def no_trans(tar_data):
    return tar_data


while stream.is_active():
    input_data = stream.read(CHUNK)  # , exception_on_overflow=False)
    if not NO_TRANS:
        trans_data = audio_trans(input_data)
    else:
        trans_data = no_trans(input_data)
    output_data = stream.write(trans_data)

stream.stop_stream()
stream.close()
p.terminate()
