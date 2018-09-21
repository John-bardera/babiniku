# coding:utf-8

import numpy as np
import pyworld as pw


def data_extraction(np_data, rate):
    np_data = np_data.astype(np.float)
    _f0, t = pw.harvest(np_data, rate)
    f0 = pw.stonemask(np_data, _f0, t, rate)
    sp = pw.cheaptrick(np_data, f0, t, rate)
    ap = pw.d4c(np_data, f0, t, rate)
    return f0, sp, ap


def synthesize(f0, sp, ap, rate):
    synthesized = pw.synthesize(f0, sp, ap, rate)
    return synthesized


def parameter_setting(f0, sp, ap, vtype):
    if vtype == 0:
        # 女性
        female_like_sp = np.zeros_like(sp)
        for f in range(female_like_sp.shape[1]):
            female_like_sp[:, f] = sp[:, int(f / 1.2)]
        return f0 * 1.9, female_like_sp, ap
    elif vtype == 1:
        # ロボット
        robot_like_f0 = np.ones_like(f0) * 100
        return robot_like_f0, sp, ap
    elif vtype == 2:
        # 声低めの男性
        male_like_sp = np.zeros_like(sp)
        for f in range(male_like_sp.shape[1]):
            male_like_sp[:, f] = sp[:, int(f / 1.0)]
        return f0 * 0.6, male_like_sp, ap
    elif vtype == 3:
        # かすれ声
        return f0 * 0, sp, ap
    else:
        return f0, sp, ap