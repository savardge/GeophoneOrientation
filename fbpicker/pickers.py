""" Simple first break pickers"""

from obspy.realtime.signal import kurtosis
import numpy as np


def get_arrival_from_kurtosis(st, win=0.1):
    """ Use max of derivative of kurtosis """
    tarr = []
    for tr in st.traces:
        k = kurtosis(st[0], win=win)
        dk = np.gradient(k)
        ix = np.argmax(np.abs(dk))
        t = tr.times("utcdatetime")
        tarr.append(t[ix])
    return min(tarr)
