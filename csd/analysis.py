import numpy as np
import scipy
import pywt # github.com/PyWavelets/pywt
import csd.icsd # github.com/espenhgn/iCSD
import quantities # github.com/python-quantities/python-quantities


def memory_map_imec(path):
    with open(path.replace('.bin', '.meta'), 'r') as f:
        metadata = dict([kv.strip('~').split('=') for kv in f.readlines()])
        n_chan = int(metadata['nSavedChans'])
        n_byte = int(metadata['fileSizeBytes'])
        n_sample = (n_byte // 2) // n_chan
    return np.memmap(path, dtype='int16', mode='r', order='C', shape=(n_sample, n_chan))


def lowpass_filter(sig, sample_hz=2500, cutoff_hz=300, order=4, **kwargs):
    cutoff = cutoff_hz / (sample_hz / 2)
    sos = scipy.signal.butter(btype='lowpass', N=order, Wn=cutoff, output='sos')
    return scipy.signal.sosfiltfilt(sos, sig, **kwargs)


def highpass_filter(sig, sample_hz=2500, cutoff_hz=1, order=4, **kwargs):
    cutoff = cutoff_hz / (sample_hz / 2)
    sos = scipy.signal.butter(btype='highpass', N=order, Wn=cutoff, output='sos')
    return scipy.signal.sosfiltfilt(sos, sig, **kwargs)


def gaussian_filter(sig, step_um=40, std_um=40, radius_um=120, **kwargs):
    assert std_um % step_um == 0, 'require std to divide evenly by step size'
    assert radius_um % step_um == 0, 'require radius to divide evenly by step size'
    std = std_um // step_um
    radius = radius_um // step_um
    return scipy.ndimage.gaussian_filter1d(sig, sigma=std, radius=radius, **kwargs)


def csd_transform(sig, step_um=40):
    coord = step_um * 1e-6 * np.arange(sig.shape[1]) * quantities.m
    sigma = 0.3 * quantities.S / quantities.m
    lfp = sig.T * quantities.V
    csd = icsd.StandardCSD(lfp, coord_electrode=coord, sigma=sigma).get_csd()
    return np.array(csd).T


def wavelet_transform(sig, sample_hz=2500, cycle_hz=7, template='cmor1.5-1.0', **kwargs):
    freq_cycles = cycle_hz / sample_hz # wavelet periods per sample duration
    wavelet = pywt.ContinuousWavelet(template)
    scale = pywt.frequency2scale(wavelet, freq_cycles)
    coefs, _ = pywt.cwt(sig, scales=scale, wavelet=wavelet, **kwargs)
    return np.array(coefs[0])

    
def neuropixels_lfp(arr):
    # (sample, neuropixels channel) -> (shank[4], sample, 40-µm channel)
    assert arr.shape[1] % 4 == 0
    n_sample, n_channel = arr.shape[0], arr.shape[1] // 4
    lfp = np.zeros((4, n_sample, n_channel))
    for i in range(4):
        sig = arr[:, i::4]
        sig = sig - np.mean(sig, axis=1, keepdims=True)
        sig = highpass_filter(sig, axis=0, sample_hz=2500, cutoff_hz=1)
        sig = lowpass_filter(sig, axis=0, sample_hz=2500, cutoff_hz=300)
        lfp[i] = sig
    return lfp


def neuropixels_csd(arr):
    # (shank[4], sample, 40-µm channel) -> (sample, 20-µm channel)
    assert arr.shape[0] == 4
    n_sample, n_channel = arr.shape[1], arr.shape[2]
    csd = np.zeros((4, n_sample, n_channel))
    for i in range(4):
        sig = arr[i]
        sig = csd_transform(sig, step_um=40)
        sig = scipy.stats.zscore(sig, axis=None) # correct for gain difference
        csd[i] = sig
    csd = np.array([csd[0] + csd[1], csd[2] + csd[3]])
    csd = csd.transpose(1, 2, 0).reshape(n_sample, n_channel * 2)
    return csd

