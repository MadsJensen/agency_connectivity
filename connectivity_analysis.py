import mne
import numpy as np
import matplotlib.pyplot as plt

from tf_analysis import single_trial_tf, morlet_analysis
from preprocessing import hilbert_process

plt.ion()

data_folder = "/home/mje/Projects/agency_connectivity/data/"

epochs = mne.read_epochs(data_folder + "P2_ds_bp_ica-epo.fif")
raw = mne.io.Raw(data_folder + "P2_ds_bp_ica-raw.fif", preload=True)

bands = {"alpha": [8, 12],
         "beta": [15, 25]}

hb_epo = hilbert_process(raw, bands)

hb_tfr = hb_epo["alpha"]["voluntary"]

# single trial morlet tests
frequencies = np.arange(6., 30., 1.)
n_cycles = 5.
times = epochs.times

tfr_vol = single_trial_tf(epochs["voluntary"])
tfr_invol = single_trial_tf(epochs["involuntary"])

tfr_vol = np.asarray(tfr_vol)
tfr_invol = np.asarray(tfr_invol)


itc_over_time_vol = np.empty(tfr_vol.shape[-1])
itc_over_time_invol = np.empty(tfr_vol.shape[-1])

chan_index = 37

for i in range(len(itc_over_time_vol)):
    itc_over_time_vol[i] = np.abs(np.mean(
        np.exp(1j*np.angle(tfr_vol[:, chan_index, 1:4, i]))))
    # itc_over_time_invol[i] = np.abs(np.mean(
    #     np.exp(1j*np.angle(tfr_invol[:, chan_index, 1:4, i].mean(axis=0)))))

hb_itc = np.empty(hb_tfr.shape[-1])
for i in range(len(hb_itc)):
    hb_itc[i] = np.abs(np.mean(np.exp(1j*np.angle(hb_tfr[:, chan_index, i]))))


plt.figure()
plt.plot(epochs.times, itc_over_time_vol, 'b', label="voluntary")
plt.plot(epochs.times, hb_itc, 'k', label="Hilbert")
# plt.plot(epochs.times, itc_over_time_invol, 'r', label="involuntary")
plt.legend()
plt.show()


pow_morlet_vol, itc_morlet_vol = morlet_analysis(epochs["voluntary"],
                                                 n_cycles=5)


plt.figure()
plt.plot(epochs.times, itc_over_time_vol, 'b', label="Mine ITC")
plt.plot(epochs.times, itc_morlet_vol.data[chan_index, 2:7, :].mean(axis=0),
         'r', label="MNE-python itc")
plt.plot(epochs.times, hb_itc, 'k', label="Hilbert")
plt.legend()
plt.show()


#  ISPC test
tfr_vol = single_trial_tf(epochs["voluntary"])
tfr_invol = single_trial_tf(epochs["involuntary"])

tfr_vol = np.asarray(tfr_vol)
tfr_invol = np.asarray(tfr_invol)


ispc_vol = np.empty(tfr_vol.shape[-1])
ispc_invol = np.empty(tfr_invol.shape[-1])

chan_A, chan_B = 37, 47
freq_idx = 4

for i in range(len(ispc_vol)):
    ispc_vol[i] = np.abs(np.mean(np.exp(
        1j*(np.angle(tfr_vol[:, chan_A, freq_idx, i]) -
            np.angle(tfr_vol[:, chan_B, freq_idx, i])))))


for i in range(len(ispc_invol)):
    ispc_invol[i] = np.abs(np.mean(np.exp(
        1j*(np.angle(tfr_invol[:, chan_A, freq_idx, i]) -
            np.angle(tfr_invol[:, chan_B, freq_idx, i])))))


plt.figure()
plt.plot(times, ispc_vol, 'b', label="Voluntary")
plt.plot(times, ispc_invol, 'r', label="Involuntary")
plt.legend()
plt.title("ISPC between %s and %s in freq: %s" % (epochs.ch_names[chan_A],
                                                  epochs.ch_names[chan_B],
                                                  frequencies[freq_idx]))
plt.show()

# ISPC over time test

ispc_vol = np.empty(tfr_vol.shape[0])
ispc_invol = np.empty(tfr_invol.shape[0])

chan_A, chan_B = 37, 47
freq_idx = 4

for i in range(len(ispc_vol)):
    ispc_vol[i] = np.abs(np.mean(np.exp(
        1j*(np.angle(tfr_vol[i, chan_A, freq_idx, :]) -
            np.angle(tfr_vol[i, chan_B, freq_idx, :])))))

for i in range(len(ispc_invol)):
    ispc_invol[i] = np.abs(np.mean(np.exp(
        1j*(np.angle(tfr_invol[i, chan_A, freq_idx, :]) -
            np.angle(tfr_invol[i, chan_B, freq_idx, :])))))


for freq in frequencies:
    print(freq)
