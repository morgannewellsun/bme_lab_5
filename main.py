"""

"""

import numpy as np
import scipy
import scipy.signal as sig
from matplotlib import pyplot as plt


def main(filepath_ecog, filepath_spikes):

    # =========================================================================
    # Part 2
    # =========================================================================

    # unpack data
    dict_ecog = scipy.io.loadmat(filepath_ecog)
    sample_frequency_Hz = 1000

    # apply common average referencing
    np_ecog_29_raw_uV = dict_ecog["ecogData"][28]
    np_ecog_car_uV = np.mean(dict_ecog["ecogData"][dict_ecog["refChannels"].flatten()], axis=0)
    np_ecog_29_V = (np_ecog_29_raw_uV - np_ecog_car_uV) * 1e-6

    # design a pair of 24db/oct digital filters for each band
    order = 4
    bands_cutoffs_Hz = ((4, 6), (14, 46), (54, 114))
    sos_pairs = []
    for band_hp_cutoff_Hz, band_lp_cutoff_Hz in bands_cutoffs_Hz:
        sos_highpass = sig.butter(N=order, Wn=band_hp_cutoff_Hz, btype="highpass", output="sos", fs=sample_frequency_Hz)
        sos_lowpass = sig.butter(N=order, Wn=band_lp_cutoff_Hz, btype="lowpass", output="sos", fs=sample_frequency_Hz)
        sos_pairs.append((sos_highpass, sos_lowpass))

    # isolate frequency bands
    np_ecog_29_bands = []
    for sos_highpass, sos_lowpass in sos_pairs:
        np_ecog_29_bands.append(sig.sosfilt(sos=sos_lowpass, x=sig.sosfilt(sos=sos_highpass, x=np_ecog_29_V)))

    # calculate power for each band
    np_ecog_29_bands_power_V2 = [np.square(np_ecog_29_band) for np_ecog_29_band in np_ecog_29_bands]

    # smooth power for each band
    smoothing_window_ms = 100
    smoothing_window_samples = int(np.round(smoothing_window_ms * 1e-3 * sample_frequency_Hz))
    np_ecog_29_bands_power_smoothed_V2 = []
    for np_ecog_29_band_power_V2 in np_ecog_29_bands_power_V2:
        np_ecog_29_bands_power_smoothed_V2.append(
            np.mean(
                axis=1,
                a=np.lib.stride_tricks.sliding_window_view(
                    np_ecog_29_band_power_V2,
                    smoothing_window_samples)))

    # plot smoothed power for each band
    fig, axes = plt.subplots(3, 1, sharex="all")
    for i, np_ecog_29_band_power_smoothed_V2 in enumerate(np_ecog_29_bands_power_smoothed_V2):
        axes[i].plot(np_ecog_29_band_power_smoothed_V2)
    plt.show()


    # =========================================================================
    # Part 3
    # =========================================================================

    dict_spikes = scipy.io.loadmat(filepath_spikes)




if __name__ == "__main__":
    _filepath_ecog = r"C:\Users\Morgan\Documents\Academics\BME517\bme_lab_5\data\ecogdatasnippet.mat"
    _filepath_spikes = r"C:\Users\Morgan\Documents\Academics\BME517\bme_lab_5\data\spikes.mat"
    main(_filepath_ecog, _filepath_spikes)
