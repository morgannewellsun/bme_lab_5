"""

"""
import os

import numpy as np
import scipy
import scipy.signal as sig
from matplotlib.ticker import ScalarFormatter
from matplotlib import pyplot as plt


class ScalarFormatterClass(ScalarFormatter):
    def _set_format(self):
        self.format = "%1.0f"


y_scalar_formatter = ScalarFormatterClass(useMathText=True)
y_scalar_formatter.set_powerlimits((0, 0))
plt.rcParams["figure.figsize"] = [6, 6]
plt.rcParams["figure.autolayout"] = True


def main(filepath_ecog, filepath_spikes, dir_output):

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

    # design a pair of 48db/oct digital filters for each band
    order = 8
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
    labels = ("low", "intermediate", "high")
    fig, axes = plt.subplots(3, 1, sharex="all", sharey="all")
    for i, (label, np_ecog_29_band_power_smoothed_V2) in enumerate(zip(labels, np_ecog_29_bands_power_smoothed_V2)):
        axes[i].set_title(f"power for {label} frequency band")
        axes[i].set_xlabel("time (s)")
        axes[i].set_ylabel("power ($V^2$)")
        axes[i].plot(
            np.arange(len(np_ecog_29_band_power_smoothed_V2)) / sample_frequency_Hz,
            np_ecog_29_band_power_smoothed_V2)
        axes[i].yaxis.set_major_formatter(y_scalar_formatter)
    plt.tight_layout()
    plt.savefig(os.path.join(dir_output, "power for frequency bands.png"))
    plt.close()

    # =========================================================================
    # Part 3
    # =========================================================================

    # unpack data
    dict_spikes = scipy.io.loadmat(filepath_spikes)
    np_spikes = dict_spikes["spikes"]

    # normalize spikes
    np_spikes_mean = np.mean(np_spikes, axis=0, keepdims=True)
    np_spikes_std = np.std(np_spikes, axis=0, keepdims=True)
    np_spikes_normalized = (np_spikes - np_spikes_mean) / np_spikes_std

    # PCA
    n_dims_original = np_spikes_normalized.shape[1]
    np_unitary, np_singular_values, np_row_eigenvectors = np.linalg.svd(np_spikes_normalized, full_matrices=False)
    np_eigenvalues = np.square(np_singular_values) / (n_dims_original - 1)
    np_col_pcs = np_unitary @ np.diag(np_singular_values)

    # Determine number of necessary principal components to capture 90% of variance
    np_captured_variance = np.cumsum(np_eigenvalues) / np.sum(np_eigenvalues)
    k = np.min(np.argwhere(np_captured_variance > 0.9)) + 1
    for i in range(k):
        print(f"Variance captured with {i+1} PCs: {np_captured_variance[i]}")

    # Plot top k principal components for one spike
    sample_index = 4
    fig, axes = plt.subplots(1, 1, sharex="all", sharey="all")
    np_components = np_col_pcs[sample_index][:, np.newaxis] * np_row_eigenvectors
    np_components *= np_spikes_std.flatten()[:, np.newaxis]
    np_components += np_spikes_mean.flatten()[:, np.newaxis]
    for i in range(k):
        axes.plot(np_components[i], label=f"PC {i+1}")
    axes.set_title(f"Individual PCs for spike {sample_index+1}")
    axes.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(dir_output, "individual pcs.png"))

    # Spike reconstruction
    results_dict = {}
    for n_dims_reconstruction in (32, 6):
        np_spikes_reconstructed = np_col_pcs[:, :n_dims_reconstruction] @ np_row_eigenvectors[:n_dims_reconstruction, :]
        np_spikes_reconstructed *= np_spikes_std
        np_spikes_reconstructed += np_spikes_mean
        results_dict[n_dims_reconstruction] = np_spikes_reconstructed

    # Plotting reconstructed spikes
    fig, axes = plt.subplots(2, 1, sharex="all", sharey="all")
    [a.plot(np_spikes[4], linewidth=6, color="black", label="original") for a in axes]
    axes[0].set_title("Original data v.s. 32-dim PCA")
    axes[0].plot(results_dict[32][sample_index], color="red", label="32-dim PCA")
    axes[1].set_title("Original data v.s. 6-dim PCA")
    axes[1].axes.plot(results_dict[6][sample_index], color="red", label="6-dim PCA")
    [a.legend() for a in axes]
    plt.tight_layout()
    plt.savefig(os.path.join(dir_output, "original data vs pca.png"))
    plt.close()


if __name__ == "__main__":
    _filepath_ecog = r"D:\Documents\Academics\BME517\bme_lab_5\data\ecogdatasnippet.mat"
    _filepath_spikes = r"D:\Documents\Academics\BME517\bme_lab_5\data\spikes.mat"
    _dir_output = r"D:\Documents\Academics\BME517\bme_lab_5_6_report"
    main(_filepath_ecog, _filepath_spikes, _dir_output)
