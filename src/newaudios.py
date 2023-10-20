import os
import pickle

import numpy as np
import soundfile as sf

from vae import VAE
import librosa
from preprocessing import MinMaxNormaliser


"""
This is a script that provides functions to go from spectrograms to audios.

if runned: generates reconstructed audios of a batch of random audios from the dataset
"""

HOP_LENGTH = 256
SAVE_DIR_ORIGINAL = "samples/original/"
SAVE_DIR_GENERATED = "samples/generated/"
SPECTROGRAMS_PATH = "fsdd/spectrograms/"
MIN_MAX_VALUES_PATH = "fsdd/min_max_values.pkl"


class SoundGenerator:
    """
    Go from spectrograms to audio (after de decoder)
    """

    def __init__(self, vae, hop_length):
        self.vae = vae
        self.hop_length = hop_length
        self._min_max_normaliser = MinMaxNormaliser(0, 1)

    def generate(self, spectrograms, min_max_values):
        """
        General function to go from spectrogram data to audio signal
        """
        generated_spectrograms, latent_representations = \
            self.vae.reconstruct(spectrograms)
        signals = self.convert_spectrograms_to_audio(generated_spectrograms, min_max_values)
        return signals, latent_representations

    def convert_spectrograms_to_audio(self, spectrograms, min_max_values):
        signals = []
        for spectrogram, min_max_value in zip(spectrograms, min_max_values):
            log_spectrogram = spectrogram[:, :, 0]
            # Apply denormalisation
            denorm_log_spec = self._min_max_normaliser.denormalise(
                log_spectrogram, min_max_value["min"], min_max_value["max"])
            # Go from log spectrogram to spectrogram
            spec = librosa.db_to_amplitude(denorm_log_spec)
            # Apply inverse short time fourier transform
            signal = librosa.istft(spec, hop_length=self.hop_length)
            
            signals.append(signal)
        return signals
    



def load_fsdd(spectrograms_path):
    """
    Function to load the spectrograms generated in preprocessing.py
    """
    x_train = []
    file_paths = []
    for root, _, file_names in os.walk(spectrograms_path):
        for file_name in file_names:
            file_path = os.path.join(root, file_name)
            spectrogram = np.load(file_path) 
            x_train.append(spectrogram)
            file_paths.append(file_path)
    x_train = np.array(x_train)
    x_train = x_train[..., np.newaxis] 
    return x_train, file_paths


def select_spectrograms(spectrograms,
                        file_paths,
                        min_max_values,
                        num_spectrograms=2):
    """
    Function to select a number of spectrograms from the batch
    """
    sampled_indexes = np.random.choice(range(len(spectrograms)), num_spectrograms)
    sampled_spectrogrmas = spectrograms[sampled_indexes]
    file_paths = [file_paths[index] for index in sampled_indexes]
    sampled_min_max_values = [min_max_values[file_path] for file_path in
                           file_paths]
    print(file_paths)
    print(sampled_min_max_values)
    return sampled_spectrogrmas, sampled_min_max_values


def save_signals(signals, save_dir, sample_rate=22050):
    """
    Function to save the signal as .wav
    """
    for i, signal in enumerate(signals):
        save_path = os.path.join(save_dir, str(i) + ".wav")
        sf.write(save_path, signal, sample_rate)


def spect_to_reconstructed_audio(spectrogram_path,min_max_values_path, generated_sound_path,model_path):
    """
    Master function to go from spectrogram to a reconstructed audio
    """
    vae = VAE.load(model_path)
    sound_generator = SoundGenerator(vae, HOP_LENGTH)
    

    with open(min_max_values_path, "rb") as f:
        min_max_values = pickle.load(f)

    specs, file_paths = load_fsdd(spectrogram_path)

    sampled_specs, sampled_min_max_values = select_spectrograms(specs,
                                                                file_paths,
                                                                min_max_values,
                                                                1)

    signals, _ = sound_generator.generate(sampled_specs,
                                          sampled_min_max_values)

    
    save_signals(signals, generated_sound_path)



if __name__ == "__main__":
    # Example of audios reconstructed by the model

    
    # Model and sound generated are loaded
    vae = VAE.load("model/vae1200-150")
    sound_generator = SoundGenerator(vae, HOP_LENGTH)

    # Spectrograms and min max values are loaded
    with open(MIN_MAX_VALUES_PATH, "rb") as f:
        min_max_values = pickle.load(f)

    specs, file_paths = load_fsdd(SPECTROGRAMS_PATH)

    # Select a number of them
    sampled_specs, sampled_min_max_values = select_spectrograms(specs,
                                                                file_paths,
                                                                min_max_values,
                                                                10)

    # Go grom spectrograms to audio
    signals, _ = sound_generator.generate(sampled_specs,
                                          sampled_min_max_values)
    # Save the original audio
    original_signals = sound_generator.convert_spectrograms_to_audio(
        sampled_specs, sampled_min_max_values)

    # Save audio reconstructed by the model 
    save_signals(signals, SAVE_DIR_GENERATED)
    save_signals(original_signals, SAVE_DIR_ORIGINAL)