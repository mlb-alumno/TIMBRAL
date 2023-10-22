import os
import pickle
import librosa
import numpy as np


class Loader:
    """For loading the audio file"""

    def __init__(self, sample_rate, duration, mono):
        self.sample_rate = sample_rate
        self.duration = duration
        self.mono = mono

    def load(self, file_path):
        signal = librosa.load(file_path,
                              sr=self.sample_rate,
                              duration=self.duration,
                              mono=self.mono)[0]
        return signal



class LogSpectrogramExtractor:
    """Extracts log spectrograms (in dB) from a
    time-series signal.
    """

    def __init__(self, frame_size, hop_length):
        self.frame_size = frame_size
        self.hop_length = hop_length

    def extract(self, signal):
        stft = librosa.stft(signal,
                            n_fft=self.frame_size,
                            hop_length=self.hop_length)[:-1]
        spectrogram = np.abs(stft)
        log_spectrogram = librosa.amplitude_to_db(spectrogram)
        return log_spectrogram


class MinMaxNormaliser:
    """We apply minmax normalization"""

    def __init__(self, min_val, max_val):
        self.min = min_val
        self.max = max_val

    def normalise(self, array):
        norm_array = (array - array.min()) / (array.max() - array.min())
        norm_array = norm_array * (self.max - self.min) + self.min
        return norm_array

    def denormalise(self, norm_array, original_min, original_max):
        array = (norm_array - self.min) / (self.max - self.min)
        array = array * (original_max - original_min) + original_min
        return array


class Saver:
    """For saving the spectrograms and mix max amplitude values"""

    def __init__(self, feature_save_dir, min_max_values_save_dir):
        self.feature_save_dir = feature_save_dir
        self.min_max_values_save_dir = min_max_values_save_dir

    def save_feature(self, feature, file_path):
        save_path = self._generate_save_path(file_path)
        np.save(save_path, feature)
        return save_path

    def save_min_max_values(self, min_max_values):
        save_path = os.path.join(self.min_max_values_save_dir,
                                 "min_max_values.pkl")
        self._save(min_max_values, save_path)

    @staticmethod
    def _save(data, save_path):
        with open(save_path, "wb") as f:
            pickle.dump(data, f)

    def _generate_save_path(self, file_path):
        file_name = os.path.split(file_path)[1]
        save_path = os.path.join(self.feature_save_dir, file_name + ".npy")
        return save_path


class PreprocessingPipeline:
    """
    Preprocessing consists process:

    - load the files
    - from signal to log spectrogram with short time fourier transforms 
    (stft: a signal in the time-frequency domain by computing discrete Fourier transforms (DFT) 
    over short overlapping windows)
    - MinMax Normalization is applied
    - Normalised spectrogram is saved

    """

    def __init__(self):
        #self.padder = None
        self.extractor = None
        self.normaliser = None
        self.saver = None
        self.min_max_values = {}
        self._loader = None
        self._num_expected_samples = None

    @property
    def loader(self):
        return self._loader

    @loader.setter
    def loader(self, loader):
        self._loader = loader
        self._num_expected_samples = int(loader.sample_rate * loader.duration)

    def process(self, audio_files_dir,file_limit=699):
        for root, _, files in os.walk(audio_files_dir):
            for file in files[:file_limit]: # Change range for smaller set of data [:1200]
                file_path = os.path.join(root, file)
                self._process_file(file_path)
                print(f"Processed file {file_path}")
        self.saver.save_min_max_values(self.min_max_values)
    
    def process_individual(self, file_dir):
        self._process_file(file_dir)
        self.saver.save_min_max_values(self.min_max_values)

    def _process_file(self, file_path):
        signal = self.loader.load(file_path)
        feature = self.extractor.extract(signal)
        norm_feature = self.normaliser.normalise(feature)
        save_path = self.saver.save_feature(norm_feature, file_path)
        self._store_min_max_value(save_path, feature.min(), feature.max())

    def _store_min_max_value(self, save_path, min_val, max_val):
        self.min_max_values[save_path] = {
            "min": min_val,
            "max": max_val
        }

def audio_to_spectr(audio_origin_path, spectr_save_dir,minmax_save_dir,individual=True,file_limit=0):
        # Function to go from audio to spectrogram
        FRAME_SIZE = 512
        HOP_LENGTH = 256
        DURATION = 0.74  # seconds of audio processed
        SAMPLE_RATE = 22050
        MONO = True
        SPECTROGRAMS_SAVE_DIR = spectr_save_dir
        MIN_MAX_VALUES_SAVE_DIR = minmax_save_dir
        FILES_DIR = audio_origin_path   

        loader = Loader(SAMPLE_RATE, DURATION, MONO)
        log_spectrogram_extractor = LogSpectrogramExtractor(FRAME_SIZE, HOP_LENGTH)
        min_max_normaliser = MinMaxNormaliser(0, 1)
        saver = Saver(SPECTROGRAMS_SAVE_DIR, MIN_MAX_VALUES_SAVE_DIR)
        preprocessing_pipeline = PreprocessingPipeline()
        preprocessing_pipeline.loader = loader
        preprocessing_pipeline.extractor = log_spectrogram_extractor
        preprocessing_pipeline.normaliser = min_max_normaliser
        preprocessing_pipeline.saver = saver
        if individual:
            preprocessing_pipeline.process_individual(FILES_DIR)    
        else:
            preprocessing_pipeline.process(FILES_DIR,file_limit=file_limit)


if __name__ == "__main__":
    # Preprocessing a batch of audios for training

    FRAME_SIZE = 512
    HOP_LENGTH = 256
    DURATION = 0.74  # seconds of audio processed
    SAMPLE_RATE = 22050
    MONO = True

    SPECTROGRAMS_SAVE_DIR = "fsdd/spectrograms/"
    MIN_MAX_VALUES_SAVE_DIR = "fsdd/"
    FILES_DIR = "src/partial-dataset/"   # path of your sound files


    # instantiation of objects
    loader = Loader(SAMPLE_RATE, DURATION, MONO)
    log_spectrogram_extractor = LogSpectrogramExtractor(FRAME_SIZE, HOP_LENGTH)
    min_max_normaliser = MinMaxNormaliser(0, 1)
    saver = Saver(SPECTROGRAMS_SAVE_DIR, MIN_MAX_VALUES_SAVE_DIR)
    preprocessing_pipeline = PreprocessingPipeline()
    preprocessing_pipeline.loader = loader
    preprocessing_pipeline.extractor = log_spectrogram_extractor
    preprocessing_pipeline.normaliser = min_max_normaliser
    preprocessing_pipeline.saver = saver

    preprocessing_pipeline.process(FILES_DIR)
