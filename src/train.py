import os
import numpy as np
from vae import VAE
from pathlib import Path

# Train parameters
LEARNING_RATE = 0.001
BATCH_SIZE = 128
EPOCHS = 150

# Spectrograms parameters

SPECTROGRAMS_PATH = "fsdd/spectrograms/"


def load_fsdd(spectrograms_path):
    """
    Loads spctrograms data from preprocessing.py
    """
    x_train = []
    for root, _, file_names in os.walk(spectrograms_path):
        for file_name in file_names:
            file_path = os.path.join(root, file_name)
            spectrogram = np.load(file_path)
            x_train.append(spectrogram)
    x_train = np.array(x_train)
    x_train = x_train[..., np.newaxis] 
    return x_train


def train(x_train, learning_rate, batch_size, epochs):
    ## 
    # Parameter tunings from paper :
    # https://arrow.tudublin.ie/cgi/viewcontent.cgi?article=1420&context=scschcomcon 
    
    vautoencoder = VAE(
        input_shape=(256, 64, 1),
        conv_filters=(32, 32,32,32),
        conv_kernels=(3, 3,3,3),
        conv_strides=(2, 2,2,2),
        latent_space_dim=128
    )
    vautoencoder.summary()
    vautoencoder.compile(learning_rate)
    vautoencoder.train(x_train, batch_size, epochs)
    return vautoencoder


if __name__ == "__main__":
    x_train = load_fsdd(SPECTROGRAMS_PATH)
    x_train_no_nans = np.nan_to_num(x_train,nan=np.nanmedian(x_train))
    vautoencoder = train(x_train_no_nans, LEARNING_RATE, BATCH_SIZE, EPOCHS)
    vautoencoder.save('model/vae1200-150/')
    
    
   