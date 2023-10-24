# TIMBRAL
Linear interpolation in the latent space of a VAE for sound design

Flow:
1. preprocessing.py: preprocessing functions to turn audio to log spectrograms using librosa. Run it for turning dataset to spectrogram saved data.

2. vae.py: model architecture with keras.

3. train.py: script to train the model

4. newaudios.py: functions for turning from spectrograms to audios. Run it for making reconstruction tests (audio-tests/reconstructed-audios)

5. latentspaceanalysis.ipybn: sprectrogram and latent space visualization notebook

6. interpolation.py: uses preprocessing, loads the model and uses newaudios for generating the new interpolated audios (between two random audios from the dataset or two audios selected in the api). Run to test the function (generated audios in audio-tests/interpolation/generated)

7. app.py: flask API with css for deploying the web app (index.html goes to audios.html once clicked generate audios)