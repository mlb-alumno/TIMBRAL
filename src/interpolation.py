import random
from playsound import playsound
import os
import shutil
import librosa
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from pydub import AudioSegment
from vae import VAE
import newaudios
import preprocessing

def interpolate_2_audios_web(n_sounds=11):
    """
    Function that 
    - gets two random audios from the dataset
    - converts them to log spectrograms
    - gets the linear interpolation in the latent space of them (that gets feeded to the decoder)
    - goes from final spectrogram to audio displayed in the web
    - zips the files for download 

    (this is looped for various alpha interpolation values)
    """

    "We take two random audios from our dataset folder and we copy them into our test folder"
    path="src/partial-dataset/"
    save_direct = "src/audio-tests/interpolation/"
    files=os.listdir(path)
    sound1=path+random.choice(files)
    sound2=path+random.choice(files)
    
    shutil.copy(sound1, save_direct+'generated/original/sound1.wav') 
    shutil.copy(sound2, save_direct+'generated/original/sound2.wav') 

    "We then generate and visualize the log spectrograms. We also play the audios"
    preprocessing.audio_to_spectr(save_direct+'generated/original/sound1.wav',save_direct+'data/sound1/spec/',save_direct+'data/sound1')
    preprocessing.audio_to_spectr(save_direct+'generated/original/sound2.wav',save_direct+'data/sound2/spec/',save_direct+'data/sound2')

    audio1 , _ = newaudios.load_fsdd(save_direct+'data/sound1/spec/')
    audio2 , _ = newaudios.load_fsdd(save_direct+'data/sound2/spec/')

    my_alphas = np.arange(0,1.1,0.1) # Alpha parameter to modulate the interpolation

    vae2 = VAE.load("model/vae1200def-h")
    vae2.load_weights("model/vae1200def-h/weights.h5")
    _, latent1 = vae2.reconstruct(audio1)
    _, latent2 = vae2.reconstruct(audio2)
    generator= newaudios.SoundGenerator(vae2,hop_length=256)
    for i in range(len(my_alphas)):
        alpha = my_alphas[i]
        
        latent_interpolation = latent1*(1-alpha)+latent2*(alpha)
        final_spectr = vae2.decoder.predict(latent_interpolation)
        signals = generator.convert_spectrograms_to_audio(final_spectr,[{'min': -46+i,
'max': 44+i}])
        for j, signal in enumerate(signals):
                save_path = os.path.join(save_direct+'generated/', str(i)+str(j) + ".wav")
                sf.write(save_path, signal, 22050)
    
    sound1 = AudioSegment.from_file(f"src/audio-tests/interpolation/generated/00.wav", format="wav")
    combined = sound1
    for i in range(1,n_sounds):
        sound2 = AudioSegment.from_file(f"src/audio-tests/interpolation/generated/{i}0.wav", format="wav")
        combined += sound2
    combined.export("src/audio-tests/interpolation/generated/all.wav", format="wav")
    
    shutil.make_archive('src/audio-tests/interpolation/generated-audios', 'zip', 'src/audio-tests/interpolation/generated')


def interpolate_2_audios_web_choose(path1,path2,n_sounds=11):
    """
    Function that 
    - gets two audios chosen by the user from the sound-selection folder
    - converts them to log spectrograms
    - gets the linear interpolation in the latent space of them (that gets feeded to the decoder)
    - goes from final spectrogram to audio displayed in the web
    - zips the files for download 

    (this is looped for various alpha interpolation values)
    """
    
    save_direct = "src/audio-tests/interpolation/"
    
    shutil.copy(path1, save_direct+'generated/original/sound1.wav') 
    shutil.copy(path2, save_direct+'generated/original/sound2.wav') 

    "We then generate and visualize the log spectrograms. We also play the audios"
    preprocessing.audio_to_spectr(save_direct+'generated/original/sound1.wav',save_direct+'data/sound1/spec/',save_direct+'data/sound1')
    preprocessing.audio_to_spectr(save_direct+'generated/original/sound2.wav',save_direct+'data/sound2/spec/',save_direct+'data/sound2')

    audio1 , _ = newaudios.load_fsdd(save_direct+'data/sound1/spec/')
    audio2 , _ = newaudios.load_fsdd(save_direct+'data/sound2/spec/')

    my_alphas = np.arange(0,1.1,0.1) # Alpha parameter to modulate the interpolation

    vae2 = VAE.load("model/vae1200def-h")
    vae2.load_weights("model/vae1200def-h/weights.h5")
    _, latent1 = vae2.reconstruct(audio1)
    _, latent2 = vae2.reconstruct(audio2)
    generator= newaudios.SoundGenerator(vae2,hop_length=256)
    for i in range(len(my_alphas)):
        alpha = my_alphas[i]
        
        latent_interpolation = latent1*(1-alpha)+latent2*(alpha)
        final_spectr = vae2.decoder.predict(latent_interpolation)
        signals = generator.convert_spectrograms_to_audio(final_spectr,[{'min': -46+i,
'max': 44+i}])
        for j, signal in enumerate(signals):
                save_path = os.path.join(save_direct+'generated/', str(i)+str(j) + ".wav")
                sf.write(save_path, signal, 22050)
    
    sound1 = AudioSegment.from_file(f"src/audio-tests/interpolation/generated/00.wav", format="wav")
    combined = sound1
    for i in range(1,n_sounds):
        sound2 = AudioSegment.from_file(f"src/audio-tests/interpolation/generated/{i}0.wav", format="wav")
        combined += sound2
    combined.export("src/audio-tests/interpolation/generated/all.wav", format="wav")

    shutil.make_archive('src/audio-tests/interpolation/generated-audios', 'zip', 'src/audio-tests/interpolation/generated')




def interpolate_2_audios_local():
    """
    Function that 
    - gets two random audios from the dataset
    - converts them to log spectrograms
    - visualizes them and plays the audios.
    - gets the linear interpolation in the latent space of them (that gets feeded to the decoder)
    - goes from final spectrogram to audio 

    (this is looped for various alpha interpolation values)
    """

    "We take two random audios from our dataset folder and we copy them into our test folder"
    path="src/partial-dataset/"
    save_direct = "src/audio-tests/interpolation/"
    files=os.listdir(path)
    sound1=path+random.choice(files)
    sound2=path+random.choice(files)
    
    shutil.copy(sound1, save_direct+'/original/sound1.wav') 
    shutil.copy(sound2, save_direct+'/original/sound2.wav') 

    "We then generate and visualize the log spectrograms. We also play the audios"
    preprocessing.audio_to_spectr(save_direct+'original/sound1.wav',save_direct+'data/sound1/spec/',save_direct+'data/sound1')
    preprocessing.audio_to_spectr(save_direct+'original/sound2.wav',save_direct+'data/sound2/spec/',save_direct+'data/sound2')

    audio1 , _ = newaudios.load_fsdd(save_direct+'data/sound1/spec/')
    _show_spectrogram(audio1)
    playsound(sound1)
    audio2 , _ = newaudios.load_fsdd(save_direct+'data/sound2/spec/')
    _show_spectrogram(audio2)
    playsound(sound2)

    my_alphas = np.arange(0,1,0.1) # Alpha parameter to modulate the interpolation

    print('Generating interpolations')

    vae2 = VAE.load("model/vae1200def-h")
    vae2.load_weights("model/vae1200def-h/weights.h5")
    _, latent1 = vae2.reconstruct(audio1)
    _, latent2 = vae2.reconstruct(audio2)
    generator= newaudios.SoundGenerator(vae2,hop_length=256)
    for i in range(len(my_alphas)):
        alpha = my_alphas[i]
        print('Working on alpha {:2.1f}'.format(alpha))
        
        latent_interpolation = latent1*(1-alpha)+latent2*(alpha)
        final_spectr = vae2.decoder.predict(latent_interpolation)
        signals = generator.convert_spectrograms_to_audio(final_spectr,[{'min': -46,
'max': 44}])
        for j, signal in enumerate(signals):
                save_path = os.path.join(save_direct+'generated/', str(i)+str(j) + ".wav")
                sf.write(save_path, signal, 22050)
                playsound(save_path)
    plt.show()

def _show_spectrogram(audio1):
    fig, ax = plt.subplots()
    img = librosa.display.specshow(np.squeeze(audio1[-1]), x_axis='time', y_axis='log', ax=ax,hop_length=256,win_length=512)
    ax.set(title='Using a logarithmic frequency axis')
    fig.colorbar(img, ax=ax, format="%+2.f dB")
    plt.show(block=False)
    plt.pause(0.001)


if __name__ == "__main__":
    #interpolate_2_audios_local()
        interpolate_2_audios_web()

    
