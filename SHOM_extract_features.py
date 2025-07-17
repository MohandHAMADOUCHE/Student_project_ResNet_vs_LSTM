from audio_processing import extract_features
from audio_processing import visualize_features
import os
import numpy as np
import librosa
import librosa.display
import matplotlib
matplotlib.use('TkAgg',force=True)
from matplotlib import pyplot as plt
from matplotlib.widgets import Button
from spafe.features import gfcc
from spafe.utils.preprocessing import SlidingWindow
from spafe.fbanks.gammatone_fbanks import gammatone_filter_banks
import scipy



def read_audio(file_path, first_frame =0,last_frame =-1):
    """
    Applique le pre-emphasis, framing, and windowing au signal audio.

    Args:
        file_path (str) : Chemin d'acc�s au fichier audio.
        frame_length (int) : Longueur de chaque trame.
        hop_length (int) : Chevauchement entre les images.
        show_example (bool) : Indique s'il faut le signal audio et le signal accentu�.

    Returns:
        tuple: Emphasized audio, framed and windowed signal, sample rate.
    """
    audio, sample_rate = librosa.load(file_path, sr=None)
    print('audio size = ', str(audio.size))

    # audio = audio[200000::] # pour le flac ressach (bruit elec au d�but)
    audio = audio[first_frame:last_frame] # pour le wav d'OBS ressach (audio bcp trop long..)
    return audio, sample_rate
def emphasized_audio(audio):
    """
    Applique le pre-emphasis, framing, and windowing au signal audio.

    Args:
        file_path (str) : Chemin d'acc�s au fichier audio.
        frame_length (int) : Longueur de chaque trame.
        hop_length (int) : Chevauchement entre les images.
        show_example (bool) : Indique s'il faut le signal audio et le signal accentu�.

    Returns:
        tuple: Emphasized audio, framed and windowed signal, sample rate.
    """
    # Apply pre-emphasis
    pre_emphasis = 0.97
    emphasized_audio = np.append(audio[0], audio[1:] - pre_emphasis * audio[:-1])
    return emphasized_audio
def decimate_audio(audio, fe, nfe):
    """
    :param audio: (1-D array) of the signal to decimate
    :param t: (1-D array) of the time (could be retrieved from fs...)
    :param fe: sampling frequency of the input signal
    :param nfe: new sampling frequency to consider
    :return: decimated_signal, t_decim, decimation_factor, nfe
    """

    decimation_factor = int(fe / nfe)
    if decimation_factor == 1:
        decimated_signal = audio
    else :
        # t = np.arange(0,1/fe*(len(audio)),1/fe)
        # Apply filter before decimating the signal
        # wnfilt = 2*np.pi*nfe # pulsation uniquement pour filtre analogue
        wnfilt = 1/decimation_factor # fr�quence normalis�e par f nyquist
        b, a = scipy.signal.butter(8, wnfilt, 'low')
        filtered_signal = scipy.signal.filtfilt(b, a, audio)

        decimated_signal = scipy.signal.decimate(filtered_signal, decimation_factor)
        # t_decim = t[::decimation_factor]
        # nfe = fe / decimation_factor
    return decimated_signal, fe/decimation_factor

def extract_stft(afile, nfft=2048, hop_length=512):
    rpz = librosa.stft(afile, n_fft=nfft, hop_length=hop_length)
    return rpz

def extract_mel(afile, fe, nfft=2048, hop_length=512, nmels=128):
    rpz = librosa.feature.melspectrogram(y=afile, sr=fe, n_fft=nfft, hop_length=hop_length, n_mels = nmels)
    # Calculate the time vector
    t_mel = librosa.times_like(rpz, sr=fe, hop_length=hop_length)
    # Calculate the frequency vector
    f_mel = librosa.mel_frequencies(n_mels=nmels, fmin=0, fmax=fe/2)

    return rpz, t_mel, f_mel

# def extract_mfcc(afile, fs, n_filters):
#     rpz = librosa.feature.mfcc(y=afile, sr=fs, n_mfcc=n_filters)
#     return rpz
# def extract_gfcc(sig,fe,nfft,hop_length,nfilters,nceps=100):
#     # stft = extract_stft(afile, nfft=nfft, hop_length=hop_length)
#     # rpz = librosa.feature.mfcc(S=librosa.amplitude_to_db(stft), n_mfcc=nfilters)#la variable S est cens�e �tre un log scaled mel spectrogram
#     winwin = SlidingWindow(win_len=nfft/fe, win_hop=hop_length/fe, win_type='hamming')
#     rpz = gfcc.gfcc(sig, fs=fe, num_ceps=nceps,window = winwin, nfilts=nfilters, nfft=nfft)#, low_freq=None, high_freq=None,
#     return rpz
def extract_cqt(afile, fs, nb, nbin,f_min=1):
    rpz = np.abs(librosa.cqt(y=afile, sr=fs, bins_per_octave=nb, n_bins = nbin,fmin=f_min))
    t_cqt = librosa.times_like(rpz, sr=fs)
    f_cqt = librosa.cqt_frequencies(n_bins=nbin, fmin=f_min, bins_per_octave=nb)

    return rpz, t_cqt, f_cqt

def extract_lofar(audio,fs,nfft=32768,hop_length=32768/4, freq_range=[10,500],df=5):
    # resample audio depending on freq range
    # audioLF, feLF = decimate_audio(audio, fs, 4*freq_range[1])
    # Compute the STFT
    stft_result = extract_stft(audio,nfft, hop_length)
    # Convert to decibels
    stft_db = librosa.amplitude_to_db(np.abs(stft_result), ref=np.max)
    # Get the frequency bins corresponding to the STFT
    freqs = librosa.fft_frequencies(sr=fs, n_fft=nfft)
    # # Filter frequencies
    min_freq, max_freq = freq_range
    freq_indices = [i for i, f in enumerate(freqs) if min_freq <= f <= max_freq]
    # Downsample to match the desired frequency interval
    target_freqs = np.arange(min_freq, max_freq + df, df)
    lofar_spectrum = []
    for target_freq in target_freqs:
        # Find the closest frequency bin to the target frequency
        closest_index = np.argmin(np.abs(freqs[freq_indices] - target_freq))
        lofar_spectrum.append(stft_db[freq_indices[closest_index], :])
        # Stack the results into a matrix of shape (51, number of frames)
    rpz = np.array(lofar_spectrum)
    # Calculate the time vector

    t_lofar = librosa.times_like(stft_result, sr=fs, hop_length=int(hop_length))

    # Calculate the frequency vector
    f_lofar = target_freqs

    return rpz, t_lofar, f_lofar

def compute_cepstrogram(rpz, fe, analytic=False):
    """
    Compute the cepstrogram of a time-frequency representation.

    Cepstrogram can be usefull to compute time difference between different
    arrivals or to study the harmonic structure of a signal.

    Parameters
    ----------
    analytic: bool, optional
        Whether to return the analytical signal of the cepstrogram.


    """
    f = np.arange(0,fe/2,fe/2/rpz.shape[0])
    df = f[1] - f[0]
    q = np.fft.rfftfreq(2*(f.size - 1), df)
    data = np.log(np.abs(rpz))
    if analytic:
        data = np.concatenate((
            data[0:1, :],
            2*data[1:-1, :],
            data[-1:, :],
            0*data[-2:0:-1, :]), axis=-2)
        data = np.fft.ifft(data, axis=-2)
    else:
        data = np.fft.irfft(data, axis=-2)
    data = data[..., :q.size, :]
    return data, q

def remove_mean(rpz):
    "enl�ve la moyenne temporelle, ligne par ligne"
    demean_rpz = rpz - np.dot(np.array([np.ones(rpz.shape[1])]).T, [np.mean(rpz, axis=1)]).T
    return demean_rpz
def remove_neg_et_zeros(rpz,seuil=0.001):
    new_rpz = rpz
    for i in range(np.shape(rpz)[0]):
        for j in range(np.shape(rpz)[1]):
            if rpz[i, j] < seuil:
                new_rpz[i, j] = seuil
    return new_rpz
def translate_positif_dom(rpz,seuil=0.001):
    new_rpz = rpz
    minimum = np.min(rpz)
    if minimum <=0:
        new_rpz = rpz - minimum + seuil

    return new_rpz




# file_path = "/tools/mohand_postdoc/datasets/DeepShip/ClassesrandomlySplitted/train_DeepShip_Segments_5000/Cargo/99_segment_3060.wav"
# file_path = os.path.abspath("./Segments/Cargo/99_segment_3060.wav")
# first_frame = 0
# file_path = os.path.abspath("./Segments/Passengership/12_segment_0240.wav")
# first_frame = 0
# last_frame = first_frame + 15*32000
# file_path = os.path.abspath("./Segments/Tug/020446_segment_1088.wav")
# first_frame = 0
file_path = os.path.abspath("./5084.220822235505.flac")
# first_frame = 200000
# last_frame = first_frame + 60*48000
# file_path = os.path.abspath("./Segments/Tanker/6_segment_1786.wav")
# first_frame = 0
# file_path = os.path.abspath("D:\enreg_annotation/channelH_2024-10-03_01-01-57.wav")
# first_frame = 45000000
# last_frame = first_frame + 5*60*2000
# first_frame = 45000000
# first_frame = 70000000
# last_frame = first_frame + 5*60*2000
# last_frame = -1
# file_path = os.path.abspath("./Segments/RESSACH/SampleVocClickBateau_channelA_2019-09-08_13-20-06.wav")
# first_frame = 0
# last_frame = first_frame + 15*39062
# file_path = os.path.abspath("./Segments/RESSACH/channelA_2019-09-17_05-17-29.wav")
first_frame = 24700000-15*39000
# last_frame = first_frame + 3*60*39000
last_frame = first_frame + 15*39000




# file_path = os.path.abspath("D:/Donnees_UBO_1/Annotations_complete_ubo/Annotations/6338.210818115958.flac")
# first_frame = 45000000
# last_frame = first_frame + 15*288000


######    LECTURE DE L'AUDIO ET PRETRAITEMENTS   ######
audio, fe = read_audio(file_path, first_frame,last_frame)
emphasized_audio = emphasized_audio(audio)
audioBF, feBF = decimate_audio(audio,fe,2000)
# audioBF, feBF = decimate_audio(audio,fe,30000)
t = np.arange(0,1/fe*(len(audioBF)),1/fe)

######    PARAMETRES GENERAUX    ######
nfft = 2048
hop_length = 512

######    PARAMETRES ET CALCULS SPECTROGRAMME    ######
f_stft = librosa.fft_frequencies(sr=feBF, n_fft=nfft)
stft = extract_stft(audioBF,nfft = nfft, hop_length=hop_length)
t_stft = np.arange(0,hop_length/feBF*(np.shape(stft)[1]),hop_length/feBF)
rpz1 = np.log10(np.abs(stft))

######    PARAMETRES ET CALCULS CEPSTROGRAMME    ######
cepstro,q = compute_cepstrogram(stft,feBF,True)
rpz6 = np.log10(remove_neg_et_zeros(np.real(cepstro)))
rpz7 = np.log10(remove_neg_et_zeros(np.imag(cepstro)))

cepstro_imag_demean = remove_mean(np.imag(cepstro))
cepstro_imag_demean = remove_neg_et_zeros(cepstro_imag_demean)

cepstro_real_demean = remove_mean(np.real(cepstro))
cepstro_real_demean = remove_neg_et_zeros(cepstro_real_demean)

rpz8 = np.log10(cepstro_real_demean)
rpz9 = np.log10(cepstro_imag_demean)

######    PARAMETRES ET CALCULS LOFAR    ######
nfft_lfr = 2048*8
hop_length_lfr = hop_length
lofar, t_lofar, f_lofar = extract_lofar(audioBF,feBF,nfft= nfft_lfr ,
                                        hop_length=hop_length_lfr, freq_range=[10,500],df=2)
tLF = np.arange(0,hop_length_lfr/feBF*np.shape(lofar)[1],hop_length_lfr/feBF)

new_lofar = np.zeros((len(f_lofar),len(t_stft)))
for j in range(len(f_lofar)):
    new_lofar[j,:] = np.interp(t_stft, tLF,lofar[j,:])
rpz2 = new_lofar

######    PARAMETRES ET CALCULS CQT    ######
nbinperoct = 45
fmin = 3.9375
# nbin = 350
# nbin est calcul� pour que la fr�quence la plus haute soit proche de 1000 Hz pour exploiter toute la bande passante sur signal r��chantillonn�
nbin = int(nbinperoct*np.log10(feBF/2/fmin)/np.log10(2))
# le seul param�tre qui controle la r�solution temporelle du cqt est la limite HF du signal est une fe � 2000 Hz limite vraiment cette r�solution, de mani�re assez �trange.
# Plus le signal contient des HF, meilleure est la r�solution. Du coup pour avoir une r�solution t acceptable, et suffisamment d'�chantillons par la m�me occasion, j'ai
# appliqu� le calcul sur le signal non r��chantillonn�. Fonctionne aussi sur signaux BF type OBS ?

cqt, t_cqt, f_cqt = extract_cqt(audio, fe, nbinperoct, nbin-nbinperoct, f_min=fmin*2)
# Projection sur une �chelle temporelle qui match la stft
new_cqt = np.zeros((len(f_cqt),len(t_stft)))
for j in range(len(f_cqt)):
    new_cqt[j,:] = np.interp(t_stft, t_cqt,cqt[j,:])
cqt = new_cqt
rpz3 = np.log10(cqt)

######    PARAMETRES ET CALCULS MEL SPECTROGRAMME    ######
# f_mel= np.arange(0,fe/2,fe/2/128)
n_filt_mel = 200
nfft_mel = nfft
hop_mel = hop_length

taille = int((len(audioBF)-nfft_mel)/hop_mel)+5
ind = np.arange(0,taille,1)
t_mel = (hop_mel/feBF)*ind

print(len(t_mel))
f_stft_m = librosa.fft_frequencies(sr=feBF, n_fft=nfft_mel)
mel, t_mel, f_mel = extract_mel(audioBF, feBF, nfft=nfft_mel, hop_length=hop_mel, nmels=n_filt_mel)

rpz4 = np.log10(mel)

######    PARAMETRES ET CALCULS MFCC    ######
MFCC,qM = compute_cepstrogram(np.abs(mel),feBF,True)
rpz10 = np.log10(remove_neg_et_zeros(np.real(MFCC)))
# rpz11 = np.log10(np.imag(MFCC))
MFCC_real_demean = remove_mean(np.real(MFCC))
MFCC_real_demean = remove_neg_et_zeros(MFCC_real_demean)
rpz11 = np.log10(MFCC_real_demean)

MFCC_imag_demean = remove_mean(np.imag(MFCC))
MFCC_imag_demean = remove_neg_et_zeros(MFCC_imag_demean)
rpz12 = np.log10(MFCC_imag_demean)

######    PARAMETRES ET CALCULS GAMMATONE FREQ SPECTROGRAMME    ######
n_filt_GT = 200
nfft_gt = int(nfft)

hop_gt = hop_length
freq = librosa.fft_frequencies(sr=feBF, n_fft=nfft_gt)
f = feBF
winwin = SlidingWindow(win_len=nfft_gt/feBF, win_hop=hop_gt/feBF, win_type='hamming')
features, fourrier_transform = gfcc.erb_spectrogram(sig=audioBF, fs=f, window=winwin, nfilts=n_filt_GT,
                                                    nfft=nfft_gt, low_freq=0.1, high_freq=f/2, scale='constant')

t_gt = np.arange(0,int((len(audioBF)-nfft_gt)/hop_gt +1)*hop_gt/f,hop_gt/f)
filtres_GT,trash = gammatone_filter_banks(nfft=nfft_gt,fs=feBF,nfilts=n_filt_GT,low_freq=0.10, high_freq = feBF/2)

for i in range(len(filtres_GT)):
    for j in range(len(filtres_GT[i])):
        if filtres_GT[i,j] < 0.1 :
            filtres_GT[i,j] = 0

f_GT = [np.sum(freq*filtres_GT[i])/np.sum(filtres_GT[i]) for i in range(len(filtres_GT))]

# Projection sur une �chelle temporelle qui match la stft
new_GT = np.zeros((len(f_GT),len(t_stft)))
for j in range(len(f_GT)):
    new_GT[j,:] = np.interp(t_stft, t_gt,features.T[j,:])
features = new_GT.T
rpz5 = np.log10(features.T)

######    PARAMETRES ET CALCULS GFCC    ######
GFCC,qM = compute_cepstrogram(np.abs(features.T),feBF,True)
rpz13 = np.log10(remove_neg_et_zeros(np.real(GFCC)))

GFCC_real_demean = remove_mean(np.real(GFCC))
GFCC_real_demean = remove_neg_et_zeros(GFCC_real_demean)
rpz14 = np.log10(GFCC_real_demean)

GFCC_imag_demean = remove_mean(np.imag(GFCC))
GFCC_imag_demean = remove_neg_et_zeros(GFCC_imag_demean)
rpz15 = np.log10(GFCC_imag_demean)



print('rpz1.shape',rpz1.shape)
print('rpz2.shape',rpz2.shape)
print('rpz3.shape',rpz3.shape)
print('rpz4.shape',rpz4.shape)
print('rpz5.shape',rpz5.shape)
print('rpz6.shape',rpz6.shape)
print('rpz7.shape',rpz7.shape)
print('rpz8.shape',rpz8.shape)
print('rpz9.shape',rpz9.shape)
print('rpz10.shape',rpz10.shape)
print('rpz11.shape',rpz11.shape)
print('rpz12.shape',rpz12.shape)
print('rpz13.shape',rpz13.shape)
print('rpz14.shape',rpz14.shape)
print('rpz15.shape',rpz15.shape)


fig, axes = plt.subplots(2, 3, figsize=(30,25), sharex=False)
img1 = axes[0,0].pcolormesh(rpz1,cmap='viridis')
axes[0,0].set_title("rpz1")
fig.colorbar(img1)

img2 = axes[0,1].pcolormesh(rpz2,cmap='viridis')
axes[0,1].set_title("rpz2")
fig.colorbar(img2)

img3 = axes[0,2].pcolormesh(rpz3,cmap='viridis')
axes[0,2].set_title("rpz3")
fig.colorbar(img3)

img4 = axes[1,0].pcolormesh(rpz4,cmap='viridis')
axes[1,0].set_title("rpz4")
fig.colorbar(img4)

img5 = axes[1,1].pcolormesh(rpz5,cmap='viridis')
axes[1,1].set_title("rpz5")
fig.colorbar(img5)

img6 = axes[1,2].pcolormesh(rpz6,cmap='viridis')
axes[1,2].set_title("rpz6")
fig.colorbar(img6)

fig, axes = plt.subplots(2, 3, figsize=(30,25), sharex=False)
img1 = axes[0,0].pcolormesh(rpz7,cmap='viridis')
axes[0,0].set_title("rpz7")
fig.colorbar(img1)

img1 = axes[0,1].pcolormesh(rpz8,cmap='viridis')
axes[0,1].set_title("rpz8")
fig.colorbar(img1)

img1 = axes[0,2].pcolormesh(rpz9,cmap='viridis')
axes[0,2].set_title("rpz9")
fig.colorbar(img1)

img1 = axes[1,0].pcolormesh(rpz10,cmap='viridis')
axes[1,0].set_title("rpz10")
fig.colorbar(img1)

img1 = axes[1,1].pcolormesh(rpz11,cmap='viridis')
axes[1,1].set_title("rpz11")
fig.colorbar(img1)

img1 = axes[1,2].pcolormesh(rpz12,cmap='viridis')
axes[1,2].set_title("rpz12")
fig.colorbar(img1)

fig, axes = plt.subplots(2, 3, figsize=(30,25), sharex=False)

img1 = axes[1,0].pcolormesh(rpz13,cmap='viridis')
axes[1,0].set_title("rpz13")
fig.colorbar(img1)

img1 = axes[1,1].pcolormesh(rpz14,cmap='viridis')
axes[1,1].set_title("rpz14")
fig.colorbar(img1)

img1 = axes[1,2].pcolormesh(rpz15,cmap='viridis')
axes[1,2].set_title("rpz15")
fig.colorbar(img1)