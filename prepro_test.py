import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.preprocessing import StandardScaler
from scipy.signal import convolve
from scipy.ndimage import zoom

def reduce_dimension(features, labels, target_dim):
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    nca = NeighborhoodComponentsAnalysis(n_components=target_dim, random_state=42)
    features_reduced = nca.fit_transform(features_scaled, labels)
    return features_reduced

def gammatone_filterbank(signal, sample_rate, n_filters=64, f_min=50, filter_length=1024):
    f_max = sample_rate / 2
    center_frequencies = np.logspace(np.log10(f_min), np.log10(f_max), n_filters)
    filtered_signals = []

    for f in center_frequencies:
        bandwidth = 1.019 * (24.7 * (4.37 * f / 1000 + 1))  # Banda ERB
        t = np.linspace(0, (filter_length - 1) / sample_rate, filter_length)
        impulse_response = t ** 3 * np.exp(-2 * np.pi * bandwidth * t) * np.cos(2 * np.pi * f * t)
        filtered_signal = convolve(signal, impulse_response, mode='same')
        filtered_signals.append(np.abs(filtered_signal))

    return np.array(filtered_signals)

def preprocess_audio(file_path):
    print(f"Carregando áudio do arquivo: {file_path}")
    audio, sample_rate = librosa.load(file_path, sr=None)

    # Pré-ênfase
    audio = np.append(audio[0], audio[1:] - 0.97 * audio[:-1])
    print(f"Áudio carregado e pré-ênfase aplicada. Taxa de amostragem: {sample_rate}")

    # Enquadramento e aplicação de janela
    hop_length = 512
    frame_length = 2048
    windowed = librosa.stft(audio, n_fft=frame_length, hop_length=hop_length)
    print(f"Transformada de Short-Time Fourier aplicada. Forma da matriz de janela: {windowed.shape}")

    return audio, sample_rate, windowed

def extract_mfcc(audio, sample_rate, n_mfcc=30, n_fft=2048, hop_length=512):
    print("Extraindo características MFCC...")
    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    print(f"Características MFCC extraídas. Forma: {mfcc.shape}")
    return mfcc

def extract_gfcc(audio, sample_rate, n_filters=64):
    print("Extraindo características GFCC...")
    gammatone_bank = gammatone_filterbank(audio, sample_rate, n_filters=n_filters)
    if gammatone_bank.ndim == 1:
        gfcc = gammatone_bank.flatten()  # Se apenas uma linha, convertê-la adequadamente
    else:
        gfcc = np.log1p(np.mean(gammatone_bank, axis=1))
    print(f"Características GFCC extraídas. Forma: {gfcc.shape}")
    return gfcc

def extract_cqt(audio, sample_rate):
    print("Extraindo características CQT...")
    cqt = np.abs(librosa.cqt(y=audio, sr=sample_rate, hop_length=512))
    print(f"Características CQT extraídas. Forma: {cqt.shape}")
    return cqt

def extract_lofar(audio, sample_rate, n_fft=2048, hop_length=512):
    print("Extraindo características LOFAR...")
    stft = np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop_length))
    return stft[:int(500 / (sample_rate / n_fft))]  # Filtro de 0 a 500 Hz

def calculate_delta(features):
    print("Calculando características delta...")
    # Verificar se o número de frames (eixo 1) é suficiente
    if features.shape[1] < 9:  # 'width=9' é o padrão no delta do Librosa
        padding = 9 - features.shape[1]
        features = np.pad(features, ((0, 0), (0, padding)), mode='edge')
    delta = librosa.feature.delta(features)
    print(f"Características delta calculadas. Forma: {delta.shape}")
    return delta

def resize_features(features, target_shape):
    print("Redimensionando características...")
    if features.shape[1] != target_shape:
        scale = target_shape / features.shape[1]
        features = zoom(features, (1, scale), order=1)  # Interpolação linear
    print(f"Características redimensionadas para: {features.shape}")
    return features

def preprocess_and_fuse(audio, sample_rate, labels, target_dim):
    # Extrair características
    mfcc = extract_mfcc(audio, sample_rate)  # Dimensão (n_mfcc, frames)
    gfcc = extract_gfcc(audio, sample_rate)  # Pode ser (n_filters, frames)
    cqt = extract_cqt(audio, sample_rate)    # Dimensão (bins, frames)
    lofar = extract_lofar(audio, sample_rate)  # Dimensão (freq_bins, frames)

    # Garantir que todas as características tenham pelo menos duas dimensões
    if len(gfcc.shape) == 1:
        gfcc = np.expand_dims(gfcc, axis=1)  # Adicionar dimensão temporal (n_filters, 1)

    # Determinar o número alvo de frames (usamos a maior dimensão encontrada)
    target_frames = max(mfcc.shape[1], gfcc.shape[1], cqt.shape[1], lofar.shape[1])

    # Ajustar dimensões temporais
    mfcc = resize_features(mfcc, target_frames)
    gfcc = resize_features(gfcc, target_frames)
    cqt = resize_features(cqt, target_frames)
    lofar = resize_features(lofar, target_frames)

    # Cálculo das características Delta
    mfcc_delta = calculate_delta(mfcc)
    gfcc_delta = calculate_delta(gfcc)
    cqt_delta = calculate_delta(cqt)
    lofar_delta = calculate_delta(lofar)

    # Fusão inicial das características
    fused_features = np.concatenate([mfcc, gfcc, cqt, lofar], axis=0)  # Dimensão (total_features, frames)
    fused_delta = np.concatenate([mfcc_delta, gfcc_delta, cqt_delta, lofar_delta], axis=0)  # Dimensão (total_features, frames)
    fused_combined = np.concatenate([fused_features, fused_delta], axis=0)  # Dimensão (2 * total_features, frames)

    # Redução de dimensionalidade
    fused_combined_flat = fused_combined.reshape(fused_combined.shape[0], -1).T  # Dimensão (frames, 2 * total_features)

    # Replicar rótulos para corresponder ao número de frames
    repeated_labels = np.tile(labels, fused_combined_flat.shape[0] // len(labels) + 1)[:fused_combined_flat.shape[0]]

    # Reduzir dimensionalidade com NCA
    fused_reduced = reduce_dimension(fused_combined_flat, repeated_labels, target_dim)  # Reduz para target_dim dimensões

    return fused_reduced

def plot_features(features_list, titles, output_file=None):
    plt.figure(figsize=(15, 10))
    for i, (feature, title) in enumerate(zip(features_list, titles)):
        plt.subplot(2, 2, i + 1)
        librosa.display.specshow(feature, x_axis='time', y_axis='mel', cmap='viridis')
        plt.colorbar(format='%+2.0f dB')
        plt.title(title)
    plt.tight_layout()
    if output_file:
        plt.savefig(output_file)
    else:
        plt.show()

def process_and_visualize(data_path, labels, target_dim=96):
    for class_folder in os.listdir(data_path):
        class_path = os.path.join(data_path, class_folder)
        if os.path.isdir(class_path):
            audio_files = [f for f in os.listdir(class_path) if f.endswith(('.wav', '.mp3'))]
            if not audio_files:
                continue

            # Selecionar um áudio por classe
            file_path = os.path.join(class_path, audio_files[0])
            audio, sample_rate, _ = preprocess_audio(file_path)

            # Preprocessar e fundir características
            fused_features = preprocess_and_fuse(audio, sample_rate, labels, target_dim)

            # Plotar características originais e delta
            mfcc = extract_mfcc(audio, sample_rate)
            gfcc = extract_gfcc(audio, sample_rate)
            cqt = extract_cqt(audio, sample_rate)
            lofar = extract_lofar(audio, sample_rate)
            plot_features(
                [mfcc, gfcc, cqt, lofar],
                ["MFCC", "GFCC", "CQT", "LOFAR"],
                output_file=f"features_{class_folder}.png"
            )

# Caminho dos dados de entrada e rótulos
data_path = r"C:\\Users\\Gustavo\\Desktop\\deepship"
labels = [0, 1, 2, 3]  # Substitua pelos rótulos apropriados para cada classe
process_and_visualize(data_path, labels)
