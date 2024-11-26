import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from analysis import save_figure_to_list  # Importe a função para salvar imagens

# Process a single audio file and extract MFCC features
def process_single_audio(file_path, class_label, n_mfcc=20):
    try:
        audio, sample_rate = librosa.load(file_path, sr=None)
        n_fft = 512 if len(audio) < 2048 else 2048
        mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc, n_fft=n_fft, n_mels=20)
        mfcc_scaled = np.mean(mfcc.T, axis=0)
        return mfcc_scaled, class_label
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# Load and process multiple audio files using parallel processing
def load_audio_files(data_path, metadata, n_mfcc=20):
    from concurrent.futures import ThreadPoolExecutor

    X, y = [], []
    with ThreadPoolExecutor() as executor:
        futures = []
        for i in range(len(metadata)):
            file_path = os.path.join(data_path, f"fold{metadata.fold[i]}", metadata.slice_file_name[i])
            class_label = metadata['classID'][i]
            futures.append(executor.submit(process_single_audio, file_path, class_label, n_mfcc))
        
        for future in futures:
            result = future.result()
            if result:
                mfcc_scaled, class_label = result
                X.append(mfcc_scaled)
                y.append(class_label)
    return np.array(X), np.array(y)

def plot_spectrograms_per_class_from_files(data_path, metadata, class_names):
    """
    Plot spectrograms for each class using an example file and save to the global image list.
    """
    unique_classes = np.unique(metadata['classID'])

    for class_id in unique_classes:
        try:
            # Obter um exemplo do arquivo para a classe atual
            example_row = metadata[metadata['classID'] == class_id].iloc[0]
            file_path = os.path.join(data_path, f"fold{example_row['fold']}", example_row['slice_file_name'])

            # Carregar o arquivo de áudio
            audio, sr = librosa.load(file_path, sr=None)

            # Calcular o espectrograma
            n_fft = min(2048, len(audio) // 2)
            hop_length = n_fft // 4
            D = np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop_length))
            S_db = librosa.amplitude_to_db(D, ref=np.max)

            # Plotar o espectrograma
            plt.figure(figsize=(10, 6))
            librosa.display.specshow(S_db, sr=sr, hop_length=hop_length, x_axis='time', y_axis='log', cmap='viridis')
            plt.colorbar(format='%+2.0f dB')
            plt.title(f"Spectrogram - Class {class_names[class_id]}")
            plt.xlabel("Time (s)")
            plt.ylabel("Frequency (Hz)")

            # Salvar a figura na lista de imagens
            save_figure_to_list()

        except Exception as e:
            print(f"Error processing file for class {class_id}: {e}")