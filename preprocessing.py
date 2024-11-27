import json
import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical # type: ignore
from analysis import save_figure_to_list
from utils import config, validate_path  # Importe a função para salvar imagens

def load_dataset():
    # Validate paths
    data_path = validate_path(config["data_path"])
    metadata_path = validate_path(config["metadata_path"])

    # Load metadata
    metadata = pd.read_csv(metadata_path)

    # Load and process audio files
    print("Loading and processing audio files...")
    X, y = load_audio_files(data_path, metadata)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config["test_size"], random_state=config["random_state"])

    # Prepare labels
    y_train = to_categorical(y_train, num_classes=config["num_classes"])
    y_test = to_categorical(y_test, num_classes=config["num_classes"])

    # Reshape data for model input
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    # Plot spectrograms
    print("Plotting spectrograms...")
    plot_spectrograms_per_class_from_files(data_path, metadata, [f"Class {i}" for i in range(config["num_classes"])])

    return X_train, y_train, X_test, y_test

def load_class_names(file_path="classes.json"):
    """
    Carrega as classes a partir de um arquivo JSON.
    """
    try:
        with open(file_path, 'r') as file:
            class_names = json.load(file)
        return {int(k): v for k, v in class_names.items()}  # Converte as chaves para inteiros
    except Exception as e:
        raise ValueError(f"Error loading class names: {e}")

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