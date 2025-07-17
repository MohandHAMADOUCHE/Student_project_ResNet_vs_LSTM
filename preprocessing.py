import json
import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical  # type: ignore
from analysis import save_figure_to_list
from utils import validate_path, config
from scipy.signal import hilbert
from scipy.signal import spectrogram
from concurrent.futures import ThreadPoolExecutor

def load_dataset():
    print("Loading and processing audio files...")
    # Validate paths
    data_path = validate_path(config["data_path"])
    metadata_path = validate_path(config["metadata_path"])

    # Load metadata
    metadata = pd.read_csv(metadata_path)

    # Load and process audio files
    X, y = load_audio_files(data_path, metadata)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config["test_size"], random_state=config["random_state"])

    # Prepare labels
    y_train = to_categorical(y_train, num_classes=config["num_classes"])
    y_test = to_categorical(y_test, num_classes=config["num_classes"])

    # Reshape data for model input
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    print("Audio files loaded!")

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

def compute_LOFAR(audio, sr, n_fft=2048, hop_length=512):
    """
    Calcula o espectrograma LOFAR para análise de frequências.
    """
    S = np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop_length))
    LOFAR = librosa.amplitude_to_db(S, ref=np.max)
    return LOFAR

def compute_DEMON(audio, sr, low_freq=20, high_freq=500):
    """
    DEMON: Técnica de demodulação para análise de cavitação e frequência de eixo.
    """
    envelope = np.abs(hilbert(audio))  # Calcula o envelope do sinal
    f, t, Sxx = spectrogram(envelope, sr, nperseg=2048)
    DEMON = Sxx[(f >= low_freq) & (f <= high_freq)]
    return DEMON

def process_single_audio(file_path, class_label):
    try:
        audio, sample_rate = librosa.load(file_path, sr=None)
        features = []

        # Parâmetros configuráveis diretamente no código
        n_mfcc = 20
        n_mels = 40  # Número de bandas Mel
        fmax = sample_rate // 2  # Frequência máxima

        # Pré-processamentos ativados no config
        if config["preprocessing_methods"]["MFCC"]:
            mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc, n_mels=n_mels, fmax=fmax)
            features.append(np.mean(mfcc.T, axis=0))
        
        if config["preprocessing_methods"]["GFCC"]:
            pass

        if config["preprocessing_methods"]["Delta"]:
            mfcc_delta = librosa.feature.delta(mfcc)
            features.append(np.mean(mfcc_delta.T, axis=0))

        if config["preprocessing_methods"]["STFT"]:
            stft = np.abs(librosa.stft(audio))
            features.append(np.mean(stft, axis=1))

        if config["preprocessing_methods"]["LOFAR"]:
            lofar = compute_LOFAR(audio, sample_rate)
            features.append(np.mean(lofar, axis=1))

        if config["preprocessing_methods"]["DEMON"]:
            demon = compute_DEMON(audio, sample_rate)
            features.append(np.mean(demon, axis=1))

        # Garantir que todas as características têm o mesmo comprimento
        max_length = max([len(f) for f in features])
        padded_features = [np.pad(f, (0, max_length - len(f)), mode='constant') for f in features]

        return np.concatenate(padded_features), class_label
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def load_audio_files(data_path, metadata):
    X, y = [], []
    with ThreadPoolExecutor() as executor:
        futures = []
        for i in range(len(metadata)):
            file_path = os.path.join(data_path, f"fold{metadata.fold[i]}", metadata.slice_file_name[i])
            class_label = metadata['classID'][i]
            futures.append(executor.submit(process_single_audio, file_path, class_label))
        
        for future in futures:
            result = future.result()
            if result:
                features, class_label = result
                X.append(features)
                y.append(class_label)
    
    # Garantir que X e y tenham o formato correto para o modelo
    X = np.array(X)
    y = np.array(y)

    return X, y

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


def load_dataset_from_folders(data_path):
    """
    Carrega os dados de áudio diretamente de pastas que representam classes.
    """
    print("Loading and processing audio files from folders...")
    # Validate the data path
    data_path = validate_path(data_path)

    # Load audio files and their corresponding labels
    X, y, class_mapping = load_audio_files_from_folders(data_path)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config["test_size"], random_state=config["random_state"])

    # Prepare labels
    y_train = to_categorical(y_train, num_classes=len(class_mapping))
    y_test = to_categorical(y_test, num_classes=len(class_mapping))

    # Reshape data for model input
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    print("Audio files loaded!")

    # Plot spectrograms
    print("Plotting spectrograms...")
    plot_spectrograms_per_class_from_folders(data_path, class_mapping)

    return X_train, y_train, X_test, y_test, class_mapping

def load_audio_files_from_folders(data_path):
    """
    Carrega os arquivos de áudio a partir de pastas organizadas por classes.
    """
    X, y = [], []
    class_mapping = {}  # Para mapear os índices das classes para seus nomes
    class_id = 0

    with ThreadPoolExecutor() as executor:
        futures = []
        for class_folder in os.listdir(data_path):
            class_path = os.path.join(data_path, class_folder)
            if os.path.isdir(class_path):
                # Atualiza o mapeamento de classes
                class_mapping[class_id] = class_folder
                for audio_file in os.listdir(class_path):
                    file_path = os.path.join(class_path, audio_file)
                    if os.path.isfile(file_path) and file_path.endswith(('.wav', '.mp3')):
                        futures.append(executor.submit(process_single_audio, file_path, class_id))
                class_id += 1
        
        for future in futures:
            result = future.result()
            if result:
                features, class_label = result
                X.append(features)
                y.append(class_label)

    # Garantir que X e y tenham o formato correto para o modelo
    X = np.array(X)
    y = np.array(y)

    return X, y, class_mapping

def plot_spectrograms_per_class_from_folders(data_path, class_mapping):
    """
    Plot spectrograms for each class using an example file and save to the global image list.
    """
    for class_id, class_name in class_mapping.items():
        try:
            class_path = os.path.join(data_path, class_name)
            # Obter um exemplo de arquivo da classe
            example_file = next(file for file in os.listdir(class_path) if file.endswith(('.wav', '.mp3')))
            file_path = os.path.join(class_path, example_file)

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
            plt.title(f"Spectrogram - Class {class_name}")
            plt.xlabel("Time (s)")
            plt.ylabel("Frequency (Hz)")

            # Salvar a figura na lista de imagens
            save_figure_to_list()

        except Exception as e:
            print(f"Error processing file for class {class_name}: {e}")
