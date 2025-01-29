import os
import random
import librosa
import numpy as np
from tqdm import tqdm
import librosa.display
import soundfile as sf
import matplotlib.pyplot as plt
from itertools import combinations
from gammatone.gtgram import gtgram
from concurrent.futures import ThreadPoolExecutor
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from tensorflow.keras.models import load_model  # type: ignore

# Salvar uma figura na lista de imagens
def save_figure_to_list(image_list):
    import io
    from PIL import Image
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image = Image.open(buf)
    image_list.append(np.array(image))
    buf.close()
    plt.close()

import numpy as np
def classify_audio(file_path, config, model_paths, class_mapping):
    # Extraindo as características do áudio
    features = extract_features(
        file_path=file_path,
        config=config,
        frame_length=config["general"]["frame_length"],
        hop_length=config["general"]["hop_length"],
        max_frames=config["general"]["max_frames"],
        n_features=config["general"]["n_features"],
        show_example=config["general"].get("visual_feedback", False),
    )

    print(f"Features shape: {features.shape}")  # Verificando a forma das características

    # Garantir que as características tenham o formato reduzido esperado
    # Aqui assumimos que o PCA reduziu para 2 componentes
    features_flattened = features.flatten()[:2]  # Pegando apenas as duas primeiras características
    features_resized = features_flattened.reshape(1, 2, 1)  # Redimensionando para o formato (1, 2, 1)

    print(f"Features resized shape (ajustado): {features_resized.shape}")  # Verifique a nova forma

    predictions = []  
    model_results = {}

    for model_name, model_path in model_paths.items():
        print(f"Modelo: {model_name}")

        model = load_model(model_path)

        # Predição com o modelo
        prediction = model.predict(features_resized)
        print(f"Prediction: {prediction}")  # Verificando a predição

        # Identificando a classe predita
        prediction_index = prediction.argmax(axis=-1)[0]
        predicted_class = [key for key, value in class_mapping.items() if value == prediction_index][0]
        confidence = prediction.max()  # Confiança na predição

        predictions.append(predicted_class)
        model_results[model_name] = {
            "predicted_class": predicted_class,
            "confidence": confidence,
        }

        print(f"Predição para {model_name}: {predicted_class} com confiança {confidence:.2f}")

    # Obtendo a classe final pela moda (mais comum)
    unique_classes, counts = np.unique(predictions, return_counts=True)
    most_common_class = unique_classes[np.argmax(counts)]

    print(f"Predição final (Moda): {most_common_class}")  # Mostrando a predição final

    return {
        "model_results": model_results,
        "final_prediction": most_common_class,
    }



def plot_spectrograms_per_class(data_path, class_mapping, image_list):
    """
    Plot spectrograms for each class using an example file and save to the global image list.
    """
    for class_name in class_mapping.keys():  # Use apenas as chaves do mapeamento
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
            save_figure_to_list(image_list)

        except Exception as e:
            print(f"Error processing file for class {class_name}: {e}")

# ==============================
# Audio Preprocessing Functions
# ==============================

def preprocess_audio(file_path, config, image_list, frame_length=2048, hop_length=512, show_example=False):
    """
    Applies pre-emphasis, framing, and windowing to the audio signal.

    Args:
        file_path (str): Path to the audio file.
        frame_length (int): Length of each frame.
        hop_length (int): Overlap between frames.
        show_example (bool): Whether to plot the audio and emphasized signal.

    Returns:
        tuple: Emphasized audio, framed and windowed signal, sample rate.
    """
    audio, sample_rate = librosa.load(file_path, sr=None, duration=config["general"]["audio_duration"])

    # Apply pre-emphasis
    pre_emphasis = 0.97
    emphasized_audio = np.append(audio[0], audio[1:] - pre_emphasis * audio[:-1])

    # Frame the signal
    frames = librosa.util.frame(emphasized_audio, frame_length=frame_length, hop_length=hop_length).T

    # Apply Hamming window
    windowed_frames = frames * np.hamming(frame_length)

    # Optional visualization
    if show_example:
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.title("Original Audio")
        librosa.display.waveshow(audio, sr=sample_rate)
        plt.subplot(1, 2, 2)
        plt.title("Emphasized Audio")
        librosa.display.waveshow(emphasized_audio, sr=sample_rate)
        plt.tight_layout()
        save_figure_to_list(image_list)

        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.title("Emphasized Audio")
        librosa.display.waveshow(emphasized_audio, sr=sample_rate)
        plt.subplot(1, 2, 2)
        plt.title("Emphasized Audio + Framing")
        librosa.display.waveshow(frames, sr=sample_rate)
        plt.tight_layout()
        save_figure_to_list(image_list)

        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.title("Emphasized Audio + Framing")
        librosa.display.waveshow(frames, sr=sample_rate)
        plt.subplot(1, 2, 2)
        plt.title("Emphasized Audio + Framing + Windowed")
        librosa.display.waveshow(windowed_frames, sr=sample_rate)
        plt.tight_layout()
        save_figure_to_list(image_list)

    return emphasized_audio, windowed_frames, sample_rate


def plot_feature(feature, sample_rate, title, image_list):
    """Helper function to visualize features."""
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(feature, sr=sample_rate, x_axis='time', cmap='viridis')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    save_figure_to_list(image_list)

# ============================
# Feature Extraction Functions
# ============================

def extract_features(file_path, config, image_list=[], frame_length=2048, hop_length=512, max_frames=100, n_features=13, save_path=None, show_example=False):
    """
    Extracts audio features, combines basic and delta features, and visualizes the process based on configuration.

    Args:
        file_path (str): Path to the audio file.
        config (dict): Configuration dictionary specifying which methods to apply.
        image_list (list): List to store generated images for visualization.
        frame_length (int): Length of each frame.
        hop_length (int): Overlap between frames.
        max_frames (int): Maximum number of frames per feature.
        n_features (int): Number of features to extract.
        save_path (str): Path to save the extracted features.
        show_example (bool): Whether to visualize the features.

    Returns:
        np.ndarray: Final fused feature matrix.
    """

    def compute_gfcc(audio, sr, n_filters=64, frame_length=0.025, hop_length=0.01, f_min=50):
        # Aplicar filtro Gammatone
        gammatone_features = gtgram(audio, sr, frame_length, hop_length, n_filters, f_min)
        # Aplicar log para simular compressão cepstral
        gfcc = np.log(np.abs(gammatone_features) + 1e-8)
        return gfcc

    if save_path and os.path.exists(save_path):
        return np.load(save_path)

    emphasized_audio, _, sample_rate = preprocess_audio(file_path, config, image_list, frame_length, hop_length)

    # Pad or truncate features
    def pad_or_truncate(feature, max_frames):
        if feature.shape[1] > max_frames:
            return feature[:, :max_frames]
        elif feature.shape[1] < max_frames:
            return np.pad(feature, ((0, 0), (0, max_frames - feature.shape[1])), mode='constant')
        return feature

    # Extract basic features
    basic_features = []
    feature_names = []  # Para rastrear o nome das features básicas
    if config["preprocessing_methods"].get("MFCC", False):
        mfcc = pad_or_truncate(librosa.feature.mfcc(y=emphasized_audio, sr=sample_rate, n_mfcc=n_features), max_frames)
        basic_features.append(mfcc)
        feature_names.append("MFCC")
        if show_example:
            plot_feature(mfcc, sample_rate, "MFCC", image_list)

    if config["preprocessing_methods"].get("GFCC", False):
        gfcc = pad_or_truncate(compute_gfcc(emphasized_audio, sample_rate), max_frames)
        basic_features.append(gfcc)
        feature_names.append("GFCC")
        if show_example:
            plot_feature(gfcc, sample_rate, "GFCC", image_list)

    if config["preprocessing_methods"].get("CQT", False):
        cqt = pad_or_truncate(
            np.abs(librosa.cqt(
                y=emphasized_audio,
                sr=sample_rate,
                n_bins=60,  # Reduzido para evitar exceder Nyquist
                bins_per_octave=12,  # Número de bins por oitava
                fmin=librosa.note_to_hz('C2'),  # Frequência mínima ajustada
            )),
            max_frames
        )
        basic_features.append(cqt)
        feature_names.append("CQT")
        if show_example:
            plot_feature(cqt, sample_rate, "CQT", image_list)

    if config["preprocessing_methods"].get("LOFAR", False):
        lofar = pad_or_truncate(
            librosa.amplitude_to_db(np.abs(librosa.stft(emphasized_audio, n_fft=frame_length))[:n_features, :], ref=np.max),
            max_frames,
        )
        basic_features.append(lofar)
        feature_names.append("LOFAR")
        if show_example:
            plot_feature(lofar, sample_rate, "LOFAR", image_list)

    # Combine basic features
    combined_basic_features = np.concatenate(basic_features, axis=0) if basic_features else None
    if show_example:
        plot_feature(combined_basic_features, sample_rate, "Combined Basic Features", image_list)

    # Extract delta features
    delta_features = []
    if config["preprocessing_methods"].get("DELTAS", False):
        for feature, name in zip(basic_features, feature_names):
            delta = pad_or_truncate(librosa.feature.delta(feature), max_frames)
            delta_features.append(delta)
            if show_example:
                plot_feature(delta, sample_rate, f"Delta Features ({name})", image_list)

    # Combine delta features
    combined_delta_features = np.concatenate(delta_features, axis=0) if delta_features else None
    if show_example and combined_delta_features is not None:
        plot_feature(combined_delta_features, sample_rate, "Combined Delta Features", image_list)

    # Combine all features (basic + delta)
    all_features = np.concatenate([combined_basic_features, combined_delta_features], axis=0) if combined_delta_features is not None else combined_basic_features
    if show_example and combined_delta_features is not None:
        plot_feature(all_features, sample_rate, "All Features (Basic + Delta)", image_list)

    # Save features
    if save_path:
        np.save(save_path, all_features)

    return all_features


def reduce_and_visualize_with_nca(features, labels, config, image_list, n_components=2):
    """
    Reduces dimensionality using NCA and visualizes the result.

    Args:
        features (np.ndarray): Input features.
        labels (np.ndarray): Corresponding labels.
        n_components (int): Number of components for NCA.

    Returns:
        np.ndarray: Reduced features.
    """
    print("Reducing dimensionality with NCA...")

    # Flatten features for NCA
    num_samples = features.shape[0]
    num_features = np.prod(features.shape[1:])
    flattened_features = features.reshape(num_samples, num_features)

    # Apply NCA
    nca = NeighborhoodComponentsAnalysis(n_components=n_components, random_state=config["general"]["random_state"])
    reduced_features = nca.fit_transform(flattened_features, labels)

    # Visualize reduced features
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=labels, cmap='viridis', alpha=0.8)
    plt.colorbar(scatter, label='Classes')
    plt.title("Feature Screening Diagram (MGCL-Delta)")
    plt.xlabel("NCA Dimension 1")
    plt.ylabel("NCA Dimension 2")
    plt.tight_layout()
    save_figure_to_list(image_list)

    return reduced_features


# Function to process a single audio file
def process_audio_file(args):
    """
    Processes a single audio file to extract features.

    Args:
        args (tuple): Contains the file path, class index, frame length, hop length, 
                      maximum frames, number of features, and save directory.

    Returns:
        tuple: A tuple containing the extracted features and class index.
    """
    file_path, class_index, frame_length, hop_length, max_frames, n_features, save_dir, config = args
    save_path = None
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, os.path.basename(file_path) + '.npy')
    
    # Extract features from the audio file
    features = extract_features(file_path, config, frame_length, hop_length, max_frames, n_features, save_path=save_path)
    return features, class_index

# Function to load a dataset organized in subdirectories using ThreadPoolExecutor
def load_dataset(image_list, config, frame_length=2048, hop_length=512, max_frames=100, n_features=13, save_dir=None):
    """
    Loads a dataset organized in subdirectories, processes the audio files in parallel, 
    and extracts their features.

    Args:
        dataset_path (str): Path to the root directory containing the dataset.
        frame_length (int): Length of each frame for feature extraction.
        hop_length (int): Hop length for feature extraction.
        max_frames (int): Maximum number of frames per feature vector.
        n_features (int): Number of features to extract.
        save_dir (str, optional): Directory to save the extracted features.

    Returns:
        tuple: A tuple containing the features matrix (X), the labels (y), and 
               a dictionary mapping class names to indices.
    """

    X, y = [], []
    classes = sorted(os.listdir(config["general"]["data_path"]))  # List all class folders
    class_to_index = {cls_name: idx for idx, cls_name in enumerate(classes)}  # Map class names to indices
    
    if config["general"]["visual_feedback"]:
        plot_spectrograms_per_class(config["general"]["data_path"], class_to_index, image_list)
        example_audio_path = os.path.join(config["general"]["data_path"], os.listdir(config["general"]["data_path"])[0], os.listdir(os.path.join(config["general"]["data_path"], os.listdir(config["general"]["data_path"])[0]))[0])
        preprocess_audio(example_audio_path, config, image_list, show_example=True)
        extract_features(example_audio_path, config, image_list, show_example=True)

    tasks = []
    for cls_name in classes:
        cls_path = os.path.join(config["general"]["data_path"], cls_name)
        if os.path.isdir(cls_path):  # Check if it is a directory
            for file_name in os.listdir(cls_path):  # List all files in the class folder
                file_path = os.path.join(cls_path, file_name)
                if file_name.endswith(('.wav', '.mp3')):  # Check if the file is an audio file
                    tasks.append((file_path, class_to_index[cls_name], frame_length, hop_length, max_frames, n_features, save_dir, config))

    # Process the files in parallel using ThreadPoolExecutor
    with ThreadPoolExecutor() as executor:
        results = list(tqdm(executor.map(process_audio_file, tasks), total=len(tasks), desc="Processing audio files"))
        for features, label in results:
            X.append(features)
            y.append(label)

    return np.array(X), np.array(y), class_to_index

from itertools import combinations, product

def mix_audio_files(file_paths, output_path, sample_rate=22050):
    """
    Função para misturar múltiplos arquivos de áudio e salvar o resultado em um novo arquivo.
    """
    # Carregar os áudios e padronizar a taxa de amostragem
    audios = [librosa.load(file_path, sr=sample_rate)[0] for file_path in file_paths]
    
    # Ajustar todos os áudios para o mesmo comprimento (mínimo comprimento encontrado)
    min_length = min(len(audio) for audio in audios)
    audios = [audio[:min_length] for audio in audios]
    
    # Misturar os áudios (média dos sinais)
    mixed_audio = np.mean(audios, axis=0)
    
    # Garantir que os valores estão no intervalo [-1, 1]
    mixed_audio = np.clip(mixed_audio, -1.0, 1.0)
    
    # Salvar o áudio misturado no arquivo de saída
    sf.write(output_path, mixed_audio, sample_rate)


def create_mixed_classes(dataset_path, sample_rate=22050):
    """
    Função para criar novas classes misturadas a partir de combinações de classes existentes.
    """
    # Listar as subpastas (classes) dentro do dataset
    class_dirs = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    
    # Contar a quantidade de arquivos em cada classe
    class_file_counts = {cls: len(os.listdir(os.path.join(dataset_path, cls))) for cls in class_dirs}
    max_count = max(class_file_counts.values())
    
    # Criar combinações de classes
    for num_classes in range(2, len(class_dirs) + 1):
        for class_combination in combinations(class_dirs, num_classes):
            combined_class_name = "_".join(class_combination)
            combined_class_dir = os.path.join(dataset_path, combined_class_name)
            os.makedirs(combined_class_dir, exist_ok=True)
            
            # Obter os arquivos de cada classe na combinação
            class_files = [
                [os.path.join(dataset_path, class_dir, f) for f in os.listdir(os.path.join(dataset_path, class_dir)) if f.endswith(('.wav', '.mp3'))]
                for class_dir in class_combination
            ]
            
            # Gerar combinações de arquivos (um de cada classe)
            combined_files = list(product(*class_files))
            random.shuffle(combined_files)  # Aleatorizar as combinações
            
            # Limitar o número de combinações para o máximo de arquivos em qualquer classe
            limited_combinations = combined_files[:max_count]
            
            # Misturar os arquivos e salvar o resultado
            for file_paths in limited_combinations:
                output_file_name = f"mix_{'_'.join([os.path.splitext(os.path.basename(f))[0] for f in file_paths])}.wav"
                output_file_path = os.path.join(combined_class_dir, output_file_name)
                mix_audio_files(file_paths, output_file_path, sample_rate)
            
            print(f"Combinações salvas em: {combined_class_dir}")

def preprocessing(config, image_list):
    print("Começando preprocessamento...")

    if config["general"]["mix_classes"]:
        print("Começando mixagem de classes...")
        create_mixed_classes(config["general"]["data_path"])
        print("Mixagem de classes finalizada...")

    # Load dataset
    X, y, class_to_index = load_dataset(image_list, config, save_dir=config["general"]["processed_data_path"])

    # Reduce dimensionality and visualize
    X_reduced = reduce_and_visualize_with_nca(X, y, config, image_list)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=config["general"]["test_size"], random_state=config["general"]["random_state"])

    X_train = X_train.reshape(X_train.shape[0], 2, 1, 1)
    X_test = X_test.reshape(X_test.shape[0], 2, 1, 1)

    print("Preprocessamento finalizado...")
    return X_train, y_train, X_test, y_test, class_to_index

def load_preprocessed(config, image_list):
    # Load dataset
    X, y, class_to_index = load_dataset(image_list, config, save_dir=config["general"]["processed_data_path"])

    # Reduce dimensionality and visualize
    X_reduced = reduce_and_visualize_with_nca(X, y, config, image_list)

    # Train-test split
    _, X_test, _, y_test = train_test_split(X_reduced, y, test_size=config["general"]["test_size"], random_state=config["general"]["random_state"])

    X_test = X_test.reshape(X_test.shape[0], 2, 1, 1)

    return X_test, y_test, class_to_index