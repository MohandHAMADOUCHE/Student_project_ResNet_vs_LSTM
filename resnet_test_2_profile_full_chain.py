import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress warnings and informational logs
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from gammatone.gtgram import gtgram
from sklearn.metrics import confusion_matrix

# ==============================
# Audio Preprocessing Functions
# ==============================

def preprocess_audio(file_path, frame_length=2048, hop_length=512, show_example=False):
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
    audio, sample_rate = librosa.load(file_path, sr=None, duration=30.0)

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
        plt.show()

        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.title("Emphasized Audio")
        librosa.display.waveshow(emphasized_audio, sr=sample_rate)
        plt.subplot(1, 2, 2)
        plt.title("Emphasized Audio + Framing")
        librosa.display.waveshow(frames, sr=sample_rate)
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.title("Emphasized Audio + Framing")
        librosa.display.waveshow(frames, sr=sample_rate)
        plt.subplot(1, 2, 2)
        plt.title("Emphasized Audio + Framing + Windowed")
        librosa.display.waveshow(windowed_frames, sr=sample_rate)
        plt.tight_layout()
        plt.show()

    return emphasized_audio, windowed_frames, sample_rate

# ============================
# Feature Extraction Functions
# ============================

def extract_features(file_path, frame_length=2048, hop_length=512, max_frames=100, n_features=13, save_path=None, show_example=False):
    """
    Extracts audio features including MFCC, GFCC, CQT, LOFAR, and their delta features.

    Args:
        file_path (str): Path to the audio file.
        frame_length (int): Length of each frame.
        hop_length (int): Overlap between frames.
        max_frames (int): Maximum number of frames per feature.
        n_features (int): Number of features to extract.
        save_path (str): Path to save the extracted features.
        show_example (bool): Whether to visualize the features.

    Returns:
        np.ndarray: Concatenated feature matrix.
    """

    def compute_gfcc(audio, sr, n_filters=64, frame_length=0.025, hop_length=0.01, f_min=50):
        # Aplicar filtro Gammatone
        gammatone_features = gtgram(audio, sr, frame_length, hop_length, n_filters, f_min)

        # Aplicar log para simular compressão cepstral
        gfcc = np.log(np.abs(gammatone_features) + 1e-8)

        return gfcc

    if save_path and os.path.exists(save_path):
        return np.load(save_path)

    emphasized_audio, _, sample_rate = preprocess_audio(file_path, frame_length, hop_length)

    # Extract features
    mfcc = librosa.feature.mfcc(y=emphasized_audio, sr=sample_rate, n_mfcc=n_features)
    gfcc = compute_gfcc(emphasized_audio, sample_rate)
    cqt = np.abs(librosa.cqt(y=emphasized_audio, sr=sample_rate, n_bins=84))
    lofar = librosa.amplitude_to_db(np.abs(librosa.stft(emphasized_audio, n_fft=frame_length))[:n_features, :], ref=np.max)
    delta_mfcc = librosa.feature.delta(mfcc)
    delta_gfcc = librosa.feature.delta(gfcc)
    delta_cqt = librosa.feature.delta(cqt)
    delta_lofar = librosa.feature.delta(lofar)

    # Pad or truncate features
    def pad_or_truncate(feature, max_frames):
        if feature.shape[1] > max_frames:
            return feature[:, :max_frames]
        elif feature.shape[1] < max_frames:
            return np.pad(feature, ((0, 0), (0, max_frames - feature.shape[1])), mode='constant')
        return feature

    mfcc = pad_or_truncate(mfcc, max_frames)
    gfcc = pad_or_truncate(gfcc, max_frames)
    cqt = pad_or_truncate(cqt, max_frames)
    lofar = pad_or_truncate(lofar, max_frames)
    delta_mfcc = pad_or_truncate(delta_mfcc, max_frames)
    delta_gfcc = pad_or_truncate(delta_gfcc, max_frames)
    delta_cqt = pad_or_truncate(delta_cqt, max_frames)
    delta_lofar = pad_or_truncate(delta_lofar, max_frames)

    # Fuse features
    mgcl = np.concatenate([mfcc, gfcc, cqt, lofar], axis=0)
    delta_mgcl = np.concatenate([delta_mfcc, delta_gfcc, delta_cqt, delta_lofar], axis=0)
    fused_features = np.concatenate([mgcl, delta_mgcl], axis=0)

    # Save features
    if save_path:
        np.save(save_path, fused_features)

    # Optional visualization
    if show_example:
        features = {
            "MFCC": mfcc, "GFCC": gfcc, "CQT": cqt, "LOFAR": lofar,
            "Delta-MFCC": delta_mfcc, "Delta-GFCC": delta_gfcc,
            "Delta-CQT": delta_cqt, "Delta-LOFAR": delta_lofar,
            "MGCL": mgcl, "Delta-MGCL": delta_mgcl
        }
        for name, feature in features.items():
            plt.figure(figsize=(10, 4))
            librosa.display.specshow(feature, sr=sample_rate, x_axis='time', cmap='viridis')
            plt.colorbar(format='%+2.0f dB')
            plt.title(name)
            plt.tight_layout()
            plt.show()

    return fused_features

def reduce_and_visualize_with_nca(features, labels, n_components=2):
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
    nca = NeighborhoodComponentsAnalysis(n_components=n_components, random_state=42)
    reduced_features = nca.fit_transform(flattened_features, labels)

    # Visualize reduced features
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=labels, cmap='viridis', alpha=0.8)
    plt.colorbar(scatter, label='Classes')
    plt.title("Feature Screening Diagram (MGCL-Delta)")
    plt.xlabel("NCA Dimension 1")
    plt.ylabel("NCA Dimension 2")
    plt.tight_layout()
    plt.show()

    return reduced_features

# ======================
# Model Construction
# ======================

def build_resnet18(input_shape, num_classes):
    """
    Builds a ResNet18 model for classification.

    Args:
        input_shape (tuple): Shape of the input features.
        num_classes (int): Number of output classes.

    Returns:
        keras.Model: Compiled ResNet18 model.
    """
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(64, kernel_size=7, strides=2, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    def resnet_block(x, filters, downsample=False):
        shortcut = x
        if downsample:
            shortcut = layers.Conv2D(filters, kernel_size=1, strides=2, padding='same')(shortcut)
        x = layers.Conv2D(filters, kernel_size=3, strides=(2 if downsample else 1), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Conv2D(filters, kernel_size=3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.add([x, shortcut])
        x = layers.ReLU()(x)
        return x

    for filters, num_blocks in zip([64, 128, 256, 512], [2, 2, 2, 2]):
        for i in range(num_blocks):
            x = resnet_block(x, filters, downsample=(i == 0 and filters != 64))

    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs)
    model.compile(optimizer=optimizers.SGD(learning_rate=0.01, momentum=0.9),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

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
    file_path, class_index, frame_length, hop_length, max_frames, n_features, save_dir = args
    save_path = None
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, os.path.basename(file_path) + '.npy')
    
    # Extract features from the audio file
    features = extract_features(file_path, frame_length, hop_length, max_frames, n_features, save_path=save_path)
    return features, class_index

# Function to load a dataset organized in subdirectories using ThreadPoolExecutor
def load_dataset(dataset_path, frame_length=2048, hop_length=512, max_frames=100, n_features=13, save_dir=None):
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
    classes = sorted(os.listdir(dataset_path))  # List all class folders
    class_to_index = {cls_name: idx for idx, cls_name in enumerate(classes)}  # Map class names to indices

    tasks = []
    for cls_name in classes:
        cls_path = os.path.join(dataset_path, cls_name)
        if os.path.isdir(cls_path):  # Check if it is a directory
            for file_name in os.listdir(cls_path):  # List all files in the class folder
                file_path = os.path.join(cls_path, file_name)
                if file_name.endswith(('.wav', '.mp3')):  # Check if the file is an audio file
                    tasks.append((file_path, class_to_index[cls_name], frame_length, hop_length, max_frames, n_features, save_dir))

    # Process the files in parallel using ThreadPoolExecutor
    with ThreadPoolExecutor() as executor:
        results = list(tqdm(executor.map(process_audio_file, tasks), total=len(tasks), desc="Processing audio files"))
        for features, label in results:
            X.append(features)
            y.append(label)

    return np.array(X), np.array(y), class_to_index


import soundfile as sf
from itertools import combinations
import random


import os
import numpy as np
import librosa
import soundfile as sf
from itertools import chain, combinations
import random


def mix_audio_files(file_paths, output_path, sample_rate=22050):
    """
    Combina múltiplos arquivos de áudio em um único arquivo mixado.

    Args:
        file_paths (list): Lista de caminhos dos arquivos de áudio.
        output_path (str): Caminho para salvar o arquivo de áudio mixado.
        sample_rate (int): Taxa de amostragem dos arquivos de áudio.

    """
    audios = [librosa.load(file_path, sr=sample_rate)[0] for file_path in file_paths]

    # Garantir que os áudios tenham o mesmo comprimento
    min_length = min(len(audio) for audio in audios)
    audios = [audio[:min_length] for audio in audios]

    # Mixar os áudios
    mixed_audio = np.mean(audios, axis=0)

    # Normalizar o áudio mixado
    mixed_audio = np.clip(mixed_audio, -1.0, 1.0)

    # Salvar o áudio mixado
    sf.write(output_path, mixed_audio, sample_rate)


def create_mixed_classes(dataset_path, sample_rate=22050):
    """
    Gera todas as combinações possíveis de áudios das classes existentes dentro das subpastas do dataset.

    Args:
        dataset_path (str): Caminho do diretório do dataset.
        sample_rate (int): Taxa de amostragem dos arquivos de áudio.
    """
    class_dirs = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    class_file_counts = {cls: len(os.listdir(os.path.join(dataset_path, cls))) for cls in class_dirs}
    max_count = max(class_file_counts.values())

    # Gerar todas as combinações possíveis de 2 a n classes
    for num_classes in range(2, len(class_dirs) + 1):
        for class_combination in combinations(class_dirs, num_classes):
            combined_class_name = "_".join(class_combination)
            combined_class_dir = os.path.join(dataset_path, combined_class_name)
            os.makedirs(combined_class_dir, exist_ok=True)

            # Carregar arquivos de áudio de cada classe
            class_files = [
                [os.path.join(dataset_path, class_dir, f) for f in os.listdir(os.path.join(dataset_path, class_dir)) if f.endswith(('.wav', '.mp3'))]
                for class_dir in class_combination
            ]

            # Limitar o número de combinações ao tamanho da maior classe
            combined_files = list(combinations(sum(class_files, []), num_classes))
            random.shuffle(combined_files)
            limited_combinations = combined_files[:max_count]

            for file_paths in limited_combinations:
                output_file_name = f"mix_{'_'.join([os.path.splitext(os.path.basename(f))[0] for f in file_paths])}.wav"
                output_file_path = os.path.join(combined_class_dir, output_file_name)

                mix_audio_files(file_paths, output_file_path, sample_rate)

            print(f"Combinações salvas em: {combined_class_dir}")


# ======================
# Main Workflow
# ======================

def main():
    version = "V4"
    dataset_path = fr'/tools/mohand_postdoc/datasets/DeepShip/DeepShip_organized_V2'
    save_dir = fr'/tools/mohand_postdoc/datasets/DeepShip/DeepShip_{version}_processed_features'
    model_save_path = fr"/tools/mohand_postdoc/datasets/models/resnet18_model_{version}_.h5"

    #create_mixed_classes(dataset_path)


    # Example audio and feature extraction
    #example_audio_path = os.path.join(dataset_path, os.listdir(dataset_path)[0], os.listdir(os.path.join(dataset_path, os.listdir(dataset_path)[0]))[0])
    #preprocess_audio(example_audio_path, show_example=True)
    #extract_features(example_audio_path, show_example=True)

    # Load dataset
    X, y, class_to_index = load_dataset(dataset_path, save_dir=save_dir)

    # Reduce dimensionality and visualize
    X_reduced = reduce_and_visualize_with_nca(X, y)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.2, random_state=42)

    X_train = X_train.reshape(X_train.shape[0], 2, 1, 1)
    X_test = X_test.reshape(X_test.shape[0], 2, 1, 1)

    # Build or load model
    if os.path.exists(model_save_path):
        print("Loading saved model...")
        model = models.load_model(model_save_path)
    else:
        print("Training the model...")
        model = build_resnet18(input_shape=X_train.shape[1:], num_classes=len(class_to_index))
        early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
        lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
        history = model.fit(X_train, y_train, epochs=100, batch_size=5, validation_data=(X_test, y_test), callbacks=[early_stopping, lr_scheduler])
        model.save(model_save_path)
        print(f"Model saved to {model_save_path}")

# Évaluation globale du modèle
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Accuracy globale: {accuracy * 100:.2f}%")

    # 🔥 Ajout du calcul de l'accuracy par classe 🔥
    print("\n🔍 Calcul de l'accuracy par classe...")
    
    # Prédictions sur le jeu de test
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)  # Convertir les probabilités en indices de classes
    y_true_classes = y_test  # y_test contient déjà les indices des classes


    # Matrice de confusion
    conf_matrix = confusion_matrix(y_true_classes, y_pred_classes)

    # Compter le nombre d'échantillons par classe dans y_test
    import collections
    counts = collections.Counter(y_true_classes)
    print("\n📊 Nombre d'échantillons par classe dans y_test:")
    for class_id, count in sorted(counts.items()):
        print(f"Classe {class_id}: {count} échantillons")

    # Accuracy par classe
    accuracy_per_class = conf_matrix.diagonal() / conf_matrix.sum(axis=1)

    # ✅ Vérifier les classes présentes dans y_test
    classes_presentes = np.unique(y_true_classes)  # Liste des classes présentes dans y_test
    print(f"\n🔢 Classes présentes dans y_test: {classes_presentes.tolist()} ({len(classes_presentes)} classes)\n")

    # Affichage des résultats par classe
    print("\n📊 Résultats par classe:")
    for class_name, idx in class_to_index.items():
        if idx in classes_presentes:  # Vérifie si la classe est bien présente
            print(f"Classe {class_name} (index {idx}) - Accuracy: {accuracy_per_class[idx] * 100:.2f}%")
        else:
            print(f"⚠️ Classe {class_name} (index {idx}) n'a pas d'échantillons dans y_test.")

if __name__ == "__main__":
    main()

