import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress warnings and informational logs
import numpy as np
import librosa
import matplotlib.pyplot as plt
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models, optimizers
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# Função de pré-processamento com pré-ênfase, framing e windowing
def preprocess_audio(file_path, frame_length=2048, hop_length=512, show_example=False):
    audio, sample_rate = librosa.load(file_path, sr=None)

    # Pré-ênfase
    pre_emphasis = 0.97
    emphasized_audio = np.append(audio[0], audio[1:] - pre_emphasis * audio[:-1])

    # Divisão em frames e aplicação de janela Hamming
    frames = librosa.util.frame(emphasized_audio, frame_length=frame_length, hop_length=hop_length).T
    windowed_frames = frames * np.hamming(frame_length)

    # Visualização de exemplo
    if show_example:
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.title("Áudio Original")
        librosa.display.waveshow(audio, sr=sample_rate)
        plt.subplot(1, 2, 2)
        plt.title("Áudio com Pré-Ênfase")
        librosa.display.waveshow(emphasized_audio, sr=sample_rate)
        plt.tight_layout()
        plt.show()

    return emphasized_audio, windowed_frames, sample_rate

def extract_features(file_path, frame_length=2048, hop_length=512, max_frames=100, n_features=13, save_path=None, show_example=False):
    if save_path and os.path.exists(save_path):
        return np.load(save_path)

    emphasized_audio, frames, sample_rate = preprocess_audio(file_path, frame_length, hop_length)

    # Extração das características
    mfcc = librosa.feature.mfcc(y=emphasized_audio, sr=sample_rate, n_mfcc=n_features)
    gfcc = librosa.feature.mfcc(y=emphasized_audio, sr=sample_rate, n_mfcc=n_features)  # Substituir por GFCC real
    cqt = np.abs(librosa.cqt(y=emphasized_audio, sr=sample_rate, n_bins=n_features))  # Ajustar n_bins para n_features
    lofar = librosa.amplitude_to_db(np.abs(librosa.stft(emphasized_audio, n_fft=frame_length))[:n_features, :], ref=np.max)
    delta_mfcc = librosa.feature.delta(mfcc)

    # Ajustar o número de frames para o mesmo tamanho
    def pad_or_truncate(feature, max_frames):
        if feature.shape[1] > max_frames:  # Truncar
            return feature[:, :max_frames]
        elif feature.shape[1] < max_frames:  # Preencher com zeros
            return np.pad(feature, ((0, 0), (0, max_frames - feature.shape[1])), mode='constant')
        return feature

    mfcc = pad_or_truncate(mfcc, max_frames)
    gfcc = pad_or_truncate(gfcc, max_frames)
    cqt = pad_or_truncate(cqt, max_frames)
    lofar = pad_or_truncate(lofar, max_frames)
    delta_mfcc = pad_or_truncate(delta_mfcc, max_frames)

    # Fusão de características
    fused_features = np.concatenate([mfcc, gfcc, cqt, lofar, delta_mfcc], axis=0)

    # Salvar features
    if save_path:
        np.save(save_path, fused_features)

    # Visualizar exemplo de cada feature e das features fusionadas
    if show_example:
        features = {"MFCC": mfcc, "GFCC": gfcc, "CQT": cqt, "LOFAR": lofar, "Delta-MFCC": delta_mfcc, "Fused Features": fused_features}
        for name, feature in features.items():
            plt.figure(figsize=(10, 4))
            librosa.display.specshow(feature, sr=sample_rate, x_axis='time', cmap='viridis')
            plt.colorbar(format='%+2.0f dB')
            plt.title(name)
            plt.tight_layout()
            plt.show()

    return fused_features

# Função para processar um único arquivo de áudio
def process_audio_file(args):
    file_path, class_index, frame_length, hop_length, max_frames, n_features, save_dir = args
    save_path = None
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, os.path.basename(file_path) + '.npy')
    features = extract_features(file_path, frame_length, hop_length, max_frames, n_features, save_path=save_path)
    return features, class_index

# Função para carregar dataset organizado em subpastas com ThreadPoolExecutor
def load_dataset(dataset_path, frame_length=2048, hop_length=512, max_frames=100, n_features=13, save_dir=None):
    X, y = [], []
    classes = sorted(os.listdir(dataset_path))
    class_to_index = {cls_name: idx for idx, cls_name in enumerate(classes)}

    tasks = []
    for cls_name in classes:
        cls_path = os.path.join(dataset_path, cls_name)
        if os.path.isdir(cls_path):
            for file_name in os.listdir(cls_path):
                file_path = os.path.join(cls_path, file_name)
                if file_name.endswith(('.wav', '.mp3')):
                    tasks.append((file_path, class_to_index[cls_name], frame_length, hop_length, max_frames, n_features, save_dir))

    with ThreadPoolExecutor() as executor:
        results = list(tqdm(executor.map(process_audio_file, tasks), total=len(tasks), desc="Processando arquivos de áudio"))
        for features, label in results:
            X.append(features)
            y.append(label)

    return np.array(X), np.array(y), class_to_index

# Função para construir modelo ResNet18
def build_resnet18(input_shape, num_classes):
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
    model.compile(optimizer=optimizers.Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Caminho para o dataset
dataset_path = r'C:\Users\gumar\OneDrive\Área de Trabalho\Pesquisa UBO\DeepShip-main'
save_dir = r'C:\Users\gumar\OneDrive\Área de Trabalho\Pesquisa UBO\DeepShip-main\processed_features'

# Mostrar um exemplo de áudio e suas features antes de carregar todo o dataset
example_audio_path = os.path.join(dataset_path, os.listdir(dataset_path)[0], os.listdir(os.path.join(dataset_path, os.listdir(dataset_path)[0]))[0])
preprocess_audio(example_audio_path, show_example=True)
extract_features(example_audio_path, show_example=True)

# Carregar o dataset
X, y, class_to_index = load_dataset(dataset_path, save_dir=save_dir)

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Ajustar formato para entrada no modelo ResNet
X_train = X_train[..., np.newaxis]  # Adicionar canal para 4D
X_test = X_test[..., np.newaxis]

# Construir e treinar o modelo ResNet18
model = build_resnet18(input_shape=X_train.shape[1:], num_classes=len(class_to_index))
history = model.fit(X_train, y_train, epochs=100, batch_size=4, validation_data=(X_test, y_test))

# Avaliar o modelo
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Acurácia: {accuracy * 100:.2f}%")
