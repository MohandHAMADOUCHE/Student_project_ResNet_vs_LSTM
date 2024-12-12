import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers, models, optimizers
import pickle

# Configurações
DATA_PATH = r"C:\Users\gumar\OneDrive\Área de Trabalho\Pesquisa UBO\DeepShip-main"
MAX_LENGTH = 100  # Comprimento máximo das características
SAVE_FILE = "preprocessed_data.pkl"

# Função para plotar características
def plot_features(features, title="Características", cmap="viridis"):
    plt.figure(figsize=(10, 4))
    plt.imshow(features, aspect="auto", origin="lower", cmap=cmap)
    plt.colorbar()
    plt.title(title)
    plt.xlabel("Tempo")
    plt.ylabel("Coeficientes")
    plt.tight_layout()
    plt.show()

# Função para processar um único arquivo de áudio
def process_single_audio(file_path, class_label, features_type="MFCC", max_length=MAX_LENGTH, plot=False):
    try:
        audio, sample_rate = librosa.load(file_path, sr=None)

        # Extração de características
        if features_type == "MFCC":
            features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
        elif features_type == "GFCC":
            features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)  # Substituir por GFCC adequado
        elif features_type == "CQT":
            features = np.abs(librosa.cqt(y=audio, sr=sample_rate, n_bins=84))
        elif features_type == "LOFAR":
            features = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
        elif features_type == "MGCL":
            mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
            gfcc = mfcc  # Substituir por GFCC adequado
            cqt = np.abs(librosa.cqt(y=audio, sr=sample_rate, n_bins=84))
            lofar = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
            features = np.concatenate([mfcc, gfcc, cqt, lofar], axis=0)
        elif features_type == "MGCL-Delta":
            mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
            delta_mfcc = librosa.feature.delta(mfcc)
            delta2_mfcc = librosa.feature.delta(mfcc, order=2)
            features = np.concatenate([mfcc, delta_mfcc, delta2_mfcc], axis=0)

        # Normalizar
        std = np.std(features, axis=1, keepdims=True)
        std[std == 0] = 1e-8  # Evita divisão por zero
        features = (features - np.mean(features, axis=1, keepdims=True)) / std

        # Padding ou truncamento
        if features.shape[1] > max_length:
            features = features[:, :max_length]
        else:
            features = np.pad(features, ((0, 0), (0, max_length - features.shape[1])), mode='constant')

        # Plotar as características, se necessário
        if plot:
            plot_features(features, title=f"{features_type} - {os.path.basename(file_path)}")

        return features, class_label
    except Exception as e:
        print(f"Erro ao processar {file_path}: {e}")
        return None

# Função para carregar o dataset
def load_dataset(data_path, features_type="MFCC", max_length=MAX_LENGTH, save_file=SAVE_FILE, plot_samples=False):
    if os.path.exists(save_file):
        print("Carregando dados preprocessados...")
        with open(save_file, "rb") as f:
            return pickle.load(f)

    X, y = [], []
    class_mapping = {}
    class_id = 0

    for class_folder in os.listdir(data_path):
        class_path = os.path.join(data_path, class_folder)
        if os.path.isdir(class_path):
            class_mapping[class_id] = class_folder
            for audio_file in os.listdir(class_path):
                file_path = os.path.join(class_path, audio_file)
                if os.path.isfile(file_path) and file_path.endswith(('.wav', '.mp3')):
                    result = process_single_audio(file_path, class_id, features_type=features_type, max_length=max_length, plot=plot_samples)
                    if result:
                        features, label = result
                        X.append(features)
                        y.append(label)
            class_id += 1

    X = np.array(X)
    y = np.array(y)

    # Salvar os dados preprocessados
    with open(save_file, "wb") as f:
        pickle.dump((X, y, class_mapping), f)

    return X, y, class_mapping

# Função para construir o modelo ResNet
def build_resnet_model(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv1D(64, kernel_size=7, strides=2, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Blocos ResNet
    def resnet_block(x, filters, downsample=False):
        shortcut = x
        if downsample:
            shortcut = layers.Conv1D(filters, kernel_size=1, strides=2, padding='same')(shortcut)
        x = layers.Conv1D(filters, kernel_size=3, strides=(2 if downsample else 1), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Conv1D(filters, kernel_size=3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.add([x, shortcut])
        x = layers.ReLU()(x)
        return x

    for filters, num_blocks in zip([64, 128, 256, 512], [4, 3, 2, 1]):
        for i in range(num_blocks):
            x = resnet_block(x, filters, downsample=(i == 0 and filters != 64))

    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs)
    model.compile(optimizer=optimizers.Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Comparação entre os métodos
results = {}

for features_type in ["MFCC", "MGCL", "MGCL-Delta"]:
    print(f"Processando características: {features_type}...")
    X, y, class_mapping = load_dataset(DATA_PATH, features_type=features_type, max_length=MAX_LENGTH, save_file=f"data_{features_type}.pkl", plot_samples=True)

    # Divisão treino/teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    y_train = to_categorical(y_train, num_classes=len(class_mapping))
    y_test = to_categorical(y_test, num_classes=len(class_mapping))

    X_train = X_train[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    # Construção e treinamento do modelo
    model = build_resnet_model((X_train.shape[1], X_train.shape[2]), len(class_mapping))
    history = model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_test, y_test), verbose=1)

    # Avaliação
    accuracy = model.evaluate(X_test, y_test, verbose=0)[1]
    print(f"Acurácia para {features_type}: {accuracy * 100:.2f}%")
    results[features_type] = accuracy

print("Resultados Finais:", results)
