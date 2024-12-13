import os
import numpy as np
import librosa
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models, optimizers

# Função para preprocessar e fundir características
def extract_features(file_path, max_length=100):
    audio, sample_rate = librosa.load(file_path, sr=None)

    # Extração de características individuais
    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
    gfcc = mfcc  # Substituir por implementação GFCC
    cqt = np.abs(librosa.cqt(y=audio, sr=sample_rate, n_bins=84))
    lofar = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)

    # Fusão das características
    features = np.concatenate([mfcc, gfcc, cqt, lofar], axis=0)

    # Normalizar e ajustar tamanho
    std = np.std(features, axis=1, keepdims=True)
    std[std == 0] = 1e-8
    features = (features - np.mean(features, axis=1, keepdims=True)) / std

    if features.shape[1] > max_length:
        features = features[:, :max_length]
    else:
        features = np.pad(features, ((0, 0), (0, max_length - features.shape[1])), mode='constant')

    return features

# Função para carregar o dataset e aplicar fusão
def load_dataset(data_path, max_length=100):
    X, y = [], []
    class_mapping = {}

    for class_id, class_folder in enumerate(os.listdir(data_path)):
        class_path = os.path.join(data_path, class_folder)
        if os.path.isdir(class_path):
            class_mapping[class_id] = class_folder
            for audio_file in os.listdir(class_path):
                file_path = os.path.join(class_path, audio_file)
                if file_path.endswith(('.wav', '.mp3')):
                    features = extract_features(file_path, max_length)
                    X.append(features)
                    y.append(class_id)

    return np.array(X), np.array(y), class_mapping

# Redução de dimensionalidade com NCA
def reduce_dimensionality(X, y):
    nca = NeighborhoodComponentsAnalysis(random_state=42)
    X_reduced = nca.fit_transform(X.reshape(len(X), -1), y)
    return X_reduced

# Construção do modelo ResNet18
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

# Caminho dos dados
data_path = r"C:\\Users\\Gustavo\\Desktop\\deepship"
X, y, class_mapping = load_dataset(data_path, max_length=100)

# Reduzir dimensionalidade
X_reduced = reduce_dimensionality(X, y)

# Divisão treino/teste
X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.25, random_state=42)

# Ajustar formato para entrada no modelo
X_train = X_train.reshape((-1, 10, 10, 1))  # Ajustar para formato 2D
X_test = X_test.reshape((-1, 10, 10, 1))

# Construir e treinar o modelo
model = build_resnet18(input_shape=X_train.shape[1:], num_classes=len(class_mapping))
history = model.fit(X_train, y_train, epochs=30, batch_size=5, validation_data=(X_test, y_test))

# Avaliar modelo
accuracy = model.evaluate(X_test, y_test, verbose=1)[1]
print(f"Acurácia: {accuracy * 100:.2f}%")
