from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical

# Fonction pour construire le modèle ResNet
def build_resnet_model(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv1D(64, kernel_size=7, strides=2, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling1D(pool_size=3, strides=2, padding='same')(x)

    for _ in range(3):
        shortcut = x
        x = layers.Conv1D(64, kernel_size=3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Conv1D(64, kernel_size=3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        if shortcut.shape[-1] != x.shape[-1]:
            shortcut = layers.Conv1D(64, kernel_size=1, padding='same')(shortcut)
        x = layers.add([x, shortcut])
        x = layers.ReLU()(x)
    
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128, activation='relu')(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    return model

# Fonction pour construire le modèle LSTM
def build_lstm_model(input_shape, num_classes):
    model = models.Sequential()
    model.add(layers.LSTM(64, input_shape=input_shape, return_sequences=True))
    model.add(layers.LSTM(64))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))
    return model

# Fonction pour entraîner le modèle
def train_model(X_train, y_train, X_test, y_test, model_type='resnet', epochs=3):
    num_classes = y_train.shape[1]
    if model_type == 'resnet':
        model = build_resnet_model(input_shape=(X_train.shape[1], 1), num_classes=num_classes)
    else:
        model = build_lstm_model(input_shape=(X_train.shape[1], 1), num_classes=num_classes)
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=32, validation_data=(X_test, y_test))
    return model, history
