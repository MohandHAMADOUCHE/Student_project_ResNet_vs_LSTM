import json
import time
from utils import config
from tensorflow.keras import layers, models # type: ignore
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.callbacks import History # type: ignore

# Build an LSTM model for classification
def build_lstm_model(input_shape, num_classes):
    model = models.Sequential()
    model.add(layers.Input(shape=input_shape))  # Adiciona explicitamente a camada Input
    model.add(layers.LSTM(64, return_sequences=True))
    model.add(layers.LSTM(64))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))
    return model

# Build a ResNet model for classification
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
    return models.Model(inputs=inputs, outputs=outputs)

# Build a Transformer model for classification
def build_transformer_model(input_shape, num_classes, num_heads=4, ff_dim=128, num_layers=2):
    inputs = layers.Input(shape=input_shape)

    # Embed the input
    x = layers.Dense(ff_dim, activation='relu')(inputs)
    
    for _ in range(num_layers):
        # Multi-head attention block
        attention_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=ff_dim)(x, x)
        attention_output = layers.LayerNormalization(epsilon=1e-6)(attention_output + x)
        
        # Feed-forward block
        ffn_output = layers.Dense(ff_dim, activation='relu')(attention_output)
        ffn_output = layers.Dense(ff_dim)(ffn_output)
        x = layers.LayerNormalization(epsilon=1e-6)(ffn_output + attention_output)
    
    # Global pooling and classification layers
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128, activation='relu')(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    return models.Model(inputs=inputs, outputs=outputs)

# Após carregar o modelo, recompilá-lo
def load_and_compile_model(model_path):
    model = load_model(model_path)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Função para salvar o modelo
def save_model(model, file_path):
    model.save(file_path)
    print(f"Model saved to {file_path}")

def save_training_data(training_times, history_resnet=None, history_lstm=None, history_transformer=None, training_params=None, filename="training_times.json"):
    """
    Salva os tempos de treinamento, o histórico dos modelos e os parâmetros de treinamento no arquivo JSON.
    """
    try:
        data_to_save = {
            "training_params": training_params,
            "training_times": training_times,
            "history_resnet": history_resnet.history if history_resnet else None,  # Salva o histórico como dicionário
            "history_lstm": history_lstm.history if history_lstm else None,
            "history_transformer": history_transformer.history if history_transformer else None
        }

        with open(filename, 'w') as file:
            json.dump(data_to_save, file, indent=4)
        print(f"Training times, history, and params saved to {filename}")
    except Exception as e:
        print(f"Error saving training times: {e}")

def dict_to_history(history_dict):
    if not history_dict:
        return None
    history = History()
    history.history = history_dict
    return history

def load_training_data(filename="training_times.json"):
    """
    Carrega os tempos de treinamento, histórico e parâmetros de um arquivo JSON.
    """
    try:
        with open(filename, 'r') as file:
            data = json.load(file)
            training_times = data.get("training_times", {})
            history_resnet = dict_to_history(data.get("history_resnet", {}))
            history_lstm = dict_to_history(data.get("history_lstm", {}))
            history_transformer = dict_to_history(data.get("history_transformer", {}))
            training_params = data.get("training_params", {})
        print("Training data loaded successfully!")
        return training_times, history_resnet, history_lstm, history_transformer, training_params
    except Exception as e:
        print(f"Error loading training data: {e}")
        return {}, None, None, None, {}

def training_models(X_train, y_train, X_test, y_test):

    # Armazenar tempos de treinamento
    training_times = {}

    # Armazenar tempos de treinamento
    history_resnet, history_lstm, history_transformer = None, None, None

    # Parâmetros de treinamento
    training_params = {
        "epochs": config["epochs"],
        "batch_size": config["batch_size"],
        "test_size": config["test_size"],
        "random_state": config["random_state"],
        "num_classes": config["num_classes"]
    }

    # Treinamento dos modelos, salvando os tempos
    print("Training ResNet...")
    start_time_resnet = time.time()
    resnet_model = build_resnet_model((X_train.shape[1], 1), config["num_classes"])
    resnet_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    history_resnet = resnet_model.fit(X_train, y_train, epochs=config["epochs"], batch_size=config["batch_size"], validation_data=(X_test, y_test))
    training_times["resnet"] = time.time() - start_time_resnet

    # Train and evaluate LSTM
    print("Training LSTM...")
    start_time_lstm = time.time()
    lstm_model = build_lstm_model((X_train.shape[1], 1), config["num_classes"])
    lstm_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    history_lstm = lstm_model.fit(X_train, y_train, epochs=config["epochs"], batch_size=config["batch_size"], validation_data=(X_test, y_test))
    training_times["lstm"] = time.time() - start_time_lstm

    # Train and evaluate Transformer
    print("Training Transformer...")
    start_time_transformer = time.time()
    transformer_model = build_transformer_model((X_train.shape[1], 1), config["num_classes"])
    transformer_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    history_transformer = transformer_model.fit(X_train, y_train, epochs=config["epochs"], batch_size=config["batch_size"], validation_data=(X_test, y_test))
    training_times["transformer"] = time.time() - start_time_transformer

    # Salvar os modelos
    # Salvar os modelos usando o formato .keras
    save_model(resnet_model, "resnet_model.keras")
    save_model(lstm_model, "lstm_model.keras")
    save_model(transformer_model, "transformer_model.keras")

    print("Models trained and saved successfully!")

    # Salvar os tempos de treinamento em um arquivo JSON
    save_training_data(training_times, history_resnet, history_lstm, history_transformer, training_params)