import json
import time

import tensorflow as tf
from utils import config
from tensorflow.keras import layers, models, optimizers, regularizers, callbacks  # type: ignore
from tensorflow.keras.models import load_model  # type: ignore
from tensorflow.keras.callbacks import History  # type: ignore
# Function to build an LSTM model
def build_lstm_model(input_shape, config):
    model = models.Sequential([
        # Input layer to define the shape of the input
        layers.Input(shape=input_shape),
        # First LSTM layer with sequence output and dropout
        layers.LSTM(config["lstm_units"], return_sequences=True, dropout=config["lstm_dropout"]),
        # Second LSTM layer without sequence output
        layers.LSTM(config["lstm_units"], dropout=config["lstm_dropout"]),
        # Dense layer to extract features
        layers.Dense(config["dense_units"], activation='relu'),
        # Dropout layer for regularization
        layers.Dropout(config["dropout_rate"]),
        # Output layer for multi-class classification
        layers.Dense(config["num_classes"], activation='softmax')
    ])
    # Compile the model with Adam optimizer and categorical cross-entropy loss
    model.compile(optimizer=optimizers.Adam(learning_rate=config["learning_rate"]),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Function to build a ResNet model
def build_resnet_model(input_shape, config):
    # Input layer
    inputs = layers.Input(shape=input_shape)
    # Initial convolution, batch normalization, activation, and pooling
    x = layers.Conv1D(config["resnet_filters"], kernel_size=7, strides=2, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling1D(pool_size=3, strides=2, padding='same')(x)

    # Residual blocks with skip connections
    for _ in range(config["resnet_blocks"]):
        shortcut = x  # Preserve the input for the skip connection
        x = layers.Conv1D(config["resnet_filters"], kernel_size=config["resnet_kernel_size"], padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv1D(config["resnet_filters"], kernel_size=config["resnet_kernel_size"], padding='same')(x)
        x = layers.BatchNormalization()(x)
        # Adjust dimensions of the shortcut if needed
        if shortcut.shape[-1] != x.shape[-1]:
            shortcut = layers.Conv1D(config["resnet_filters"], kernel_size=1, padding='same')(shortcut)
        # Add the skip connection
        x = layers.add([x, shortcut])
        x = layers.ReLU()(x)

    # Global average pooling to summarize features
    x = layers.GlobalAveragePooling1D()(x)
    # Fully connected layer for feature extraction
    x = layers.Dense(config["dense_units"], activation='relu')(x)
    # Dropout layer to prevent overfitting
    x = layers.Dropout(config["dropout_rate"])(x)
    # Output layer for classification
    outputs = layers.Dense(config["num_classes"], activation='softmax')(x)

    # Compile the model
    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizers.Adam(learning_rate=config["learning_rate"]),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


# Function to build a Transformer model
def build_transformer_model(input_shape, config):
    # Input layer
    inputs = layers.Input(shape=input_shape)
    # Dense layer for embedding
    x = layers.Dense(config["transformer_ff_dim"], activation='relu')(inputs)

    # Transformer encoder layers
    for _ in range(config["transformer_layers"]):
        # Multi-head attention layer
        attention_output = layers.MultiHeadAttention(
            num_heads=config["transformer_heads"],
            key_dim=config["transformer_ff_dim"] // config["transformer_heads"]
        )(x, x)
        # Add and normalize for stability
        attention_output = layers.LayerNormalization(epsilon=config["transformer_epsilon"])(attention_output + x)
        # Feed-forward network with residual connection
        ffn_output = layers.Dense(config["transformer_ff_dim"], activation='relu')(attention_output)
        ffn_output = layers.Dense(config["transformer_ff_dim"])(ffn_output)
        x = layers.LayerNormalization(epsilon=config["transformer_epsilon"])(ffn_output + attention_output)

    # Pooling layers for fixed-size output
    x = layers.GlobalAveragePooling1D()(x)
    # Fully connected layer for feature extraction
    x = layers.Dense(config["dense_units"], activation='relu')(x)
    # Dropout layer for regularization
    x = layers.Dropout(config["dropout_rate"])(x)
    # Output layer for classification
    outputs = layers.Dense(config["num_classes"], activation='softmax')(x)

    # Compile the model
    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizers.Adam(learning_rate=config["learning_rate"]),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Função para carregar e compilar um modelo
def load_and_compile_model(model_path):
    model = load_model(model_path)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Função para salvar o modelo
def save_model(model, file_path):
    model.save(file_path)
    print(f"Model saved to {file_path}")

# Função para salvar os tempos de treinamento e os históricos dos modelos
def save_training_data(training_times, history_dict, training_params, filename="training_times.json"):
    try:
        data_to_save = {
            "training_params": training_params,
            "training_times": training_times,
            "history_dict": history_dict
        }
        with open(filename, 'w') as file:
            json.dump(data_to_save, file, indent=4)
        print(f"Training data saved to {filename}")
    except Exception as e:
        print(f"Error saving training data: {e}")

# Função para converter um dicionário para histórico
def dict_to_history(history_dict):
    if not history_dict:
        return None
    history = History()
    history.history = history_dict
    return history

# Função para carregar os dados de treinamento
def load_training_data(filename="training_times.json"):
    try:
        with open(filename, 'r') as file:
            data = json.load(file)
            training_times = data.get("training_times", {})
            history_dict = data.get("history_dict", {})
            training_params = data.get("training_params", {})
        print("Training data loaded successfully!")
        return training_times, history_dict, training_params
    except Exception as e:
        print(f"Error loading training data: {e}")
        return {}, {}, {}

# Função para treinar o modelo
def train_model(model_name, X_train, y_train, X_test, y_test, config):
    print(f"Training {model_name}...")
    start_time = time.time()
    
    if model_name == 'resnet':
        model = build_resnet_model((X_train.shape[1], 1), config)
    elif model_name == 'lstm':
        model = build_lstm_model((X_train.shape[1], 1), config)
    elif model_name == 'transformer':
        model = build_transformer_model((X_train.shape[1], 1), config)
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=config["epochs"], batch_size=config["batch_size"], validation_data=(X_test, y_test))
    
    training_time = time.time() - start_time
    save_model(model, f"{model_name}_model.keras")
    
    return history, training_time

# Função principal para treinar os modelos selecionados
def training_models(X_train, y_train, X_test, y_test):
    training_times = {}
    history_dict = {}

    training_params = {
        "epochs": config["epochs"],
        "batch_size": config["batch_size"],
        "test_size": config["test_size"],
        "random_state": config["random_state"],
        "num_classes": config["num_classes"]
    }

    # Modelos selecionados no arquivo de configuração
    for model_name in config["selected_models"]:
        history, training_time = train_model(model_name, X_train, y_train, X_test, y_test, config)
        history_dict[model_name] = history.history
        training_times[model_name] = training_time

    save_training_data(training_times, history_dict, training_params)
    print("Models trained and saved successfully!")