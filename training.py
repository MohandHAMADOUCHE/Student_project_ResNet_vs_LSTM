import json
import time
from tensorflow.keras import layers, models, optimizers  # type: ignore
from tensorflow.keras.models import load_model  # type: ignore
from tensorflow.keras.callbacks import History  # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore

# Function to build an LSTM model
def build_lstm_model(input_shape, config, num_classes):
    model = models.Sequential([
        # Input layer to define the shape of the input
        layers.Input(shape=input_shape),
        # First LSTM layer with sequence output and dropout
        layers.LSTM(config["lstm"]["lstm_units"], return_sequences=True, dropout=config["lstm"]["lstm_dropout"]),
        # Second LSTM layer without sequence output
        layers.LSTM(config["lstm"]["lstm_units"], dropout=config["lstm"]["lstm_dropout"]),
        # Dense layer to extract features
        layers.Dense(config["general"]["dense_units"], activation='relu'),
        # Dropout layer for regularization
        layers.Dropout(rate=config["lstm"]["lstm_dropout"]),
        # Output layer for multi-class classification
        layers.Dense(num_classes, activation='softmax')
    ])
    # Compile the model with Adam optimizer and categorical cross-entropy loss
    model.compile(optimizer=optimizers.Adam(learning_rate=config["general"]["learning_rate"]),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Function to build a ResNet model
def build_resnet_model(input_shape, config, num_classes):
    # Input layer
    inputs = layers.Input(shape=input_shape)
    # Initial convolution, batch normalization, activation, and pooling
    x = layers.Conv1D(config["resnet"]["resnet_filters"], kernel_size=7, strides=2, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling1D(pool_size=3, strides=2, padding='same')(x)

    # Residual blocks with skip connections
    for _ in range(config["resnet"]["resnet_blocks"]):
        shortcut = x  # Preserve the input for the skip connection
        x = layers.Conv1D(config["resnet"]["resnet_filters"], kernel_size=config["resnet"]["resnet_kernel_size"], padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv1D(config["resnet"]["resnet_filters"], kernel_size=config["resnet"]["resnet_kernel_size"], padding='same')(x)
        x = layers.BatchNormalization()(x)
        # Adjust dimensions of the shortcut if needed
        if shortcut.shape[-1] != x.shape[-1]:
            shortcut = layers.Conv1D(config["resnet"]["resnet_filters"], kernel_size=1, padding='same')(shortcut)
        # Add the skip connection
        x = layers.add([x, shortcut])
        x = layers.ReLU()(x)

    # Global average pooling to summarize features
    x = layers.GlobalAveragePooling1D()(x)
    # Fully connected layer for feature extraction
    x = layers.Dense(config["general"]["dense_units"], activation='relu')(x)
    # Dropout layer to prevent overfitting
    x = layers.Dropout(config["general"]["dropout_rate"])(x)
    # Output layer for classification
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    # Compile the model
    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizers.Adam(learning_rate=config["general"]["learning_rate"]),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


# Function to build a Transformer model
def build_transformer_model(input_shape, config, num_classes):
    # Input layer
    inputs = layers.Input(shape=input_shape)
    # Dense layer for embedding
    x = layers.Dense(config["transformer"]["transformer_ff_dim"], activation='relu')(inputs)

    # Transformer encoder layers
    for _ in range(config["transformer"]["transformer_layers"]):
        # Multi-head attention layer
        attention_output = layers.MultiHeadAttention(
            num_heads=config["transformer"]["transformer_heads"],
            key_dim=config["transformer"]["transformer_ff_dim"] // config["transformer"]["transformer_heads"]
        )(x, x)
        # Add and normalize for stability
        attention_output = layers.LayerNormalization(epsilon=config["transformer"]["transformer_epsilon"])(attention_output + x)
        # Feed-forward network with residual connection
        ffn_output = layers.Dense(config["transformer"]["transformer_ff_dim"], activation='relu')(attention_output)
        ffn_output = layers.Dense(config["transformer"]["transformer_ff_dim"])(ffn_output)
        x = layers.LayerNormalization(epsilon=config["transformer"]["transformer_epsilon"])(ffn_output + attention_output)

    # Pooling layers for fixed-size output
    x = layers.GlobalAveragePooling1D()(x)
    # Fully connected layer for feature extraction
    x = layers.Dense(config["general"]["dense_units"], activation='relu')(x)
    # Dropout layer for regularization
    x = layers.Dropout(config["general"]["dropout_rate"])(x)
    # Output layer for classification
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    # Compile the model
    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizers.Adam(learning_rate=config["general"]["learning_rate"]),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# ======================
# Model Construction
# ======================

def build_resnet18_model(input_shape, config, num_classes):
    """
    Builds a ResNet18 model for classification.

    Args:
        input_shape (tuple): Shape of the input features.
        num_classes (int): Number of output classes.

    Returns:
        keras.Model: Compiled ResNet18 model.
    """
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(config["resnet18"]["initial_filters"], kernel_size=7, strides=2, padding='same')(inputs)
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
    model.compile(optimizer=optimizers.SGD(learning_rate=config["general"]["learning_rate"], momentum=config["resnet18"]["momentum"]),
                  loss='sparse_categorical_crossentropy',
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
def train_model(model_name, X_train, y_train, X_test, y_test, config, class_to_index):
    print(f"Training {model_name}...")
    start_time = time.time()
    num_classes = len(class_to_index)
    
    if model_name == 'resnet':
        model = build_resnet_model((X_train.shape[1], 1), config, num_classes)
    elif model_name == 'lstm':
        model = build_lstm_model((X_train.shape[1], 1), config, num_classes)
    elif model_name == 'transformer':
        model = build_transformer_model((X_train.shape[1], 1), config, num_classes)
    elif model_name == 'resnet18':
        model = build_resnet18_model(X_train.shape[1:], config, num_classes)


    early_stopping = EarlyStopping(monitor='val_loss', patience=config["general"]["patience"], restore_best_weights=True)
    history = model.fit(X_train, y_train, epochs=config["general"]["epochs"], batch_size=config["general"]["batch_size"], validation_data=(X_test, y_test), callbacks=[early_stopping])
    # Evaluate model
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Accuracy: {accuracy * 100:.2f}%")

    training_time = time.time() - start_time
    save_model(model, f"{model_name}_model.keras")
    
    return history, training_time


def train(config, X_train, y_train, X_test, y_test, class_to_index):

    print("Training the model...")

    training_times = {}
    history_dict = {}

    for model_name, is_selected in config["selected_models"].items():
        if is_selected:
            history, training_time = train_model(model_name, X_train, y_train, X_test, y_test, config, class_to_index)
            history_dict[model_name] = history.history
            training_times[model_name] = training_time

    save_training_data(training_times, history_dict, config)

    print("Models trained and saved successfully!")

    